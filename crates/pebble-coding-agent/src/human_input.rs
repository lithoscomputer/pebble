//! Asking the person a question mid-prompt.
//!
//! A session can only ask if the application gave it a way to: pebble defines
//! the seam, [`HumanInputProvider`], and the application implements it over
//! whatever it has — a terminal prompt, a web form, a chat message, a
//! workflow that parks until someone answers.
//!
//! Three rules hold wherever the provider comes from:
//!
//! - **One question tool per round.** The question tools are the point where a
//!   prompt waits for a person, so the execution layer runs at most one of them
//!   per tool round and refuses its peers. A model that wants to ask several
//!   things asks them in one batch.
//! - **Root sessions only.** Child sessions never register a question tool;
//!   they report back to their parent instead.
//! - **The call is interruptible.**
//!   [`ask_questions`](HumanInputProvider::ask_questions) receives the tool
//!   call's cancellation token, and an interrupt cancels a pending question
//!   rather than waiting for an answer.
//!
//! Answers come back as data, not as an error: a person who declines, or a
//! prompt that is interrupted, produces an [`Answer`] with the matching
//! [`AnswerStatus`]. An `Err` means the provider itself failed.

use std::error::Error as StdError;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::tool::{NativeTool, ToolError};
use crate::types::ToolErrorKind;

/// Whether `tool_name` names one of pebble's human-question tools.
///
/// Both tools keep their names in every vocabulary, so this answers the same
/// way whichever profile is running.
#[must_use]
pub(crate) fn is_question_tool(tool_name: &str) -> bool {
    matches!(
        NativeTool::from_any_name(tool_name),
        Some(NativeTool::AskUserQuestion | NativeTool::RequestUserInput)
    )
}

/// What a question expects back.
///
/// Pebble's question tools ask for a choice, so these are the two kinds they
/// produce. More may appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum QuestionKind {
    /// Exactly one of the options.
    MultipleChoice,
    /// Any number of the options.
    MultiSelect,
}

/// One answer a question offers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionOption {
    /// The stable identifier an answer names.
    pub key:         String,
    /// The short text shown beside the option.
    pub label:       String,
    /// What choosing this option means, when the label is not enough.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A longer sample of what this option would produce, for a caller that
    /// can show one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview:     Option<String>,
}

/// One question a session is asking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    /// The identifier the model gave this question, when the tool shape it
    /// called carries one.
    ///
    /// An [`Answer`] echoes it back unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_id:       Option<String>,
    /// The question text exactly as the model wrote it, untrimmed.
    ///
    /// This is the key that matches an [`Answer`] to its question, so a
    /// provider must echo it back unchanged rather than the displayed
    /// [`text`](Self::text).
    pub original_question: String,
    /// A short heading for the question, when the model supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header:            Option<String>,
    /// The question as it should be shown, header included.
    pub text:              String,
    /// What the question expects back.
    pub kind:              QuestionKind,
    /// The options offered, in the order they should be shown.
    #[serde(default)]
    pub options:           Vec<QuestionOption>,
    /// Whether an answer outside the offered options is accepted.
    pub allow_freeform:    bool,
}

/// How a question ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AnswerStatus {
    /// The person answered.
    Answered,
    /// The prompt was cancelled while the question was open.
    Cancelled,
    /// The person interrupted the prompt instead of answering.
    Interrupted,
    /// The person chose not to answer this question.
    Skipped,
    /// Nobody answered in the time the application allows.
    Timeout,
}

/// What came back for one [`Question`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answer {
    /// The question's [`original_id`](Question::original_id), echoed back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_id:       Option<String>,
    /// The question's [`original_question`](Question::original_question),
    /// echoed back unchanged. This is what matches the answer to its question.
    pub original_question: String,
    /// What the person chose or wrote. One entry for a single choice, several
    /// for a multi-select, none for any status but
    /// [`Answered`](AnswerStatus::Answered).
    #[serde(default)]
    pub answers:           Vec<String>,
    /// How the question ended.
    pub status:            AnswerStatus,
}

impl Answer {
    /// An answered question.
    #[must_use]
    pub fn answered(question: &Question, answers: Vec<String>) -> Self {
        Self {
            original_id: question.original_id.clone(),
            original_question: question.original_question.clone(),
            answers,
            status: AnswerStatus::Answered,
        }
    }

    /// A question that ended without an answer.
    #[must_use]
    pub fn unanswered(question: &Question, status: AnswerStatus) -> Self {
        Self {
            original_id: question.original_id.clone(),
            original_question: question.original_question.clone(),
            answers: Vec::new(),
            status,
        }
    }
}

/// A failure of the question channel itself.
///
/// Not the answer being "no", and not an interrupted prompt — those are an
/// [`Answer`] with the matching [`AnswerStatus`]. This is the transport
/// failing: a closed UI, a dropped connection, a queue that will not accept
/// the question.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct HumanInputError {
    message: String,
    #[source]
    source:  Option<Box<dyn StdError + Send + Sync + 'static>>,
}

impl HumanInputError {
    /// Builds an error with no underlying cause.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source:  None,
        }
    }

    /// Builds an error that keeps `source` as its cause.
    #[must_use]
    pub fn with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source:  Some(Box::new(source)),
        }
    }

    /// The message, without its cause.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<HumanInputError> for ToolError {
    /// The question tool reports a failed channel as a failed call, keeping
    /// the failure as the cause.
    fn from(error: HumanInputError) -> Self {
        Self::with_rendered_source(ToolErrorKind::Execution, "Could not ask the user", error)
    }
}

/// Where a session's questions go.
///
/// An application installs one on the session, or on a single prompt when a
/// reused session should bind its questions to the current stage. A session
/// without a provider registers no question tool, so a model cannot ask.
///
/// Implementations run inside a tool call: they hold the round open until the
/// person answers, so they must respect `cancel_token` and return promptly
/// once it fires.
#[async_trait]
pub trait HumanInputProvider: Send + Sync {
    /// Asks one batch of questions and waits for the answers.
    ///
    /// `tool_call_id` identifies the call that is waiting, so an application
    /// can correlate the prompt with the events it already received.
    ///
    /// Returning fewer answers than there were questions is allowed; the
    /// question tool reports the unanswered ones to the model. An answer is
    /// matched to its question by
    /// [`original_question`](Question::original_question), not by position.
    ///
    /// # Errors
    ///
    /// Returns [`HumanInputError`] only when the channel itself failed. A
    /// person declining, an interrupt, or a timeout is an [`Answer`] with the
    /// matching [`AnswerStatus`].
    async fn ask_questions(
        &self,
        tool_call_id: &str,
        questions: Vec<Question>,
        cancel_token: CancellationToken,
    ) -> Result<Vec<Answer>, HumanInputError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::tool::ToolVocabulary;

    fn question(text: &str) -> Question {
        Question {
            original_id:       Some("q1".to_owned()),
            original_question: text.to_owned(),
            header:            Some("Deploy".to_owned()),
            text:              format!("Deploy\n\n{text}"),
            kind:              QuestionKind::MultipleChoice,
            options:           vec![QuestionOption {
                key: "yes".to_owned(),
                label: "Yes".to_owned(),
                ..QuestionOption::default()
            }],
            allow_freeform:    true,
        }
    }

    struct ScriptedProvider {
        answers: Vec<Answer>,
    }

    #[async_trait]
    impl HumanInputProvider for ScriptedProvider {
        async fn ask_questions(
            &self,
            _tool_call_id: &str,
            _questions: Vec<Question>,
            _cancel_token: CancellationToken,
        ) -> Result<Vec<Answer>, HumanInputError> {
            Ok(self.answers.clone())
        }
    }

    #[test]
    fn question_tools_are_recognized_under_every_name() {
        assert!(is_question_tool("AskUserQuestion"));
        assert!(is_question_tool("request_user_input"));
        for vocabulary in ToolVocabulary::ALL.iter().copied() {
            assert!(is_question_tool(
                NativeTool::AskUserQuestion.name(vocabulary)
            ));
            assert!(is_question_tool(
                NativeTool::RequestUserInput.name(vocabulary)
            ));
        }
    }

    #[test]
    fn other_tools_are_not_question_tools() {
        assert!(!is_question_tool("read_file"));
        assert!(!is_question_tool("Bash"));
        assert!(!is_question_tool("mcp__forms__ask_user_question"));
    }

    #[test]
    fn an_answer_echoes_the_question_identity() {
        let question = question("Ship it?");

        let answered = Answer::answered(&question, vec!["yes".to_owned()]);
        assert_eq!(answered.original_id, question.original_id);
        assert_eq!(answered.original_question, "Ship it?");
        assert_eq!(answered.status, AnswerStatus::Answered);

        let skipped = Answer::unanswered(&question, AnswerStatus::Skipped);
        assert!(skipped.answers.is_empty());
        assert_eq!(skipped.status, AnswerStatus::Skipped);
    }

    #[test]
    fn a_question_serializes_without_its_absent_members() {
        let question = Question {
            original_id: None,
            header: None,
            ..question("Ship it?")
        };

        assert_eq!(
            serde_json::to_value(&question).expect("serializes"),
            json!({
                "original_question": "Ship it?",
                "text": "Deploy\n\nShip it?",
                "kind": "multiple_choice",
                "options": [{"key": "yes", "label": "Yes"}],
                "allow_freeform": true,
            })
        );
    }

    #[test]
    fn an_answer_round_trips() {
        let answer = Answer {
            original_id:       Some("q1".to_owned()),
            original_question: "Ship it?".to_owned(),
            answers:           vec!["yes".to_owned()],
            status:            AnswerStatus::Answered,
        };
        let value = serde_json::to_value(&answer).expect("serializes");
        assert_eq!(
            value,
            json!({
                "original_id": "q1",
                "original_question": "Ship it?",
                "answers": ["yes"],
                "status": "answered",
            })
        );
        assert_eq!(
            serde_json::from_value::<Answer>(value).expect("parses"),
            answer
        );
    }

    #[test]
    fn answer_statuses_are_snake_case_on_the_wire() {
        for (status, wire) in [
            (AnswerStatus::Answered, "answered"),
            (AnswerStatus::Cancelled, "cancelled"),
            (AnswerStatus::Interrupted, "interrupted"),
            (AnswerStatus::Skipped, "skipped"),
            (AnswerStatus::Timeout, "timeout"),
        ] {
            assert_eq!(
                serde_json::to_value(status).expect("serializes"),
                json!(wire)
            );
        }
    }

    #[test]
    fn question_kinds_are_snake_case_on_the_wire() {
        assert_eq!(
            serde_json::to_value(QuestionKind::MultipleChoice).expect("serializes"),
            json!("multiple_choice")
        );
        assert_eq!(
            serde_json::to_value(QuestionKind::MultiSelect).expect("serializes"),
            json!("multi_select")
        );
    }

    #[test]
    fn a_channel_failure_becomes_a_failed_tool_call() {
        let error = ToolError::from(HumanInputError::new("the interviewer disconnected"));
        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert_eq!(
            error.message(),
            "Could not ask the user: the interviewer disconnected"
        );
    }

    #[tokio::test]
    async fn a_provider_is_usable_behind_a_trait_object() {
        let question = question("Ship it?");
        let provider: Arc<dyn HumanInputProvider> = Arc::new(ScriptedProvider {
            answers: vec![Answer::answered(&question, vec!["yes".to_owned()])],
        });

        let answers = provider
            .ask_questions("call_1", vec![question], CancellationToken::new())
            .await
            .expect("the provider answers");

        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].answers, vec!["yes".to_owned()]);
    }
}
