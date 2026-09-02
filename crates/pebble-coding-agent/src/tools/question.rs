//! Asking the person a question, in the shape each harness expects.
//!
//! Three tools, one flow. OpenAI's `request_user_input` sends questions with
//! identifiers and wants the answers back keyed by them; Anthropic's
//! `AskUserQuestion` sends questions with headers and options and wants them
//! back as a sentence; Claude 5 sends the same shape under a contract that
//! bounds almost every field. All three normalize into pebble's own
//! [`Question`], hand it to the session's [`HumanInputProvider`], and render
//! what came back the way their model expects to read it.
//!
//! A question tool is registered only in a root session, because a child
//! reports back to its parent rather than to a person, and only where the
//! application gave the session somewhere to ask. Both are enforced: a child is
//! built without a provider, so no question tool is registered in one, and the
//! executor refuses a call whose session is not the root of its tree even if
//! one somehow was.
//!
//! [`Question`]: crate::extensions::Question
//! [`HumanInputProvider`]: crate::extensions::HumanInputProvider

use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::human_input::{Answer, AnswerStatus, Question, QuestionKind, QuestionOption};
use crate::tool::{NativeTool, RegisteredTool, ToolContext, ToolError};
use crate::types::{AgentProfileKind, ToolErrorKind, ToolSource};

/// How much of an option's description a person is shown.
const OPTION_DESCRIPTION_MAX_CHARS: usize = 2_000;

/// How much of an option's preview a person is shown.
const OPTION_PREVIEW_MAX_CHARS: usize = 4_000;

/// What a call outside a root session is told.
const ROOT_SESSION_REQUIRED_ERROR: &str =
    "human-question tools are available only during a root agent session";

/// The tool that asks a person a question in `profile`'s vocabulary, or `None`
/// where the harness has none.
///
/// This is the whole registration rule: a session builder asks once, and only
/// when the application gave the session a
/// [`HumanInputProvider`](crate::extensions::HumanInputProvider) to ask
/// through. Gemini's harness has no question tool, so a Gemini session never
/// advertises one.
#[must_use]
pub fn make_question_tool(profile: AgentProfileKind) -> Option<RegisteredTool> {
    match profile {
        // Codex names this tool `request_user_input` for GPT-5.6 too.
        AgentProfileKind::OpenAi | AgentProfileKind::Gpt56 => Some(make_openai_question_tool()),
        // Kimi Code names it `AskUserQuestion` with the same question and
        // option shape, so the Anthropic tool is the match.
        AgentProfileKind::Anthropic | AgentProfileKind::Kimi => {
            Some(make_anthropic_question_tool())
        }
        AgentProfileKind::Claude5 => Some(make_claude5_question_tool()),
        AgentProfileKind::Gemini => None,
    }
}

/// One question as the OpenAI harness sends it.
#[derive(Debug, Deserialize)]
struct OpenAiQuestionToolArgs {
    questions: Vec<OpenAiQuestion>,
}

#[derive(Debug, Deserialize)]
struct OpenAiQuestion {
    id:       String,
    header:   String,
    question: String,
    #[serde(default)]
    options:  Vec<OpenAiOption>,
}

#[derive(Debug, Deserialize)]
struct OpenAiOption {
    label:       String,
    #[serde(default)]
    description: Option<String>,
}

/// One question as the Anthropic harnesses send it.
#[derive(Debug, Deserialize)]
struct AnthropicQuestionToolArgs {
    questions: Vec<AnthropicQuestion>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnthropicQuestion {
    question:     String,
    #[serde(default)]
    header:       Option<String>,
    #[serde(default)]
    options:      Vec<AnthropicOption>,
    #[serde(default)]
    multi_select: bool,
}

#[derive(Debug, Deserialize)]
struct AnthropicOption {
    label:       String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    preview:     Option<String>,
}

/// The contract rules a JSON schema cannot state, and which differ between the
/// two harnesses that share one normalizer.
struct QuestionLimits {
    questions: RangeInclusive<usize>,
    questions_error: &'static str,
    /// `None` leaves the option count unbounded.
    options: Option<RangeInclusive<usize>>,
    options_error: &'static str,
    max_header_chars: Option<usize>,
    /// Claude 5's schema marks `header` and every option `description`
    /// required, so both are checked rather than passed through as given.
    require_header_and_descriptions: bool,
    /// Claude 5 renders a multi-select without a preview pane.
    allow_preview_with_multi_select: bool,
}

const ANTHROPIC_QUESTION_LIMITS: QuestionLimits = QuestionLimits {
    questions: 1..=usize::MAX,
    questions_error: "questions must contain at least one question",
    options: None,
    options_error: "",
    max_header_chars: None,
    require_header_and_descriptions: false,
    allow_preview_with_multi_select: true,
};

const CLAUDE5_QUESTION_LIMITS: QuestionLimits = QuestionLimits {
    questions: 1..=4,
    questions_error: "questions must contain between one and four questions",
    options: Some(2..=4),
    options_error: "each question must contain between two and four options",
    max_header_chars: Some(12),
    require_header_and_descriptions: true,
    allow_preview_with_multi_select: false,
};

/// Asks a person one or more questions, the way the OpenAI harness does.
///
/// Every question carries an identifier the model chose, and the answers come
/// back as JSON keyed by it.
#[must_use]
pub fn make_openai_question_tool() -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::RequestUserInput.canonical_name(),
            "Ask the human one or more questions and wait for their answers before continuing \
             this stage.",
            json!({
                "type": "object",
                "required": ["questions"],
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["id", "header", "question", "options"],
                            "properties": {
                                "id": { "type": "string" },
                                "header": { "type": "string" },
                                "question": { "type": "string" },
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["label"],
                                        "properties": {
                                            "label": { "type": "string" },
                                            "description": { "type": "string" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }),
        ),
        Arc::new(|args, ctx| {
            Box::pin(async move {
                let parsed: OpenAiQuestionToolArgs = parse_tool_args(args)?;
                let questions = normalize_openai_questions(parsed)?;
                let answers = ask(ctx, questions).await?;
                format_openai_answers(&answers)
            })
        }),
    )
    .with_source(ToolSource::Native)
    .requires_human_input()
}

/// Asks a person one or more questions, the way the Anthropic harnesses do.
///
/// The contract is lenient: a header and an option description are optional,
/// and there is no bound on how many questions or options one call carries.
#[must_use]
pub fn make_anthropic_question_tool() -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::AskUserQuestion.canonical_name(),
            "Ask the human one or more questions and wait for their answers before continuing \
             this stage.",
            json!({
                "type": "object",
                "required": ["questions"],
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["question", "options", "multiSelect"],
                            "properties": {
                                "question": { "type": "string" },
                                "header": { "type": "string" },
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["label"],
                                        "properties": {
                                            "label": { "type": "string" },
                                            "description": { "type": "string" },
                                            "preview": { "type": "string" }
                                        }
                                    }
                                },
                                "multiSelect": { "type": "boolean" }
                            }
                        }
                    }
                }
            }),
        ),
        Arc::new(|args, ctx| {
            Box::pin(async move {
                let parsed: AnthropicQuestionToolArgs = parse_tool_args(args)?;
                let questions = normalize_anthropic_questions(parsed, &ANTHROPIC_QUESTION_LIMITS)?;
                let answers = ask(ctx, questions).await?;
                format_anthropic_answers(&answers)
            })
        }),
    )
    .with_source(ToolSource::Native)
    .requires_human_input()
}

/// Asks a person one or more questions, the way Claude 5 does.
///
/// The same wire shape as [`make_anthropic_question_tool`] under a much tighter
/// contract: one to four questions, two to four options each, a header of at
/// most twelve characters, a description on every option, and no preview on a
/// multi-select. Pebble's schema check is structural, so the normalizer
/// enforces all of it rather than trusting the schema to.
#[must_use]
pub fn make_claude5_question_tool() -> RegisteredTool {
    RegisteredTool::new(ToolDefinition::function(
            NativeTool::AskUserQuestion.canonical_name(),
            "Ask the human up to four questions when a decision is genuinely theirs to make. The \
             UI automatically provides an Other option for custom text.",
            json!({
                "type": "object",
                "properties": {
                    "questions": {
                        "description": "Questions to ask the user (1-4 questions)",
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "description": "The complete, clear, and specific question to ask.",
                                    "type": "string"
                                },
                                "header": {
                                    "description": "Very short label displayed as a chip/tag (max 12 chars).",
                                    "type": "string"
                                },
                                "options": {
                                    "description": "Two to four choices. Do not include Other; the UI adds it automatically.",
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {
                                                "description": "Concise display text for the option.",
                                                "type": "string"
                                            },
                                            "description": {
                                                "description": "What the option means and its relevant trade-offs.",
                                                "type": "string"
                                            },
                                            "preview": {
                                                "description": "Optional Markdown preview for single-select visual comparisons.",
                                                "type": "string"
                                            }
                                        },
                                        "required": ["label", "description"],
                                        "additionalProperties": false
                                    }
                                },
                                "multiSelect": {
                                    "description": "Whether the user may select multiple options.",
                                    "default": false,
                                    "type": "boolean"
                                }
                            },
                            "required": ["question", "header", "options", "multiSelect"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["questions"],
                "additionalProperties": false
            }),
        ), Arc::new(|args, ctx| {
            Box::pin(async move {
                let parsed: AnthropicQuestionToolArgs = parse_tool_args(args)?;
                let questions = normalize_anthropic_questions(parsed, &CLAUDE5_QUESTION_LIMITS)?;
                let answers = ask(ctx, questions).await?;
                format_anthropic_answers(&answers)
            })
        })).with_source(ToolSource::Native)
        .requires_human_input()
}

/// The arguments as the tool's own wire shape.
fn parse_tool_args<T: for<'de> Deserialize<'de>>(args: Value) -> Result<T, ToolError> {
    serde_json::from_value(args).map_err(|error| {
        ToolError::with_source(
            ToolErrorKind::InvalidArguments,
            format!("invalid question tool arguments: {error}"),
            error,
        )
    })
}

/// Puts the questions to the person and waits.
///
/// The four refusals ahead of the call are the ones that make this tool
/// root-only and application-provided: a call outside a session, a call from a
/// child, a call the provider could not be correlated with, and a session with
/// nobody to ask.
async fn ask(ctx: ToolContext, questions: Vec<Question>) -> Result<Vec<Answer>, ToolError> {
    let session_id = ctx
        .session_id
        .as_deref()
        .ok_or_else(|| ToolError::unavailable(ROOT_SESSION_REQUIRED_ERROR))?;
    let root_session_id = ctx
        .root_session_id
        .as_deref()
        .ok_or_else(|| ToolError::unavailable(ROOT_SESSION_REQUIRED_ERROR))?;
    if session_id != root_session_id {
        return Err(ToolError::denied(
            "human-question tools are only available to the root agent; subagents must report \
             back to their parent",
        ));
    }
    let tool_call_id = ctx.tool_call_id.as_deref().ok_or_else(|| {
        ToolError::execution("human-question tool call is missing a provider tool_call_id")
    })?;
    let provider = ctx.human_input.as_ref().ok_or_else(|| {
        ToolError::unavailable(
            "human-question tools are available only where the application gave the session \
             someone to ask",
        )
    })?;

    Ok(provider
        .ask_questions(tool_call_id, questions, ctx.cancel.clone())
        .await?)
}

/// The OpenAI payload as pebble's own questions.
fn normalize_openai_questions(args: OpenAiQuestionToolArgs) -> Result<Vec<Question>, ToolError> {
    if args.questions.is_empty() {
        return Err(ToolError::invalid_arguments(
            "questions must contain at least one question",
        ));
    }
    args.questions
        .into_iter()
        .map(|question| {
            let original_question = question.question.trim().to_owned();
            Ok(Question {
                original_id: Some(non_empty(&question.id, "question id")?),
                text: display_text(Some(question.header.as_str()), &question.question),
                header: Some(question.header),
                original_question,
                kind: QuestionKind::MultipleChoice,
                options: options_from_openai(question.options),
                allow_freeform: true,
            })
        })
        .collect()
}

/// The Anthropic payload as pebble's own questions, under `limits`.
fn normalize_anthropic_questions(
    args: AnthropicQuestionToolArgs,
    limits: &QuestionLimits,
) -> Result<Vec<Question>, ToolError> {
    if !limits.questions.contains(&args.questions.len()) {
        return Err(ToolError::invalid_arguments(limits.questions_error));
    }

    args.questions
        .into_iter()
        .map(|question| {
            let original_question = non_empty(&question.question, "question")?;
            let header = if limits.require_header_and_descriptions {
                let header = non_empty(
                    question.header.as_deref().unwrap_or_default(),
                    "question header",
                )?;
                if limits
                    .max_header_chars
                    .is_some_and(|max| header.chars().count() > max)
                {
                    return Err(ToolError::invalid_arguments(format!(
                        "question header must contain at most {} characters",
                        limits.max_header_chars.unwrap_or_default()
                    )));
                }
                Some(header)
            } else {
                question.header
            };

            if let Some(bounds) = &limits.options
                && !bounds.contains(&question.options.len())
            {
                return Err(ToolError::invalid_arguments(limits.options_error));
            }
            if !limits.allow_preview_with_multi_select
                && question.multi_select
                && question
                    .options
                    .iter()
                    .any(|option| option.preview.is_some())
            {
                return Err(ToolError::invalid_arguments(
                    "option previews are not supported for multi-select questions",
                ));
            }

            let text = display_text(header.as_deref(), &original_question);

            Ok(Question {
                original_id: None,
                text,
                header,
                original_question,
                kind: if question.multi_select {
                    QuestionKind::MultiSelect
                } else {
                    QuestionKind::MultipleChoice
                },
                options: options_from_anthropic(question.options, limits)?,
                allow_freeform: true,
            })
        })
        .collect()
}

/// The OpenAI options as pebble's own.
fn options_from_openai(options: Vec<OpenAiOption>) -> Vec<QuestionOption> {
    options
        .into_iter()
        .enumerate()
        .map(|(index, option)| QuestionOption {
            key:         option_key(index),
            label:       option.label,
            description: option
                .description
                .map(|value| bounded_display_field(&value, OPTION_DESCRIPTION_MAX_CHARS)),
            preview:     None,
        })
        .collect()
}

/// The Anthropic options as pebble's own, under `limits`.
fn options_from_anthropic(
    options: Vec<AnthropicOption>,
    limits: &QuestionLimits,
) -> Result<Vec<QuestionOption>, ToolError> {
    options
        .into_iter()
        .enumerate()
        .map(|(index, option)| {
            let (label, description) = if limits.require_header_and_descriptions {
                (
                    non_empty(&option.label, "option label")?,
                    Some(non_empty(
                        option.description.as_deref().unwrap_or_default(),
                        "option description",
                    )?),
                )
            } else {
                (option.label, option.description)
            };
            Ok(QuestionOption {
                key: option_key(index),
                label,
                description: description
                    .map(|value| bounded_display_field(&value, OPTION_DESCRIPTION_MAX_CHARS)),
                preview: option
                    .preview
                    .map(|value| bounded_display_field(&value, OPTION_PREVIEW_MAX_CHARS)),
            })
        })
        .collect()
}

/// The key an option is identified by, which is its position.
fn option_key(index: usize) -> String {
    format!("option_{}", index + 1)
}

/// One trimmed field, or the error the model is given instead.
fn non_empty(value: &str, field: &str) -> Result<String, ToolError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(ToolError::invalid_arguments(format!(
            "{field} must not be empty"
        )))
    } else {
        Ok(trimmed.to_owned())
    }
}

/// The question as it should be shown, header included.
fn display_text(header: Option<&str>, question: &str) -> String {
    let header = header.map(str::trim).filter(|value| !value.is_empty());
    let question = question.trim();
    match (header, question.is_empty()) {
        (Some(header), false) => format!("{header}\n\n{question}"),
        (Some(header), true) => header.to_owned(),
        (None, false) => question.to_owned(),
        (None, true) => String::new(),
    }
}

/// The first `max_chars` characters of `value`, cut at a character boundary.
fn bounded_display_field(value: &str, max_chars: usize) -> String {
    match value.char_indices().nth(max_chars) {
        Some((byte_index, _)) => value[..byte_index].to_owned(),
        None => value.to_owned(),
    }
}

/// Refuses the round when any question came back unanswered.
///
/// The model asked because it could not proceed, so an answer it did not get is
/// a failed call rather than an empty result it might act on.
fn ensure_all_answered(answers: &[Answer]) -> Result<(), ToolError> {
    if let Some(answer) = answers
        .iter()
        .find(|answer| answer.status != AnswerStatus::Answered)
    {
        return Err(ToolError::new(
            unanswered_kind(answer.status),
            format!(
                "human-question request ended before the user answered `{}`: {}",
                answer.original_question,
                answer_status_label(answer.status)
            ),
        ));
    }
    Ok(())
}

/// Why the round failed, as the completion event reports it.
const fn unanswered_kind(status: AnswerStatus) -> ToolErrorKind {
    match status {
        // A person who declines has refused the call.
        AnswerStatus::Skipped => ToolErrorKind::Denied,
        // Nothing else was a decision about the question: the prompt stopped
        // waiting for it.
        _ => ToolErrorKind::Cancelled,
    }
}

/// How an unanswered question is described to the model.
const fn answer_status_label(status: AnswerStatus) -> &'static str {
    match status {
        AnswerStatus::Answered => "answered",
        AnswerStatus::Cancelled => "cancelled",
        AnswerStatus::Interrupted => "interrupted",
        AnswerStatus::Skipped => "skipped",
        AnswerStatus::Timeout => "timed out",
    }
}

/// The answers as the OpenAI harness reads them: JSON keyed by the identifier
/// the model sent.
fn format_openai_answers(answers: &[Answer]) -> Result<String, ToolError> {
    ensure_all_answered(answers)?;
    let mut answer_map = BTreeMap::new();
    for answer in answers {
        let Some(original_id) = answer.original_id.as_ref() else {
            return Err(ToolError::execution(
                "OpenAI question answer is missing the original model question id",
            ));
        };
        answer_map.insert(original_id.clone(), json!({ "answers": answer.answers }));
    }
    serde_json::to_string(&json!({ "answers": answer_map })).map_err(|error| {
        ToolError::with_source(
            ToolErrorKind::Execution,
            format!("failed to serialize answers: {error}"),
            error,
        )
    })
}

/// The answers as the Anthropic harnesses read them: one sentence naming each
/// question and what was chosen.
fn format_anthropic_answers(answers: &[Answer]) -> Result<String, ToolError> {
    ensure_all_answered(answers)?;
    let pairs = answers
        .iter()
        .map(|answer| {
            let question = json!(answer.original_question);
            let answer_text = json!(answer.answers.join(", "));
            format!("{question}={answer_text}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "User has answered your questions: {pairs}. You can now continue with the task."
    ))
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::human_input::{HumanInputError, HumanInputProvider};
    use crate::test_support::MockEnvironment;
    use crate::tool::{ToolRegistry, ToolVocabulary};
    use crate::tools::testing::{context, schema_of};

    /// A provider that answers every question with what it was given.
    struct Scripted {
        answers: Vec<Answer>,
    }

    #[async_trait]
    impl HumanInputProvider for Scripted {
        async fn ask_questions(
            &self,
            _tool_call_id: &str,
            _questions: Vec<Question>,
            _cancel_token: CancellationToken,
        ) -> Result<Vec<Answer>, HumanInputError> {
            Ok(self.answers.clone())
        }
    }

    /// A provider that reports its own channel failing.
    struct Broken;

    #[async_trait]
    impl HumanInputProvider for Broken {
        async fn ask_questions(
            &self,
            _tool_call_id: &str,
            _questions: Vec<Question>,
            _cancel_token: CancellationToken,
        ) -> Result<Vec<Answer>, HumanInputError> {
            Err(HumanInputError::new("the interviewer disconnected"))
        }
    }

    fn answered(original_id: Option<&str>, question: &str, answers: &[&str]) -> Answer {
        Answer {
            original_id:       original_id.map(ToOwned::to_owned),
            original_question: question.to_owned(),
            answers:           answers.iter().map(|value| (*value).to_owned()).collect(),
            status:            AnswerStatus::Answered,
        }
    }

    /// A root session with `provider` to ask through.
    fn root_context(provider: Arc<dyn HumanInputProvider>) -> ToolContext {
        context(MockEnvironment::default())
            .with_session("root", "root")
            .with_tool_call_id("call_1")
            .with_human_input(provider)
    }

    fn claude5_question() -> Value {
        json!({
            "questions": [{
                "header": "Approach",
                "question": "Which approach?",
                "multiSelect": false,
                "options": [
                    {"label": "Simple", "description": "Use the simple approach."},
                    {"label": "Flexible", "description": "Use the flexible approach."}
                ]
            }]
        })
    }

    // --- normalization ---

    #[test]
    fn openai_request_with_descriptions_normalizes_to_multiple_choice() {
        let args: OpenAiQuestionToolArgs = serde_json::from_value(json!({
            "questions": [{
                "id": "q1",
                "header": "Decision",
                "question": "Which path?",
                "options": [{ "label": "Ship", "description": "Deploy now" }]
            }]
        }))
        .expect("the payload parses");

        let questions = normalize_openai_questions(args).expect("the payload normalizes");

        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].original_id.as_deref(), Some("q1"));
        assert_eq!(questions[0].kind, QuestionKind::MultipleChoice);
        assert!(questions[0].allow_freeform);
        assert_eq!(questions[0].text, "Decision\n\nWhich path?");
        assert_eq!(questions[0].options[0].key, "option_1");
        assert_eq!(questions[0].options[0].label, "Ship");
        assert_eq!(
            questions[0].options[0].description.as_deref(),
            Some("Deploy now")
        );
    }

    #[test]
    fn anthropic_multiselect_preserves_preview_and_formats_comma_joined_answers() {
        let args: AnthropicQuestionToolArgs = serde_json::from_value(json!({
            "questions": [{
                "header": "Pick features",
                "question": "Which features?",
                "multiSelect": true,
                "options": [{
                    "label": "Auth",
                    "description": "Login support",
                    "preview": "auth diff"
                }]
            }]
        }))
        .expect("the payload parses");

        let questions = normalize_anthropic_questions(args, &ANTHROPIC_QUESTION_LIMITS)
            .expect("the payload normalizes");

        assert_eq!(questions[0].kind, QuestionKind::MultiSelect);
        assert_eq!(
            questions[0].options[0].preview.as_deref(),
            Some("auth diff")
        );

        let text =
            format_anthropic_answers(&[answered(None, "Which features?", &["Auth", "Billing"])])
                .expect("the answers render");
        assert!(text.contains("\"Which features?\"=\"Auth, Billing\""));
    }

    #[test]
    fn openai_answers_are_keyed_by_original_model_question_id() {
        let text = format_openai_answers(&[
            answered(Some("first"), "First?", &["Yes"]),
            answered(Some("second"), "Second?", &["No"]),
        ])
        .expect("the answers render");

        assert_eq!(
            serde_json::from_str::<Value>(&text).expect("the answers are JSON"),
            json!({
                "answers": {
                    "first": { "answers": ["Yes"] },
                    "second": { "answers": ["No"] }
                }
            })
        );
    }

    #[test]
    fn option_description_and_preview_are_bounded() {
        let long = "x".repeat(OPTION_PREVIEW_MAX_CHARS + 10);

        assert_eq!(
            bounded_display_field(&long, OPTION_DESCRIPTION_MAX_CHARS)
                .chars()
                .count(),
            OPTION_DESCRIPTION_MAX_CHARS
        );
        assert_eq!(
            bounded_display_field(&long, OPTION_PREVIEW_MAX_CHARS)
                .chars()
                .count(),
            OPTION_PREVIEW_MAX_CHARS
        );
    }

    /// The bound counts characters, so a field of multi-byte characters is cut
    /// where a character ends rather than inside one.
    #[test]
    fn a_bounded_field_is_cut_at_a_character_boundary() {
        let long = "é".repeat(OPTION_DESCRIPTION_MAX_CHARS + 5);

        let bounded = bounded_display_field(&long, OPTION_DESCRIPTION_MAX_CHARS);

        assert_eq!(bounded.chars().count(), OPTION_DESCRIPTION_MAX_CHARS);
        assert_eq!(bounded.len(), OPTION_DESCRIPTION_MAX_CHARS * 2);
    }

    // --- registration ---

    #[test]
    fn each_harness_gets_the_question_tool_it_expects() {
        let registered = |profile: AgentProfileKind, vocabulary: ToolVocabulary| {
            let mut registry = ToolRegistry::with_vocabulary(vocabulary);
            if let Some(tool) = make_question_tool(profile) {
                registry.register(tool);
            }
            registry
        };

        let openai = registered(AgentProfileKind::OpenAi, ToolVocabulary::Canonical);
        assert!(openai.get("request_user_input").is_some());
        assert!(openai.get("AskUserQuestion").is_none());

        let gpt56 = registered(AgentProfileKind::Gpt56, ToolVocabulary::Codex);
        assert!(gpt56.get("request_user_input").is_some());
        assert!(gpt56.get("AskUserQuestion").is_none());

        let anthropic = registered(AgentProfileKind::Anthropic, ToolVocabulary::Canonical);
        assert!(anthropic.get("AskUserQuestion").is_some());
        assert!(anthropic.get("request_user_input").is_none());

        let kimi = registered(AgentProfileKind::Kimi, ToolVocabulary::KimiCode);
        assert!(kimi.get("AskUserQuestion").is_some());
        assert!(kimi.get("request_user_input").is_none());

        let gemini = registered(AgentProfileKind::Gemini, ToolVocabulary::Canonical);
        assert!(gemini.names().is_empty(), "Gemini's harness asks nobody");
    }

    #[test]
    fn claude5_is_given_the_strict_schema() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::Claude5);
        registry.register(make_question_tool(AgentProfileKind::Claude5).expect("Claude 5 can ask"));

        let tool = registry.get("AskUserQuestion").expect("registered");
        let schema = schema_of(tool);
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(
            schema["properties"]
                .as_object()
                .expect("the schema has properties")
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["questions"]
        );
        assert_eq!(schema["properties"]["questions"]["maxItems"], json!(4));
        assert!(registry.get("request_user_input").is_none());
    }

    /// Both question tools keep their names in every vocabulary, so a session
    /// never advertises one under a name the model was not trained on.
    #[test]
    fn the_question_tools_keep_their_names_in_every_vocabulary() {
        for vocabulary in ToolVocabulary::ALL.iter().copied() {
            assert_eq!(
                NativeTool::AskUserQuestion.name(vocabulary),
                "AskUserQuestion"
            );
            assert_eq!(
                NativeTool::RequestUserInput.name(vocabulary),
                "request_user_input"
            );
        }
    }

    // --- the Claude 5 contract ---

    #[test]
    fn claude5_question_contract_is_strict_and_preserves_preview() {
        let args: AnthropicQuestionToolArgs = serde_json::from_value(json!({
            "questions": [{
                "header": "Approach",
                "question": "Which approach should we use?",
                "multiSelect": false,
                "options": [
                    {
                        "label": "Simple",
                        "description": "Use the smallest implementation.",
                        "preview": "fn simple() {}"
                    },
                    {
                        "label": "Flexible",
                        "description": "Allow future extension."
                    }
                ]
            }]
        }))
        .expect("the payload parses");

        let questions = normalize_anthropic_questions(args, &CLAUDE5_QUESTION_LIMITS)
            .expect("the payload normalizes");

        assert_eq!(questions[0].header.as_deref(), Some("Approach"));
        assert_eq!(
            questions[0].options[0].preview.as_deref(),
            Some("fn simple() {}")
        );
        assert!(questions[0].allow_freeform);
    }

    /// The Claude 5 payload is read through the lenient struct, so every rule
    /// its own struct used to enforce is the normalizer's to enforce now.
    #[test]
    fn claude5_limits_reject_what_the_lenient_contract_allows() {
        let question = |patch: Value| {
            let mut base = json!({
                "question": "Which approach?",
                "header": "Approach",
                "multiSelect": false,
                "options": [
                    {"label": "First", "description": "One"},
                    {"label": "Second", "description": "Two"}
                ]
            });
            let object = base.as_object_mut().expect("the question is an object");
            for (key, value) in patch.as_object().expect("the patch is an object") {
                if value.is_null() {
                    object.remove(key);
                } else {
                    object.insert(key.clone(), value.clone());
                }
            }
            base
        };
        let normalize = |questions: Value| {
            let args: AnthropicQuestionToolArgs =
                serde_json::from_value(json!({"questions": questions}))
                    .expect("the payload parses");
            normalize_anthropic_questions(args, &CLAUDE5_QUESTION_LIMITS)
        };

        // A missing header and a missing option description used to be caught
        // by serde; the normalizer has to reject them now.
        assert!(normalize(json!([question(json!({"header": null}))])).is_err());
        assert!(
            normalize(json!([question(json!({
                "options": [{"label": "First"}, {"label": "Second"}]
            }))]))
            .is_err()
        );

        assert!(
            normalize(json!([question(json!({"header": "ThirteenChars"}))])).is_err(),
            "a header longer than twelve characters"
        );
        assert!(
            normalize(json!([question(json!({
                "options": [{"label": "Only", "description": "One"}]
            }))]))
            .is_err(),
            "fewer than two options"
        );
        assert!(
            normalize(json!(vec![question(json!({})); 5])).is_err(),
            "more than four questions"
        );

        assert!(normalize(json!([question(json!({}))])).is_ok());
    }

    /// The same payloads stay acceptable under the lenient contract, so the
    /// shared normalizer has not tightened the Anthropic tool.
    #[test]
    fn anthropic_limits_still_accept_optional_headers_and_descriptions() {
        let args: AnthropicQuestionToolArgs = serde_json::from_value(json!({
            "questions": [{
                "question": "Which approach?",
                "options": [{"label": "First"}]
            }]
        }))
        .expect("the payload parses");

        let questions = normalize_anthropic_questions(args, &ANTHROPIC_QUESTION_LIMITS)
            .expect("the payload normalizes");

        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].header, None);
        assert_eq!(questions[0].options[0].description, None);
    }

    #[test]
    fn claude5_rejects_previews_for_multi_select_questions() {
        let args: AnthropicQuestionToolArgs = serde_json::from_value(json!({
            "questions": [{
                "header": "Features",
                "question": "Which features should we enable?",
                "multiSelect": true,
                "options": [
                    {
                        "label": "Auth",
                        "description": "Enable authentication.",
                        "preview": "auth = true"
                    },
                    {"label": "Metrics", "description": "Enable metrics."}
                ]
            }]
        }))
        .expect("the payload parses");

        assert!(normalize_anthropic_questions(args, &CLAUDE5_QUESTION_LIMITS).is_err());
    }

    // --- the flow ---

    #[tokio::test]
    async fn a_root_session_puts_its_questions_to_the_person() {
        let tool = make_claude5_question_tool();
        let provider = Arc::new(Scripted {
            answers: vec![answered(None, "Which approach?", &["Simple"])],
        });

        let output = (tool.executor)(claude5_question(), root_context(provider))
            .await
            .expect("the person answered");

        assert_eq!(
            output,
            "User has answered your questions: \"Which approach?\"=\"Simple\". You can now \
             continue with the task."
        );
    }

    /// The second layer of the root-only rule. A child is built without a
    /// provider, so it never registers a question tool; this is what answers if
    /// one somehow reaches a child anyway.
    #[tokio::test]
    async fn claude5_question_tool_rejects_subagent_sessions() {
        let tool = make_claude5_question_tool();
        let child = context(MockEnvironment::default())
            .with_session("child", "root")
            .with_tool_call_id("call")
            .with_human_input(Arc::new(Scripted {
                answers: Vec::new(),
            }));

        let error = (tool.executor)(claude5_question(), child)
            .await
            .expect_err("a child asks its parent, not a person");

        assert!(
            error.message().contains("only available to the root agent"),
            "{}",
            error.message()
        );
        assert_eq!(error.kind(), ToolErrorKind::Denied);
    }

    #[tokio::test]
    async fn a_session_with_nobody_to_ask_says_so() {
        let tool = make_anthropic_question_tool();
        let context = context(MockEnvironment::default())
            .with_session("root", "root")
            .with_tool_call_id("call");

        let error = (tool.executor)(
            json!({"questions": [{
                "question": "Ship it?",
                "options": [{"label": "Yes"}],
                "multiSelect": false
            }]}),
            context,
        )
        .await
        .expect_err("there is nobody to ask");

        assert_eq!(error.kind(), ToolErrorKind::Unavailable);
        assert!(error.message().contains("someone to ask"), "{error:?}");
    }

    #[tokio::test]
    async fn a_question_asked_outside_a_session_is_unavailable() {
        let tool = make_anthropic_question_tool();

        let error = (tool.executor)(
            json!({"questions": [{
                "question": "Ship it?",
                "options": [{"label": "Yes"}],
                "multiSelect": false
            }]}),
            context(MockEnvironment::default()),
        )
        .await
        .expect_err("there is no session");

        assert_eq!(error.message(), ROOT_SESSION_REQUIRED_ERROR);
        assert_eq!(error.kind(), ToolErrorKind::Unavailable);
    }

    /// The model asked because it could not proceed, so a question that came
    /// back unanswered fails the call rather than returning nothing.
    #[tokio::test]
    async fn a_question_nobody_answered_fails_the_call() {
        let tool = make_anthropic_question_tool();
        let provider = Arc::new(Scripted {
            answers: vec![Answer {
                original_id:       None,
                original_question: "Ship it?".to_owned(),
                answers:           Vec::new(),
                status:            AnswerStatus::Timeout,
            }],
        });

        let error = (tool.executor)(
            json!({"questions": [{
                "question": "Ship it?",
                "options": [{"label": "Yes"}],
                "multiSelect": false
            }]}),
            root_context(provider),
        )
        .await
        .expect_err("nobody answered");

        assert_eq!(
            error.message(),
            "human-question request ended before the user answered `Ship it?`: timed out"
        );
        assert_eq!(error.kind(), ToolErrorKind::Cancelled);
    }

    #[test]
    fn a_declined_question_is_reported_as_a_refusal() {
        let error = ensure_all_answered(&[Answer {
            original_id:       None,
            original_question: "Ship it?".to_owned(),
            answers:           Vec::new(),
            status:            AnswerStatus::Skipped,
        }])
        .expect_err("the person declined");

        assert_eq!(
            error.message(),
            "human-question request ended before the user answered `Ship it?`: skipped"
        );
        assert_eq!(error.kind(), ToolErrorKind::Denied);
    }

    #[tokio::test]
    async fn a_broken_question_channel_is_an_execution_failure() {
        let tool = make_openai_question_tool();

        let error = (tool.executor)(
            json!({"questions": [{
                "id": "q1",
                "header": "Ship",
                "question": "Ship it?",
                "options": [{"label": "Yes"}]
            }]}),
            root_context(Arc::new(Broken)),
        )
        .await
        .expect_err("the channel failed");

        assert_eq!(
            error.message(),
            "Could not ask the user: the interviewer disconnected"
        );
        assert_eq!(error.kind(), ToolErrorKind::Execution);
    }

    #[tokio::test]
    async fn arguments_that_are_not_questions_are_refused_by_shape() {
        let tool = make_openai_question_tool();

        let error = (tool.executor)(
            json!({"questions": [{"header": "Ship"}]}),
            root_context(Arc::new(Scripted {
                answers: Vec::new(),
            })),
        )
        .await
        .expect_err("the question has no text and no id");

        assert!(
            error
                .message()
                .starts_with("invalid question tool arguments:"),
            "{}",
            error.message()
        );
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }
}
