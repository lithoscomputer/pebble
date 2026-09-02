//! Lifecycle stages at stable model-turn boundaries.

use std::error::Error as StdError;
use std::fmt;
use std::result::Result as StdResult;

use async_trait::async_trait;
use lithos_llm::types::{Message, Response};
use tokio_util::sync::CancellationToken;

use crate::agent::UserMessage;

/// What the agent does after a model turn answers without tool calls.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum AfterAnswerAction {
    /// Finish the prompt with this answer.
    #[default]
    Complete,
    /// Start another model turn without committing another message.
    Continue,
    /// Commit another user message and start another model turn.
    ContinueWith(UserMessage),
}

/// An immutable view of the conversation at one model turn.
#[derive(Clone, Copy, Debug)]
pub struct TurnContext<'a> {
    model:    &'a str,
    turn:     usize,
    messages: &'a [Message],
}

impl<'a> TurnContext<'a> {
    pub(crate) const fn new(model: &'a str, turn: usize, messages: &'a [Message]) -> Self {
        Self {
            model,
            turn,
            messages,
        }
    }

    /// The model selector for this turn.
    #[must_use]
    pub const fn model(&self) -> &str {
        self.model
    }

    /// The zero-based model-turn number in the current prompt.
    #[must_use]
    pub const fn turn(&self) -> usize {
        self.turn
    }

    /// The committed conversation at this boundary.
    #[must_use]
    pub const fn messages(&self) -> &[Message] {
        self.messages
    }
}

/// A typed conversation change returned from a lifecycle stage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConversationUpdate {
    replacement: Option<Vec<Message>>,
}

impl ConversationUpdate {
    /// Leaves the canonical conversation unchanged.
    #[must_use]
    pub const fn unchanged() -> Self {
        Self { replacement: None }
    }

    /// Replaces the canonical conversation at a stable turn boundary.
    ///
    /// The replacement must preserve tool-call and tool-result pairing.
    #[must_use]
    pub fn replace(messages: Vec<Message>) -> Self {
        Self {
            replacement: Some(messages),
        }
    }

    pub(crate) fn apply(self, messages: &mut Vec<Message>) -> bool {
        if let Some(replacement) = self.replacement {
            *messages = replacement;
            true
        } else {
            false
        }
    }
}

/// Work performed at stable model-turn boundaries.
///
/// The default methods do nothing. A coding layer can compact history before
/// or after a model turn and can return a background result after a natural
/// answer. Returning a message from [`after_answer`](Self::after_answer)
/// continues the same prompt with that message.
#[async_trait]
pub trait AgentLifecycle: Send + Sync {
    /// Runs after steering is committed and before tools are resolved.
    async fn before_model(
        &self,
        _context: TurnContext<'_>,
        _cancel: &CancellationToken,
    ) -> StdResult<ConversationUpdate, LifecycleError> {
        Ok(ConversationUpdate::unchanged())
    }

    /// Runs after the assistant response is committed.
    ///
    /// An error ends the prompt, but only after the tool calls the response
    /// made are answered as `Cancelled` without running, so the committed
    /// turn is never left without its results.
    async fn after_model(
        &self,
        _context: TurnContext<'_>,
        _response: &Response,
        _cancel: &CancellationToken,
    ) -> StdResult<ConversationUpdate, LifecycleError> {
        Ok(ConversationUpdate::unchanged())
    }

    /// Runs before a natural answer completes the prompt.
    ///
    /// The returned action decides whether the answer completes the prompt.
    async fn after_answer(
        &self,
        _context: TurnContext<'_>,
        _response: &Response,
        _cancel: &CancellationToken,
    ) -> StdResult<AfterAnswerAction, LifecycleError> {
        Ok(AfterAnswerAction::Complete)
    }
}

/// A lifecycle-stage failure with an optional source chain.
#[derive(Debug)]
pub struct LifecycleError {
    message: String,
    source:  Option<Box<dyn StdError + Send + Sync>>,
}

impl LifecycleError {
    /// Creates a failure with no lower-level source.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source:  None,
        }
    }

    /// Creates a failure that preserves its lower-level source.
    pub fn with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source:  Some(Box::new(source)),
        }
    }
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for LifecycleError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}
