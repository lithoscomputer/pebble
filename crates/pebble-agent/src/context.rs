//! Context transformation before a model turn.

use std::error::Error as StdError;
use std::fmt;
use std::result::Result as StdResult;

use async_trait::async_trait;
use lithos_llm::types::Message;
use tokio_util::sync::CancellationToken;

/// Mutable conversation state presented to a context transformation.
///
/// A transformation can summarize, remove, or inject messages before the next
/// request. It must preserve valid tool-call and tool-result pairing.
pub struct TransformContext<'a> {
    model:    &'a str,
    messages: &'a mut Vec<Message>,
}

impl<'a> TransformContext<'a> {
    pub(crate) fn new(model: &'a str, messages: &'a mut Vec<Message>) -> Self {
        Self { model, messages }
    }

    /// The model selector the next request will use.
    #[must_use]
    pub const fn model(&self) -> &str {
        self.model
    }

    /// The messages the next request will contain.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        self.messages
    }

    /// Mutates the messages the next request will contain.
    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        self.messages
    }
}

/// Prepares conversation context before each model turn.
#[async_trait]
pub trait ContextTransform: Send + Sync {
    /// Transforms the current context.
    ///
    /// Implementations must stop promptly when `cancel` is cancelled.
    async fn transform(
        &self,
        context: TransformContext<'_>,
        cancel: &CancellationToken,
    ) -> StdResult<(), ContextTransformError>;
}

/// A context transformation failure with an optional source chain.
#[derive(Debug)]
pub struct ContextTransformError {
    message: String,
    source:  Option<Box<dyn StdError + Send + Sync>>,
}

impl ContextTransformError {
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

impl fmt::Display for ContextTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for ContextTransformError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}
