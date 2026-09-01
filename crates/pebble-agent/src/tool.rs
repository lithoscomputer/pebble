//! Generic tools exposed to an agent.

use std::error::Error as StdError;
use std::fmt;
use std::future::Future;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::{ContentPart, ToolDefinition};
use serde_json::Value;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::event::AgentEvent;

/// The context supplied to one tool call.
#[derive(Clone)]
pub struct ToolContext {
    tool_call_id: String,
    tool_name:    String,
    cancel:       CancellationToken,
    events:       broadcast::Sender<AgentEvent>,
}

impl ToolContext {
    pub(crate) fn new(
        tool_call_id: String,
        tool_name: String,
        cancel: CancellationToken,
        events: broadcast::Sender<AgentEvent>,
    ) -> Self {
        Self {
            tool_call_id,
            tool_name,
            cancel,
            events,
        }
    }

    /// The provider's identifier for this call.
    #[must_use]
    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    /// The registered tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// The cooperative cancellation signal for this call.
    #[must_use]
    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancel
    }

    /// Publishes incremental output for observers.
    ///
    /// This does not add the fragment to the result returned to the model.
    pub fn emit_output_delta(&self, delta: impl Into<String>) {
        let _ = self.events.send(AgentEvent::ToolOutputDelta {
            tool_call_id: self.tool_call_id.clone(),
            delta:        delta.into(),
        });
    }
}

impl fmt::Debug for ToolContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolContext")
            .field("tool_call_id", &self.tool_call_id)
            .field("tool_name", &self.tool_name)
            .field("cancelled", &self.cancel.is_cancelled())
            .finish_non_exhaustive()
    }
}

/// Executes one model-requested tool call.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Executes the call.
    async fn execute(
        &self,
        context: ToolContext,
        arguments: Value,
    ) -> StdResult<ToolOutput, ToolError>;
}

/// A model-visible tool definition paired with its executor.
#[derive(Clone)]
pub struct Tool {
    definition: ToolDefinition,
    executor:   Arc<dyn ToolExecutor>,
}

impl Tool {
    /// Pairs an existing definition with an executor.
    #[must_use]
    pub fn new(definition: ToolDefinition, executor: Arc<dyn ToolExecutor>) -> Self {
        Self {
            definition,
            executor,
        }
    }

    /// Defines a function tool with an asynchronous closure.
    pub fn function<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(ToolContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = StdResult<ToolOutput, ToolError>> + Send + 'static,
    {
        Self::new(
            ToolDefinition::function(name, description, input_schema),
            Arc::new(FunctionExecutor { execute }),
        )
    }

    /// The definition sent to the model.
    #[must_use]
    pub const fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    pub(crate) async fn execute(
        &self,
        context: ToolContext,
        arguments: Value,
    ) -> StdResult<ToolOutput, ToolError> {
        self.executor.execute(context, arguments).await
    }
}

impl fmt::Debug for Tool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tool")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}

struct FunctionExecutor<F> {
    execute: F,
}

#[async_trait]
impl<F, Fut> ToolExecutor for FunctionExecutor<F>
where
    F: Fn(ToolContext, Value) -> Fut + Send + Sync,
    Fut: Future<Output = StdResult<ToolOutput, ToolError>> + Send + 'static,
{
    async fn execute(
        &self,
        context: ToolContext,
        arguments: Value,
    ) -> StdResult<ToolOutput, ToolError> {
        (self.execute)(context, arguments).await
    }
}

/// Content returned from a successful tool execution.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutput {
    content: Vec<ContentPart>,
}

impl ToolOutput {
    /// Creates output from provider-neutral content parts.
    #[must_use]
    pub fn new(content: impl IntoIterator<Item = ContentPart>) -> Self {
        Self {
            content: content.into_iter().collect(),
        }
    }

    /// The content returned to the model.
    #[must_use]
    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    pub(crate) fn into_content(self) -> Vec<ContentPart> {
        self.content
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::new([ContentPart::Text { text }])
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

/// A tool execution failure returned to the model as an error result.
#[derive(Debug)]
pub struct ToolError {
    message: String,
    source:  Option<Box<dyn StdError + Send + Sync>>,
}

impl ToolError {
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

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for ToolError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}
