//! Generic tools exposed to an agent.

mod system;

use std::error::Error as StdError;
use std::fmt;
use std::future::Future;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::{ContentPart, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub use self::system::{
    ToolCallNext, ToolCallRequest, ToolCatalog, ToolDescriptor, ToolDiscoveryContext,
    ToolDiscoveryNext, ToolId, ToolIdError, ToolMiddleware, ToolOutcome, ToolScheduling,
    ToolService, ToolSystem, ToolSystemError,
};

/// Why a tool call failed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolErrorKind {
    /// The call arguments did not match the tool schema.
    InvalidArguments,
    /// The tool refused the call after the round allowed execution.
    Denied,
    /// The call was cancelled before it completed.
    Cancelled,
    /// No resolved tool matched the requested name.
    Unavailable,
    /// The tool ran and returned an error.
    Execution,
}

/// The context supplied to one tool call.
#[derive(Clone)]
pub struct ToolContext {
    request: ToolCallRequest,
}

impl ToolContext {
    const fn new(request: ToolCallRequest) -> Self {
        Self { request }
    }

    /// The provider's identifier for this call.
    #[must_use]
    pub fn tool_call_id(&self) -> &str {
        &self.request.call().id
    }

    /// The registered tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.request.call().name
    }

    /// The cooperative cancellation signal for this call.
    #[must_use]
    pub const fn cancellation(&self) -> &CancellationToken {
        self.request.cancellation()
    }

    /// Publishes incremental output for observers.
    ///
    /// This does not add the fragment to the result returned to the model.
    pub fn emit_output_delta(&self, delta: impl Into<String>) {
        self.request.emit_output_delta(delta);
    }
}

impl fmt::Debug for ToolContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolContext")
            .field("tool_call_id", &self.tool_call_id())
            .field("tool_name", &self.tool_name())
            .field("cancelled", &self.cancellation().is_cancelled())
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

/// A described tool paired with its executor.
#[derive(Clone)]
pub struct Tool {
    descriptor: ToolDescriptor,
    executor:   Arc<dyn ToolExecutor>,
}

impl Tool {
    /// Pairs an existing descriptor with an executor.
    #[must_use]
    pub fn new(descriptor: ToolDescriptor, executor: Arc<dyn ToolExecutor>) -> Self {
        Self {
            descriptor,
            executor,
        }
    }

    /// Defines a function tool whose stable identity is its initial name.
    ///
    /// # Errors
    ///
    /// Returns [`ToolIdError`] when `name` is blank.
    pub fn function<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        execute: F,
    ) -> StdResult<Self, ToolIdError>
    where
        F: Fn(ToolContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = StdResult<ToolOutput, ToolError>> + Send + 'static,
    {
        let name = name.into();
        let id = ToolId::try_new(name.clone())?;
        Ok(Self::new(
            ToolDescriptor::new(
                id,
                ToolDefinition::function(name, description, input_schema),
            ),
            Arc::new(FunctionExecutor { execute }),
        ))
    }

    /// Sets how calls to this tool are scheduled.
    #[must_use]
    pub fn with_scheduling(mut self, scheduling: ToolScheduling) -> Self {
        self.descriptor = self.descriptor.with_scheduling(scheduling);
        self
    }

    /// The stable identity, model definition, and scheduling rule.
    #[must_use]
    pub const fn descriptor(&self) -> &ToolDescriptor {
        &self.descriptor
    }

    /// The definition sent to the model.
    #[must_use]
    pub const fn definition(&self) -> &ToolDefinition {
        self.descriptor.definition()
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
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

pub(crate) struct StaticToolService {
    tools: Vec<Tool>,
}

impl StaticToolService {
    pub(crate) const fn new(tools: Vec<Tool>) -> Self {
        Self { tools }
    }
}

#[async_trait]
impl ToolService for StaticToolService {
    async fn discover(
        &self,
        _context: ToolDiscoveryContext<'_>,
    ) -> StdResult<ToolCatalog, ToolSystemError> {
        Ok(ToolCatalog::new(
            self.tools.iter().map(|tool| tool.descriptor.clone()),
        ))
    }

    async fn call(&self, request: ToolCallRequest) -> StdResult<ToolOutcome, ToolSystemError> {
        let Some(tool) = self
            .tools
            .iter()
            .find(|tool| tool.descriptor.id() == request.descriptor().id())
        else {
            return Ok(ToolOutcome::failure(
                ToolErrorKind::Unavailable,
                format!("unknown tool `{}`", request.call().name),
            ));
        };
        let arguments = request.call().arguments.clone();
        Ok(
            match tool.execute(ToolContext::new(request), arguments).await {
                Ok(output) => ToolOutcome::success(output),
                Err(error) => ToolOutcome::failure(ToolErrorKind::Execution, error.to_string()),
            },
        )
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
