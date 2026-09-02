//! Generic tools exposed to an agent.

use std::error::Error as StdError;
use std::fmt;
use std::future::Future;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::{ContentPart, ToolCall, ToolDefinition, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::event::{AgentEvent, EventHub};
use crate::turn::TurnContext;

/// Resolves tools from the current conversation before each model turn.
pub trait ToolProvider: Send + Sync {
    /// Returns the tools available for this turn.
    fn tools_for_turn(&self, context: TurnContext<'_>) -> Vec<Tool>;
}

impl<F> ToolProvider for F
where
    F: for<'a> Fn(TurnContext<'a>) -> Vec<Tool> + Send + Sync,
{
    fn tools_for_turn(&self, context: TurnContext<'_>) -> Vec<Tool> {
        self(context)
    }
}

/// What an access policy decided for one tool in one turn.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ToolAccess {
    /// Advertise and execute the tool.
    #[default]
    Allowed,
    /// Do not advertise or execute the tool.
    Denied {
        /// The explanation returned if the model still requests the tool.
        reason: String,
    },
}

/// The input to a per-turn tool access decision.
#[derive(Clone, Copy, Debug)]
pub struct ToolAccessContext<'a> {
    turn:       TurnContext<'a>,
    definition: &'a ToolDefinition,
}

impl<'a> ToolAccessContext<'a> {
    pub(crate) const fn new(turn: TurnContext<'a>, definition: &'a ToolDefinition) -> Self {
        Self { turn, definition }
    }

    /// The model turn being prepared.
    #[must_use]
    pub const fn turn(&self) -> TurnContext<'a> {
        self.turn
    }

    /// The tool definition under consideration.
    #[must_use]
    pub const fn definition(&self) -> &ToolDefinition {
        self.definition
    }
}

/// Decides which resolved tools may be advertised and executed in each turn.
pub trait ToolAccessPolicy: Send + Sync {
    /// Returns this turn's access for one tool.
    fn access(&self, context: ToolAccessContext<'_>) -> ToolAccess;
}

impl<F> ToolAccessPolicy for F
where
    F: for<'a> Fn(ToolAccessContext<'a>) -> ToolAccess + Send + Sync,
{
    fn access(&self, context: ToolAccessContext<'_>) -> ToolAccess {
        self(context)
    }
}

/// The context supplied to tool-call hooks.
#[derive(Clone, Copy, Debug)]
pub struct ToolCallContext<'a> {
    turn: usize,
    call: &'a ToolCall,
}

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

/// The completed result supplied to an after-call hook.
#[derive(Clone, Copy, Debug)]
pub struct ToolCallOutcome<'a> {
    result:     &'a ToolResult,
    error_kind: Option<ToolErrorKind>,
}

impl<'a> ToolCallOutcome<'a> {
    pub(crate) const fn new(result: &'a ToolResult, error_kind: Option<ToolErrorKind>) -> Self {
        Self { result, error_kind }
    }

    /// The result committed to conversation history.
    #[must_use]
    pub const fn result(&self) -> &ToolResult {
        self.result
    }

    /// Why the call failed, or `None` when it succeeded.
    #[must_use]
    pub const fn error_kind(&self) -> Option<ToolErrorKind> {
        self.error_kind
    }
}

impl<'a> ToolCallContext<'a> {
    pub(crate) const fn new(turn: usize, call: &'a ToolCall) -> Self {
        Self { turn, call }
    }

    /// The zero-based model turn that requested this call.
    #[must_use]
    pub const fn turn(&self) -> usize {
        self.turn
    }

    /// The requested call.
    #[must_use]
    pub const fn call(&self) -> &ToolCall {
        self.call
    }
}

/// What a before-call hook decided.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum BeforeToolCall {
    /// Continue to argument validation and execution.
    #[default]
    Proceed,
    /// Return an error result without executing the tool.
    Block {
        /// The explanation returned to the model.
        reason: String,
    },
}

/// Hooks around each tool call.
///
/// Completion events are projected before
/// [`after_tool_call`](Self::after_tool_call) runs. The after hook therefore
/// observes the same final result as event consumers and the next model turn.
#[async_trait]
pub trait ToolCallHooks: Send + Sync {
    /// Runs after access policy and before argument validation.
    async fn before_tool_call(
        &self,
        _context: ToolCallContext<'_>,
        _cancel: &CancellationToken,
    ) -> BeforeToolCall {
        BeforeToolCall::Proceed
    }

    /// Runs after the call's completion event is projected.
    async fn after_tool_call(
        &self,
        _context: ToolCallContext<'_>,
        _outcome: ToolCallOutcome<'_>,
        _cancel: &CancellationToken,
    ) {
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ToolRoundAccess {
    name:   String,
    access: ToolAccess,
}

impl ToolRoundAccess {
    pub(crate) fn new(name: String, access: ToolAccess) -> Self {
        Self { name, access }
    }
}

/// The input to a custom executor for one complete tool round.
#[derive(Clone, Copy)]
pub struct ToolRoundContext<'a> {
    turn:       usize,
    calls:      &'a [ToolCall],
    tools:      &'a [ToolDefinition],
    access:     &'a [ToolRoundAccess],
    tool_hooks: Option<&'a dyn ToolCallHooks>,
}

impl<'a> ToolRoundContext<'a> {
    pub(crate) const fn new(
        turn: usize,
        calls: &'a [ToolCall],
        tools: &'a [ToolDefinition],
        access: &'a [ToolRoundAccess],
        tool_hooks: Option<&'a dyn ToolCallHooks>,
    ) -> Self {
        Self {
            turn,
            calls,
            tools,
            access,
            tool_hooks,
        }
    }

    /// The zero-based model turn that requested these calls.
    #[must_use]
    pub const fn turn(&self) -> usize {
        self.turn
    }

    /// The calls in model order.
    #[must_use]
    pub const fn calls(&self) -> &[ToolCall] {
        self.calls
    }

    /// The definitions advertised for the turn.
    #[must_use]
    pub const fn tools(&self) -> &[ToolDefinition] {
        self.tools
    }

    /// Returns the access decision already made for a requested tool.
    ///
    /// A call for an unknown name is allowed through this gate so the round
    /// executor can return its normal unavailable-tool result.
    #[must_use]
    pub fn access_for_call(&self, call: &ToolCall) -> ToolAccess {
        self.access
            .iter()
            .find(|entry| entry.name == call.name)
            .map_or_else(ToolAccess::default, |entry| entry.access.clone())
    }

    /// Runs the configured before-call hook for `call`.
    pub async fn before_tool_call(
        &self,
        call: &ToolCall,
        cancel: &CancellationToken,
    ) -> BeforeToolCall {
        match self.tool_hooks {
            Some(hooks) => {
                hooks
                    .before_tool_call(ToolCallContext::new(self.turn, call), cancel)
                    .await
            }
            None => BeforeToolCall::Proceed,
        }
    }

    /// Runs the configured after-call hook for `call`.
    pub async fn after_tool_call(
        &self,
        call: &ToolCall,
        result: &ToolResult,
        error_kind: Option<ToolErrorKind>,
        cancel: &CancellationToken,
    ) {
        if let Some(hooks) = self.tool_hooks {
            hooks
                .after_tool_call(
                    ToolCallContext::new(self.turn, call),
                    ToolCallOutcome::new(result, error_kind),
                    cancel,
                )
                .await;
        }
    }
}

impl fmt::Debug for ToolRoundContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRoundContext")
            .field("turn", &self.turn)
            .field("calls", &self.calls)
            .field("tools", &self.tools)
            .field("has_tool_hooks", &self.tool_hooks.is_some())
            .finish_non_exhaustive()
    }
}

/// Executes a complete tool round for a specialized agent layer.
///
/// The executor returns exactly one result per call, in call order. It owns
/// detailed tool events and any layer-specific output policy. It must apply
/// [`ToolRoundContext::access_for_call`] and call the context's before and
/// after hooks around every call it executes. The generic agent still commits
/// the returned results before it observes cancellation.
///
/// A round may open with `cancel` already fired: the prompt was ended, or a
/// turn boundary failed, after the assistant turn was committed. The executor
/// must then answer every call as cancelled without starting one, because
/// the paired conversation the agent leaves behind depends on those results.
#[async_trait]
pub trait ToolRoundExecutor: Send + Sync {
    /// Executes the round.
    async fn execute_round(
        &self,
        context: ToolRoundContext<'_>,
        cancel: &CancellationToken,
    ) -> Vec<ToolResult>;
}

/// The context supplied to one tool call.
#[derive(Clone)]
pub struct ToolContext {
    tool_call_id: String,
    tool_name:    String,
    cancel:       CancellationToken,
    events:       EventHub,
}

impl ToolContext {
    pub(crate) fn new(
        tool_call_id: String,
        tool_name: String,
        cancel: CancellationToken,
        events: EventHub,
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
        self.events.emit(AgentEvent::ToolOutputDelta {
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
