//! A composable service for tool discovery and invocation.

use std::error::Error as StdError;
use std::fmt;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::{
    ContentPart, ToolCall, ToolCallKind, ToolDefinition, ToolDefinitionKind, ToolResult,
};
use tokio_util::sync::CancellationToken;

use super::{ToolContext, ToolErrorKind, ToolOutput, ToolOutputStats};
use crate::event::EventHub;
use crate::turn::TurnContext;
use crate::validation::validate_tool_arguments;

/// What a call that never ran, or was cancelled while it ran, is told.
pub(crate) const CANCELLED: &str = "Cancelled";

/// A stable tool identity that does not depend on its model-visible name.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ToolId(String);

impl ToolId {
    /// Creates an identity from a non-empty value.
    ///
    /// # Errors
    ///
    /// Returns [`ToolIdError`] when `value` is empty or contains only
    /// whitespace.
    pub fn try_new(value: impl Into<String>) -> StdResult<Self, ToolIdError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ToolIdError);
        }
        Ok(Self(value))
    }

    /// Returns the identity as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A tool identity was empty.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("a tool identity must not be empty")]
pub struct ToolIdError;

/// How the round scheduler may run a tool call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ToolScheduling {
    /// The call may run beside other concurrent calls.
    #[default]
    Concurrent,
    /// The entire round runs in model order when it contains this tool.
    Sequential,
    /// The call must be the only call executed from its round.
    ExclusiveRound,
}

/// One tool's stable identity, model-visible definition, and scheduling rule.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDescriptor {
    id:         ToolId,
    definition: ToolDefinition,
    scheduling: ToolScheduling,
}

impl ToolDescriptor {
    /// Describes one concurrently callable tool.
    #[must_use]
    pub const fn new(id: ToolId, definition: ToolDefinition) -> Self {
        Self {
            id,
            definition,
            scheduling: ToolScheduling::Concurrent,
        }
    }

    /// Sets how calls to this tool are scheduled.
    #[must_use]
    pub const fn with_scheduling(mut self, scheduling: ToolScheduling) -> Self {
        self.scheduling = scheduling;
        self
    }

    /// The stable identity used by policy and dispatch.
    #[must_use]
    pub const fn id(&self) -> &ToolId {
        &self.id
    }

    /// The definition sent to the model when middleware exposes this tool.
    #[must_use]
    pub const fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    /// How the round scheduler may run this tool.
    #[must_use]
    pub const fn scheduling(&self) -> ToolScheduling {
        self.scheduling
    }
}

/// The complete tool catalog produced for one model turn.
///
/// Middleware narrows the visible part of this catalog without discarding
/// hidden descriptors. The kernel can therefore resolve an unadvertised call
/// and send it through call middleware for a second policy check.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCatalog {
    entries: Vec<ToolCatalogEntry>,
}

#[derive(Clone, Debug, PartialEq)]
struct ToolCatalogEntry {
    descriptor: ToolDescriptor,
    visible:    bool,
}

impl ToolCatalog {
    /// Creates a catalog from tool descriptors.
    #[must_use]
    pub fn new(tools: impl IntoIterator<Item = ToolDescriptor>) -> Self {
        Self {
            entries: tools
                .into_iter()
                .map(|descriptor| ToolCatalogEntry {
                    descriptor,
                    visible: true,
                })
                .collect(),
        }
    }

    /// All tools in catalog order, including hidden tools.
    pub fn tools(&self) -> impl ExactSizeIterator<Item = &ToolDescriptor> {
        self.entries.iter().map(|entry| &entry.descriptor)
    }

    /// The tools that middleware left visible to the model.
    pub fn visible_tools(&self) -> impl Iterator<Item = &ToolDescriptor> {
        self.entries
            .iter()
            .filter(|entry| entry.visible)
            .map(|entry| &entry.descriptor)
    }

    /// Keeps accepted tools visible and hides rejected tools.
    ///
    /// A tool hidden by an inner middleware stays hidden. This makes several
    /// policy layers compose by narrowing access.
    pub fn retain(&mut self, mut predicate: impl FnMut(&ToolDescriptor) -> bool) {
        for entry in &mut self.entries {
            entry.visible &= predicate(&entry.descriptor);
        }
    }

    /// Finds a tool by its model-visible name, whether visible or hidden.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&ToolDescriptor> {
        self.entries
            .iter()
            .find(|entry| entry.descriptor.definition().name == name)
            .map(|entry| &entry.descriptor)
    }

    /// Resolves and validates one model-requested call.
    ///
    /// The returned request binds the model-visible name and call kind to the
    /// descriptor that policy and terminal dispatch use.
    pub fn resolve(
        &self,
        turn: usize,
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> StdResult<ToolCallRequest, ToolOutcome> {
        let Some(descriptor) = self.find_by_name(&call.name) else {
            return Err(ToolOutcome::failure(
                ToolErrorKind::Unavailable,
                format!("unknown tool `{}`", call.name),
            ));
        };
        let kind_matches = matches!(
            (&descriptor.definition().kind, call.kind),
            (ToolDefinitionKind::Function { .. }, ToolCallKind::Function)
                | (ToolDefinitionKind::Custom { .. }, ToolCallKind::Custom)
        );
        if !kind_matches {
            return Err(ToolOutcome::failure(
                ToolErrorKind::InvalidArguments,
                format!("tool call kind does not match `{}`", call.name),
            ));
        }
        if let Err(error) = validate_tool_arguments(&descriptor.definition().kind, &call.arguments)
        {
            return Err(ToolOutcome::failure(
                ToolErrorKind::InvalidArguments,
                error.to_string(),
            ));
        }
        Ok(ToolCallRequest::new(
            turn,
            call,
            descriptor.clone(),
            cancellation,
        ))
    }
}

/// One resolved tool invocation passed through middleware.
#[derive(Clone)]
pub struct ToolCallRequest {
    turn:         usize,
    call:         ToolCall,
    descriptor:   ToolDescriptor,
    cancellation: CancellationToken,
    events:       Option<EventHub>,
}

impl ToolCallRequest {
    const fn new(
        turn: usize,
        call: ToolCall,
        descriptor: ToolDescriptor,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            turn,
            call,
            descriptor,
            cancellation,
            events: None,
        }
    }

    /// The zero-based model turn that requested this call.
    #[must_use]
    pub const fn turn(&self) -> usize {
        self.turn
    }

    /// The call as the model requested it.
    #[must_use]
    pub const fn call(&self) -> &ToolCall {
        &self.call
    }

    /// The resolved tool.
    #[must_use]
    pub const fn descriptor(&self) -> &ToolDescriptor {
        &self.descriptor
    }

    /// The signal that asks this call to stop.
    #[must_use]
    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Takes the call and cancellation signal from this resolved request.
    #[must_use]
    pub fn into_call(self) -> (ToolCall, CancellationToken) {
        (self.call, self.cancellation)
    }

    /// Publishes incremental output for observers.
    ///
    /// This does not add the fragment to the result returned to the model.
    pub fn emit_output_delta(&self, delta: impl Into<String>) {
        super::emit_output_delta(self.events.as_ref(), &self.call.id, delta.into());
    }

    pub(crate) fn with_events(mut self, events: Option<EventHub>) -> Self {
        self.events = events;
        self
    }

    pub(crate) fn into_context_and_arguments(self) -> (ToolContext, serde_json::Value) {
        let Self {
            call,
            cancellation,
            events,
            ..
        } = self;
        let arguments = call.arguments;
        let context = ToolContext::new(call.id, call.name, cancellation, events);
        (context, arguments)
    }
}

impl fmt::Debug for ToolCallRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolCallRequest")
            .field("turn", &self.turn)
            .field("call", &self.call)
            .field("descriptor", &self.descriptor)
            .field("cancelled", &self.cancellation.is_cancelled())
            .finish_non_exhaustive()
    }
}

/// The logical result of one tool invocation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ToolOutcome {
    /// The tool completed successfully.
    Success {
        /// The content returned to the model.
        output:       ToolOutput,
        /// Output byte counts supplied by an execution layer.
        output_stats: Option<ToolOutputStats>,
    },
    /// The call was refused or failed.
    Failure {
        /// Why the call failed.
        kind:         ToolErrorKind,
        /// The safe message returned to the model.
        message:      String,
        /// Output byte counts supplied by an execution layer.
        output_stats: Option<ToolOutputStats>,
    },
}

impl ToolOutcome {
    /// A successful call.
    #[must_use]
    pub const fn success(output: ToolOutput) -> Self {
        Self::Success {
            output,
            output_stats: None,
        }
    }

    /// A failed call with a model-facing message.
    #[must_use]
    pub fn failure(kind: ToolErrorKind, message: impl Into<String>) -> Self {
        Self::Failure {
            kind,
            message: message.into(),
            output_stats: None,
        }
    }

    /// Attaches the byte counts observed while producing this outcome.
    #[must_use]
    pub const fn with_output_stats(mut self, stats: ToolOutputStats) -> Self {
        match &mut self {
            Self::Success { output_stats, .. } | Self::Failure { output_stats, .. } => {
                *output_stats = Some(stats);
            }
        }
        self
    }

    /// The output byte counts, when an execution layer supplied them.
    #[must_use]
    pub const fn output_stats(&self) -> Option<ToolOutputStats> {
        match self {
            Self::Success { output_stats, .. } | Self::Failure { output_stats, .. } => {
                *output_stats
            }
        }
    }

    /// Why this call failed, or `None` when it succeeded.
    #[must_use]
    pub const fn error_kind(&self) -> Option<ToolErrorKind> {
        match self {
            Self::Success { .. } => None,
            Self::Failure { kind, .. } => Some(*kind),
        }
    }

    /// Converts the logical outcome to the provider-neutral call result.
    #[must_use]
    pub fn into_result(self, call: &ToolCall) -> ToolResult {
        match self {
            Self::Success { output, .. } => {
                let mut content = output.into_content();
                if content.is_empty() {
                    content.push(ContentPart::Text {
                        text: String::new(),
                    });
                }
                ToolResult {
                    tool_call_id: call.id.clone(),
                    name: Some(call.name.clone()),
                    content,
                    is_error: false,
                }
            }
            Self::Failure { message, .. } => ToolResult {
                tool_call_id: call.id.clone(),
                name:         Some(call.name.clone()),
                content:      vec![ContentPart::Text { text: message }],
                is_error:     true,
            },
        }
    }
}

/// An infrastructure failure in tool discovery or middleware.
///
/// This ends the owning operation. A refusal or tool failure that the model
/// can act on is a [`ToolOutcome::Failure`] instead.
#[derive(Debug)]
pub struct ToolSystemError {
    message: String,
    source:  Option<Box<dyn StdError + Send + Sync>>,
}

impl ToolSystemError {
    /// Creates an infrastructure failure with no lower-level source.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source:  None,
        }
    }

    /// Creates an infrastructure failure that preserves its source.
    pub fn with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source:  Some(Box::new(source)),
        }
    }

    /// The safe description of the failure.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ToolSystemError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for ToolSystemError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

/// The terminal source of tool discovery and invocation.
#[async_trait]
pub trait ToolService: Send + Sync {
    /// Returns the unfiltered tools available for one model turn.
    async fn discover(&self, context: TurnContext<'_>) -> StdResult<ToolCatalog, ToolSystemError>;

    /// Invokes one resolved tool.
    async fn call(&self, request: ToolCallRequest) -> StdResult<ToolOutcome, ToolSystemError>;
}

/// Middleware around tool discovery and invocation.
///
/// The first middleware installed on a [`ToolSystem`] is outermost. Discovery
/// middleware usually delegates first and filters the returned catalog. Call
/// middleware may delegate, or return an outcome without invoking the next
/// service.
#[async_trait]
pub trait ToolMiddleware: Send + Sync {
    /// Filters or annotates the tools available for a model turn.
    async fn discover(
        &self,
        context: TurnContext<'_>,
        next: ToolDiscoveryNext<'_>,
    ) -> StdResult<ToolCatalog, ToolSystemError> {
        next.run(context).await
    }

    /// Observes, changes, or refuses one resolved invocation.
    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> StdResult<ToolOutcome, ToolSystemError> {
        next.run(request).await
    }
}

/// The rest of a tool discovery middleware chain.
#[derive(Clone, Copy)]
pub struct ToolDiscoveryNext<'a> {
    remaining: &'a [Arc<dyn ToolMiddleware>],
    terminal:  &'a dyn ToolService,
}

impl ToolDiscoveryNext<'_> {
    /// Runs the next middleware, or the terminal service at the end.
    pub async fn run(self, context: TurnContext<'_>) -> StdResult<ToolCatalog, ToolSystemError> {
        match self.remaining.split_first() {
            Some((middleware, remaining)) => {
                middleware
                    .discover(context, ToolDiscoveryNext {
                        remaining,
                        terminal: self.terminal,
                    })
                    .await
            }
            None => self.terminal.discover(context).await,
        }
    }
}

impl fmt::Debug for ToolDiscoveryNext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolDiscoveryNext")
            .field("remaining", &self.remaining.len())
            .finish_non_exhaustive()
    }
}

/// The rest of a tool call middleware chain.
#[derive(Clone, Copy)]
pub struct ToolCallNext<'a> {
    remaining: &'a [Arc<dyn ToolMiddleware>],
    terminal:  &'a dyn ToolService,
}

impl ToolCallNext<'_> {
    /// Runs the next middleware, or the terminal service at the end.
    pub async fn run(self, request: ToolCallRequest) -> StdResult<ToolOutcome, ToolSystemError> {
        match self.remaining.split_first() {
            Some((middleware, remaining)) => {
                middleware
                    .call(request, ToolCallNext {
                        remaining,
                        terminal: self.terminal,
                    })
                    .await
            }
            None => self.terminal.call(request).await,
        }
    }
}

impl fmt::Debug for ToolCallNext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolCallNext")
            .field("remaining", &self.remaining.len())
            .finish_non_exhaustive()
    }
}

/// A type-erased tool service and its ordered middleware.
#[derive(Clone)]
pub struct ToolSystem {
    terminal:   Arc<dyn ToolService>,
    middleware: Vec<Arc<dyn ToolMiddleware>>,
}

impl ToolSystem {
    /// Starts a tool system around its terminal service.
    #[must_use]
    pub fn new(terminal: Arc<dyn ToolService>) -> Self {
        Self {
            terminal,
            middleware: Vec::new(),
        }
    }

    /// Adds middleware inside every middleware already installed.
    ///
    /// The first installed middleware is outermost.
    #[must_use]
    pub fn middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.middleware.push(middleware);
        self
    }

    /// Discovers tools through the complete middleware chain.
    pub async fn discover(
        &self,
        context: TurnContext<'_>,
    ) -> StdResult<ToolCatalog, ToolSystemError> {
        ToolDiscoveryNext {
            remaining: &self.middleware,
            terminal:  self.terminal.as_ref(),
        }
        .run(context)
        .await
    }

    /// Invokes a resolved tool through the complete middleware chain.
    pub async fn call(&self, request: ToolCallRequest) -> StdResult<ToolOutcome, ToolSystemError> {
        ToolCallNext {
            remaining: &self.middleware,
            terminal:  self.terminal.as_ref(),
        }
        .run(request)
        .await
    }

    /// Resolves, validates, and invokes one call through the complete stack.
    pub async fn execute(
        &self,
        catalog: &ToolCatalog,
        turn: usize,
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> StdResult<ToolOutcome, ToolSystemError> {
        self.execute_with(catalog, turn, call, cancellation, None)
            .await
    }

    /// [`execute`](Self::execute), with `events` for the tool to stream
    /// output fragments on.
    pub(crate) async fn execute_with(
        &self,
        catalog: &ToolCatalog,
        turn: usize,
        call: ToolCall,
        cancellation: CancellationToken,
        events: Option<EventHub>,
    ) -> StdResult<ToolOutcome, ToolSystemError> {
        let request = match catalog.resolve(turn, call, cancellation.clone()) {
            Ok(request) => request.with_events(events),
            Err(outcome) => return Ok(outcome),
        };
        let outcome = self.call(request).await?;
        if cancellation.is_cancelled() {
            Ok(ToolOutcome::failure(ToolErrorKind::Cancelled, CANCELLED))
        } else {
            Ok(outcome)
        }
    }
}

impl fmt::Debug for ToolSystem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolSystem")
            .field("middleware", &self.middleware.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, PoisonError};

    use lithos_llm::types::{ContentPart, ToolCallKind, ToolDefinition};
    use serde_json::json;

    use super::*;

    struct RecordingService {
        calls: Arc<Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl ToolService for RecordingService {
        async fn discover(
            &self,
            _context: TurnContext<'_>,
        ) -> StdResult<ToolCatalog, ToolSystemError> {
            self.calls
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push("terminal:discover");
            Ok(ToolCatalog::new([descriptor("inspect")]))
        }

        async fn call(&self, _request: ToolCallRequest) -> StdResult<ToolOutcome, ToolSystemError> {
            self.calls
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push("terminal:call");
            Ok(ToolOutcome::success("done".into()))
        }
    }

    struct RecordingMiddleware {
        name:  &'static str,
        calls: Arc<Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl ToolMiddleware for RecordingMiddleware {
        async fn discover(
            &self,
            context: TurnContext<'_>,
            next: ToolDiscoveryNext<'_>,
        ) -> StdResult<ToolCatalog, ToolSystemError> {
            self.record(match self.name {
                "outer" => "outer:discover:before",
                _ => "inner:discover:before",
            });
            let catalog = next.run(context).await?;
            self.record(match self.name {
                "outer" => "outer:discover:after",
                _ => "inner:discover:after",
            });
            Ok(catalog)
        }

        async fn call(
            &self,
            request: ToolCallRequest,
            next: ToolCallNext<'_>,
        ) -> StdResult<ToolOutcome, ToolSystemError> {
            self.record(match self.name {
                "outer" => "outer:call:before",
                _ => "inner:call:before",
            });
            let outcome = next.run(request).await?;
            self.record(match self.name {
                "outer" => "outer:call:after",
                _ => "inner:call:after",
            });
            Ok(outcome)
        }
    }

    impl RecordingMiddleware {
        fn record(&self, entry: &'static str) {
            self.calls
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(entry);
        }
    }

    struct Refuse;

    #[async_trait]
    impl ToolMiddleware for Refuse {
        async fn call(
            &self,
            _request: ToolCallRequest,
            _next: ToolCallNext<'_>,
        ) -> StdResult<ToolOutcome, ToolSystemError> {
            Ok(ToolOutcome::failure(ToolErrorKind::Denied, "not allowed"))
        }
    }

    fn descriptor(name: &str) -> ToolDescriptor {
        ToolDescriptor::new(
            ToolId::try_new(name).expect("the test tool identity is valid"),
            ToolDefinition::function(name, "test tool", json!({"type": "object"})),
        )
    }

    fn request() -> ToolCallRequest {
        ToolCallRequest::new(
            0,
            ToolCall {
                id:                "call_1".to_owned(),
                name:              "inspect".to_owned(),
                arguments:         json!({}),
                kind:              ToolCallKind::Function,
                raw_arguments:     None,
                provider_metadata: BTreeMap::default(),
            },
            descriptor("inspect"),
            CancellationToken::new(),
        )
    }

    #[tokio::test]
    async fn middleware_is_nested_in_installation_order() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let system = ToolSystem::new(Arc::new(RecordingService {
            calls: Arc::clone(&calls),
        }))
        .middleware(Arc::new(RecordingMiddleware {
            name:  "outer",
            calls: Arc::clone(&calls),
        }))
        .middleware(Arc::new(RecordingMiddleware {
            name:  "inner",
            calls: Arc::clone(&calls),
        }));

        let messages = [];
        let context = TurnContext::new("test/model", 0, &messages);
        let _ = system.discover(context).await.expect("discovery succeeds");
        let _ = system.call(request()).await.expect("the call succeeds");

        assert_eq!(*calls.lock().unwrap_or_else(PoisonError::into_inner), [
            "outer:discover:before",
            "inner:discover:before",
            "terminal:discover",
            "inner:discover:after",
            "outer:discover:after",
            "outer:call:before",
            "inner:call:before",
            "terminal:call",
            "inner:call:after",
            "outer:call:after",
        ]);
    }

    #[tokio::test]
    async fn middleware_can_refuse_without_calling_the_terminal() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let system = ToolSystem::new(Arc::new(RecordingService {
            calls: Arc::clone(&calls),
        }))
        .middleware(Arc::new(Refuse));

        let outcome = system.call(request()).await.expect("the refusal succeeds");

        assert_eq!(outcome, ToolOutcome::Failure {
            kind:         ToolErrorKind::Denied,
            message:      "not allowed".to_owned(),
            output_stats: None,
        });
        assert!(
            calls
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        );
    }

    #[test]
    fn resolution_rejects_a_call_kind_that_does_not_match_the_descriptor() {
        let catalog = ToolCatalog::new([descriptor("inspect")]);
        let outcome = catalog
            .resolve(
                0,
                ToolCall::custom("call_1", "inspect", "free form"),
                CancellationToken::new(),
            )
            .expect_err("a custom call cannot invoke a function descriptor");

        assert!(matches!(outcome, ToolOutcome::Failure {
            kind: ToolErrorKind::InvalidArguments,
            ..
        }));
    }

    #[test]
    fn tool_identity_rejects_blank_values() {
        assert_eq!(ToolId::try_new("  "), Err(ToolIdError));
    }

    #[test]
    fn a_catalog_can_be_filtered_without_changing_tool_identity() {
        let mut catalog = ToolCatalog::new([descriptor("read"), descriptor("write")]);

        catalog.retain(|tool| tool.id().as_str() == "read");

        assert_eq!(catalog.tools().len(), 2);
        let visible = catalog.visible_tools().collect::<Vec<_>>();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id().as_str(), "read");
        assert_eq!(
            catalog
                .find_by_name("write")
                .expect("the hidden tool remains resolvable")
                .id()
                .as_str(),
            "write"
        );
        assert!(matches!(
            ToolOutcome::success(ToolOutput::new([ContentPart::Text {
                text: "ok".to_owned(),
            }])),
            ToolOutcome::Success { .. }
        ));
    }
}
