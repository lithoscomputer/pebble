//! Running the tools a model asked for.
//!
//! [`CodingToolService`] is the terminal of a session's tool system and its
//! fixed output envelope. The generic agent loop resolves and schedules each
//! call and publishes its lifecycle; this module runs the tool and answers
//! with a result the session is willing to carry.
//!
//! Each call runs through one tool service and its middleware. Application
//! middleware may continue or refuse, the terminal runs the tool, and the
//! fixed outer layer bounds the result. History sees a further-truncated copy.
//!
//! Output is bounded twice, by
//! [`CodingAgentOptions::tool_output_retention_bytes`] and
//! [`CodingAgentOptions::tool_output_serialized_bytes`] first and by the
//! tool's own character and line limits second. See
//! [`crate::truncate_tool_output`].
//!
//! A tool that parks a prompt on a person's answer is discovered as
//! [`ExclusiveRound`](agent::ToolScheduling::ExclusiveRound), so the agent
//! loop runs it alone and refuses its peers with an explanation the model can
//! act on.

use std::borrow::Cow;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::{ContentPart, ToolCall, ToolResult};
use pebble_agent as agent;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::error::ToolError;
use super::permissions::canonical_tool_name;
use super::registry::{
    CodingEventEmitter, RegisteredTool, ToolContext, ToolEnvProvider, ToolRegistry,
};
use crate::config::CodingAgentOptions;
use crate::environment::Environment;
use crate::event::{Emitter, OutputCaptureStats, SessionBoundEmitter};
use crate::human_input::HumanInputProvider;
use crate::redact::Redactor;
use crate::truncation::{
    OutputBudgets, ToolOutputLimits, preview_tool_output, serialized_json_bytes,
    truncate_tool_output,
};
use crate::types::{CodingEvent, ToolErrorKind};

/// The coding-tool terminal and its fixed output envelope.
///
/// A coding agent and a standalone runner both put this service outside their
/// application middleware. This keeps output limits identical, including when
/// inner middleware refuses a call without reaching the tool.
#[derive(Clone)]
pub(crate) struct CodingToolService {
    registry:          Arc<ToolRegistry>,
    /// What every turn discovers, described once: the registry is frozen
    /// before the first prompt.
    descriptors:       Vec<agent::ToolDescriptor>,
    env:               Arc<dyn Environment>,
    config:            Arc<CodingAgentOptions>,
    emitter:           Emitter,
    session_id:        String,
    /// Equal to `session_id` in a root session; a child carries the root of
    /// its tree, which is how a root-only tool knows it is running somewhere
    /// it should not.
    root_session_id:   String,
    tool_env_provider: Option<Arc<dyn ToolEnvProvider>>,
    /// Absent in a child session and wherever the application installed no
    /// provider, which is what makes a question tool report that it cannot
    /// ask.
    human_input:       Option<Arc<dyn HumanInputProvider>>,
    /// Strips secrets out of the process output a tool publishes and out of
    /// the message of every failed call, which can carry an OS error naming a
    /// path the model asked for.
    redactor:          Arc<dyn Redactor>,
}

impl CodingToolService {
    /// Creates a service with no optional call providers.
    pub(crate) fn new(
        registry: Arc<ToolRegistry>,
        env: Arc<dyn Environment>,
        config: Arc<CodingAgentOptions>,
        emitter: Emitter,
        session_id: String,
        root_session_id: String,
        redactor: Arc<dyn Redactor>,
    ) -> Self {
        Self {
            descriptors: describe(&registry),
            registry,
            env,
            config,
            emitter,
            session_id,
            root_session_id,
            tool_env_provider: None,
            human_input: None,
            redactor,
        }
    }

    /// Sets where a call gets its extra environment variables.
    #[must_use]
    pub(crate) fn with_tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env_provider = Some(provider);
        self
    }

    /// Sets where a human-question tool gets its answers.
    #[must_use]
    pub(crate) fn with_human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Publishes the start of one standalone call.
    pub(crate) fn begin_standalone(&self, call: &ToolCall) {
        self.emit_started(call);
    }

    /// Applies output policy and publishes one standalone result.
    pub(crate) fn complete_standalone(
        &self,
        call: &ToolCall,
        outcome: agent::ToolOutcome,
    ) -> ToolResult {
        let outcome = if outcome.output_stats().is_some() {
            outcome
        } else {
            self.finish_terminal(&call.id, &call.name, outcome)
        };
        let stats = outcome.output_stats();
        let error_kind = outcome.error_kind();
        let result = outcome.into_result(call);
        self.emit_result(&result, stats, error_kind);
        result
    }

    /// Runs one call after the shared agent middleware approved it.
    async fn execute_terminal(&self, request: agent::ToolCallRequest) -> agent::ToolOutcome {
        let tool_id = request.descriptor().id().clone();
        let (mut call, cancellation) = request.into_call();
        let executed = self
            .run_tool(&mut call, self.registry.get_by_id(&tool_id), cancellation)
            .await;
        outcome_from_result(executed.result, executed.error_kind, executed.output_stats)
    }

    /// Applies coding output policy to the final call result.
    fn finish_terminal(
        &self,
        call_id: &str,
        tool_name: &str,
        outcome: agent::ToolOutcome,
    ) -> agent::ToolOutcome {
        let previous = outcome.output_stats();
        let (result, error_kind) = match outcome {
            agent::ToolOutcome::Success { output, .. } => (
                ToolResult {
                    tool_call_id: call_id.to_owned(),
                    name:         Some(tool_name.to_owned()),
                    content:      output.content().to_vec(),
                    is_error:     false,
                },
                None,
            ),
            agent::ToolOutcome::Failure { kind, message, .. } => (
                self.error_result_for(call_id, tool_name, &ToolError::new(kind, message)),
                Some(kind),
            ),
            _ => (
                self.error_result_for(
                    call_id,
                    tool_name,
                    &ToolError::execution("the tool returned an unsupported outcome"),
                ),
                Some(ToolErrorKind::Execution),
            ),
        };
        let retained = self.retain(result, previous);
        let result = self.truncate_for_history(retained.result, tool_name);
        outcome_from_result(result, error_kind, Some(retained.output_stats))
    }

    /// Runs one resolved tool.
    async fn run_tool(
        &self,
        call: &mut ToolCall,
        registered: Option<&RegisteredTool>,
        cancel: CancellationToken,
    ) -> ExecutedTool {
        let Some(tool) = registered else {
            return self.failed(
                call,
                &ToolError::unavailable(format!("Unknown tool: {}", call.name)),
            );
        };

        let bound = Arc::new(SessionBoundEmitter::new(
            self.emitter.clone(),
            &self.session_id,
            Some(call.id.clone()),
        ));
        let mut context = ToolContext::new(Arc::clone(&self.env))
            .with_cancel(cancel)
            .with_session(&self.session_id, &self.root_session_id)
            .with_tool_call_id(call.id.clone())
            .with_coding_event_emitter(Arc::clone(&bound) as Arc<dyn CodingEventEmitter>)
            .with_redactor(Arc::clone(&self.redactor));
        if let Some(provider) = &self.tool_env_provider {
            context = context.with_tool_env_provider(Arc::clone(provider));
        }
        if let Some(provider) = &self.human_input {
            context = context.with_human_input(Arc::clone(provider));
        }

        let arguments = match call.input.to_value() {
            Ok(arguments) => arguments,
            Err(error) => {
                return self.failed(call, &ToolError::invalid_arguments(error.to_string()));
            }
        };
        let (result, error_kind) = match (tool.executor)(arguments, context).await {
            Ok(output) => (text_result(call, output, false), None),
            Err(error) => (
                self.error_result_for(&call.id, &call.name, &error),
                Some(error.kind()),
            ),
        };

        ExecutedTool {
            result,
            error_kind,
            output_stats: bound.take_tool_output_stats(),
        }
    }

    /// A failure the tool never got to see.
    fn failed(&self, call: &ToolCall, error: &ToolError) -> ExecutedTool {
        ExecutedTool {
            result:       self.error_result_for(&call.id, &call.name, error),
            error_kind:   Some(error.kind()),
            output_stats: None,
        }
    }

    /// A failed call, rendering the error's model-facing message.
    ///
    /// The message passes through the session's redactor first. A message is
    /// written without secrets, but an environment failure carries an OS
    /// error under it, and the path in that error is whatever the model
    /// asked for.
    fn error_result_for(&self, call_id: &str, tool_name: &str, error: &ToolError) -> ToolResult {
        let message = self.redactor.redact(error.message()).into_owned();
        text_result_for(call_id, tool_name, message, true)
    }

    /// Cuts a tool's output down to what the session is willing to carry.
    ///
    /// This is the form middleware, events, and the model all see. Bytes the
    /// environment already dropped while draining the process are carried in
    /// through `previous`, so the counters describe everything the tool
    /// produced rather than everything that reached this point.
    fn retain(&self, mut result: ToolResult, previous: Option<OutputCaptureStats>) -> Retained {
        let budgets = OutputBudgets::new(
            self.config.tool_output_retention_bytes,
            self.config.tool_output_serialized_bytes,
        );
        let output_stats = match text_part_mut(&mut result) {
            Some(text) => {
                let previously_omitted = previous.map_or(0, |stats| stats.omitted_bytes);
                let previewed = preview_tool_output(text, budgets, previously_omitted);
                let stats = previewed.stats;
                if let Cow::Owned(bounded) = previewed.output {
                    *text = bounded;
                }
                stats
            }
            // Structured output is left alone; only its size is reported.
            None => OutputCaptureStats::complete(serialized_json_bytes(&result.content)),
        };

        Retained {
            result,
            output_stats,
        }
    }

    /// Cuts the copy history keeps to the limits this tool deserves.
    fn truncate_for_history(&self, mut result: ToolResult, tool_name: &str) -> ToolResult {
        if let Some(text) = text_part_mut(&mut result) {
            let limits = ToolOutputLimits::resolve(
                tool_name,
                canonical_tool_name(tool_name),
                &self.config.tool_output_limits,
                &self.config.tool_line_limits,
            );
            *text = truncate_tool_output(text, limits);
        }
        result
    }

    /// Publishes that a call started.
    pub(crate) fn emit_started(&self, call: &ToolCall) {
        self.emit(&call.id, CodingEvent::ToolCallStarted {
            tool_name:    call.name.clone(),
            tool_call_id: call.id.clone(),
            arguments:    call
                .input
                .to_value()
                .unwrap_or_else(|_| Value::String(call.input.raw().to_owned())),
        });
    }

    /// Publishes a call's result.
    ///
    /// `output_stats` is what the output envelope counted. A call the kernel
    /// refused before the envelope ran carries none, and is counted as its
    /// message.
    pub(crate) fn emit_result(
        &self,
        result: &ToolResult,
        output_stats: Option<OutputCaptureStats>,
        error_kind: Option<ToolErrorKind>,
    ) {
        let output = output_value(result);
        let text = result_text(result);
        let stats = output_stats.unwrap_or_else(|| OutputCaptureStats::complete(text.len()));
        // The same bounded output goes out twice on purpose: the delta is the
        // live-streaming feed and `ToolCallCompleted` is the durable record,
        // so a store keeps the completed event and drops deltas as ephemeral.
        // No tool streams incremental deltas yet, which makes the two payloads
        // equal today.
        self.emit(&result.tool_call_id, CodingEvent::ToolCallOutputDelta {
            delta: text.into_owned(),
        });
        self.emit(&result.tool_call_id, CodingEvent::ToolCallCompleted {
            tool_name: result.name.clone().unwrap_or_default(),
            tool_call_id: result.tool_call_id.clone(),
            output,
            is_error: result.is_error,
            error_kind,
            output_bytes_observed: stats.observed_bytes,
            output_bytes_retained: stats.retained_bytes,
            output_bytes_omitted: stats.omitted_bytes,
        });
    }

    /// Publishes an event about one call, stamping the call on the envelope so
    /// a fragment with no identity of its own is still attributable.
    fn emit(&self, tool_call_id: &str, event: CodingEvent) {
        self.emitter
            .emit_with_tool_call_id(&self.session_id, event, Some(tool_call_id.to_owned()));
    }
}

#[async_trait]
impl agent::ToolService for CodingToolService {
    async fn discover(
        &self,
        _context: agent::TurnContext<'_>,
    ) -> StdResult<agent::ToolCatalog, agent::ToolSystemError> {
        Ok(agent::ToolCatalog::new(self.descriptors.iter().cloned()))
    }

    async fn call(
        &self,
        request: agent::ToolCallRequest,
    ) -> StdResult<agent::ToolOutcome, agent::ToolSystemError> {
        Ok(self.execute_terminal(request).await)
    }
}

#[async_trait]
impl agent::ToolMiddleware for CodingToolService {
    async fn call(
        &self,
        request: agent::ToolCallRequest,
        next: agent::ToolCallNext<'_>,
    ) -> StdResult<agent::ToolOutcome, agent::ToolSystemError> {
        let call_id = request.call().id.clone();
        let tool_name = request.call().name.clone();
        let outcome = next.run(request).await?;
        Ok(self.finish_terminal(&call_id, &tool_name, outcome))
    }
}

/// Describes every registered tool to the generic agent.
///
/// A tool's stable identity is its canonical name, so policy written against
/// pebble's names holds whichever vocabulary the model sees. A tool that parks
/// the prompt on a person runs alone in its round.
fn describe(registry: &ToolRegistry) -> Vec<agent::ToolDescriptor> {
    registry
        .tools_with_ids()
        .map(|(id, tool)| {
            let scheduling = if tool.needs_human_input() {
                agent::ToolScheduling::ExclusiveRound
            } else {
                agent::ToolScheduling::Concurrent
            };
            agent::ToolDescriptor::new(id.clone(), tool.definition.clone())
                .with_scheduling(scheduling)
        })
        .collect()
}

/// What a tool produced, before any budget applied.
struct ExecutedTool {
    result:       ToolResult,
    error_kind:   Option<ToolErrorKind>,
    /// What the tool reported about its own output, when it reported anything.
    output_stats: Option<OutputCaptureStats>,
}

/// One call's result inside the session's output budgets.
struct Retained {
    result:       ToolResult,
    output_stats: OutputCaptureStats,
}

/// The logical outcome of a result the coding layer built.
fn outcome_from_result(
    result: ToolResult,
    error_kind: Option<ToolErrorKind>,
    output_stats: Option<OutputCaptureStats>,
) -> agent::ToolOutcome {
    let outcome = if result.is_error {
        agent::ToolOutcome::failure(
            error_kind.unwrap_or(ToolErrorKind::Execution),
            result_text(&result).into_owned(),
        )
    } else {
        agent::ToolOutcome::success(agent::ToolOutput::new(result.content))
    };
    match output_stats {
        Some(stats) => outcome.with_output_stats(stats),
        None => outcome,
    }
}

/// A result carrying one block of text, which is what every pebble tool
/// produces.
fn text_result(call: &ToolCall, text: String, is_error: bool) -> ToolResult {
    text_result_for(&call.id, &call.name, text, is_error)
}

fn text_result_for(
    tool_call_id: &str,
    tool_name: &str,
    text: String,
    is_error: bool,
) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call_id.to_owned(),
        // Kept so a provider that labels a tool message by name has it; the
        // call id alone does not survive every wire format.
        name: Some(tool_name.to_owned()),
        content: vec![ContentPart::Text { text }],
        is_error,
    }
}

/// The text of a single-text-part result, for rewriting in place.
fn text_part_mut(result: &mut ToolResult) -> Option<&mut String> {
    match result.content.as_mut_slice() {
        [ContentPart::Text { text }] => Some(text),
        _ => None,
    }
}

/// A result's output as text, which is what the output fragment carries.
///
/// Also what the modules that read finished results — file tracking and the
/// compaction transcript — see, so a tool's output is rendered the same way
/// everywhere.
pub(crate) fn result_text(result: &ToolResult) -> Cow<'_, str> {
    match result.content.as_slice() {
        [ContentPart::Text { text }] => Cow::Borrowed(text),
        other => Cow::Owned(serde_json::to_string(other).unwrap_or_default()),
    }
}

/// A result's output as the JSON its completion event carries.
fn output_value(result: &ToolResult) -> Value {
    match result.content.as_slice() {
        [ContentPart::Text { text }] => Value::String(text.clone()),
        other => serde_json::to_value(other).unwrap_or(Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Mutex, PoisonError};

    use lithos_llm::types::{Message, ToolDefinition};
    use serde_json::json;
    use tokio::sync::broadcast;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::environment::EnvironmentError;
    use crate::error::Result as PebbleResult;
    use crate::event::{EventOptions, EventPump};
    use crate::redact::NoRedaction;
    use crate::test_support::MockEnvironment;
    use crate::types::{CodingAgentEvent, CommandTermination, ToolSource};

    /// An event pipeline whose events can be read once the call is over.
    struct Events {
        emitter:  Emitter,
        pump:     JoinHandle<PebbleResult<()>>,
        received: broadcast::Receiver<CodingAgentEvent>,
    }

    impl Events {
        fn new() -> Self {
            let (emitter, pump) = EventPump::new(EventOptions::default());
            let received = emitter.subscribe();
            Self {
                emitter,
                pump: tokio::spawn(pump.run()),
                received,
            }
        }

        /// Stops the pipeline and returns everything it published.
        ///
        /// The service under test holds its own emitter clone, so the pump is
        /// closed explicitly rather than by dropping the last sender.
        async fn drain(self) -> Vec<CodingAgentEvent> {
            let Self {
                emitter,
                pump,
                mut received,
            } = self;
            emitter.close().await.expect("the pump closes");
            pump.await
                .expect("the pump task joins")
                .expect("the pump finishes");

            let mut events = Vec::new();
            while let Ok(event) = received.try_recv() {
                events.push(event);
            }
            events
        }

        /// The names of the tool-call events, in published order.
        fn order(events: &[CodingAgentEvent]) -> Vec<&'static str> {
            events
                .iter()
                .filter_map(|event| match &event.event {
                    CodingEvent::ToolCallStarted { .. } => Some("started"),
                    CodingEvent::ToolProcessCompleted { .. } => Some("process"),
                    CodingEvent::ToolCallOutputDelta { .. } => Some("delta"),
                    CodingEvent::ToolCallCompleted { .. } => Some("completed"),
                    _ => None,
                })
                .collect()
        }
    }

    /// The service a session builds, over `tools`, publishing on `events`.
    fn service(
        tools: impl IntoIterator<Item = RegisteredTool>,
        events: &Events,
        redactor: Arc<dyn Redactor>,
    ) -> Arc<CodingToolService> {
        let mut registry = ToolRegistry::new();
        for tool in tools {
            registry
                .register(tool)
                .expect("tool registration is unique");
        }
        Arc::new(CodingToolService::new(
            Arc::new(registry),
            Arc::new(MockEnvironment::default()),
            Arc::new(CodingAgentOptions::default()),
            events.emitter.clone(),
            "ses_1".to_owned(),
            "ses_1".to_owned(),
            redactor,
        ))
    }

    fn plain_service(
        tools: impl IntoIterator<Item = RegisteredTool>,
        events: &Events,
    ) -> Arc<CodingToolService> {
        service(tools, events, Arc::new(NoRedaction))
    }

    /// Answers one call the way a session does: through the service as both
    /// terminal and output envelope, with the kernel resolving the call.
    async fn run(
        service: &Arc<CodingToolService>,
        call: &ToolCall,
        cancel: CancellationToken,
    ) -> ToolResult {
        let system = agent::ToolSystem::new(service.clone()).middleware(service.clone());
        let catalog = discover(service).await;
        service.begin_standalone(call);
        let outcome = system
            .execute(&catalog, 0, call.clone(), cancel)
            .await
            .expect("the call completes");
        service.complete_standalone(call, outcome)
    }

    async fn discover(service: &Arc<CodingToolService>) -> agent::ToolCatalog {
        let messages: [Message; 0] = [];
        agent::ToolService::discover(
            service.as_ref(),
            agent::TurnContext::new("test/model", 0, &messages),
        )
        .await
        .expect("discovery succeeds")
    }

    fn call(name: &str, id: &str, arguments: Value) -> ToolCall {
        ToolCall::function(id, name, arguments)
    }

    fn echo_tool() -> RegisteredTool {
        RegisteredTool::new(
            ToolDefinition::function(
                "echo",
                "Echo input",
                json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }),
            ),
            Arc::new(|arguments: Value, _context| {
                Box::pin(async move {
                    let text = arguments["text"].as_str().unwrap_or_default();
                    Ok(format!("echo: {text}"))
                })
            }),
        )
        .with_source(ToolSource::Native)
    }

    fn failing_tool() -> RegisteredTool {
        RegisteredTool::new(
            ToolDefinition::function("fail_tool", "Always fails", json!({})),
            Arc::new(|_arguments, _context| {
                Box::pin(async { Err(ToolError::execution("tool failed")) })
            }),
        )
        .with_source(ToolSource::Native)
    }

    /// A tool shaped like the shell tool: it reports its subprocess itself,
    /// through the context, before the service completes the call.
    fn process_tool(exit_code: i32) -> RegisteredTool {
        RegisteredTool::new(
            ToolDefinition::function("shell", "Runs a command", json!({})),
            Arc::new(move |_arguments, context: ToolContext| {
                Box::pin(async move {
                    context.record_tool_output_stats(OutputCaptureStats::complete(3));
                    context.emit_coding_event(CodingEvent::ToolProcessCompleted {
                        exit_code:             Some(exit_code),
                        termination:           CommandTermination::Exited,
                        duration_ms:           12,
                        streams_separated:     true,
                        exec_output_tail:      None,
                        output_bytes_observed: 3,
                        output_bytes_retained: 3,
                        output_bytes_omitted:  0,
                    });
                    if exit_code == 0 {
                        Ok("out".to_owned())
                    } else {
                        Err(ToolError::execution(format!("Exit code: {exit_code}")))
                    }
                })
            }),
        )
        .with_source(ToolSource::Native)
    }

    fn text_of(result: &ToolResult) -> String {
        result_text(result).into_owned()
    }

    fn completion(events: &[CodingAgentEvent]) -> &CodingEvent {
        events
            .iter()
            .map(|event| &event.event)
            .find(|event| matches!(event, CodingEvent::ToolCallCompleted { .. }))
            .expect("a completion event")
    }

    #[tokio::test]
    async fn a_tool_runs_and_reports_its_output() {
        let events = Events::new();
        let service = plain_service([echo_tool()], &events);

        let result = run(
            &service,
            &call("echo", "call_1", json!({"text": "hello"})),
            CancellationToken::new(),
        )
        .await;

        assert!(!result.is_error);
        assert_eq!(text_of(&result), "echo: hello");
        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.name.as_deref(), Some("echo"));

        let published = events.drain().await;
        assert_eq!(Events::order(&published), ["started", "delta", "completed"]);
        for event in &published {
            assert_eq!(event.session_id, "ses_1");
            assert_eq!(event.tool_call_id.as_deref(), Some("call_1"));
        }
        assert!(matches!(
            completion(&published),
            CodingEvent::ToolCallCompleted {
                output,
                is_error: false,
                error_kind: None,
                ..
            } if output == &json!("echo: hello")
        ));
    }

    #[tokio::test]
    async fn a_failing_tool_reports_only_its_safe_message() {
        let events = Events::new();
        let service = plain_service([failing_tool()], &events);

        let result = run(
            &service,
            &call("fail_tool", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        assert!(result.is_error);
        assert_eq!(text_of(&result), "tool failed");
        assert!(matches!(
            completion(&events.drain().await),
            CodingEvent::ToolCallCompleted {
                is_error: true,
                error_kind: Some(ToolErrorKind::Execution),
                ..
            }
        ));
    }

    /// An environment failure's OS cause reaches the model, and the redactor
    /// sees it on the way: the path in an io error is whatever the model asked
    /// to read.
    #[tokio::test]
    async fn a_failed_call_is_redacted_before_the_model_reads_it() {
        struct DropKeys;

        impl Redactor for DropKeys {
            fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
                Cow::Owned(text.replace("AKIAYRWQG5EJLPZLBYNP", "[REDACTED]"))
            }
        }

        let tool = RegisteredTool::new(
            ToolDefinition::function("read_secret", "Fails with a secret", json!({})),
            Arc::new(|_arguments, _context| {
                Box::pin(async {
                    Err(ToolError::from(EnvironmentError::io(
                        "Failed to read /work/keys/AKIAYRWQG5EJLPZLBYNP.pem",
                        io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "Permission denied opening AKIAYRWQG5EJLPZLBYNP.pem",
                        ),
                    )))
                })
            }),
        )
        .with_source(ToolSource::Native);
        let events = Events::new();
        let service = service([tool], &events, Arc::new(DropKeys));

        let result = run(
            &service,
            &call("read_secret", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        assert!(result.is_error);
        assert_eq!(
            text_of(&result),
            "Failed to read /work/keys/[REDACTED].pem\n  caused by: Permission denied opening \
             [REDACTED].pem"
        );
        assert!(matches!(
            completion(&events.drain().await),
            CodingEvent::ToolCallCompleted {
                output,
                is_error: true,
                error_kind: Some(ToolErrorKind::Execution),
                ..
            } if !output.to_string().contains("AKIA")
        ));
    }

    #[tokio::test]
    async fn an_unknown_tool_is_reported_as_unavailable() {
        let events = Events::new();
        let service = plain_service([], &events);

        let result = run(
            &service,
            &call("nope", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        assert!(result.is_error);
        assert_eq!(text_of(&result), "unknown tool `nope`");
        assert!(matches!(
            completion(&events.drain().await),
            CodingEvent::ToolCallCompleted {
                error_kind: Some(ToolErrorKind::Unavailable),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn arguments_that_miss_the_schema_never_reach_the_tool() {
        let runs = Arc::new(Mutex::new(0_usize));
        let counter = Arc::clone(&runs);
        let mut tool = echo_tool();
        tool.executor = Arc::new(move |_arguments, _context| {
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                *counter.lock().unwrap_or_else(PoisonError::into_inner) += 1;
                Ok("ran".to_owned())
            })
        });
        let events = Events::new();
        let service = plain_service([tool], &events);

        let result = run(
            &service,
            &call("echo", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        assert!(result.is_error);
        assert!(
            text_of(&result).contains("missing required property \"text\""),
            "{}",
            text_of(&result)
        );
        assert_eq!(*runs.lock().unwrap_or_else(PoisonError::into_inner), 0);
        assert!(matches!(
            completion(&events.drain().await),
            CodingEvent::ToolCallCompleted {
                error_kind: Some(ToolErrorKind::InvalidArguments),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn tool_output_is_bounded_before_events_and_history() {
        let budget = CodingAgentOptions::default().tool_output_retention_bytes;
        let text = "x".repeat(budget + 100);
        let events = Events::new();
        let service = plain_service([echo_tool()], &events);

        let result = run(
            &service,
            &call("echo", "call_large", json!({"text": text})),
            CancellationToken::new(),
        )
        .await;

        let output = text_of(&result);
        assert!(output.len() <= budget);
        assert!(output.starts_with("Warning: truncated output"));
        assert!(output.contains("bytes omitted"));
        assert!(output.contains("tokens truncated"));

        let published = events.drain().await;
        let CodingEvent::ToolCallCompleted {
            output: event_output,
            is_error,
            output_bytes_observed,
            output_bytes_retained,
            output_bytes_omitted,
            ..
        } = completion(&published)
        else {
            panic!("a completion event");
        };
        assert_eq!(event_output.as_str(), Some(output.as_str()));
        assert!(!is_error, "truncation must not make the tool an error");
        assert_eq!(*output_bytes_observed, budget + 100 + "echo: ".len());
        assert!(*output_bytes_retained < budget);
        assert_eq!(
            *output_bytes_omitted,
            output_bytes_observed - output_bytes_retained
        );
        assert!(output.contains(&format!("... {output_bytes_omitted} bytes omitted ...")));
    }

    #[tokio::test]
    async fn serialized_tool_output_stays_within_the_serialized_budget() {
        let config = CodingAgentOptions::default();
        // Output made almost entirely of characters JSON escapes: it fits the
        // retained budget as text and blows past it once serialized.
        let text = format!(
            "HEAD{}TAIL",
            "\0".repeat(config.tool_output_retention_bytes - "echo: HEADTAIL".len())
        );
        let events = Events::new();
        let service = plain_service([echo_tool()], &events);

        let result = run(
            &service,
            &call("echo", "call_escaped", json!({"text": text})),
            CancellationToken::new(),
        )
        .await;

        assert!(!result.is_error);
        let published = events.drain().await;
        let CodingEvent::ToolCallCompleted {
            output: event_output,
            ..
        } = completion(&published)
        else {
            panic!("a completion event");
        };
        let serialized = serde_json::to_vec(event_output)
            .expect("tool output serializes")
            .len();
        assert!(
            serialized <= config.tool_output_serialized_bytes,
            "serialized output was {serialized} bytes"
        );
    }

    #[tokio::test]
    async fn a_tool_that_runs_a_process_reports_it_before_its_own_completion() {
        let events = Events::new();
        let service = plain_service([process_tool(7)], &events);

        let result = run(
            &service,
            &call("shell", "call_1", json!({"command": "make test"})),
            CancellationToken::new(),
        )
        .await;

        assert!(result.is_error);
        assert_eq!(text_of(&result), "Exit code: 7");

        let published = events.drain().await;
        assert_eq!(Events::order(&published), [
            "started",
            "process",
            "delta",
            "completed"
        ]);
        let process = published
            .iter()
            .find(|event| matches!(event.event, CodingEvent::ToolProcessCompleted { .. }))
            .expect("a process event");
        assert_eq!(process.tool_call_id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn what_a_tool_reported_about_its_own_output_reaches_the_counters() {
        let events = Events::new();
        let service = plain_service([process_tool(0)], &events);

        run(
            &service,
            &call("shell", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        let published = events.drain().await;
        assert!(matches!(
            completion(&published),
            CodingEvent::ToolCallCompleted {
                output_bytes_observed: 3,
                output_bytes_retained: 3,
                output_bytes_omitted: 0,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn truncation_preserves_the_call_id_and_the_error_state() {
        let tool = RegisteredTool::new(
            ToolDefinition::function("shell", "Runs a command", json!({})),
            Arc::new(|_arguments, _context| {
                Box::pin(async { Err(ToolError::execution("x".repeat(60_000))) })
            }),
        )
        .with_source(ToolSource::Native);
        let events = Events::new();
        let service = plain_service([tool], &events);

        let result = run(
            &service,
            &call("shell", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        assert_eq!(result.tool_call_id, "call_1");
        assert!(result.is_error);
        assert!(text_of(&result).len() < 60_000);
        drop(events.drain().await);
    }

    /// The agent loop runs an exclusive tool alone in its round, and the
    /// service marks the tools that need a person that way.
    #[tokio::test]
    async fn a_tool_that_needs_a_person_is_discovered_as_exclusive() {
        let asks = RegisteredTool::function(
            "asks",
            "Asks a person",
            json!({"type": "object"}),
            |_context, _arguments| async { Ok("asked".to_owned()) },
        )
        .requires_human_input();
        let events = Events::new();
        let service = plain_service([asks, echo_tool()], &events);

        let catalog = discover(&service).await;

        let scheduling = |name: &str| {
            catalog
                .find_by_name(name)
                .unwrap_or_else(|| panic!("{name} is discovered"))
                .scheduling()
        };
        assert_eq!(scheduling("asks"), agent::ToolScheduling::ExclusiveRound);
        assert_eq!(scheduling("echo"), agent::ToolScheduling::Concurrent);
        drop(events.drain().await);
    }
}
