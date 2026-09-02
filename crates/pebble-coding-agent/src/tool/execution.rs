//! Running the tools a model asked for.
//!
//! One round of tool calls goes through a [`ToolDispatch`], which answers every
//! call it is given — including the ones it refuses — so a conversation never
//! carries a call without its result.
//!
//! Each call runs through one tool service and its middleware. The fixed outer
//! layer publishes the start, application middleware may continue or refuse,
//! the terminal runs the tool, and the outer layer bounds and publishes the
//! result. History sees a further-truncated copy.
//!
//! Output is bounded twice, by
//! [`CodingAgentOptions::tool_output_retention_bytes`] and
//! [`CodingAgentOptions::tool_output_serialized_bytes`] first and by the
//! tool's own character and line limits second. See
//! [`crate::truncate_tool_output`].
//!
//! Human-question tools are the one exception to running calls together: they
//! park a prompt until a person answers, so at most one of them runs per round
//! and its peers are refused with an explanation the model can act on.

use std::borrow::Cow;
use std::collections::HashMap;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
#[cfg(test)]
use futures_util::future::join_all;
use lithos_llm::types::{ContentPart, ToolCall, ToolCallKind, ToolDefinitionKind, ToolResult};
use pebble_agent as agent;
use pebble_agent::integration::validate_tool_arguments;
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
use crate::human_input::{HumanInputProvider, is_question_tool};
use crate::redact::Redactor;
use crate::truncation::{
    OutputBudgets, ToolOutputLimits, preview_tool_output, serialized_json_bytes,
    truncate_tool_output,
};
use crate::types::{CodingEvent, ToolErrorKind};

/// What a second human-question call in one round is told.
#[cfg(test)]
const ONE_QUESTION_PER_ROUND: &str = "Only one human-question tool call may be used in a tool \
                                      round. Combine all questions into a single questions[] \
                                      batch and call the question tool once.";

/// What a call that shared a round with a human-question call is told.
#[cfg(test)]
const QUESTIONS_RUN_ALONE: &str = "This tool call was not executed because human-question tools \
                                   must run alone in a tool round. Retry non-question tools in a \
                                   later round after the user answers.";

/// What a call that never started is told.
#[cfg(test)]
const CANCELLED: &str = "Cancelled";

/// The coding-tool terminal and its fixed event/output envelope.
///
/// A coding agent and a standalone runner both put this service outside their
/// application middleware. This keeps call events and output limits identical,
/// including when inner middleware refuses a call without reaching the tool.
#[derive(Clone)]
pub(crate) struct CodingToolService {
    registry:          ToolRegistry,
    env:               Arc<dyn Environment>,
    config:            CodingAgentOptions,
    emitter:           Emitter,
    session_id:        String,
    root_session_id:   String,
    tool_env_provider: Arc<Mutex<Option<Arc<dyn ToolEnvProvider>>>>,
    human_input:       Option<Arc<dyn HumanInputProvider>>,
    redactor:          Arc<dyn Redactor>,
    output_stats:      Arc<Mutex<HashMap<String, OutputCaptureStats>>>,
}

impl CodingToolService {
    /// Creates a service with no optional call providers.
    pub(crate) fn new(
        registry: ToolRegistry,
        env: Arc<dyn Environment>,
        config: CodingAgentOptions,
        emitter: Emitter,
        session_id: String,
        root_session_id: String,
        redactor: Arc<dyn Redactor>,
    ) -> Self {
        Self {
            registry,
            env,
            config,
            emitter,
            session_id,
            root_session_id,
            tool_env_provider: Arc::new(Mutex::new(None)),
            human_input: None,
            redactor,
            output_stats: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Sets where a call gets its extra environment variables.
    #[must_use]
    pub(crate) fn with_tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env_provider = Arc::new(Mutex::new(Some(provider)));
        self
    }

    /// Sets where a human-question tool gets its answers.
    #[must_use]
    pub(crate) fn with_human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Replaces the provider used by later calls.
    #[cfg(test)]
    pub(crate) fn set_tool_env_provider(&self, provider: Arc<dyn ToolEnvProvider>) {
        *self
            .tool_env_provider
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(provider);
    }

    /// Clears process-output state left by a cancelled prompt.
    pub(crate) fn begin_prompt(&self) {
        self.output_stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    fn dispatch(&self) -> ToolDispatch<'_> {
        ToolDispatch::new(
            &self.registry,
            &self.env,
            &self.config,
            &self.emitter,
            &self.session_id,
            &self.root_session_id,
        )
        .with_redactor(&self.redactor)
    }

    /// Answers a call that the shared kernel could not enter into middleware.
    pub(crate) fn answer_failure(
        &self,
        call: &ToolCall,
        kind: ToolErrorKind,
        message: impl Into<String>,
    ) -> agent::ToolOutcome {
        let dispatch = self.dispatch();
        dispatch.begin_terminal(call);
        dispatch.finish_terminal(call, agent::ToolOutcome::failure(kind, message), None)
    }
}

#[async_trait]
impl agent::ToolService for CodingToolService {
    async fn discover(
        &self,
        _context: agent::ToolDiscoveryContext<'_>,
    ) -> StdResult<agent::ToolCatalog, agent::ToolSystemError> {
        let tools = self
            .registry
            .definitions_with_source()
            .into_iter()
            .map(|tool| {
                let name = tool.definition.name.clone();
                let id = agent::ToolId::try_new(canonical_tool_name(&name)).map_err(|source| {
                    agent::ToolSystemError::with_source(
                        format!("tool `{name}` has no stable identity"),
                        source,
                    )
                })?;
                let scheduling = if is_question_tool(&name) {
                    agent::ToolScheduling::ExclusiveRound
                } else {
                    agent::ToolScheduling::Concurrent
                };
                Ok(agent::ToolDescriptor::new(id, tool.definition).with_scheduling(scheduling))
            })
            .collect::<StdResult<Vec<_>, agent::ToolSystemError>>()?;
        Ok(agent::ToolCatalog::new(tools))
    }

    async fn call(
        &self,
        request: agent::ToolCallRequest,
    ) -> StdResult<agent::ToolOutcome, agent::ToolSystemError> {
        let call_id = request.call().id.clone();
        let tool_env_provider = self
            .tool_env_provider
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let mut dispatch = self.dispatch();
        if let Some(provider) = tool_env_provider.as_ref() {
            dispatch = dispatch.with_tool_env_provider(provider);
        }
        if let Some(provider) = self.human_input.as_ref() {
            dispatch = dispatch.with_human_input(provider);
        }
        let (outcome, stats) = dispatch.execute_terminal(request).await;
        if let Some(stats) = stats {
            self.output_stats
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(call_id, stats);
        }
        Ok(outcome)
    }
}

#[async_trait]
impl agent::ToolMiddleware for CodingToolService {
    async fn call(
        &self,
        request: agent::ToolCallRequest,
        next: agent::ToolCallNext<'_>,
    ) -> StdResult<agent::ToolOutcome, agent::ToolSystemError> {
        let call = request.call().clone();
        let dispatch = self.dispatch();
        dispatch.begin_terminal(&call);
        let outcome = next.run(request).await;
        let previous = self
            .output_stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&call.id);
        match outcome {
            Ok(outcome) => Ok(dispatch.finish_terminal(&call, outcome, previous)),
            Err(error) => {
                let _ = dispatch.finish_terminal(
                    &call,
                    agent::ToolOutcome::failure(ToolErrorKind::Execution, error.message()),
                    previous,
                );
                Err(error)
            }
        }
    }
}

/// One session's tool dispatch: the registry, the environment, and everything
/// a call is answered with.
///
/// Built per round and borrowed from the session, so nothing here outlives the
/// round it belongs to. Construct it with [`new`](Self::new) and add the
/// optional seams with the `with_*` methods.
#[derive(Clone, Copy)]
pub(crate) struct ToolDispatch<'a> {
    registry:          &'a ToolRegistry,
    env:               &'a Arc<dyn Environment>,
    config:            &'a CodingAgentOptions,
    emitter:           &'a Emitter,
    session_id:        &'a str,
    root_session_id:   &'a str,
    tool_env_provider: Option<&'a Arc<dyn ToolEnvProvider>>,
    human_input:       Option<&'a Arc<dyn HumanInputProvider>>,
    redactor:          Option<&'a Arc<dyn Redactor>>,
}

impl<'a> ToolDispatch<'a> {
    /// Dispatch for one session.
    ///
    /// `root_session_id` equals `session_id` in a root session; a child
    /// session passes the root of its tree, which is how a root-only tool
    /// knows it is running somewhere it should not.
    #[must_use]
    pub(crate) fn new(
        registry: &'a ToolRegistry,
        env: &'a Arc<dyn Environment>,
        config: &'a CodingAgentOptions,
        emitter: &'a Emitter,
        session_id: &'a str,
        root_session_id: &'a str,
    ) -> Self {
        Self {
            registry,
            env,
            config,
            emitter,
            session_id,
            root_session_id,
            tool_env_provider: None,
            human_input: None,
            redactor: None,
        }
    }

    /// Sets where a call's extra environment variables come from.
    #[must_use]
    pub(crate) fn with_tool_env_provider(mut self, provider: &'a Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env_provider = Some(provider);
        self
    }

    /// Sets where a question tool asks the person.
    ///
    /// Absent in a child session and wherever the application installed no
    /// provider, which is what makes a question tool report that it cannot ask.
    #[must_use]
    pub(crate) fn with_human_input(mut self, provider: &'a Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Sets what strips secrets out of text a tool publishes.
    ///
    /// It runs over process output a tool puts on the event stream and over
    /// the message of every failed call, which can carry an OS error naming a
    /// path. Without one, both reach the model and the event stream exactly
    /// as they were written.
    #[must_use]
    pub(crate) fn with_redactor(mut self, redactor: &'a Arc<dyn Redactor>) -> Self {
        self.redactor = Some(redactor);
        self
    }

    /// Answers every call in one round, in call order.
    ///
    /// `parallel` lets independent calls run together; a round holding a
    /// human-question call ignores it and runs sequentially, because the
    /// question runs alone. A call that finds `cancel` already fired is
    /// answered without being started, so the round still pairs a result with
    /// every call.
    #[cfg(test)]
    pub(crate) async fn execute(
        &self,
        calls: &[ToolCall],
        parallel: bool,
        cancel: &CancellationToken,
    ) -> Vec<ToolResult> {
        self.execute_round(calls, parallel, cancel).await
    }

    #[cfg(test)]
    async fn execute_round(
        &self,
        calls: &[ToolCall],
        parallel: bool,
        cancel: &CancellationToken,
    ) -> Vec<ToolResult> {
        // A round that opens after its cancellation fired — the prompt was
        // ended while the turn was being committed — answers every call without
        // starting one, whichever way it would have run. The sequential paths
        // below check again per call for a cancellation that lands mid-round.
        if cancel.is_cancelled() {
            return calls
                .iter()
                .map(|call| self.cancelled_result(call))
                .collect();
        }

        if calls.iter().any(|call| is_question_tool(&call.name)) {
            return self.execute_question_round(calls, cancel).await;
        }

        if parallel && calls.len() > 1 {
            self.execute_parallel(calls, cancel).await
        } else {
            self.execute_sequential(calls, cancel).await
        }
    }

    /// Answers one call without application middleware.
    #[cfg(test)]
    pub(crate) async fn execute_one(
        &self,
        call: &ToolCall,
        cancel: CancellationToken,
    ) -> ToolResult {
        self.emit_started(call);
        let executed = self
            .run_tool(call, self.registry.get(&call.name), cancel)
            .await;
        let retained = self.retain(executed.result, executed.output_stats);
        self.emit_result(
            call,
            &retained.result,
            retained.output_stats,
            executed.error_kind,
        );
        self.truncate_for_history(retained.result, &call.name)
    }

    /// Runs one call after the shared agent middleware approved it.
    pub(crate) async fn execute_terminal(
        &self,
        request: agent::ToolCallRequest,
    ) -> (agent::ToolOutcome, Option<OutputCaptureStats>) {
        let call = request.call().clone();
        let executed = self
            .run_tool(
                &call,
                self.registry.get(&call.name),
                request.cancellation().clone(),
            )
            .await;
        let outcome = if executed.result.is_error {
            agent::ToolOutcome::failure(
                executed.error_kind.unwrap_or(ToolErrorKind::Execution),
                result_text(&executed.result).into_owned(),
            )
        } else {
            agent::ToolOutcome::success(agent::ToolOutput::new(executed.result.content))
        };
        (outcome, executed.output_stats)
    }

    /// Publishes the start of a call entering the shared middleware stack.
    pub(crate) fn begin_terminal(&self, call: &ToolCall) {
        self.emit_started(call);
    }

    /// Applies coding output policy and publishes the final call result.
    pub(crate) fn finish_terminal(
        &self,
        call: &ToolCall,
        outcome: agent::ToolOutcome,
        previous: Option<OutputCaptureStats>,
    ) -> agent::ToolOutcome {
        let (result, error_kind) = match outcome {
            agent::ToolOutcome::Success(output) => (
                ToolResult {
                    tool_call_id: call.id.clone(),
                    name:         Some(call.name.clone()),
                    content:      output.content().to_vec(),
                    is_error:     false,
                },
                None,
            ),
            agent::ToolOutcome::Failure { kind, message } => (
                self.error_result(call, &ToolError::new(kind, message)),
                Some(kind),
            ),
            _ => (
                self.error_result(
                    call,
                    &ToolError::execution("the tool returned an unsupported outcome"),
                ),
                Some(ToolErrorKind::Execution),
            ),
        };
        let retained = self.retain(result, previous);
        self.emit_result(call, &retained.result, retained.output_stats, error_kind);
        let result = self.truncate_for_history(retained.result, &call.name);
        if result.is_error {
            agent::ToolOutcome::failure(
                error_kind.unwrap_or(ToolErrorKind::Execution),
                result_text(&result).into_owned(),
            )
        } else {
            agent::ToolOutcome::success(agent::ToolOutput::new(result.content))
        }
    }

    #[cfg(test)]
    async fn execute_sequential(
        &self,
        calls: &[ToolCall],
        cancel: &CancellationToken,
    ) -> Vec<ToolResult> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            if cancel.is_cancelled() {
                results.push(self.cancelled_result(call));
                continue;
            }
            results.push(self.execute_one(call, cancel.child_token()).await);
        }
        results
    }

    #[cfg(test)]
    async fn execute_parallel(
        &self,
        calls: &[ToolCall],
        cancel: &CancellationToken,
    ) -> Vec<ToolResult> {
        join_all(
            calls
                .iter()
                .map(|call| self.execute_one(call, cancel.child_token())),
        )
        .await
    }

    /// Runs the first human-question call and refuses the rest of the round.
    ///
    /// A question parks the prompt until a person answers, so running its peers
    /// would either race the answer or waste work the answer invalidates. Both
    /// refusals name what to do instead, and the results stay in call order.
    #[cfg(test)]
    async fn execute_question_round(
        &self,
        calls: &[ToolCall],
        cancel: &CancellationToken,
    ) -> Vec<ToolResult> {
        let first_question = calls.iter().position(|call| is_question_tool(&call.name));
        let mut results = Vec::with_capacity(calls.len());

        for (index, call) in calls.iter().enumerate() {
            if cancel.is_cancelled() {
                results.push(self.cancelled_result(call));
                continue;
            }

            let result = if Some(index) == first_question {
                self.execute_one(call, cancel.child_token()).await
            } else if is_question_tool(&call.name) {
                self.refuse(call, &ToolError::denied(ONE_QUESTION_PER_ROUND))
            } else {
                self.refuse(call, &ToolError::denied(QUESTIONS_RUN_ALONE))
            };
            results.push(result);
        }

        results
    }

    /// Validates the arguments and runs the tool.
    async fn run_tool(
        &self,
        call: &ToolCall,
        registered: Option<&RegisteredTool>,
        cancel: CancellationToken,
    ) -> ExecutedTool {
        let Some(tool) = registered else {
            return self.failed(
                call,
                &ToolError::unavailable(format!("Unknown tool: {}", call.name)),
            );
        };

        // A custom tool's input is free-form text the model wrote against a
        // provider grammar, not JSON this crate can judge.
        if !matches!(call.kind, ToolCallKind::Custom)
            && let Err(error) = validate_tool_args(&tool.definition.kind, &call.arguments)
        {
            return self.failed(call, &error);
        }

        let bound = Arc::new(SessionBoundEmitter::new(
            self.emitter.clone(),
            self.session_id,
            Some(call.id.clone()),
        ));
        let mut context = ToolContext::new(Arc::clone(self.env))
            .with_cancel(cancel)
            .with_session(self.session_id, self.root_session_id)
            .with_tool_call_id(call.id.clone())
            .with_coding_event_emitter(Arc::clone(&bound) as Arc<dyn CodingEventEmitter>);
        if let Some(provider) = self.tool_env_provider {
            context = context.with_tool_env_provider(Arc::clone(provider));
        }
        if let Some(provider) = self.human_input {
            context = context.with_human_input(Arc::clone(provider));
        }
        if let Some(redactor) = self.redactor {
            context = context.with_redactor(Arc::clone(redactor));
        }

        let (result, error_kind) = match (tool.executor)(call.arguments.clone(), context).await {
            Ok(output) => (text_result(call, output, false), None),
            Err(error) => (self.error_result(call, &error), Some(error.kind())),
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
            result:       self.error_result(call, error),
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
    fn error_result(&self, call: &ToolCall, error: &ToolError) -> ToolResult {
        let message = match self.redactor {
            Some(redactor) => redactor.redact(error.message()).into_owned(),
            None => error.message().to_owned(),
        };
        text_result(call, message, true)
    }

    /// A call that was cancelled before it started.
    #[cfg(test)]
    fn cancelled_result(&self, call: &ToolCall) -> ToolResult {
        self.error_result(call, &ToolError::cancelled(CANCELLED))
    }

    /// Publishes the started event this call never got, then refuses it.
    #[cfg(test)]
    fn refuse(&self, call: &ToolCall, error: &ToolError) -> ToolResult {
        self.emit_started(call);
        self.finish_error(call, error)
    }

    /// Bounds, publishes, and truncates a failure whose started event is out.
    #[cfg(test)]
    fn finish_error(&self, call: &ToolCall, error: &ToolError) -> ToolResult {
        let retained = self.retain(self.error_result(call, error), None);
        self.emit_result(
            call,
            &retained.result,
            retained.output_stats,
            Some(error.kind()),
        );
        self.truncate_for_history(retained.result, &call.name)
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

    fn emit_started(&self, call: &ToolCall) {
        self.emit(call, CodingEvent::ToolCallStarted {
            tool_name:    call.name.clone(),
            tool_call_id: call.id.clone(),
            arguments:    call.arguments.clone(),
        });
    }

    fn emit_result(
        &self,
        call: &ToolCall,
        result: &ToolResult,
        output_stats: OutputCaptureStats,
        error_kind: Option<ToolErrorKind>,
    ) {
        // The same bounded output goes out twice on purpose: the delta is the
        // live-streaming feed and `ToolCallCompleted` is the durable record,
        // so a store keeps the completed event and drops deltas as ephemeral.
        // No tool streams incremental deltas yet, which makes the two payloads
        // equal today.
        self.emit(call, CodingEvent::ToolCallOutputDelta {
            delta: result_text(result).into_owned(),
        });
        self.emit(call, CodingEvent::ToolCallCompleted {
            tool_name: call.name.clone(),
            tool_call_id: call.id.clone(),
            output: output_value(result),
            is_error: result.is_error,
            error_kind,
            output_bytes_observed: output_stats.observed_bytes,
            output_bytes_retained: output_stats.retained_bytes,
            output_bytes_omitted: output_stats.omitted_bytes,
        });
    }

    /// Publishes an event about one call, stamping the call on the envelope so
    /// a fragment with no identity of its own is still attributable.
    fn emit(&self, call: &ToolCall, event: CodingEvent) {
        self.emitter
            .emit_with_tool_call_id(self.session_id, event, Some(call.id.clone()));
    }
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

/// Checks a call's arguments against the tool's schema.
///
/// The check is structural rather than a full JSON Schema evaluation: an
/// object is required where the schema says `object`, every `required` property
/// must be present, and a declared property that is present must have its
/// declared `type`. Nested `properties` and array `items` are checked the same
/// way.
///
/// Everything else in a schema — `enum`, `oneOf`/`anyOf`, `minimum`/`maximum`,
/// `pattern`, formats — is left to the tool. A tool must defend itself against
/// a model's arguments in any case, so a tool that treats one of those keywords
/// as a guarantee is already relying on something this crate does not promise.
///
/// A tool described by a provider format rather than a schema is not checked,
/// because its input is free-form text.
///
/// # Errors
///
/// Returns a [`ToolError`] of kind
/// [`InvalidArguments`](ToolErrorKind::InvalidArguments) naming every problem
/// found, so a model can fix them all in one retry.
pub(crate) fn validate_tool_args(
    kind: &ToolDefinitionKind,
    arguments: &Value,
) -> Result<(), ToolError> {
    validate_tool_arguments(kind, arguments)
        .map_err(|error| ToolError::invalid_arguments(error.to_string()))
}

/// A result carrying one block of text, which is what every pebble tool
/// produces.
fn text_result(call: &ToolCall, text: String, is_error: bool) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        // Kept so a provider that labels a tool message by name has it; the
        // call id alone does not survive every wire format.
        name: Some(call.name.clone()),
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

    use async_trait::async_trait;
    use lithos_llm::types::ToolDefinition;
    use serde_json::json;
    use tokio::sync::broadcast;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::environment::EnvironmentError;
    use crate::error::Result as PebbleResult;
    use crate::event::{EventOptions, EventPump};
    use crate::human_input::{Answer, AnswerStatus, HumanInputError, Question, QuestionKind};
    use crate::test_support::MockEnvironment;
    use crate::types::{CodingAgentEvent, CommandTermination, ToolSource};

    /// An event pipeline whose events can be read once the round is over.
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
        async fn drain(self) -> Vec<CodingAgentEvent> {
            let Self {
                emitter,
                pump,
                mut received,
            } = self;
            drop(emitter);
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

    fn environment() -> Arc<dyn Environment> {
        Arc::new(MockEnvironment::default())
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
    /// through the context, before the dispatch layer completes the call.
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

    /// The smallest tool that asks a person something: it answers through
    /// whatever provider the context carries, so what is under test here is
    /// the dispatch path rather than any shipped question tool's schema.
    fn question_tool() -> RegisteredTool {
        RegisteredTool::new(
            ToolDefinition::function(
                "request_user_input",
                "Ask the person a question",
                json!({"type": "object"}),
            ),
            Arc::new(|_arguments, context: ToolContext| {
                Box::pin(async move {
                    let provider = context
                        .human_input
                        .clone()
                        .ok_or_else(|| ToolError::unavailable("No one is available to ask"))?;
                    let answers = provider
                        .ask_questions(
                            context.tool_call_id.as_deref().unwrap_or_default(),
                            vec![Question {
                                original_id:       Some("q1".to_owned()),
                                original_question: "Ship it?".to_owned(),
                                header:            None,
                                text:              "Ship it?".to_owned(),
                                kind:              QuestionKind::MultipleChoice,
                                options:           Vec::new(),
                                allow_freeform:    true,
                            }],
                            context.cancel.clone(),
                        )
                        .await?;
                    Ok(answers
                        .into_iter()
                        .flat_map(|answer| answer.answers)
                        .collect::<Vec<_>>()
                        .join(", "))
                })
            }),
        )
        .with_source(ToolSource::Native)
    }

    struct StubHumanInput;

    #[async_trait]
    impl HumanInputProvider for StubHumanInput {
        async fn ask_questions(
            &self,
            _tool_call_id: &str,
            questions: Vec<Question>,
            _cancel_token: CancellationToken,
        ) -> Result<Vec<Answer>, HumanInputError> {
            Ok(questions
                .into_iter()
                .map(|question| Answer {
                    original_id:       question.original_id,
                    original_question: question.original_question,
                    answers:           vec!["Ship".to_owned()],
                    status:            AnswerStatus::Answered,
                })
                .collect())
        }
    }

    fn registry_with(tools: impl IntoIterator<Item = RegisteredTool>) -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        for tool in tools {
            registry.register(tool);
        }
        registry
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
        let registry = registry_with([echo_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
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
        let registry = registry_with([failing_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
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
        let registry = registry_with([tool]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();
        let redactor: Arc<dyn Redactor> = Arc::new(DropKeys);

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .with_redactor(&redactor)
        .execute_one(
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
        let registry = ToolRegistry::new();
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(&call("nope", "call_1", json!({})), CancellationToken::new())
        .await;

        assert!(result.is_error);
        assert_eq!(text_of(&result), "Unknown tool: nope");
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
        let registry = registry_with([tool]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(&call("echo", "call_1", json!({})), CancellationToken::new())
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
        let registry = registry_with([echo_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let budget = config.tool_output_retention_bytes;
        let text = "x".repeat(budget + 100);
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
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
        let registry = registry_with([echo_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        // Output made almost entirely of characters JSON escapes: it fits the
        // retained budget as text and blows past it once serialized.
        let text = format!(
            "HEAD{}TAIL",
            "\0".repeat(config.tool_output_retention_bytes - "echo: HEADTAIL".len())
        );
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
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
        let registry = registry_with([process_tool(7)]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
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
        let registry = registry_with([process_tool(0)]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
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
    async fn a_question_round_runs_one_question_and_refuses_its_peers() {
        let registry = registry_with([question_tool(), echo_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let provider: Arc<dyn HumanInputProvider> = Arc::new(StubHumanInput);
        let events = Events::new();
        let calls = [
            call("request_user_input", "call_question", json!({})),
            call("echo", "call_echo", json!({"text": "hello"})),
        ];

        let results = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .with_human_input(&provider)
        .execute(&calls, true, &CancellationToken::new())
        .await;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool_call_id, "call_question");
        assert!(!results[0].is_error);
        assert_eq!(text_of(&results[0]), "Ship");
        assert_eq!(results[1].tool_call_id, "call_echo");
        assert!(results[1].is_error);
        assert!(text_of(&results[1]).contains("human-question tools must run alone"));
        drop(events.drain().await);
    }

    #[tokio::test]
    async fn only_the_first_of_several_question_calls_runs() {
        let registry = registry_with([question_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let provider: Arc<dyn HumanInputProvider> = Arc::new(StubHumanInput);
        let events = Events::new();
        let calls = [
            call("request_user_input", "call_first", json!({})),
            call("request_user_input", "call_second", json!({})),
        ];

        let results = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .with_human_input(&provider)
        .execute(&calls, true, &CancellationToken::new())
        .await;

        assert!(!results[0].is_error);
        assert!(results[1].is_error);
        assert!(
            text_of(&results[1]).contains("Combine all questions into a single questions[] batch")
        );
        drop(events.drain().await);
    }

    #[tokio::test]
    async fn a_cancelled_round_answers_every_call_without_starting_one() {
        let registry = registry_with([echo_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let events = Events::new();
        let calls = [
            call("echo", "call_1", json!({"text": "one"})),
            call("echo", "call_2", json!({"text": "two"})),
        ];

        let results = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute(&calls, false, &cancel)
        .await;

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.is_error);
            assert_eq!(text_of(result), "Cancelled");
        }
        assert!(
            events.drain().await.is_empty(),
            "a call that never started publishes nothing"
        );
    }

    #[tokio::test]
    async fn a_cancelled_parallel_round_answers_every_call_without_starting_one() {
        // A parallel round starts all its calls at once, so the check that a
        // sequential round makes per call has to happen before any of them.
        let registry = registry_with([echo_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let events = Events::new();
        let calls = [
            call("echo", "call_1", json!({"text": "one"})),
            call("echo", "call_2", json!({"text": "two"})),
        ];

        let results = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute(&calls, true, &cancel)
        .await;

        assert_eq!(
            results
                .iter()
                .map(|result| result.tool_call_id.as_str())
                .collect::<Vec<_>>(),
            ["call_1", "call_2"]
        );
        for result in &results {
            assert!(result.is_error);
            assert_eq!(text_of(result), "Cancelled");
        }
        assert!(
            events.drain().await.is_empty(),
            "a call that never started publishes nothing"
        );
    }

    #[tokio::test]
    async fn parallel_calls_come_back_in_call_order() {
        let registry = registry_with([echo_tool()]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();
        let calls = [
            call("echo", "call_1", json!({"text": "one"})),
            call("echo", "call_2", json!({"text": "two"})),
            call("echo", "call_3", json!({"text": "three"})),
        ];

        let results = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute(&calls, true, &CancellationToken::new())
        .await;

        assert_eq!(
            results
                .iter()
                .map(|result| result.tool_call_id.clone())
                .collect::<Vec<_>>(),
            ["call_1", "call_2", "call_3"]
        );
        assert_eq!(results.iter().map(text_of).collect::<Vec<_>>(), [
            "echo: one",
            "echo: two",
            "echo: three"
        ]);
        drop(events.drain().await);
    }

    #[tokio::test]
    async fn history_keeps_a_smaller_copy_than_the_events_carried() {
        let registry = registry_with([RegisteredTool::new(
            ToolDefinition::function("shell", "Runs a command", json!({})),
            Arc::new(|_arguments, _context| Box::pin(async { Ok("x".repeat(60_000)) })),
        )
        .with_source(ToolSource::Native)]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
            &call("shell", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        // The shell tool's history limit is 30,000 characters; the retention
        // budget is far larger, so the event kept the whole output.
        assert!(text_of(&result).len() < 60_000);
        assert_eq!(result.tool_call_id, "call_1");

        let published = events.drain().await;
        let CodingEvent::ToolCallCompleted { output, .. } = completion(&published) else {
            panic!("a completion event");
        };
        assert_eq!(output.as_str().map(str::len), Some(60_000));
    }

    #[tokio::test]
    async fn truncation_preserves_the_call_id_and_the_error_state() {
        let registry = registry_with([RegisteredTool::new(
            ToolDefinition::function("shell", "Runs a command", json!({})),
            Arc::new(|_arguments, _context| {
                Box::pin(async { Err(ToolError::execution("x".repeat(60_000))) })
            }),
        )
        .with_source(ToolSource::Native)]);
        let environment = environment();
        let config = CodingAgentOptions::default();
        let events = Events::new();

        let result = ToolDispatch::new(
            &registry,
            &environment,
            &config,
            &events.emitter,
            "ses_1",
            "ses_1",
        )
        .execute_one(
            &call("shell", "call_1", json!({})),
            CancellationToken::new(),
        )
        .await;

        assert_eq!(result.tool_call_id, "call_1");
        assert!(result.is_error);
        assert!(text_of(&result).len() < 60_000);
        drop(events.drain().await);
    }

    #[test]
    fn a_schema_with_nothing_to_say_accepts_anything() {
        for schema in [json!(null), json!({})] {
            let kind = ToolDefinitionKind::Function {
                input_schema: schema,
            };
            assert!(validate_tool_args(&kind, &json!("anything")).is_ok());
        }
    }

    #[test]
    fn a_custom_tool_is_not_schema_checked() {
        let kind = ToolDefinitionKind::Custom {
            format: json!({"type": "grammar"}),
        };
        assert!(validate_tool_args(&kind, &json!("*** Begin Patch")).is_ok());
    }

    #[test]
    fn arguments_of_the_wrong_shape_are_rejected() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({"type": "object", "properties": {}}),
        };

        let error = validate_tool_args(&kind, &json!("not an object"))
            .expect_err("a string is not an object");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert!(
            error
                .message()
                .contains("arguments: expected object, got string"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn every_missing_required_property_is_named_at_once() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
                "required": ["name", "age"],
            }),
        };

        let error = validate_tool_args(&kind, &json!({})).expect_err("both properties are missing");

        assert!(error.message().contains("\"name\""), "{}", error.message());
        assert!(error.message().contains("\"age\""), "{}", error.message());
    }

    #[test]
    fn a_declared_property_is_checked_against_its_type() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            }),
        };

        assert!(validate_tool_args(&kind, &json!({"count": 3})).is_ok());
        assert!(validate_tool_args(&kind, &json!({"count": "three"})).is_err());
        // A property the schema does not describe is not pebble's to judge.
        assert!(validate_tool_args(&kind, &json!({"other": "three"})).is_ok());
    }

    #[test]
    fn nested_objects_and_array_items_are_checked_too() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                },
            }),
        };

        assert!(validate_tool_args(&kind, &json!({"questions": [{"text": "Ship it?"}]})).is_ok());
        let error = validate_tool_args(&kind, &json!({"questions": [{"header": "Decision"}]}))
            .expect_err("the item misses a required property");
        assert!(
            error.message().contains("arguments.questions[0]"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn a_type_union_accepts_either_member() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {"limit": {"type": ["integer", "null"]}},
            }),
        };

        assert!(validate_tool_args(&kind, &json!({"limit": 10})).is_ok());
        assert!(validate_tool_args(&kind, &json!({"limit": null})).is_ok());
        assert!(validate_tool_args(&kind, &json!({"limit": "ten"})).is_err());
    }
}
