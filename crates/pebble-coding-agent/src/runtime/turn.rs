//! The coding layer that projects Pebble's durable behavior onto `Agent`.

use std::collections::VecDeque;
use std::mem;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Instant, SystemTime};

use async_trait::async_trait;
use lithos_llm::middleware::CallContext;
use lithos_llm::types::{
    ContentPart, Error as LlmError, Message as LlmMessage, Request, Response, ResponseStream,
    ToolCall, ToolResult,
};
use pebble_agent as agent;
use pebble_agent::integration::{ToolRoundContext, ToolRoundExecutor};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::control::SteeringItem;
use super::retry::RetryEventBridge;
use super::{CodingRuntime, PromptTotals};
use crate::compaction::{CompactionRequest, check_context_usage, compact_context};
use crate::config::{CodingAgentOptions, ToolHookDecision};
use crate::context_window::{
    ContextWindowInput, build_local_snapshot, context_window_from_response_usage,
};
use crate::environment::Environment;
use crate::error::{Error, ErrorData, Result};
use crate::event::Emitter;
use crate::file_tracker::FileTracker;
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::loop_detection::detect_loop;
use crate::profile::ModelFacts;
use crate::reasoning::ReasoningOutput;
use crate::redact::Redactor;
use crate::skills::{ExpandedInput, Skill, SkillExpansion, expand_skill};
use crate::subagent::SubagentSupervisor;
use crate::task_reminder::maybe_task_reminder;
use crate::tool::{
    NativeTool, ToolDefinitionWithSource, ToolDispatch, ToolEnvProvider, ToolRegistry,
    canonical_tool_name,
};
use crate::types::{
    CodingAgentState, CodingEvent, ContextWindowSnapshot, CostSource, LlmOutputKind, LlmRetryPhase,
    Message, SkillActivationSource, TokenUsage, ToolErrorKind, message_text,
};

/// How many failed response streams Pebble replays after the first attempt.
const STREAM_CONSUME_RETRIES: u32 = 3;

/// What the model is told when it keeps making the same calls.
const LOOP_WARNING: &str = "WARNING: Loop detected. You appear to be repeating the same tool \
                            calls. Please try a different approach or ask for clarification.";

#[derive(Clone)]
pub(super) struct CodingAgentBridge {
    state:             Arc<Mutex<BridgeState>>,
    client:            lithos_llm::Client,
    model_selector:    String,
    model:             String,
    provider:          String,
    system_prompt:     String,
    facts:             ModelFacts,
    config:            CodingAgentOptions,
    registry:          ToolRegistry,
    env:               Arc<dyn Environment>,
    human_input:       Option<Arc<dyn HumanInputProvider>>,
    tool_env_provider: Arc<Mutex<Option<Arc<dyn ToolEnvProvider>>>>,
    redactor:          Arc<dyn Redactor>,
    emitter:           Emitter,
    session_id:        String,
    root_session_id:   String,
    memory_tokens:     u64,
    skills_tokens:     u64,
    control_state:     Arc<Mutex<super::control::ControlState>>,
    control_notify:    Arc<Notify>,
    /// The token that ends the prompt in progress. A child of the runtime's
    /// terminal token, so it also fires when the session is shut down, and set
    /// afresh by [`begin_prompt`](Self::begin_prompt) for every prompt.
    prompt_cancel:     Arc<Mutex<CancellationToken>>,
    followup_queue:    Arc<Mutex<VecDeque<String>>>,
    subagents:         Option<SubagentSupervisor>,
    skills:            Vec<Skill>,
}

struct BridgeState {
    history: History,
    file_tracker: FileTracker,
    totals: PromptTotals,
    activated_skill_context_observed: bool,
    compaction_failed: bool,
    pending_task_reminder: Option<Message>,
    local_context_window: Option<ContextWindowSnapshot>,
    inference_start: Option<Instant>,
    boundary_error: Option<Error>,
}

impl CodingAgentBridge {
    fn from_runtime(runtime: &mut CodingRuntime) -> Self {
        Self {
            state:             Arc::new(Mutex::new(BridgeState {
                history: mem::take(&mut runtime.history),
                file_tracker: mem::take(&mut runtime.file_tracker),
                totals: PromptTotals::default(),
                activated_skill_context_observed: runtime.activated_skill_context_observed,
                compaction_failed: false,
                pending_task_reminder: None,
                local_context_window: None,
                inference_start: None,
                boundary_error: None,
            })),
            client:            runtime.client.clone(),
            model_selector:    runtime.model_selector.clone(),
            model:             runtime.model.clone(),
            provider:          runtime.provider.clone(),
            system_prompt:     runtime.system_prompt.clone(),
            facts:             runtime.facts,
            config:            runtime.config.clone(),
            registry:          runtime.registry.clone(),
            env:               Arc::clone(&runtime.env),
            human_input:       runtime.human_input.clone(),
            tool_env_provider: Arc::new(Mutex::new(runtime.tool_env_provider.clone())),
            redactor:          Arc::clone(&runtime.redactor),
            emitter:           runtime.emitter.clone(),
            session_id:        runtime.id.clone(),
            root_session_id:   runtime.root_session_id.clone(),
            memory_tokens:     runtime.memory_tokens,
            skills_tokens:     runtime.skills_tokens,
            control_state:     Arc::clone(&runtime.control_state),
            control_notify:    Arc::clone(&runtime.control_notify),
            prompt_cancel:     Arc::new(Mutex::new(runtime.cancel_token.clone())),
            followup_queue:    Arc::clone(&runtime.followup_queue),
            subagents:         runtime.subagents.clone(),
            skills:            runtime.skills.clone(),
        }
    }

    fn begin_prompt(&self, prompt_cancel: CancellationToken) {
        *self
            .prompt_cancel
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = prompt_cancel;
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.totals = PromptTotals::default();
        state.compaction_failed = false;
        state.pending_task_reminder = None;
        state.local_context_window = None;
        state.inference_start = None;
        state.boundary_error = None;
    }

    fn restore_runtime(&self, runtime: &mut CodingRuntime) {
        self.finish_inference();
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        runtime.history = state.history.clone();
        runtime.file_tracker = state.file_tracker.clone();
        runtime.activated_skill_context_observed = state.activated_skill_context_observed;
        runtime.last_prompt = state.totals;
    }

    #[cfg(test)]
    pub(super) fn set_tool_env_provider(&self, provider: Arc<dyn ToolEnvProvider>) {
        *self
            .tool_env_provider
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(provider);
    }

    /// The token that ends the prompt in progress.
    fn prompt_cancel(&self) -> CancellationToken {
        self.prompt_cancel
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn take_boundary_error(&self) -> Option<Error> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .boundary_error
            .take()
    }

    fn emit(&self, event: CodingEvent) {
        self.emitter.emit(self.session_id.clone(), event);
    }

    fn effective_tools(&self) -> Vec<ToolDefinitionWithSource> {
        self.registry.definitions_with_source_for_policy(
            self.config.tool_access_policy.as_deref(),
            self.config.tool_exposure_mode,
        )
    }

    fn finish_inference(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(started) = state.inference_start.take() {
            state.totals.timing.inference = state
                .totals
                .timing
                .inference
                .saturating_add(started.elapsed());
        }
    }

    fn settle_interrupts(&self) {
        let generations = {
            let mut control = self
                .control_state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let first = control.settled_interrupt_generation.saturating_add(1);
            let last = control.interrupt_generation;
            control.settled_interrupt_generation = last;
            if first <= last {
                (first..=last).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };
        for generation in generations {
            self.emit(CodingEvent::RoundInterrupted { generation });
        }
    }

    async fn wait_for_steer_if_needed(&self, cancel: &CancellationToken) {
        let prompt_cancel = self.prompt_cancel();
        loop {
            let notified = self.control_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let should_wait = {
                let control = self
                    .control_state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                control.waiting_for_steer && control.queue.is_empty()
            };
            if !should_wait {
                return;
            }
            tokio::select! {
                () = prompt_cancel.cancelled() => return,
                () = cancel.cancelled() => return,
                () = notified => {}
            }
        }
    }

    async fn compact_if_needed(&self) -> bool {
        let estimate = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            check_context_usage(
                &self.system_prompt,
                &state.history,
                self.facts.context_window_tokens,
                self.config.compaction_threshold_percent,
                &self.emitter,
                &self.session_id,
            )
        };
        let Some(estimate) = estimate else {
            return false;
        };
        if !self.config.enable_context_compaction {
            return false;
        }

        let (mut history, file_tracker) = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            (state.history.clone(), state.file_tracker.clone())
        };
        let request = CompactionRequest {
            model: &self.model_selector,
            facts: self.facts,
            preserve_turns: self.config.compaction_preserve_turns,
            estimate,
        };
        if let Err(error) = compact_context(
            &mut history,
            &self.client,
            &file_tracker,
            request,
            &self.emitter,
            &self.session_id,
        )
        .await
        {
            self.emit(CodingEvent::Error {
                error: ErrorData::from(&error),
            });
            return true;
        }
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .history = history;
        false
    }

    /// Compacts once per prompt, remembering a failure so it is not retried.
    async fn compact_once_if_needed(&self) {
        let compaction_failed = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .compaction_failed;
        if !compaction_failed && self.compact_if_needed().await {
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .compaction_failed = true;
        }
    }

    fn stage_task_reminder(&self) {
        let tools = self.effective_tools();
        let names = tools
            .iter()
            .map(|tool| tool.definition.name.as_str())
            .collect::<Vec<_>>();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.pending_task_reminder =
            maybe_task_reminder(&state.history, &names).map(|content| Message::System {
                content,
                timestamp: SystemTime::now(),
            });
    }

    fn sync_messages(&self, context: &mut agent::TurnBoundaryContext<'_>, include_reminder: bool) {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *context.messages_mut() = state.history.to_llm_messages();
        if include_reminder && let Some(reminder) = &state.pending_task_reminder {
            context.messages_mut().push(reminder.to_llm_message());
        }
    }

    fn commit_user_message(&self, message: &LlmMessage) {
        let text = message_text(message);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.compaction_failed = false;
        state.history.push(Message::User {
            content:   text.clone(),
            timestamp: SystemTime::now(),
        });
        drop(state);
        self.emit(CodingEvent::UserInput { text });
    }

    fn commit_steering(&self, message: &LlmMessage) {
        self.settle_interrupts();
        let queued = self
            .control_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .queue
            .pop_front();
        let fallback = message_text(message);
        let timestamp = SystemTime::now();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match queued.unwrap_or_else(|| SteeringItem::steering(fallback)) {
            SteeringItem::Steering { text, actor } => {
                state.history.push(Message::Steering {
                    content: text.clone(),
                    timestamp,
                });
                drop(state);
                self.emit(CodingEvent::SteeringInjected { text, actor });
            }
        }
    }

    fn commit_assistant(&self, response: &Response) {
        self.finish_inference();
        let text = response.text();
        let tool_calls = tool_calls_of(response);
        let reasoning = ReasoningOutput::from_content(&response.content);
        let provider_parts = provider_parts_of(response);
        let usage = TokenUsage::from(response.usage);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let context_window = state
            .local_context_window
            .take()
            .map(|local| context_window_from_response_usage(&local, usage));

        state.totals.usage = state.totals.usage.saturating_add(usage);
        if let Some(cost) = response.cost {
            state.totals.cost_usd_micros = Some(
                state
                    .totals
                    .cost_usd_micros
                    .unwrap_or(0)
                    .saturating_add(cost.usd_micros),
            );
        }
        if let Some(reminder) = state.pending_task_reminder.take() {
            state.history.push(reminder);
        }
        state.history.push(Message::Assistant {
            content: text.clone(),
            tool_calls: tool_calls.clone(),
            provider_parts,
            usage,
            response_id: response.id.clone().unwrap_or_default(),
            timestamp: SystemTime::now(),
        });
        drop(state);

        let answering_model = response.model.model().as_str();
        self.emit(CodingEvent::AssistantMessage {
            text,
            model: if answering_model.is_empty() {
                self.model.clone()
            } else {
                answering_model.to_owned()
            },
            usage,
            cost_usd_micros: response.cost.map(|cost| cost.usd_micros),
            cost_source: response.cost.map(|cost| CostSource::from(cost.source)),
            tool_call_count: tool_calls.len(),
            context_window,
            reasoning,
        });
    }

    fn measure_request(&self, request: &Request) -> ContextWindowSnapshot {
        let tools = self.effective_tools();
        let activated = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .activated_skill_context_observed;
        build_local_snapshot(ContextWindowInput {
            request,
            tools: &tools,
            system_prompt: &self.system_prompt,
            memory_tokens: self.memory_tokens,
            skills_tokens: self.skills_tokens,
            activated_skill_context_observed: activated,
            provider: &self.provider,
            model: &self.model,
            context_window_tokens: self.facts.context_window_tokens,
        })
    }
}

impl agent::EventProjection for CodingAgentBridge {
    fn project(&self, event: &agent::AgentEvent) {
        match event {
            agent::AgentEvent::UserMessage { message } => self.commit_user_message(message),
            agent::AgentEvent::SteeringMessage { message } => self.commit_steering(message),
            agent::AgentEvent::ModelRequestStarted { request, .. } => {
                let local = self.measure_request(request);
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                state.local_context_window = Some(local);
                state.inference_start = Some(Instant::now());
                drop(state);
                self.emit(CodingEvent::LlmRequestStarted {
                    requested_model: self.model.clone(),
                });
            }
            agent::AgentEvent::FirstOutput { kind } => {
                let kind = match kind {
                    agent::FirstOutputKind::Text => LlmOutputKind::Text,
                    agent::FirstOutputKind::Reasoning => LlmOutputKind::Reasoning,
                    agent::FirstOutputKind::ToolCall => LlmOutputKind::ToolCall,
                    _ => return,
                };
                self.emit(CodingEvent::LlmFirstOutput { kind });
            }
            agent::AgentEvent::TextDelta { delta } => {
                self.emit(CodingEvent::TextDelta {
                    delta: delta.clone(),
                });
            }
            agent::AgentEvent::ReasoningDelta { delta } => {
                self.emit(CodingEvent::ReasoningDelta {
                    delta: delta.clone(),
                });
            }
            agent::AgentEvent::OutputReplaced => {
                self.emit(CodingEvent::AssistantOutputReplace {
                    text:      String::new(),
                    reasoning: None,
                });
            }
            agent::AgentEvent::TurnReplay {
                failed_attempt,
                delay_seconds,
                error,
            } => {
                self.emit(CodingEvent::LlmRetry {
                    provider:   self.provider.clone(),
                    model:      self.model.clone(),
                    attempt:    usize::try_from(failed_attempt.saturating_sub(1))
                        .unwrap_or(usize::MAX),
                    delay_secs: *delay_seconds,
                    error:      ErrorData::from(error),
                    phase:      LlmRetryPhase::Consume,
                });
            }
            agent::AgentEvent::AssistantMessage { response } => self.commit_assistant(response),
            agent::AgentEvent::TurnInterrupted => {
                self.finish_inference();
                self.state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .pending_task_reminder = None;
                self.settle_interrupts();
            }
            _ => {}
        }
    }
}

impl agent::ToolProvider for CodingAgentBridge {
    fn tools_for_turn(&self, _context: agent::TurnContext<'_>) -> Vec<agent::Tool> {
        let executor: Arc<dyn agent::ToolExecutor> = Arc::new(UnusedToolExecutor);
        self.registry
            .definitions_with_source()
            .into_iter()
            .map(|tool| agent::Tool::new(tool.definition, Arc::clone(&executor)))
            .collect()
    }
}

impl agent::ToolAccessPolicy for CodingAgentBridge {
    fn access(&self, context: agent::ToolAccessContext<'_>) -> agent::ToolAccess {
        self.config
            .tool_access_denial_reason(&context.definition().name)
            .map_or(agent::ToolAccess::Allowed, |reason| {
                agent::ToolAccess::Denied { reason }
            })
    }
}

#[async_trait]
impl agent::ToolCallHooks for CodingAgentBridge {
    async fn before_tool_call(
        &self,
        context: agent::ToolCallContext<'_>,
        _cancel: &CancellationToken,
    ) -> agent::BeforeToolCall {
        let Some(hooks) = self.config.tool_hooks.as_ref() else {
            return agent::BeforeToolCall::Proceed;
        };
        match hooks
            .pre_tool_use(&context.call().name, &context.call().arguments)
            .await
        {
            ToolHookDecision::Proceed => agent::BeforeToolCall::Proceed,
            ToolHookDecision::Block { reason } => agent::BeforeToolCall::Block { reason },
        }
    }

    async fn after_tool_call(
        &self,
        context: agent::ToolCallContext<'_>,
        outcome: agent::ToolCallOutcome<'_>,
        _cancel: &CancellationToken,
    ) {
        let Some(hooks) = self.config.tool_hooks.as_ref() else {
            return;
        };
        let result = outcome.result();
        let content = result
            .content
            .iter()
            .find_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        if result.is_error {
            hooks
                .post_tool_use_failure(
                    &context.call().name,
                    &context.call().id,
                    content,
                    outcome.error_kind().unwrap_or(ToolErrorKind::Execution),
                )
                .await;
        } else {
            hooks
                .post_tool_use(&context.call().name, &context.call().id, content)
                .await;
        }
    }
}

struct UnusedToolExecutor;

#[async_trait]
impl agent::ToolExecutor for UnusedToolExecutor {
    async fn execute(
        &self,
        _context: agent::ToolContext,
        _arguments: serde_json::Value,
    ) -> StdResult<agent::ToolOutput, agent::ToolError> {
        Err(agent::ToolError::new(
            "coding tools must run through Pebble's round executor",
        ))
    }
}

#[async_trait]
impl ToolRoundExecutor for CodingAgentBridge {
    async fn execute_round(
        &self,
        context: ToolRoundContext<'_>,
        cancel: &CancellationToken,
    ) -> Vec<ToolResult> {
        let mut dispatch = ToolDispatch::new(
            &self.registry,
            &self.env,
            &self.config,
            &self.emitter,
            &self.session_id,
            &self.root_session_id,
        );
        let tool_env_provider = self
            .tool_env_provider
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(provider) = tool_env_provider.as_ref() {
            dispatch = dispatch.with_tool_env_provider(provider);
        }
        if let Some(provider) = self.human_input.as_ref() {
            dispatch = dispatch.with_human_input(provider);
        }
        dispatch = dispatch.with_redactor(&self.redactor);

        let started = Instant::now();
        let results = dispatch.execute_agent_round(context, cancel).await;
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.totals.timing.tool = state.totals.timing.tool.saturating_add(started.elapsed());
        if activated_a_skill(context.calls(), &results) {
            state.activated_skill_context_observed = true;
        }
        state
            .file_tracker
            .record_from_tool_calls(context.calls(), &results);
        state.history.push(Message::ToolResults {
            results:   results.clone(),
            timestamp: SystemTime::now(),
        });
        let loop_detected = self.config.enable_loop_detection
            && detect_loop(&state.history, self.config.loop_detection_window);
        if loop_detected {
            state.history.push(Message::Steering {
                content:   LOOP_WARNING.to_owned(),
                timestamp: SystemTime::now(),
            });
        }
        drop(state);
        if loop_detected {
            self.emit(CodingEvent::LoopDetected);
        }
        results
    }
}

#[async_trait]
impl agent::TurnBoundaryHooks for CodingAgentBridge {
    async fn before_model(
        &self,
        mut context: agent::TurnBoundaryContext<'_>,
        cancel: &CancellationToken,
    ) -> StdResult<(), agent::TurnBoundaryError> {
        self.settle_interrupts();
        self.wait_for_steer_if_needed(cancel).await;
        if cancel.is_cancelled() || self.prompt_cancel().is_cancelled() {
            return Ok(());
        }

        self.compact_once_if_needed().await;
        self.stage_task_reminder();
        self.sync_messages(&mut context, true);
        Ok(())
    }

    async fn after_model(
        &self,
        mut context: agent::TurnBoundaryContext<'_>,
        _response: &Response,
        _cancel: &CancellationToken,
    ) -> StdResult<(), agent::TurnBoundaryError> {
        self.compact_once_if_needed().await;
        self.sync_messages(&mut context, false);
        Ok(())
    }

    async fn after_answer(
        &self,
        _context: agent::TurnContext<'_>,
        _response: &Response,
        cancel: &CancellationToken,
    ) -> StdResult<agent::TurnBoundaryAction, agent::TurnBoundaryError> {
        // The completion close-door race. A steer may be queued right now, or a
        // steering lease may be held by an external source that is about to
        // send one. Anything queued sends the loop around again to drain it; an
        // open lease parks the prompt until the last lease drops. Both the
        // check and the park read the shared control state under its lock, so a
        // steer arriving mid-decision is never lost between the two.
        let prompt_cancel = self.prompt_cancel();
        loop {
            let notified = self.control_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let parked = {
                let control = self
                    .control_state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                if !control.queue.is_empty() {
                    return Ok(agent::TurnBoundaryAction::Continue);
                }
                control.steering_leases > 0
            };
            if !parked || cancel.is_cancelled() || prompt_cancel.is_cancelled() {
                break;
            }
            tokio::select! {
                () = prompt_cancel.cancelled() => break,
                () = cancel.cancelled() => break,
                () = notified => {}
            }
        }

        let followup = self
            .followup_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        if let Some(followup) = followup {
            let expanded = if self.skills.is_empty() {
                ExpandedInput {
                    text:       followup,
                    skill_name: None,
                }
            } else {
                expand_skill(&self.skills, &followup).map_err(|error| {
                    agent::TurnBoundaryError::new(format!("expanding follow-up input: {error}"))
                })?
            };
            if let Some(name) = expanded.skill_name {
                self.state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .activated_skill_context_observed = true;
                self.emit(CodingEvent::SkillActivated {
                    skill_name: name,
                    source:     SkillActivationSource::Slash,
                });
            }
            return Ok(agent::TurnBoundaryAction::ContinueWith(
                expanded.text.into(),
            ));
        }

        let Some(supervisor) = self.subagents.as_ref() else {
            return Ok(agent::TurnBoundaryAction::Complete);
        };
        match supervisor.next_parent_notification_turn(cancel).await {
            Ok(Some(turn)) => Ok(agent::TurnBoundaryAction::ContinueWith(turn.into())),
            Ok(None) => Ok(agent::TurnBoundaryAction::Complete),
            Err(error) => {
                let message = error.to_string();
                self.state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .boundary_error = Some(error);
                Err(agent::TurnBoundaryError::new(message))
            }
        }
    }
}

#[derive(Clone)]
struct CodingModelService {
    client:     lithos_llm::Client,
    emitter:    Emitter,
    session_id: String,
    provider:   String,
    model:      String,
}

#[async_trait]
impl agent::ModelService for CodingModelService {
    async fn stream(
        &self,
        request: Request,
        mut context: CallContext,
    ) -> StdResult<ResponseStream, LlmError> {
        context.extensions_mut().insert(RetryEventBridge::new(
            self.emitter.clone(),
            self.session_id.clone(),
            self.provider.clone(),
            self.model.clone(),
        ));
        self.client.stream_with_context(request, context).await
    }
}

impl CodingRuntime {
    /// Processes one input through the shared generic agent loop.
    ///
    /// `prompt_cancel` ends this prompt alone. It is a child of the runtime's
    /// terminal token, so a shutdown ends the prompt too; the two are told
    /// apart afterwards, because only the terminal one closes the session.
    pub(super) async fn process_input(
        &mut self,
        input: &str,
        skill_expansion: SkillExpansion,
        prompt_cancel: &CancellationToken,
    ) -> Result<Option<String>> {
        if self.state == CodingAgentState::Closed {
            return Err(Error::SessionClosed);
        }
        self.check_pump().await?;
        self.transition(CodingAgentState::Thinking);

        let expanded = self.expand_input(input, skill_expansion)?;
        if let Some(name) = &expanded.skill_name {
            self.activated_skill_context_observed = true;
            self.emit(CodingEvent::SkillActivated {
                skill_name: name.clone(),
                source:     SkillActivationSource::Slash,
            });
        }

        self.ensure_coding_agent()?;
        let bridge = self
            .coding_bridge
            .clone()
            .ok_or_else(|| Error::InvalidState("the coding bridge was not built".to_owned()))?;
        bridge.begin_prompt(prompt_cancel.clone());
        let mut agent = self
            .coding_agent
            .take()
            .ok_or_else(|| Error::InvalidState("the coding agent was not built".to_owned()))?;

        self.install_agent_control(agent.control_handle());
        let result = agent
            .prompt_with_cancellation(expanded.text, prompt_cancel)
            .await;
        self.clear_agent_control();
        self.coding_agent = Some(agent);
        bridge.restore_runtime(self);
        if let Some(error) = bridge.take_boundary_error() {
            return Err(error);
        }

        match result {
            Ok(outcome) => {
                let text = outcome.text();
                Ok((!text.trim().is_empty()).then_some(text))
            }
            Err(agent::AgentError::Aborted) => Err(self.prompt_aborted().await),
            Err(agent::AgentError::Model { source }) => Err(self.emit_llm_error(source)),
            Err(error) => {
                self.check_pump().await?;
                Err(Error::InvalidState(format!(
                    "the coding agent could not process the prompt: {error}"
                )))
            }
        }
    }

    fn ensure_coding_agent(&mut self) -> Result<()> {
        if self.coding_agent.is_some() {
            return Ok(());
        }

        let bridge = Arc::new(CodingAgentBridge::from_runtime(self));
        let model_service = CodingModelService {
            client:     self.client.clone(),
            emitter:    self.emitter.clone(),
            session_id: self.id.clone(),
            provider:   self.provider.clone(),
            model:      self.model.clone(),
        };
        let messages = bridge
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .history
            .to_llm_messages();
        let config = agent::AgentConfig {
            max_output_tokens: self.max_output_tokens(),
            reasoning_effort: self.config.reasoning_effort,
            speed: self.config.speed,
            turn_replay: self.config.turn_replay,
            max_turn_replays: STREAM_CONSUME_RETRIES,
            ..agent::AgentConfig::default()
        };
        let agent = agent::Agent::builder(model_service, self.model_selector.clone())
            .system_prompt(self.system_prompt.clone())
            .messages(messages)
            .tool_provider(bridge.clone())
            .tool_access_policy(bridge.clone())
            .tool_call_hooks(bridge.clone())
            .tool_round_executor(bridge.clone())
            .turn_boundary_hooks(bridge.clone())
            .event_projection(bridge.clone())
            .config(config)
            .build()
            .map_err(|error| Error::InvalidState(format!("building the coding agent: {error}")))?;

        self.coding_bridge = Some(bridge);
        self.coding_agent = Some(agent);
        Ok(())
    }

    fn install_agent_control(&self, agent: pebble_agent::AgentControlHandle) {
        let control = self
            .control_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for item in &control.queue {
            agent.enqueue_steering(item.text().to_owned());
        }
        *self
            .active_agent_control
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(agent);
    }

    fn clear_agent_control(&self) {
        *self
            .active_agent_control
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// Expands a `/name` reference in the input, where one is allowed.
    fn expand_input(&self, input: &str, expansion: SkillExpansion) -> Result<ExpandedInput> {
        if self.skills.is_empty() || expansion == SkillExpansion::Skip {
            return Ok(ExpandedInput {
                text:       input.to_owned(),
                skill_name: None,
            });
        }
        expand_skill(&self.skills, input).map_err(|error| Error::InvalidState(error.to_string()))
    }

    /// The most tokens the model may produce in one turn.
    fn max_output_tokens(&self) -> Option<u32> {
        let configured = self
            .config
            .max_tokens
            .and_then(|tokens| u32::try_from(tokens).ok());
        let from_catalog = || {
            self.facts
                .max_output_tokens
                .map(|tokens| u32::try_from(tokens).unwrap_or(u32::MAX))
        };
        configured
            .or_else(from_catalog)
            .filter(|tokens| *tokens > 0)
    }
}

fn tool_calls_of(response: &Response) -> Vec<ToolCall> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

fn provider_parts_of(response: &Response) -> Vec<ContentPart> {
    response
        .content
        .iter()
        .filter(|part| matches!(part, ContentPart::Opaque { .. } | ContentPart::Reasoning(_)))
        .cloned()
        .collect()
}

fn activated_a_skill(calls: &[ToolCall], results: &[ToolResult]) -> bool {
    calls.iter().zip(results).any(|(call, result)| {
        !result.is_error && canonical_tool_name(&call.name) == NativeTool::UseSkill.canonical_name()
    })
}

#[cfg(test)]
mod tests {
    use std::slice;

    use lithos_llm::catalog::{ModelId, ProviderId};
    use lithos_llm::types::ReasoningContent;

    use super::*;

    fn response_with(content: Vec<ContentPart>) -> Response {
        Response::new(ProviderId::new("test"), ModelId::new("model"), content)
    }

    #[test]
    fn tool_calls_come_out_in_the_order_they_were_asked() {
        let response = response_with(vec![
            ContentPart::Text {
                text: "working".to_owned(),
            },
            ContentPart::ToolCall(ToolCall::function("call_1", "read", serde_json::json!({}))),
            ContentPart::ToolCall(ToolCall::function("call_2", "shell", serde_json::json!({}))),
        ]);

        let calls = tool_calls_of(&response);

        assert_eq!(
            calls
                .iter()
                .map(|call| call.id.as_str())
                .collect::<Vec<_>>(),
            ["call_1", "call_2"]
        );
    }

    #[test]
    fn only_provider_native_parts_are_kept_for_replay() {
        let response = response_with(vec![
            ContentPart::Reasoning(ReasoningContent {
                text:             "thinking".to_owned(),
                signature:        None,
                signature_origin: None,
                redacted:         false,
            }),
            ContentPart::Text {
                text: "answer".to_owned(),
            },
            ContentPart::Opaque {
                kind: "openai.reasoning".to_owned(),
                data: serde_json::json!({"id": "rs_1"}),
            },
            ContentPart::ToolCall(ToolCall::function("call_1", "read", serde_json::json!({}))),
        ]);

        let parts = provider_parts_of(&response);

        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], ContentPart::Reasoning(_)));
        assert!(matches!(parts[1], ContentPart::Opaque { .. }));
    }

    #[test]
    fn a_successful_skill_call_counts_as_an_activation() {
        let call = ToolCall::function(
            "call_1",
            NativeTool::UseSkill.canonical_name(),
            serde_json::json!({}),
        );
        let ok = ToolResult {
            tool_call_id: "call_1".to_owned(),
            name:         None,
            content:      vec![ContentPart::Text {
                text: "loaded".to_owned(),
            }],
            is_error:     false,
        };
        let failed = ToolResult {
            is_error: true,
            ..ok.clone()
        };

        assert!(activated_a_skill(slice::from_ref(&call), &[ok]));
        assert!(!activated_a_skill(&[call], &[failed]));
    }
}
