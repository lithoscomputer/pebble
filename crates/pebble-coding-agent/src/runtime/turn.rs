//! The coding layer that projects Pebble's durable behavior onto `Agent`.

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
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::control::{actor_from_attribution, input_message, input_source_from_attribution};
use super::retry::RetryEventBridge;
use super::{CodingRuntime, PromptTotals, StateMachine};
use crate::coding_agent::CodingInput;
use crate::compaction::{
    CompactionControl, CompactionReason, CompactionRequest, check_context_usage, compact_context,
};
use crate::config::CodingAgentOptions;
use crate::context_window::{
    ContextWindowInput, build_local_snapshot, context_window_from_response_usage,
};
use crate::error::{Error, ErrorData, InterruptReason, Result};
use crate::event::Emitter;
use crate::file_tracker::FileTracker;
use crate::history::History;
use crate::loop_detection::detect_loop;
use crate::profile::ModelFacts;
use crate::reasoning::ReasoningOutput;
use crate::skills::{ExpandedInput, Skill, SkillExpansion, expand_skill};
use crate::subagent::SubagentSupervisor;
use crate::task_reminder::maybe_task_reminder;
use crate::tool::{CodingToolService, NativeTool, ToolRegistry, canonical_tool_name};
use crate::types::{
    CodingAgentState, CodingEvent, ContextWindowSnapshot, ContextWindowStaleness, CostSource,
    InputContent, InputSource, LlmOutputKind, LlmRetryPhase, Message, SkillActivationSource,
    TokenUsage,
};

/// How many failed response streams Pebble replays after the first attempt.
const STREAM_CONSUME_RETRIES: u32 = 3;

/// What the model is told when it keeps making the same calls.
const LOOP_WARNING: &str = "WARNING: Loop detected. You appear to be repeating the same tool \
                            calls. Please try a different approach or ask for clarification.";

#[derive(Clone)]
pub(super) struct CodingAgentBridge {
    state:          Arc<Mutex<ConversationState>>,
    client:         lithos_llm::Client,
    model_selector: String,
    model:          String,
    provider:       String,
    system_prompt:  String,
    facts:          ModelFacts,
    config:         Arc<CodingAgentOptions>,
    registry:       Arc<ToolRegistry>,
    tools:          Arc<CodingToolService>,
    emitter:        Emitter,
    session_id:     String,
    memory_tokens:  u64,
    skills_tokens:  u64,
    /// The session's state, moved to `Executing` for the length of a tool
    /// round and back to `Thinking` after it.
    state_machine:  StateMachine,
    /// The token that ends the prompt in progress. A child of the runtime's
    /// terminal token, so it also fires when the session is shut down, and set
    /// afresh by [`begin_prompt`](Self::begin_prompt) for every prompt.
    prompt_cancel:  Arc<Mutex<CancellationToken>>,
    compaction:     CompactionControl,
    subagents:      Option<SubagentSupervisor>,
    skills:         Vec<Skill>,
}

/// The conversation and what one prompt accumulates around it.
///
/// One copy, shared by the runtime and the bridge: the runtime reads it for
/// records, exports, and the public history, and the bridge writes it as the
/// loop commits turns. The first four members outlive a prompt; the rest are
/// reset by [`CodingAgentBridge::begin_prompt`].
pub(super) struct ConversationState {
    pub(super) history: History,
    pub(super) file_tracker: FileTracker,
    pub(super) totals: PromptTotals,
    pub(super) activated_skill_context_observed: bool,
    pub(super) context_window: Option<ContextWindowSnapshot>,
    compaction_failed: bool,
    pending_task_reminder: Option<Message>,
    local_context_window: Option<ContextWindowSnapshot>,
    inference_start: Option<Instant>,
    tool_start: Option<Instant>,
    boundary_error: Option<Error>,
}

impl ConversationState {
    /// A conversation that starts from `history`, with nothing accumulated.
    pub(super) fn new(history: History) -> Self {
        Self {
            history,
            file_tracker: FileTracker::default(),
            totals: PromptTotals::default(),
            activated_skill_context_observed: false,
            context_window: None,
            compaction_failed: false,
            pending_task_reminder: None,
            local_context_window: None,
            inference_start: None,
            tool_start: None,
            boundary_error: None,
        }
    }
}

impl CodingAgentBridge {
    fn from_runtime(runtime: &CodingRuntime) -> Self {
        let registry = Arc::new(runtime.registry.clone());
        let config = Arc::new(runtime.config.clone());
        let mut tools = CodingToolService::new(
            Arc::clone(&registry),
            Arc::clone(&runtime.env),
            Arc::clone(&config),
            runtime.emitter.clone(),
            runtime.id.clone(),
            runtime.root_session_id.clone(),
            Arc::clone(&runtime.redactor),
        );
        if let Some(provider) = runtime.tool_env_provider.as_ref() {
            tools = tools.with_tool_env_provider(Arc::clone(provider));
        }
        if let Some(provider) = runtime.human_input.as_ref() {
            tools = tools.with_human_input(Arc::clone(provider));
        }
        Self {
            state: Arc::clone(&runtime.conversation),
            client: runtime.client.clone(),
            model_selector: runtime.model_selector.clone(),
            model: runtime.model.clone(),
            provider: runtime.provider.clone(),
            system_prompt: runtime.system_prompt.clone(),
            facts: runtime.facts,
            config,
            registry,
            tools: Arc::new(tools),
            emitter: runtime.emitter.clone(),
            session_id: runtime.id.clone(),
            memory_tokens: runtime.memory_tokens,
            skills_tokens: runtime.skills_tokens,
            state_machine: runtime.state.clone(),
            prompt_cancel: Arc::new(Mutex::new(runtime.cancel_token.clone())),
            compaction: runtime.compaction.clone(),
            subagents: runtime.subagents.clone(),
            skills: runtime.skills.clone(),
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
        state.tool_start = None;
        state.boundary_error = None;
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

    /// Keeps a typed coding-layer failure while the generic loop receives its
    /// boundary representation.
    fn record_boundary_error(&self, error: Error) -> agent::LifecycleError {
        let message = ErrorData::from(&error).message;
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .boundary_error = Some(error);
        agent::LifecycleError::new(message)
    }

    /// Waits for the tree stream at a stable turn boundary.
    async fn flush_events(&self) -> StdResult<(), agent::LifecycleError> {
        self.emitter
            .flush()
            .await
            .map(|_| ())
            .map_err(|failure| self.record_boundary_error(failure.into_runtime_error()))
    }

    fn emit(&self, event: CodingEvent) {
        self.emitter.emit(self.session_id.clone(), event);
    }

    pub(super) fn finish_inference(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(started) = state.inference_start.take() {
            state.totals.timing.inference = state
                .totals
                .timing
                .inference
                .saturating_add(started.elapsed());
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
        let prompt_cancel = self.prompt_cancel();
        let operation = self.compaction.begin(&prompt_cancel);
        let request = CompactionRequest {
            model: &self.model_selector,
            facts: self.facts,
            preserve_turns: self.config.compaction_preserve_turns,
            estimate,
            reason: CompactionReason::Threshold,
            instructions: None,
            cancel: operation.token(),
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
            if !matches!(error, Error::Interrupted(InterruptReason::Cancelled)) {
                self.emit(CodingEvent::Error {
                    error: ErrorData::from(&error),
                });
            }
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

    fn stage_task_reminder(&self, tools: &agent::ToolCatalog) {
        let names = tools
            .visible_tools()
            .map(|tool| tool.definition().name.as_str())
            .collect::<Vec<_>>();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.pending_task_reminder =
            maybe_task_reminder(&state.history, &names).map(|content| Message::System {
                content,
                timestamp: SystemTime::now(),
            });
    }

    fn conversation_update(&self, include_reminder: bool) -> agent::ConversationUpdate {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let mut messages = state.history.to_llm_messages();
        if include_reminder && let Some(reminder) = &state.pending_task_reminder {
            messages.push(reminder.to_llm_message());
        }
        agent::ConversationUpdate::replace(messages)
    }

    fn commit_user_message(&self, message: &LlmMessage, attribution: Option<&Value>) {
        let content = InputContent::from(message.content());
        let text = content.text_content().to_owned();
        let source = input_source_from_attribution(attribution).unwrap_or(InputSource::Agent);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.compaction_failed = false;
        state.history.push(Message::User {
            content:   content.clone(),
            timestamp: SystemTime::now(),
        });
        drop(state);
        self.emit(CodingEvent::UserInput {
            text,
            content: content.event_content(),
            source,
        });
    }

    fn commit_steering(&self, message: &LlmMessage, attribution: Option<&Value>) {
        let content = InputContent::from(message.content());
        let text = content.text_content().to_owned();
        let actor = actor_from_attribution(attribution);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.history.push(Message::Steering {
            content:   content.clone(),
            timestamp: SystemTime::now(),
        });
        drop(state);
        self.emit(CodingEvent::SteeringInjected {
            text,
            content: content.event_content(),
            actor,
        });
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
        if let Some(context_window) = &context_window {
            let mut stored = context_window.clone();
            stored.staleness = ContextWindowStaleness::Stored;
            state.context_window = Some(stored);
        }

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

    fn commit_tool_results(&self, calls: &[ToolCall], results: &[ToolResult], cancelled: bool) {
        self.state_machine.transition(CodingAgentState::Thinking);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(started) = state.tool_start.take() {
            state.totals.timing.tool = state.totals.timing.tool.saturating_add(started.elapsed());
        }
        if activated_a_skill(calls, results) {
            state.activated_skill_context_observed = true;
        }
        state.file_tracker.record_from_tool_calls(calls, results);
        state.history.push(Message::ToolResults {
            results:   results.to_vec(),
            timestamp: SystemTime::now(),
        });

        if cancelled || self.prompt_cancel().is_cancelled() {
            return;
        }
        let loop_detected = self.config.enable_loop_detection
            && detect_loop(&state.history, self.config.loop_detection_window);
        if loop_detected {
            state.history.push(Message::Steering {
                content:   LOOP_WARNING.into(),
                timestamp: SystemTime::now(),
            });
        }
        drop(state);
        if loop_detected {
            self.emit(CodingEvent::LoopDetected);
        }
    }

    fn measure_request(&self, request: &Request) -> ContextWindowSnapshot {
        let tools = self.registry.sources_for(request.tools());
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
            agent::AgentEvent::ToolStarted { call } => {
                self.state_machine.transition(CodingAgentState::Executing);
                self.state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .tool_start
                    .get_or_insert_with(Instant::now);
                self.tools.emit_started(call);
            }
            agent::AgentEvent::ToolCompleted {
                result,
                error_kind,
                output_stats,
            } => {
                self.tools.emit_result(result, *output_stats, *error_kind);
            }
            agent::AgentEvent::TurnInterrupted { generation } => {
                self.finish_inference();
                self.state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .pending_task_reminder = None;
                self.emit(CodingEvent::RoundInterrupted {
                    generation: *generation,
                });
            }
            _ => {}
        }
    }
}

#[async_trait]
impl agent::AgentLifecycle for CodingAgentBridge {
    async fn before_model(
        &self,
        _context: agent::TurnContext<'_>,
        cancel: &CancellationToken,
    ) -> StdResult<agent::ConversationUpdate, agent::LifecycleError> {
        if cancel.is_cancelled() || self.prompt_cancel().is_cancelled() {
            return Ok(agent::ConversationUpdate::unchanged());
        }

        // Commit the input and any steering before work on the next request.
        self.flush_events().await?;
        self.compact_once_if_needed().await;
        let update = self.conversation_update(false);
        // Compaction can publish its own result or failure.
        self.flush_events().await?;
        Ok(update)
    }

    async fn after_tool_discovery(
        &self,
        _context: agent::TurnContext<'_>,
        tools: &agent::ToolCatalog,
        cancel: &CancellationToken,
    ) -> StdResult<agent::ConversationUpdate, agent::LifecycleError> {
        if cancel.is_cancelled() || self.prompt_cancel().is_cancelled() {
            return Ok(agent::ConversationUpdate::unchanged());
        }
        self.stage_task_reminder(tools);
        Ok(self.conversation_update(true))
    }

    async fn after_model(
        &self,
        _context: agent::TurnContext<'_>,
        _response: &Response,
        _cancel: &CancellationToken,
    ) -> StdResult<agent::ConversationUpdate, agent::LifecycleError> {
        // Do not run tools until the response that requested them is durable.
        self.flush_events().await?;
        self.compact_once_if_needed().await;
        let update = self.conversation_update(false);
        self.flush_events().await?;
        Ok(update)
    }

    async fn prepare_follow_up(
        &self,
        _context: agent::TurnContext<'_>,
        message: agent::UserMessage,
        _cancel: &CancellationToken,
    ) -> StdResult<agent::UserMessage, agent::LifecycleError> {
        let attribution = message.attribution().cloned();
        let content = InputContent::from(message.content());
        let text = content.text_content();
        let expanded = if self.skills.is_empty() {
            ExpandedInput {
                text:       text.to_owned(),
                skill_name: None,
            }
        } else {
            expand_skill(&self.skills, text)
                .map_err(|source| self.record_boundary_error(Error::SkillExpansion(source)))?
        };
        if let Some(name) = &expanded.skill_name {
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .activated_skill_context_observed = true;
            self.emit(CodingEvent::SkillActivated {
                skill_name: name.clone(),
                source:     SkillActivationSource::Slash,
            });
        }
        let content = content_after_skill_expansion(content, &expanded);
        let message = agent::UserMessage::new(content.into_parts());
        Ok(match attribution {
            Some(attribution) => message.with_attribution(attribution),
            None => message,
        })
    }

    async fn after_answer(
        &self,
        _context: agent::TurnContext<'_>,
        _response: &Response,
        cancel: &CancellationToken,
    ) -> StdResult<agent::AfterAnswerAction, agent::LifecycleError> {
        let Some(supervisor) = self.subagents.as_ref() else {
            return Ok(agent::AfterAnswerAction::Complete);
        };
        match supervisor.next_parent_notification_turn(cancel).await {
            Ok(Some(turn)) => Ok(agent::AfterAnswerAction::ContinueWith(turn.into())),
            Ok(None) => Ok(agent::AfterAnswerAction::Complete),
            Err(error) => Err(self.record_boundary_error(error)),
        }
    }
}

impl agent::ConversationProjection for CodingAgentBridge {
    fn user_message_committed(&self, message: &LlmMessage, attribution: Option<&Value>) {
        self.commit_user_message(message, attribution);
    }

    fn steering_message_committed(&self, message: &LlmMessage, attribution: Option<&Value>) {
        self.commit_steering(message, attribution);
    }

    fn assistant_message_committed(&self, response: &Response) {
        self.commit_assistant(response);
    }

    fn tool_results_committed(&self, calls: &[ToolCall], results: &[ToolResult], cancelled: bool) {
        self.commit_tool_results(calls, results, cancelled);
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
        input: CodingInput,
        skill_expansion: SkillExpansion,
        prompt_cancel: &CancellationToken,
    ) -> Result<Option<String>> {
        if self.state.current() == CodingAgentState::Closed {
            return Err(Error::SessionClosed);
        }
        self.check_pump().await?;
        self.state.transition(CodingAgentState::Thinking);

        let source = input.source();
        let content = input.content().clone();
        let expanded = self.expand_input(content.text_content(), skill_expansion)?;
        if let Some(name) = &expanded.skill_name {
            self.conversation().activated_skill_context_observed = true;
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

        let result = agent
            .prompt_with_cancellation(
                input_message(content_after_skill_expansion(content, &expanded), source),
                prompt_cancel,
            )
            .await;
        self.coding_agent = Some(agent);
        bridge.finish_inference();
        // A failed stream cancels the generic loop. Recover its exact error
        // before an abort can be mistaken for caller cancellation.
        self.check_pump().await?;
        if let Some(error) = bridge.take_boundary_error() {
            // The supervisor's wait answers a cancellation with a plain
            // `Cancelled`, whoever cancelled and whatever reason they recorded
            // first. The prompt ends the way an abort at any other checkpoint
            // does: with the recorded reason, and closed when the cancellation
            // was the session's own.
            if matches!(error, Error::Interrupted(InterruptReason::Cancelled)) {
                return Err(self.prompt_aborted().await);
            }
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
                Err(Error::Agent(error))
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
        let mut builder = agent::Agent::builder(model_service, self.model_selector.clone())
            .system_prompt(self.system_prompt.clone())
            .messages(messages)
            .tool_service(bridge.tools.clone())
            .tool_middleware(bridge.tools.clone())
            .control_handle(self.agent_control.clone())
            .lifecycle(bridge.clone())
            .conversation_projection(bridge.clone())
            .event_projection(bridge.clone())
            .config(config);
        for middleware in &self.tool_middleware {
            builder = builder.tool_middleware(Arc::clone(middleware));
        }
        let agent = builder.build().map_err(Error::AgentBuild)?;

        self.coding_bridge = Some(bridge);
        self.coding_agent = Some(agent);
        Ok(())
    }

    /// Expands a `/name` reference in the input, where one is allowed.
    fn expand_input(&self, input: &str, expansion: SkillExpansion) -> Result<ExpandedInput> {
        if self.skills.is_empty() || expansion == SkillExpansion::Skip {
            return Ok(ExpandedInput {
                text:       input.to_owned(),
                skill_name: None,
            });
        }
        expand_skill(&self.skills, input).map_err(Error::SkillExpansion)
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

/// Keeps attachments unchanged when a slash command replaces readable text.
fn content_after_skill_expansion(content: InputContent, expanded: &ExpandedInput) -> InputContent {
    if expanded.skill_name.is_none() {
        return content;
    }

    let mut parts = vec![ContentPart::Text {
        text: expanded.text.clone(),
    }];
    parts.extend(
        content
            .into_parts()
            .into_iter()
            .filter(|part| !matches!(part, ContentPart::Text { .. })),
    );
    InputContent::new(parts)
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
