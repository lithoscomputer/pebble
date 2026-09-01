//! The agent facade and turn loop.

use std::collections::HashSet;
use std::result::Result as StdResult;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use lithos_llm::middleware::{CallContext, RetryPolicy};
use lithos_llm::types::{
    ContentPart, Error as LlmError, Message, ReasoningEffort, Request, Response, Role, Speed,
    ToolCall, ToolChoice, ToolResult,
};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::advanced::{StreamObserver, StreamOutcome, stream_response};
use crate::context::{ContextTransform, TransformContext};
use crate::control::{AgentControlHandle, Control};
use crate::error::{AgentBuildError, AgentError, Result};
use crate::event::{AgentEvent, EventHub, EventProjection, FirstOutputKind};
use crate::model::ModelService;
use crate::tool::{
    BeforeToolCall, Tool, ToolAccess, ToolAccessContext, ToolAccessPolicy, ToolCallContext,
    ToolCallHooks, ToolContext, ToolProvider, ToolRoundContext, ToolRoundExecutor,
};
use crate::turn::{TurnBoundaryAction, TurnBoundaryContext, TurnBoundaryHooks, TurnContext};
use crate::validation::validate_tool_arguments;

/// The default number of lifecycle events held for each subscriber.
const DEFAULT_EVENT_CAPACITY: usize = 256;

/// A user message accepted by [`Agent::prompt`], steering, or follow-up.
#[derive(Clone, Debug, PartialEq)]
pub struct UserMessage {
    content: Vec<ContentPart>,
}

impl UserMessage {
    /// Creates a message from provider-neutral content parts.
    #[must_use]
    pub fn new(content: impl IntoIterator<Item = ContentPart>) -> Self {
        Self {
            content: content.into_iter().collect(),
        }
    }

    /// Creates a text message.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::new([ContentPart::Text { text: text.into() }])
    }

    /// The message content.
    #[must_use]
    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    pub(crate) fn into_message(self) -> Message {
        Message::new(Role::User, self.content)
    }
}

impl From<String> for UserMessage {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for UserMessage {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

/// Whether independent tool calls in one model response run together.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ToolExecution {
    /// Run calls in model order.
    Sequential,
    /// Run calls concurrently and commit their results in model order.
    #[default]
    Parallel,
}

/// Turn-loop policy for an [`Agent`].
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// The most tokens one model turn may produce.
    pub max_output_tokens: Option<u32>,
    /// The requested reasoning effort.
    pub reasoning_effort:  Option<ReasoningEffort>,
    /// The requested latency or cost tier.
    pub speed:             Option<Speed>,
    /// Replay policy for a response stream that fails after it opens.
    pub turn_replay:       RetryPolicy,
    /// Hard limit on replays after the first response stream opens.
    pub max_turn_replays:  u32,
    /// Whether independent tool calls run together.
    pub tool_execution:    ToolExecution,
    /// Events held for each live subscriber.
    pub event_capacity:    usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: None,
            reasoning_effort:  None,
            speed:             None,
            turn_replay:       RetryPolicy::exponential().max_attempts(4),
            max_turn_replays:  3,
            tool_execution:    ToolExecution::default(),
            event_capacity:    DEFAULT_EVENT_CAPACITY,
        }
    }
}

/// Collects the required model service and optional agent behavior.
#[must_use = "a builder does nothing until `build` is called"]
pub struct AgentBuilder {
    model_service:     Arc<dyn ModelService>,
    model:             String,
    system_prompt:     String,
    messages:          Vec<Message>,
    tools:             Vec<Tool>,
    tool_provider:     Option<Arc<dyn ToolProvider>>,
    tool_access:       Option<Arc<dyn ToolAccessPolicy>>,
    tool_hooks:        Option<Arc<dyn ToolCallHooks>>,
    tool_round:        Option<Arc<dyn ToolRoundExecutor>>,
    context_transform: Option<Arc<dyn ContextTransform>>,
    turn_hooks:        Option<Arc<dyn TurnBoundaryHooks>>,
    event_projection:  Option<Arc<dyn EventProjection>>,
    config:            AgentConfig,
}

impl AgentBuilder {
    fn new(service: impl ModelService + 'static, model: impl Into<String>) -> Self {
        Self {
            model_service:     Arc::new(service),
            model:             model.into(),
            system_prompt:     String::new(),
            messages:          Vec::new(),
            tools:             Vec::new(),
            tool_provider:     None,
            tool_access:       None,
            tool_hooks:        None,
            tool_round:        None,
            context_transform: None,
            turn_hooks:        None,
            event_projection:  None,
            config:            AgentConfig::default(),
        }
    }

    /// Sets the system prompt sent before conversation history.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Starts with existing provider-neutral conversation history.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Adds model-visible tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Adds tools resolved from conversation state before each model turn.
    ///
    /// Resolved tools are combined with tools added through
    /// [`tools`](Self::tools).
    pub fn tool_provider(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        self.tool_provider = Some(provider);
        self
    }

    /// Sets the access policy evaluated for every resolved tool in every turn.
    pub fn tool_access_policy(mut self, policy: Arc<dyn ToolAccessPolicy>) -> Self {
        self.tool_access = Some(policy);
        self
    }

    /// Sets the hooks called before and after every tool call.
    pub fn tool_call_hooks(mut self, hooks: Arc<dyn ToolCallHooks>) -> Self {
        self.tool_hooks = Some(hooks);
        self
    }

    /// Replaces the default generic executor for complete tool rounds.
    pub fn tool_round_executor(mut self, executor: Arc<dyn ToolRoundExecutor>) -> Self {
        self.tool_round = Some(executor);
        self
    }

    /// Sets the transformation run before each model request.
    pub fn context_transform(mut self, transform: Arc<dyn ContextTransform>) -> Self {
        self.context_transform = Some(transform);
        self
    }

    /// Sets hooks at model-turn and natural-answer boundaries.
    pub fn turn_boundary_hooks(mut self, hooks: Arc<dyn TurnBoundaryHooks>) -> Self {
        self.turn_hooks = Some(hooks);
        self
    }

    /// Projects lifecycle events into an embedding layer's event model.
    pub fn event_projection(mut self, projection: Arc<dyn EventProjection>) -> Self {
        self.event_projection = Some(projection);
        self
    }

    /// Replaces the turn-loop policy.
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the agent.
    ///
    /// # Errors
    ///
    /// Returns an error for a blank model, a zero event capacity, or duplicate
    /// tool names.
    pub fn build(self) -> StdResult<Agent, AgentBuildError> {
        if self.model.trim().is_empty() {
            return Err(AgentBuildError::EmptyModel);
        }
        if self.config.event_capacity == 0 {
            return Err(AgentBuildError::ZeroEventCapacity);
        }
        let mut names = HashSet::new();
        for tool in &self.tools {
            let name = tool.definition().name.clone();
            if !names.insert(name.clone()) {
                return Err(AgentBuildError::DuplicateTool { name });
            }
        }

        let events = EventHub::new(self.config.event_capacity, self.event_projection);
        Ok(Agent {
            model_service: self.model_service,
            model: self.model,
            system_prompt: self.system_prompt,
            messages: self.messages,
            tools: self.tools,
            tool_provider: self.tool_provider,
            tool_access: self.tool_access,
            tool_hooks: self.tool_hooks,
            tool_round: self.tool_round,
            context_transform: self.context_transform,
            turn_hooks: self.turn_hooks,
            config: self.config,
            events,
            control: Control::new(),
        })
    }
}

/// What an agent is doing now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AgentState {
    /// Waiting for input.
    Idle,
    /// Processing a prompt.
    Running,
    /// Shut down.
    Closed,
}

/// An owned, immutable view of an agent's current state.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentSnapshot {
    state:    AgentState,
    model:    String,
    messages: Vec<Message>,
}

impl AgentSnapshot {
    /// The lifecycle state captured by this snapshot.
    #[must_use]
    pub const fn state(&self) -> AgentState {
        self.state
    }

    /// The configured model selector.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The conversation history captured by this snapshot.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}

/// The final result of a prompt and any queued follow-up input.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptOutcome {
    response:        Response,
    turn_count:      usize,
    tool_call_count: usize,
}

impl PromptOutcome {
    /// The final complete model response.
    #[must_use]
    pub const fn response(&self) -> &Response {
        &self.response
    }

    /// Consumes the outcome and returns the final response.
    #[must_use]
    pub fn into_response(self) -> Response {
        self.response
    }

    /// The readable text in the final response.
    #[must_use]
    pub fn text(&self) -> String {
        self.response.text()
    }

    /// The number of model turns completed for this prompt.
    #[must_use]
    pub const fn turn_count(&self) -> usize {
        self.turn_count
    }

    /// The total number of tool calls executed for this prompt.
    #[must_use]
    pub const fn tool_call_count(&self) -> usize {
        self.tool_call_count
    }
}

/// One provider-neutral active conversation.
pub struct Agent {
    model_service:     Arc<dyn ModelService>,
    model:             String,
    system_prompt:     String,
    messages:          Vec<Message>,
    tools:             Vec<Tool>,
    tool_provider:     Option<Arc<dyn ToolProvider>>,
    tool_access:       Option<Arc<dyn ToolAccessPolicy>>,
    tool_hooks:        Option<Arc<dyn ToolCallHooks>>,
    tool_round:        Option<Arc<dyn ToolRoundExecutor>>,
    context_transform: Option<Arc<dyn ContextTransform>>,
    turn_hooks:        Option<Arc<dyn TurnBoundaryHooks>>,
    config:            AgentConfig,
    events:            EventHub,
    control:           Arc<Control>,
}

impl Agent {
    /// Starts a builder with an injected model service and model selector.
    pub fn builder(service: impl ModelService + 'static, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(service, model)
    }

    /// Subscribes to lifecycle events.
    ///
    /// The stream is bounded and lossy for a receiver that falls behind.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Returns a handle that can steer, follow up, or abort an active prompt.
    #[must_use]
    pub fn control_handle(&self) -> AgentControlHandle {
        AgentControlHandle::new(Arc::clone(&self.control))
    }

    /// Queues steering for the next turn and interrupts the current turn.
    pub fn steer(&self, message: impl Into<UserMessage>) -> bool {
        self.control_handle().steer(message)
    }

    /// Queues input to process after the current answer.
    pub fn follow_up(&self, message: impl Into<UserMessage>) -> bool {
        self.control_handle().follow_up(message)
    }

    /// Aborts the active prompt.
    pub fn abort(&self) -> bool {
        self.control_handle().abort()
    }

    /// Interrupts the current turn and waits for steering before another.
    pub fn interrupt(&self) -> bool {
        self.control_handle().interrupt()
    }

    /// Claims the next turn boundary for steering without interrupting now.
    pub fn park_for_steer(&self) -> bool {
        self.control_handle().park_for_steer()
    }

    /// Waits until no prompt is running.
    pub async fn wait_for_idle(&self) {
        self.control_handle().wait_for_idle().await;
    }

    /// The current lifecycle state.
    #[must_use]
    pub fn state(&self) -> AgentState {
        if self.control.is_closed() {
            AgentState::Closed
        } else if self.control.is_running() {
            AgentState::Running
        } else {
            AgentState::Idle
        }
    }

    /// The current conversation history.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Captures an owned view of the current state and history.
    #[must_use]
    pub fn snapshot(&self) -> AgentSnapshot {
        AgentSnapshot {
            state:    self.state(),
            model:    self.model.clone(),
            messages: self.messages.clone(),
        }
    }

    /// Clears conversation history and queued input while the agent is idle.
    ///
    /// Returns `false` while a prompt is active or after shutdown.
    pub fn reset(&mut self) -> bool {
        if self.state() != AgentState::Idle {
            return false;
        }
        self.messages.clear();
        self.control.clear_queues();
        true
    }

    /// Processes one input and every queued follow-up to completion.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is empty, the agent is closed, the
    /// prompt is aborted, context preparation fails, or the model call
    /// fails.
    pub async fn prompt(&mut self, message: impl Into<UserMessage>) -> Result<PromptOutcome> {
        self.prompt_inner(message.into(), None).await
    }

    /// Processes one input until it completes or `cancel` is cancelled.
    ///
    /// The external signal is combined with cancellation from this agent's
    /// control handle. A cancellation that is already set still commits the
    /// input before the prompt aborts.
    pub async fn prompt_with_cancellation(
        &mut self,
        message: impl Into<UserMessage>,
        cancel: &CancellationToken,
    ) -> Result<PromptOutcome> {
        self.prompt_inner(message.into(), Some(cancel)).await
    }

    async fn prompt_inner(
        &mut self,
        message: UserMessage,
        parent_cancel: Option<&CancellationToken>,
    ) -> Result<PromptOutcome> {
        if message.content.is_empty() {
            return Err(AgentError::EmptyInput);
        }
        let Some(prompt_cancel) = self.control.begin_prompt(parent_cancel) else {
            return Err(AgentError::Closed);
        };

        self.emit(AgentEvent::PromptStarted);
        let result = self
            .process_prompt(message.into_message(), &prompt_cancel)
            .await;
        match &result {
            Ok(outcome) => self.emit(AgentEvent::PromptCompleted {
                response: outcome.response.clone(),
            }),
            Err(AgentError::Aborted) => self.emit(AgentEvent::PromptAborted),
            Err(_) => {}
        }
        self.control.finish_prompt();
        result
    }

    /// Closes an idle agent.
    ///
    /// Returns whether this call performed the transition.
    pub fn shutdown(&mut self) -> bool {
        if !self.control.close() {
            return false;
        }
        self.emit(AgentEvent::AgentClosed);
        true
    }

    async fn process_prompt(
        &mut self,
        first_message: Message,
        prompt_cancel: &CancellationToken,
    ) -> Result<PromptOutcome> {
        let mut next_message = first_message;
        let mut turn_count = 0;
        let mut tool_call_count = 0;

        loop {
            self.messages.push(next_message.clone());
            self.emit(AgentEvent::UserMessage {
                message: next_message,
            });

            let final_response = loop {
                if prompt_cancel.is_cancelled() {
                    return Err(AgentError::Aborted);
                }

                if !self.control.wait_until_resumed(prompt_cancel).await {
                    return Err(AgentError::Aborted);
                }

                let (round_cancel, steering) = self.control.begin_round();
                for steering in steering {
                    self.messages.push(steering.clone());
                    self.emit(AgentEvent::SteeringMessage { message: steering });
                }
                if let Some(hooks) = self.turn_hooks.clone() {
                    let context =
                        TurnBoundaryContext::new(&self.model, turn_count, &mut self.messages);
                    let prepared = hooks.before_model(context, &round_cancel).await;
                    if prompt_cancel.is_cancelled() {
                        return Err(AgentError::Aborted);
                    }
                    if round_cancel.is_cancelled() {
                        self.emit(AgentEvent::TurnInterrupted);
                        continue;
                    }
                    prepared.map_err(|source| AgentError::TurnBoundary { source })?;
                }
                if let Some(transform) = &self.context_transform {
                    let context = TransformContext::new(&self.model, &mut self.messages);
                    let transformed = transform.transform(context, &round_cancel).await;
                    if prompt_cancel.is_cancelled() {
                        return Err(AgentError::Aborted);
                    }
                    if round_cancel.is_cancelled() {
                        self.emit(AgentEvent::TurnInterrupted);
                        continue;
                    }
                    transformed.map_err(|source| AgentError::ContextTransform { source })?;
                }

                let tools = self.resolve_tools(turn_count)?;
                self.emit(AgentEvent::TurnStarted { turn: turn_count });
                let request = self.build_request(&tools)?;
                self.emit(AgentEvent::ModelRequestStarted {
                    model:   self.model.clone(),
                    request: request.clone(),
                });

                let response = match self
                    .stream_response(request, prompt_cancel, &round_cancel)
                    .await?
                {
                    StreamResult::Completed(response) => *response,
                    StreamResult::Interrupted => {
                        self.emit(AgentEvent::TurnInterrupted);
                        continue;
                    }
                };
                let turn = turn_count;
                turn_count += 1;

                let assistant = response_message(&response);
                self.messages.push(assistant);
                self.emit(AgentEvent::AssistantMessage {
                    response: response.clone(),
                });

                if let Some(hooks) = self.turn_hooks.clone() {
                    let context = TurnBoundaryContext::new(&self.model, turn, &mut self.messages);
                    hooks
                        .after_model(context, &response, &round_cancel)
                        .await
                        .map_err(|source| AgentError::TurnBoundary { source })?;
                    if prompt_cancel.is_cancelled() {
                        return Err(AgentError::Aborted);
                    }
                }

                let calls = tool_calls(&response);
                if calls.is_empty() {
                    if round_cancel.is_cancelled() || self.control.is_paused() {
                        if round_cancel.is_cancelled() {
                            self.emit(AgentEvent::TurnInterrupted);
                        }
                        continue;
                    }
                    if let Some(hooks) = self.turn_hooks.clone() {
                        let context = TurnContext::new(&self.model, turn, &self.messages);
                        let boundary_action = hooks
                            .after_answer(context, &response, prompt_cancel)
                            .await
                            .map_err(|source| AgentError::TurnBoundary { source })?;
                        if prompt_cancel.is_cancelled() {
                            return Err(AgentError::Aborted);
                        }
                        match boundary_action {
                            TurnBoundaryAction::Complete => {}
                            TurnBoundaryAction::Continue => continue,
                            TurnBoundaryAction::ContinueWith(message) => {
                                let message = message.into_message();
                                self.messages.push(message.clone());
                                self.emit(AgentEvent::UserMessage { message });
                                continue;
                            }
                        }
                        if round_cancel.is_cancelled() {
                            self.emit(AgentEvent::TurnInterrupted);
                            continue;
                        }
                    }
                    break response;
                }
                tool_call_count += calls.len();

                let results = self
                    .execute_tools(turn, &calls, &tools, prompt_cancel, &round_cancel)
                    .await;
                self.messages.push(tool_results_message(&results));

                if prompt_cancel.is_cancelled() {
                    return Err(AgentError::Aborted);
                }
                if round_cancel.is_cancelled() {
                    self.emit(AgentEvent::TurnInterrupted);
                }
            };

            if let Some(follow_up) = self.control.pop_follow_up() {
                next_message = follow_up;
                continue;
            }

            return Ok(PromptOutcome {
                response: final_response,
                turn_count,
                tool_call_count,
            });
        }
    }

    fn resolve_tools(&self, turn: usize) -> Result<Vec<ResolvedTool>> {
        let context = TurnContext::new(&self.model, turn, &self.messages);
        let mut tools = self.tools.clone();
        if let Some(provider) = &self.tool_provider {
            tools.extend(provider.tools_for_turn(context));
        }

        let mut names = HashSet::new();
        let mut resolved = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = tool.definition().name.clone();
            if !names.insert(name.clone()) {
                return Err(AgentError::DuplicateTool { name });
            }
            let access = self
                .tool_access
                .as_ref()
                .map_or_else(ToolAccess::default, |policy| {
                    policy.access(ToolAccessContext::new(context, tool.definition()))
                });
            resolved.push(ResolvedTool { tool, access });
        }
        Ok(resolved)
    }

    fn build_request(&self, tools: &[ResolvedTool]) -> Result<Request> {
        let mut builder = Request::builder().model(self.model.clone());
        if !self.system_prompt.trim().is_empty() {
            builder = builder.system(self.system_prompt.clone());
        }
        for message in &self.messages {
            builder = builder.message(message.clone());
        }
        for tool in tools.iter().filter(|tool| tool.is_allowed()) {
            builder = builder.tool(tool.tool.definition().clone());
        }
        if tools.iter().any(ResolvedTool::is_allowed) {
            builder = builder.tool_choice(ToolChoice::Auto);
        }
        if let Some(tokens) = self.config.max_output_tokens {
            builder = builder.max_output_tokens(tokens);
        }
        if let Some(effort) = self.config.reasoning_effort {
            builder = builder.reasoning_effort(effort);
        }
        if let Some(speed) = self.config.speed {
            builder = builder.speed(speed);
        }
        builder
            .build()
            .map_err(|source| AgentError::Request { source })
    }

    async fn stream_response(
        &self,
        request: Request,
        prompt_cancel: &CancellationToken,
        round_cancel: &CancellationToken,
    ) -> Result<StreamResult> {
        let observer = AgentStreamObserver {
            events: &self.events,
        };
        match stream_response(
            self.model_service.as_ref(),
            request,
            prompt_cancel,
            round_cancel,
            self.config.turn_replay,
            self.config.max_turn_replays,
            CallContext::new,
            &observer,
        )
        .await
        {
            StreamOutcome::Completed(response) => Ok(StreamResult::Completed(response)),
            StreamOutcome::Aborted => Err(AgentError::Aborted),
            StreamOutcome::Interrupted => Ok(StreamResult::Interrupted),
            StreamOutcome::Failed(source) => Err(AgentError::Model { source: *source }),
        }
    }

    async fn execute_tools(
        &self,
        turn: usize,
        calls: &[ToolCall],
        tools: &[ResolvedTool],
        prompt_cancel: &CancellationToken,
        round_cancel: &CancellationToken,
    ) -> Vec<ToolResult> {
        let cancel = CancellationToken::new();
        let advertised = tools
            .iter()
            .filter(|tool| tool.is_allowed())
            .map(|tool| tool.tool.definition().clone())
            .collect::<Vec<_>>();
        let running = async {
            if let Some(executor) = &self.tool_round {
                return executor
                    .execute_round(ToolRoundContext::new(turn, calls, &advertised), &cancel)
                    .await;
            }
            match self.config.tool_execution {
                ToolExecution::Sequential => {
                    let mut results = Vec::with_capacity(calls.len());
                    for call in calls {
                        results.push(self.execute_tool(turn, call, tools, &cancel).await);
                    }
                    results
                }
                ToolExecution::Parallel => {
                    join_all(
                        calls
                            .iter()
                            .map(|call| self.execute_tool(turn, call, tools, &cancel)),
                    )
                    .await
                }
            }
        };
        tokio::pin!(running);

        let mut prompt_cancelled = false;
        let mut round_cancelled = false;
        loop {
            tokio::select! {
                biased;
                () = prompt_cancel.cancelled(), if !prompt_cancelled => {
                    prompt_cancelled = true;
                    cancel.cancel();
                }
                () = round_cancel.cancelled(), if !round_cancelled => {
                    round_cancelled = true;
                    cancel.cancel();
                }
                results = &mut running => return normalize_tool_results(calls, results),
            }
        }
    }

    async fn execute_tool(
        &self,
        turn: usize,
        call: &ToolCall,
        tools: &[ResolvedTool],
        cancel: &CancellationToken,
    ) -> ToolResult {
        self.emit(AgentEvent::ToolStarted { call: call.clone() });
        let resolved = tools
            .iter()
            .find(|tool| tool.tool.definition().name == call.name);
        let denied = resolved.and_then(|resolved| match &resolved.access {
            ToolAccess::Allowed => None,
            ToolAccess::Denied { reason } => Some(reason.clone()),
        });
        let (result, call_after_hook) = if let Some(reason) = denied {
            (error_tool_result(call, reason), false)
        } else {
            let hook_context = ToolCallContext::new(turn, call);
            let decision = match &self.tool_hooks {
                Some(hooks) => hooks.before_tool_call(hook_context, cancel).await,
                None => BeforeToolCall::Proceed,
            };
            match decision {
                BeforeToolCall::Block { reason } => (error_tool_result(call, reason), false),
                BeforeToolCall::Proceed => {
                    let result = match resolved {
                        Some(resolved) => {
                            if let Err(error) = validate_tool_arguments(
                                &resolved.tool.definition().kind,
                                &call.arguments,
                            ) {
                                error_tool_result(call, error.to_string())
                            } else {
                                let context = ToolContext::new(
                                    call.id.clone(),
                                    call.name.clone(),
                                    cancel.clone(),
                                    self.events.clone(),
                                );
                                match resolved.tool.execute(context, call.arguments.clone()).await {
                                    Ok(output) => ToolResult {
                                        tool_call_id: call.id.clone(),
                                        name:         Some(call.name.clone()),
                                        content:      nonempty_content(output.into_content()),
                                        is_error:     false,
                                    },
                                    Err(error) => error_tool_result(call, error.to_string()),
                                }
                            }
                        }
                        None => error_tool_result(call, format!("unknown tool `{}`", call.name)),
                    };
                    (result, true)
                }
            }
        };
        self.emit(AgentEvent::ToolCompleted {
            result: result.clone(),
        });
        if call_after_hook && let Some(hooks) = &self.tool_hooks {
            hooks
                .after_tool_call(ToolCallContext::new(turn, call), &result, cancel)
                .await;
        }
        result
    }

    fn emit(&self, event: AgentEvent) {
        self.events.emit(event);
    }
}

struct ResolvedTool {
    tool:   Tool,
    access: ToolAccess,
}

impl ResolvedTool {
    const fn is_allowed(&self) -> bool {
        matches!(&self.access, ToolAccess::Allowed)
    }
}

enum StreamResult {
    Completed(Box<Response>),
    Interrupted,
}

struct AgentStreamObserver<'a> {
    events: &'a EventHub,
}

impl StreamObserver for AgentStreamObserver<'_> {
    fn first_output(&self, kind: FirstOutputKind) {
        self.emit(AgentEvent::FirstOutput { kind });
    }

    fn text_delta(&self, delta: &str) {
        self.emit(AgentEvent::TextDelta {
            delta: delta.to_owned(),
        });
    }

    fn reasoning_delta(&self, delta: &str) {
        self.emit(AgentEvent::ReasoningDelta {
            delta: delta.to_owned(),
        });
    }

    fn output_replaced(&self) {
        self.emit(AgentEvent::OutputReplaced);
    }

    fn replay(&self, failed_attempt: u32, delay: Duration, error: &LlmError) {
        self.emit(AgentEvent::TurnReplay {
            failed_attempt,
            delay_seconds: delay.as_secs_f64(),
            error: error.data(),
        });
    }
}

impl AgentStreamObserver<'_> {
    fn emit(&self, event: AgentEvent) {
        self.events.emit(event);
    }
}

fn response_message(response: &Response) -> Message {
    Message::new(Role::Assistant, nonempty_content(response.content.clone()))
}

fn tool_results_message(results: &[ToolResult]) -> Message {
    let message = Message::new(
        Role::Tool,
        results.iter().cloned().map(ContentPart::ToolResult),
    );
    match results.first() {
        Some(result) => message.with_tool_call_id(result.tool_call_id.clone()),
        None => message,
    }
}

fn nonempty_content(mut content: Vec<ContentPart>) -> Vec<ContentPart> {
    if content.is_empty() {
        content.push(ContentPart::Text {
            text: String::new(),
        });
    }
    content
}

fn error_tool_result(call: &ToolCall, message: String) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        name:         Some(call.name.clone()),
        content:      vec![ContentPart::Text { text: message }],
        is_error:     true,
    }
}

fn normalize_tool_results(calls: &[ToolCall], results: Vec<ToolResult>) -> Vec<ToolResult> {
    let mut results = results.into_iter();
    calls
        .iter()
        .map(|call| match results.next() {
            Some(result) if result.tool_call_id == call.id => result,
            Some(_) => error_tool_result(
                call,
                "the tool-round executor returned a result for a different call".to_owned(),
            ),
            None => error_tool_result(
                call,
                "the tool-round executor returned no result for this call".to_owned(),
            ),
        })
        .collect()
}

fn tool_calls(response: &Response) -> Vec<ToolCall> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures_util::stream;
    use lithos_llm::catalog::{ModelId, ProviderId};
    use lithos_llm::middleware::CallContext;
    use lithos_llm::types::{ResponseStream, StreamEvent};
    use serde_json::json;

    use super::*;

    struct ScriptedModel {
        responses: Mutex<VecDeque<Response>>,
        requests:  Option<Arc<Mutex<Vec<Request>>>>,
    }

    impl ScriptedModel {
        fn new(responses: impl IntoIterator<Item = Response>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().collect()),
                requests:  None,
            }
        }

        fn recording(
            responses: impl IntoIterator<Item = Response>,
        ) -> (Self, Arc<Mutex<Vec<Request>>>) {
            let requests = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    responses: Mutex::new(responses.into_iter().collect()),
                    requests:  Some(Arc::clone(&requests)),
                },
                requests,
            )
        }
    }

    #[async_trait]
    impl ModelService for ScriptedModel {
        async fn stream(
            &self,
            request: Request,
            _context: CallContext,
        ) -> StdResult<ResponseStream, LlmError> {
            if let Some(requests) = &self.requests {
                requests
                    .lock()
                    .expect("the request lock is healthy")
                    .push(request);
            }
            let response = self
                .responses
                .lock()
                .expect("the script lock is healthy")
                .pop_front()
                .expect("the script has another response");
            Ok(Box::pin(stream::iter([Ok(StreamEvent::Completed {
                response,
            })])))
        }
    }

    fn response(content: impl IntoIterator<Item = ContentPart>) -> Response {
        Response::new(
            ProviderId::new("test"),
            ModelId::new("model"),
            content.into_iter().collect(),
        )
    }

    fn text_response(text: &str) -> Response {
        response([ContentPart::Text {
            text: text.to_owned(),
        }])
    }

    struct RecordingToolHooks {
        trace: Arc<Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl ToolCallHooks for RecordingToolHooks {
        async fn before_tool_call(
            &self,
            _context: ToolCallContext<'_>,
            _cancel: &CancellationToken,
        ) -> BeforeToolCall {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push("hook:before");
            BeforeToolCall::Proceed
        }

        async fn after_tool_call(
            &self,
            _context: ToolCallContext<'_>,
            _result: &ToolResult,
            _cancel: &CancellationToken,
        ) {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push("hook:after");
        }
    }

    struct BackgroundBoundary {
        trace: Arc<Mutex<Vec<String>>>,
    }

    struct FixedRoundExecutor;

    #[async_trait]
    impl ToolRoundExecutor for FixedRoundExecutor {
        async fn execute_round(
            &self,
            context: ToolRoundContext<'_>,
            _cancel: &CancellationToken,
        ) -> Vec<ToolResult> {
            context
                .calls()
                .iter()
                .map(|call| ToolResult {
                    tool_call_id: call.id.clone(),
                    name:         Some(call.name.clone()),
                    content:      vec![ContentPart::Text {
                        text: "specialized".to_owned(),
                    }],
                    is_error:     false,
                })
                .collect()
        }
    }

    #[async_trait]
    impl TurnBoundaryHooks for BackgroundBoundary {
        async fn before_model(
            &self,
            context: TurnBoundaryContext<'_>,
            _cancel: &CancellationToken,
        ) -> StdResult<(), crate::TurnBoundaryError> {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push(format!("before:{}", context.turn()));
            Ok(())
        }

        async fn after_model(
            &self,
            context: TurnBoundaryContext<'_>,
            _response: &Response,
            _cancel: &CancellationToken,
        ) -> StdResult<(), crate::TurnBoundaryError> {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push(format!("after:{}", context.turn()));
            Ok(())
        }

        async fn after_answer(
            &self,
            context: TurnContext<'_>,
            _response: &Response,
            _cancel: &CancellationToken,
        ) -> StdResult<TurnBoundaryAction, crate::TurnBoundaryError> {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push(format!("answer:{}", context.turn()));
            Ok(if context.turn() == 0 {
                TurnBoundaryAction::ContinueWith(UserMessage::text("background result"))
            } else {
                TurnBoundaryAction::Complete
            })
        }
    }

    #[tokio::test]
    async fn prompt_returns_the_final_response() {
        let mut agent = Agent::builder(ScriptedModel::new([text_response("done")]), "test/model")
            .build()
            .expect("the agent builds");

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "done");
        assert_eq!(outcome.turn_count(), 1);
        assert_eq!(outcome.tool_call_count(), 0);
        assert_eq!(agent.messages().len(), 2);
        assert_eq!(agent.state(), AgentState::Idle);
    }

    #[tokio::test]
    async fn tools_run_until_the_model_answers() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({"name": "parser"}),
        ))]);
        let tool = Tool::function(
            "inspect",
            "Inspect a value",
            json!({"type": "object"}),
            |_context, arguments| async move { Ok(format!("inspected {}", arguments["name"]).into()) },
        );
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("finished")]),
            "test/model",
        )
        .tools([tool])
        .build()
        .expect("the agent builds");

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "finished");
        assert_eq!(outcome.turn_count(), 2);
        assert_eq!(outcome.tool_call_count(), 1);
        assert_eq!(agent.messages().len(), 4);
        let tool_message = &agent.messages()[2];
        assert_eq!(tool_message.role(), Role::Tool);
        assert!(matches!(
            &tool_message.content()[0],
            ContentPart::ToolResult(result) if !result.is_error
        ));
    }

    #[tokio::test]
    async fn tools_and_access_are_resolved_for_each_turn() {
        let calls_hidden = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "hidden",
            json!({}),
        ))]);
        let (model, requests) = ScriptedModel::recording([calls_hidden, text_response("finished")]);
        let hidden_executions = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&hidden_executions);
        let provider = Arc::new(move |context: TurnContext<'_>| {
            let tool = |name: &'static str| {
                let observed = Arc::clone(&observed);
                Tool::function(name, name, json!({}), move |_context, _arguments| {
                    let observed = Arc::clone(&observed);
                    async move {
                        if name == "hidden" {
                            observed.fetch_add(1, Ordering::SeqCst);
                        }
                        Ok(name.into())
                    }
                })
            };
            if context.turn() == 0 {
                vec![tool("allowed"), tool("hidden")]
            } else {
                vec![tool("later")]
            }
        });
        let policy = Arc::new(|context: ToolAccessContext<'_>| {
            if context.definition().name == "hidden" {
                ToolAccess::Denied {
                    reason: "hidden for this turn".to_owned(),
                }
            } else {
                ToolAccess::Allowed
            }
        });
        let mut agent = Agent::builder(model, "test/model")
            .tool_provider(provider)
            .tool_access_policy(policy)
            .build()
            .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(hidden_executions.load(Ordering::SeqCst), 0);
        let requests = requests.lock().expect("the request lock is healthy");
        let names = |request: &Request| {
            request
                .tools()
                .iter()
                .map(|tool| tool.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&requests[0]), ["allowed".to_owned()]);
        assert_eq!(names(&requests[1]), ["later".to_owned()]);
        let ContentPart::ToolResult(result) = &agent.messages()[2].content()[0] else {
            panic!("the denied tool call has a result");
        };
        assert!(result.is_error);
        assert!(matches!(
            &result.content[0],
            ContentPart::Text { text } if text == "hidden for this turn"
        ));
    }

    #[tokio::test]
    async fn tool_hooks_and_event_projection_keep_call_order() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({}),
        ))]);
        let trace = Arc::new(Mutex::new(Vec::new()));
        let projected = Arc::clone(&trace);
        let projection = Arc::new(move |event: &AgentEvent| {
            let item = match event {
                AgentEvent::ToolStarted { .. } => Some("event:start"),
                AgentEvent::ToolCompleted { .. } => Some("event:complete"),
                _ => None,
            };
            if let Some(item) = item {
                projected
                    .lock()
                    .expect("the trace lock is healthy")
                    .push(item);
            }
        });
        let executed = Arc::clone(&trace);
        let tool = Tool::function(
            "inspect",
            "inspect",
            json!({}),
            move |_context, _arguments| {
                executed
                    .lock()
                    .expect("the trace lock is healthy")
                    .push("tool");
                async { Ok("done".into()) }
            },
        );
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("finished")]),
            "test/model",
        )
        .tools([tool])
        .tool_call_hooks(Arc::new(RecordingToolHooks {
            trace: Arc::clone(&trace),
        }))
        .event_projection(projection)
        .build()
        .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(*trace.lock().expect("the trace lock is healthy"), [
            "event:start",
            "hook:before",
            "tool",
            "event:complete",
            "hook:after"
        ]);
    }

    #[tokio::test]
    async fn a_specialized_layer_can_execute_a_complete_tool_round() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({}),
        ))]);
        let tool = Tool::function(
            "inspect",
            "inspect",
            json!({}),
            |_context, _arguments| async { panic!("the generic executor must not run") },
        );
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("finished")]),
            "test/model",
        )
        .tools([tool])
        .tool_round_executor(Arc::new(FixedRoundExecutor))
        .build()
        .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        let ContentPart::ToolResult(result) = &agent.messages()[2].content()[0] else {
            panic!("the call has a result");
        };
        assert!(matches!(
            &result.content[0],
            ContentPart::Text { text } if text == "specialized"
        ));
    }

    #[tokio::test]
    async fn turn_boundary_hooks_can_continue_with_a_background_result() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let mut agent = Agent::builder(
            ScriptedModel::new([text_response("first"), text_response("second")]),
            "test/model",
        )
        .turn_boundary_hooks(Arc::new(BackgroundBoundary {
            trace: Arc::clone(&trace),
        }))
        .build()
        .expect("the agent builds");

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "second");
        assert_eq!(agent.messages().len(), 4);
        assert_eq!(*trace.lock().expect("the trace lock is healthy"), [
            "before:0", "after:0", "answer:0", "before:1", "after:1", "answer:1"
        ]);
    }

    #[tokio::test]
    async fn invalid_tool_arguments_are_answered_without_running_the_tool() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({}),
        ))]);
        let executions = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&executions);
        let tool = Tool::function(
            "inspect",
            "Inspect a value",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            move |_context, _arguments| {
                observed.fetch_add(1, Ordering::SeqCst);
                async { Ok("unexpected".into()) }
            },
        );
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("corrected")]),
            "test/model",
        )
        .tools([tool])
        .build()
        .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        let ContentPart::ToolResult(result) = &agent.messages()[2].content()[0] else {
            panic!("the tool call has a result");
        };
        let ContentPart::Text { text } = &result.content[0] else {
            panic!("the validation result is text");
        };
        assert!(result.is_error);
        assert!(text.contains("missing required property"));
    }

    #[test]
    fn duplicate_tool_names_are_rejected() {
        let make_tool = || {
            Tool::function("same", "same", json!({}), |_context, _arguments| async {
                Ok("ok".into())
            })
        };
        let result = Agent::builder(ScriptedModel::new([]), "test/model")
            .tools([make_tool(), make_tool()])
            .build();
        let Err(error) = result else {
            panic!("duplicate names are ambiguous");
        };

        assert!(matches!(
            error,
            AgentBuildError::DuplicateTool { ref name } if name == "same"
        ));
    }

    #[tokio::test]
    async fn follow_up_is_processed_before_the_agent_becomes_idle() {
        let mut agent = Agent::builder(
            ScriptedModel::new([text_response("first"), text_response("second")]),
            "test/model",
        )
        .build()
        .expect("the agent builds");
        assert!(agent.follow_up("next"));

        let outcome = agent.prompt("start").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "second");
        assert_eq!(outcome.turn_count(), 2);
        assert_eq!(agent.messages().len(), 4);
    }

    #[tokio::test]
    async fn reset_clears_history_and_queued_input() {
        let mut agent = Agent::builder(ScriptedModel::new([text_response("done")]), "test/model")
            .build()
            .expect("the agent builds");
        assert!(agent.follow_up("later"));
        assert!(agent.steer("change course"));

        assert!(agent.reset());
        let outcome = agent.prompt("start").await.expect("the prompt succeeds");

        assert_eq!(outcome.turn_count(), 1);
        assert_eq!(agent.messages().len(), 2);
    }

    #[tokio::test]
    async fn shutdown_refuses_new_prompts() {
        let mut agent = Agent::builder(ScriptedModel::new([]), "test/model")
            .build()
            .expect("the agent builds");

        assert!(agent.shutdown());
        assert!(!agent.shutdown());
        assert!(matches!(
            agent.prompt("work").await,
            Err(AgentError::Closed)
        ));
    }
}
