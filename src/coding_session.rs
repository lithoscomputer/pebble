//! The ready-to-run coding-agent facade.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::Client;
use lithos_llm::types::{ReasoningEffort, Speed};
use thiserror::Error;
use tokio::sync::{Notify, broadcast};
use tokio_util::sync::CancellationToken;

use crate::config::CodingSessionOptions;
use crate::environment::Environment;
use crate::error::{Error, InterruptReason};
use crate::event::{EventCapacity, EventSink};
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::redact::Redactor;
use crate::runtime::{
    CodingRuntime, CodingRuntimeBuildError, CodingRuntimeBuilder, InterruptReasonHandle,
    PromptTiming, SessionControlHandle, ShutdownReason,
};
use crate::search::SearchProvider;
use crate::subagent::{SessionFactory, SubagentLimits};
use crate::tool::{RegisteredTool, ToolEnvProvider};
use crate::types::{CodingSessionEvent, Message, SessionState, TokenUsage};

/// Why a ready coding session could not be built.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CodingSessionBuildError {
    /// Required configuration or model metadata was invalid.
    #[error("building the coding session")]
    Configuration {
        /// The configuration failure.
        #[source]
        source: CodingRuntimeBuildError,
    },
    /// Resource discovery or system-prompt construction failed.
    #[error("initializing the coding session")]
    Initialization {
        /// The initialization failure.
        #[source]
        source: Error,
    },
}

/// The completed result of one coding-agent prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptOutcome {
    text:            Option<String>,
    final_message:   Option<Message>,
    usage:           TokenUsage,
    cost_usd_micros: Option<u64>,
    timing:          PromptTiming,
}

impl PromptOutcome {
    /// The final readable assistant text, when the response carried any.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// The final committed assistant message.
    #[must_use]
    pub const fn final_message(&self) -> Option<&Message> {
        self.final_message.as_ref()
    }

    /// Token usage summed across all model turns in this prompt.
    #[must_use]
    pub const fn usage(&self) -> TokenUsage {
        self.usage
    }

    /// Provider-reported cost summed across all model turns, in USD micros.
    #[must_use]
    pub const fn cost_usd_micros(&self) -> Option<u64> {
        self.cost_usd_micros
    }

    /// Time spent in inference and tool execution.
    #[must_use]
    pub const fn timing(&self) -> PromptTiming {
        self.timing
    }
}

/// Builds an initialized [`CodingSession`].
#[must_use = "a builder does nothing until `build().await` is called"]
pub struct CodingSessionBuilder {
    inner: CodingRuntimeBuilder,
}

impl CodingSessionBuilder {
    fn new(client: Client, environment: Arc<dyn Environment>) -> Self {
        Self {
            inner: CodingRuntime::builder(client).environment(environment),
        }
    }

    /// Names the model through the client's catalog.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.inner = self.inner.model(model);
        self
    }

    /// Adds tools on top of the selected coding profile.
    pub fn tools(mut self, tools: impl IntoIterator<Item = RegisteredTool>) -> Self {
        self.inner = self.inner.tools(tools);
        self
    }

    /// Sets where the root session asks a person a question.
    pub fn human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.inner = self.inner.human_input(provider);
        self
    }

    /// Sets where per-round tool environment variables come from.
    pub fn tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.inner = self.inner.tool_env_provider(provider);
        self
    }

    /// Sets fixed extra environment variables for tool calls.
    pub fn tool_env(mut self, env: HashMap<String, String>) -> Self {
        self.inner = self.inner.tool_env(env);
        self
    }

    /// Sets the redactor used for event-visible process output.
    pub fn redactor(mut self, redactor: Arc<dyn Redactor>) -> Self {
        self.inner = self.inner.redactor(redactor);
        self
    }

    /// Selects the model used to summarize fetched web pages.
    pub fn web_fetch_summarizer(mut self, model: impl Into<String>) -> Self {
        self.inner = self.inner.web_fetch_summarizer(model);
        self
    }

    /// Sets the provider behind the profile's web-search tool.
    pub fn search_provider(mut self, provider: Arc<dyn SearchProvider>) -> Self {
        self.inner = self.inner.search_provider(provider);
        self
    }

    /// Replaces coding-session policy.
    pub fn options(mut self, options: CodingSessionOptions) -> Self {
        self.inner = self.inner.options(options);
        self
    }

    /// Records every coding event before a live subscriber sees it.
    pub fn event_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.inner = self.inner.event_sink(sink);
        self
    }

    /// Sets the capacity of each live event subscription.
    pub fn event_capacity(mut self, capacity: impl Into<EventCapacity>) -> Self {
        self.inner = self.inner.event_capacity(capacity);
        self
    }

    /// Enables coding subagents built by `factory`.
    pub fn subagents(mut self, factory: SessionFactory) -> Self {
        self.inner = self.inner.subagents(factory);
        self
    }

    /// Sets the open-session limit for the subagent tree.
    pub fn subagent_limits(mut self, limits: SubagentLimits) -> Self {
        self.inner = self.inner.subagent_limits(limits);
        self
    }

    /// Builds and initializes the coding session.
    ///
    /// # Errors
    ///
    /// Returns an error when required configuration is absent, model metadata
    /// does not select a supported coding profile, or resource initialization
    /// fails.
    pub async fn build(self) -> Result<CodingSession, CodingSessionBuildError> {
        let mut inner = self
            .inner
            .build()
            .map_err(|source| CodingSessionBuildError::Configuration { source })?;
        if let Err(source) = inner.initialize().await {
            let _ = inner.shutdown(ShutdownReason::Error).await;
            return Err(CodingSessionBuildError::Initialization { source });
        }
        let control = CodingControl::new(&inner);
        Ok(CodingSession { inner, control })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CodingPromptState {
    running: bool,
    closed:  bool,
}

struct CodingControl {
    session:          SessionControlHandle,
    follow_up:        Arc<Mutex<VecDeque<String>>>,
    cancel:           CancellationToken,
    interrupt_reason: InterruptReasonHandle,
    prompt_state:     Mutex<CodingPromptState>,
    idle:             Notify,
}

impl CodingControl {
    fn new(session: &CodingRuntime) -> Arc<Self> {
        Arc::new(Self {
            session:          session.control_handle(),
            follow_up:        session.followup_queue_handle(),
            cancel:           session.cancel_token(),
            interrupt_reason: session.interrupt_reason_handle(),
            prompt_state:     Mutex::new(CodingPromptState::default()),
            idle:             Notify::new(),
        })
    }

    fn begin_prompt(&self) {
        let mut state = self
            .prompt_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        debug_assert!(!state.running, "a mutable session cannot start two prompts");
        state.running = true;
    }

    fn finish_prompt(&self, closed: bool) {
        let mut state = self
            .prompt_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state.running = false;
        state.closed |= closed;
        drop(state);
        self.idle.notify_waiters();
    }

    fn mark_closed(&self) {
        let mut state = self
            .prompt_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state.closed = true;
    }

    fn state(&self) -> CodingPromptState {
        *self
            .prompt_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Controls a coding session while another task awaits its prompt.
///
/// Cloning is cheap. Every clone steers, follows up, aborts, and observes the
/// same session.
#[derive(Clone)]
pub struct CodingSessionControlHandle {
    control: Arc<CodingControl>,
}

impl CodingSessionControlHandle {
    fn new(control: Arc<CodingControl>) -> Self {
        Self { control }
    }

    /// Queues steering for the next model turn and interrupts the current one.
    ///
    /// Steering queued while idle is applied in the next prompt. Returns
    /// `false` after the session closes.
    pub fn steer(&self, message: impl Into<String>) -> bool {
        let state = self.control.state();
        if state.closed {
            return false;
        }
        if state.running {
            self.control.session.interrupt_then_steer(message, None);
        } else {
            self.control.session.steer(message, None);
        }
        true
    }

    /// Queues input to run after the current prompt reaches an answer.
    ///
    /// Returns `false` after the session closes.
    pub fn follow_up(&self, message: impl Into<String>) -> bool {
        if self.is_closed() {
            return false;
        }
        self.control
            .follow_up
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back(message.into());
        true
    }

    /// Aborts the active prompt and closes the session.
    ///
    /// Returns whether a prompt was running.
    pub fn abort(&self) -> bool {
        {
            let mut state = self
                .control
                .prompt_state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if !state.running || state.closed {
                return false;
            }
            state.closed = true;
        }
        self.control
            .interrupt_reason
            .record(InterruptReason::Cancelled);
        self.control.cancel.cancel();
        true
    }

    /// Waits until no prompt is running.
    pub async fn wait_for_idle(&self) {
        loop {
            let notified = self.control.idle.notified();
            if !self.is_running() {
                return;
            }
            notified.await;
        }
    }

    /// Whether a prompt is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.control.state().running
    }

    /// Whether the session is closed or is finishing an abort.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.control.state().closed
    }
}

impl fmt::Debug for CodingSessionControlHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodingSessionControlHandle")
            .field("running", &self.is_running())
            .field("closed", &self.is_closed())
            .finish()
    }
}

/// A ready coding-agent conversation.
///
/// This facade owns coding profiles, resources, tools, environment access,
/// compaction policy, and subagents. Use
/// [`pebble::advanced::agent`](crate::advanced::agent) when those
/// coding-specific facilities are not needed.
pub struct CodingSession {
    inner:   CodingRuntime,
    control: Arc<CodingControl>,
}

impl CodingSession {
    /// Starts a builder with the model client and tool environment.
    pub fn builder(client: Client, environment: Arc<dyn Environment>) -> CodingSessionBuilder {
        CodingSessionBuilder::new(client, environment)
    }

    /// Processes one user prompt and every queued follow-up to completion.
    ///
    /// # Errors
    ///
    /// Returns the same prompt failures as [`CodingRuntime::prompt`].
    pub async fn prompt(&mut self, input: &str) -> Result<PromptOutcome, Error> {
        self.control.begin_prompt();
        let result = self.inner.prompt(input).await;
        self.control
            .finish_prompt(self.inner.state() == SessionState::Closed);
        let text = result?;
        let final_message = self
            .inner
            .history()
            .turns()
            .iter()
            .rev()
            .find(|message| matches!(message, Message::Assistant { .. }))
            .cloned();
        Ok(PromptOutcome {
            text,
            final_message,
            usage: self.inner.last_prompt_usage(),
            cost_usd_micros: self.inner.last_prompt_cost_usd_micros(),
            timing: self.inner.last_prompt_timing(),
        })
    }

    /// Subscribes to coding-agent events.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<CodingSessionEvent> {
        self.inner.subscribe()
    }

    /// Returns a handle that can steer, follow up, abort, or await a prompt.
    #[must_use]
    pub fn control_handle(&self) -> CodingSessionControlHandle {
        CodingSessionControlHandle::new(Arc::clone(&self.control))
    }

    /// Queues steering and interrupts the current model turn.
    pub fn steer(&self, text: impl Into<String>) -> bool {
        self.control_handle().steer(text)
    }

    /// Queues input after the current prompt reaches an answer.
    pub fn follow_up(&self, message: impl Into<String>) -> bool {
        self.control_handle().follow_up(message)
    }

    /// Aborts the active prompt and closes this session.
    pub fn abort(&self) -> bool {
        self.control_handle().abort()
    }

    /// Waits until no prompt is running.
    pub async fn wait_for_idle(&self) {
        self.control_handle().wait_for_idle().await;
    }

    /// Changes the reasoning effort used by later model turns.
    pub fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.inner.set_reasoning_effort(effort);
    }

    /// Changes the speed tier used by later model turns.
    pub fn set_speed(&mut self, speed: Option<Speed>) {
        self.inner.set_speed(speed);
    }

    /// The stable session identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        self.inner.id()
    }

    /// The resolved provider identifier.
    #[must_use]
    pub fn provider(&self) -> &str {
        self.inner.provider()
    }

    /// The resolved model identifier.
    #[must_use]
    pub fn model(&self) -> &str {
        self.inner.model()
    }

    /// The current session state.
    #[must_use]
    pub const fn state(&self) -> SessionState {
        self.inner.state()
    }

    /// The committed coding-agent conversation.
    #[must_use]
    pub const fn history(&self) -> &History {
        self.inner.history()
    }

    /// Closes the session and joins its owned tasks.
    pub async fn shutdown(&mut self, reason: ShutdownReason) -> Result<bool, Error> {
        let result = self.inner.shutdown(reason).await;
        self.control.mark_closed();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockEnvironment, ScriptedCall, scripted_client, text_response};

    #[tokio::test]
    async fn build_returns_a_ready_session_and_prompt_returns_an_outcome() {
        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let environment = Arc::new(MockEnvironment::linux());
        let mut session = CodingSession::builder(client, environment)
            .model("test/model")
            .build()
            .await
            .expect("the coding session builds and initializes");

        let outcome = session.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), Some("done"));
        assert!(outcome.final_message().is_some());
        assert_eq!(outcome.usage(), session.inner.last_prompt_usage());
        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the session shuts down");
    }

    #[tokio::test]
    async fn control_handle_aborts_an_active_prompt_and_waits_for_it_to_settle() {
        let (client, provider) = scripted_client(vec![ScriptedCall::PendingOpen]);
        let environment = Arc::new(MockEnvironment::linux());
        let mut session = CodingSession::builder(client, environment)
            .model("test/model")
            .build()
            .await
            .expect("the coding session builds and initializes");
        let control = session.control_handle();

        let prompting = session.prompt("work");
        let controlling = async {
            provider.wait_for_call().await;
            assert!(control.is_running());
            assert!(control.abort());
            control.wait_for_idle().await;
            assert!(!control.is_running());
            assert!(control.is_closed());
        };
        let (result, ()) = tokio::join!(prompting, controlling);

        assert!(matches!(
            result,
            Err(Error::Interrupted(InterruptReason::Cancelled))
        ));
        assert_eq!(session.state(), SessionState::Closed);
    }
}
