//! The ready-to-run coding-agent facade.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use lithos_llm::Client;
use lithos_llm::catalog::MetadataError;
use lithos_llm::resolver::ModelSelectionError;
use lithos_llm::types::{ReasoningEffort, RequestBuildError, Speed};
use pebble_agent::{QueueOutcome, UserMessage};
use tokio::sync::{Notify, broadcast};
use tokio_util::sync::CancellationToken;

use crate::config::CodingAgentOptions;
use crate::environment::Environment;
use crate::error::{Error, InterruptReason};
use crate::event::{EventCapacity, EventSink};
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::prompt_transform::SystemPromptTransform;
use crate::record::SessionRecord;
use crate::redact::Redactor;
use crate::runtime::{
    CodingRuntime, CodingRuntimeBuilder, InterruptReasonHandle, SessionControlHandle,
    SteeringLease, WarmState, actor_from_attribution,
};
use crate::search::SearchProvider;
use crate::subagent::{ChildAgentSpec, SubagentOptions};
use crate::tool::{RegisteredTool, ToolEnvProvider};
use crate::types::{Actor, CodingAgentEvent, CodingAgentState, Message, TokenUsage};

/// Why a coding agent is being shut down.
///
/// Recorded for the application's benefit; it never reaches the event stream,
/// which reports the end of a session the same way whatever ended it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShutdownReason {
    /// The work finished.
    Completed,
    /// Someone cancelled the prompt.
    Cancelled,
    /// The prompt failed.
    Error,
}

/// Where one prompt spent its time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptTiming {
    /// Time spent waiting on the model.
    pub inference: Duration,
    /// Time spent running tools.
    pub tool:      Duration,
}

/// A coding agent could not be built.
///
/// Every variant names something the application chose: a missing dependency, a
/// selector that resolves to nothing, or a model whose catalog entry does not
/// say which harness it expects.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CodingAgentBuildError {
    /// Resource discovery or system-prompt construction failed.
    #[error("initializing the coding agent")]
    Initialization {
        /// The initialization failure.
        #[source]
        source: Box<Error>,
    },

    /// No model selector was given.
    #[error("a coding agent needs a model selector")]
    MissingModel,

    /// No environment was given, so the tools would have nowhere to act.
    #[error("a coding agent needs an environment for its tools to act through")]
    MissingEnvironment,

    /// The selector names no model the client can reach.
    #[error("model selector `{selector}` resolves to no available model: {source}")]
    ModelSelection {
        /// The selector the application gave.
        selector: String,
        /// What the client's resolver said.
        #[source]
        source:   ModelSelectionError,
    },

    /// The request pebble resolves the selector with could not be built, which
    /// an empty or blank selector runs into.
    #[error("model selector `{selector}` is not usable: {source}")]
    Selector {
        /// The selector the application gave.
        selector: String,
        /// What request building said.
        #[source]
        source:   RequestBuildError,
    },

    /// Neither the model nor its provider says which harness the model expects.
    #[error(
        "model {model} names no agent profile: neither it nor its provider carries \
         `metadata.pebble.profile`"
    )]
    MissingProfileMetadata {
        /// The model that was resolved.
        model: String,
    },

    /// The `pebble` metadata is present but not shaped as pebble reads it.
    #[error("the `pebble` catalog metadata for model {model} could not be read: {source}")]
    InvalidProfileMetadata {
        /// The model that was resolved.
        model:  String,
        /// What reading the namespace said.
        #[source]
        source: MetadataError,
    },

    /// The catalog names a harness pebble does not know.
    #[error("model {model} names agent profile `{profile}`, which pebble does not know")]
    UnknownProfile {
        /// The model that was resolved.
        model:   String,
        /// The identifier the catalog carried.
        profile: String,
    },

    /// The stored record's format version is one this build does not read.
    #[error(
        "session record format version {version} is not one this build reads (it reads up to \
         {supported})"
    )]
    UnsupportedRecord {
        /// The version the record declares.
        version:   u32,
        /// The newest version this build reads.
        supported: u32,
    },

    /// Resume was asked to restore the recorded model, and the record names
    /// no provider or no model to restore.
    #[error("session record {session_id} names no provider and model to resume on")]
    RecordedRouteMissing {
        /// The session the record describes.
        session_id: String,
    },

    /// The exact route the record names is not one this client can reach, so
    /// the session is not resumed on a different one.
    #[error("the recorded route {provider}/{model} is not available to this client: {source}")]
    RecordedRouteUnavailable {
        /// The provider the record names.
        provider: String,
        /// The model the record names.
        model:    String,
        /// What the client's resolver said.
        #[source]
        source:   ModelSelectionError,
    },

    /// The recorded route resolved to a different provider or model, which a
    /// resume on the recorded model never accepts.
    #[error("the recorded route {recorded} resolved to {resolved}, which is a different model")]
    RecordedRouteMismatch {
        /// The `provider/model` pair the record names.
        recorded: String,
        /// The `provider/model` pair the client resolved.
        resolved: String,
    },

    /// A model was set on a builder that resumes a session, and the resume
    /// mode already decides the model.
    #[error("a resumed agent takes its model from the resume mode, not from the builder")]
    ModelConflictsWithResume,
}

/// Which model a resumed session runs on.
///
/// Resume never infers this from how the builder was called. The recorded
/// route is restored exactly or the caller names a replacement, and the choice
/// is visible where the agent is resumed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResumeMode {
    /// Restore the exact `provider/model` route the record names. A record
    /// that names none, or names a route this client cannot reach, is refused
    /// rather than resumed on a different model.
    RecordedModel,
    /// Keep the conversation but run it on `selector`, which the client's
    /// catalog resolves. This is the failover path, and the new route is what
    /// the next record stores.
    UseModel(String),
}

/// A warm, in-memory handoff from one live coding agent to its successor.
///
/// Where a [`SessionRecord`] is the minimal durable form and a session
/// resumed from one re-initializes — probing the environment, loading memory
/// and skills, rebuilding the system prompt — an export also carries that
/// derived state, so the successor skips initialization. The successor is
/// built on the exported route: the system prompt names the model, so a
/// change of model goes through [`CodingAgent::resume`] instead.
///
/// Taking an export does not consume or close the agent it came from. The
/// application shuts the predecessor down when it is done with it.
#[derive(Clone, Debug)]
pub struct CodingAgentExport {
    inner: WarmState,
}

impl CodingAgentExport {
    /// The durable record inside the export, for storing beside it.
    #[must_use]
    pub const fn record(&self) -> &SessionRecord {
        &self.inner.record
    }
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

/// What a builder resumes from, when it resumes at all.
enum ResumeSource {
    Record(SessionRecord, ResumeMode),
    Export(WarmState),
}

/// Builds an initialized [`CodingAgent`].
#[must_use = "a builder does nothing until `build().await` is called"]
pub struct CodingAgentBuilder {
    inner:  CodingRuntimeBuilder,
    resume: Option<ResumeSource>,
}

impl CodingAgentBuilder {
    fn new(client: Client, environment: Arc<dyn Environment>) -> Self {
        Self {
            inner:  CodingRuntime::builder(client).environment(environment),
            resume: None,
        }
    }

    /// Names the model through the client's catalog.
    ///
    /// A builder that resumes a session takes its model from the resume mode
    /// instead, and refuses one set here.
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

    /// Lets the application adjust the system prompt the selected profile
    /// writes: use it, add to it, or replace it.
    ///
    /// The tools, their names, and the runtime behavior around them are the
    /// profile's whatever the transform answers. It applies to this agent only,
    /// not to the subagents it spawns.
    pub fn system_prompt_transform(mut self, transform: Arc<dyn SystemPromptTransform>) -> Self {
        self.inner = self.inner.system_prompt_transform(transform);
        self
    }

    /// Replaces coding-agent policy.
    pub fn options(mut self, options: CodingAgentOptions) -> Self {
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

    /// Configures the subagents this agent may spawn.
    ///
    /// Pebble builds the children itself. A child acts through this agent's
    /// environment, runs on its model under its tool access policy and hooks,
    /// forwards its events through this agent's stream, and inherits only the
    /// application tools marked
    /// [`allow_in_subagents`](RegisteredTool::allow_in_subagents) — never one
    /// that [`requires_human_input`](RegisteredTool::requires_human_input).
    /// Without this call, or with [`SubagentOptions::disabled`], no subagent
    /// tools are advertised.
    pub fn subagents(mut self, options: SubagentOptions) -> Self {
        if options.is_enabled() {
            self.inner = self
                .inner
                .subagents(Arc::new(ChildAgentSpec::build))
                .subagent_limits(options.limits());
        }
        self
    }

    /// Builds and initializes the coding agent.
    ///
    /// # Errors
    ///
    /// Returns an error when required configuration is absent, model metadata
    /// does not select a supported coding profile, or resource initialization
    /// fails.
    pub async fn build(self) -> Result<CodingAgent, CodingAgentBuildError> {
        if self.resume.is_some() && self.inner.has_model() {
            return Err(CodingAgentBuildError::ModelConflictsWithResume);
        }
        let inner = match self.resume {
            Some(ResumeSource::Record(record, mode)) => {
                let mut inner = CodingRuntime::from_record(record, &mode, self.inner)?;
                if let Err(source) = inner.initialize().await {
                    let _ = inner.shutdown(ShutdownReason::Error).await;
                    return Err(CodingAgentBuildError::Initialization {
                        source: Box::new(source),
                    });
                }
                inner
            }
            Some(ResumeSource::Export(state)) => {
                let mut inner = CodingRuntime::from_warm_state(state, self.inner)?;
                inner.start_from_warm_state();
                inner
            }
            None => {
                let mut inner = self.inner.build()?;
                if let Err(source) = inner.initialize().await {
                    let _ = inner.shutdown(ShutdownReason::Error).await;
                    return Err(CodingAgentBuildError::Initialization {
                        source: Box::new(source),
                    });
                }
                inner
            }
        };
        let control = CodingControl::new(&inner);
        Ok(CodingAgent { inner, control })
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

/// Out-of-band input for a coding agent: what to say, and who said it.
///
/// Steering reaches the model as its own turn at the next round boundary and
/// publishes [`SteeringInjected`](crate::events::CodingEvent::SteeringInjected)
/// with the same author. A follow-up runs as ordinary user input once the
/// current prompt reaches an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SteeringMessage {
    text:  String,
    actor: Option<Actor>,
}

impl SteeringMessage {
    /// A message with no named author.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text:  text.into(),
            actor: None,
        }
    }

    /// Names who wrote the message.
    #[must_use]
    pub fn with_actor(mut self, actor: Actor) -> Self {
        self.actor = Some(actor);
        self
    }

    /// What the message says.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Who wrote it, when the application said.
    #[must_use]
    pub const fn actor(&self) -> Option<&Actor> {
        self.actor.as_ref()
    }

    /// The message the generic agent evicted from its queue, read back as it
    /// was given.
    fn from_user_message(message: &UserMessage) -> Self {
        Self {
            text:  message.text_content(),
            actor: actor_from_attribution(message.attribution()),
        }
    }
}

impl From<&str> for SteeringMessage {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

impl From<String> for SteeringMessage {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

/// What the control handle did with one message.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SteeringOutcome {
    /// The message is queued.
    Accepted,
    /// The message is queued, and the queue was full: this older message was
    /// dropped to make room and the agent will never see it. The application
    /// decides whether that is worth reporting.
    Evicted(SteeringMessage),
    /// The agent is closed and nothing was queued.
    Closed,
}

impl SteeringOutcome {
    /// Whether the message was queued, with or without an eviction.
    #[must_use]
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted | Self::Evicted(_))
    }

    fn from_queue(outcome: QueueOutcome) -> Self {
        match outcome {
            QueueOutcome::Queued => Self::Accepted,
            QueueOutcome::Evicted(message) => {
                Self::Evicted(SteeringMessage::from_user_message(&message))
            }
            // The generic agent's outcome may grow; anything else means the
            // message was not queued.
            QueueOutcome::Closed | _ => Self::Closed,
        }
    }
}

/// A read-only view of a coding agent's control state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlSnapshot {
    running:            bool,
    closed:             bool,
    parked:             bool,
    pending_steering:   usize,
    pending_follow_ups: usize,
}

impl ControlSnapshot {
    /// Whether a prompt is running.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    /// Whether the agent is closed or is finishing an abort.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// Whether the prompt is parked after an interrupt, waiting for a steer.
    #[must_use]
    pub const fn is_parked(&self) -> bool {
        self.parked
    }

    /// Steering messages queued for the next round boundary.
    #[must_use]
    pub const fn pending_steering(&self) -> usize {
        self.pending_steering
    }

    /// Follow-ups queued to run after the current answer.
    #[must_use]
    pub const fn pending_follow_ups(&self) -> usize {
        self.pending_follow_ups
    }
}

/// Controls a coding agent while another task awaits its prompt.
///
/// Cloning is cheap. Every clone steers, follows up, aborts, and observes the
/// same agent. Every operation is safe to call from any task at any time,
/// including from several handles at once: the queue and the interrupt ledger
/// are updated under one lock, so two steers sent together both land and an
/// interrupt is announced exactly once.
#[derive(Clone)]
pub struct CodingAgentControlHandle {
    control: Arc<CodingControl>,
}

impl CodingAgentControlHandle {
    fn new(control: Arc<CodingControl>) -> Self {
        Self { control }
    }

    /// How many steering messages wait for a round boundary before the oldest
    /// is dropped.
    ///
    /// Pebble owns the bound and the rule: a full queue evicts its oldest
    /// message, and the outcome carries what was evicted so the application
    /// can report it. Follow-ups are not bounded; they run one at a time after
    /// each answer and never race a round boundary.
    pub const STEERING_QUEUE_CAPACITY: usize = 64;

    /// Queues steering for the next round boundary without interrupting the
    /// round in progress.
    ///
    /// The model finishes what it is doing and reads the message before its
    /// next turn. Steering queued while idle opens the next prompt.
    pub fn queue_steering(&self, message: impl Into<SteeringMessage>) -> SteeringOutcome {
        if self.is_closed() {
            return SteeringOutcome::Closed;
        }
        let message = message.into();
        SteeringOutcome::from_queue(self.control.session.queue_steering(
            message.text,
            message.actor,
            Some(Self::STEERING_QUEUE_CAPACITY),
        ))
    }

    /// Interrupts the round in progress and queues `message` as what replaces
    /// it, in one step.
    ///
    /// The interrupt and the enqueue happen under one lock, so the prompt can
    /// never observe the interrupt with an empty queue and park when the steer
    /// was already on its way. While no prompt is running this only queues, and
    /// the message opens the next prompt.
    pub fn steer_now(&self, message: impl Into<SteeringMessage>) -> SteeringOutcome {
        let state = self.control.state();
        if state.closed {
            return SteeringOutcome::Closed;
        }
        let message = message.into();
        let capacity = Some(Self::STEERING_QUEUE_CAPACITY);
        let outcome = if state.running {
            self.control
                .session
                .steer_now(message.text, message.actor, capacity)
        } else {
            self.control
                .session
                .queue_steering(message.text, message.actor, capacity)
        };
        SteeringOutcome::from_queue(outcome)
    }

    /// Interrupts the round in progress without saying what comes next.
    ///
    /// The prompt abandons its round, publishes one
    /// [`RoundInterrupted`](crate::events::CodingEvent::RoundInterrupted), and
    /// parks at the next boundary until a steer arrives or the prompt is
    /// cancelled. Returns whether a prompt was running to interrupt.
    pub fn interrupt(&self) -> bool {
        let state = self.control.state();
        if !state.running || state.closed {
            return false;
        }
        self.control.session.interrupt()
    }

    /// Holds natural completion open while an external steering source is
    /// attached.
    ///
    /// While the returned [`SteeringLease`] is alive, a prompt that reaches a
    /// plain answer parks rather than completing, so a steer the source is
    /// about to send cannot lose the completion close-door race. Dropping the
    /// final lease wakes a parked prompt and lets it complete. This is the
    /// supported replacement for reaching into the drain, park, and generation
    /// protocol directly.
    pub fn hold_open_for_steering(&self) -> SteeringLease {
        SteeringLease::acquire(self.control.session.clone())
    }

    /// Queues input to run as its own user turn once the current prompt
    /// reaches an answer.
    ///
    /// A follow-up is ordinary input, not steering: it does not interrupt
    /// anything and is not bounded. The author is not recorded on the turn it
    /// becomes.
    pub fn queue_follow_up(&self, message: impl Into<SteeringMessage>) -> SteeringOutcome {
        if self.is_closed() {
            return SteeringOutcome::Closed;
        }
        self.control
            .follow_up
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back(message.into().text);
        SteeringOutcome::Accepted
    }

    /// Aborts the active prompt and closes the agent for good.
    ///
    /// This is the terminal gesture. To end one prompt and keep the agent,
    /// cancel the token given to
    /// [`CodingAgent::prompt_with_cancellation`] instead. Returns whether a
    /// prompt was running.
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

    /// Whether the agent is closed or is finishing an abort.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.control.state().closed
    }

    /// A read-only view of the agent's state and queued input.
    #[must_use]
    pub fn snapshot(&self) -> ControlSnapshot {
        let state = self.control.state();
        ControlSnapshot {
            running:            state.running,
            closed:             state.closed,
            parked:             self.control.session.is_parked(),
            pending_steering:   self.control.session.pending_steering(),
            pending_follow_ups: self
                .control
                .follow_up
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .len(),
        }
    }
}

impl fmt::Debug for CodingAgentControlHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodingAgentControlHandle")
            .field("running", &self.is_running())
            .field("closed", &self.is_closed())
            .finish()
    }
}

/// A ready coding-agent conversation.
///
/// This type owns coding profiles, resources, tools, environment access,
/// compaction policy, and subagents. Use [`pebble_agent::Agent`] directly when
/// those coding-specific facilities are not needed.
pub struct CodingAgent {
    inner:   CodingRuntime,
    control: Arc<CodingControl>,
}

impl CodingAgent {
    /// Starts a builder with the model client and tool environment.
    pub fn builder(client: Client, environment: Arc<dyn Environment>) -> CodingAgentBuilder {
        CodingAgentBuilder::new(client, environment)
    }

    /// Starts a builder that resumes a stored session on the model `mode`
    /// names.
    ///
    /// The record supplies the identity, the conversation, and where the event
    /// stream had got to; the builder supplies everything a record cannot hold
    /// and must not also name a model. The resumed agent initializes again —
    /// probing the environment and rebuilding its system prompt — so a
    /// [`ResumeMode::UseModel`] failover gets a prompt written for the model it
    /// runs on.
    ///
    /// ```no_run
    /// use pebble_coding_agent::{CodingAgent, ResumeMode};
    /// # use std::sync::Arc;
    /// # async fn example(
    /// #     client: lithos_llm::Client,
    /// #     environment: Arc<dyn pebble_coding_agent::environment::Environment>,
    /// #     record: pebble_coding_agent::state::SessionRecord,
    /// # ) -> Result<(), Box<dyn std::error::Error>> {
    /// let same = CodingAgent::resume(
    ///     client.clone(),
    ///     Arc::clone(&environment),
    ///     record.clone(),
    ///     ResumeMode::RecordedModel,
    /// )
    /// .build()
    /// .await?;
    ///
    /// let fallback = CodingAgent::resume(
    ///     client,
    ///     environment,
    ///     record,
    ///     ResumeMode::UseModel("openai/gpt-fallback".into()),
    /// )
    /// .build()
    /// .await?;
    /// # let _ = (same, fallback);
    /// # Ok(())
    /// # }
    /// ```
    pub fn resume(
        client: Client,
        environment: Arc<dyn Environment>,
        record: SessionRecord,
        mode: ResumeMode,
    ) -> CodingAgentBuilder {
        let mut builder = CodingAgentBuilder::new(client, environment);
        builder.resume = Some(ResumeSource::Record(record, mode));
        builder
    }

    /// Starts a builder that continues a live agent from its
    /// [`export`](Self::export), without initializing again.
    ///
    /// This is the handoff between two agents in one process: the successor
    /// keeps the identity, the conversation, the system prompt, and the
    /// discovered skills, and binds whatever services the builder gives it —
    /// a different human-input provider, different tool hooks — for its own
    /// life. It runs on the exported route; a change of model goes through
    /// [`resume`](Self::resume). Event numbering continues from the export, on
    /// a fresh event stream.
    pub fn resume_from_export(
        client: Client,
        environment: Arc<dyn Environment>,
        export: CodingAgentExport,
    ) -> CodingAgentBuilder {
        let mut builder = CodingAgentBuilder::new(client, environment);
        builder.resume = Some(ResumeSource::Export(export.inner));
        builder
    }

    /// Processes one user prompt and every queued follow-up to completion.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SessionClosed`] after shutdown,
    /// [`Error::Interrupted`] after cancellation or timeout, and the
    /// applicable model, tool, compaction, or event failure otherwise.
    pub async fn prompt(&mut self, input: &str) -> Result<PromptOutcome, Error> {
        self.prompt_inner(input, None).await
    }

    /// Processes one user prompt until it completes or `cancel` fires.
    ///
    /// Cancelling ends this prompt alone. The loop unwinds through its
    /// checkpoints, so every tool call the model made still gets its result and
    /// history stays paired; the prompt returns [`Error::Interrupted`] and the
    /// agent returns to [`Idle`](CodingAgentState::Idle), ready for the next
    /// prompt. Only [`shutdown`](Self::shutdown) or
    /// [`abort`](CodingAgentControlHandle::abort) closes the agent.
    ///
    /// # Errors
    ///
    /// As [`prompt`](Self::prompt).
    pub async fn prompt_with_cancellation(
        &mut self,
        input: &str,
        cancel: &CancellationToken,
    ) -> Result<PromptOutcome, Error> {
        self.prompt_inner(input, Some(cancel)).await
    }

    async fn prompt_inner(
        &mut self,
        input: &str,
        cancel: Option<&CancellationToken>,
    ) -> Result<PromptOutcome, Error> {
        self.control.begin_prompt();
        let result = self.inner.prompt_with_cancellation(input, cancel).await;
        self.control
            .finish_prompt(self.inner.state() == CodingAgentState::Closed);
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
    pub fn subscribe(&self) -> broadcast::Receiver<CodingAgentEvent> {
        self.inner.subscribe()
    }

    /// Returns a handle that can steer, follow up, abort, or await a prompt.
    #[must_use]
    pub fn control_handle(&self) -> CodingAgentControlHandle {
        CodingAgentControlHandle::new(Arc::clone(&self.control))
    }

    /// Interrupts the current round and queues `message` as what replaces it.
    pub fn steer_now(&self, message: impl Into<SteeringMessage>) -> SteeringOutcome {
        self.control_handle().steer_now(message)
    }

    /// Queues input to run after the current prompt reaches an answer.
    pub fn queue_follow_up(&self, message: impl Into<SteeringMessage>) -> SteeringOutcome {
        self.control_handle().queue_follow_up(message)
    }

    /// Aborts the active prompt and closes this agent for good.
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

    /// The stable durable-session identifier.
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

    /// The current agent state.
    #[must_use]
    pub const fn state(&self) -> CodingAgentState {
        self.inner.state()
    }

    /// The committed coding-agent conversation.
    #[must_use]
    pub const fn history(&self) -> &History {
        self.inner.history()
    }

    /// Captures the durable session state.
    ///
    /// The record can be taken between prompts or during a prompt as a crash
    /// checkpoint. Restore it with [`CodingAgent::resume`].
    #[must_use]
    pub fn to_record(&self) -> SessionRecord {
        self.inner.to_record()
    }

    /// Captures everything a successor in the same process needs to carry on
    /// without initializing again.
    ///
    /// Taking an export leaves this agent as it was. Continue from it with
    /// [`CodingAgent::resume_from_export`].
    #[must_use]
    pub fn export(&self) -> CodingAgentExport {
        CodingAgentExport {
            inner: self.inner.warm_state(),
        }
    }

    /// Closes the agent and joins its owned tasks.
    pub async fn shutdown(&mut self, reason: ShutdownReason) -> Result<bool, Error> {
        let result = self.inner.shutdown(reason).await;
        self.control.mark_closed();
        result
    }
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::ToolDefinition;
    use serde_json::json;
    use tokio::time::timeout;

    use super::*;
    use crate::runtime::testing::{blocking_tool, drained, wait_for_event};
    use crate::test_support::{
        MockEnvironment, ScriptedCall, scripted_client, text_delta_events, text_response,
        tool_call_response,
    };
    use crate::types::{CodingEvent, ToolSource};

    /// How long a test waits for a prompt another task has to unblock.
    const PATIENCE: Duration = Duration::from_secs(5);

    /// A tool that says when it starts and waits for the test to let it finish.
    fn checkpoint_tool(reached: Arc<Notify>, release: Arc<Notify>) -> RegisteredTool {
        RegisteredTool::new(
            ToolDefinition::function(
                "checkpoint",
                "Waits for the test",
                json!({"type": "object"}),
            ),
            Arc::new(move |_arguments, _context| {
                let reached = Arc::clone(&reached);
                let release = Arc::clone(&release);
                Box::pin(async move {
                    reached.notify_one();
                    release.notified().await;
                    Ok("ready".to_owned())
                })
            }),
        )
        .with_source(ToolSource::Native)
    }

    async fn agent_with(
        calls: Vec<ScriptedCall>,
        tools: impl IntoIterator<Item = RegisteredTool>,
    ) -> CodingAgent {
        let (client, _provider) = scripted_client(calls);
        CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .tools(tools)
            .build()
            .await
            .expect("the coding agent builds and initializes")
    }

    fn steering_texts(agent: &CodingAgent) -> Vec<&str> {
        agent
            .history()
            .turns()
            .iter()
            .filter_map(|turn| match turn {
                Message::Steering { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn build_returns_a_ready_session_and_prompt_returns_an_outcome() {
        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let environment = Arc::new(MockEnvironment::linux());
        let mut session = CodingAgent::builder(client, environment)
            .model("test/model")
            .build()
            .await
            .expect("the coding agent builds and initializes");

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
        let mut session = CodingAgent::builder(client, environment)
            .model("test/model")
            .build()
            .await
            .expect("the coding agent builds and initializes");
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
        assert_eq!(session.state(), CodingAgentState::Closed);
    }

    #[tokio::test]
    async fn a_coding_agent_resumes_from_its_durable_record() {
        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("first"))]);
        let environment = Arc::new(MockEnvironment::linux());
        let mut agent = CodingAgent::builder(client, environment)
            .model("test/model")
            .build()
            .await
            .expect("the coding agent builds");
        agent
            .prompt("start")
            .await
            .expect("the first prompt succeeds");
        let record = agent.to_record();
        let original_id = agent.id().to_owned();
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the first agent shuts down");

        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("second"))]);
        let environment = Arc::new(MockEnvironment::linux());
        let mut resumed =
            CodingAgent::resume(client, environment, record, ResumeMode::RecordedModel)
                .build()
                .await
                .expect("the durable record resumes");

        assert_eq!(resumed.id(), original_id);
        assert!(!resumed.history().is_empty());
        assert_eq!(
            resumed
                .prompt("continue")
                .await
                .expect("the resumed prompt succeeds")
                .text(),
            Some("second")
        );
        resumed
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the resumed agent shuts down");
    }

    // --- Steering ---

    #[tokio::test]
    async fn two_handles_steering_at_once_both_land() {
        let reached = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let mut agent = agent_with(
            vec![
                ScriptedCall::response(tool_call_response("checkpoint", "call_1", json!({}))),
                ScriptedCall::response(text_response("done")),
            ],
            [checkpoint_tool(Arc::clone(&reached), Arc::clone(&release))],
        )
        .await;
        let first = agent.control_handle();
        let second = agent.control_handle();

        let steering = tokio::spawn(async move {
            reached.notified().await;
            // Two steers from two handles with nothing between them. Both must
            // be queued: the queue and the interrupt ledger share one lock.
            let (one, two) =
                tokio::join!(async { first.steer_now("from the first handle") }, async {
                    second.steer_now(
                        SteeringMessage::new("from the second handle").with_actor(Actor::System),
                    )
                },);
            assert!(one.is_accepted());
            assert!(two.is_accepted());
            release.notify_one();
        });

        let outcome = timeout(PATIENCE, agent.prompt("start"))
            .await
            .expect("the prompt finishes")
            .expect("the prompt succeeds");
        steering.await.expect("the steering task finishes");

        assert_eq!(outcome.text(), Some("done"));
        let mut texts = steering_texts(&agent);
        texts.sort_unstable();
        assert_eq!(texts, ["from the first handle", "from the second handle"]);
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    #[tokio::test]
    async fn a_full_steering_queue_evicts_the_oldest_and_reports_it() {
        let mut agent = agent_with(vec![ScriptedCall::response(text_response("done"))], []).await;
        let control = agent.control_handle();
        let capacity = CodingAgentControlHandle::STEERING_QUEUE_CAPACITY;

        for index in 0..capacity {
            assert_eq!(
                control.queue_steering(format!("steer {index}")),
                SteeringOutcome::Accepted
            );
        }
        let outcome =
            control.queue_steering(SteeringMessage::new("one too many").with_actor(Actor::System));

        assert_eq!(
            outcome,
            SteeringOutcome::Evicted(SteeringMessage::new("steer 0")),
            "the oldest message makes room and comes back to the caller"
        );
        assert!(outcome.is_accepted());
        let snapshot = control.snapshot();
        assert_eq!(snapshot.pending_steering(), capacity);
        assert!(!snapshot.is_running());
        assert!(!snapshot.is_parked());
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    #[tokio::test]
    async fn an_interrupt_from_the_handle_is_announced_exactly_once() {
        let mut agent = agent_with(
            vec![
                ScriptedCall::EventsThenPending(text_delta_events("a long answer that")),
                ScriptedCall::response(text_response("DONE")),
            ],
            [],
        )
        .await;
        let control = agent.control_handle();
        let mut watched = agent.subscribe();
        let mut recorded = agent.subscribe();

        let controller = tokio::spawn(async move {
            wait_for_event(&mut watched, |event| {
                matches!(event, CodingEvent::TextDelta { .. })
            })
            .await;
            assert!(control.interrupt(), "a prompt was running to interrupt");
            wait_for_event(&mut watched, |event| {
                matches!(event, CodingEvent::RoundInterrupted { .. })
            })
            .await;
            assert!(control.snapshot().is_parked(), "a bare interrupt parks");
            assert_eq!(
                control.queue_steering("say DONE"),
                SteeringOutcome::Accepted
            );
        });

        let outcome = timeout(PATIENCE, agent.prompt("describe everything"))
            .await
            .expect("the interrupt unblocks the hanging stream")
            .expect("the prompt succeeds");
        controller.await.expect("the controller finishes");

        assert_eq!(outcome.text(), Some("DONE"));
        let published = drained(&mut recorded).await;
        assert_eq!(
            published
                .iter()
                .filter(|event| matches!(event, CodingEvent::RoundInterrupted { .. }))
                .count(),
            1,
            "one gesture is announced once"
        );
        assert!(!agent.control_handle().snapshot().is_parked());
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    #[tokio::test]
    async fn a_follow_up_runs_after_the_answer() {
        let mut agent = agent_with(
            vec![
                ScriptedCall::response(text_response("first")),
                ScriptedCall::response(text_response("second")),
            ],
            [],
        )
        .await;
        let control = agent.control_handle();
        assert_eq!(
            control.queue_follow_up("and then this"),
            SteeringOutcome::Accepted
        );
        assert_eq!(control.snapshot().pending_follow_ups(), 1);

        let outcome = agent.prompt("start").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), Some("second"));
        assert_eq!(control.snapshot().pending_follow_ups(), 0);
        assert!(matches!(
            agent.history().turns(),
            [
                Message::User { .. },
                Message::Assistant { .. },
                Message::User { content, .. },
                Message::Assistant { .. },
            ] if content == "and then this"
        ));
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    // --- Cancelling ---

    #[tokio::test]
    async fn cancelling_a_prompt_leaves_the_agent_idle_and_reusable() {
        let mut agent = agent_with(
            vec![
                ScriptedCall::response(tool_call_response("slow_tool", "call_1", json!({}))),
                ScriptedCall::response(text_response("second prompt done")),
            ],
            [blocking_tool("slow_tool")],
        )
        .await;
        let control = agent.control_handle();
        let mut watched = agent.subscribe();
        let cancel = CancellationToken::new();
        let canceller = cancel.clone();

        let controller = tokio::spawn(async move {
            wait_for_event(&mut watched, |event| {
                matches!(event, CodingEvent::ToolCallStarted { .. })
            })
            .await;
            canceller.cancel();
        });

        let error = timeout(PATIENCE, agent.prompt_with_cancellation("start", &cancel))
            .await
            .expect("cancellation unblocks the tool")
            .expect_err("the prompt was cancelled");
        controller.await.expect("the controller finishes");

        assert!(
            matches!(error, Error::Interrupted(InterruptReason::Cancelled)),
            "{error:?}"
        );
        assert_eq!(agent.state(), CodingAgentState::Idle);
        assert!(
            !control.is_closed(),
            "only shutdown or abort closes the agent"
        );
        assert!(!control.is_running());
        assert!(
            matches!(
                agent.history().turns().last(),
                Some(Message::ToolResults { results, .. })
                    if results.len() == 1 && results[0].tool_call_id == "call_1"
            ),
            "the cancelled call still has its result: {:?}",
            agent.history().turns()
        );

        let outcome = timeout(PATIENCE, agent.prompt("again"))
            .await
            .expect("the next prompt runs")
            .expect("the next prompt succeeds");
        assert_eq!(outcome.text(), Some("second prompt done"));
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    #[tokio::test]
    async fn a_closed_agent_refuses_control_input() {
        let mut agent = agent_with(vec![ScriptedCall::response(text_response("done"))], []).await;
        let control = agent.control_handle();
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");

        assert_eq!(control.queue_steering("late"), SteeringOutcome::Closed);
        assert_eq!(control.steer_now("late"), SteeringOutcome::Closed);
        assert_eq!(control.queue_follow_up("late"), SteeringOutcome::Closed);
        assert!(!control.interrupt());
        assert!(!control.abort());
        let snapshot = control.snapshot();
        assert!(snapshot.is_closed());
        assert_eq!(snapshot.pending_steering(), 0);
    }
}
