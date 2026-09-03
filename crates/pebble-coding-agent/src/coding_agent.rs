//! The ready-to-run coding-agent facade.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use lithos_llm::Client;
use lithos_llm::catalog::MetadataError;
use lithos_llm::resolver::ModelSelectionError;
use lithos_llm::types::{ContentPart, ReasoningEffort, RequestBuildError, Speed};
use pebble_agent::{
    AgentControlHandle, AgentPendingInput, QueueOutcome, ToolMiddleware, UserMessage,
};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::compaction::{CompactionControl, CompactionOptions, CompactionOutcome};
use crate::config::CodingAgentOptions;
use crate::environment::Environment;
use crate::error::{Error, InterruptReason};
use crate::event::{EventCapacity, EventSink, EventSinkTimeout};
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::prompt_transform::SystemPromptTransform;
use crate::record::SessionRecord;
use crate::redact::Redactor;
use crate::runtime::{
    CodingRuntime, CodingRuntimeBuilder, InterruptReasonHandle, StateMachine, SteeringLease,
    WarmState, actor_from_attribution, input_message, steering_message,
};
use crate::search::SearchProvider;
use crate::subagent::SubagentOptions;
use crate::tool::{RegisteredTool, ToolEnvProvider};
use crate::types::{
    Actor, AgentProfileKind, CodingAgentEvent, CodingAgentState, ContextWindowSnapshot,
    InputContent, InputSource, MemoryFileSummary, Message, SkillSummary, TokenUsage, ToolSummary,
};

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
    #[error("resolving model selector `{selector}`")]
    ModelSelection {
        /// The selector the application gave.
        selector: String,
        /// What the client's resolver said.
        #[source]
        source:   ModelSelectionError,
    },

    /// The request pebble resolves the selector with could not be built, which
    /// an empty or blank selector runs into.
    #[error("building a request for model selector `{selector}`")]
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
    #[error("reading the `pebble` catalog metadata for model {model}")]
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
    #[error("resolving the recorded route {provider}/{model}")]
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

/// An owned view of a coding agent at one committed event boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct CodingAgentSnapshot {
    session_id:          String,
    stream_id:           String,
    state:               CodingAgentState,
    provider:            String,
    model:               String,
    profile:             AgentProfileKind,
    history:             History,
    pending_input:       PendingInput,
    memory:              Vec<MemoryFileSummary>,
    skills:              Vec<SkillSummary>,
    tools:               Vec<ToolSummary>,
    context_window:      Option<ContextWindowSnapshot>,
    committed_event_seq: u64,
}

impl CodingAgentSnapshot {
    /// The durable session identifier.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The root event-stream identifier.
    #[must_use]
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// What the session was doing when captured.
    #[must_use]
    pub const fn state(&self) -> CodingAgentState {
        self.state
    }

    /// The resolved provider identifier.
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The resolved model identifier.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The coding harness selected for the model.
    #[must_use]
    pub const fn profile(&self) -> AgentProfileKind {
        self.profile
    }

    /// The committed conversation.
    #[must_use]
    pub const fn history(&self) -> &History {
        &self.history
    }

    /// Input waiting outside the committed conversation.
    #[must_use]
    pub const fn pending_input(&self) -> &PendingInput {
        &self.pending_input
    }

    /// Memory files loaded into the system prompt.
    #[must_use]
    pub fn memory(&self) -> &[MemoryFileSummary] {
        &self.memory
    }

    /// Skills available to the session.
    #[must_use]
    pub fn skills(&self) -> &[SkillSummary] {
        &self.skills
    }

    /// Tools registered for the session, ordered by name.
    #[must_use]
    pub fn tools(&self) -> &[ToolSummary] {
        &self.tools
    }

    /// The latest context-window measurement.
    #[must_use]
    pub const fn context_window(&self) -> Option<&ContextWindowSnapshot> {
        self.context_window.as_ref()
    }

    /// The last event committed before this snapshot was returned.
    #[must_use]
    pub const fn committed_event_seq(&self) -> u64 {
        self.committed_event_seq
    }
}

/// A coherent starting point for a live coding-agent projection.
pub struct CodingAgentObservation {
    snapshot: CodingAgentSnapshot,
    events:   broadcast::Receiver<CodingAgentEvent>,
}

impl CodingAgentObservation {
    /// The complete state at the observation boundary.
    #[must_use]
    pub const fn snapshot(&self) -> &CodingAgentSnapshot {
        &self.snapshot
    }

    /// Consumes the observation into its snapshot and live event receiver.
    #[must_use]
    pub fn into_parts(self) -> (CodingAgentSnapshot, broadcast::Receiver<CodingAgentEvent>) {
        (self.snapshot, self.events)
    }
}

impl fmt::Debug for CodingAgentObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodingAgentObservation")
            .field("snapshot", &self.snapshot)
            .finish_non_exhaustive()
    }
}

/// What a builder resumes from, when it resumes at all.
enum ResumeSource {
    Record(SessionRecord, ResumeMode),
    Export(Box<WarmState>),
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

    /// Adds one tool middleware to this agent and its descendants.
    ///
    /// Middleware runs in installation order around tool calls. The first
    /// middleware added here is the outermost application layer.
    pub fn tool_middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.inner = self.inner.tool_middleware(middleware);
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

    /// Records the whole session tree's ordered event stream durably.
    ///
    /// The sink receives every event before live subscribers do. A refusal
    /// closes the tree and fails the operation that observed it.
    pub fn event_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.inner = self.inner.event_sink(sink);
        self
    }

    /// Sets the pending event queue and each live subscription's capacity.
    ///
    /// Filling the producer queue is fatal to the stream. A live subscriber
    /// that falls behind instead receives `RecvError::Lagged`.
    pub fn event_capacity(mut self, capacity: impl Into<EventCapacity>) -> Self {
        self.inner = self.inner.event_capacity(capacity);
        self
    }

    /// Sets the longest one call to the durable event sink may take.
    ///
    /// Exceeding this limit stops the stream and closes the session tree.
    pub fn event_sink_timeout(mut self, timeout: impl Into<EventSinkTimeout>) -> Self {
        self.inner = self.inner.event_sink_timeout(timeout);
        self
    }

    /// Configures the subagents this agent may spawn.
    ///
    /// Pebble builds the children itself. A child acts through this agent's
    /// environment, runs on its model under the same tool middleware,
    /// writes events to this tree's shared stream, and inherits only the
    /// application tools marked
    /// [`allow_in_subagents`](RegisteredTool::allow_in_subagents) — never one
    /// that [`requires_human_input`](RegisteredTool::requires_human_input).
    /// Without this call, or with [`SubagentOptions::disabled`], no subagent
    /// tools are advertised.
    pub fn subagents(mut self, options: SubagentOptions) -> Self {
        self.inner = self.inner.subagents(options);
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
                let mut inner = CodingRuntime::from_warm_state(*state, self.inner)?;
                if let Err(source) = inner.start_from_warm_state().await {
                    let _ = inner.shutdown(ShutdownReason::Error).await;
                    return Err(CodingAgentBuildError::Initialization {
                        source: Box::new(source),
                    });
                }
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

/// What a control handle reaches: generic agent control and terminal
/// cancellation.
///
/// Whether a prompt is running and whether the agent is closed are read from
/// the session's own control rather than tracked again here.
struct CodingControl {
    session:          AgentControlHandle,
    cancel:           CancellationToken,
    interrupt_reason: InterruptReasonHandle,
    state:            StateMachine,
    compaction:       CompactionControl,
}

/// Input submitted through a coding agent's main prompt operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CodingInput {
    content: InputContent,
    source:  InputSource,
}

impl CodingInput {
    /// Creates input from provider-neutral content parts.
    #[must_use]
    pub fn new(content: impl IntoIterator<Item = ContentPart>) -> Self {
        Self {
            content: InputContent::new(content),
            source:  InputSource::Prompt,
        }
    }

    /// Creates plain text input.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: InputContent::text(text),
            source:  InputSource::Prompt,
        }
    }

    /// Records where this input came from.
    #[must_use]
    pub const fn with_source(mut self, source: InputSource) -> Self {
        self.source = source;
        self
    }

    /// The ordered content sent to the model.
    #[must_use]
    pub const fn content(&self) -> &InputContent {
        &self.content
    }

    /// Where this input came from.
    #[must_use]
    pub const fn source(&self) -> InputSource {
        self.source
    }
}

impl From<&str> for CodingInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<String> for CodingInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&String> for CodingInput {
    fn from(text: &String) -> Self {
        Self::text(text)
    }
}

impl From<InputContent> for CodingInput {
    fn from(content: InputContent) -> Self {
        Self {
            content,
            source: InputSource::Prompt,
        }
    }
}

impl From<Vec<ContentPart>> for CodingInput {
    fn from(content: Vec<ContentPart>) -> Self {
        Self::new(content)
    }
}

impl CodingControl {
    fn new(session: &CodingRuntime) -> Arc<Self> {
        Arc::new(Self {
            session:          session.control_handle(),
            cancel:           session.cancel_token(),
            interrupt_reason: session.interrupt_reason_handle(),
            state:            session.state_machine(),
            compaction:       session.compaction_control(),
        })
    }
}

/// Out-of-band input for a coding agent: what to say, and who said it.
///
/// Steering reaches the model as its own turn at the next round boundary and
/// publishes [`SteeringInjected`](crate::events::CodingEvent::SteeringInjected)
/// with the same author. A follow-up runs as ordinary user input once the
/// current prompt reaches an answer.
#[derive(Clone, Debug, PartialEq)]
pub struct SteeringMessage {
    content: InputContent,
    actor:   Option<Actor>,
}

impl SteeringMessage {
    /// A message with no named author.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content: InputContent::text(text),
            actor:   None,
        }
    }

    /// A message with provider-neutral content and no named author.
    #[must_use]
    pub fn from_content(content: impl IntoIterator<Item = ContentPart>) -> Self {
        Self {
            content: InputContent::new(content),
            actor:   None,
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
        self.content.text_content()
    }

    /// The ordered content sent to the model.
    #[must_use]
    pub const fn content(&self) -> &InputContent {
        &self.content
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
            content: InputContent::new(message.content().iter().cloned()),
            actor:   actor_from_attribution(message.attribution()),
        }
    }
}

/// Input waiting outside the committed conversation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PendingInput {
    steering:   Vec<SteeringMessage>,
    follow_ups: Vec<SteeringMessage>,
}

impl PendingInput {
    fn from_agent(input: AgentPendingInput) -> Self {
        let (steering, follow_ups) = input.into_parts();
        Self {
            steering:   steering
                .iter()
                .map(SteeringMessage::from_user_message)
                .collect(),
            follow_ups: follow_ups
                .iter()
                .map(SteeringMessage::from_user_message)
                .collect(),
        }
    }

    /// Steering waiting for the next turn boundary, oldest first.
    #[must_use]
    pub fn steering(&self) -> &[SteeringMessage] {
        &self.steering
    }

    /// Follow-up input waiting for a natural answer, oldest first.
    #[must_use]
    pub fn follow_ups(&self) -> &[SteeringMessage] {
        &self.follow_ups
    }

    /// Whether neither queue contains input.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steering.is_empty() && self.follow_ups.is_empty()
    }

    /// Consumes the snapshot and returns both queues, oldest first.
    #[must_use]
    pub fn into_parts(self) -> (Vec<SteeringMessage>, Vec<SteeringMessage>) {
        (self.steering, self.follow_ups)
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

impl From<Vec<ContentPart>> for SteeringMessage {
    fn from(content: Vec<ContentPart>) -> Self {
        Self::from_content(content)
    }
}

/// What the control handle did with one message.
#[derive(Clone, Debug, PartialEq)]
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
    compacting:         bool,
    pending_steering:   usize,
    pending_follow_ups: usize,
}

impl ControlSnapshot {
    /// Whether a prompt is running.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    /// Whether the agent is permanently closed.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// Whether the prompt is parked after an interrupt, waiting for a steer.
    ///
    /// Never `true` while [`pending_steering`](Self::pending_steering) is
    /// above zero: queued steering is what a park waits for.
    #[must_use]
    pub const fn is_parked(&self) -> bool {
        self.parked
    }

    /// Whether conversation history is being compacted.
    #[must_use]
    pub const fn is_compacting(&self) -> bool {
        self.compacting
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
/// including from several handles at once. The generic control state updates
/// each queue and interrupt generation under one lock.
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
    /// can report it.
    pub const STEERING_QUEUE_CAPACITY: usize = 64;

    /// How many follow-up messages wait for a natural answer before the oldest
    /// is dropped.
    pub const FOLLOW_UP_QUEUE_CAPACITY: usize = 64;

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
        SteeringOutcome::from_queue(self.control.session.enqueue_steering_bounded(
            steering_message(message.content, message.actor),
            Self::STEERING_QUEUE_CAPACITY,
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
        if self.is_closed() {
            return SteeringOutcome::Closed;
        }
        let message = message.into();
        SteeringOutcome::from_queue(self.control.session.steer_bounded(
            steering_message(message.content, message.actor),
            Self::STEERING_QUEUE_CAPACITY,
        ))
    }

    /// Interrupts the round in progress without saying what comes next.
    ///
    /// The prompt abandons its round, publishes one
    /// [`RoundInterrupted`](crate::events::CodingEvent::RoundInterrupted), and
    /// parks at the next boundary until a steer arrives or the prompt is
    /// cancelled. With steering already queued the prompt does not park: the
    /// queued message opens the next round, so a steer followed by an
    /// interrupt needs no second gesture. Returns whether a prompt was running
    /// to interrupt.
    pub fn interrupt(&self) -> bool {
        if !self.is_running() || self.is_closed() {
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
        SteeringLease::acquire(&self.control.session)
    }

    /// Queues input to run as its own user turn once the current prompt
    /// reaches an answer.
    ///
    /// A follow-up is ordinary input, not steering, and does not interrupt.
    /// The author is not recorded on the turn it becomes. A full queue evicts
    /// its oldest message and returns it.
    pub fn queue_follow_up(&self, message: impl Into<SteeringMessage>) -> SteeringOutcome {
        if self.is_closed() {
            return SteeringOutcome::Closed;
        }
        let message = message.into();
        SteeringOutcome::from_queue(self.control.session.follow_up_bounded(
            input_message(message.content, InputSource::FollowUp),
            Self::FOLLOW_UP_QUEUE_CAPACITY,
        ))
    }

    /// Aborts the active prompt and leaves the agent ready for another.
    ///
    /// The loop unwinds through its cancellation checkpoints and keeps tool
    /// calls paired with results. Returns whether a prompt was running.
    pub fn abort(&self) -> bool {
        if self.is_closed() || !self.control.session.abort() {
            return false;
        }
        self.control
            .interrupt_reason
            .record(InterruptReason::Cancelled);
        true
    }

    /// Permanently closes the agent from an out-of-band handle.
    ///
    /// Active work is cancelled immediately. The task that owns the
    /// [`CodingAgent`] must still call [`CodingAgent::shutdown`] to publish the
    /// terminal event and join owned tasks.
    pub fn close(&self) -> bool {
        if !self.control.session.close() {
            return false;
        }
        self.control
            .interrupt_reason
            .record(InterruptReason::Cancelled);
        self.control.state.transition(CodingAgentState::Closed);
        self.control.cancel.cancel();
        true
    }

    /// Cancels the compaction currently in progress.
    ///
    /// Cancelling automatic compaction leaves the surrounding prompt running.
    /// Cancelling manual compaction ends that operation with
    /// [`Error::Interrupted`] and leaves the agent reusable.
    pub fn cancel_compaction(&self) -> bool {
        self.control.compaction.cancel()
    }

    /// Waits until no prompt or compaction is running.
    pub async fn wait_for_idle(&self) {
        loop {
            self.control.session.wait_for_idle().await;
            self.control.compaction.wait_for_idle().await;
            if !self.is_running() && !self.is_compacting() {
                return;
            }
        }
    }

    /// Whether a prompt is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.control.session.is_running()
    }

    /// Whether the agent is permanently closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.control.session.is_closed() || self.control.cancel.is_cancelled()
    }

    /// Whether conversation history is being compacted.
    #[must_use]
    pub fn is_compacting(&self) -> bool {
        self.control.compaction.is_active()
    }

    /// A read-only view of the agent's state and queued input.
    #[must_use]
    pub fn snapshot(&self) -> ControlSnapshot {
        let snapshot = self.control.session.snapshot();
        ControlSnapshot {
            running:            snapshot.is_running(),
            closed:             self.is_closed(),
            parked:             snapshot.is_paused(),
            compacting:         self.is_compacting(),
            pending_steering:   snapshot.pending_steering(),
            pending_follow_ups: snapshot.pending_follow_ups(),
        }
    }

    /// Clones the input currently waiting in both queues.
    #[must_use]
    pub fn pending_input(&self) -> PendingInput {
        PendingInput::from_agent(self.control.session.pending_input())
    }

    /// Removes and returns the input currently waiting in both queues.
    ///
    /// If a steer already interrupted the active round, removing it can leave
    /// the prompt parked at its next boundary. Abort the prompt first when the
    /// application is restoring queued input to an editor.
    pub fn take_pending_input(&self) -> PendingInput {
        PendingInput::from_agent(self.control.session.take_pending_input())
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
    /// If the event sink and record were not saved in one transaction, first
    /// reconcile the record with
    /// [`SessionRecord::advance_event_cursor`](crate::state::SessionRecord::advance_event_cursor).
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
    /// a different human-input provider or different tool middleware for its
    /// own life. It runs on the exported route; a change of model goes
    /// through [`resume`](Self::resume). The successor uses a new event
    /// pump but keeps the stream identity and continues numbering from the
    /// export.
    pub fn resume_from_export(
        client: Client,
        environment: Arc<dyn Environment>,
        export: CodingAgentExport,
    ) -> CodingAgentBuilder {
        let mut builder = CodingAgentBuilder::new(client, environment);
        builder.resume = Some(ResumeSource::Export(Box::new(export.inner)));
        builder
    }

    /// Processes one user prompt and every queued follow-up to completion.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SessionClosed`] after shutdown,
    /// [`Error::Interrupted`] after cancellation or timeout, and the
    /// applicable model, tool, compaction, or event failure otherwise.
    pub async fn prompt(&mut self, input: impl Into<CodingInput>) -> Result<PromptOutcome, Error> {
        self.prompt_inner(input, None).await
    }

    /// Processes one user prompt until it completes or `cancel` fires.
    ///
    /// Cancelling ends this prompt alone. The loop unwinds through its
    /// checkpoints, so every tool call the model made still gets its result and
    /// history stays paired: a running call is cancelled through its token and
    /// keeps the result it returns, and a call that has not started yet — the
    /// cancellation landed while the assistant turn was being committed or
    /// compacted — is answered `Cancelled` without running. The prompt returns
    /// [`Error::Interrupted`] and the agent returns to
    /// [`Idle`](CodingAgentState::Idle), ready for the next prompt. Only
    /// [`shutdown`](Self::shutdown) or
    /// [`close`](CodingAgentControlHandle::close) closes the agent.
    ///
    /// # Errors
    ///
    /// As [`prompt`](Self::prompt).
    pub async fn prompt_with_cancellation(
        &mut self,
        input: impl Into<CodingInput>,
        cancel: &CancellationToken,
    ) -> Result<PromptOutcome, Error> {
        self.prompt_inner(input, Some(cancel)).await
    }

    /// Replaces older conversation turns with a model-generated summary.
    ///
    /// This operation requires an idle agent. Use
    /// [`compact_with_cancellation`](Self::compact_with_cancellation) with a
    /// caller token, or [`CodingAgentControlHandle::cancel_compaction`] from
    /// another task to stop it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] while a prompt is active,
    /// [`Error::Interrupted`] when cancelled, and the applicable model,
    /// compaction, or event-stream error otherwise.
    pub async fn compact(
        &mut self,
        options: CompactionOptions,
    ) -> Result<CompactionOutcome, Error> {
        self.inner.compact(options, None).await
    }

    /// Replaces older turns unless `cancel` fires first.
    ///
    /// Cancellation leaves history unchanged and the agent reusable.
    ///
    /// # Errors
    ///
    /// As [`compact`](Self::compact).
    pub async fn compact_with_cancellation(
        &mut self,
        options: CompactionOptions,
        cancel: &CancellationToken,
    ) -> Result<CompactionOutcome, Error> {
        self.inner.compact(options, Some(cancel)).await
    }

    async fn prompt_inner(
        &mut self,
        input: impl Into<CodingInput>,
        cancel: Option<&CancellationToken>,
    ) -> Result<PromptOutcome, Error> {
        let text = self.inner.prompt_with_cancellation(input, cancel).await?;
        let final_message = self.inner.final_assistant_message();
        Ok(PromptOutcome {
            text,
            final_message,
            usage: self.inner.last_prompt_usage(),
            cost_usd_micros: self.inner.last_prompt_cost_usd_micros(),
            timing: self.inner.last_prompt_timing(),
        })
    }

    /// Watches events published after this call.
    ///
    /// This path is bounded and can lag. Use an [`EventSink`] for a complete
    /// record. The receiver ends after [`shutdown`](Self::shutdown) returns,
    /// even while this agent value is still alive.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<CodingAgentEvent> {
        self.inner.subscribe()
    }

    /// Captures the state currently available to an application view.
    ///
    /// Use [`observe`](Self::observe) when the snapshot and a live event
    /// receiver must share one committed boundary.
    #[must_use]
    pub fn snapshot(&self) -> CodingAgentSnapshot {
        self.snapshot_at(self.committed_event_seq())
    }

    /// Starts a coherent live observation of this agent.
    ///
    /// The receiver is installed before the event pipeline is flushed. The
    /// returned snapshot then records that flush's committed sequence. A
    /// consumer starts with the snapshot, discards queued events at or below
    /// its sequence, and applies later events in order.
    ///
    /// # Errors
    ///
    /// Returns an event-stream error when the durability barrier fails. As
    /// with [`flush_events`](Self::flush_events), such a failure closes the
    /// agent.
    pub async fn observe(&mut self) -> Result<CodingAgentObservation, Error> {
        let events = self.subscribe();
        let committed_event_seq = self.flush_events().await?;
        Ok(CodingAgentObservation {
            snapshot: self.snapshot_at(committed_event_seq),
            events,
        })
    }

    fn snapshot_at(&self, committed_event_seq: u64) -> CodingAgentSnapshot {
        CodingAgentSnapshot {
            session_id: self.inner.id().to_owned(),
            stream_id: self.inner.root_session_id().to_owned(),
            state: self.inner.state(),
            provider: self.inner.provider().to_owned(),
            model: self.inner.model().to_owned(),
            profile: self.inner.profile_kind(),
            history: self.inner.history(),
            pending_input: self.control_handle().pending_input(),
            memory: self.inner.memory_summaries().to_vec(),
            skills: self.inner.skill_summaries(),
            tools: self.inner.tool_summaries(),
            context_window: self.inner.context_window(),
            committed_event_seq,
        }
    }

    /// Waits until every event currently queued has reached the durable sink.
    ///
    /// The returned value is the highest committed sequence number at this
    /// barrier. A sink failure closes the agent and is returned here.
    pub async fn flush_events(&mut self) -> Result<u64, Error> {
        match self.inner.flush_events().await {
            Ok(seq) => Ok(seq),
            Err(error) => {
                // This is a public operation boundary. Finish the close that
                // the failed pipeline began before returning.
                let _ = self.inner.shutdown(ShutdownReason::Error).await;
                Err(error)
            }
        }
    }

    /// The highest event sequence committed by the event pipeline.
    ///
    /// With a durable sink, commitment means the sink accepted the event. This
    /// is a committed cursor, not the number most recently queued.
    #[must_use]
    pub fn committed_event_seq(&self) -> u64 {
        self.inner.committed_event_seq()
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

    /// Aborts the active prompt and leaves this agent ready for another.
    pub fn abort(&self) -> bool {
        self.control_handle().abort()
    }

    /// Permanently closes the agent from a shared handle.
    ///
    /// Call [`shutdown`](Self::shutdown) afterwards to join owned tasks.
    pub fn close(&self) -> bool {
        self.control_handle().close()
    }

    /// Waits until no prompt or compaction is running.
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

    /// The stable durable-session and root-stream identifier.
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
    pub fn state(&self) -> CodingAgentState {
        self.inner.state()
    }

    /// A snapshot of the committed coding-agent conversation.
    #[must_use]
    pub fn history(&self) -> History {
        self.inner.history()
    }

    /// Captures the durable session state.
    ///
    /// Take records between prompts, after the prompt's event barrier has
    /// committed the same conversation. Restore one with
    /// [`CodingAgent::resume`].
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
        self.inner.shutdown(reason).await
    }
}

#[cfg(test)]
mod tests {
    use std::result::Result as StdResult;

    use async_trait::async_trait;
    use lithos_llm::types::{ImageContent, MediaSource, Role, ToolDefinition};
    use serde_json::json;
    use tokio::sync::Notify;
    use tokio::time::timeout;

    use super::*;
    use crate::compaction::CompactionReason;
    use crate::error::ErrorKind;
    use crate::event::EventSinkError;
    use crate::runtime::testing::{blocking_tool, drained, wait_for_event};
    use crate::test_support::{
        MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
        message_text, scripted_client, text_delta_events, text_response, tool_call_response,
    };
    use crate::types::{CodingAgentEvent, CodingEvent, ContextWindowStaleness, ToolSource};

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

    fn steering_texts(agent: &CodingAgent) -> Vec<String> {
        agent
            .history()
            .turns()
            .iter()
            .filter_map(|turn| match turn {
                Message::Steering { content, .. } => Some(content.text_content().to_owned()),
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
    async fn rich_input_reaches_events_history_and_the_model() {
        let (client, provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/vision")
            .build()
            .await
            .expect("the coding agent builds");
        let mut events = agent.subscribe();
        let parts = vec![
            ContentPart::Text {
                text: "describe this image".into(),
            },
            ContentPart::Image(ImageContent::new(MediaSource::url_with_media_type(
                "https://example.test/image.png",
                "image/png",
            ))),
        ];
        let expected = InputContent::new(parts.clone());

        agent
            .prompt(CodingInput::new(parts.clone()).with_source(InputSource::External))
            .await
            .expect("the rich prompt succeeds");

        let requests = provider.requests();
        let user = requests[0]
            .messages()
            .iter()
            .find(|message| message.role() == Role::User)
            .expect("the request has user input");
        assert_eq!(user.content(), parts);
        assert!(matches!(
            agent.history().turns().first(),
            Some(Message::User { content, .. }) if content == &expected
        ));
        assert!(drained(&mut events).await.iter().any(|event| matches!(
            event,
            CodingEvent::UserInput {
                text,
                content: Some(content),
                source: InputSource::External,
            } if text == "describe this image" && content == &expected
        )));
        assert_eq!(
            agent
                .snapshot()
                .context_window()
                .expect("the completed prompt was measured")
                .staleness,
            ContextWindowStaleness::Stored
        );
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    #[tokio::test]
    async fn observe_returns_a_snapshot_before_later_events() {
        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .build()
            .await
            .expect("the coding agent builds");

        let observation = agent.observe().await.expect("observation starts");
        let (snapshot, mut events) = observation.into_parts();
        assert_eq!(snapshot.session_id(), agent.id());
        assert_eq!(snapshot.model(), "model");
        assert_eq!(snapshot.state(), CodingAgentState::Idle);
        assert!(snapshot.history().is_empty());
        assert!(snapshot.pending_input().is_empty());
        assert!(!snapshot.tools().is_empty());
        assert_eq!(snapshot.committed_event_seq(), agent.committed_event_seq());

        agent.prompt("work").await.expect("the prompt succeeds");
        let first_later = timeout(PATIENCE, events.recv())
            .await
            .expect("a later event arrives")
            .expect("the event stream stays open");
        assert!(first_later.seq > snapshot.committed_event_seq());
        assert!(matches!(first_later.event, CodingEvent::UserInput { .. }));
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    struct RefusingSink;

    #[async_trait]
    impl EventSink for RefusingSink {
        async fn record(&self, _event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
            Err(EventSinkError::new("the event store is unavailable"))
        }
    }

    #[tokio::test]
    async fn build_waits_for_initialization_events_to_be_recorded() {
        let (client, _provider) = scripted_client(Vec::new());
        let result = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .event_sink(Arc::new(RefusingSink))
            .build()
            .await;

        let Err(CodingAgentBuildError::Initialization { source }) = result else {
            panic!("a sink failure during initialization must fail the build");
        };
        assert_eq!(source.kind(), ErrorKind::EventStream);
    }

    #[tokio::test]
    async fn control_handle_aborts_only_the_active_prompt() {
        let (client, provider) = scripted_client(vec![
            ScriptedCall::PendingOpen,
            ScriptedCall::response(text_response("done")),
        ]);
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
            assert!(!control.is_closed());
        };
        let (result, ()) = tokio::join!(prompting, controlling);

        assert!(matches!(
            result,
            Err(Error::Interrupted(InterruptReason::Cancelled))
        ));
        assert_eq!(session.state(), CodingAgentState::Idle);
        assert_eq!(
            session
                .prompt("try again")
                .await
                .expect("the agent remains reusable")
                .text(),
            Some("done")
        );
        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
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
    async fn pending_input_can_be_inspected_and_taken() {
        let mut agent = agent_with(Vec::new(), []).await;
        let control = agent.control_handle();
        let actor = Actor::User {
            id:           Some("u_1".into()),
            display_name: Some("Ada".into()),
        };

        assert!(
            control
                .queue_steering(SteeringMessage::new("steer").with_actor(actor.clone()))
                .is_accepted()
        );
        assert!(control.queue_follow_up("follow up").is_accepted());

        let pending = control.pending_input();
        assert_eq!(pending.steering()[0].text(), "steer");
        assert_eq!(pending.steering()[0].actor(), Some(&actor));
        assert_eq!(pending.follow_ups()[0].text(), "follow up");
        assert_eq!(control.snapshot().pending_steering(), 1);
        assert_eq!(control.snapshot().pending_follow_ups(), 1);

        let taken = control.take_pending_input();
        assert_eq!(taken, pending);
        assert!(control.pending_input().is_empty());
        assert_eq!(control.snapshot().pending_steering(), 0);
        assert_eq!(control.snapshot().pending_follow_ups(), 0);
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
    async fn a_full_follow_up_queue_evicts_the_oldest_and_reports_it() {
        let mut agent = agent_with(vec![ScriptedCall::response(text_response("done"))], []).await;
        let control = agent.control_handle();
        let capacity = CodingAgentControlHandle::FOLLOW_UP_QUEUE_CAPACITY;
        for index in 0..capacity {
            assert_eq!(
                control.queue_follow_up(format!("follow up {index}")),
                SteeringOutcome::Accepted
            );
        }

        let outcome = control.queue_follow_up("one too many");

        assert_eq!(
            outcome,
            SteeringOutcome::Evicted(SteeringMessage::new("follow up 0"))
        );
        assert_eq!(control.snapshot().pending_follow_ups(), capacity);
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
    async fn an_interrupt_with_steering_queued_never_reports_a_parked_snapshot() {
        let mut agent = agent_with(
            vec![
                ScriptedCall::response(tool_call_response("slow_tool", "call_1", json!({}))),
                ScriptedCall::response(text_response("steered")),
            ],
            [blocking_tool("slow_tool")],
        )
        .await;
        let control = agent.control_handle();
        let mut watched = agent.subscribe();
        let mut recorded = agent.subscribe();

        let controller = tokio::spawn(async move {
            wait_for_event(&mut watched, |event| {
                matches!(event, CodingEvent::ToolCallStarted { .. })
            })
            .await;
            assert_eq!(
                control.queue_steering("change course"),
                SteeringOutcome::Accepted
            );
            assert!(control.interrupt(), "a prompt was running to interrupt");
            let snapshot = control.snapshot();
            assert!(
                !snapshot.is_parked(),
                "queued steering keeps the prompt from parking: {snapshot:?}"
            );
            assert_eq!(snapshot.pending_steering(), 1);
        });

        let outcome = timeout(PATIENCE, agent.prompt("start"))
            .await
            .expect("the queued steer resumes the prompt without a second gesture")
            .expect("the prompt succeeds");
        controller.await.expect("the controller finishes");

        assert_eq!(outcome.text(), Some("steered"));
        assert_eq!(steering_texts(&agent), ["change course"]);
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
        let mut events = agent.subscribe();
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
        let sources: Vec<_> = drained(&mut events)
            .await
            .iter()
            .filter_map(|event| match event {
                CodingEvent::UserInput { source, .. } => Some(*source),
                _ => None,
            })
            .collect();
        assert_eq!(sources, vec![InputSource::Prompt, InputSource::FollowUp]);
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    // --- Compaction ---

    #[tokio::test]
    async fn manual_compaction_returns_and_stores_structured_metadata() {
        let provider = ScriptedProvider::new(vec![
            ScriptedCall::response(text_response("first answer")),
            ScriptedCall::response(text_response("second answer")),
        ])
        .completing(vec![ScriptedCompletion::response(text_response(
            "Work completed and work remaining.",
        ))]);
        let (client, provider) = client_from(provider);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .build()
            .await
            .expect("the coding agent builds");
        agent
            .prompt("first task")
            .await
            .expect("first prompt succeeds");
        agent
            .prompt("second task")
            .await
            .expect("second prompt succeeds");
        let history_before_cancel = agent.history();
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(matches!(
            agent
                .compact_with_cancellation(CompactionOptions::new().preserve_turns(1), &cancelled,)
                .await,
            Err(Error::Interrupted(InterruptReason::Cancelled))
        ));
        assert_eq!(provider.completion_count(), 0);
        assert_eq!(agent.history(), history_before_cancel);
        let mut events = agent.subscribe();

        let outcome = agent
            .compact(
                CompactionOptions::new()
                    .preserve_turns(1)
                    .instructions("Keep the failing test name."),
            )
            .await
            .expect("manual compaction succeeds");
        let CompactionOutcome::Compacted(result) = outcome else {
            panic!("there was history to compact");
        };

        assert_eq!(result.reason(), CompactionReason::Manual);
        assert!(result.original_turn_count() >= 4);
        assert!(result.summary().contains("Work completed"));
        assert!(agent.history().turns().iter().any(|turn| matches!(
            turn,
            Message::Compaction {
                reason: CompactionReason::Manual,
                original_turn_count,
                ..
            } if *original_turn_count == result.original_turn_count()
        )));
        let requests = provider.completion_requests();
        let system = requests[0]
            .messages()
            .iter()
            .find(|message| message.role() == Role::System)
            .expect("the summary has system instructions");
        assert!(message_text(system).contains("Keep the failing test name."));
        let published = drained(&mut events).await;
        assert!(
            published
                .iter()
                .any(|event| matches!(event, CodingEvent::CompactionStarted {
                    reason: CompactionReason::Manual,
                    ..
                }))
        );
        assert!(
            published
                .iter()
                .any(|event| matches!(event, CodingEvent::CompactionCompleted {
                    reason: CompactionReason::Manual,
                    ..
                }))
        );
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the agent shuts down");
    }

    #[tokio::test]
    async fn a_control_handle_cancels_manual_compaction() {
        let provider = ScriptedProvider::new(vec![
            ScriptedCall::response(text_response("first answer")),
            ScriptedCall::response(text_response("second answer")),
        ])
        .completing(vec![ScriptedCompletion::Pending]);
        let (client, _provider) = client_from(provider);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .build()
            .await
            .expect("the coding agent builds");
        agent
            .prompt("first task")
            .await
            .expect("first prompt succeeds");
        agent
            .prompt("second task")
            .await
            .expect("second prompt succeeds");
        let history_before = agent.history();
        let control = agent.control_handle();
        let mut events = agent.subscribe();

        let controller = async {
            wait_for_event(&mut events, |event| {
                matches!(event, CodingEvent::CompactionStarted { .. })
            })
            .await;
            assert!(control.is_compacting());
            assert!(control.cancel_compaction());
            wait_for_event(&mut events, |event| {
                matches!(event, CodingEvent::CompactionCancelled {
                    reason: CompactionReason::Manual,
                })
            })
            .await;
            control.wait_for_idle().await;
            assert!(!control.is_compacting());
        };
        let (result, ()) = tokio::join!(
            agent.compact(CompactionOptions::new().preserve_turns(1)),
            controller
        );

        assert!(matches!(
            result,
            Err(Error::Interrupted(InterruptReason::Cancelled))
        ));
        assert_eq!(agent.state(), CodingAgentState::Idle);
        assert_eq!(agent.history(), history_before);
        assert_eq!(
            agent
                .prompt("try again")
                .await
                .expect("the agent remains reusable")
                .text(),
            Some("second answer")
        );
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
            "only close or shutdown closes the agent"
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

    #[tokio::test]
    async fn close_is_the_terminal_control_gesture() {
        let mut agent = agent_with(Vec::new(), []).await;
        let control = agent.control_handle();

        assert!(control.close());
        assert!(!control.close());
        assert!(control.is_closed());
        assert_eq!(agent.state(), CodingAgentState::Closed);
        assert!(matches!(
            agent.prompt("late").await,
            Err(Error::SessionClosed)
        ));
        assert!(
            agent
                .shutdown(ShutdownReason::Cancelled)
                .await
                .expect("the terminal event is published")
        );
    }
}
