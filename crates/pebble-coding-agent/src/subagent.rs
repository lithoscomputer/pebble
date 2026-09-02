//! The supervisor a session delegates work to.
//!
//! A session that was given subagent support hands spawned tasks to a
//! [`SubagentSupervisor`], asks it at every input boundary whether a background
//! child has finished, and closes its children before it ends itself.
//!
//! # What a child is built from
//!
//! Pebble builds each child itself, from the parent's own [`ChildDeps`] — the
//! same client, model, harness, environment, policy, and the tools marked
//! inheritable — so a child inherits what its parent had and cannot be given
//! more. Nothing outside the crate takes part in building one, and no
//! [`HumanInputProvider`](crate::extensions::HumanInputProvider) travels down,
//! which is what makes questions root-only.
//!
//! # How many sessions a tree may hold open
//!
//! [`SubagentLimits`] bounds the whole tree, counting the root. The counter is
//! created once by the root session and inherited through every spec, so a
//! grandchild competes for the same slots as its uncle. A slot is taken when a
//! child is spawned and released when that child reaches
//! [`Closed`](SubagentStatus::Closed) — a finished child that could still take
//! another turn holds its slot, because its session is still open.
//!
//! # What the parent sees
//!
//! Lifecycle facts about a child — spawned, turn started, completed, failed,
//! closed — are published on the *parent's* stream, because they are the
//! parent's news. Each child writes its own events directly to the tree's one
//! shared pipeline. Those events keep the child's `session_id`, name its
//! immediate parent in `parent_session_id`, and take the next shared sequence
//! number.

mod tools;

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::result::Result as StdResult;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};
use std::time::Duration;

use futures_util::future::join_all;
use lithos_llm::Client;
use pebble_agent::AgentControlHandle;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{AbortHandle, JoinHandle};
use tokio::time::{Instant, timeout_at};
use tokio_util::sync::CancellationToken;
use tracing::warn;

pub(crate) use self::tools::{subagent_tools, tree_position};
use crate::config::CodingAgentOptions;
use crate::environment::Environment;
use crate::error::{Error, ErrorData, ErrorKind, InterruptReason, Result, TaskKind};
use crate::event::Emitter;
use crate::profile::AgentProfile;
use crate::redact::Redactor;
use crate::runtime::{CodingAgentBuildError, CodingRuntime, ShutdownReason};
use crate::search::SearchProvider;
use crate::tool::{RegisteredTool, ToolEnvProvider, ToolError};
use crate::types::{CodingAgentState, CodingEvent, INITIAL_SUBAGENT_GENERATION, ToolErrorKind};

/// How long a closing child has to stop on its own before it is aborted.
const SUBAGENT_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// One idle child accepts one next turn. `send_input` reserves this single slot
/// before it makes the agent running, so an agent can never be running with no
/// turn on its way; input for a running agent goes to the follow-up queue
/// instead.
const SUBAGENT_COMMAND_CAPACITY: usize = 1;

/// How many sessions one tree may hold open at once, counting the root.
const DEFAULT_MAX_OPEN_SESSIONS: usize = 4;

/// How many sessions one tree of agents may hold open at once.
///
/// The count includes the root session, so the default of four leaves three
/// open children across the whole tree however they are nested. A child that
/// has finished still holds its slot — its session is open, and another turn
/// can be sent to it — and releases it when it is closed.
///
/// There is no depth limit and no limit on how many children a session spawns
/// over its life; what is bounded is how many sessions exist at one moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SubagentLimits {
    /// The most sessions one tree may hold open at once, counting the root.
    pub max_open_sessions: usize,
}

impl SubagentLimits {
    /// Limits that allow `max_open_sessions` open sessions, root included.
    ///
    /// A limit below two allows no children at all, which is a way to keep the
    /// subagent tools registered and answering while refusing every spawn.
    #[must_use]
    pub const fn new(max_open_sessions: usize) -> Self {
        Self { max_open_sessions }
    }
}

impl Default for SubagentLimits {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_OPEN_SESSIONS)
    }
}

/// Whether an agent may spawn subagents, and how many the tree may hold open.
///
/// Pebble builds the children itself, so this is the whole of what an
/// application decides. The default is enabled with the default limits; give it
/// to [`CodingAgentBuilder::subagents`](crate::CodingAgentBuilder::subagents)
/// to turn subagents on.
///
/// ```
/// use pebble_coding_agent::subagents::{SubagentLimits, SubagentOptions};
///
/// let options = SubagentOptions::default().with_limits(SubagentLimits::new(3));
/// assert!(options.is_enabled());
/// assert_eq!(options.limits().max_open_sessions, 3);
/// assert!(!SubagentOptions::disabled().is_enabled());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubagentOptions {
    enabled: bool,
    limits:  SubagentLimits,
}

impl SubagentOptions {
    /// Subagents on, with the default limits.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            limits:  SubagentLimits::default(),
        }
    }

    /// Subagents off: no subagent tools are advertised.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            limits:  SubagentLimits::default(),
        }
    }

    /// The same options, bounding the tree by `limits`.
    #[must_use]
    pub const fn with_limits(mut self, limits: SubagentLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Whether subagents are on.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// How many sessions the tree may hold open at once.
    #[must_use]
    pub const fn limits(&self) -> SubagentLimits {
        self.limits
    }
}

impl Default for SubagentOptions {
    fn default() -> Self {
        Self::enabled()
    }
}

/// The open-session count one tree of agents shares.
///
/// Created by the root session and inherited by every child through its
/// [`ChildIdentity`], so the limit is a property of the tree rather than of
/// one supervisor.
#[derive(Debug)]
pub(crate) struct OpenSessions {
    max:  usize,
    open: AtomicUsize,
}

impl OpenSessions {
    /// The counter a root session starts, already counting itself.
    pub(crate) fn root(limits: SubagentLimits) -> Arc<Self> {
        Arc::new(Self {
            max:  limits.max_open_sessions,
            open: AtomicUsize::new(1),
        })
    }

    /// Takes a slot for one more open session, or answers `None` when the tree
    /// is full.
    fn try_acquire(self: &Arc<Self>) -> Option<SessionSlot> {
        let mut open = self.open.load(Ordering::Relaxed);
        loop {
            if open >= self.max {
                return None;
            }
            match self.open.compare_exchange_weak(
                open,
                open + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(SessionSlot {
                        sessions: Arc::clone(self),
                        released: AtomicBool::new(false),
                    });
                }
                Err(current) => open = current,
            }
        }
    }

    /// How many sessions in this tree are open right now.
    #[cfg(test)]
    fn open(&self) -> usize {
        self.open.load(Ordering::Relaxed)
    }

    /// The most sessions this tree may hold open.
    const fn max(&self) -> usize {
        self.max
    }
}

/// One open session's place in its tree's budget.
///
/// Releasing is idempotent and happens on drop as well as on the close that
/// owns it, so a slot cannot leak however a child ends.
#[derive(Debug)]
struct SessionSlot {
    sessions: Arc<OpenSessions>,
    released: AtomicBool,
}

impl SessionSlot {
    fn release(&self) {
        if !self.released.swap(true, Ordering::AcqRel) {
            self.sessions.open.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for SessionSlot {
    fn drop(&mut self) {
        self.release();
    }
}

/// Observes each child the moment it is built, before it is supervised.
///
/// The crate's own tests keep hold of a child's identity and its supervisor
/// this way. Nothing in production observes children, and an observer cannot
/// change what it is shown: what a child inherits is decided by
/// [`build_child`] alone.
pub(crate) type ChildObserver = Arc<dyn Fn(&CodingRuntime) + Send + Sync>;

/// Builds one child session from its parent's dependencies.
///
/// Everything a child is comes from `deps` and its place in the tree: the
/// parent's client, model, harness, environment, options, and the tools marked
/// inheritable. Nothing else can be handed to it.
fn build_child(
    deps: &Arc<ChildDeps>,
    parent_session_id: String,
    root_session_id: String,
    depth: usize,
) -> StdResult<CodingRuntime, CodingAgentBuildError> {
    // A child is given a task, not the application's integrations: an
    // application tool reaches it only when marked inheritable, and a tool
    // that needs a person never does, because a child has nobody to ask.
    let inherited = deps
        .tools
        .iter()
        .filter(|tool| tool.is_inheritable())
        .cloned();
    let mut builder = CodingRuntime::builder(deps.client.clone())
        .model(deps.model_selector.clone())
        .environment(Arc::clone(&deps.environment))
        .with_profile(Arc::clone(&deps.profile))
        .tools(inherited)
        .options(deps.options.clone())
        .subagents(SubagentOptions::enabled())
        .child_of(ChildIdentity {
            event_emitter: deps.event_emitter.for_child(parent_session_id.clone()),
            parent_session_id,
            root_session_id,
            depth,
            open_sessions: Arc::clone(&deps.open_sessions),
            observer: deps.observer.clone(),
        });
    for middleware in &deps.tool_middleware {
        builder = builder.tool_middleware(Arc::clone(middleware));
    }
    if let Some(provider) = deps.tool_env_provider.as_ref() {
        builder = builder.tool_env_provider(Arc::clone(provider));
    }
    builder = builder.redactor(Arc::clone(&deps.redactor));
    if let Some(provider) = deps.search_provider.as_ref() {
        builder = builder.search_provider(Arc::clone(provider));
    }
    builder.build()
}

/// What one session gives its children.
///
/// Assembled by the session builder from the session it is building, so every
/// child of that session is built from the same set. Notably absent: the
/// human-input provider, which is why a child can never ask a person a
/// question.
pub(crate) struct ChildDeps {
    pub(crate) client:            Client,
    pub(crate) model_selector:    String,
    pub(crate) profile:           Arc<dyn AgentProfile>,
    pub(crate) environment:       Arc<dyn Environment>,
    pub(crate) tools:             Vec<RegisteredTool>,
    pub(crate) tool_middleware:   Vec<Arc<dyn pebble_agent::ToolMiddleware>>,
    pub(crate) options:           CodingAgentOptions,
    pub(crate) tool_env_provider: Option<Arc<dyn ToolEnvProvider>>,
    /// What strips secrets out of what a child publishes. Inherited, because
    /// a child's process output reaches the same stream its parent's does.
    pub(crate) redactor:          Arc<dyn Redactor>,
    /// Where a child's web searches go. Inherited, because a child researches
    /// the task its parent gave it.
    pub(crate) search_provider:   Option<Arc<dyn SearchProvider>>,
    /// The one lossless pipeline shared by this whole session tree.
    pub(crate) event_emitter:     Emitter,
    /// What sees each child as it is built, for the crate's own tests.
    pub(crate) observer:          Option<ChildObserver>,
    pub(crate) open_sessions:     Arc<OpenSessions>,
    /// The depth of the session these deps belong to. Its children sit one
    /// deeper.
    pub(crate) depth:             usize,
}

/// Where a child sits in its tree, and which budget it spends.
pub(crate) struct ChildIdentity {
    /// This child's view of the tree's shared event pipeline.
    pub(crate) event_emitter:     Emitter,
    pub(crate) parent_session_id: String,
    pub(crate) root_session_id:   String,
    pub(crate) depth:             usize,
    pub(crate) open_sessions:     Arc<OpenSessions>,
    /// What sees this child's own children as they are built, inherited
    /// from the root for the crate's own tests.
    pub(crate) observer:          Option<ChildObserver>,
}

/// Where a supervisor sends its child-lifecycle events.
pub(crate) type SubagentEventCallback = Arc<dyn Fn(CodingEvent) + Send + Sync>;

/// What one turn of a child produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubagentResult {
    /// The child's final answer.
    pub(crate) output:     String,
    /// Whether the turn succeeded.
    pub(crate) success:    bool,
    /// How many turns the child spent on it.
    pub(crate) turns_used: usize,
}

/// A finished background result waiting to be delivered to its parent at a safe
/// turn boundary.
#[derive(Debug, Clone)]
pub(crate) struct SubagentParentNotification {
    pub(crate) agent_id:    String,
    pub(crate) description: String,
    pub(crate) result:      StdResult<SubagentResult, ErrorData>,
}

/// Renders one batch as the text of the parent turn that carries it.
///
/// The envelope is model-visible API: the parent reads it as a user turn, so
/// the shape and the wording are frozen. A failure renders
/// [`ErrorData::message`], which is a whole self-contained line — pebble's
/// projection carries the causes in `source_chain` and renders them into the
/// message already, where fabro walked the live error's source chain here.
fn format_parent_notification_batch(notifications: &[SubagentParentNotification]) -> String {
    notifications
        .iter()
        .map(|notification| {
            let (status, result) = match &notification.result {
                Ok(result) if result.success => {
                    ("completed", Cow::Borrowed(result.output.as_str()))
                }
                Ok(result) => ("failed", Cow::Borrowed(result.output.as_str())),
                Err(error) => ("failed", Cow::Borrowed(error.message.as_str())),
            };
            format!(
                "<task-notification>\n  <task-id>{}</task-id>\n  <status>{status}</status>\n  \
                 <description>{}</description>\n  <result>{}</result>\n</task-notification>",
                escape_notification_xml(&notification.agent_id),
                escape_notification_xml(&notification.description),
                escape_notification_xml(&result),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Escapes the five characters that would otherwise close the envelope early.
fn escape_notification_xml(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

/// Where one child is in its life.
#[derive(Debug, Clone)]
pub(crate) enum SubagentStatus {
    /// A turn is running.
    Running,
    /// The turn ended. `reusable` reports whether the child session survived it
    /// and can start another turn, so a finished-but-spent agent and a
    /// finished-and-ready one cannot be confused. What the turn produced lives
    /// in the results cache, where `wait` answers from.
    Finished {
        /// Whether the child session survived the turn.
        reusable: bool,
    },
    /// The child is being torn down.
    Closing,
    /// The child is gone and its slot is released.
    Closed,
}

/// Start the next turn of an existing child session.
#[derive(Debug)]
struct StartTurn {
    generation: u64,
    prompt:     String,
}

/// Which generations of one child should reach the parent by themselves.
struct ParentNotificationState {
    description:         String,
    pending_generations: VecDeque<u64>,
}

/// One supervised child.
struct SubAgent {
    status:              watch::Sender<SubagentStatus>,
    generation:          u64,
    results:             HashMap<u64, StdResult<SubagentResult, ErrorData>>,
    command_tx:          mpsc::Sender<StartTurn>,
    runner_stop:         CancellationToken,
    cleanup_done:        watch::Sender<bool>,
    monitor_task:        Option<JoinHandle<()>>,
    cleanup_task:        Option<JoinHandle<()>>,
    child_abort_handle:  AbortHandle,
    control:             AgentControlHandle,
    cancel_token:        CancellationToken,
    depth:               usize,
    /// Registration for generations whose results should be delivered to the
    /// parent automatically. The description remains available so a later turn
    /// in the same child session can register its own result.
    ///
    /// Keeping this beside the generation results means a notification cannot
    /// be registered before — or suppressed after — the state it describes:
    /// there is only one lock and one ordering.
    parent_notification: Option<ParentNotificationState>,
    /// Spawn order, so a batch is delivered oldest-first rather than in
    /// whatever order the map happens to iterate.
    spawn_seq:           u64,
    /// This child's place in the tree's open-session budget, released when it
    /// closes or, failing that, when the entry is dropped.
    slot:                Option<SessionSlot>,
}

impl Drop for SubAgent {
    fn drop(&mut self) {
        self.runner_stop.cancel();
        self.cancel_token.cancel();
        self.child_abort_handle.abort();
        if let Some(task) = self.monitor_task.take() {
            task.abort();
        }
        if let Some(task) = self.cleanup_task.take() {
            task.abort();
        }
    }
}

/// Everything one supervisor knows about its children.
#[derive(Default)]
struct SupervisorState {
    agents:             HashMap<String, SubAgent>,
    next_spawn_seq:     u64,
    lifecycle_events:   VecDeque<CodingEvent>,
    lifecycle_draining: bool,
}

impl SupervisorState {
    fn agent(&self, agent_id: &str) -> StdResult<&SubAgent, ToolError> {
        self.agents
            .get(agent_id)
            .ok_or_else(|| unknown_agent(agent_id))
    }

    fn agent_mut(&mut self, agent_id: &str) -> StdResult<&mut SubAgent, ToolError> {
        self.agents
            .get_mut(agent_id)
            .ok_or_else(|| unknown_agent(agent_id))
    }

    fn queue_lifecycle_event(&mut self, event: CodingEvent) {
        self.lifecycle_events.push_back(event);
    }
}

/// The answer to a call naming an agent that was never spawned.
fn unknown_agent(agent_id: &str) -> ToolError {
    ToolError::execution(format!(
        "No agent found with id: {agent_id} (it was never spawned)"
    ))
}

/// The answer to a spawn the tree has no room for.
///
/// A refused spawn is the tool's answer, not the parent's failure: the model
/// reads it, and can close a finished agent and try again.
fn limit_reached(max_open_sessions: usize) -> ToolError {
    ToolError::new(
        ToolErrorKind::Denied,
        format!(
            "Cannot spawn another agent: this session tree already holds the most open sessions \
             it may have ({max_open_sessions}, counting the root). Close an agent that is no \
             longer needed and try again."
        ),
    )
}

/// The answer to a call naming an agent that is closing or closed.
fn already_closed(agent_id: &str) -> ToolError {
    ToolError::execution(format!("Agent {agent_id} has been closed"))
}

/// A child's own failure, reported to whoever was waiting on it.
///
/// The projection's category crosses with it: a child that was interrupted —
/// cancelled, or out of wall-clock time — is reported as
/// [`ToolErrorKind::Cancelled`], and every other failure as
/// [`ToolErrorKind::Execution`]. The message is [`ErrorData::message`]
/// unchanged, which is the whole failure on one line.
///
/// [`ToolErrorKind`] is a narrower vocabulary than [`ErrorKind`], so a caller
/// that needs the exact category reads it from the
/// [`SubAgentFailed`](CodingEvent::SubAgentFailed) event, which carries the
/// projection whole.
fn child_failure(error: ErrorData) -> ToolError {
    let kind = match error.kind {
        ErrorKind::Interrupted => ToolErrorKind::Cancelled,
        _ => ToolErrorKind::Execution,
    };
    ToolError::new(kind, error.message)
}

/// What the one thread that leads a child's shutdown carries out of the lock.
struct ShutdownWork {
    handle:              SubagentHandle,
    generation:          u64,
    close_running_agent: bool,
    status:              watch::Sender<SubagentStatus>,
    cleanup_done:        watch::Sender<bool>,
    monitor_task:        Option<JoinHandle<()>>,
    child_abort_handle:  AbortHandle,
    cancel_token:        CancellationToken,
    runner_stop:         CancellationToken,
    slot:                Option<SessionSlot>,
}

impl Drop for ShutdownWork {
    fn drop(&mut self) {
        self.runner_stop.cancel();
        self.cancel_token.cancel();
        self.child_abort_handle.abort();
        if let Some(task) = self.monitor_task.take() {
            task.abort();
        }
    }
}

/// What one caller of `begin_shutdown` is asked to do.
enum ShutdownDisposition {
    /// This caller runs the shutdown.
    Lead(Box<ShutdownWork>),
    /// Another caller is running it; wait for that.
    Follow(watch::Receiver<bool>),
    /// It has already run.
    Done,
}

/// Flips the cleanup flag however the shutdown ends, so a follower is never
/// left waiting on a leader that failed.
struct CleanupDoneGuard(watch::Sender<bool>);

impl Drop for CleanupDoneGuard {
    fn drop(&mut self) {
        self.0.send_replace(true);
    }
}

/// Wakes anything parked in
/// [`SubagentSupervisor::next_parent_notification_batch`] so it can re-evaluate
/// which children are deliverable.
fn signal_notifications(changed: &watch::Sender<u64>) {
    changed.send_modify(|generation| {
        *generation = generation.wrapping_add(1);
    });
}

/// Clears the draining flag if a callback panics, so one panicking callback
/// cannot silence every later lifecycle event.
///
/// A drain that ends normally clears the flag itself, under the same lock that
/// found the queue empty, and disarms this guard: clearing it twice would let a
/// second drainer start while a third is still running.
struct DrainingGuard<'a> {
    state: &'a Arc<Mutex<SupervisorState>>,
    armed: bool,
}

impl<'a> DrainingGuard<'a> {
    fn new(state: &'a Arc<Mutex<SupervisorState>>) -> Self {
        Self { state, armed: true }
    }

    /// Says the drain has already cleared the flag.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for DrainingGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .lifecycle_draining = false;
    }
}

/// Delivers lifecycle callbacks in the same order as the state transitions that
/// queued them.
///
/// The queue exists for cross-thread ordering: a runner thread that releases
/// the lock after committing one generation would otherwise race a `send_input`
/// thread emitting the next generation's start, and consumers would see the
/// turns out of order. Callbacks run with no lock held, so one may also call
/// back into the supervisor without deadlocking.
fn drain_lifecycle_events(
    state: &Arc<Mutex<SupervisorState>>,
    event_callback: &Arc<RwLock<Option<SubagentEventCallback>>>,
) {
    {
        let mut locked = state.lock().unwrap_or_else(PoisonError::into_inner);
        if locked.lifecycle_draining {
            return;
        }
        locked.lifecycle_draining = true;
    }
    let mut draining = DrainingGuard::new(state);

    loop {
        let event = {
            let mut locked = state.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(event) = locked.lifecycle_events.pop_front() else {
                // Finding the queue empty and clearing the flag are one
                // critical section, so a publisher racing this exit either
                // queues before the clear, and this loop takes the event, or
                // takes the lock after it, and drains the event itself.
                // Clearing the flag first and releasing the lock second would
                // strand that event until an unrelated later publish.
                locked.lifecycle_draining = false;
                draining.disarm();
                return;
            };
            event
        };
        let callback = event_callback
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(callback) = callback {
            callback(event);
        }
    }
}

/// What a committed turn leaves its runner to do.
enum TurnCommit {
    /// A follow-up raced the boundary; run it in the same generation.
    Continue(String),
    /// The generation is committed; wait for the next command.
    Finished,
    /// This runner is stale or the agent is gone; stop.
    Stopping,
}

/// The event one committed generation publishes on the parent's stream.
fn completion_event(
    agent_id: &str,
    depth: usize,
    generation: u64,
    result: &StdResult<SubagentResult, ErrorData>,
) -> CodingEvent {
    match result {
        Ok(result) => CodingEvent::SubAgentCompleted {
            agent_id: agent_id.to_owned(),
            depth,
            generation,
            success: result.success,
            turns_used: result.turns_used,
        },
        Err(error) => CodingEvent::SubAgentFailed {
            agent_id: agent_id.to_owned(),
            depth,
            generation,
            error: error.clone(),
        },
    }
}

/// One child's view of its supervisor: the shared state plus the identity every
/// lifecycle transition needs.
///
/// The state reference is weak because a child task reaches its supervisor
/// through this handle, and a strong reference would close the cycle
/// state -> `SubAgent` -> runner task -> handle.
#[derive(Clone)]
struct SubagentHandle {
    state:                 Weak<Mutex<SupervisorState>>,
    event_callback:        Arc<RwLock<Option<SubagentEventCallback>>>,
    notifications_changed: Arc<watch::Sender<u64>>,
    agent_id:              String,
    depth:                 usize,
}

impl SubagentHandle {
    /// Commits one generation result, or claims a follow-up that raced its
    /// final boundary.
    ///
    /// This is where the runtime error becomes the projection everything else
    /// carries: the results cache, the notification batch, and the failure
    /// event all hold [`ErrorData`], and the error itself is never stored.
    ///
    /// The supervisor state lock is acquired before the follow-up queue lock,
    /// which is also the ordering used by `send_input`.
    fn commit_turn_result(
        &self,
        generation: u64,
        result: &Result<SubagentResult>,
        reusable: bool,
    ) -> TurnCommit {
        let Some(state) = self.state.upgrade() else {
            return TurnCommit::Stopping;
        };
        let projected = match result {
            Ok(result) => Ok(result.clone()),
            Err(error) => Err(ErrorData::from(error)),
        };
        let outcome = {
            let mut locked = state.lock().unwrap_or_else(PoisonError::into_inner);
            let Ok(agent) = locked.agent_mut(&self.agent_id) else {
                return TurnCommit::Stopping;
            };
            if agent.generation != generation
                || !matches!(*agent.status.borrow(), SubagentStatus::Running)
            {
                return TurnCommit::Stopping;
            }

            if reusable {
                let next_prompt = agent
                    .control
                    .take_follow_up()
                    .map(|message| message.text_content());
                if let Some(next_prompt) = next_prompt {
                    return TurnCommit::Continue(next_prompt);
                }
            }

            agent.results.insert(generation, projected.clone());
            agent
                .status
                .send_replace(SubagentStatus::Finished { reusable });
            locked.queue_lifecycle_event(completion_event(
                &self.agent_id,
                self.depth,
                generation,
                &projected,
            ));
            TurnCommit::Finished
        };

        self.publish(&state);
        outcome
    }

    /// The generation this agent is on now, or `None` once the supervisor or
    /// the agent itself is gone.
    fn current_generation(&self) -> Option<u64> {
        let state = self.state.upgrade()?;
        let locked = state.lock().unwrap_or_else(PoisonError::into_inner);
        locked
            .agent(&self.agent_id)
            .ok()
            .map(|agent| agent.generation)
    }

    fn queue_and_publish(&self, event: CodingEvent) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .queue_lifecycle_event(event);
        self.publish(&state);
    }

    /// Wakes notification waiters and delivers queued lifecycle callbacks.
    /// Always called with no supervisor lock held.
    fn publish(&self, state: &Arc<Mutex<SupervisorState>>) {
        signal_notifications(&self.notifications_changed);
        drain_lifecycle_events(state, &self.event_callback);
    }
}

/// Runs one child session for as long as its supervisor keeps giving it turns.
#[tracing::instrument(
    name = "subagent_session",
    skip_all,
    fields(
        session_id = %session.id(),
        agent_id = %handle.agent_id,
        depth = handle.depth
    )
)]
async fn run_subagent_session(
    mut session: CodingRuntime,
    handle: SubagentHandle,
    initial_prompt: String,
    mut command_rx: mpsc::Receiver<StartTurn>,
    runner_stop: CancellationToken,
    start_rx: oneshot::Receiver<()>,
) {
    // The start gate: the supervisor releases it once the agent's entry is in
    // the map, so nothing this runner commits can arrive before there is
    // somewhere to commit it.
    if start_rx.await.is_err() {
        return;
    }

    if let Err(error) = session.initialize().await {
        handle.commit_turn_result(INITIAL_SUBAGENT_GENERATION, &Err(error), false);
        // A session that never initialized has no history worth reusing, so
        // release it and its environment now rather than holding both until the
        // parent closes the agent.
        shutdown_child(&mut session, true).await;
        return;
    }

    let mut command = StartTurn {
        generation: INITIAL_SUBAGENT_GENERATION,
        prompt:     initial_prompt,
    };
    'commands: loop {
        let StartTurn {
            generation,
            mut prompt,
        } = command;
        let generation_start_turns = session.history().turns().len();

        loop {
            let result = session
                .prompt(&prompt)
                .await
                .and_then(|output| {
                    output.ok_or_else(|| {
                        Error::InvalidState(
                            "Subagent completed without a non-empty final response".to_owned(),
                        )
                    })
                })
                .map(|output| SubagentResult {
                    output,
                    success: true,
                    turns_used: session
                        .history()
                        .turns()
                        .len()
                        .saturating_sub(generation_start_turns),
                });
            let reusable =
                session.state() == CodingAgentState::Idle && !session.cancel_token().is_cancelled();
            match handle.commit_turn_result(generation, &result, reusable) {
                TurnCommit::Continue(next_prompt) => prompt = next_prompt,
                TurnCommit::Finished => break,
                TurnCommit::Stopping => break 'commands,
            }
        }

        command = tokio::select! {
            biased;
            () = runner_stop.cancelled() => break,
            command = command_rx.recv() => {
                let Some(command) = command else {
                    break;
                };
                command
            }
        };
    }

    shutdown_child(&mut session, false).await;
}

/// Closes a child session, reporting a shutdown that failed rather than
/// dropping it.
///
/// Cancellation always wins as the reported reason; otherwise a session that
/// failed to start reports an error and one that ran reports completion.
async fn shutdown_child(session: &mut CodingRuntime, failed_to_start: bool) {
    let reason = if session.cancel_token().is_cancelled() {
        ShutdownReason::Cancelled
    } else if failed_to_start {
        ShutdownReason::Error
    } else {
        ShutdownReason::Completed
    };
    if let Err(error) = session.shutdown(reason).await {
        warn!(%error, "A subagent session failed to shut down cleanly");
    }
}

/// Reports a runner that died without committing its own result, so the agent
/// never sits in `Running` with nothing left to run.
fn spawn_runner_monitor(runner_task: JoinHandle<()>, handle: SubagentHandle) -> JoinHandle<()> {
    tokio::spawn(async move {
        let Err(error) = runner_task.await else {
            return;
        };
        let Some(generation) = handle.current_generation() else {
            return;
        };
        let task_result = Err(Error::Task {
            task:   TaskKind::SubagentSession,
            source: error,
        });
        handle.commit_turn_result(generation, &task_result, false);
    })
}

/// One session's children: spawning, background results, and teardown.
///
/// Cloning is cheap and every clone drives the same children, which is what
/// lets a session hold one and hand another to the tools that spawn.
///
/// The internal mutex protects short state transitions only; task waits and
/// callbacks always happen after the guard has been released.
#[derive(Clone)]
pub(crate) struct SubagentSupervisor {
    state:                 Arc<Mutex<SupervisorState>>,
    deps:                  Arc<ChildDeps>,
    event_callback:        Arc<RwLock<Option<SubagentEventCallback>>>,
    notifications_changed: Arc<watch::Sender<u64>>,
}

impl fmt::Debug for SubagentSupervisor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let agents = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .agents
            .len();
        formatter
            .debug_struct("SubagentSupervisor")
            .field("agents", &agents)
            .field("depth", &self.deps.depth)
            .finish_non_exhaustive()
    }
}

impl SubagentSupervisor {
    /// A supervisor whose children are built from `deps`.
    pub(crate) fn new(deps: Arc<ChildDeps>) -> Self {
        Self {
            state: Arc::new(Mutex::new(SupervisorState::default())),
            deps,
            event_callback: Arc::new(RwLock::new(None)),
            notifications_changed: Arc::new(watch::channel(0).0),
        }
    }

    /// A child's view of this supervisor, for the tasks that run that child.
    fn handle(&self, agent_id: String, depth: usize) -> SubagentHandle {
        SubagentHandle {
            state: Arc::downgrade(&self.state),
            event_callback: Arc::clone(&self.event_callback),
            notifications_changed: Arc::clone(&self.notifications_changed),
            agent_id,
            depth,
        }
    }

    /// Wakes notification waiters and delivers queued lifecycle callbacks,
    /// after the state lock has been released.
    fn publish(&self) {
        signal_notifications(&self.notifications_changed);
        drain_lifecycle_events(&self.state, &self.event_callback);
    }

    /// Sends this supervisor's child-lifecycle events to `callback`.
    ///
    /// The session builder installs the session's own callback, so an
    /// application cannot forget to connect them. Each child's own events use
    /// the shared tree pipeline directly and do not pass through this callback.
    pub(crate) fn set_event_callback(&self, callback: SubagentEventCallback) {
        *self
            .event_callback
            .write()
            .unwrap_or_else(PoisonError::into_inner) = Some(callback);
    }

    /// Starts a child on `task_prompt`, answering with its identifier.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] the calling tool gives straight to the model,
    /// because a refused spawn is the tool's answer rather than the session's
    /// failure: the tree may be full, or the factory may have refused to build
    /// the child.
    pub(crate) fn spawn(
        &self,
        parent_session_id: &str,
        root_session_id: &str,
        task_prompt: String,
    ) -> StdResult<String, ToolError> {
        self.spawn_inner(parent_session_id, root_session_id, task_prompt, None)
    }

    /// Starts a child whose terminal result should reach the parent by itself,
    /// at the next input boundary.
    pub(crate) fn spawn_with_parent_notification(
        &self,
        parent_session_id: &str,
        root_session_id: &str,
        task_prompt: String,
        description: String,
    ) -> StdResult<String, ToolError> {
        self.spawn_inner(
            parent_session_id,
            root_session_id,
            task_prompt,
            Some(description),
        )
    }

    fn spawn_inner(
        &self,
        parent_session_id: &str,
        root_session_id: &str,
        task_prompt: String,
        parent_notification_description: Option<String>,
    ) -> StdResult<String, ToolError> {
        // The slot is taken before the child is built, so a tree at its limit
        // never pays for a session it cannot keep.
        let slot = self
            .deps
            .open_sessions
            .try_acquire()
            .ok_or_else(|| limit_reached(self.deps.open_sessions.max()))?;

        let child_depth = self.deps.depth.saturating_add(1);
        let child = build_child(
            &self.deps,
            parent_session_id.to_owned(),
            root_session_id.to_owned(),
            child_depth,
        )
        .map_err(|error| {
            ToolError::with_rendered_source(
                ToolErrorKind::Execution,
                "Could not start a subagent session",
                error,
            )
        })?;
        if let Some(observer) = &self.deps.observer {
            observer(&child);
        }

        Ok(self.supervise(
            child,
            task_prompt,
            child_depth,
            parent_notification_description,
            slot,
        ))
    }

    /// Puts a built child session under supervision and starts its turn.
    fn supervise(
        &self,
        session: CodingRuntime,
        task_prompt: String,
        child_depth: usize,
        parent_notification_description: Option<String>,
        slot: SessionSlot,
    ) -> String {
        let agent_id = format!("{:08x}", uuid::Uuid::new_v4().as_fields().0);
        let control = session.agent_control_handle();
        let cancel_token = session.cancel_token();

        let (start_tx, start_rx) = oneshot::channel();
        let (command_tx, command_rx) = mpsc::channel(SUBAGENT_COMMAND_CAPACITY);
        let runner_stop = CancellationToken::new();
        let handle = self.handle(agent_id.clone(), child_depth);
        let runner_task = tokio::spawn(run_subagent_session(
            session,
            handle.clone(),
            task_prompt.clone(),
            command_rx,
            runner_stop.clone(),
            start_rx,
        ));
        let child_abort_handle = runner_task.abort_handle();
        let monitor_task = spawn_runner_monitor(runner_task, handle);
        let (status, _) = watch::channel(SubagentStatus::Running);
        let (cleanup_done, _) = watch::channel(false);

        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let spawn_seq = state.next_spawn_seq;
            state.next_spawn_seq = state.next_spawn_seq.saturating_add(1);
            let parent_notification =
                parent_notification_description.map(|description| ParentNotificationState {
                    description,
                    pending_generations: VecDeque::from([INITIAL_SUBAGENT_GENERATION]),
                });
            state.agents.insert(agent_id.clone(), SubAgent {
                status,
                generation: INITIAL_SUBAGENT_GENERATION,
                results: HashMap::new(),
                command_tx,
                runner_stop,
                cleanup_done,
                monitor_task: Some(monitor_task),
                cleanup_task: None,
                child_abort_handle,
                control,
                cancel_token,
                depth: child_depth,
                parent_notification,
                spawn_seq,
                slot: Some(slot),
            });
            state.queue_lifecycle_event(CodingEvent::SubAgentSpawned {
                agent_id:   agent_id.clone(),
                depth:      child_depth,
                task:       task_prompt,
                generation: INITIAL_SUBAGENT_GENERATION,
            });
        }
        self.publish();
        // Releasing the gate last is what guarantees the entry exists before
        // the runner can commit anything against it.
        let _ = start_tx.send(());

        agent_id
    }

    /// Gives a child more to do.
    ///
    /// A running child receives the message at its next safe turn boundary; a
    /// finished one that can still run starts a new generation with the history
    /// it already has.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] naming an agent that was never spawned, one that
    /// has been closed, or one whose session ended.
    pub(crate) fn send_input(&self, agent_id: &str, message: &str) -> StdResult<(), ToolError> {
        let resumed = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let agent = state.agent_mut(agent_id)?;
            let status = agent.status.borrow().clone();
            match status {
                SubagentStatus::Running => {
                    if !agent.control.follow_up(message) {
                        return Err(ToolError::execution(format!(
                            "Agent {agent_id} cannot accept more input because its session ended"
                        )));
                    }
                    None
                }
                SubagentStatus::Finished { reusable, .. } => {
                    if !reusable {
                        return Err(ToolError::execution(format!(
                            "Agent {agent_id} cannot accept more input because its session ended"
                        )));
                    }
                    let permit = agent
                        .command_tx
                        .clone()
                        .try_reserve_owned()
                        .map_err(|error| {
                            ToolError::with_rendered_source(
                                ToolErrorKind::Execution,
                                format!("Agent {agent_id} could not start another turn"),
                                error,
                            )
                        })?;
                    let generation = agent.generation.checked_add(1).ok_or_else(|| {
                        ToolError::execution(format!(
                            "Agent {agent_id} exhausted its turn generation"
                        ))
                    })?;
                    agent.generation = generation;
                    agent.status.send_replace(SubagentStatus::Running);
                    if let Some(notification) = agent.parent_notification.as_mut() {
                        notification.pending_generations.push_back(generation);
                    }
                    let depth = agent.depth;
                    state.queue_lifecycle_event(CodingEvent::SubAgentTurnStarted {
                        agent_id: agent_id.to_owned(),
                        depth,
                        task: message.to_owned(),
                        generation,
                    });
                    Some((permit, generation))
                }
                SubagentStatus::Closing | SubagentStatus::Closed => {
                    return Err(already_closed(agent_id));
                }
            }
        };

        if let Some((permit, generation)) = resumed {
            self.publish();
            // This send cannot fail: `OwnedPermit::send` returns `()`, and the
            // capacity it needs was reserved above while the state lock was
            // held. Should a concurrent close drop the receiver first, the
            // command is discarded and the agent still cannot hang, because
            // every path that stops the runner — `run_shutdown` and the two
            // drop impls — is reached only after `begin_shutdown` has set
            // `Closing` under this same lock. A waiter then observes the close
            // and stops instead of waiting on a turn that will never run.
            permit.send(StartTurn {
                generation,
                prompt: message.to_owned(),
            });
        }

        Ok(())
    }

    /// Waits for the generation a child is on now, or for `cancel`.
    ///
    /// The generation is pinned when the wait starts, so a later turn cannot
    /// hand this caller the wrong answer, and results are cached, so waiting
    /// twice reports the same thing.
    ///
    /// Cancelling a wait closes the child: a parent whose round was interrupted
    /// is not going to read this result, and the child would otherwise keep
    /// working for nobody.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] of kind
    /// [`Cancelled`](crate::tools::ToolErrorKind::Cancelled) when `cancel`
    /// fires, and an execution error for an agent that is unknown or
    /// closed. A child that failed reports its own failure through this
    /// error as well, with its category carried across by
    /// [`child_failure`]: an interrupted child
    /// is [`Cancelled`](crate::tools::ToolErrorKind::Cancelled), every other
    /// failure is [`Execution`](crate::tools::ToolErrorKind::Execution). A
    /// caller that must tell its own cancellation apart from the child's
    /// reads the message, or takes the whole projection from
    /// [`status`](Self::status).
    pub(crate) async fn wait_with_cancel(
        &self,
        agent_id: &str,
        cancel: &CancellationToken,
    ) -> StdResult<SubagentResult, ToolError> {
        let (generation, mut status) = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let agent = state.agent(agent_id)?;
            (agent.generation, agent.status.subscribe())
        };

        loop {
            let current = {
                let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                let agent = state.agent(agent_id)?;
                // Checked before the status, so a coalesced watch update cannot
                // hide a generation that has already finished.
                if let Some(result) = agent.results.get(&generation) {
                    return result.clone().map_err(child_failure);
                }
                agent.status.borrow().clone()
            };
            match current {
                SubagentStatus::Closing | SubagentStatus::Closed => {
                    return Err(already_closed(agent_id));
                }
                SubagentStatus::Running | SubagentStatus::Finished { .. } => {}
            }

            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    self.ensure_closed(agent_id).await?;
                    return Err(ToolError::cancelled("Cancelled"));
                }
                changed = status.changed() => {
                    changed.map_err(|_| {
                        ToolError::execution(format!(
                            "Agent {agent_id} result observer closed unexpectedly"
                        ))
                    })?;
                }
            }
        }
    }

    /// Stops automatic delivery for an agent whose result the parent retrieved
    /// explicitly.
    ///
    /// The registration itself survives, so a later turn of the same child can
    /// register its own result.
    pub(crate) fn suppress_parent_notification(&self, agent_id: &str) {
        let cleared = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state
                .agents
                .get_mut(agent_id)
                .and_then(|agent| agent.parent_notification.as_mut())
                .is_some_and(|notification| {
                    let cleared = !notification.pending_generations.is_empty();
                    notification.pending_generations.clear();
                    cleared
                })
        };
        if cleared {
            signal_notifications(&self.notifications_changed);
        }
    }

    /// The next parent turn carrying finished background results, when there is
    /// one ready.
    ///
    /// Answers `None` as soon as nothing is pending, so the session's boundary
    /// drain ends rather than waiting for children it does not have. The
    /// envelope is the supervisor's concern, so the caller receives a finished
    /// turn rather than the notifications behind it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Interrupted`] when `cancel` fires while the supervisor
    /// is waiting on a child.
    pub(crate) async fn next_parent_notification_turn(
        &self,
        cancel: &CancellationToken,
    ) -> Result<Option<String>> {
        Ok(self
            .next_parent_notification_batch(cancel)
            .await?
            .map(|notifications| format_parent_notification_batch(&notifications)))
    }

    /// The notifications behind [`Self::next_parent_notification_turn`], for
    /// tests that assert on delivery semantics rather than on the rendering.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Interrupted`] when `cancel` fires while the supervisor
    /// is waiting on a child.
    pub(crate) async fn next_parent_notification_batch(
        &self,
        cancel: &CancellationToken,
    ) -> Result<Option<Vec<SubagentParentNotification>>> {
        let mut changed = self.notifications_changed.subscribe();
        loop {
            {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                let mut ready = Vec::new();
                let mut awaiting_result = false;
                for (agent_id, agent) in &state.agents {
                    let Some(notification) = agent.parent_notification.as_ref() else {
                        continue;
                    };
                    for generation in &notification.pending_generations {
                        if let Some(result) = agent.results.get(generation) {
                            ready.push((
                                agent.spawn_seq,
                                *generation,
                                SubagentParentNotification {
                                    agent_id:    agent_id.clone(),
                                    description: notification.description.clone(),
                                    result:      result.clone(),
                                },
                            ));
                        } else if agent.generation == *generation
                            && matches!(*agent.status.borrow(), SubagentStatus::Running)
                        {
                            awaiting_result = true;
                        }
                    }
                }

                if !ready.is_empty() {
                    ready.sort_by_key(|(spawn_seq, generation, _)| (*spawn_seq, *generation));
                    let delivered = ready
                        .iter()
                        .map(|(_, generation, notification)| {
                            (notification.agent_id.clone(), *generation)
                        })
                        .collect::<Vec<_>>();
                    let batch: Vec<_> = ready
                        .into_iter()
                        .map(|(_, _, notification)| notification)
                        .collect();
                    // Delivery is remembered by clearing the registration, not
                    // by consuming the result: a later `wait` still answers.
                    for (agent_id, generation) in delivered {
                        if let Some(notification) = state
                            .agents
                            .get_mut(&agent_id)
                            .and_then(|agent| agent.parent_notification.as_mut())
                        {
                            notification
                                .pending_generations
                                .retain(|pending| *pending != generation);
                        }
                    }
                    return Ok(Some(batch));
                }
                if !awaiting_result {
                    return Ok(None);
                }
            }

            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    return Err(Error::Interrupted(InterruptReason::Cancelled));
                }
                observed = changed.changed() => {
                    observed.map_err(|_| {
                        Error::InvalidState(
                            "Background-agent notification observer closed unexpectedly".to_owned(),
                        )
                    })?;
                }
            }
        }
    }

    /// Waits for one agent with no cancellation, for tests.
    #[cfg(test)]
    async fn wait(&self, agent_id: &str) -> StdResult<SubagentResult, ToolError> {
        self.wait_with_cancel(agent_id, &CancellationToken::new())
            .await
    }

    /// Decides what one caller closing `agent_id` has to do, and commits the
    /// child to closing when that caller is the one who leads it.
    fn begin_shutdown(
        &self,
        agent_id: &str,
        strict: bool,
    ) -> StdResult<ShutdownDisposition, ToolError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let agent = state.agent_mut(agent_id)?;

        let close_running_agent = match agent.status.borrow().clone() {
            SubagentStatus::Running => true,
            // An idle session is still shut down, but its cancel token is left
            // alone: there is nothing in flight to interrupt.
            SubagentStatus::Finished { .. } => false,
            SubagentStatus::Closing | SubagentStatus::Closed if strict => {
                return Err(ToolError::execution(format!(
                    "Agent {agent_id} is already closed"
                )));
            }
            SubagentStatus::Closing => {
                return Ok(ShutdownDisposition::Follow(agent.cleanup_done.subscribe()));
            }
            SubagentStatus::Closed => return Ok(ShutdownDisposition::Done),
        };
        // Reaching here means the status was Running or Finished, so this call
        // is the one that commits shutdown: the arms above return for a status
        // already Closing or Closed, and the only write out of Closing is
        // `run_shutdown`'s move to Closed.
        debug_assert!(agent.cleanup_task.is_none());
        agent.status.send_replace(SubagentStatus::Closing);

        // Shutdown is committed, so no pending result will reach the parent.
        agent.parent_notification = None;

        Ok(ShutdownDisposition::Lead(Box::new(ShutdownWork {
            handle: self.handle(agent_id.to_owned(), agent.depth),
            generation: agent.generation,
            close_running_agent,
            status: agent.status.clone(),
            cleanup_done: agent.cleanup_done.clone(),
            monitor_task: agent.monitor_task.take(),
            child_abort_handle: agent.child_abort_handle.clone(),
            cancel_token: agent.cancel_token.clone(),
            runner_stop: agent.runner_stop.clone(),
            slot: agent.slot.take(),
        })))
    }

    /// Stops one child and joins everything it owned.
    ///
    /// Cooperative first: the runner is asked to stop, a running child is
    /// cancelled, and each task gets its own window to end on its own before it
    /// is aborted and joined anyway.
    async fn run_shutdown(mut work: ShutdownWork) {
        let _cleanup_done = CleanupDoneGuard(work.cleanup_done.clone());
        let deadline = Instant::now() + SUBAGENT_SHUTDOWN_GRACE;
        work.runner_stop.cancel();
        if work.close_running_agent {
            work.cancel_token.cancel();
        }

        if let Some(mut task) = work.monitor_task.take()
            && timeout_at(deadline, &mut task).await.is_err()
        {
            work.child_abort_handle.abort();
            let _ = task.await;
        }

        let emit_closed = work.status.send_if_modified(|status| {
            if matches!(status, SubagentStatus::Closing) {
                *status = SubagentStatus::Closed;
                true
            } else {
                false
            }
        });
        if emit_closed {
            // The tree gets its slot back only once the session is really gone.
            if let Some(slot) = work.slot.take() {
                slot.release();
            }
            work.handle.queue_and_publish(CodingEvent::SubAgentClosed {
                agent_id:   work.handle.agent_id.clone(),
                depth:      work.handle.depth,
                generation: work.generation,
            });
        }
    }

    /// Starts the shutdown task and stores it back on the agent, so whoever
    /// waits for the close also joins the task that performed it.
    fn spawn_shutdown(&self, work: ShutdownWork) -> watch::Receiver<bool> {
        let cleanup_done = work.cleanup_done.subscribe();
        let agent_id = work.handle.agent_id.clone();
        let cleanup_task = tokio::spawn(Self::run_shutdown(work));
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(agent) = state.agents.get_mut(&agent_id) {
            debug_assert!(agent.cleanup_task.is_none());
            agent.cleanup_task = Some(cleanup_task);
        }
        cleanup_done
    }

    /// Waits for a child to finish closing and joins the task that closed it.
    async fn await_shutdown(&self, agent_id: &str, cleanup_done: watch::Receiver<bool>) {
        Self::follow_shutdown(cleanup_done).await;
        let cleanup_task = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state
                .agents
                .get_mut(agent_id)
                .and_then(|agent| agent.cleanup_task.take())
        };
        if let Some(task) = cleanup_task {
            let _ = task.await;
        }
    }

    /// Waits for the leader's cleanup flag.
    async fn follow_shutdown(mut cleanup_done: watch::Receiver<bool>) {
        while !*cleanup_done.borrow() {
            if cleanup_done.changed().await.is_err() {
                break;
            }
        }
    }

    /// Closes one child: leads the shutdown, follows one already underway, or
    /// returns at once for a child already closed.
    ///
    /// `strict` decides what an agent that is closing or closed means:
    /// `begin_shutdown` answers it with an error for the model-facing close and
    /// a no-op for internal cleanup.
    async fn close(&self, agent_id: &str, strict: bool) -> StdResult<(), ToolError> {
        let disposition = self.begin_shutdown(agent_id, strict)?;
        // Signalled before the close finishes, so a parent parked on a result
        // this child will never produce re-evaluates immediately.
        signal_notifications(&self.notifications_changed);
        let cleanup_done = match disposition {
            ShutdownDisposition::Lead(work) => self.spawn_shutdown(*work),
            ShutdownDisposition::Follow(cleanup_done) => cleanup_done,
            ShutdownDisposition::Done => return Ok(()),
        };
        self.await_shutdown(agent_id, cleanup_done).await;
        Ok(())
    }

    /// Closes a child, and does nothing to one that is already closed.
    async fn ensure_closed(&self, agent_id: &str) -> StdResult<(), ToolError> {
        self.close(agent_id, false).await
    }

    /// Closes a running or idle child that is no longer needed.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] for an agent that was never spawned or that is
    /// already closed, because a model closing the same agent twice has lost
    /// track of it and should be told.
    pub(crate) async fn close_agent(&self, agent_id: &str) -> StdResult<(), ToolError> {
        self.close(agent_id, true).await
    }

    /// Closes every child and waits for each to finish.
    ///
    /// The session calls this before it publishes its own end, so a child's
    /// closing events always precede the parent's.
    pub(crate) async fn shutdown_all(&self) {
        let ids = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.agents.keys().cloned().collect::<Vec<_>>()
        };
        join_all(ids.iter().map(|id| self.ensure_closed(id))).await;
    }

    /// Where one child is now, or `None` for an agent that was never spawned.
    ///
    /// A closed agent keeps answering: its entry stays, so a model that asks
    /// about an agent it closed is told what happened rather than that it never
    /// existed.
    pub(crate) fn status(&self, agent_id: &str) -> Option<SubagentStatus> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state
            .agents
            .get(agent_id)
            .map(|agent| agent.status.borrow().clone())
    }

    /// Whether this supervisor knows `agent_id`.
    #[cfg(test)]
    fn contains(&self, agent_id: &str) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .agents
            .contains_key(agent_id)
    }

    /// Whether this supervisor has any children at all.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .agents
            .is_empty()
    }

    /// How many sessions this supervisor's tree holds open, the root included.
    #[cfg(test)]
    pub(crate) fn open_sessions(&self) -> usize {
        self.deps.open_sessions.open()
    }

    /// Supervises a bare task as though it were a child session, so a test can
    /// drive the lifecycle without a model behind it.
    #[cfg(test)]
    pub(crate) fn supervise_test_task(
        &self,
        agent_id: String,
        child_task: JoinHandle<Result<SubagentResult>>,
        cancel_token: CancellationToken,
    ) {
        let child_abort_handle = child_task.abort_handle();
        let (status, _) = watch::channel(SubagentStatus::Running);
        let (cleanup_done, _) = watch::channel(false);
        let (command_tx, command_rx) = mpsc::channel(SUBAGENT_COMMAND_CAPACITY);
        drop(command_rx);
        let runner_stop = CancellationToken::new();
        let depth = 1;
        let (monitor_start_tx, monitor_start_rx) = oneshot::channel();
        let handle = self.handle(agent_id.clone(), depth);
        let monitor_task = tokio::spawn(async move {
            let _ = monitor_start_rx.await;
            let task_result = match child_task.await {
                Ok(result) => result,
                Err(source) => Err(Error::Task {
                    task: TaskKind::SubagentSession,
                    source,
                }),
            };
            handle.commit_turn_result(INITIAL_SUBAGENT_GENERATION, &task_result, false);
        });
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.agents.insert(agent_id, SubAgent {
                status,
                generation: INITIAL_SUBAGENT_GENERATION,
                results: HashMap::new(),
                command_tx,
                runner_stop,
                cleanup_done,
                monitor_task: Some(monitor_task),
                cleanup_task: None,
                child_abort_handle,
                control: AgentControlHandle::detached(),
                cancel_token,
                depth,
                parent_notification: None,
                spawn_seq: 0,
                slot: self.deps.open_sessions.try_acquire(),
            });
        }
        let _ = monitor_start_tx.send(());
    }
}

#[cfg(test)]
mod tests {
    use std::future::pending;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicBool;
    use std::{iter, thread};

    use futures_util::poll;
    use lithos_llm::types::{Role, ToolDefinitionKind};
    use pebble_agent::ToolDescriptor;
    use serde_json::json;
    use tokio::task::yield_now;
    use tokio::time;

    use super::*;
    use crate::error::ErrorKind;
    use crate::record::SessionRecord;
    use crate::runtime::testing::{TestSession, noop_tool};
    use crate::runtime::{ResumeMode, testing};
    use crate::test_support::{
        MockEnvironment, ScriptedCall, message_text, scripted_client, text_response,
    };
    use crate::tool::{
        PermissionMiddleware, ToolContext, ToolDefinitionWithSource, ToolPermission,
        ToolPermissionPolicy,
    };
    use crate::types::{CodingAgentEvent, PermissionLevel, ToolErrorKind};

    /// Reports the moment the task holding it is dropped, which is what an
    /// abort does to a task that never returns on its own.
    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Spins a bounded number of times for `flag`, so a test that will never
    /// see it fails rather than hangs.
    ///
    /// An abort is not synchronous: the runtime drops the aborted task's future
    /// on a later pass, and a cascade of aborts takes one pass per level.
    async fn wait_for_flag(flag: &Arc<AtomicBool>) -> bool {
        for _ in 0..1_000 {
            if flag.load(Ordering::SeqCst) {
                return true;
            }
            yield_now().await;
        }
        flag.load(Ordering::SeqCst)
    }

    /// A policy that refuses one tool by name and allows the rest.
    struct DenyByName(&'static str);

    impl ToolPermissionPolicy for DenyByName {
        fn permission(&self, tool: &ToolDescriptor) -> ToolPermission {
            if tool.id().as_str() == self.0 {
                ToolPermission::Deny {
                    reason: format!("{} is forbidden", tool.id()),
                }
            } else {
                ToolPermission::Allow
            }
        }
    }

    /// The tools a session shows its model, by name, sorted so two sessions can
    /// be compared however their registries happen to iterate.
    fn tool_names(session: &CodingRuntime) -> Vec<String> {
        let mut names: Vec<String> = session
            .registered_tools()
            .iter()
            .map(|tool| tool.definition.name.clone())
            .collect();
        names.sort();
        names
    }

    /// A parent session that may spawn children, and the supervisor it drives
    /// them with. Every child answers from the same script.
    fn parent_over(answers: Vec<&str>) -> (CodingRuntime, SubagentSupervisor) {
        let calls = answers
            .into_iter()
            .map(|text| ScriptedCall::response(text_response(text)))
            .collect();
        let (session, _provider) = TestSession::new(calls).with_subagents().build();
        let supervisor = supervisor_of(&session);
        (session, supervisor)
    }

    fn supervisor_of(session: &CodingRuntime) -> SubagentSupervisor {
        session
            .subagent_supervisor()
            .expect("the test session was given a factory")
            .clone()
    }

    fn spawn(supervisor: &SubagentSupervisor, parent: &CodingRuntime, task: &str) -> String {
        supervisor
            .spawn(parent.id(), parent.root_session_id(), task.to_owned())
            .expect("the spawn succeeds")
    }

    fn spawn_notifying(
        supervisor: &SubagentSupervisor,
        parent: &CodingRuntime,
        task: &str,
        description: &str,
    ) -> String {
        supervisor
            .spawn_with_parent_notification(
                parent.id(),
                parent.root_session_id(),
                task.to_owned(),
                description.to_owned(),
            )
            .expect("the spawn succeeds")
    }

    /// One child a recording factory built, so a test can drive the tree below
    /// it.
    #[derive(Clone)]
    struct ChildHandle {
        id:         String,
        supervisor: SubagentSupervisor,
    }

    /// A parent whose factory keeps hold of each child it builds.
    ///
    /// A child session is moved into its runner task the moment it is
    /// supervised, so the factory is the only place a test can take what it
    /// needs from one — its identity, and the supervisor that drives its own
    /// children.
    fn parent_over_recording_children(
        answers: Vec<&str>,
    ) -> (
        CodingRuntime,
        SubagentSupervisor,
        Arc<Mutex<Vec<ChildHandle>>>,
    ) {
        let recorded: Arc<Mutex<Vec<ChildHandle>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&recorded);
        let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
            if let Some(supervisor) = child.subagent_supervisor() {
                recorder
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(ChildHandle {
                        id:         child.id().to_owned(),
                        supervisor: supervisor.clone(),
                    });
            }
        });
        let calls = answers
            .into_iter()
            .map(|text| ScriptedCall::response(text_response(text)))
            .collect();
        let (session, _provider) = TestSession::new(calls).observe_children(observer).build();
        let supervisor = supervisor_of(&session);
        (session, supervisor, recorded)
    }

    /// The first child the recording factory built, which exists as soon as the
    /// spawn that asked for it has answered.
    fn first_child(recorded: &Arc<Mutex<Vec<ChildHandle>>>) -> ChildHandle {
        recorded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .first()
            .cloned()
            .expect("the factory built a child")
    }

    /// Spins until a child has committed its own result, rather than forever.
    ///
    /// A commit that never arrives is a failed test, not a hung one.
    async fn wait_until_finished(supervisor: &SubagentSupervisor, agent_id: &str) {
        time::timeout(Duration::from_secs(5), async {
            while !matches!(
                supervisor.status(agent_id),
                Some(SubagentStatus::Finished { .. })
            ) {
                yield_now().await;
            }
        })
        .await
        .expect("the child commits its result on its own");
    }

    /// A callback that records everything the supervisor publishes.
    type Captured = Arc<Mutex<Vec<CodingEvent>>>;

    fn captured_events() -> (SubagentEventCallback, Captured) {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&captured);
        let callback: SubagentEventCallback = Arc::new(move |event| {
            recorder
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event);
        });
        (callback, captured)
    }

    fn lifecycle(captured: &Captured) -> Vec<CodingEvent> {
        captured
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn tool_named(tools: &[RegisteredTool], name: &str) -> RegisteredTool {
        tools
            .iter()
            .find(|tool| tool.definition.name == name)
            .cloned()
            .unwrap_or_else(|| panic!("the {name} tool is registered"))
    }

    fn schema_of(tool: &RegisteredTool) -> serde_json::Value {
        match &tool.definition.kind {
            ToolDefinitionKind::Function { input_schema } => input_schema.clone(),
            other => panic!("a subagent tool is a function tool, not {other:?}"),
        }
    }

    /// Where a drain sends what it delivers.
    type CallbackSlot = Arc<RwLock<Option<SubagentEventCallback>>>;
    /// What a drain delivered.
    type SeenEvents = Arc<Mutex<Vec<CodingEvent>>>;

    /// A callback slot that records the lifecycle events a drain delivers.
    fn recording_lifecycle_callback() -> (CallbackSlot, SeenEvents) {
        let seen: SeenEvents = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        let callback: SubagentEventCallback = Arc::new(move |event| {
            recorder
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event);
        });
        (Arc::new(RwLock::new(Some(callback))), seen)
    }

    fn closed_event(agent_id: &str) -> CodingEvent {
        CodingEvent::SubAgentClosed {
            agent_id:   agent_id.to_owned(),
            depth:      1,
            generation: INITIAL_SUBAGENT_GENERATION,
        }
    }

    #[test]
    fn a_finished_drain_leaves_the_queue_empty_and_the_flag_clear() {
        let state = Arc::new(Mutex::new(SupervisorState::default()));
        let (callback, seen) = recording_lifecycle_callback();
        state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .queue_lifecycle_event(closed_event("sa-1"));

        drain_lifecycle_events(&state, &callback);

        let locked = state.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(locked.lifecycle_events.is_empty());
        assert!(
            !locked.lifecycle_draining,
            "a finished drain leaves the flag clear for the next publisher"
        );
        assert_eq!(*seen.lock().unwrap_or_else(PoisonError::into_inner), vec![
            closed_event("sa-1")
        ]);
    }

    #[test]
    fn a_publisher_that_defers_to_a_running_drain_never_loses_its_event() {
        // A publisher returns as soon as it sees another thread draining,
        // trusting that thread to take the event it just queued. That trust
        // holds only while the drain clears its flag under the same lock that
        // found the queue empty. The window between the two is a few
        // instructions wide, so two threads publish into it round after round,
        // and each round is checked on its own: a stranded event is delivered
        // by the next round's publish, which would hide it from a check made
        // only at the end.
        const ROUNDS: usize = 30_000;

        let state = Arc::new(Mutex::new(SupervisorState::default()));
        let (callback, _seen) = recording_lifecycle_callback();
        let barrier = Barrier::new(2);
        let stranded = AtomicUsize::new(0);

        thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    for _ in 0..ROUNDS {
                        barrier.wait();
                        state
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .queue_lifecycle_event(closed_event("sa-1"));
                        drain_lifecycle_events(&state, &callback);
                        // Both publishers have returned, so nothing is left to
                        // deliver this round's events.
                        if barrier.wait().is_leader() {
                            let mut locked = state.lock().unwrap_or_else(PoisonError::into_inner);
                            stranded.fetch_add(locked.lifecycle_events.len(), Ordering::SeqCst);
                            locked.lifecycle_events.clear();
                        }
                        barrier.wait();
                    }
                });
            }
        });

        assert_eq!(
            stranded.load(Ordering::SeqCst),
            0,
            "no round left an event queued after both publishers returned"
        );
    }

    #[test]
    fn the_notification_envelope_escapes_what_would_close_it_early() {
        let envelope = format_parent_notification_batch(&[SubagentParentNotification {
            agent_id:    "agent<&".to_owned(),
            description: "Review <core> & tests".to_owned(),
            result:      Ok(SubagentResult {
                output:     "done <safely> & \"verified\"".to_owned(),
                success:    true,
                turns_used: 2,
            }),
        }]);

        assert!(envelope.contains("<status>completed</status>"));
        assert!(envelope.contains("<task-id>agent&lt;&amp;</task-id>"));
        assert!(envelope.contains("<description>Review &lt;core&gt; &amp; tests</description>"));
        assert!(
            envelope.contains("<result>done &lt;safely&gt; &amp; &quot;verified&quot;</result>")
        );
    }

    #[test]
    fn a_failed_generation_reaches_the_parent_as_a_failure() {
        let envelope = format_parent_notification_batch(&[SubagentParentNotification {
            agent_id:    "sa-1".to_owned(),
            description: "Inspect the module".to_owned(),
            result:      Err(ErrorData::from(&Error::InvalidState(
                "the child gave up".to_owned(),
            ))),
        }]);

        assert!(envelope.contains("<status>failed</status>"));
        assert!(
            envelope.contains("<result>invalid state: the child gave up</result>"),
            "{envelope}"
        );
    }

    #[test]
    fn a_batch_is_one_envelope_per_notification() {
        let one = SubagentParentNotification {
            agent_id:    "sa-1".to_owned(),
            description: "First".to_owned(),
            result:      Ok(SubagentResult {
                output:     "first".to_owned(),
                success:    true,
                turns_used: 1,
            }),
        };
        let two = SubagentParentNotification {
            agent_id: "sa-2".to_owned(),
            description: "Second".to_owned(),
            ..one.clone()
        };

        let envelope = format_parent_notification_batch(&[one, two]);

        assert_eq!(envelope.matches("<task-notification>").count(), 2);
        assert!(envelope.contains("</task-notification>\n\n<task-notification>"));
    }

    #[test]
    fn the_notification_envelope_is_frozen_text() {
        // A parent reads this as a user turn, so every character of it is model
        // -visible API. Pinned whole rather than by fragments: a stray space or
        // a reordered field is a change to what the model is trained against.
        let envelope = format_parent_notification_batch(&[
            SubagentParentNotification {
                agent_id:    "1a2b3c4d".to_owned(),
                description: "Inspect the module".to_owned(),
                result:      Ok(SubagentResult {
                    output:     "Found two dead branches".to_owned(),
                    success:    true,
                    turns_used: 3,
                }),
            },
            SubagentParentNotification {
                agent_id:    "5e6f7a8b".to_owned(),
                description: "Run the suite".to_owned(),
                result:      Err(ErrorData::from(&Error::InvalidState(
                    "the child gave up".to_owned(),
                ))),
            },
        ]);

        assert_eq!(
            envelope,
            "<task-notification>\n  <task-id>1a2b3c4d</task-id>\n  <status>completed</status>\n  \
             <description>Inspect the module</description>\n  <result>Found two dead \
             branches</result>\n</task-notification>\n\n<task-notification>\n  \
             <task-id>5e6f7a8b</task-id>\n  <status>failed</status>\n  <description>Run the \
             suite</description>\n  <result>invalid state: the child gave \
             up</result>\n</task-notification>"
        );
    }

    #[tokio::test]
    async fn a_spawn_answers_with_the_identifier_of_a_supervised_child() {
        let (mut parent, supervisor) = parent_over(vec!["child result"]);

        let agent_id = spawn(&supervisor, &parent, "review the diff");

        assert!(!agent_id.is_empty());
        assert!(supervisor.contains(&agent_id));
        parent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the parent shuts down");
    }

    #[tokio::test]
    async fn a_child_is_running_as_soon_as_it_is_spawned() {
        let (parent, supervisor) = parent_over(vec!["child result"]);

        let agent_id = spawn(&supervisor, &parent, "review the diff");

        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Running)
        ));
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_child_is_initialized_before_it_is_given_its_task() {
        let (session, provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("child result"))])
                .with_subagents()
                .build();
        let supervisor = supervisor_of(&session);

        let agent_id = spawn(&supervisor, &session, "review the diff");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let requests = provider.requests();
        let request = requests.first().expect("the child asked the model");
        let system = request
            .messages()
            .first()
            .expect("the request is not empty");
        assert_eq!(system.role(), Role::System);
        assert!(
            message_text(system).contains("test assistant"),
            "the child ran `initialize` before its first turn: {}",
            message_text(system)
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn waiting_answers_with_the_childs_result() {
        let (parent, supervisor) = parent_over(vec!["Task completed successfully"]);
        let agent_id = spawn(&supervisor, &parent, "do something");

        let result = supervisor.wait(&agent_id).await.expect("the child answers");

        assert_eq!(result.output, "Task completed successfully");
        assert!(result.success);
        assert!(result.turns_used > 0);
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Finished { .. })
        ));
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn waiting_twice_answers_from_the_cache() {
        let (parent, supervisor) = parent_over(vec!["cached output"]);
        let agent_id = spawn(&supervisor, &parent, "do something");

        let first = supervisor.wait(&agent_id).await.expect("the child answers");
        let second = supervisor.wait(&agent_id).await.expect("the cache answers");

        assert_eq!(first, second);
        assert_eq!(second.output, "cached output");
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn an_empty_final_response_is_not_reported_as_success() {
        let (parent, supervisor) = parent_over(vec![""]);
        let agent_id = spawn(&supervisor, &parent, "do something");

        let error = supervisor
            .wait(&agent_id)
            .await
            .expect_err("an empty answer is not an answer");

        assert!(
            error
                .message()
                .contains("without a non-empty final response"),
            "{}",
            error.message()
        );
        supervisor.shutdown_all().await;
    }

    /// What every call naming an agent nobody spawned answers with.
    fn assert_names_the_missing_agent(error: &ToolError) {
        assert_eq!(
            error.message(),
            "No agent found with id: nope (it was never spawned)"
        );
        assert_eq!(error.kind(), ToolErrorKind::Execution);
    }

    #[tokio::test]
    async fn sending_input_to_an_agent_that_was_never_spawned_errors() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);

        let error = supervisor
            .send_input("nope", "hello")
            .expect_err("there is no such agent");

        assert_names_the_missing_agent(&error);
    }

    #[tokio::test]
    async fn waiting_on_an_agent_that_was_never_spawned_errors() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);

        let error = supervisor
            .wait("nope")
            .await
            .expect_err("there is no such agent");

        assert_names_the_missing_agent(&error);
    }

    #[tokio::test]
    async fn closing_an_agent_that_was_never_spawned_errors() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);

        let error = supervisor
            .close_agent("nope")
            .await
            .expect_err("there is no such agent");

        assert_names_the_missing_agent(&error);
    }

    #[tokio::test]
    async fn closing_a_running_child_leaves_it_closed() {
        let (parent, supervisor) = parent_over(vec!["hello"]);
        let agent_id = spawn(&supervisor, &parent, "do something");
        assert!(supervisor.contains(&agent_id));

        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
    }

    #[tokio::test]
    async fn closing_a_finished_child_closes_its_idle_session() {
        let (parent, supervisor) = parent_over(vec!["done"]);
        let agent_id = spawn(&supervisor, &parent, "do something");
        supervisor.wait(&agent_id).await.expect("the child answers");
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Finished { .. })
        ));

        supervisor
            .close_agent(&agent_id)
            .await
            .expect("an idle session closes too");

        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
    }

    #[tokio::test]
    async fn closing_a_child_twice_is_an_error() {
        let (parent, supervisor) = parent_over(vec!["hello"]);
        let agent_id = spawn(&supervisor, &parent, "do something");
        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        let error = supervisor
            .close_agent(&agent_id)
            .await
            .expect_err("closing twice is an error");

        assert_eq!(
            error.message(),
            format!("Agent {agent_id} is already closed")
        );
    }

    #[tokio::test]
    async fn a_closed_child_takes_no_more_input() {
        let (parent, supervisor) = parent_over(vec!["hello"]);
        let agent_id = spawn(&supervisor, &parent, "do something");
        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        let error = supervisor
            .send_input(&agent_id, "more")
            .expect_err("a closed agent takes no input");

        assert_eq!(error.message(), format!("Agent {agent_id} has been closed"));
    }

    #[tokio::test]
    async fn waiting_on_a_closed_child_is_an_error() {
        let (parent, supervisor) = parent_over(vec!["hello"]);
        let agent_id = spawn(&supervisor, &parent, "do something");
        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        let error = supervisor
            .wait(&agent_id)
            .await
            .expect_err("a closed agent has nothing to wait for");

        assert_eq!(error.message(), format!("Agent {agent_id} has been closed"));
    }

    #[tokio::test]
    async fn shutting_down_closes_every_child() {
        let (parent, supervisor) = parent_over(vec!["hello"]);
        let first = spawn(&supervisor, &parent, "first");
        let second = spawn(&supervisor, &parent, "second");

        supervisor.shutdown_all().await;

        for agent_id in [&first, &second] {
            assert!(matches!(
                supervisor.status(agent_id),
                Some(SubagentStatus::Closed)
            ));
        }
    }

    #[tokio::test]
    async fn shutting_down_with_no_children_does_nothing() {
        let (_parent, supervisor) = parent_over(vec!["hello"]);

        supervisor.shutdown_all().await;

        assert!(supervisor.is_empty());
    }

    #[tokio::test]
    async fn input_for_a_running_child_joins_the_turn_it_is_already_taking() {
        let (callback, captured) = captured_events();
        let (parent, supervisor) = parent_over(vec!["initial", "after follow-up"]);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &parent, "do something");

        supervisor
            .send_input(&agent_id, "use this as well")
            .expect("the message is queued");
        let result = supervisor.wait(&agent_id).await.expect("the child answers");

        assert_eq!(result.output, "after follow-up");
        let events = lifecycle(&captured);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, CodingEvent::SubAgentTurnStarted { .. })),
            "a follow-up does not start a generation: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, CodingEvent::SubAgentCompleted { generation: 1, .. }))
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn input_for_a_finished_child_starts_another_generation_on_its_history() {
        let (callback, captured) = captured_events();
        let (session, provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("captured"))])
                .with_subagents()
                .build();
        let supervisor = supervisor_of(&session);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &session, "do something");
        let first = supervisor.wait(&agent_id).await.expect("the child answers");

        supervisor
            .send_input(&agent_id, "fix the review findings")
            .expect("a finished child takes another turn");
        let second = supervisor
            .wait(&agent_id)
            .await
            .expect("the second turn answers");

        assert_eq!(first.output, "captured");
        assert_eq!(second.turns_used, first.turns_used);

        let requests = provider.requests();
        let last = requests.last().expect("the second turn asked the model");
        let user_text: Vec<String> = last
            .messages()
            .iter()
            .filter(|message| message.role() == Role::User)
            .map(message_text)
            .collect();
        assert!(
            user_text.iter().any(|text| text.contains("do something")),
            "the child kept its history: {user_text:?}"
        );
        assert!(
            user_text
                .iter()
                .any(|text| text.contains("fix the review findings")),
            "the second turn carries the new input: {user_text:?}"
        );
        let assistant_text: Vec<String> = last
            .messages()
            .iter()
            .filter(|message| message.role() == Role::Assistant)
            .map(message_text)
            .collect();
        assert!(
            assistant_text.iter().any(|text| text.contains("captured")),
            "the child kept what it answered the first time: {assistant_text:?}"
        );

        let events = lifecycle(&captured);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, CodingEvent::SubAgentSpawned { .. }))
                .count(),
            1,
            "one session, two turns: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, CodingEvent::SubAgentTurnStarted {
                    generation: 2,
                    ..
                }))
        );
        let completions: Vec<u64> = events
            .iter()
            .filter_map(|event| match event {
                CodingEvent::SubAgentCompleted { generation, .. } => Some(*generation),
                _ => None,
            })
            .collect();
        assert_eq!(completions, vec![1, 2]);
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_finished_child_is_delivered_to_its_parent_exactly_once() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let agent_id = spawn_notifying(&supervisor, &parent, "task", "Inspect the module");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let batch = supervisor
            .next_parent_notification_batch(&CancellationToken::new())
            .await
            .expect("nothing was cancelled")
            .expect("the finished child is delivered");

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].agent_id, agent_id);
        assert_eq!(batch[0].description, "Inspect the module");
        // Delivery is remembered by clearing the registration, not by taking
        // the result: the status stays `Finished` and a later wait still works.
        assert!(
            supervisor
                .next_parent_notification_batch(&CancellationToken::new())
                .await
                .expect("nothing was cancelled")
                .is_none()
        );
        assert!(supervisor.wait(&agent_id).await.is_ok());
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_reused_child_delivers_each_generation_to_its_parent() {
        let (parent, supervisor) = parent_over(vec!["first result", "remediation result"]);
        let agent_id = spawn_notifying(&supervisor, &parent, "task", "Inspect the module");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let first = supervisor
            .next_parent_notification_batch(&CancellationToken::new())
            .await
            .expect("nothing was cancelled")
            .expect("the first generation is delivered");
        supervisor
            .send_input(&agent_id, "now fix what you found")
            .expect("a finished child takes another turn");
        supervisor.wait(&agent_id).await.expect("the child answers");
        let second = supervisor
            .next_parent_notification_batch(&CancellationToken::new())
            .await
            .expect("nothing was cancelled")
            .expect("the second generation is delivered too");

        assert_eq!(
            first[0].result.as_ref().expect("the turn succeeded").output,
            "first result"
        );
        assert_eq!(
            second[0]
                .result
                .as_ref()
                .expect("the turn succeeded")
                .output,
            "remediation result"
        );
        assert_eq!(second[0].description, "Inspect the module");
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_child_that_finishes_with_nobody_waiting_still_commits_once() {
        let (callback, captured) = captured_events();
        let (parent, supervisor) = parent_over(vec!["child result"]);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &parent, "task");

        // Nothing waits: the runner commits on its own.
        wait_until_finished(&supervisor, &agent_id).await;

        let events = lifecycle(&captured);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, CodingEvent::SubAgentCompleted { .. }))
                .count(),
            1,
            "{events:?}"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn two_waiters_share_one_result_and_one_completion_event() {
        let (callback, captured) = captured_events();
        let (parent, supervisor) = parent_over(vec!["child result"]);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &parent, "task");

        let first = supervisor.clone();
        let second = supervisor.clone();
        let one = agent_id.clone();
        let other = agent_id.clone();
        let (left, right) = tokio::join!(async move { first.wait(&one).await }, async move {
            second.wait(&other).await
        },);

        assert_eq!(
            left.expect("the child answers"),
            right.expect("the child answers")
        );
        let events = lifecycle(&captured);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, CodingEvent::SubAgentCompleted { .. }))
                .count(),
            1,
            "{events:?}"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_batch_is_delivered_oldest_spawn_first() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let spawned: Vec<String> = (0..3)
            .map(|index| {
                spawn_notifying(
                    &supervisor,
                    &parent,
                    &format!("task {index}"),
                    &format!("Task {index}"),
                )
            })
            .collect();
        // Finished in a different order than they were spawned, so completion
        // order cannot pass for spawn order.
        for index in [2, 0, 1] {
            supervisor
                .wait(&spawned[index])
                .await
                .expect("the child answers");
        }

        let batch = supervisor
            .next_parent_notification_batch(&CancellationToken::new())
            .await
            .expect("nothing was cancelled")
            .expect("all three children are ready");

        let delivered: Vec<String> = batch
            .iter()
            .map(|notification| notification.agent_id.clone())
            .collect();
        assert_eq!(delivered, spawned);
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_turn_carries_every_result_that_was_ready() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let first = spawn_notifying(&supervisor, &parent, "first", "First task");
        let second = spawn_notifying(&supervisor, &parent, "second", "Second task");
        supervisor.wait(&first).await.expect("the child answers");
        supervisor.wait(&second).await.expect("the child answers");

        let turn = supervisor
            .next_parent_notification_turn(&CancellationToken::new())
            .await
            .expect("nothing was cancelled")
            .expect("both children are ready");

        assert_eq!(turn.matches("<task-notification>").count(), 2);
        assert!(turn.contains("<description>First task</description>"));
        assert!(turn.contains("<description>Second task</description>"));
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn suppressing_a_notification_stops_its_delivery() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let agent_id = spawn_notifying(&supervisor, &parent, "task", "Inspect the module");

        supervisor.suppress_parent_notification(&agent_id);
        supervisor.wait(&agent_id).await.expect("the child answers");

        assert!(
            supervisor
                .next_parent_notification_batch(&CancellationToken::new())
                .await
                .expect("nothing was cancelled")
                .is_none(),
            "a parent that took the result itself is not told again"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn closing_a_child_stops_delivery_without_parking_its_parent() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let agent_id = spawn_notifying(&supervisor, &parent, "task", "Inspect the module");

        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        assert!(
            supervisor
                .next_parent_notification_batch(&CancellationToken::new())
                .await
                .expect("nothing was cancelled")
                .is_none(),
            "a closed child neither delivers nor parks the parent"
        );
    }

    #[tokio::test]
    async fn closing_a_finished_child_discards_its_pending_notification() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let agent_id = spawn_notifying(&supervisor, &parent, "task", "Inspect the module");
        // Finish the child, so its result is queued for automatic delivery.
        supervisor.wait(&agent_id).await.expect("the child answers");

        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        assert!(
            supervisor
                .next_parent_notification_batch(&CancellationToken::new())
                .await
                .expect("nothing was cancelled")
                .is_none(),
            "a close commits: no pending result reaches the parent after it"
        );
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
    }

    #[tokio::test]
    async fn a_wait_answers_with_the_generation_it_started_on() {
        let (parent, supervisor) = parent_over(vec!["generation one", "generation two"]);
        let agent_id = spawn(&supervisor, &parent, "implement");

        // Registering the wait pins it to generation one. It stays unpolled
        // from here, so generation one's completion and generation two's start
        // reach it as a single coalesced watch update.
        let cancel = CancellationToken::new();
        let mut waiting = Box::pin(supervisor.wait_with_cancel(&agent_id, &cancel));
        assert!(
            poll!(waiting.as_mut()).is_pending(),
            "generation one is still running"
        );

        wait_until_finished(&supervisor, &agent_id).await;
        supervisor
            .send_input(&agent_id, "fix the review findings")
            .expect("a finished child takes another turn");

        let result = time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("a coalesced status update must not hide generation one")
            .expect("generation one answered");

        assert_eq!(result.output, "generation one");
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn the_callback_sees_a_child_spawn() {
        let (callback, captured) = captured_events();
        let (parent, supervisor) = parent_over(vec!["child result"]);
        supervisor.set_event_callback(callback);

        spawn(&supervisor, &parent, "test task");

        let events = lifecycle(&captured);
        assert_eq!(events.len(), 1, "{events:?}");
        assert!(
            matches!(&events[0], CodingEvent::SubAgentSpawned { depth: 1, task, generation: 1, .. }
                if task == "test task"),
            "{events:?}"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn the_callback_sees_a_child_complete() {
        let (callback, captured) = captured_events();
        let (parent, supervisor) = parent_over(vec!["child result"]);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &parent, "task");

        supervisor.wait(&agent_id).await.expect("the child answers");

        let events = lifecycle(&captured);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, CodingEvent::SubAgentCompleted {
                    depth: 1,
                    success: true,
                    ..
                })),
            "{events:?}"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn the_callback_sees_a_child_close() {
        let (callback, captured) = captured_events();
        let (parent, supervisor) = parent_over(vec!["child result"]);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &parent, "task");

        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        let events = lifecycle(&captured);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, CodingEvent::SubAgentClosed { depth: 1, .. })),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn a_grandchild_is_announced_one_level_deeper() {
        let (parent, supervisor, children) = parent_over_recording_children(vec!["child result"]);
        let child_id = spawn(&supervisor, &parent, "delegate further");
        let child = first_child(&children);

        // The child's own supervisor is what spawns a grandchild, so the depth
        // is counted from the child rather than from the root.
        let (callback, captured) = captured_events();
        child.supervisor.set_event_callback(callback);
        let grandchild = child
            .supervisor
            .spawn(
                &child.id,
                parent.root_session_id(),
                "the leaf task".to_owned(),
            )
            .expect("a child may spawn a child of its own");

        let events = lifecycle(&captured);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, CodingEvent::SubAgentSpawned { depth: 2, .. })),
            "a grandchild sits one deeper than its parent: {events:?}"
        );
        assert!(child.supervisor.contains(&grandchild));
        assert_eq!(
            supervisor.open_sessions(),
            3,
            "root, child and grandchild share one budget"
        );

        child.supervisor.shutdown_all().await;
        supervisor.shutdown_all().await;
        assert!(supervisor.contains(&child_id));
        assert_eq!(supervisor.open_sessions(), 1, "only the root is left");
    }

    #[tokio::test]
    async fn a_childs_own_events_reach_the_parents_stream() {
        let (mut parent, supervisor) = parent_over(vec!["child result"]);
        let mut events = parent.subscribe();
        let agent_id = spawn(&supervisor, &parent, "review the diff");
        supervisor.wait(&agent_id).await.expect("the child answers");
        supervisor.shutdown_all().await;
        parent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the parent shuts down");

        let published: Vec<CodingAgentEvent> = iter::from_fn(|| events.try_recv().ok()).collect();
        let child_event = published
            .iter()
            .find(|event| event.session_id != parent.id())
            .expect("the child's own events use the tree stream");

        assert_eq!(child_event.stream_id, parent.id());
        assert_eq!(child_event.parent_session_id.as_deref(), Some(parent.id()));
        assert!(
            published.iter().all(|event| event.stream_id == parent.id()),
            "one root owns the whole tree stream: {published:?}"
        );
        assert!(
            published.iter().any(|event| {
                event.session_id != parent.id()
                    && matches!(&event.event, CodingEvent::AssistantMessage { text, .. }
                        if text == "child result")
            }),
            "what the child answered is news the parent's readers get: {published:?}"
        );
        assert!(
            published
                .iter()
                .any(|event| matches!(event.event, CodingEvent::SubAgentSpawned { .. })),
            "the parent's own news about the child is on the same stream"
        );
        // One numbering for the whole stream: the parent's own events and the
        // child's events share it, in the order the pump published
        // them, with nothing skipped and nothing repeated.
        let numbering: Vec<u64> = published.iter().map(|event| event.seq).collect();
        assert_eq!(
            numbering,
            (1..=u64::try_from(published.len()).expect("a small stream")).collect::<Vec<_>>(),
            "every tree event takes one root-stream sequence number"
        );
        assert!(
            published.iter().any(|event| event.session_id != parent.id()
                && matches!(event.event, CodingEvent::SessionEnded)),
            "the lossless tree stream includes the child's session boundary"
        );
    }

    #[tokio::test]
    async fn a_tree_refuses_a_spawn_beyond_its_open_session_limit() {
        let (session, _provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("child result"))])
                .with_subagents()
                .build();
        let supervisor = supervisor_of(&session);

        // Four open sessions, the root included, leaves room for three.
        let children: Vec<String> = (0..3)
            .map(|index| spawn(&supervisor, &session, &format!("task {index}")))
            .collect();
        let refused = supervisor
            .spawn(
                session.id(),
                session.root_session_id(),
                "one too many".to_owned(),
            )
            .expect_err("the tree is full");

        assert_eq!(refused.kind(), ToolErrorKind::Denied);
        assert!(
            refused.message().contains("(4, counting the root)"),
            "{}",
            refused.message()
        );
        assert_eq!(supervisor.open_sessions(), 4);

        // Closing one gives the slot back.
        supervisor
            .close_agent(&children[0])
            .await
            .expect("the close succeeds");
        assert_eq!(supervisor.open_sessions(), 3);
        let replacement = spawn(&supervisor, &session, "the replacement");
        assert!(supervisor.contains(&replacement));

        supervisor.shutdown_all().await;
        assert_eq!(supervisor.open_sessions(), 1, "only the root is left");
    }

    #[tokio::test]
    async fn a_finished_child_holds_its_slot_until_it_is_closed() {
        let (session, _provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("child result"))])
                .with_subagents()
                .subagent_limits(SubagentLimits::new(2))
                .build();
        let supervisor = supervisor_of(&session);
        let agent_id = spawn(&supervisor, &session, "task");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let refused = supervisor
            .spawn(
                session.id(),
                session.root_session_id(),
                "another".to_owned(),
            )
            .expect_err("a finished child still holds its session");
        assert_eq!(refused.kind(), ToolErrorKind::Denied);

        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");
        assert!(supervisor.contains(&spawn(&supervisor, &session, "another")));
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_child_is_built_from_its_parents_dependencies() {
        /// What one spawn told the factory, and what the child came out as.
        struct Built {
            model:  String,
            root:   String,
            parent: String,
        }

        let seen: Arc<Mutex<Vec<Built>>> = Arc::new(Mutex::new(Vec::new()));
        let asked = Arc::new(AtomicBool::new(false));
        let recorded = Arc::clone(&seen);
        let recorded_ask = Arc::clone(&asked);
        let environment: Arc<dyn Environment> = Arc::new(MockEnvironment::linux());
        let parent_environment = Arc::clone(&environment);
        let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
            assert!(
                Arc::ptr_eq(child.environment(), &parent_environment),
                "a child acts where its parent does"
            );
            assert!(
                child.config().memory_files.is_empty(),
                "a child is given a task, not the project briefing"
            );
            assert!(child.config().skill_dirs.is_empty());
            recorded
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(Built {
                    model:  format!("{}/{}", child.provider(), child.model()),
                    root:   child.root_session_id().to_owned(),
                    parent: child.to_record().parent_session_id.unwrap_or_default(),
                });
            recorded_ask.store(child.has_human_input(), Ordering::SeqCst);
        });
        let (session, _provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("child result"))])
                .environment(Arc::clone(&environment))
                .observe_children(observer)
                .build();
        let supervisor = supervisor_of(&session);

        let agent_id = spawn(&supervisor, &session, "task");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let (model, root, parent) = {
            let built = seen.lock().unwrap_or_else(PoisonError::into_inner);
            let built = built.first().expect("the factory built the child");
            (
                built.model.clone(),
                built.root.clone(),
                built.parent.clone(),
            )
        };
        assert_eq!(
            model,
            format!("{}/{}", session.provider(), session.model()),
            "a child runs the model its parent resolved to"
        );
        assert_eq!(parent, session.id());
        assert_eq!(
            root,
            session.root_session_id(),
            "a child belongs to its parent's tree, which root-scoped tools key on"
        );
        assert!(
            !asked.load(Ordering::SeqCst),
            "a child carries no human-input provider, so no question tool registers"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_child_is_shown_exactly_the_tools_and_permissions_its_parent_has() {
        /// What the child came out as, for the parent to be compared against.
        struct Inherited {
            tools:      Vec<String>,
            permission: Option<PermissionLevel>,
        }

        let inherited: Arc<Mutex<Option<Inherited>>> = Arc::new(Mutex::new(None));
        let recorder = Arc::clone(&inherited);
        let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
            let mut tools: Vec<String> = child
                .registered_tools()
                .into_iter()
                .map(|tool| tool.definition.name)
                .collect();
            tools.sort();
            *recorder.lock().unwrap_or_else(PoisonError::into_inner) = Some(Inherited {
                tools,
                permission: child.permission_level(),
            });
        });
        // Registered the way an application registers tools, so what travels
        // is the parent's own tool list rather than a profile both sessions
        // happen to share.
        let (client, provider) =
            scripted_client(vec![ScriptedCall::response(text_response("child result"))]);
        let session = testing::builder(client)
            .tools(vec![noop_tool("allowed"), noop_tool("forbidden")])
            .options(CodingAgentOptions {
                permission_level: Some(PermissionLevel::ReadOnly),
                ..CodingAgentOptions::default()
            })
            .tool_middleware(Arc::new(PermissionMiddleware::new(Arc::new(DenyByName(
                "forbidden",
            )))))
            .observe_children(observer)
            .build()
            .expect("the session builds");
        let supervisor = supervisor_of(&session);

        let agent_id = spawn(&supervisor, &session, "task");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let (child_tools, child_permission) = {
            let built = inherited.lock().unwrap_or_else(PoisonError::into_inner);
            let built = built.as_ref().expect("the factory built a child");
            (built.tools.clone(), built.permission)
        };
        assert_eq!(
            child_tools,
            tool_names(&session),
            "a child is shown its parent's tools, no more and no fewer"
        );
        assert!(
            child_tools.iter().any(|name| name == "allowed"),
            "the parent's own tools travel with it: {child_tools:?}"
        );
        assert!(child_tools.iter().any(|name| name == "forbidden"));
        let requests = provider.requests();
        let request = requests.first().expect("the child asked its model");
        assert!(
            request.tools().iter().all(|tool| tool.name != "forbidden"),
            "the parent's middleware filters the child's request: {:?}",
            request.tools()
        );
        assert_eq!(child_permission, Some(PermissionLevel::ReadOnly));
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_child_records_the_session_that_spawned_it() {
        let built: Arc<Mutex<Option<SessionRecord>>> = Arc::new(Mutex::new(None));
        let recorder = Arc::clone(&built);
        let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
            *recorder.lock().unwrap_or_else(PoisonError::into_inner) = Some(child.to_record());
        });
        let (session, _provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("child result"))])
                .observe_children(observer)
                .build();
        let supervisor = supervisor_of(&session);

        let agent_id = spawn(&supervisor, &session, "task");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let record = built
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .expect("the child was built");
        assert_eq!(record.parent_session_id.as_deref(), Some(session.id()));
        assert!(session.to_record().parent_session_id.is_none());

        // A resumed child says the same thing when it is stored again, even
        // though the tree above it is the application's to rebuild.
        let (client, _resumed_provider) =
            scripted_client(vec![ScriptedCall::response(text_response("resumed"))]);
        let mut resumed = CodingRuntime::from_record(
            record.clone(),
            &ResumeMode::RecordedModel,
            testing::builder(client),
        )
        .expect("the child's record resumes");
        assert_eq!(
            resumed.to_record().parent_session_id.as_deref(),
            Some(session.id())
        );
        assert!(resumed.subagent_supervisor().is_none());
        resumed
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the resumed child shuts down");
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_child_that_fails_reaches_its_parent_as_a_failure() {
        let (callback, captured) = captured_events();
        // An answer that is not an answer: the turn fails, and what the parent
        // is told carries the projection rather than the live error.
        let (parent, supervisor) = parent_over(vec![""]);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &parent, "task");

        supervisor
            .wait(&agent_id)
            .await
            .expect_err("an empty answer is a failed turn");

        let events = lifecycle(&captured);
        assert!(
            events.iter().any(|event| matches!(
                event,
                CodingEvent::SubAgentFailed {
                    depth: 1,
                    generation: 1,
                    error,
                    ..
                } if error.kind == ErrorKind::InvalidState
                    && error.message.contains("without a non-empty final response")
            )),
            "{events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, CodingEvent::SubAgentCompleted { .. })),
            "a failure is not a completion: {events:?}"
        );
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Finished { reusable: true })
        ));
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_waiter_is_told_which_kind_of_failure_ended_its_child() {
        // `ToolError` speaks a narrower vocabulary than `ErrorKind`, and this
        // is where the projection is fitted to it: a child that was interrupted
        // — cancelled, or out of wall-clock time — is a cancellation to
        // whoever waited on it, and every other failure is an execution error.
        let (_parent, supervisor) = parent_over(vec!["unused"]);
        let timed_out = tokio::spawn(async {
            Err::<SubagentResult, Error>(Error::Interrupted(InterruptReason::WallClockTimeout))
        });
        supervisor.supervise_test_task("timed-out".to_owned(), timed_out, CancellationToken::new());
        let broken = tokio::spawn(async {
            Err::<SubagentResult, Error>(Error::ToolExecution("the tool gave up".to_owned()))
        });
        supervisor.supervise_test_task("broken".to_owned(), broken, CancellationToken::new());

        let interrupted = supervisor
            .wait("timed-out")
            .await
            .expect_err("an interrupted child has no result to give");
        assert_eq!(interrupted.kind(), ToolErrorKind::Cancelled);
        assert_eq!(interrupted.message(), "interrupted: wall clock timeout");

        let failed = supervisor
            .wait("broken")
            .await
            .expect_err("a failed child has no result to give");
        assert_eq!(failed.kind(), ToolErrorKind::Execution);
        assert_eq!(failed.message(), "tool execution error: the tool gave up");

        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_session_with_subagents_enabled_is_given_the_tools_to_delegate() {
        let (session, _provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("child result"))])
                .with_subagents()
                .build();
        let supervisor = supervisor_of(&session);

        let names = tool_names(&session);
        for expected in ["spawn_agent", "send_input", "wait", "close_agent"] {
            assert!(names.iter().any(|name| name == expected), "{names:?}");
        }
        assert_eq!(supervisor.open_sessions(), 1);
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_session_with_no_factory_registers_no_subagent_tools() {
        let (session, _provider) = TestSession::answering(vec![ScriptedCall::response(
            text_response("no children here"),
        )]);

        assert!(session.subagent_supervisor().is_none());
        assert!(
            !session
                .registered_tools()
                .iter()
                .map(ToolDefinitionWithSource::to_tool_summary)
                .any(|summary| summary.name == "spawn_agent")
        );
    }

    #[tokio::test]
    async fn the_tool_descriptions_explain_the_delegation_lifecycle() {
        let (session, _provider) = TestSession::new(Vec::new()).with_subagents().build();
        let tools = subagent_tools(&supervisor_of(&session));

        let spawn = tool_named(&tools, "spawn_agent");
        let send = tool_named(&tools, "send_input");
        let wait = tool_named(&tools, "wait");
        let close = tool_named(&tools, "close_agent");

        assert!(spawn.definition.description.contains("independent work"));
        assert!(spawn.definition.description.contains("context isolation"));
        assert!(send.definition.description.contains("follow-up"));
        assert!(send.definition.description.contains("completed agent"));
        assert!(send.definition.description.contains("same session"));
        assert!(wait.definition.description.contains("synthesize"));
        assert!(close.definition.description.contains("no longer needed"));

        // Nothing from the applications fabro's vocabulary grew up in: these
        // tools describe delegation, not a chat client's background shell or a
        // code host's comment API.
        for tool in [spawn, send, wait, close] {
            let text = &tool.definition.description;
            assert!(!text.contains("background Bash"), "{text}");
            assert!(!text.contains("addComment"), "{text}");
        }
    }

    #[tokio::test]
    async fn a_tree_starts_with_one_open_session_and_room_for_three() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);

        assert_eq!(SubagentLimits::default().max_open_sessions, 4);
        assert_eq!(SubagentLimits::new(7).max_open_sessions, 7);
        assert!(supervisor.is_empty(), "a new supervisor has no children");
        assert_eq!(
            supervisor.open_sessions(),
            1,
            "the root counts against its own budget"
        );
    }

    #[tokio::test]
    async fn the_tool_schemas_name_what_each_call_needs() {
        let (session, _provider) = TestSession::new(Vec::new()).with_subagents().build();
        let tools = subagent_tools(&supervisor_of(&session));

        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["spawn_agent", "send_input", "wait", "close_agent"]
        );

        let spawn = schema_of(&tool_named(&tools, "spawn_agent"));
        assert_eq!(spawn["required"], json!(["task"]));
        assert_eq!(
            spawn["properties"]
                .as_object()
                .expect("the schema takes an object")
                .len(),
            1,
            "a spawn takes the task and nothing else"
        );
        assert!(spawn["properties"]["task"].is_object());

        let send = schema_of(&tool_named(&tools, "send_input"));
        assert_eq!(send["required"], json!(["agent_id", "message"]));
        assert!(send["properties"]["agent_id"].is_object());
        assert!(send["properties"]["message"].is_object());

        let wait = schema_of(&tool_named(&tools, "wait"));
        assert_eq!(wait["required"], json!(["agent_id"]));
        assert!(wait["properties"]["agent_id"].is_object());

        let close = schema_of(&tool_named(&tools, "close_agent"));
        assert_eq!(close["required"], json!(["agent_id"]));
        assert!(close["properties"]["agent_id"].is_object());
    }

    #[tokio::test]
    async fn the_tools_answer_the_model_in_the_words_it_was_trained_on() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let tools = subagent_tools(&supervisor);
        let context = || {
            ToolContext::new(Arc::new(MockEnvironment::linux()))
                .with_session(parent.id(), parent.root_session_id())
        };

        let agent_id = (tool_named(&tools, "spawn_agent").executor)(
            json!({ "task": "review the diff" }),
            context(),
        )
        .await
        .expect("the spawn succeeds");
        let sent = (tool_named(&tools, "send_input").executor)(
            json!({ "agent_id": agent_id, "message": "and the tests" }),
            context(),
        )
        .await
        .expect("the message is queued");
        let waited =
            (tool_named(&tools, "wait").executor)(json!({ "agent_id": agent_id }), context())
                .await
                .expect("the child answers");
        let closed = (tool_named(&tools, "close_agent").executor)(
            json!({ "agent_id": agent_id }),
            context(),
        )
        .await
        .expect("the close succeeds");

        assert_eq!(sent, format!("Message sent to agent {agent_id}"));
        assert!(
            waited.starts_with("Agent completed (success: true, turns: "),
            "{waited}"
        );
        assert!(waited.contains("\n\nchild result"), "{waited}");
        assert_eq!(closed, format!("Agent {agent_id} closed"));
    }

    #[tokio::test]
    async fn a_missing_argument_is_the_models_mistake_to_correct() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);
        let tools = subagent_tools(&supervisor);
        let context = ToolContext::new(Arc::new(MockEnvironment::linux()));

        let error = (tool_named(&tools, "wait").executor)(json!({}), context)
            .await
            .expect_err("the call names no agent");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "Missing required parameter: agent_id");
    }

    #[tokio::test]
    async fn cancelling_the_wait_tool_closes_the_child_it_was_waiting_on() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);
        let child_cancel = CancellationToken::new();
        let child_probe = child_cancel.clone();
        let task_cancel = child_cancel.clone();
        let task = tokio::spawn(async move {
            task_cancel.cancelled().await;
            Ok(SubagentResult {
                output:     String::new(),
                success:    false,
                turns_used: 0,
            })
        });
        let agent_id = "blocked-agent".to_owned();
        supervisor.supervise_test_task(agent_id.clone(), task, child_cancel);

        let tools = subagent_tools(&supervisor);
        let tool_cancel = CancellationToken::new();
        let context =
            ToolContext::new(Arc::new(MockEnvironment::linux())).with_cancel(tool_cancel.clone());
        let mut waiting =
            (tool_named(&tools, "wait").executor)(json!({ "agent_id": agent_id }), context);

        assert!(
            poll!(waiting.as_mut()).is_pending(),
            "a blocked child leaves the wait tool pending"
        );
        tool_cancel.cancel();
        let answered = time::timeout(Duration::from_millis(500), waiting.as_mut())
            .await
            .expect("a cancelled wait answers promptly");
        drop(waiting);

        let error = answered.expect_err("a cancelled wait is not a result");
        assert_eq!(error.kind(), ToolErrorKind::Cancelled);
        assert_eq!(error.message(), "Cancelled");
        assert!(child_probe.is_cancelled(), "the child was closed with it");
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
        supervisor.shutdown_all().await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_that_will_not_stop_is_aborted_after_the_grace_period() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);
        let child_cancel = CancellationToken::new();
        let child = tokio::spawn(async {
            pending::<()>().await;
            unreachable!("the task is aborted before it returns")
        });
        let agent_id = "uncooperative".to_owned();
        supervisor.supervise_test_task(agent_id.clone(), child, child_cancel.clone());

        let closer = {
            let supervisor = supervisor.clone();
            let agent_id = agent_id.clone();
            tokio::spawn(async move { supervisor.close_agent(&agent_id).await })
        };
        yield_now().await;

        // Asked to stop, but nothing has stopped: the close waits out the
        // grace period before it takes the tasks down itself.
        assert!(child_cancel.is_cancelled());
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closing)
        ));
        assert!(!closer.is_finished());

        time::advance(SUBAGENT_SHUTDOWN_GRACE).await;
        yield_now().await;
        closer
            .await
            .expect("the closing task finishes")
            .expect("the close succeeds");

        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_aborted_after_the_grace_takes_its_own_children_with_it() {
        // The state pebble reaches and fabro did not: the child that has to be
        // aborted has children of its own, and nothing cooperative runs for
        // them. The abort drops the child's session, the session drops the
        // supervisor it owns, and that drop cancels every grandchild's tokens,
        // aborts its tasks and gives its slot back. A grandchild torn down this
        // way publishes no `SubAgentClosed` — there is no longer anyone to
        // publish it to — which is the difference between this path and a
        // close.
        let (_parent, supervisor) = parent_over(vec!["unused"]);
        let grandchild_dropped = Arc::new(AtomicBool::new(false));
        let probe = Arc::clone(&grandchild_dropped);
        let child = tokio::spawn(async move {
            // A session of its own, with a child of its own, both owned by this
            // task and by nothing else: an application that kept a clone of a
            // child's supervisor would keep that child's children alive.
            let (session, _provider) = TestSession::new(Vec::new()).with_subagents().build();
            let below = session
                .subagent_supervisor()
                .expect("the test session was given a factory")
                .clone();
            let grandchild = tokio::spawn(async move {
                let _probe = DropProbe(probe);
                pending::<()>().await;
                unreachable!("the grandchild is aborted, never finished")
            });
            below.supervise_test_task(
                "grandchild".to_owned(),
                grandchild,
                CancellationToken::new(),
            );
            pending::<()>().await;
            unreachable!("the child is aborted, never finished")
        });
        let agent_id = "uncooperative".to_owned();
        supervisor.supervise_test_task(agent_id.clone(), child, CancellationToken::new());

        let closer = {
            let supervisor = supervisor.clone();
            let agent_id = agent_id.clone();
            tokio::spawn(async move { supervisor.close_agent(&agent_id).await })
        };
        yield_now().await;
        time::advance(SUBAGENT_SHUTDOWN_GRACE).await;
        yield_now().await;
        closer
            .await
            .expect("the closing task finishes")
            .expect("the close succeeds");

        assert!(
            wait_for_flag(&grandchild_dropped).await,
            "an aborted child leaves nothing of its own tree running"
        );
    }

    #[tokio::test]
    async fn two_callers_closing_one_child_wait_for_the_same_cleanup() {
        let (callback, captured) = captured_events();
        let (parent, supervisor) = parent_over(vec!["child result"]);
        supervisor.set_event_callback(callback);
        let agent_id = spawn(&supervisor, &parent, "task");

        let first = supervisor.clone();
        let second = supervisor.clone();
        let id = agent_id.clone();
        let other = agent_id.clone();
        let (left, right) = tokio::join!(async move { first.close_agent(&id).await }, async move {
            second.ensure_closed(&other).await
        },);

        assert!(right.is_ok(), "an idempotent close never fails");
        // Whichever caller followed rather than led is told the agent was
        // already closed; what matters is that the cleanup ran once.
        if let Err(error) = left {
            assert_eq!(
                error.message(),
                format!("Agent {agent_id} is already closed")
            );
        }
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
        let events = lifecycle(&captured);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, CodingEvent::SubAgentClosed { .. }))
                .count(),
            1,
            "one close, announced once: {events:?}"
        );
    }

    #[tokio::test]
    async fn two_shutdowns_of_one_child_converge() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);
        let child_cancel = CancellationToken::new();
        let task_cancel = child_cancel.clone();
        let child = tokio::spawn(async move {
            task_cancel.cancelled().await;
            Ok(SubagentResult {
                output:     String::new(),
                success:    false,
                turns_used: 0,
            })
        });
        let agent_id = "concurrent-close".to_owned();
        supervisor.supervise_test_task(agent_id.clone(), child, child_cancel);

        tokio::join!(supervisor.shutdown_all(), supervisor.shutdown_all());

        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
    }

    #[tokio::test]
    async fn a_lifecycle_callback_may_call_back_into_the_supervisor() {
        let (parent, supervisor) = parent_over(vec!["child result"]);
        let reentrant = supervisor.clone();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        supervisor.set_event_callback(Arc::new(move |event| {
            let (CodingEvent::SubAgentSpawned { agent_id, .. }
            | CodingEvent::SubAgentCompleted { agent_id, .. }) = &event
            else {
                return;
            };
            // Reading state from inside a callback deadlocks unless callbacks
            // run with no lock held — and the completion arrives from the
            // runner's own thread, which is the harder of the two.
            let status = reentrant.status(agent_id);
            recorder
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(status.is_some());
        }));

        let agent_id = spawn(&supervisor, &parent, "task");
        supervisor.wait(&agent_id).await.expect("the child answers");

        assert_eq!(*seen.lock().unwrap_or_else(PoisonError::into_inner), vec![
            true, true
        ]);
        assert!(supervisor.contains(&agent_id));
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_suppressed_child_still_delivers_its_next_generation() {
        let (parent, supervisor) = parent_over(vec!["first result", "second result"]);
        let agent_id = spawn_notifying(&supervisor, &parent, "task", "Inspect the module");
        supervisor.suppress_parent_notification(&agent_id);
        supervisor.wait(&agent_id).await.expect("the child answers");
        assert!(
            supervisor
                .next_parent_notification_batch(&CancellationToken::new())
                .await
                .expect("nothing was cancelled")
                .is_none(),
            "the generation the parent took itself is not delivered again"
        );

        // Suppressing one generation does not cancel the registration: the
        // next turn of the same child is delivered as it always was.
        supervisor
            .send_input(&agent_id, "now fix what you found")
            .expect("a finished child takes another turn");
        supervisor.wait(&agent_id).await.expect("the child answers");

        let batch = supervisor
            .next_parent_notification_batch(&CancellationToken::new())
            .await
            .expect("nothing was cancelled")
            .expect("the second generation is delivered");
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].description, "Inspect the module");
        assert_eq!(
            batch[0].result.as_ref().expect("the turn succeeded").output,
            "second result"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_child_whose_session_ended_is_told_apart_from_one_that_is_closed() {
        let (_parent, supervisor) = parent_over(vec!["unused"]);
        // A supervised task commits its one result as spent, which is what a
        // child whose session ended with its turn looks like.
        let task = tokio::spawn(async {
            Ok(SubagentResult {
                output:     "all done".to_owned(),
                success:    true,
                turns_used: 1,
            })
        });
        let agent_id = "spent-agent".to_owned();
        supervisor.supervise_test_task(agent_id.clone(), task, CancellationToken::new());
        supervisor
            .wait(&agent_id)
            .await
            .expect("the spent child answered");

        let error = supervisor
            .send_input(&agent_id, "more")
            .expect_err("a session that ended takes no further turn");

        assert_eq!(
            error.message(),
            format!("Agent {agent_id} cannot accept more input because its session ended")
        );
        assert_eq!(error.kind(), ToolErrorKind::Execution);
        supervisor.shutdown_all().await;
    }

    /// A parent whose one child takes a whole minute to answer, so the child is
    /// reliably still running when the parent reaches a boundary.
    fn parent_over_a_slow_child() -> (CodingRuntime, SubagentSupervisor) {
        let (session, _provider) =
            TestSession::new(vec![ScriptedCall::response(text_response("late result"))])
                .delayed(Duration::from_secs(60))
                .with_subagents()
                .build();
        let supervisor = supervisor_of(&session);
        (session, supervisor)
    }

    #[tokio::test(start_paused = true)]
    async fn a_parent_waits_at_a_boundary_for_a_child_it_is_expecting() {
        let (parent, supervisor) = parent_over_a_slow_child();
        let agent_id = spawn_notifying(&supervisor, &parent, "task", "Inspect the module");

        let cancel = CancellationToken::new();
        let mut waiting = Box::pin(supervisor.next_parent_notification_batch(&cancel));
        assert!(
            poll!(waiting.as_mut()).is_pending(),
            "a result that is still coming is worth waiting for"
        );

        let batch = time::timeout(Duration::from_secs(120), waiting)
            .await
            .expect("the child's completion wakes the parent")
            .expect("nothing was cancelled")
            .expect("the child is delivered once it finishes");

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].agent_id, agent_id);
        assert_eq!(
            batch[0].result.as_ref().expect("the turn succeeded").output,
            "late result"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_cancelled_boundary_stops_waiting_for_a_child() {
        let (parent, supervisor) = parent_over_a_slow_child();
        spawn_notifying(&supervisor, &parent, "task", "Inspect the module");

        let cancel = CancellationToken::new();
        let mut waiting = Box::pin(supervisor.next_parent_notification_batch(&cancel));
        assert!(poll!(waiting.as_mut()).is_pending(), "the child is running");
        cancel.cancel();

        let interrupted = time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("a cancelled boundary answers promptly")
            .expect_err("a cancelled wait is not a batch");

        assert!(
            matches!(interrupted, Error::Interrupted(InterruptReason::Cancelled)),
            "{interrupted:?}"
        );
        supervisor.shutdown_all().await;
    }

    #[tokio::test(start_paused = true)]
    async fn closing_a_child_does_not_wait_out_the_grace_period() {
        let (parent, supervisor, children) = parent_over_recording_children(vec!["child result"]);
        let agent_id = spawn(&supervisor, &parent, "task");
        // The child's own supervisor is held here, so one child view of the
        // shared emitter outlives the close. That holds nothing open: a stream
        // ends with its root pump, so a close costs nothing.
        let _child = first_child(&children);
        supervisor.wait(&agent_id).await.expect("the child answers");

        let started = Instant::now();
        supervisor
            .close_agent(&agent_id)
            .await
            .expect("the close succeeds");

        assert!(
            started.elapsed() < SUBAGENT_SHUTDOWN_GRACE,
            "a close waited out the grace period: {:?}",
            started.elapsed()
        );
        assert!(matches!(
            supervisor.status(&agent_id),
            Some(SubagentStatus::Closed)
        ));
    }
}
