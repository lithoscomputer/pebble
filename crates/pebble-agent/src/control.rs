//! Thread-safe control of an active agent prompt.

use std::collections::VecDeque;
use std::fmt;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::UserMessage;

pub(crate) struct Control {
    state:  Mutex<ControlState>,
    idle:   Notify,
    resume: Notify,
}

struct ControlState {
    running: bool,
    closed: bool,
    paused: bool,
    steering: VecDeque<UserMessage>,
    follow_up: VecDeque<UserMessage>,
    prompt_cancel: CancellationToken,
    round_cancel: CancellationToken,
    interrupt_generation: u64,
    settled_interrupt_generation: u64,
    completion_holds: usize,
}

impl ControlState {
    /// Whether the prompt waits at its next turn boundary for steering.
    ///
    /// A park only holds while the steering queue is empty. `interrupt` and
    /// `park_for_steer` never set `paused` with steering queued, and queueing
    /// steering clears it, so the queue check is the same guard fabro places
    /// at its wait point: steering that is already queued is the answer the
    /// park was waiting for.
    fn is_parked(&self) -> bool {
        self.paused && self.steering.is_empty()
    }
}

impl Control {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state:  Mutex::new(ControlState {
                running: false,
                closed: false,
                paused: false,
                steering: VecDeque::new(),
                follow_up: VecDeque::new(),
                prompt_cancel: CancellationToken::new(),
                round_cancel: CancellationToken::new(),
                interrupt_generation: 0,
                settled_interrupt_generation: 0,
                completion_holds: 0,
            }),
            idle:   Notify::new(),
            resume: Notify::new(),
        })
    }

    pub(crate) fn begin_prompt(
        &self,
        parent_cancel: Option<&CancellationToken>,
    ) -> Option<CancellationToken> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return None;
        }
        debug_assert!(!state.running, "a mutable agent cannot start two prompts");
        state.running = true;
        state.paused = false;
        state.prompt_cancel =
            parent_cancel.map_or_else(CancellationToken::new, CancellationToken::child_token);
        state.round_cancel = CancellationToken::new();
        Some(state.prompt_cancel.clone())
    }

    pub(crate) fn finish_prompt(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.running = false;
        state.paused = false;
        state.round_cancel = CancellationToken::new();
        state.settled_interrupt_generation = state.interrupt_generation;
        drop(state);
        self.idle.notify_waiters();
        self.resume.notify_waiters();
    }

    pub(crate) fn begin_round(&self) -> (CancellationToken, Vec<UserMessage>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.round_cancel = CancellationToken::new();
        let cancel = state.round_cancel.clone();
        let steering = state.steering.drain(..).collect();
        (cancel, steering)
    }

    pub(crate) fn pop_follow_up(&self) -> Option<UserMessage> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .follow_up
            .pop_front()
    }

    pub(crate) fn clear_queues(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.steering.clear();
        state.follow_up.clear();
        state.paused = false;
    }

    pub(crate) fn close(&self) -> bool {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return false;
        }
        state.closed = true;
        state.prompt_cancel.cancel();
        state.round_cancel.cancel();
        drop(state);
        self.idle.notify_waiters();
        self.resume.notify_waiters();
        true
    }

    pub(crate) fn is_running(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .running
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .closed
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_parked()
    }

    pub(crate) fn settle_interrupts(&self) -> Option<RangeInclusive<u64>> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let first = state.settled_interrupt_generation.saturating_add(1);
        let last = state.interrupt_generation;
        state.settled_interrupt_generation = last;
        (first <= last).then_some(first..=last)
    }

    pub(crate) async fn wait_for_completion(
        &self,
        cancel: &CancellationToken,
    ) -> CompletionReadiness {
        loop {
            let notified = self.resume.notified();
            let readiness = {
                let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if !state.steering.is_empty() {
                    CompletionReadiness::SteeringQueued
                } else if state.completion_holds == 0 {
                    CompletionReadiness::Ready
                } else {
                    CompletionReadiness::Waiting
                }
            };
            match readiness {
                CompletionReadiness::Ready | CompletionReadiness::SteeringQueued => {
                    return readiness;
                }
                CompletionReadiness::Waiting => {}
            }
            tokio::select! {
                () = cancel.cancelled() => return CompletionReadiness::Ready,
                () = notified => {}
            }
        }
    }

    pub(crate) async fn wait_until_resumed(&self, cancel: &CancellationToken) -> bool {
        loop {
            let notified = self.resume.notified();
            if !self.is_paused() {
                return true;
            }
            tokio::select! {
                () = cancel.cancelled() => return false,
                () = notified => {}
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompletionReadiness {
    Ready,
    SteeringQueued,
    Waiting,
}

/// What the control did with a message it was asked to queue.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum QueueOutcome {
    /// The message is queued.
    Queued,
    /// The message is queued, and the queue was at its bound: this older
    /// message was dropped to make room and the agent will never see it.
    Evicted(UserMessage),
    /// The bound was zero, so the message could not be queued.
    Rejected(UserMessage),
    /// The agent is closed and nothing was queued.
    Closed,
}

/// A hold on natural prompt completion.
///
/// While a hold exists, a prompt that reaches a natural answer waits for
/// steering. Dropping the final hold wakes the prompt. This closes the race
/// between an external input source deciding to steer and the prompt ending.
#[must_use = "completion is held only while this value exists"]
pub struct CompletionLease {
    control: Arc<Control>,
}

impl fmt::Debug for CompletionLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompletionLease")
            .finish_non_exhaustive()
    }
}

impl Drop for CompletionLease {
    fn drop(&mut self) {
        let wake = {
            let mut state = self
                .control
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            state.completion_holds = state.completion_holds.saturating_sub(1);
            state.completion_holds == 0
        };
        if wake {
            self.control.resume.notify_waiters();
        }
    }
}

/// A read-only snapshot of an agent's control state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentControlSnapshot {
    running:            bool,
    closed:             bool,
    paused:             bool,
    pending_steering:   usize,
    pending_follow_ups: usize,
}

impl AgentControlSnapshot {
    /// Whether a prompt is running.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    /// Whether the agent is closed.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// Whether the prompt is parked waiting for steering.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    /// How many steering messages are queued.
    #[must_use]
    pub const fn pending_steering(&self) -> usize {
        self.pending_steering
    }

    /// How many follow-up messages are queued.
    #[must_use]
    pub const fn pending_follow_ups(&self) -> usize {
        self.pending_follow_ups
    }
}

impl QueueOutcome {
    /// Whether the message was queued, with or without an eviction.
    #[must_use]
    pub const fn is_queued(&self) -> bool {
        matches!(self, Self::Queued | Self::Evicted(_))
    }
}

/// Controls an agent while another task awaits
/// [`Agent::prompt`](crate::Agent::prompt).
///
/// A handle can exist before the agent does:
/// [`detached`](Self::detached) creates one that
/// [`AgentBuilder::control_handle`](crate::AgentBuilder::control_handle) later
/// binds to an agent, so a layer built on this crate can hand out control
/// before it has built the loop the control drives. Steering queued on a
/// detached handle is applied by the first prompt after it is bound.
#[derive(Clone)]
pub struct AgentControlHandle {
    control: Arc<Control>,
}

impl AgentControlHandle {
    pub(crate) fn new(control: Arc<Control>) -> Self {
        Self { control }
    }

    /// A handle bound to no agent yet.
    #[must_use]
    pub fn detached() -> Self {
        Self::new(Control::new())
    }

    pub(crate) fn control(&self) -> Arc<Control> {
        Arc::clone(&self.control)
    }

    /// Queues steering for the next model turn and interrupts the current one.
    ///
    /// Returns `false` when the agent is closed. Steering queued while idle is
    /// applied after the next prompt's user message.
    pub fn steer(&self, message: impl Into<UserMessage>) -> bool {
        self.queue_steering(message.into(), true, None).is_queued()
    }

    /// Queues steering for the next model turn without interrupting this one.
    ///
    /// Returns `false` when the agent is closed.
    pub fn enqueue_steering(&self, message: impl Into<UserMessage>) -> bool {
        self.queue_steering(message.into(), false, None).is_queued()
    }

    /// Queues steering and interrupts the current turn, keeping at most
    /// `capacity` messages queued.
    ///
    /// A full queue drops its oldest message to make room, and the outcome
    /// carries what was dropped. The interrupt and the enqueue happen under one
    /// lock, so the loop can never observe the interrupt with the queue still
    /// empty. A zero capacity rejects the new message.
    pub fn steer_bounded(&self, message: impl Into<UserMessage>, capacity: usize) -> QueueOutcome {
        self.queue_steering(message.into(), true, Some(capacity))
    }

    /// Queues steering without interrupting, keeping at most `capacity`
    /// messages queued. A zero capacity rejects the new message.
    pub fn enqueue_steering_bounded(
        &self,
        message: impl Into<UserMessage>,
        capacity: usize,
    ) -> QueueOutcome {
        self.queue_steering(message.into(), false, Some(capacity))
    }

    fn queue_steering(
        &self,
        message: UserMessage,
        interrupt: bool,
        capacity: Option<usize>,
    ) -> QueueOutcome {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return QueueOutcome::Closed;
        }
        let outcome = push_bounded(&mut state.steering, message, capacity);
        if !outcome.is_queued() {
            return outcome;
        }
        state.paused = false;
        if interrupt && state.running {
            state.interrupt_generation = state.interrupt_generation.saturating_add(1);
            state.round_cancel.cancel();
        }
        drop(state);
        self.control.resume.notify_waiters();
        outcome
    }

    /// Queues input to process after the current prompt reaches an answer.
    ///
    /// Returns `false` when the agent is closed.
    pub fn follow_up(&self, message: impl Into<UserMessage>) -> bool {
        self.queue_follow_up(message.into(), None).is_queued()
    }

    /// Queues follow-up input, keeping at most `capacity` messages queued.
    ///
    /// A full queue drops its oldest message to make room and returns it.
    /// A zero capacity rejects the new message.
    pub fn follow_up_bounded(
        &self,
        message: impl Into<UserMessage>,
        capacity: usize,
    ) -> QueueOutcome {
        self.queue_follow_up(message.into(), Some(capacity))
    }

    fn queue_follow_up(&self, message: UserMessage, capacity: Option<usize>) -> QueueOutcome {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return QueueOutcome::Closed;
        }
        push_bounded(&mut state.follow_up, message, capacity)
    }

    /// Aborts the active prompt.
    ///
    /// Returns whether a prompt was running.
    pub fn abort(&self) -> bool {
        let state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !state.running {
            return false;
        }
        state.prompt_cancel.cancel();
        state.round_cancel.cancel();
        drop(state);
        self.control.resume.notify_waiters();
        true
    }

    /// Interrupts the current turn and waits for steering before another.
    ///
    /// The round is cancelled either way. With no steering queued the prompt
    /// parks at its next turn boundary until a steer arrives. With steering
    /// already queued the prompt does not park: the queued steering opens the
    /// next turn on its own, so a steer followed by an interrupt needs no
    /// second gesture.
    ///
    /// Returns whether a prompt was running.
    pub fn interrupt(&self) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !state.running {
            return false;
        }
        if state.steering.is_empty() {
            state.paused = true;
        }
        state.interrupt_generation = state.interrupt_generation.saturating_add(1);
        state.round_cancel.cancel();
        true
    }

    /// Claims the next turn boundary for steering without interrupting now.
    ///
    /// With steering already queued there is nothing to claim: the queued
    /// steering opens the next turn and the prompt does not park.
    ///
    /// Returns whether a prompt was running.
    pub fn park_for_steer(&self) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !state.running {
            return false;
        }
        if state.steering.is_empty() {
            state.paused = true;
        }
        true
    }

    /// Closes the agent from a handle: the active prompt aborts and no further
    /// prompt runs.
    ///
    /// Returns whether this call performed the transition. Closing through the
    /// handle publishes no event; [`Agent::shutdown`](crate::Agent::shutdown)
    /// does, so a layer that owns the agent shuts it down and closes the
    /// handle only where no agent was ever built.
    pub fn close(&self) -> bool {
        self.control.close()
    }

    /// Holds natural completion open for an external steering source.
    pub fn hold_completion(&self) -> CompletionLease {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state.completion_holds = state.completion_holds.saturating_add(1);
        CompletionLease {
            control: Arc::clone(&self.control),
        }
    }

    /// Waits until no prompt is running.
    pub async fn wait_for_idle(&self) {
        loop {
            let notified = self.control.idle.notified();
            if !self.control.is_running() {
                return;
            }
            notified.await;
        }
    }

    /// Whether a prompt is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.control.is_running()
    }

    /// Whether the agent is closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.control.is_closed()
    }

    /// Whether the prompt is parked after an interrupt, waiting for steering.
    ///
    /// Never `true` while [`pending_steering`](Self::pending_steering) is
    /// above zero: queued steering is what a park waits for.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.control.is_paused()
    }

    /// How many steering messages wait for the next turn boundary.
    #[must_use]
    pub fn pending_steering(&self) -> usize {
        self.control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .steering
            .len()
    }

    /// How many follow-up messages wait for a natural answer.
    #[must_use]
    pub fn pending_follow_ups(&self) -> usize {
        self.control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .follow_up
            .len()
    }

    /// Removes and returns the oldest queued follow-up.
    ///
    /// Embedding layers that supervise prompts can use this after a prompt
    /// returns to close their own completion race. A normal
    /// [`Agent`](crate::Agent) consumes follow-ups itself.
    pub fn take_follow_up(&self) -> Option<UserMessage> {
        self.control.pop_follow_up()
    }

    /// Returns one consistent view of the control state.
    #[must_use]
    pub fn snapshot(&self) -> AgentControlSnapshot {
        let state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        AgentControlSnapshot {
            running:            state.running,
            closed:             state.closed,
            paused:             state.is_parked(),
            pending_steering:   state.steering.len(),
            pending_follow_ups: state.follow_up.len(),
        }
    }
}

impl fmt::Debug for AgentControlHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = self.snapshot();
        formatter
            .debug_struct("AgentControlHandle")
            .field("running", &snapshot.is_running())
            .field("closed", &snapshot.is_closed())
            .field("paused", &snapshot.is_paused())
            .field("pending_steering", &snapshot.pending_steering())
            .field("pending_follow_ups", &snapshot.pending_follow_ups())
            .finish()
    }
}

fn push_bounded(
    queue: &mut VecDeque<UserMessage>,
    message: UserMessage,
    capacity: Option<usize>,
) -> QueueOutcome {
    if capacity == Some(0) {
        return QueueOutcome::Rejected(message);
    }
    let evicted = capacity
        .filter(|capacity| queue.len() >= *capacity)
        .and_then(|_| queue.pop_front());
    queue.push_back(message);
    evicted.map_or(QueueOutcome::Queued, QueueOutcome::Evicted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bounded_queue_evicts_its_oldest_message() {
        let handle = AgentControlHandle::detached();

        assert_eq!(
            handle.enqueue_steering_bounded("first", 2),
            QueueOutcome::Queued
        );
        assert_eq!(
            handle.enqueue_steering_bounded("second", 2),
            QueueOutcome::Queued
        );
        let outcome = handle.enqueue_steering_bounded("third", 2);

        assert_eq!(outcome, QueueOutcome::Evicted(UserMessage::text("first")));
        assert!(outcome.is_queued());
        assert_eq!(handle.pending_steering(), 2);
    }

    #[test]
    fn bounded_follow_ups_share_the_control_snapshot() {
        let handle = AgentControlHandle::detached();
        assert_eq!(handle.follow_up_bounded("first", 1), QueueOutcome::Queued);

        let outcome = handle.follow_up_bounded("second", 1);

        assert_eq!(outcome, QueueOutcome::Evicted(UserMessage::text("first")));
        assert_eq!(handle.snapshot().pending_follow_ups(), 1);
        assert_eq!(handle.take_follow_up(), Some(UserMessage::text("second")));
    }

    #[test]
    fn interrupt_generations_settle_once_in_order() {
        let handle = AgentControlHandle::detached();
        let _prompt_cancel = handle
            .control
            .begin_prompt(None)
            .expect("an open control starts a prompt");

        assert!(handle.interrupt());
        assert!(handle.steer("continue"));

        assert_eq!(handle.control.settle_interrupts(), Some(1..=2));
        assert_eq!(handle.control.settle_interrupts(), None);
    }

    #[test]
    fn a_zero_bound_rejects_the_message() {
        let handle = AgentControlHandle::detached();

        assert_eq!(
            handle.enqueue_steering_bounded("never queued", 0),
            QueueOutcome::Rejected(UserMessage::text("never queued"))
        );
        assert_eq!(handle.snapshot().pending_steering(), 0);
    }

    #[test]
    fn completion_holds_are_counted_until_the_last_drop() {
        let handle = AgentControlHandle::detached();
        let first = handle.hold_completion();
        let second = handle.hold_completion();
        let hold_count = || {
            handle
                .control
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .completion_holds
        };

        assert_eq!(hold_count(), 2);
        drop(first);
        assert_eq!(hold_count(), 1);
        drop(second);
        assert_eq!(hold_count(), 0);
    }

    #[test]
    fn a_detached_handle_queues_but_cannot_interrupt() {
        let handle = AgentControlHandle::detached();

        assert!(handle.steer("later"));
        assert!(!handle.interrupt(), "nothing is running to interrupt");
        assert!(!handle.is_paused());
        assert_eq!(handle.pending_steering(), 1);
    }

    #[test]
    fn a_park_never_holds_while_steering_is_queued() {
        let handle = AgentControlHandle::detached();
        let _prompt_cancel = handle
            .control
            .begin_prompt(None)
            .expect("an open control starts a prompt");

        assert!(handle.interrupt());
        assert!(handle.is_paused(), "a bare interrupt parks");
        assert!(handle.enqueue_steering("change course"));
        assert!(!handle.is_paused(), "queued steering releases the park");

        assert!(handle.park_for_steer());
        assert!(
            !handle.is_paused(),
            "a park claimed with steering queued does not hold"
        );
        assert!(handle.interrupt());
        assert!(
            !handle.is_paused(),
            "an interrupt with steering queued does not park"
        );
        assert_eq!(handle.pending_steering(), 1);

        let (_, drained) = handle.control.begin_round();
        assert_eq!(drained.len(), 1);
        assert!(handle.park_for_steer());
        assert!(handle.is_paused(), "an empty queue parks again");
    }

    #[test]
    fn attribution_rides_along_with_a_queued_message() {
        let handle = AgentControlHandle::detached();
        let message = UserMessage::text("with a byline").with_attribution(serde_json::json!({
            "kind": "system"
        }));

        handle.enqueue_steering(message.clone());
        let (_, drained) = handle.control.begin_round();

        assert_eq!(drained, vec![message]);
        assert_eq!(
            drained[0].attribution(),
            Some(&serde_json::json!({"kind": "system"}))
        );
    }
}
