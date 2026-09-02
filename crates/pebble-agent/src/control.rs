//! Thread-safe control of an active agent prompt.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::types::Message;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::UserMessage;

pub(crate) struct Control {
    state:  Mutex<ControlState>,
    idle:   Notify,
    resume: Notify,
}

struct ControlState {
    running:       bool,
    closed:        bool,
    paused:        bool,
    steering:      VecDeque<UserMessage>,
    follow_up:     VecDeque<Message>,
    prompt_cancel: CancellationToken,
    round_cancel:  CancellationToken,
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
                running:       false,
                closed:        false,
                paused:        false,
                steering:      VecDeque::new(),
                follow_up:     VecDeque::new(),
                prompt_cancel: CancellationToken::new(),
                round_cancel:  CancellationToken::new(),
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

    pub(crate) fn pop_follow_up(&self) -> Option<Message> {
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

/// What the control did with a message it was asked to queue.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum QueueOutcome {
    /// The message is queued.
    Queued,
    /// The message is queued, and the queue was at its bound: this older
    /// message was dropped to make room and the agent will never see it.
    Evicted(UserMessage),
    /// The agent is closed and nothing was queued.
    Closed,
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
    /// empty.
    pub fn steer_bounded(&self, message: impl Into<UserMessage>, capacity: usize) -> QueueOutcome {
        self.queue_steering(message.into(), true, Some(capacity))
    }

    /// Queues steering without interrupting, keeping at most `capacity`
    /// messages queued.
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
        let evicted = capacity
            .filter(|capacity| state.steering.len() >= *capacity)
            .and_then(|_| state.steering.pop_front());
        state.steering.push_back(message);
        state.paused = false;
        if interrupt && state.running {
            state.round_cancel.cancel();
        }
        drop(state);
        self.control.resume.notify_waiters();
        evicted.map_or(QueueOutcome::Queued, QueueOutcome::Evicted)
    }

    /// Queues input to process after the current prompt reaches an answer.
    ///
    /// Returns `false` when the agent is closed.
    pub fn follow_up(&self, message: impl Into<UserMessage>) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return false;
        }
        state.follow_up.push_back(message.into().into_message());
        true
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
}

impl fmt::Debug for AgentControlHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentControlHandle")
            .field("running", &self.is_running())
            .field("closed", &self.is_closed())
            .field("paused", &self.is_paused())
            .field("pending_steering", &self.pending_steering())
            .finish()
    }
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
