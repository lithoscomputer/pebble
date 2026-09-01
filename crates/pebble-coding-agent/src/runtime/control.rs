//! Steering a coding agent that is already running.
//!
//! The public [`CodingAgentControlHandle`](crate::CodingAgentControlHandle)
//! uses this internal handle to queue input and interrupt the current model
//! turn. The coding bridge commits the queued message and its durable event at
//! the next turn boundary.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pebble_agent::AgentControlHandle;
use tokio::sync::Notify;

use crate::types::Actor;

/// One queued item waiting for the next round boundary.
///
/// Steering stays visibly separate from the user's conversation turns and
/// publishes [`SteeringInjected`](crate::events::CodingEvent::SteeringInjected).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum SteeringItem {
    /// Guidance for the assistant, kept distinct from the user's own turns.
    Steering {
        /// What to tell the assistant.
        text:  String,
        /// Who wrote it, when the application tracks that.
        actor: Option<Actor>,
    },
}

impl SteeringItem {
    /// A steer with no named author.
    #[must_use]
    pub(crate) fn steering(text: impl Into<String>) -> Self {
        Self::Steering {
            text:  text.into(),
            actor: None,
        }
    }

    /// The text this item commits.
    #[must_use]
    pub(crate) fn text(&self) -> &str {
        match self {
            Self::Steering { text, .. } => text,
        }
    }
}

/// Coding metadata shared by the control plane and the agent bridge.
///
/// The two generation counters are the exactly-once interrupt ledger: every
/// gesture raises `interrupt_generation`, and the loop raises
/// `settled_interrupt_generation` to match as it publishes one
/// [`RoundInterrupted`](crate::events::CodingEvent::RoundInterrupted) per
/// generation it has not settled yet. Several interrupts before the loop
/// unwinds therefore produce one event each, never two for one gesture and
/// never none.
#[derive(Debug, Default)]
pub(crate) struct ControlState {
    pub(crate) queue: VecDeque<SteeringItem>,
    pub(crate) waiting_for_steer: bool,
    pub(crate) interrupt_generation: u64,
    pub(crate) settled_interrupt_generation: u64,
}

/// Decides whether a finished turn really ends the prompt.
///
/// A session that answers with no tool calls is done, unless something outside
/// it knows a steer is about to arrive. An application that feeds steering from
/// another task installs a coordinator so the race is decided by whoever owns
/// the steering source rather than by timing.
///
/// The contract is the caller's to keep: once
/// [`on_natural_completion`](Self::on_natural_completion) answers `false`, no
/// further steer may reach the queue for this prompt, because the session is on
/// its way out and would never drain it.
pub trait CompletionCoordinator: Send + Sync {
    /// Whether the session should run one more round.
    ///
    /// `true` sends the loop around again, which drains anything queued.
    fn on_natural_completion(&self) -> bool;
}

/// The internal handle that steers and interrupts a running coding agent.
#[derive(Clone, Debug)]
pub(crate) struct SessionControlHandle {
    control:      Arc<Mutex<ControlState>>,
    notify:       Arc<Notify>,
    active_agent: Arc<Mutex<Option<AgentControlHandle>>>,
}

impl Default for SessionControlHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionControlHandle {
    /// A handle attached to no session.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            control:      Arc::new(Mutex::new(ControlState::default())),
            notify:       Arc::new(Notify::new()),
            active_agent: Arc::new(Mutex::new(None)),
        }
    }

    /// Builds the handle that shares one session's control state.
    pub(crate) fn attached(
        control: Arc<Mutex<ControlState>>,
        notify: Arc<Notify>,
        active_agent: Arc<Mutex<Option<AgentControlHandle>>>,
    ) -> Self {
        Self {
            control,
            notify,
            active_agent,
        }
    }

    /// Queues guidance for the next round, and wakes a parked session.
    pub(crate) fn steer(&self, text: impl Into<String>, actor: Option<Actor>) {
        self.enqueue(SteeringItem::Steering {
            text: text.into(),
            actor,
        });
    }

    /// Abandons the current round.
    ///
    /// With nothing queued the session parks at the next round boundary and
    /// waits for a steer, so an operator can stop a prompt mid-thought and
    /// decide what to say afterwards. The gesture is counted, so the
    /// session publishes exactly one
    /// [`RoundInterrupted`](crate::events::CodingEvent::RoundInterrupted) for
    /// it.
    ///
    /// No author is taken, because nothing records one: an interrupt is
    /// announced as a generation, not as something somebody said. Where the
    /// author matters, [`interrupt_then_steer`](Self::interrupt_then_steer)
    /// carries it on the steer.
    #[cfg(test)]
    pub(crate) fn interrupt(&self) {
        {
            let mut control = self.lock();
            control.interrupt_generation = control.interrupt_generation.saturating_add(1);
            if control.queue.is_empty() {
                control.waiting_for_steer = true;
            }
        }
        let _ = self.with_active_agent(AgentControlHandle::interrupt);
        self.notify.notify_waiters();
    }

    /// Abandons the current round and delivers `text` as its replacement.
    pub(crate) fn interrupt_then_steer(&self, text: impl Into<String>, actor: Option<Actor>) {
        self.interrupt_then_enqueue(SteeringItem::Steering {
            text: text.into(),
            actor,
        });
    }

    /// Parks the session at the next round boundary without cancelling the
    /// round it is in.
    ///
    /// An external coordinator uses this to claim the session before it
    /// finishes, so a steer it is about to send cannot arrive too late.
    #[cfg(test)]
    pub(crate) fn park_for_steer(&self) {
        let mut control = self.lock();
        if control.queue.is_empty() {
            control.waiting_for_steer = true;
        }
        drop(control);
        let _ = self.with_active_agent(AgentControlHandle::park_for_steer);
    }

    /// Queues one item for the next round.
    pub(crate) fn enqueue(&self, item: SteeringItem) {
        let text = item.text().to_owned();
        let was_waiting = {
            let mut control = self.lock();
            let was_waiting = control.waiting_for_steer;
            control.waiting_for_steer = false;
            control.queue.push_back(item);
            was_waiting
        };
        self.deliver_to_active_agent(text, was_waiting);
        self.notify.notify_waiters();
    }

    /// Queues one item, dropping the oldest to stay within `cap`.
    ///
    /// Answers with whatever was evicted, so a caller can report what the
    /// session will never see.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn enqueue_bounded(&self, item: SteeringItem, cap: usize) -> Option<SteeringItem> {
        let text = item.text().to_owned();
        let (evicted, was_waiting) = {
            let mut control = self.lock();
            let evicted = (control.queue.len() >= cap)
                .then(|| control.queue.pop_front())
                .flatten();
            let was_waiting = control.waiting_for_steer;
            control.waiting_for_steer = false;
            control.queue.push_back(item);
            (evicted, was_waiting)
        };
        self.deliver_to_active_agent(text, was_waiting);
        self.notify.notify_waiters();
        evicted
    }

    /// Queues one item only while the queue is under `cap`, answering whether
    /// it was taken.
    ///
    /// The opposite trade to [`enqueue_bounded`](Self::enqueue_bounded): what
    /// is already queued is kept and the new item is refused.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn try_enqueue_bounded(&self, item: SteeringItem, cap: usize) -> bool {
        let text = item.text().to_owned();
        let was_waiting = {
            let mut control = self.lock();
            if control.queue.len() >= cap {
                return false;
            }
            control.queue.push_back(item);
            let was_waiting = control.waiting_for_steer;
            control.waiting_for_steer = false;
            was_waiting
        };
        self.deliver_to_active_agent(text, was_waiting);
        self.notify.notify_waiters();
        true
    }

    /// Abandons the current round and queues one item within `cap`, in one
    /// step.
    ///
    /// Counting the interrupt and queueing its replacement happen under one
    /// lock, so the session can never observe the interrupt with the queue
    /// still empty and park when a steer was already on its way.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn interrupt_then_enqueue_bounded(
        &self,
        item: SteeringItem,
        cap: usize,
    ) -> Option<SteeringItem> {
        let text = item.text().to_owned();
        let evicted = {
            let mut control = self.lock();
            let evicted = (control.queue.len() >= cap)
                .then(|| control.queue.pop_front())
                .flatten();
            control.interrupt_generation = control.interrupt_generation.saturating_add(1);
            control.queue.push_back(item);
            control.waiting_for_steer = false;
            evicted
        };
        let _ = self.with_active_agent(|agent| agent.steer(text));
        self.notify.notify_waiters();
        evicted
    }

    /// Whether nothing is queued.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn queue_is_empty(&self) -> bool {
        self.lock().queue.is_empty()
    }

    /// Whether the session still has control work to do: something queued, or
    /// a park waiting to be filled.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn has_pending_control_work(&self) -> bool {
        let control = self.lock();
        !control.queue.is_empty() || control.waiting_for_steer
    }

    /// Whether the session is parked waiting for a steer.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn is_waiting_for_steer(&self) -> bool {
        self.lock().waiting_for_steer
    }

    /// How many items are queued.
    ///
    /// For diagnostics. A caller enforcing a bound wants
    /// [`enqueue_bounded`](Self::enqueue_bounded), which decides under the same
    /// lock it counts with.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn queue_len(&self) -> usize {
        self.lock().queue.len()
    }

    fn interrupt_then_enqueue(&self, item: SteeringItem) {
        let text = item.text().to_owned();
        {
            let mut control = self.lock();
            control.interrupt_generation = control.interrupt_generation.saturating_add(1);
            control.queue.push_back(item);
            control.waiting_for_steer = false;
        }
        let _ = self.with_active_agent(|agent| agent.steer(text));
        self.notify.notify_waiters();
    }

    fn deliver_to_active_agent(&self, text: String, interrupt: bool) {
        let _ = self.with_active_agent(|agent| {
            if interrupt {
                agent.steer(text)
            } else {
                agent.enqueue_steering(text)
            }
        });
    }

    fn with_active_agent(&self, use_agent: impl FnOnce(&AgentControlHandle) -> bool) -> bool {
        self.active_agent
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .is_some_and(use_agent)
    }

    fn lock(&self) -> MutexGuard<'_, ControlState> {
        self.control.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::timeout;

    use super::*;

    #[test]
    fn a_steer_queues_and_names_its_author() {
        let handle = SessionControlHandle::new();

        handle.steer("also update the changelog", Some(Actor::System));

        assert_eq!(handle.queue_len(), 1);
        assert!(!handle.queue_is_empty());
        assert!(handle.has_pending_control_work());
        assert!(!handle.is_waiting_for_steer());
    }

    #[test]
    fn a_pure_interrupt_parks_without_queueing_text() {
        let handle = SessionControlHandle::new();

        handle.interrupt();
        handle.interrupt();

        assert!(handle.is_waiting_for_steer());
        assert!(handle.queue_is_empty());
        assert!(handle.has_pending_control_work());
    }

    #[test]
    fn every_interrupt_gesture_raises_a_generation() {
        let handle = SessionControlHandle::new();

        handle.interrupt();
        handle.interrupt_then_steer("stop now", None);

        let control = handle.lock();
        assert_eq!(control.interrupt_generation, 2);
        assert_eq!(control.settled_interrupt_generation, 0);
    }

    #[test]
    fn a_steer_clears_a_park() {
        let handle = SessionControlHandle::new();
        handle.interrupt();

        handle.steer("carry on", None);

        assert!(!handle.is_waiting_for_steer());
        assert_eq!(handle.queue_len(), 1);
    }

    #[test]
    fn parking_claims_a_session() {
        let handle = SessionControlHandle::new();

        handle.park_for_steer();

        assert!(handle.is_waiting_for_steer());
    }

    #[test]
    fn parking_does_nothing_while_work_is_queued() {
        let handle = SessionControlHandle::new();
        handle.steer("first", None);

        handle.park_for_steer();

        assert!(!handle.is_waiting_for_steer());
    }

    #[test]
    fn a_bounded_enqueue_evicts_the_oldest() {
        let handle = SessionControlHandle::new();
        handle.enqueue(SteeringItem::steering("first"));
        handle.enqueue(SteeringItem::steering("second"));

        let evicted = handle.enqueue_bounded(SteeringItem::steering("third"), 2);

        assert_eq!(evicted.as_ref().map(SteeringItem::text), Some("first"));
        assert_eq!(handle.queue_len(), 2);
    }

    #[test]
    fn a_try_enqueue_refuses_instead_of_evicting() {
        let handle = SessionControlHandle::new();
        handle.enqueue(SteeringItem::steering("first"));

        assert!(!handle.try_enqueue_bounded(SteeringItem::steering("second"), 1));
        assert_eq!(handle.queue_len(), 1);
        assert!(handle.try_enqueue_bounded(SteeringItem::steering("second"), 2));
        assert_eq!(handle.queue_len(), 2);
    }

    #[test]
    fn an_interrupting_bounded_enqueue_does_both_at_once() {
        let handle = SessionControlHandle::new();
        handle.enqueue(SteeringItem::steering("first"));

        let evicted = handle.interrupt_then_enqueue_bounded(SteeringItem::steering("second"), 1);

        assert_eq!(evicted.as_ref().map(SteeringItem::text), Some("first"));
        assert_eq!(handle.lock().interrupt_generation, 1);
        assert!(!handle.is_waiting_for_steer());
    }

    #[tokio::test]
    async fn a_queued_steer_wakes_a_waiter_that_registered_first() {
        let handle = SessionControlHandle::new();
        let mut notified = std::pin::pin!(handle.notify.notified());
        // Registering before the queue is read is what makes the wait immune
        // to a steer that lands in between; the loop's park does the same.
        notified.as_mut().enable();

        handle.steer("wake up", None);

        timeout(Duration::from_secs(1), notified)
            .await
            .expect("a queued steer wakes a registered waiter");
    }
}
