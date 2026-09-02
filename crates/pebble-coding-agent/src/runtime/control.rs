//! Steering a coding agent that is already running.
//!
//! The steering queue itself belongs to the generic agent: every message a
//! coding agent is steered with goes straight into
//! [`AgentControlHandle`]'s queue, carrying its author as an attribution the
//! bridge reads back when the message is committed. What this module adds is
//! the coding layer's own bookkeeping around that queue — the exactly-once
//! interrupt ledger and the completion leases — and the public
//! [`CodingAgentControlHandle`](crate::CodingAgentControlHandle) drives both
//! through [`SessionControlHandle`].

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pebble_agent::{AgentControlHandle, QueueOutcome, UserMessage};
use serde_json::Value;
use tokio::sync::Notify;
use tokio::sync::futures::Notified;

use crate::types::Actor;

/// The coding layer's bookkeeping around the generic agent's steering queue.
///
/// The two generation counters are the exactly-once interrupt ledger: every
/// gesture raises `interrupt_generation`, and the loop raises
/// `settled_interrupt_generation` to match as it publishes one
/// [`RoundInterrupted`](crate::events::CodingEvent::RoundInterrupted) per
/// generation it has not settled yet. Several interrupts before the loop
/// unwinds therefore produce one event each, never two for one gesture and
/// never none.
///
/// `steering_leases` counts the prompt-scoped holds an external steering source
/// keeps on natural completion. While the count is above zero, a prompt that
/// reaches a plain answer parks instead of completing, so a steer already on
/// its way cannot lose the close-door race. The last lease to drop wakes the
/// parked prompt and lets it finish.
#[derive(Debug, Default)]
pub(crate) struct InterruptLedger {
    interrupt_generation:         u64,
    settled_interrupt_generation: u64,
    steering_leases:              usize,
}

/// A prompt-scoped hold that keeps natural completion parked.
///
/// While a lease is alive, a prompt that reaches a plain answer waits instead
/// of finishing, so an external steering source attached to the session cannot
/// lose the completion close-door race. Dropping the last lease wakes the
/// parked prompt and lets it complete. This is the supported replacement for
/// reaching into the drain, park, and generation protocol directly.
#[must_use = "the lease parks completion only while it is held"]
pub struct SteeringLease {
    control: SessionControlHandle,
}

impl SteeringLease {
    /// Takes one hold on the session's completion.
    pub(crate) fn acquire(control: SessionControlHandle) -> Self {
        control.acquire_steering_lease();
        Self { control }
    }
}

impl fmt::Debug for SteeringLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SteeringLease")
            .finish_non_exhaustive()
    }
}

impl Drop for SteeringLease {
    fn drop(&mut self) {
        self.control.release_steering_lease();
    }
}

/// The internal handle that steers and interrupts a running coding agent.
///
/// Cheap to clone. The agent handle inside is the one the session's generic
/// agent is bound to, created before the agent exists so control can be handed
/// out from the moment the session is built.
#[derive(Clone, Debug)]
pub(crate) struct SessionControlHandle {
    agent:  AgentControlHandle,
    ledger: Arc<Mutex<InterruptLedger>>,
    notify: Arc<Notify>,
}

impl SessionControlHandle {
    /// A handle over an agent that does not exist yet.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::attached(
            AgentControlHandle::detached(),
            Arc::default(),
            Arc::default(),
        )
    }

    /// Builds the handle that shares one session's control state.
    pub(crate) fn attached(
        agent: AgentControlHandle,
        ledger: Arc<Mutex<InterruptLedger>>,
        notify: Arc<Notify>,
    ) -> Self {
        Self {
            agent,
            ledger,
            notify,
        }
    }

    /// Queues steering for the next round boundary without interrupting the
    /// round in progress.
    ///
    /// `capacity` bounds the queue; a full queue evicts its oldest message and
    /// the outcome carries it. Wakes a prompt parked on a completion lease so
    /// it drains the queue.
    pub(crate) fn queue_steering(
        &self,
        text: impl Into<String>,
        actor: Option<Actor>,
        capacity: Option<usize>,
    ) -> QueueOutcome {
        let message = steering_message(text, actor);
        let outcome = match capacity {
            Some(capacity) => self.agent.enqueue_steering_bounded(message, capacity),
            None if self.agent.enqueue_steering(message) => QueueOutcome::Queued,
            None => QueueOutcome::Closed,
        };
        self.notify.notify_waiters();
        outcome
    }

    /// Interrupts the round in progress and queues `text` as what replaces it.
    ///
    /// The generic agent cancels the round and queues the message under one
    /// lock, so the loop can never observe the interrupt with an empty queue.
    /// The gesture is counted here only while a prompt is running: with nothing
    /// running there is no round to interrupt, and the message simply opens the
    /// next prompt.
    pub(crate) fn steer_now(
        &self,
        text: impl Into<String>,
        actor: Option<Actor>,
        capacity: Option<usize>,
    ) -> QueueOutcome {
        if self.agent.is_running() {
            self.raise_generation();
        }
        let message = steering_message(text, actor);
        let outcome = match capacity {
            Some(capacity) => self.agent.steer_bounded(message, capacity),
            None if self.agent.steer(message) => QueueOutcome::Queued,
            None => QueueOutcome::Closed,
        };
        self.notify.notify_waiters();
        outcome
    }

    /// Abandons the current round without saying what comes next.
    ///
    /// The prompt parks at its next boundary until a steer arrives, and the
    /// gesture is counted so exactly one
    /// [`RoundInterrupted`](crate::events::CodingEvent::RoundInterrupted) is
    /// published for it. Answers whether a prompt was running to interrupt.
    pub(crate) fn interrupt(&self) -> bool {
        if !self.agent.interrupt() {
            return false;
        }
        self.raise_generation();
        true
    }

    /// Counts a gesture whose round cancellation was lost, so the announcement
    /// it is owed is still published.
    #[cfg(test)]
    pub(crate) fn record_interrupt_without_a_round(&self) {
        self.raise_generation();
    }

    /// The generations raised since the last settlement, oldest first, and
    /// marks them settled.
    pub(crate) fn settle_interrupts(&self) -> Vec<u64> {
        let mut ledger = self.lock();
        let first = ledger.settled_interrupt_generation.saturating_add(1);
        let last = ledger.interrupt_generation;
        ledger.settled_interrupt_generation = last;
        if first <= last {
            (first..=last).collect()
        } else {
            Vec::new()
        }
    }

    /// Takes one prompt-scoped hold on natural completion.
    pub(crate) fn acquire_steering_lease(&self) {
        let mut ledger = self.lock();
        ledger.steering_leases = ledger.steering_leases.saturating_add(1);
    }

    /// Releases one hold taken by
    /// [`acquire_steering_lease`](Self::acquire_steering_lease). The call that
    /// drops the count to zero wakes a prompt parked at its completion
    /// boundary.
    pub(crate) fn release_steering_lease(&self) {
        let woke = {
            let mut ledger = self.lock();
            ledger.steering_leases = ledger.steering_leases.saturating_sub(1);
            ledger.steering_leases == 0
        };
        if woke {
            self.notify.notify_waiters();
        }
    }

    /// How many completion holds are outstanding.
    #[must_use]
    pub(crate) fn steering_lease_count(&self) -> usize {
        self.lock().steering_leases
    }

    /// How many steering messages wait for the next round boundary.
    #[must_use]
    pub(crate) fn pending_steering(&self) -> usize {
        self.agent.pending_steering()
    }

    /// Whether the prompt is parked after an interrupt, waiting for a steer.
    #[must_use]
    pub(crate) fn is_parked(&self) -> bool {
        self.agent.is_paused()
    }

    /// A wake-up that fires when steering is queued or the last lease drops.
    ///
    /// Register it before reading the queue or the lease count, the way the
    /// bridge's completion park does, so a change that lands in between is not
    /// missed.
    pub(crate) fn notified(&self) -> Notified<'_> {
        self.notify.notified()
    }

    /// Queues steering with no bound, for the crate's own tests.
    #[cfg(test)]
    pub(crate) fn steer(&self, text: impl Into<String>, actor: Option<Actor>) {
        let _ = self.queue_steering(text, actor, None);
    }

    /// Interrupts and steers with no bound, for the crate's own tests.
    #[cfg(test)]
    pub(crate) fn interrupt_then_steer(&self, text: impl Into<String>, actor: Option<Actor>) {
        let _ = self.steer_now(text, actor, None);
    }

    fn raise_generation(&self) {
        let mut ledger = self.lock();
        ledger.interrupt_generation = ledger.interrupt_generation.saturating_add(1);
    }

    fn lock(&self) -> MutexGuard<'_, InterruptLedger> {
        self.ledger.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A steering message for the generic agent's queue, carrying its author as
/// an attribution the bridge reads back when the message is committed.
pub(crate) fn steering_message(text: impl Into<String>, actor: Option<Actor>) -> UserMessage {
    let message = UserMessage::text(text);
    match actor.and_then(|actor| serde_json::to_value(actor).ok()) {
        Some(attribution) => message.with_attribution(attribution),
        None => message,
    }
}

/// The author a queued steering message named, if it named one.
pub(crate) fn actor_from_attribution(attribution: Option<&Value>) -> Option<Actor> {
    attribution.and_then(|value| serde_json::from_value(value.clone()).ok())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::timeout;

    use super::*;

    #[test]
    fn a_steer_reaches_the_agents_queue_with_its_author() {
        let handle = SessionControlHandle::new();

        let outcome = handle.queue_steering("also update the changelog", Some(Actor::System), None);

        assert_eq!(outcome, QueueOutcome::Queued);
        assert_eq!(handle.pending_steering(), 1);
        assert!(!handle.is_parked());
    }

    #[test]
    fn an_author_survives_the_round_trip_through_an_attribution() {
        let actor = Actor::User {
            id:           Some("u_1".into()),
            display_name: Some("Ada".into()),
        };

        let message = steering_message("hello", Some(actor.clone()));

        assert_eq!(message.text_content(), "hello");
        assert_eq!(actor_from_attribution(message.attribution()), Some(actor));
        assert_eq!(actor_from_attribution(None), None);
    }

    #[test]
    fn an_interrupt_with_nothing_running_raises_no_generation() {
        let handle = SessionControlHandle::new();

        assert!(!handle.interrupt());
        assert!(handle.settle_interrupts().is_empty());
    }

    #[test]
    fn raised_generations_settle_once_in_order() {
        let handle = SessionControlHandle::new();
        handle.record_interrupt_without_a_round();
        handle.record_interrupt_without_a_round();

        assert_eq!(handle.settle_interrupts(), [1, 2]);
        assert!(
            handle.settle_interrupts().is_empty(),
            "nothing is owed twice"
        );
    }

    #[test]
    fn a_bounded_steer_reports_what_it_evicted() {
        let handle = SessionControlHandle::new();
        let _ = handle.queue_steering("first", None, Some(1));

        let outcome = handle.queue_steering("second", Some(Actor::System), Some(1));

        let QueueOutcome::Evicted(evicted) = outcome else {
            panic!("the bound evicts the oldest message");
        };
        assert_eq!(evicted.text_content(), "first");
        assert_eq!(handle.pending_steering(), 1);
    }

    #[test]
    fn leases_are_counted_and_the_last_drop_clears_the_hold() {
        let handle = SessionControlHandle::new();

        let first = SteeringLease::acquire(handle.clone());
        let second = SteeringLease::acquire(handle.clone());
        assert_eq!(handle.steering_lease_count(), 2);

        drop(first);
        assert_eq!(handle.steering_lease_count(), 1, "one hold still parks");
        drop(second);
        assert_eq!(handle.steering_lease_count(), 0);
    }

    #[tokio::test]
    async fn dropping_the_last_lease_wakes_a_waiter_that_registered_first() {
        let handle = SessionControlHandle::new();
        let lease = SteeringLease::acquire(handle.clone());
        let mut notified = std::pin::pin!(handle.notified());
        notified.as_mut().enable();

        drop(lease);

        timeout(Duration::from_secs(1), notified)
            .await
            .expect("the final release wakes a registered waiter");
    }

    #[tokio::test]
    async fn a_queued_steer_wakes_a_waiter_that_registered_first() {
        let handle = SessionControlHandle::new();
        let mut notified = std::pin::pin!(handle.notified());
        // Registering before the queue is read is what makes the wait immune
        // to a steer that lands in between; the bridge's park does the same.
        notified.as_mut().enable();

        handle.steer("wake up", None);

        timeout(Duration::from_secs(1), notified)
            .await
            .expect("a queued steer wakes a registered waiter");
    }
}
