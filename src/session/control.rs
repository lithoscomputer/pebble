//! Steering a session that is already running.
//!
//! A running session is busy inside [`Session::run`](crate::Session::run), so
//! everything an application wants to say to it mid-run arrives through a
//! [`SessionControlHandle`]: a cheap clone of the three shared pieces the
//! session and its callers both hold. The handle queues messages and cancels
//! the current round; the session drains the queue at round boundaries.
//!
//! Two gestures are distinct and often confused. [`SessionControlHandle`]
//! *interrupts a round*: the current turn is abandoned and the session picks up
//! whatever is queued. [`Session::interrupt`](crate::Session::interrupt) ends
//! the whole run. Only the second closes the session.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::types::Actor;

/// A steer with the actor who wrote it.
///
/// A convenience for applications that carry text and authorship together;
/// [`SteeringItem`] is what the queue holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SteeringMessage {
    /// What to tell the session.
    pub text:  String,
    /// Who is telling it, when the application tracks that.
    pub actor: Option<Actor>,
}

impl SteeringMessage {
    /// A steer from nobody in particular.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text:  text.into(),
            actor: None,
        }
    }

    /// Records who wrote the steer.
    #[must_use]
    pub fn with_actor(mut self, actor: Actor) -> Self {
        self.actor = Some(actor);
        self
    }
}

/// One queued item waiting for the next round boundary.
///
/// The three kinds differ in what the model is told they are: guidance that
/// stays visibly separate from the conversation, another user turn, or a system
/// note. Only [`Steering`](Self::Steering) publishes
/// [`SteeringInjected`](crate::AgentEvent::SteeringInjected); the other two are
/// how a surrounding system writes into the conversation without claiming a
/// person said it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SteeringItem {
    /// Guidance for the assistant, kept distinct from the user's own turns.
    Steering {
        /// What to tell the assistant.
        text:  String,
        /// Who wrote it, when the application tracks that.
        actor: Option<Actor>,
    },
    /// Text committed as a real user turn.
    User {
        /// The user's words.
        text: String,
    },
    /// Text committed as a system turn.
    System {
        /// The note for the model.
        text: String,
    },
}

impl SteeringItem {
    /// A steer with no named author.
    #[must_use]
    pub fn steering(text: impl Into<String>) -> Self {
        Self::Steering {
            text:  text.into(),
            actor: None,
        }
    }

    /// Who wrote this item, for the kinds that name an author.
    #[must_use]
    pub fn actor(&self) -> Option<&Actor> {
        match self {
            Self::Steering { actor, .. } => actor.as_ref(),
            Self::User { .. } | Self::System { .. } => None,
        }
    }

    /// The text this item commits.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Steering { text, .. } | Self::User { text } | Self::System { text } => text,
        }
    }
}

impl From<SteeringMessage> for SteeringItem {
    fn from(message: SteeringMessage) -> Self {
        Self::Steering {
            text:  message.text,
            actor: message.actor,
        }
    }
}

/// Everything the control plane and the loop share, under one lock.
///
/// The two generation counters are the exactly-once interrupt ledger: every
/// gesture raises `interrupt_generation`, and the loop raises
/// `settled_interrupt_generation` to match as it publishes one
/// [`RoundInterrupted`](crate::AgentEvent::RoundInterrupted) per generation it
/// has not settled yet. Several interrupts before the loop unwinds therefore
/// produce one event each, never two for one gesture and never none.
#[derive(Debug, Default)]
pub(crate) struct ControlState {
    pub(crate) queue: VecDeque<SteeringItem>,
    pub(crate) waiting_for_steer: bool,
    pub(crate) interrupt_generation: u64,
    pub(crate) settled_interrupt_generation: u64,
}

/// Decides whether a finished turn really ends the run.
///
/// A session that answers with no tool calls is done, unless something outside
/// it knows a steer is about to arrive. An application that feeds steering from
/// another task installs a coordinator so the race is decided by whoever owns
/// the steering source rather than by timing.
///
/// The contract is the caller's to keep: once
/// [`on_natural_completion`](Self::on_natural_completion) answers `false`, no
/// further steer may reach the queue for this run, because the session is on
/// its way out and would never drain it.
pub trait CompletionCoordinator: Send + Sync {
    /// Whether the session should run one more round.
    ///
    /// `true` sends the loop around again, which drains anything queued.
    fn on_natural_completion(&self) -> bool;
}

/// The handle that steers and interrupts a running session.
///
/// Cloning is cheap, and every clone drives the same session:
/// [`Session::control_handle`](crate::Session::control_handle) builds one from
/// the state the session itself holds. An unattached handle from
/// [`SessionControlHandle::new`] drives nothing, which is what a test or a
/// half-built application wants.
///
/// Every method returns as soon as the item is queued. Delivery happens at the
/// session's next round boundary.
#[derive(Clone, Debug)]
pub struct SessionControlHandle {
    control:     Arc<Mutex<ControlState>>,
    round_token: Arc<RwLock<CancellationToken>>,
    notify:      Arc<Notify>,
}

impl Default for SessionControlHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionControlHandle {
    /// A handle attached to no session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            control:     Arc::new(Mutex::new(ControlState::default())),
            round_token: Arc::new(RwLock::new(CancellationToken::new())),
            notify:      Arc::new(Notify::new()),
        }
    }

    /// Builds the handle that shares one session's control state.
    pub(crate) fn attached(
        control: Arc<Mutex<ControlState>>,
        round_token: Arc<RwLock<CancellationToken>>,
        notify: Arc<Notify>,
    ) -> Self {
        Self {
            control,
            round_token,
            notify,
        }
    }

    /// Queues guidance for the next round, and wakes a parked session.
    pub fn steer(&self, text: impl Into<String>, actor: Option<Actor>) {
        self.enqueue(SteeringItem::Steering {
            text: text.into(),
            actor,
        });
    }

    /// Abandons the current round.
    ///
    /// With nothing queued the session parks at the next round boundary and
    /// waits for a steer, so an operator can stop a run mid-thought and decide
    /// what to say afterwards. The gesture is counted, so the session publishes
    /// exactly one [`RoundInterrupted`](crate::AgentEvent::RoundInterrupted)
    /// for it.
    ///
    /// This does not end the run.
    /// [`Session::interrupt`](crate::Session::interrupt) does that.
    ///
    /// No author is taken, because nothing records one: an interrupt is
    /// announced as a generation, not as something somebody said. Where the
    /// author matters, [`interrupt_then_steer`](Self::interrupt_then_steer)
    /// carries it on the steer.
    pub fn interrupt(&self) {
        {
            let mut control = self.lock();
            control.interrupt_generation = control.interrupt_generation.saturating_add(1);
            if control.queue.is_empty() {
                control.waiting_for_steer = true;
            }
        }
        self.cancel_round();
        self.notify.notify_waiters();
    }

    /// Abandons the current round and delivers `text` as its replacement.
    pub fn interrupt_then_steer(&self, text: impl Into<String>, actor: Option<Actor>) {
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
    pub fn park_for_steer(&self) {
        let mut control = self.lock();
        if control.queue.is_empty() {
            control.waiting_for_steer = true;
        }
    }

    /// Queues one item for the next round.
    pub fn enqueue(&self, item: SteeringItem) {
        {
            let mut control = self.lock();
            control.waiting_for_steer = false;
            control.queue.push_back(item);
        }
        self.notify.notify_waiters();
    }

    /// Queues one item, dropping the oldest to stay within `cap`.
    ///
    /// Answers with whatever was evicted, so a caller can report what the
    /// session will never see.
    #[must_use]
    pub fn enqueue_bounded(&self, item: SteeringItem, cap: usize) -> Option<SteeringItem> {
        let evicted = {
            let mut control = self.lock();
            let evicted = (control.queue.len() >= cap)
                .then(|| control.queue.pop_front())
                .flatten();
            control.waiting_for_steer = false;
            control.queue.push_back(item);
            evicted
        };
        self.notify.notify_waiters();
        evicted
    }

    /// Queues one item only while the queue is under `cap`, answering whether
    /// it was taken.
    ///
    /// The opposite trade to [`enqueue_bounded`](Self::enqueue_bounded): what
    /// is already queued is kept and the new item is refused.
    #[must_use]
    pub fn try_enqueue_bounded(&self, item: SteeringItem, cap: usize) -> bool {
        {
            let mut control = self.lock();
            if control.queue.len() >= cap {
                return false;
            }
            control.queue.push_back(item);
            control.waiting_for_steer = false;
        }
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
    pub fn interrupt_then_enqueue_bounded(
        &self,
        item: SteeringItem,
        cap: usize,
    ) -> Option<SteeringItem> {
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
        self.cancel_round();
        self.notify.notify_waiters();
        evicted
    }

    /// Whether nothing is queued.
    #[must_use]
    pub fn queue_is_empty(&self) -> bool {
        self.lock().queue.is_empty()
    }

    /// Whether the session still has control work to do: something queued, or
    /// a park waiting to be filled.
    #[must_use]
    pub fn has_pending_control_work(&self) -> bool {
        let control = self.lock();
        !control.queue.is_empty() || control.waiting_for_steer
    }

    /// Whether the session is parked waiting for a steer.
    #[must_use]
    pub fn is_waiting_for_steer(&self) -> bool {
        self.lock().waiting_for_steer
    }

    /// How many items are queued.
    ///
    /// For diagnostics. A caller enforcing a bound wants
    /// [`enqueue_bounded`](Self::enqueue_bounded), which decides under the same
    /// lock it counts with.
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.lock().queue.len()
    }

    fn interrupt_then_enqueue(&self, item: SteeringItem) {
        {
            let mut control = self.lock();
            control.interrupt_generation = control.interrupt_generation.saturating_add(1);
            control.queue.push_back(item);
            control.waiting_for_steer = false;
        }
        self.cancel_round();
        self.notify.notify_waiters();
    }

    /// Cancels the round the session is in right now.
    ///
    /// The cell is shared; the session swaps a fresh token into it as each
    /// round starts, so this always reaches the live round.
    fn cancel_round(&self) {
        self.round_token
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .cancel();
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
    fn an_interrupt_cancels_the_round_the_session_is_in() {
        let handle = SessionControlHandle::new();
        let round = handle.round_token.read().expect("not poisoned").clone();

        handle.interrupt();

        assert!(round.is_cancelled());
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
    fn parking_claims_a_session_without_cancelling_its_round() {
        let handle = SessionControlHandle::new();
        let round = handle.round_token.read().expect("not poisoned").clone();

        handle.park_for_steer();

        assert!(handle.is_waiting_for_steer());
        assert!(!round.is_cancelled());
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
        let round = handle.round_token.read().expect("not poisoned").clone();
        handle.enqueue(SteeringItem::steering("first"));

        let evicted = handle.interrupt_then_enqueue_bounded(SteeringItem::steering("second"), 1);

        assert_eq!(evicted.as_ref().map(SteeringItem::text), Some("first"));
        assert!(round.is_cancelled());
        assert_eq!(handle.lock().interrupt_generation, 1);
        assert!(!handle.is_waiting_for_steer());
    }

    #[test]
    fn only_a_steer_names_an_author() {
        let steer = SteeringItem::Steering {
            text:  "guidance".to_owned(),
            actor: Some(Actor::System),
        };

        assert_eq!(steer.actor(), Some(&Actor::System));
        assert_eq!(
            SteeringItem::User {
                text: "hello".to_owned(),
            }
            .actor(),
            None
        );
        assert_eq!(
            SteeringItem::System {
                text: "note".to_owned(),
            }
            .actor(),
            None
        );
    }

    #[test]
    fn a_steering_message_becomes_a_steering_item() {
        let message = SteeringMessage::new("wrap up").with_actor(Actor::User {
            id:           Some("u_1".to_owned()),
            display_name: None,
        });

        let item = SteeringItem::from(message);

        assert_eq!(item.text(), "wrap up");
        assert!(matches!(item.actor(), Some(Actor::User { .. })));
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
