//! A bus that carries steering to many sessions.
//!
//! An application that runs several sessions at once, or one after another,
//! has one control plane for all of them: a steer or an interrupt from an
//! operator goes to every session that is live, a steer that arrives while
//! none is live waits for the next one, and a party that has attached to a
//! session, a human pairing with it, holds the session open so a plain answer
//! parks instead of ending under them. Both embedders wrote that hub; this
//! is it, once.
//!
//! The bus keys attachments by whatever the application uses to name a
//! session's place, a stage or a node, and remembers which session holds
//! each key so a stale detach cannot remove a successor. It knows nothing
//! about the application's events: every operation returns what it did, the
//! sessions it reached, the steer it buffered, the messages that were
//! dropped, and the application records those in its own vocabulary and
//! order.
//!
//! A message travels in one of two modes. A steer reaches the session at
//! its next round boundary, inside the answer in progress; a follow-up waits
//! and runs as a new user turn once the current answer is reached. The bus
//! remembers the mode of every message it buffers, so text that arrived before
//! a session attached is delivered the way its sender meant.
//!
//! Sessions implement [`SteerableSession`]. Pebble's own
//! [`CodingAgentControlHandle`] does; a session on another backend
//! implements it with an adapter.
//!
//! Lock discipline: `attached` is a read-write lock held for a whole
//! decide-and-push step, `pending` is a mutex taken under it, and no method
//! awaits, so an agent loop can call
//! [`detach_if_idle`](SteeringBus::detach_if_idle) from its completion path.

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use crate::coding_agent::{CodingAgentControlHandle, SteeringMessage, SteeringOutcome};
use crate::types::Actor;

/// A live session the bus can steer.
pub trait SteerableSession: Send + Sync {
    /// Queues `message` for the session's next round boundary. The session
    /// bounds its own queue; a full one evicts its oldest message and
    /// reports it.
    fn steer(&self, message: SteeringMessage) -> SteeringOutcome;

    /// Interrupts the round in progress without saying what comes next.
    /// Returns whether there was a round to interrupt.
    fn interrupt(&self) -> bool;

    /// Interrupts the round in progress and queues `message` as what
    /// replaces it, in one step.
    fn steer_now(&self, message: SteeringMessage) -> SteeringOutcome;

    /// Whether steering the session has not yet read is waiting, or an
    /// interrupt it has not yet acted on.
    fn has_pending_steering(&self) -> bool;

    /// Holds the session's natural completion open while a party is
    /// attached. `None` when the session cannot be held.
    fn hold_open(&self) -> Option<SessionHold> {
        None
    }

    /// Queues `message` to run as a new user turn once the current answer is
    /// reached, rather than inside it. A session that can queue a new turn
    /// should override this; the default steers, which is the nearest thing
    /// a session with one queue can do.
    fn follow_up(&self, message: SteeringMessage) -> SteeringOutcome {
        self.steer(message)
    }
}

impl SteerableSession for CodingAgentControlHandle {
    fn steer(&self, message: SteeringMessage) -> SteeringOutcome {
        self.queue_steering(message)
    }

    fn follow_up(&self, message: SteeringMessage) -> SteeringOutcome {
        self.queue_follow_up(message)
    }

    fn interrupt(&self) -> bool {
        Self::interrupt(self)
    }

    fn steer_now(&self, message: SteeringMessage) -> SteeringOutcome {
        Self::steer_now(self, message)
    }

    fn has_pending_steering(&self) -> bool {
        self.snapshot().pending_steering() > 0
    }

    fn hold_open(&self) -> Option<SessionHold> {
        Some(SessionHold::new(self.hold_open_for_steering()))
    }
}

/// A guard that holds a session's natural completion open until dropped.
pub struct SessionHold {
    _guard: Box<dyn Any + Send + Sync>,
}

impl SessionHold {
    /// Wraps whatever a session releases on drop.
    #[must_use]
    pub fn new(guard: impl Any + Send + Sync) -> Self {
        Self {
            _guard: Box::new(guard),
        }
    }
}

impl fmt::Debug for SessionHold {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionHold")
            .finish_non_exhaustive()
    }
}

/// One session's place on the bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment<K> {
    pub key:        K,
    pub session_id: String,
}

/// How a message reaches a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeliveryMode {
    /// At the session's next round boundary, inside the answer in progress:
    /// [`SteerableSession::steer`].
    Steer,
    /// As a new user turn once the current answer is reached:
    /// [`SteerableSession::follow_up`].
    FollowUp,
}

/// A message waiting on the bus for the next attachment, with the mode its
/// sender chose.
struct Pending {
    mode:    DeliveryMode,
    message: SteeringMessage,
}

/// Why a steer was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DropReason {
    /// A queue was full and this message made room for a newer one.
    QueueFull,
    /// The bus was drained at the end with steers nobody read.
    Ended,
}

/// Steering, or a follow-up, the agent will never see.
#[derive(Debug, Clone, PartialEq)]
pub struct DroppedSteer<K> {
    pub reason:     DropReason,
    pub count:      usize,
    /// Who wrote the dropped message, when one message was dropped and it
    /// named its author.
    pub actor:      Option<Actor>,
    /// The session whose queue dropped it; `None` for the bus's own buffer.
    pub attachment: Option<Attachment<K>>,
}

/// What one steer or follow-up did.
#[derive(Debug, Clone, PartialEq)]
pub struct Delivery<K> {
    /// The sessions that queued the message.
    pub delivered: Vec<Attachment<K>>,
    /// The message waits on the bus for the next attachment.
    pub buffered:  bool,
    pub dropped:   Vec<DroppedSteer<K>>,
}

impl<K> Delivery<K> {
    fn none() -> Self {
        Self {
            delivered: Vec::new(),
            buffered:  false,
            dropped:   Vec::new(),
        }
    }
}

/// What one interrupt did.
#[derive(Debug, Clone, PartialEq)]
pub struct Interruption<K> {
    /// The sessions that were told to stop.
    pub interrupted: Vec<Attachment<K>>,
    pub dropped:     Vec<DroppedSteer<K>>,
}

/// Why a session could not attach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachError {
    /// Another session holds the key.
    Occupied { session_id: String },
}

impl fmt::Display for AttachError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Occupied { session_id } => {
                write!(formatter, "session {session_id} is already attached here")
            }
        }
    }
}

impl StdError for AttachError {}

/// Why an operation on one attachment did not happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetError {
    /// No session, or a different session, is attached at the key.
    NotAttached,
    /// The session cannot hold its completion open.
    Unsupported,
    /// The session is already held.
    AlreadyHeld,
}

impl fmt::Display for TargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotAttached => "no such session is attached",
            Self::Unsupported => "the session cannot be held open",
            Self::AlreadyHeld => "the session is already held open",
        })
    }
}

impl StdError for TargetError {}

struct Attached {
    session_id: String,
    session:    Arc<dyn SteerableSession>,
    hold:       Option<SessionHold>,
}

/// The bus. `K` names a session's place: a stage, a node, a slot.
pub struct SteeringBus<K> {
    attached:         RwLock<BTreeMap<K, Attached>>,
    pending:          Mutex<VecDeque<Pending>>,
    pending_capacity: usize,
}

impl<K> fmt::Debug for SteeringBus<K> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SteeringBus")
            .field("attached", &self.attached_count())
            .field("pending", &self.pending_len())
            .field("pending_capacity", &self.pending_capacity)
            .finish()
    }
}

impl<K> Default for SteeringBus<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> SteeringBus<K> {
    /// How many messages wait on the bus with no session attached before the
    /// oldest is dropped.
    pub const DEFAULT_PENDING_CAPACITY: usize = 32;

    /// A bus with nothing attached and the default buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::with_pending_capacity(Self::DEFAULT_PENDING_CAPACITY)
    }

    /// A bus whose buffer holds `capacity` messages.
    #[must_use]
    pub fn with_pending_capacity(capacity: usize) -> Self {
        Self {
            attached:         RwLock::new(BTreeMap::new()),
            pending:          Mutex::new(VecDeque::new()),
            pending_capacity: capacity.max(1),
        }
    }

    /// How many messages, steers and follow-ups alike, wait for the next
    /// attachment.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// How many sessions are attached.
    #[must_use]
    pub fn attached_count(&self) -> usize {
        self.attached
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Drops every buffered message, as at the end of a run. Returns what was
    /// dropped, if anything was.
    pub fn drain_pending(&self) -> Option<DroppedSteer<K>> {
        let count = {
            let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
            let count = pending.len();
            pending.clear();
            count
        };
        (count > 0).then_some(DroppedSteer {
            reason: DropReason::Ended,
            count,
            actor: None,
            attachment: None,
        })
    }
}

impl<K: Ord + Clone> SteeringBus<K> {
    /// Attaches `session` at `key`. Re-attaching the same session replaces
    /// its handle; a different session is refused while the first holds the
    /// key. Buffered messages stay on the bus until
    /// [`drain_pending_into`](Self::drain_pending_into), so the application
    /// can record the attachment first.
    pub fn attach(
        &self,
        key: K,
        session_id: impl Into<String>,
        session: Arc<dyn SteerableSession>,
    ) -> Result<(), AttachError> {
        let session_id = session_id.into();
        let mut attached = self
            .attached
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        match attached.get_mut(&key) {
            Some(entry) if entry.session_id != session_id => Err(AttachError::Occupied {
                session_id: entry.session_id.clone(),
            }),
            Some(entry) => {
                entry.session = session;
                Ok(())
            }
            None => {
                attached.insert(key, Attached {
                    session_id,
                    session,
                    hold: None,
                });
                Ok(())
            }
        }
    }

    /// Moves every buffered message into the session attached at `key`, in
    /// order, each the way its sender chose: a steer through
    /// [`steer`](SteerableSession::steer), a follow-up through
    /// [`follow_up`](SteerableSession::follow_up). Nothing moves when no
    /// session is attached there.
    pub fn drain_pending_into(&self, key: &K) -> Delivery<K> {
        let attached = self.attached.read().unwrap_or_else(PoisonError::into_inner);
        let Some(entry) = attached.get(key) else {
            return Delivery::none();
        };
        let pending: Vec<Pending> = {
            let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
            pending.drain(..).collect()
        };
        let attachment = Attachment {
            key:        key.clone(),
            session_id: entry.session_id.clone(),
        };
        let mut delivery = Delivery::none();
        for Pending { mode, message } in pending {
            Self::push(
                entry.session.as_ref(),
                mode,
                message,
                &attachment,
                &mut delivery,
            );
        }
        delivery
    }

    /// Detaches the session at `key`. A different session id is a stale
    /// detach and changes nothing. Any hold on the session is released.
    pub fn detach(&self, key: &K, session_id: &str) -> bool {
        let mut attached = self
            .attached
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if attached
            .get(key)
            .is_none_or(|entry| entry.session_id != session_id)
        {
            return false;
        }
        attached.remove(key);
        true
    }

    /// Detaches the session at `key` only if it has no steering waiting: the
    /// close-the-door check an agent loop makes before it lets a prompt end,
    /// atomic against a steer arriving.
    pub fn detach_if_idle(&self, key: &K, session_id: &str) -> bool {
        let mut attached = self
            .attached
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(entry) = attached.get(key) else {
            return false;
        };
        if entry.session_id != session_id || entry.session.has_pending_steering() {
            return false;
        }
        attached.remove(key);
        true
    }

    /// Whether `session_id` is the session attached at `key`.
    #[must_use]
    pub fn is_attached(&self, key: &K, session_id: &str) -> bool {
        self.attached
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .is_some_and(|entry| entry.session_id == session_id)
    }

    /// Every attached session, in key order.
    #[must_use]
    pub fn attachments(&self) -> Vec<Attachment<K>> {
        self.attached
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|(key, entry)| Attachment {
                key:        key.clone(),
                session_id: entry.session_id.clone(),
            })
            .collect()
    }

    /// Steers every attached session, or buffers the message when none is
    /// attached.
    pub fn steer(&self, message: SteeringMessage) -> Delivery<K> {
        self.deliver(DeliveryMode::Steer, message)
    }

    /// Queues `message` as a follow-up on every attached session, to run as
    /// a new user turn once each reaches its current answer, or buffers it
    /// as a follow-up when none is attached.
    pub fn follow_up(&self, message: SteeringMessage) -> Delivery<K> {
        self.deliver(DeliveryMode::FollowUp, message)
    }

    /// Delivers `message` in `mode` to every attached session, or buffers it
    /// with its mode when none is attached.
    fn deliver(&self, mode: DeliveryMode, message: SteeringMessage) -> Delivery<K> {
        let attached = self.attached.read().unwrap_or_else(PoisonError::into_inner);
        let mut delivery = Delivery::none();
        if attached.is_empty() {
            let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
            if pending.len() >= self.pending_capacity
                && let Some(evicted) = pending.pop_front()
            {
                delivery.dropped.push(DroppedSteer {
                    reason:     DropReason::QueueFull,
                    count:      1,
                    actor:      evicted.message.actor().cloned(),
                    attachment: None,
                });
            }
            pending.push_back(Pending { mode, message });
            delivery.buffered = true;
            return delivery;
        }
        for (key, entry) in attached.iter() {
            let attachment = Attachment {
                key:        key.clone(),
                session_id: entry.session_id.clone(),
            };
            Self::push(
                entry.session.as_ref(),
                mode,
                message.clone(),
                &attachment,
                &mut delivery,
            );
        }
        delivery
    }

    /// Interrupts every attached session. An interrupt is not buffered:
    /// with nothing attached there is nothing to stop.
    pub fn interrupt(&self) -> Interruption<K> {
        let attached = self.attached.read().unwrap_or_else(PoisonError::into_inner);
        let mut interrupted = Vec::new();
        for (key, entry) in attached.iter() {
            entry.session.interrupt();
            interrupted.push(Attachment {
                key:        key.clone(),
                session_id: entry.session_id.clone(),
            });
        }
        Interruption {
            interrupted,
            dropped: Vec::new(),
        }
    }

    /// Interrupts every attached session and queues `message` as what
    /// replaces its round, in one step per session. Not buffered.
    pub fn interrupt_then_steer(&self, message: &SteeringMessage) -> Interruption<K> {
        let attached = self.attached.read().unwrap_or_else(PoisonError::into_inner);
        let mut interruption = Interruption {
            interrupted: Vec::new(),
            dropped:     Vec::new(),
        };
        for (key, entry) in attached.iter() {
            let attachment = Attachment {
                key:        key.clone(),
                session_id: entry.session_id.clone(),
            };
            let outcome = entry.session.steer_now(message.clone());
            Self::record_outcome(outcome, &attachment, &mut interruption.dropped);
            interruption.interrupted.push(attachment);
        }
        interruption
    }

    /// Interrupts one attached session. Returns whether it had a round to
    /// stop.
    pub fn interrupt_at(&self, key: &K, session_id: &str) -> Result<bool, TargetError> {
        let attached = self.attached.read().unwrap_or_else(PoisonError::into_inner);
        let entry = attached
            .get(key)
            .filter(|entry| entry.session_id == session_id)
            .ok_or(TargetError::NotAttached)?;
        Ok(entry.session.interrupt())
    }

    /// Steers one attached session. The caller reads the outcome: a paired
    /// human's message that evicted an older steer may not count as
    /// accepted to them.
    pub fn send_to(
        &self,
        key: &K,
        session_id: &str,
        message: SteeringMessage,
    ) -> Result<SteeringOutcome, TargetError> {
        let attached = self.attached.read().unwrap_or_else(PoisonError::into_inner);
        let entry = attached
            .get(key)
            .filter(|entry| entry.session_id == session_id)
            .ok_or(TargetError::NotAttached)?;
        Ok(entry.session.steer(message))
    }

    /// Queues `message` as a follow-up on one attached session, to run as a
    /// new user turn once its current answer is reached. The caller reads the
    /// outcome as for [`send_to`](Self::send_to).
    pub fn follow_up_to(
        &self,
        key: &K,
        session_id: &str,
        message: SteeringMessage,
    ) -> Result<SteeringOutcome, TargetError> {
        let attached = self.attached.read().unwrap_or_else(PoisonError::into_inner);
        let entry = attached
            .get(key)
            .filter(|entry| entry.session_id == session_id)
            .ok_or(TargetError::NotAttached)?;
        Ok(entry.session.follow_up(message))
    }

    /// Holds the session at `key` open: while held, a prompt that reaches a
    /// plain answer parks instead of completing. The hold lasts until
    /// [`release_hold`](Self::release_hold) or the session detaches.
    pub fn hold_open(&self, key: &K, session_id: &str) -> Result<(), TargetError> {
        let mut attached = self
            .attached
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let entry = attached
            .get_mut(key)
            .filter(|entry| entry.session_id == session_id)
            .ok_or(TargetError::NotAttached)?;
        if entry.hold.is_some() {
            return Err(TargetError::AlreadyHeld);
        }
        entry.hold = Some(entry.session.hold_open().ok_or(TargetError::Unsupported)?);
        Ok(())
    }

    /// Releases a hold. Returns whether one was held.
    pub fn release_hold(&self, key: &K, session_id: &str) -> bool {
        let mut attached = self
            .attached
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        attached
            .get_mut(key)
            .filter(|entry| entry.session_id == session_id)
            .and_then(|entry| entry.hold.take())
            .is_some()
    }

    /// Whether the session at `key` is held open.
    #[must_use]
    pub fn is_held(&self, key: &K, session_id: &str) -> bool {
        self.attached
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .is_some_and(|entry| entry.session_id == session_id && entry.hold.is_some())
    }

    fn push(
        session: &dyn SteerableSession,
        mode: DeliveryMode,
        message: SteeringMessage,
        attachment: &Attachment<K>,
        delivery: &mut Delivery<K>,
    ) {
        let outcome = match mode {
            DeliveryMode::Steer => session.steer(message),
            DeliveryMode::FollowUp => session.follow_up(message),
        };
        if outcome.is_accepted() {
            delivery.delivered.push(attachment.clone());
        }
        Self::record_outcome(outcome, attachment, &mut delivery.dropped);
    }

    fn record_outcome(
        outcome: SteeringOutcome,
        attachment: &Attachment<K>,
        dropped: &mut Vec<DroppedSteer<K>>,
    ) {
        if let SteeringOutcome::Evicted(evicted) = outcome {
            dropped.push(DroppedSteer {
                reason:     DropReason::QueueFull,
                count:      1,
                actor:      evicted.actor().cloned(),
                attachment: Some(attachment.clone()),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A session with a bounded steering queue and a follow-up queue,
    /// standing in for a live agent.
    struct FakeSession {
        queue:       Mutex<VecDeque<SteeringMessage>>,
        follow_ups:  Mutex<Vec<SteeringMessage>>,
        capacity:    usize,
        interrupted: AtomicUsize,
        holds:       Arc<AtomicUsize>,
    }

    struct FakeHold(Arc<AtomicUsize>);

    impl Drop for FakeHold {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl FakeSession {
        fn with_capacity(capacity: usize) -> Arc<Self> {
            Arc::new(Self {
                queue: Mutex::new(VecDeque::new()),
                follow_ups: Mutex::new(Vec::new()),
                capacity,
                interrupted: AtomicUsize::new(0),
                holds: Arc::new(AtomicUsize::new(0)),
            })
        }

        fn new() -> Arc<Self> {
            Self::with_capacity(32)
        }

        fn texts(&self) -> Vec<String> {
            self.queue
                .lock()
                .expect("queue lock")
                .iter()
                .map(|message| message.text().to_owned())
                .collect()
        }

        fn follow_up_texts(&self) -> Vec<String> {
            self.follow_ups
                .lock()
                .expect("follow-up lock")
                .iter()
                .map(|message| message.text().to_owned())
                .collect()
        }

        fn interrupts(&self) -> usize {
            self.interrupted.load(Ordering::SeqCst)
        }

        fn held(&self) -> usize {
            self.holds.load(Ordering::SeqCst)
        }
    }

    impl SteerableSession for FakeSession {
        fn steer(&self, message: SteeringMessage) -> SteeringOutcome {
            let mut queue = self.queue.lock().expect("queue lock");
            let evicted = (queue.len() >= self.capacity)
                .then(|| queue.pop_front())
                .flatten();
            queue.push_back(message);
            evicted.map_or(SteeringOutcome::Accepted, SteeringOutcome::Evicted)
        }

        fn interrupt(&self) -> bool {
            self.interrupted.fetch_add(1, Ordering::SeqCst);
            true
        }

        fn steer_now(&self, message: SteeringMessage) -> SteeringOutcome {
            self.interrupt();
            self.steer(message)
        }

        fn has_pending_steering(&self) -> bool {
            !self.queue.lock().expect("queue lock").is_empty()
        }

        fn hold_open(&self) -> Option<SessionHold> {
            self.holds.fetch_add(1, Ordering::SeqCst);
            Some(SessionHold::new(FakeHold(Arc::clone(&self.holds))))
        }

        fn follow_up(&self, message: SteeringMessage) -> SteeringOutcome {
            self.follow_ups
                .lock()
                .expect("follow-up lock")
                .push(message);
            SteeringOutcome::Accepted
        }
    }

    /// A session with one queue and no idea of a follow-up, like an adapter
    /// written before the mode existed.
    struct SteerOnly {
        queue: Mutex<Vec<SteeringMessage>>,
    }

    impl SteerOnly {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                queue: Mutex::new(Vec::new()),
            })
        }

        fn texts(&self) -> Vec<String> {
            self.queue
                .lock()
                .expect("queue lock")
                .iter()
                .map(|message| message.text().to_owned())
                .collect()
        }
    }

    impl SteerableSession for SteerOnly {
        fn steer(&self, message: SteeringMessage) -> SteeringOutcome {
            self.queue.lock().expect("queue lock").push(message);
            SteeringOutcome::Accepted
        }

        fn interrupt(&self) -> bool {
            false
        }

        fn steer_now(&self, message: SteeringMessage) -> SteeringOutcome {
            self.steer(message)
        }

        fn has_pending_steering(&self) -> bool {
            !self.queue.lock().expect("queue lock").is_empty()
        }
    }

    fn steer(text: &str) -> SteeringMessage {
        SteeringMessage::new(text)
    }

    fn bus() -> SteeringBus<&'static str> {
        SteeringBus::new()
    }

    #[test]
    fn a_steer_with_nothing_attached_is_buffered_and_drains_into_the_next_attachment() {
        let bus = bus();
        let first = bus.steer(steer("queued1"));
        assert!(first.buffered);
        assert!(first.delivered.is_empty());
        bus.steer(steer("queued2"));
        assert_eq!(bus.pending_len(), 2);

        let session = FakeSession::new();
        bus.attach("agent", "session-a", session.clone())
            .expect("attaches");
        assert_eq!(bus.pending_len(), 2, "attaching alone moves nothing");
        let delivery = bus.drain_pending_into(&"agent");

        assert_eq!(session.texts(), ["queued1", "queued2"]);
        assert_eq!(bus.pending_len(), 0);
        assert_eq!(delivery.delivered.len(), 2);
        assert_eq!(bus.attached_count(), 1);
    }

    #[test]
    fn the_buffer_evicts_its_oldest_at_capacity_and_reports_it() {
        let bus: SteeringBus<&str> = SteeringBus::with_pending_capacity(2);
        bus.steer(steer("a").with_actor(Actor::System));
        bus.steer(steer("b"));
        let delivery = bus.steer(steer("c"));
        assert_eq!(bus.pending_len(), 2);
        assert_eq!(delivery.dropped, [DroppedSteer {
            reason:     DropReason::QueueFull,
            count:      1,
            actor:      Some(Actor::System),
            attachment: None,
        }]);
    }

    #[test]
    fn draining_at_the_end_reports_what_nobody_read() {
        let bus = bus();
        assert_eq!(bus.drain_pending(), None);
        bus.steer(steer("a"));
        bus.steer(steer("b"));
        let dropped = bus.drain_pending().expect("two were dropped");
        assert_eq!(dropped.reason, DropReason::Ended);
        assert_eq!(dropped.count, 2);
        assert_eq!(bus.pending_len(), 0);
    }

    #[test]
    fn a_steer_reaches_every_attached_session() {
        let bus = bus();
        let a = FakeSession::new();
        let b = FakeSession::new();
        bus.attach("a", "session-a", a.clone()).expect("attaches");
        bus.attach("b", "session-b", b.clone()).expect("attaches");

        let delivery = bus.steer(steer("hello"));

        assert_eq!(a.texts(), ["hello"]);
        assert_eq!(b.texts(), ["hello"]);
        assert_eq!(delivery.delivered.len(), 2);
        assert!(!delivery.buffered);
        assert_eq!(bus.pending_len(), 0);
    }

    #[test]
    fn a_different_session_cannot_take_an_occupied_key_and_a_stale_detach_changes_nothing() {
        let bus = bus();
        let first = FakeSession::new();
        bus.attach("a", "session-a", first.clone())
            .expect("attaches");

        let second = FakeSession::new();
        assert_eq!(
            bus.attach("a", "session-b", second.clone()),
            Err(AttachError::Occupied {
                session_id: "session-a".into(),
            })
        );
        assert!(!bus.detach(&"a", "session-b"));
        bus.steer(steer("still-active"));
        assert_eq!(first.texts(), ["still-active"]);
        assert!(second.texts().is_empty());

        assert!(bus.detach(&"a", "session-a"));
        assert!(!bus.detach(&"a", "session-a"), "a second detach is a no-op");
        assert_eq!(bus.attached_count(), 0);
    }

    #[test]
    fn detach_if_idle_respects_the_session_id_and_waiting_steering() {
        let bus = bus();
        let session = FakeSession::new();
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");

        assert!(!bus.detach_if_idle(&"a", "session-b"));
        bus.steer(steer("queued"));
        assert!(!bus.detach_if_idle(&"a", "session-a"));
        assert_eq!(bus.attached_count(), 1);

        session.queue.lock().expect("queue lock").clear();
        assert!(bus.detach_if_idle(&"a", "session-a"));
        assert_eq!(bus.attached_count(), 0);
    }

    #[test]
    fn an_interrupt_stops_sessions_without_queueing_text_and_is_not_buffered() {
        let bus = bus();
        assert!(bus.interrupt().interrupted.is_empty());
        assert_eq!(bus.pending_len(), 0);

        let session = FakeSession::new();
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");
        let interruption = bus.interrupt();
        bus.interrupt();

        assert_eq!(session.interrupts(), 2);
        assert!(session.texts().is_empty());
        assert_eq!(interruption.interrupted, [Attachment {
            key:        "a",
            session_id: "session-a".into(),
        }]);
    }

    #[test]
    fn interrupt_then_steer_stops_and_queues_the_text_in_one_step() {
        let bus = bus();
        let session = FakeSession::new();
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");

        let interruption = bus.interrupt_then_steer(&steer("stop"));

        assert_eq!(session.interrupts(), 1);
        assert_eq!(session.texts(), ["stop"]);
        assert_eq!(interruption.interrupted.len(), 1);
        assert!(interruption.dropped.is_empty());
    }

    #[test]
    fn a_full_session_queue_reports_the_message_it_evicted() {
        let bus = bus();
        let session = FakeSession::with_capacity(2);
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");
        bus.steer(steer("m0").with_actor(Actor::System));
        bus.steer(steer("m1"));

        let delivery = bus.steer(steer("m2"));

        assert_eq!(session.texts(), ["m1", "m2"]);
        assert_eq!(delivery.delivered.len(), 1, "evicting still delivers");
        assert_eq!(delivery.dropped, [DroppedSteer {
            reason:     DropReason::QueueFull,
            count:      1,
            actor:      Some(Actor::System),
            attachment: Some(Attachment {
                key:        "a",
                session_id: "session-a".into(),
            }),
        }]);
    }

    #[test]
    fn a_targeted_send_reaches_only_the_named_session() {
        let bus = bus();
        let a = FakeSession::new();
        let b = FakeSession::new();
        bus.attach("a", "session-a", a.clone()).expect("attaches");
        bus.attach("b", "session-b", b.clone()).expect("attaches");

        assert_eq!(
            bus.send_to(&"a", "session-a", steer("just you")),
            Ok(SteeringOutcome::Accepted)
        );
        assert_eq!(
            bus.send_to(&"a", "session-b", steer("stale")),
            Err(TargetError::NotAttached)
        );
        assert_eq!(a.texts(), ["just you"]);
        assert!(b.texts().is_empty());

        assert_eq!(bus.interrupt_at(&"b", "session-b"), Ok(true));
        assert_eq!(
            bus.interrupt_at(&"b", "session-a"),
            Err(TargetError::NotAttached)
        );
        assert_eq!(a.interrupts(), 0);
        assert_eq!(b.interrupts(), 1);
    }

    #[test]
    fn a_hold_lasts_until_released_or_detached() {
        let bus = bus();
        let session = FakeSession::new();
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");

        assert_eq!(
            bus.hold_open(&"a", "session-b"),
            Err(TargetError::NotAttached)
        );
        bus.hold_open(&"a", "session-a").expect("holds");
        assert_eq!(
            bus.hold_open(&"a", "session-a"),
            Err(TargetError::AlreadyHeld)
        );
        assert!(bus.is_held(&"a", "session-a"));
        assert_eq!(session.held(), 1);

        assert!(bus.release_hold(&"a", "session-a"));
        assert!(!bus.release_hold(&"a", "session-a"));
        assert_eq!(session.held(), 0);

        bus.hold_open(&"a", "session-a").expect("holds again");
        assert!(bus.detach(&"a", "session-a"));
        assert_eq!(session.held(), 0, "detaching releases the hold");
    }

    #[test]
    fn a_session_that_cannot_be_held_says_so() {
        struct Unholdable;
        impl SteerableSession for Unholdable {
            fn steer(&self, _: SteeringMessage) -> SteeringOutcome {
                SteeringOutcome::Accepted
            }

            fn interrupt(&self) -> bool {
                false
            }

            fn steer_now(&self, _: SteeringMessage) -> SteeringOutcome {
                SteeringOutcome::Accepted
            }

            fn has_pending_steering(&self) -> bool {
                false
            }
        }
        let bus = bus();
        bus.attach("a", "session-a", Arc::new(Unholdable))
            .expect("attaches");
        assert_eq!(
            bus.hold_open(&"a", "session-a"),
            Err(TargetError::Unsupported)
        );
    }

    #[test]
    fn a_follow_up_with_nothing_attached_is_delivered_as_a_follow_up_on_attach() {
        let bus = bus();
        let delivery = bus.follow_up(steer("later"));
        assert!(delivery.buffered);
        assert!(delivery.delivered.is_empty());
        assert_eq!(bus.pending_len(), 1);

        let session = FakeSession::new();
        bus.attach("agent", "session-a", session.clone())
            .expect("attaches");
        let delivery = bus.drain_pending_into(&"agent");

        assert_eq!(session.follow_up_texts(), ["later"]);
        assert!(session.texts().is_empty(), "a follow-up is not a steer");
        assert_eq!(delivery.delivered.len(), 1);
        assert_eq!(bus.pending_len(), 0);
    }

    #[test]
    fn a_follow_up_to_an_attached_session_reaches_follow_up_and_a_steer_reaches_steer() {
        let bus = bus();
        let session = FakeSession::new();
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");

        let follow_up = bus.follow_up(steer("after the answer"));
        let steered = bus.steer(steer("right now"));

        assert_eq!(session.follow_up_texts(), ["after the answer"]);
        assert_eq!(session.texts(), ["right now"]);
        assert_eq!(follow_up.delivered.len(), 1);
        assert!(!follow_up.buffered);
        assert_eq!(steered.delivered.len(), 1);

        assert_eq!(
            bus.follow_up_to(&"a", "session-a", steer("just you, later")),
            Ok(SteeringOutcome::Accepted)
        );
        assert_eq!(
            bus.follow_up_to(&"a", "session-b", steer("stale")),
            Err(TargetError::NotAttached)
        );
        assert_eq!(session.follow_up_texts(), [
            "after the answer",
            "just you, later"
        ]);
        assert_eq!(session.texts(), ["right now"]);
    }

    #[test]
    fn a_session_without_a_follow_up_queue_receives_follow_ups_as_steers() {
        let bus = bus();
        bus.follow_up(steer("buffered"));
        let session = SteerOnly::new();
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");

        bus.drain_pending_into(&"a");
        bus.follow_up(steer("live"));
        assert_eq!(
            bus.follow_up_to(&"a", "session-a", steer("targeted")),
            Ok(SteeringOutcome::Accepted)
        );

        assert_eq!(session.texts(), ["buffered", "live", "targeted"]);
    }

    #[test]
    fn mixed_pending_messages_keep_their_order_and_their_modes() {
        let bus = bus();
        bus.steer(steer("s1"));
        bus.follow_up(steer("f1"));
        bus.steer(steer("s2"));
        bus.follow_up(steer("f2"));
        assert_eq!(bus.pending_len(), 4);

        // The buffer's capacity counts both modes, and the oldest goes first.
        let small: SteeringBus<&str> = SteeringBus::with_pending_capacity(2);
        small.follow_up(steer("old").with_actor(Actor::System));
        small.steer(steer("kept"));
        let delivery = small.follow_up(steer("newest"));
        assert_eq!(delivery.dropped, [DroppedSteer {
            reason:     DropReason::QueueFull,
            count:      1,
            actor:      Some(Actor::System),
            attachment: None,
        }]);

        let session = FakeSession::new();
        bus.attach("a", "session-a", session.clone())
            .expect("attaches");
        let delivery = bus.drain_pending_into(&"a");

        assert_eq!(delivery.delivered.len(), 4);
        assert_eq!(session.texts(), ["s1", "s2"]);
        assert_eq!(session.follow_up_texts(), ["f1", "f2"]);
        assert_eq!(bus.pending_len(), 0);

        let drained = small.drain_pending().expect("two were dropped");
        assert_eq!(drained.count, 2, "a follow-up and a steer, both unread");
    }
}
