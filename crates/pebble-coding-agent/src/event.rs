//! The event pipeline: how a session publishes what it does.
//!
//! One session owns one pipeline. Every producer — the turn loop, a running
//! tool, the retry observer, a subagent forwarding its child stream — holds a
//! cheap [`Emitter`] clone and calls [`Emitter::emit`] from ordinary
//! synchronous code, which queues the event and returns. The session drives
//! one [`EventPump`], and the pump alone publishes: it stamps each event with
//! the next per-session sequence number, hands it to the configured
//! [`EventSink`] and waits for that to succeed, and only then broadcasts it to
//! live subscribers.
//!
//! That ordering is the point. The sink and every subscriber observe the same
//! events in the same order, and a sink failure stops the prompt rather than
//! losing an event.
//!
//! Live delivery is lossy by design. [`Emitter::subscribe`] hands out a
//! bounded broadcast receiver holding [`DEFAULT_EVENT_CAPACITY`] events, and a
//! subscriber that falls behind observes `RecvError::Lagged` instead of
//! stalling the session. An application that must see every event configures
//! an [`EventSink`].
//!
//! The stream ends with the pipeline rather than with the last handle. The
//! pump owns the only broadcast sender; an [`Emitter`] holds a weak one, which
//! publishes and subscribes but keeps nothing open. So once the pump has
//! stopped and been joined, every live receiver reads out whatever it still
//! holds and then observes `RecvError::Closed`, however many emitter clones
//! are still alive. That is what lets a reader loop until `Closed` and end,
//! and it is the guarantee [`crate::CodingAgent::shutdown`] rests
//! on.

use std::error::Error as StdError;
use std::fmt;
use std::result::Result as StdResult;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::SystemTime;

use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc};

use crate::error::Result;
use crate::types::{CodingAgentEvent, CodingEvent};

/// The broadcast capacity a pipeline uses when the caller names none.
pub const DEFAULT_EVENT_CAPACITY: usize = 1024;

/// A durable recorder of one session's event stream.
///
/// Implementors receive every published event exactly once, in sequence order,
/// and the pump waits for each call to return before the event reaches live
/// subscribers. A slow sink therefore slows the whole session, and a failing
/// sink stops it: the pump returns [`crate::Error::EventSink`] and publishes
/// nothing further, and the session it belongs to closes rather than run on
/// with nothing recording it.
///
/// Implementations must be cheap enough to run on the session's critical path
/// and must not call back into the session that owns them.
#[async_trait]
pub trait EventSink: Send + Sync {
    /// Records one event durably.
    ///
    /// Returning an error stops the prompt, so report only failures that make
    /// the recorded stream untrustworthy.
    async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError>;
}

/// A failure reported by an [`EventSink`].
///
/// The message is rendered into [`crate::events::ErrorData`] on the way to
/// consumers, so keep it free of credentials and provider payloads. An optional
/// source is retained for logging and is not projected.
#[derive(Debug)]
pub struct EventSinkError {
    message: String,
    source:  Option<Box<dyn StdError + Send + Sync>>,
}

impl EventSinkError {
    /// Reports a sink failure with a message safe to show a consumer.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source:  None,
        }
    }

    /// Attaches the underlying failure, which is kept for logging only.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<Box<dyn StdError + Send + Sync>>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// The failure rendered for a human.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for EventSinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for EventSinkError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_ref()
            .map(|source| &**source as &(dyn StdError + 'static))
    }
}

/// The monotonic sequence numbers stamped on one session's events.
///
/// Numbering starts at 1, so a `seq` of 0 means an event was built but never
/// published. A resumed session continues where the stored record left off, so
/// sequence numbers stay unique for the life of a session across restarts.
///
/// Crate-private: no public signature takes or returns one. The numbers
/// themselves reach an application through
/// [`CodingAgentEvent::seq`](crate::events::CodingAgentEvent::seq),
/// [`Emitter::last_seq`], and [`EventOptions::resume_after_seq`].
#[derive(Debug)]
pub(crate) struct EventSequence {
    stamped:  AtomicU64,
    reserved: AtomicU64,
}

impl EventSequence {
    /// Starts a fresh session's numbering, so the first event is 1.
    pub(crate) const fn new() -> Self {
        Self {
            stamped:  AtomicU64::new(0),
            reserved: AtomicU64::new(0),
        }
    }

    /// Continues numbering after the last event a previous prompt published.
    pub(crate) const fn resuming_after(last_seq: u64) -> Self {
        Self {
            stamped:  AtomicU64::new(last_seq),
            reserved: AtomicU64::new(last_seq),
        }
    }

    /// Claims the number one newly queued event will be published with.
    ///
    /// An emitter claims the number as it queues; the pump stamps it as it
    /// publishes. The two counters run in step, one event apart at a time, and
    /// the claim is what a stored record has to cover: an event waiting in the
    /// queue is already numbered in every sense that matters to a session
    /// resumed from that record.
    pub(crate) fn reserve(&self) {
        self.reserved.fetch_add(1, Ordering::Relaxed);
    }

    /// Takes the next sequence number for an event being published.
    pub(crate) fn assign(&self) -> u64 {
        self.stamped
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1)
    }

    /// The highest number any event queued so far can be published with.
    ///
    /// Read from outside the publishing task this is a snapshot: another
    /// producer may claim the next number before the caller acts on it.
    pub(crate) fn last_reserved(&self) -> u64 {
        self.reserved.load(Ordering::Relaxed)
    }
}

impl Default for EventSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// How one session's event pipeline is wired.
///
/// A plain configuration record: name the fields that differ and take the rest
/// from [`Default`].
///
/// ```ignore
/// # use pebble_coding_agent::events::{EventCapacity, EventOptions};
/// let options = EventOptions {
///     capacity: EventCapacity::new(64),
///     ..EventOptions::default()
/// };
/// ```
#[derive(Clone, Default)]
pub(crate) struct EventOptions {
    /// How many events the broadcast channel buffers for live subscribers.
    ///
    /// A subscriber that falls further behind than this observes
    /// `RecvError::Lagged`.
    pub(crate) capacity: EventCapacity,

    /// The durable recorder, when the application configured one.
    pub(crate) sink: Option<Arc<dyn EventSink>>,

    /// The last sequence number a previous prompt of this session published.
    ///
    /// Zero for a new session;
    /// [`crate::state::SessionRecord::last_event_seq`] for a
    /// resumed one.
    pub(crate) resume_after_seq: u64,
}

impl fmt::Debug for EventOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EventOptions")
            .field("capacity", &self.capacity)
            .field("sink", &self.sink.as_ref().map(|_| "<sink>"))
            .field("resume_after_seq", &self.resume_after_seq)
            .finish()
    }
}

/// How many events a pipeline buffers for live subscribers.
///
/// Defaults to [`DEFAULT_EVENT_CAPACITY`]; a capacity of zero is raised to one
/// because a tokio broadcast channel rejects an empty buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventCapacity(usize);

impl EventCapacity {
    /// Buffers `events` for each live subscriber.
    #[must_use]
    pub const fn new(events: usize) -> Self {
        Self(if events == 0 { 1 } else { events })
    }

    /// The buffered event count.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl Default for EventCapacity {
    fn default() -> Self {
        Self(DEFAULT_EVENT_CAPACITY)
    }
}

impl From<usize> for EventCapacity {
    fn from(events: usize) -> Self {
        Self::new(events)
    }
}

/// The publishing handle every event producer holds.
///
/// Cloning is cheap and every clone feeds the same pipeline. Emitting is
/// synchronous and never blocks: the event is queued and the owning
/// [`EventPump`] publishes it.
///
/// An emitter holds no half of the pipeline open. Once the pump has stopped,
/// emitting from a clone that outlived it is inert and the live stream is
/// closed — a tool task unwinding long after a session shut down changes
/// nothing either way.
#[derive(Clone, Debug)]
pub(crate) struct Emitter {
    outbox:    mpsc::UnboundedSender<Queued>,
    /// A weak handle to the broadcast side, upgraded only for the moment
    /// [`Emitter::subscribe`] takes to hand out a receiver.
    ///
    /// Weak on purpose: the pump holds the one sender, so the live stream ends
    /// when the pump is joined rather than when the last emitter is dropped.
    published: broadcast::WeakSender<CodingAgentEvent>,
    sequence:  Arc<EventSequence>,
}

/// One item on the queue between the emitters and the pump.
///
/// `None` is the stop signal a session sends at shutdown, so the pump can be
/// joined without waiting for every [`Emitter`] clone to be dropped. Anything
/// queued before it is still published.
type Queued = Option<CodingAgentEvent>;

impl Emitter {
    /// Publishes an event this session produced.
    pub(crate) fn emit(&self, session_id: impl Into<String>, event: CodingEvent) {
        self.emit_with_tool_call_id(session_id, event, None);
    }

    /// Publishes an event produced while a tool call was running.
    ///
    /// The event is traced as it is queued, so the tracing order is the order
    /// producers emitted in rather than the published order.
    pub(crate) fn emit_with_tool_call_id(
        &self,
        session_id: impl Into<String>,
        event: CodingEvent,
        tool_call_id: Option<String>,
    ) {
        let session_id = session_id.into();
        event.trace(&session_id);
        self.queue(CodingAgentEvent {
            seq: 0,
            event,
            timestamp: SystemTime::now(),
            session_id,
            parent_session_id: None,
            tool_call_id,
        });
    }

    /// Republishes an envelope another session built.
    ///
    /// The child's `session_id`, `parent_session_id`, and timestamp pass
    /// through untouched; only the sequence number is reassigned, because a
    /// forwarded event takes its place in the parent's stream.
    pub(crate) fn forward(&self, event: CodingAgentEvent) {
        self.queue(event);
    }

    /// Subscribes to the live event stream.
    ///
    /// The receiver observes only events published after this call, and misses
    /// events if it falls more than the configured capacity behind.
    ///
    /// It ends when the pipeline does: once the pump has stopped and been
    /// joined, the receiver reads out whatever it still holds and then
    /// observes `RecvError::Closed`. Subscribing after that answers with a
    /// receiver that is closed from its first read, which is what a stream
    /// nothing will ever publish to looks like.
    #[must_use]
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<CodingAgentEvent> {
        match self.published.upgrade() {
            Some(published) => published.subscribe(),
            None => ended_stream(),
        }
    }

    /// The highest sequence number this session has claimed for an event.
    ///
    /// Every event emitted so far will be published with this number or a
    /// lower one, including events still queued, so a record storing it names
    /// a point the live stream has passed rather than one it is about to
    /// reach.
    ///
    /// This is a snapshot taken from outside the publishing task, so another
    /// producer may claim the next number before the caller acts on it. It
    /// counts claims, not delivery: an event still in the queue is counted,
    /// and after a sink refusal, or an emit into a stopped pipeline, the
    /// number names an event that was never published.
    #[must_use]
    pub(crate) fn last_seq(&self) -> u64 {
        self.sequence.last_reserved()
    }

    /// Whether the pump has stopped, so nothing further will be published.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn is_closed(&self) -> bool {
        self.outbox.is_closed()
    }

    /// Stops the pump once everything already queued has been published.
    ///
    /// The session sends this at shutdown so it can join the pump task without
    /// having to drop every emitter clone first. Events queued afterwards are
    /// discarded, which is what emitting into a session that has ended means.
    pub(crate) fn close(&self) {
        let _ = self.outbox.send(None);
    }

    /// Queues an event, dropping it when the pump has already stopped.
    ///
    /// A session that has shut down still holds emitter clones — a tool task
    /// unwinding, say — and emitting from one must not panic.
    ///
    /// The number is claimed before the event is queued, so a record taken
    /// from another task never names a number the pump is still about to
    /// stamp.
    fn queue(&self, event: CodingAgentEvent) {
        self.sequence.reserve();
        let _ = self.outbox.send(Some(event));
    }
}

/// A receiver on a stream that has already ended.
///
/// What [`Emitter::subscribe`] answers with once the pump is gone, so a late
/// subscriber gets the ordinary end of a stream on its first read rather than
/// a receiver that never speaks.
fn ended_stream() -> broadcast::Receiver<CodingAgentEvent> {
    let (sender, receiver) = broadcast::channel(1);
    drop(sender);
    receiver
}

/// The publishing half of one session's event pipeline.
///
/// The session owns the pump and decides where it runs, normally by spawning
/// [`EventPump::run`] and joining the handle at shutdown so a sink failure is
/// reported rather than lost.
///
/// The pump holds the only broadcast sender, which is what makes joining it
/// the end of the live stream: when the pump is dropped — as it is the moment
/// [`EventPump::run`] returns — every subscriber drains what it holds and then
/// observes `RecvError::Closed`.
#[must_use = "a pipeline publishes nothing until its pump runs"]
pub(crate) struct EventPump {
    inbox:     mpsc::UnboundedReceiver<Queued>,
    published: broadcast::Sender<CodingAgentEvent>,
    sequence:  Arc<EventSequence>,
    sink:      Option<Arc<dyn EventSink>>,
}

impl fmt::Debug for EventPump {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EventPump")
            .field("inbox", &self.inbox)
            .field("published", &self.published)
            .field("sequence", &self.sequence)
            .field("sink", &self.sink.as_ref().map(|_| "<sink>"))
            .finish()
    }
}

impl EventPump {
    /// Creates one session's pipeline.
    ///
    /// The [`Emitter`] is cloned to every producer; the pump is driven once.
    /// The pump keeps the sender and the emitter takes a weak handle, so the
    /// live stream lasts exactly as long as the pump does.
    pub(crate) fn new(options: EventOptions) -> (Emitter, Self) {
        let EventOptions {
            capacity,
            sink,
            resume_after_seq,
        } = options;
        let (outbox, inbox) = mpsc::unbounded_channel();
        let (published, _) = broadcast::channel(capacity.get());
        let sequence = Arc::new(EventSequence::resuming_after(resume_after_seq));
        let emitter = Emitter {
            outbox,
            published: published.downgrade(),
            sequence: Arc::clone(&sequence),
        };
        let pump = Self {
            inbox,
            published,
            sequence,
            sink,
        };
        (emitter, pump)
    }

    /// Publishes queued events until the session stops the pipeline or every
    /// [`Emitter`] has been dropped.
    ///
    /// Returns [`crate::Error::EventSink`] as soon as the sink refuses an
    /// event. The refused event, and anything still queued, is not published,
    /// because the prompt is over.
    pub(crate) async fn run(mut self) -> Result<()> {
        while let Some(message) = self.inbox.recv().await {
            let Some(event) = message else {
                break;
            };
            self.publish(event).await?;
        }
        Ok(())
    }

    async fn publish(&mut self, mut event: CodingAgentEvent) -> Result<()> {
        event.seq = self.sequence.assign();
        if let Some(sink) = self.sink.as_ref() {
            sink.record(&event).await?;
        }
        // Having no live subscriber is the normal case, not a failure.
        let _ = self.published.send(event);
        Ok(())
    }
}

/// Byte counts for the output one tool call produced.
///
/// A tool observes every byte, retains what the output budget allows, and
/// omits the rest; the three counters land on
/// [`CodingEvent::ToolCallCompleted`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutputCaptureStats {
    /// Bytes the tool produced.
    pub observed_bytes: usize,
    /// Bytes kept for the model and for history.
    pub retained_bytes: usize,
    /// Bytes the budget dropped.
    pub omitted_bytes:  usize,
}

impl OutputCaptureStats {
    /// Counts output that was kept whole.
    #[must_use]
    pub const fn complete(byte_count: usize) -> Self {
        Self {
            observed_bytes: byte_count,
            retained_bytes: byte_count,
            omitted_bytes:  0,
        }
    }

    /// Sums two captures, as when a process reports stdout and stderr apart.
    #[must_use]
    pub const fn combine(self, other: Self) -> Self {
        Self {
            observed_bytes: self.observed_bytes.saturating_add(other.observed_bytes),
            retained_bytes: self.retained_bytes.saturating_add(other.retained_bytes),
            omitted_bytes:  self.omitted_bytes.saturating_add(other.omitted_bytes),
        }
    }
}

/// A session- and tool-bound view of an [`Emitter`], handed to a running tool.
///
/// It stamps the session identity and the active tool call on every event, so
/// a tool never has to know either, and it carries the side channel a tool
/// uses to report how much output it produced.
#[derive(Clone, Debug)]
pub(crate) struct SessionBoundEmitter {
    emitter:      Emitter,
    session_id:   String,
    tool_call_id: Option<String>,
    output_stats: Arc<Mutex<Option<OutputCaptureStats>>>,
}

impl SessionBoundEmitter {
    /// Binds an emitter to one session and, when there is one, one tool call.
    #[must_use]
    pub(crate) fn new(
        emitter: Emitter,
        session_id: impl Into<String>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            emitter,
            session_id: session_id.into(),
            tool_call_id,
            output_stats: Arc::new(Mutex::new(None)),
        }
    }

    /// Publishes an event stamped with the bound identities.
    pub(crate) fn emit(&self, event: CodingEvent) {
        self.emitter.emit_with_tool_call_id(
            self.session_id.clone(),
            event,
            self.tool_call_id.clone(),
        );
    }

    /// Reports how much output the running tool produced.
    ///
    /// The last report wins; the execution layer drains it once the tool
    /// returns.
    pub(crate) fn record_tool_output_stats(&self, stats: OutputCaptureStats) {
        *self
            .output_stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(stats);
    }

    /// Takes the reported output counts, leaving none behind.
    pub(crate) fn take_tool_output_stats(&self) -> Option<OutputCaptureStats> {
        self.output_stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// The session this view is bound to.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The tool call this view is bound to, when it is bound to one.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use tokio::sync::broadcast::error::{RecvError, TryRecvError};
    use tokio::task::{JoinHandle, yield_now};

    use super::*;
    use crate::error::ErrorKind;

    /// Records every event it is handed, and can be told to fail once it has
    /// seen a given number of them.
    #[derive(Debug, Default)]
    struct RecordingSink {
        recorded: Mutex<Vec<CodingAgentEvent>>,
        fail_at:  Option<usize>,
    }

    impl RecordingSink {
        fn failing_at(count: usize) -> Self {
            Self {
                recorded: Mutex::new(Vec::new()),
                fail_at:  Some(count),
            }
        }

        fn recorded(&self) -> Vec<CodingAgentEvent> {
            self.recorded
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    #[async_trait]
    impl EventSink for RecordingSink {
        async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
            let mut recorded = self.recorded.lock().unwrap_or_else(PoisonError::into_inner);
            if self.fail_at == Some(recorded.len()) {
                return Err(EventSinkError::new("disk is full"));
            }
            recorded.push(event.clone());
            Ok(())
        }
    }

    fn pipeline() -> (Emitter, JoinHandle<Result<()>>) {
        spawn_pipeline(EventOptions::default())
    }

    fn spawn_pipeline(options: EventOptions) -> (Emitter, JoinHandle<Result<()>>) {
        let (emitter, pump) = EventPump::new(options);
        (emitter, tokio::spawn(pump.run()))
    }

    fn session_started() -> CodingEvent {
        CodingEvent::SessionStarted {
            provider: Some("anthropic".into()),
            model:    Some("claude-sonnet-5".into()),
        }
    }

    #[tokio::test]
    async fn emit_and_receive_event() {
        let (emitter, _pump) = pipeline();
        let mut receiver = emitter.subscribe();

        emitter.emit("ses_1", session_started());

        let event = receiver.recv().await.unwrap();
        assert!(matches!(event.event, CodingEvent::SessionStarted {
            provider: Some(_),
            model:    Some(_),
        }));
        assert_eq!(event.session_id, "ses_1");
        assert_eq!(event.parent_session_id, None);
        assert_eq!(event.seq, 1);
    }

    #[tokio::test]
    async fn emit_carries_its_payload() {
        let (emitter, _pump) = pipeline();
        let mut receiver = emitter.subscribe();

        emitter.emit("ses_2", CodingEvent::UserInput {
            text: "fix the failing test".into(),
        });

        let event = receiver.recv().await.unwrap();
        assert!(
            matches!(&event.event, CodingEvent::UserInput { text } if text == "fix the failing test")
        );
    }

    #[tokio::test]
    async fn multiple_subscribers_each_see_every_event() {
        let (emitter, _pump) = pipeline();
        let mut first = emitter.subscribe();
        let mut second = emitter.subscribe();

        emitter.emit("ses_3", CodingEvent::SessionEnded);

        let from_first = first.recv().await.unwrap();
        let from_second = second.recv().await.unwrap();
        assert!(matches!(from_first.event, CodingEvent::SessionEnded));
        assert!(matches!(from_second.event, CodingEvent::SessionEnded));
        assert_eq!(from_first.session_id, "ses_3");
        assert_eq!(from_second.session_id, "ses_3");
    }

    #[tokio::test]
    async fn emitting_without_subscribers_is_not_a_failure() {
        let (emitter, pump) = pipeline();

        emitter.emit("ses_4", CodingEvent::LoopDetected);
        drop(emitter);

        pump.await.unwrap().unwrap();
    }

    #[test]
    fn emitting_after_the_pump_stops_is_ignored() {
        let (emitter, pump) = EventPump::new(EventOptions::default());
        drop(pump);

        emitter.emit("ses_5", CodingEvent::LoopDetected);

        assert!(emitter.is_closed());
    }

    #[tokio::test]
    async fn forward_preserves_the_child_envelope() {
        let (emitter, _pump) = pipeline();
        let mut receiver = emitter.subscribe();
        let stamped = SystemTime::UNIX_EPOCH;

        emitter.forward(CodingAgentEvent {
            seq:               17,
            event:             session_started(),
            timestamp:         stamped,
            session_id:        "ses_child".into(),
            parent_session_id: Some("ses_root".into()),
            tool_call_id:      None,
        });

        let event = receiver.recv().await.unwrap();
        assert_eq!(event.session_id, "ses_child");
        assert_eq!(event.parent_session_id.as_deref(), Some("ses_root"));
        assert_eq!(event.timestamp, stamped);
        assert_eq!(
            event.seq, 1,
            "a forwarded event is renumbered into the parent stream"
        );
    }

    #[tokio::test]
    async fn sequence_numbers_are_monotonic_from_one() {
        let (emitter, _pump) = pipeline();
        let mut receiver = emitter.subscribe();

        for _ in 0..3 {
            emitter.emit("ses_1", CodingEvent::LoopDetected);
        }

        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(receiver.recv().await.unwrap().seq);
        }
        assert_eq!(seen, vec![1, 2, 3]);
        assert_eq!(emitter.last_seq(), 3);
    }

    #[tokio::test]
    async fn a_resumed_session_continues_its_numbering() {
        let (emitter, _pump) = spawn_pipeline(EventOptions {
            resume_after_seq: 41,
            ..EventOptions::default()
        });
        let mut receiver = emitter.subscribe();

        emitter.emit("ses_1", CodingEvent::LoopDetected);

        assert_eq!(receiver.recv().await.unwrap().seq, 42);
    }

    #[tokio::test]
    async fn the_sink_sees_every_event_in_order() {
        let sink = Arc::new(RecordingSink::default());
        let (emitter, pump) = spawn_pipeline(EventOptions {
            sink: Some(Arc::clone(&sink) as Arc<dyn EventSink>),
            ..EventOptions::default()
        });

        emitter.emit("ses_1", CodingEvent::UserInput { text: "one".into() });
        emitter.emit("ses_1", CodingEvent::UserInput { text: "two".into() });
        drop(emitter);
        pump.await.unwrap().unwrap();

        let recorded = sink.recorded();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].seq, 1);
        assert_eq!(recorded[1].seq, 2);
    }

    #[tokio::test]
    async fn a_sink_failure_stops_the_prompt_and_withholds_the_event() {
        let sink = Arc::new(RecordingSink::failing_at(1));
        let (emitter, pump) = spawn_pipeline(EventOptions {
            sink: Some(Arc::clone(&sink) as Arc<dyn EventSink>),
            ..EventOptions::default()
        });
        let mut receiver = emitter.subscribe();

        emitter.emit("ses_1", CodingEvent::UserInput { text: "one".into() });
        emitter.emit("ses_1", CodingEvent::UserInput { text: "two".into() });

        let error = pump.await.unwrap().expect_err("the sink refused an event");
        assert_eq!(error.kind(), ErrorKind::EventSink);
        assert!(error.to_string().contains("disk is full"));

        assert_eq!(receiver.recv().await.unwrap().seq, 1);
        assert!(
            matches!(receiver.try_recv(), Err(TryRecvError::Closed)),
            "the refused event never reaches subscribers, and the stream ends \
             with the pump that stopped for it"
        );
    }

    #[tokio::test]
    async fn a_slow_subscriber_lags_instead_of_stalling_the_session() {
        let (emitter, _pump) = spawn_pipeline(EventOptions {
            capacity: EventCapacity::new(2),
            ..EventOptions::default()
        });
        let mut receiver = emitter.subscribe();

        for _ in 0..4 {
            emitter.emit("ses_1", CodingEvent::LoopDetected);
        }
        // Let the pump drain the queue before the subscriber reads.
        yield_now().await;

        let mut lagged = false;
        for _ in 0..4 {
            match receiver.try_recv() {
                Err(TryRecvError::Lagged(_)) => lagged = true,
                Err(TryRecvError::Empty) => break,
                _ => {}
            }
        }
        assert!(lagged, "a capacity of 2 cannot buffer 4 events");
    }

    #[test]
    fn a_zero_capacity_is_raised_to_one() {
        assert_eq!(EventCapacity::new(0).get(), 1);
        assert_eq!(EventCapacity::default().get(), DEFAULT_EVENT_CAPACITY);
    }

    #[tokio::test]
    async fn closing_publishes_what_was_queued_and_then_stops() {
        let sink = Arc::new(RecordingSink::default());
        let (emitter, pump) = spawn_pipeline(EventOptions {
            sink: Some(Arc::clone(&sink) as Arc<dyn EventSink>),
            ..EventOptions::default()
        });

        emitter.emit("ses_1", CodingEvent::UserInput { text: "one".into() });
        emitter.close();
        emitter.emit("ses_1", CodingEvent::UserInput { text: "two".into() });

        pump.await.unwrap().unwrap();

        let recorded = sink.recorded();
        assert_eq!(recorded.len(), 1, "only the event queued before the stop");
        assert!(
            emitter.is_closed(),
            "a stopped pump leaves its emitters closed"
        );
    }

    #[tokio::test]
    async fn a_joined_pump_ends_the_stream_though_its_emitters_live_on() {
        let (emitter, pump) = pipeline();
        let mut receiver = emitter.subscribe();
        let held = emitter.clone();

        emitter.emit("ses_1", CodingEvent::LoopDetected);
        emitter.close();
        pump.await.unwrap().unwrap();

        assert_eq!(
            receiver.recv().await.unwrap().seq,
            1,
            "what was queued before the stop is still read out"
        );
        assert!(
            matches!(receiver.recv().await, Err(RecvError::Closed)),
            "two emitters are still alive and the stream has still ended"
        );
        assert!(
            matches!(held.subscribe().recv().await, Err(RecvError::Closed)),
            "subscribing after the pump has gone answers with an ended stream"
        );
    }

    #[tokio::test]
    async fn the_pump_finishes_once_every_emitter_is_dropped() {
        let (emitter, pump) = pipeline();
        let clone = emitter.clone();

        drop(emitter);
        drop(clone);

        pump.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn a_bound_emitter_stamps_its_session_and_tool_call() {
        let (emitter, _pump) = pipeline();
        let mut receiver = emitter.subscribe();
        let bound = SessionBoundEmitter::new(emitter, "ses_1", Some("call_1".into()));

        bound.emit(CodingEvent::ToolCallOutputDelta {
            delta: "running".into(),
        });

        let event = receiver.recv().await.unwrap();
        assert_eq!(event.session_id, "ses_1");
        assert_eq!(event.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(bound.session_id(), "ses_1");
        assert_eq!(bound.tool_call_id(), Some("call_1"));
    }

    #[test]
    fn output_stats_are_taken_once() {
        let (emitter, _pump) = EventPump::new(EventOptions::default());
        let bound = SessionBoundEmitter::new(emitter, "ses_1", None);

        assert_eq!(bound.take_tool_output_stats(), None);
        bound.record_tool_output_stats(OutputCaptureStats::complete(64));

        assert_eq!(
            bound.take_tool_output_stats(),
            Some(OutputCaptureStats {
                observed_bytes: 64,
                retained_bytes: 64,
                omitted_bytes:  0,
            })
        );
        assert_eq!(bound.take_tool_output_stats(), None);
    }

    #[test]
    fn output_stats_combine_by_summing() {
        let combined = OutputCaptureStats::complete(10).combine(OutputCaptureStats {
            observed_bytes: 30,
            retained_bytes: 5,
            omitted_bytes:  25,
        });

        assert_eq!(combined, OutputCaptureStats {
            observed_bytes: 40,
            retained_bytes: 15,
            omitted_bytes:  25,
        });
    }

    #[test]
    fn a_sink_error_keeps_its_source_for_logging() {
        let error = EventSinkError::new("write failed")
            .with_source(io::Error::other("no space left on device"));

        assert_eq!(error.message(), "write failed");
        assert_eq!(error.to_string(), "write failed");
        assert!(
            error
                .source()
                .is_some_and(|source| source.to_string().contains("no space left"))
        );
    }

    #[test]
    fn sequence_assignment_is_shared_across_handles() {
        let sequence = Arc::new(EventSequence::new());
        let shared = Arc::clone(&sequence);

        let first = sequence.assign();
        let second = shared.assign();

        assert_eq!((first, second), (1, 2));
        assert_eq!(EventSequence::resuming_after(7).assign(), 8);
    }

    #[test]
    fn a_claimed_number_is_the_one_the_pump_stamps() {
        let sequence = EventSequence::resuming_after(7);

        sequence.reserve();
        sequence.reserve();

        assert_eq!(sequence.last_reserved(), 9);
        assert_eq!(sequence.assign(), 8);
        assert_eq!(sequence.assign(), 9);
        assert_eq!(
            sequence.last_reserved(),
            9,
            "the claims and the stamps meet once the queue is drained"
        );
    }

    #[tokio::test]
    async fn last_seq_counts_events_the_pump_has_not_reached() {
        // Nothing drives this pump until the assertions are made, so every
        // event is still queued.
        let (emitter, pump) = EventPump::new(EventOptions::default());
        let mut receiver = emitter.subscribe();

        for _ in 0..3 {
            emitter.emit("ses_1", CodingEvent::LoopDetected);
        }

        assert_eq!(
            emitter.last_seq(),
            3,
            "a queued event has already claimed its number"
        );

        emitter.close();
        pump.run().await.unwrap();
        let published: Vec<u64> = (0..3)
            .map(|_| receiver.try_recv().expect("the event is published").seq)
            .collect();
        assert_eq!(published, vec![1, 2, 3]);
        assert_eq!(emitter.last_seq(), 3);
    }
}
