//! The event pipeline as an application sees it.
//!
//! These tests live outside the crate on purpose. They exercise the seam an
//! embedder actually touches: build [`EventOptions`], implement [`EventSink`],
//! drive the pump, and subscribe. A change that keeps the crate's own tests
//! passing but makes the pipeline unusable from outside fails here.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pebble::{
    AgentEvent, ErrorKind, EventCapacity, EventOptions, EventPump, EventSink, EventSinkError,
    SessionEvent,
};

/// A sink an application could plausibly write: it appends to a log and can be
/// told to start failing.
#[derive(Default)]
struct RunLog {
    recorded:    Mutex<Vec<(u64, String)>>,
    refuse_from: Option<usize>,
}

impl RunLog {
    fn refusing_from(count: usize) -> Self {
        Self {
            recorded:    Mutex::new(Vec::new()),
            refuse_from: Some(count),
        }
    }

    fn recorded(&self) -> Vec<(u64, String)> {
        self.recorded.lock().expect("the log lock is held").clone()
    }
}

#[async_trait]
impl EventSink for RunLog {
    async fn record(&self, event: &SessionEvent) -> Result<(), EventSinkError> {
        let mut recorded = self.recorded.lock().expect("the log lock is held");
        if self.refuse_from == Some(recorded.len()) {
            return Err(EventSinkError::new("the run log is not writable"));
        }
        recorded.push((event.seq, event.session_id.clone()));
        Ok(())
    }
}

fn user_input(text: &str) -> AgentEvent {
    AgentEvent::UserInput { text: text.into() }
}

#[tokio::test]
async fn an_application_records_and_watches_the_same_ordered_stream() {
    let log = Arc::new(RunLog::default());
    let (emitter, pump) = EventPump::new(EventOptions {
        capacity: EventCapacity::new(8),
        sink: Some(Arc::clone(&log) as Arc<dyn EventSink>),
        ..EventOptions::default()
    });
    let mut watcher = emitter.subscribe();
    let pump = tokio::spawn(pump.run());

    emitter.emit("ses_root", user_input("one"));
    emitter.emit("ses_root", user_input("two"));
    emitter.emit("ses_root", AgentEvent::SessionEnded);
    drop(emitter);

    pump.await
        .expect("the pump task finishes")
        .expect("the sink accepted every event");

    let mut watched = Vec::new();
    while let Ok(event) = watcher.try_recv() {
        watched.push((event.seq, event.session_id));
    }
    assert_eq!(watched, vec![
        (1, "ses_root".to_owned()),
        (2, "ses_root".to_owned()),
        (3, "ses_root".to_owned()),
    ]);
    assert_eq!(
        log.recorded(),
        watched,
        "the sink and the live stream must agree on content and order"
    );
}

#[tokio::test]
async fn a_refusing_sink_stops_the_run_with_a_typed_error() {
    let log = Arc::new(RunLog::refusing_from(1));
    let (emitter, pump) = EventPump::new(EventOptions {
        sink: Some(Arc::clone(&log) as Arc<dyn EventSink>),
        ..EventOptions::default()
    });
    let pump = tokio::spawn(pump.run());

    emitter.emit("ses_root", user_input("one"));
    emitter.emit("ses_root", user_input("two"));

    let error = pump
        .await
        .expect("the pump task finishes")
        .expect_err("the sink refused the second event");

    assert_eq!(error.kind(), ErrorKind::EventSink);
    assert!(error.to_string().contains("the run log is not writable"));
    assert_eq!(log.recorded().len(), 1);
}

#[tokio::test]
async fn a_resumed_session_keeps_numbering_where_its_record_left_off() {
    let (emitter, pump) = EventPump::new(EventOptions {
        resume_after_seq: 41,
        ..EventOptions::default()
    });
    let mut watcher = emitter.subscribe();
    let pump = tokio::spawn(pump.run());

    emitter.emit("ses_root", user_input("carry on"));

    let event = watcher.recv().await.expect("the event is published");
    assert_eq!(event.seq, 42);
    assert_eq!(emitter.last_seq(), 42);

    drop(emitter);
    pump.await
        .expect("the pump task finishes")
        .expect("no sink refused anything");
}

#[tokio::test]
async fn a_forwarded_child_event_keeps_its_identity_in_the_parent_stream() {
    let (parent, parent_pump) = EventPump::new(EventOptions::default());
    let (child, child_pump) = EventPump::new(EventOptions::default());
    let mut watcher = parent.subscribe();
    let mut child_events = child.subscribe();
    let parent_pump = tokio::spawn(parent_pump.run());
    let child_pump = tokio::spawn(child_pump.run());

    parent.emit("ses_root", user_input("delegate this"));
    child.emit("ses_child", AgentEvent::SessionStarted {
        provider: Some("anthropic".into()),
        model:    Some("claude-sonnet-5".into()),
    });

    let mut from_child = child_events.recv().await.expect("the child publishes");
    from_child.parent_session_id = Some("ses_root".into());
    parent.forward(from_child);

    let first = watcher.recv().await.expect("the parent publishes");
    let forwarded = watcher.recv().await.expect("the parent forwards");

    assert_eq!((first.seq, first.session_id.as_str()), (1, "ses_root"));
    assert_eq!(
        forwarded.seq, 2,
        "the forwarded event joins the parent stream"
    );
    assert_eq!(forwarded.session_id, "ses_child");
    assert_eq!(forwarded.parent_session_id.as_deref(), Some("ses_root"));

    drop(parent);
    drop(child);
    parent_pump
        .await
        .expect("the pump finishes")
        .expect("no sink");
    child_pump
        .await
        .expect("the pump finishes")
        .expect("no sink");
}
