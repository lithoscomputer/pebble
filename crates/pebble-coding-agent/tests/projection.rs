//! The projection agrees with the agent that emitted the events.
//!
//! An application folds the event stream into a view; the agent reports the
//! same prompt through `PromptReport`. The two must agree on what a prompt
//! spent and touched, whether the fold runs live or over the events after a
//! stored value.

use std::mem;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pebble_coding_agent::events::{CodingAgentEvent, EventSink, EventSinkError, PermissionLevel};
use pebble_coding_agent::projection::{SessionActivity, SessionProjection};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedProvider, client_from, multi_tool_call_response,
    text_response,
};
use pebble_coding_agent::{CodingAgent, CodingAgentOptions, ShutdownReason};
use serde_json::{Value, json};

fn write(path: &str) -> (&'static str, &str, Value) {
    (
        "write_file",
        path,
        json!({"file_path": path, "content": "text"}),
    )
}

/// A durable sink that keeps every event, from `SessionStarted` on.
#[derive(Default)]
struct RecordingSink(Mutex<Vec<CodingAgentEvent>>);

impl RecordingSink {
    fn drain(&self) -> Vec<CodingAgentEvent> {
        mem::take(&mut *self.0.lock().expect("sink lock"))
    }
}

#[async_trait]
impl EventSink for RecordingSink {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        self.0.lock().expect("sink lock").push(event.clone());
        Ok(())
    }
}

#[tokio::test]
async fn the_projection_agrees_with_the_prompt_report_live_and_resumed() {
    let (client, _provider) = client_from(ScriptedProvider::new(vec![
        ScriptedCall::response(multi_tool_call_response(vec![
            write("/home/test/zeta.txt"),
            write("/home/test/alpha.txt"),
        ])),
        ScriptedCall::response(text_response("first done")),
        ScriptedCall::response(text_response("second done")),
    ]));
    let sink = Arc::new(RecordingSink::default());
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .permission_level(PermissionLevel::Full)
        .options(CodingAgentOptions::default().with_loop_detection(false))
        .event_sink(sink.clone())
        .build()
        .await
        .expect("the coding agent builds");

    let first = agent.prompt("write two files").await;
    assert!(first.result.is_ok(), "{first:?}");
    let first_events = sink.drain();
    let mut live = SessionProjection::new();
    live.apply_all(&first_events);

    assert_eq!(live.route.model.as_deref(), Some("model"));
    assert_eq!(live.activity, SessionActivity::Idle);
    assert!(live.prompt.completed);
    assert_eq!(live.prompt.files_touched, first.files_touched);
    assert_eq!(live.prompt.last_file_touched, first.last_file_touched);
    assert_eq!(
        live.prompt.usage, first.usage,
        "the fold spends what the report spends"
    );
    assert_eq!(live.usage, first.usage);
    assert_eq!(live.prompt.tool_calls, 2);
    assert_eq!(live.tools["write_file"].calls, 2);
    assert_eq!(live.tools["write_file"].open, 0);
    assert_eq!(live.prompts, 1);

    // A view stores the value between prompts and resumes from it.
    let stored = serde_json::to_vec(&live).expect("serializes");

    let second = agent.prompt("now just talk").await;
    assert!(second.result.is_ok(), "{second:?}");
    let second_events = sink.drain();
    let mut resumed: SessionProjection = serde_json::from_slice(&stored).expect("parses");
    resumed.apply_all(&second_events);
    live.apply_all(&second_events);

    assert_eq!(resumed, live, "resuming from a stored value loses nothing");
    assert_eq!(resumed.prompts, 2);
    assert!(resumed.prompt.files_touched.is_empty());
    assert_eq!(resumed.prompt.usage, second.usage);
    assert_eq!(
        resumed.usage,
        first.usage.saturating_add(second.usage),
        "the lifetime total spans prompts"
    );
    assert_eq!(
        resumed.files_touched,
        ["/home/test/alpha.txt", "/home/test/zeta.txt"],
        "the lifetime list keeps the first prompt's files"
    );

    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
    resumed.apply_all(&sink.drain());
    assert_eq!(resumed.activity, SessionActivity::Ended);
}
