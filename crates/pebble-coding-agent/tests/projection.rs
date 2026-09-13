//! The projection agrees with the agent that emitted the events.
//!
//! An application folds the event stream into a view; the agent reports the
//! same prompt through `PromptReport`. The two must agree on what a prompt
//! spent and touched, whether the fold runs live or over the events after a
//! stored value.

use std::mem;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lithos_llm::types::TokenCounts;
use pebble_coding_agent::events::{
    CodingAgentEvent, EventSink, EventSinkError, PermissionLevel, TokenUsage,
};
use pebble_coding_agent::projection::{SessionActivity, SessionProjection};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    multi_tool_call_response, text_response, with_cost, with_usage,
};
use pebble_coding_agent::{CodingAgent, CodingAgentOptions, PromptReport, ShutdownReason};
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

/// An input long enough to fill the 100-token window of `test/small`.
fn large_input() -> String {
    "x".repeat(400)
}

/// The summary call's answer, with the usage and cost the provider reports
/// for it.
fn priced_summary(usage: TokenCounts, usd_micros: u64) -> ScriptedCompletion {
    ScriptedCompletion::response(with_cost(
        with_usage(
            text_response("Here is the summary of the conversation so far."),
            usage,
        ),
        usd_micros,
    ))
}

/// The property every embedder's accounting rests on: the report bills the
/// root session's prompt and the fold's `PromptDelta` bills the same, so the
/// tree's spend is the report's plus `descendant_usage()`, which the report
/// keeps out and the fold keeps beside.
fn assert_agrees_with_the_report(projection: &SessionProjection, report: &PromptReport) {
    assert_eq!(
        projection.prompt.usage, report.usage,
        "the delta spends what the report spends"
    );
    assert_eq!(projection.prompt.cost_usd_micros, report.cost_usd_micros);
}

#[tokio::test]
async fn the_projection_bills_a_compaction_as_the_report_does() {
    let summary_usage = TokenCounts {
        input: 70,
        output: 12,
        ..TokenCounts::default()
    };
    // The one response reports tokens of its own, enough to put the window
    // over the threshold, so the totals are the response plus the summary
    // call, and a fold that missed either would show it.
    let (client, _provider) = client_from(
        ScriptedProvider::new(vec![ScriptedCall::response(with_cost(
            with_usage(text_response("OK"), TokenCounts {
                input: 90,
                output: 5,
                ..TokenCounts::default()
            }),
            3,
        ))])
        .completing(vec![priced_summary(summary_usage, 5)]),
    );
    let sink = Arc::new(RecordingSink::default());
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/small")
        .options(CodingAgentOptions::default().with_compaction_preserve_turns(1))
        .event_sink(sink.clone())
        .build()
        .await
        .expect("the coding agent builds");

    let report = agent.prompt(&large_input()).await;

    assert!(report.result.is_ok(), "{report:?}");
    let mut live = SessionProjection::new();
    live.apply_all(&sink.drain());
    let [account] = report.compactions.as_slice() else {
        panic!("one compaction is on the report: {report:?}");
    };
    let [compaction] = live.prompt.compactions.as_slice() else {
        panic!("one compaction is in the fold: {live:?}");
    };
    assert_eq!(compaction.usage, account.usage);
    assert_eq!(compaction.usage, TokenUsage::from(summary_usage));
    assert_eq!(compaction.cost_usd_micros, Some(5));
    assert_eq!(report.usage.input, 160, "the response and the summary call");
    assert_eq!(report.cost_usd_micros, Some(8));
    assert_agrees_with_the_report(&live, &report);
    assert_eq!(live.usage, report.usage);
    assert_eq!(live.compactions, live.prompt.compactions);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}
