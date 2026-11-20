//! The model call that summarizes a fetched page: cancelling the fetch ends
//! it, and it is sent in the fetch's trace span.
use std::sync::Arc;
use std::time::Duration;

use futures_util::poll;
use pebble_coding_agent::events::{CodingEvent, ToolErrorKind};
use pebble_coding_agent::state::Message;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    text_response, tool_call_response,
};
use pebble_coding_agent::{CodingAgent, Error, ShutdownReason, TraceContext};
use serde_json::json;
use tokio::task::yield_now;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const PATIENCE: Duration = Duration::from_secs(5);

#[tokio::test]
async fn cancelling_a_summary_pairs_the_tool_result_and_keeps_the_agent_reusable() {
    for through_handle in [false, true] {
        let script = ScriptedProvider::new(vec![
            ScriptedCall::response(tool_call_response(
                "web_fetch",
                "fetch",
                json!({"url":"https://example.test", "prompt":"summarize"}),
            )),
            ScriptedCall::response(text_response("next answer")),
        ])
        .completing(vec![ScriptedCompletion::Pending]);
        let (client, provider) = client_from(script);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .web_fetch_summarizer("test/model")
            .build()
            .await
            .expect("builds");
        let control = agent.control_handle();
        let mut events = agent.subscribe();
        let cancel = CancellationToken::new();
        let mut prompt = Box::pin(agent.prompt_with_cancellation("fetch a page", &cancel));
        timeout(PATIENCE, async {
            loop {
                assert!(poll!(&mut prompt).is_pending());
                if !provider.completion_requests().is_empty() {
                    break;
                }
                yield_now().await;
            }
        })
        .await
        .expect("summary starts");
        if through_handle {
            assert!(control.abort());
        } else {
            cancel.cancel();
        }
        assert!(matches!(
            timeout(PATIENCE, &mut prompt)
                .await
                .expect("summary cancellation finishes")
                .result,
            Err(Error::Interrupted(_))
        ));
        drop(prompt);
        assert!(!control.is_running());
        assert!(!control.is_closed());
        let history = agent.history();
        let results = history
            .turns()
            .iter()
            .find_map(|message| match message {
                Message::ToolResults { results, .. } => Some(results),
                _ => None,
            })
            .expect("tool results committed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool_call_id, "fetch");
        assert!(results[0].is_error);
        let mut cancellation_reported = false;
        while let Ok(event) = events.try_recv() {
            if matches!(event.event, CodingEvent::ToolCallCompleted {
                error_kind: Some(ToolErrorKind::Cancelled),
                ..
            }) {
                cancellation_reported = true;
            }
        }
        assert!(cancellation_reported);
        assert_eq!(
            agent
                .prompt("continue")
                .await
                .result
                .expect("agent reusable")
                .text
                .as_deref(),
            Some("next answer")
        );
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shutdown");
    }
}

/// The summary is made on the `web_fetch` call's behalf, so it is sent in
/// that call's span: the gateway's parent span is one the call's events
/// record, and the baggage names the call.
#[tokio::test]
async fn a_summary_is_sent_in_the_span_its_fetch_records() {
    let trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
    let caller_span = "00f067aa0ba902b7";
    let script = ScriptedProvider::new(vec![
        ScriptedCall::response(tool_call_response(
            "web_fetch",
            "fetch",
            json!({"url":"https://example.test", "prompt":"summarize"}),
        )),
        ScriptedCall::response(text_response("done")),
    ])
    .completing(vec![ScriptedCompletion::response(text_response(
        "a summary",
    ))]);
    let (client, provider) = client_from(script);
    let context = TraceContext::from_traceparent(&format!("00-{trace_id}-{caller_span}-01"))
        .and_then(|context| context.with_baggage("run_id", "run_1"))
        .expect("a valid context");
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .web_fetch_summarizer("test/model")
        .trace_context(context)
        .build()
        .await
        .expect("builds");
    let mut events = agent.subscribe();

    let report = agent.prompt("fetch a page").await;

    assert!(report.result.is_ok(), "{report:?}");
    let session = agent.id().to_owned();
    let mut fetch_spans = Vec::new();
    while let Ok(event) = events.try_recv() {
        if matches!(
            event.event,
            CodingEvent::ToolCallStarted { .. } | CodingEvent::ToolCallCompleted { .. }
        ) {
            assert_eq!(event.tool_call_id.as_deref(), Some("fetch"));
            assert_eq!(event.trace_id.as_deref(), Some(trace_id), "{event:?}");
            assert_eq!(event.parent_span_id.as_deref(), Some(caller_span));
            fetch_spans.push(event.span_id.clone().expect("the event records its span"));
        }
    }
    assert_eq!(fetch_spans.len(), 2);
    assert_eq!(fetch_spans[0], fetch_spans[1]);
    let span = &fetch_spans[0];
    let sent = provider.completion_request_headers();
    assert_eq!(sent.len(), 1, "one summary");
    assert_eq!(
        sent[0].get("traceparent"),
        Some(&format!("00-{trace_id}-{span}-01"))
    );
    assert_eq!(
        sent[0].get("baggage"),
        Some(&format!(
            "run_id=run_1,session_id={session},tool_call_id=fetch"
        ))
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
}
