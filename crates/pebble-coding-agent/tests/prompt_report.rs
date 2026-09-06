//! Prompt accounting survives failures and belongs to one invocation.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lithos_llm::types::ErrorKind;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent, EventSink, EventSinkError};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedFailure, ScriptedProvider, client_from, text_response,
    tool_call_response, with_cost,
};
use pebble_coding_agent::tools::{RegisteredTool, ToolError};
use pebble_coding_agent::{
    CodingAgent, CodingAgentOptions, Error, InterruptReason, PromptTiming, ShutdownReason,
};
use serde_json::json;
use tokio::sync::Notify;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug)]
enum Failure {
    Provider,
    Cancellation,
    Timeout,
    Sink,
}

struct RefusePromptEnd;

#[async_trait]
impl EventSink for RefusePromptEnd {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        if matches!(event.event, CodingEvent::ProcessingEnd) {
            return Err(EventSinkError::new("cannot save prompt end"));
        }
        Ok(())
    }
}

#[tokio::test]
async fn failures_preserve_observed_usage_cost_and_elapsed_work() {
    for failure in [
        Failure::Provider,
        Failure::Cancellation,
        Failure::Timeout,
        Failure::Sink,
    ] {
        let second = match failure {
            Failure::Provider => ScriptedCall::Failure(ScriptedFailure::terminal(
                ErrorKind::Authentication,
                "credentials rejected",
            )),
            Failure::Cancellation | Failure::Timeout => ScriptedCall::PendingOpen,
            Failure::Sink => ScriptedCall::response(text_response("done")),
        };
        let (client, _) = client_from(
            ScriptedProvider::new(vec![
                ScriptedCall::response(with_cost(
                    tool_call_response("work", "call", json!({})),
                    42,
                )),
                second,
            ])
            .delayed(Duration::from_millis(2)),
        );
        let started = Arc::new(Notify::new());
        let tool_started = started.clone();
        let tool = RegisteredTool::function(
            "work",
            "Work",
            json!({"type":"object"}),
            move |context, _| {
                let started = tool_started.clone();
                async move {
                    sleep(Duration::from_millis(2)).await;
                    started.notify_one();
                    if matches!(failure, Failure::Cancellation) {
                        context.cancel().cancelled().await;
                        return Err(ToolError::cancelled("cancelled"));
                    }
                    Ok("worked".to_owned())
                }
            },
        );
        let mut builder = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .tools([tool]);
        if matches!(failure, Failure::Timeout) {
            builder = builder.options(
                CodingAgentOptions::default().with_wall_clock_timeout(Duration::from_millis(100)),
            );
        }
        if matches!(failure, Failure::Sink) {
            builder = builder.event_sink(Arc::new(RefusePromptEnd));
        }
        let mut agent = builder.build().await.expect("builds");
        let cancel = CancellationToken::new();
        let (report, ()) = tokio::join!(agent.prompt_with_cancellation("work", &cancel), async {
            started.notified().await;
            if matches!(failure, Failure::Cancellation) {
                cancel.cancel();
            }
        });
        match (failure, &report.result) {
            (Failure::Provider, Err(Error::Llm(_)))
            | (Failure::Cancellation, Err(Error::Interrupted(InterruptReason::Cancelled)))
            | (Failure::Timeout, Err(Error::Interrupted(InterruptReason::WallClockTimeout)))
            | (Failure::Sink, Err(Error::EventSink(_))) => {}
            _ => panic!("unexpected {failure:?} report: {report:?}"),
        }
        assert!(report.usage.input >= 10, "{report:?}");
        assert!(report.usage.output >= 5, "{report:?}");
        assert_eq!(report.cost_usd_micros, Some(42));
        assert!(
            report.timing.inference >= Duration::from_millis(2),
            "{report:?}"
        );
        assert!(report.timing.tool >= Duration::from_millis(2), "{report:?}");
        // Sink failure closes the session; the report still remains readable.
        let _shutdown = agent.shutdown(ShutdownReason::Error).await;
        let rejected = agent.prompt("closed").await;
        assert!(matches!(rejected.result, Err(Error::SessionClosed)));
        assert_eq!(rejected.usage.total(), 0);
        assert_eq!(rejected.cost_usd_micros, None);
        assert_eq!(rejected.timing, PromptTiming::default());
    }
}

#[tokio::test]
async fn each_report_contains_only_its_own_invocation() {
    let (client, _) = client_from(ScriptedProvider::new(vec![
        ScriptedCall::response(with_cost(text_response("first"), 42)),
        ScriptedCall::response(text_response("second")),
    ]));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .build()
        .await
        .expect("builds");
    let first = agent.prompt("first").await;
    let second = agent.prompt("second").await;
    assert_eq!(
        first.result.expect("first succeeds").text.as_deref(),
        Some("first")
    );
    assert_eq!(
        second.result.expect("second succeeds").text.as_deref(),
        Some("second")
    );
    assert_eq!(first.usage, second.usage);
    assert_eq!(first.cost_usd_micros, Some(42));
    assert_eq!(second.cost_usd_micros, None);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shuts down");
}
