//! The same lifecycle guarantees at model, tool, and compaction boundaries.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent, EventSink, EventSinkError};
use pebble_coding_agent::state::Message;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    text_delta_events, text_response, tool_call_response,
};
use pebble_coding_agent::tools::{RegisteredTool, ToolError};
use pebble_coding_agent::{CodingAgent, CompactionOptions, Error, ShutdownReason};
use serde_json::json;
use tokio::sync::{Notify, mpsc};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const PATIENCE: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    Model,
    Tool,
    Compaction,
}

impl Boundary {
    fn matches(self, event: &CodingEvent) -> bool {
        matches!(
            (self, event),
            (Self::Model, CodingEvent::TextDelta { .. })
                | (Self::Tool, CodingEvent::ToolCallOutputDelta { .. })
                | (Self::Compaction, CodingEvent::CompactionStarted { .. })
        )
    }

    async fn run(self, agent: &mut CodingAgent, cancel: &CancellationToken) -> Result<(), Error> {
        match self {
            Self::Model | Self::Tool => agent
                .prompt_with_cancellation("work", cancel)
                .await
                .result
                .map(|_| ()),
            Self::Compaction => agent
                .compact_with_cancellation(CompactionOptions::new().preserve_turns(1), cancel)
                .await
                .map(|_| ()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure {
    Cancel,
    Drop,
    Sink,
}

struct BoundarySink {
    boundary: Boundary,
    failure:  Failure,
    armed:    AtomicBool,
    entered:  Notify,
    release:  Notify,
    recorded: Mutex<Vec<CodingAgentEvent>>,
}

#[async_trait]
impl EventSink for BoundarySink {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        if self.boundary.matches(&event.event) && self.armed.swap(false, Ordering::SeqCst) {
            self.entered.notify_one();
            self.release.notified().await;
            if self.failure == Failure::Sink {
                return Err(EventSinkError::new("injected boundary failure"));
            }
        }
        self.recorded.lock().expect("sink lock").push(event.clone());
        Ok(())
    }
}

async fn check_case(boundary: Boundary, failure: Failure) {
    let mut script = match boundary {
        Boundary::Model => vec![ScriptedCall::EventsThenPending(text_delta_events(
            "partial",
        ))],
        Boundary::Tool => vec![ScriptedCall::response(tool_call_response(
            "wait",
            "call",
            json!({}),
        ))],
        Boundary::Compaction => vec![
            ScriptedCall::response(text_response("first")),
            ScriptedCall::response(text_response("second")),
        ],
    };
    script.push(ScriptedCall::response(text_response("recovered")));
    let (client, provider) =
        client_from(ScriptedProvider::new(script).completing(vec![ScriptedCompletion::Pending]));
    let sink = Arc::new(BoundarySink {
        boundary,
        failure,
        armed: AtomicBool::new(false),
        entered: Notify::new(),
        release: Notify::new(),
        recorded: Mutex::new(Vec::new()),
    });
    let (started, mut tokens) = mpsc::unbounded_channel();
    let tool = RegisteredTool::function(
        "wait",
        "Wait",
        json!({"type":"object"}),
        move |context, _| {
            let started = started.clone();
            async move {
                started
                    .send(context.cancel().clone())
                    .expect("receiver lives");
                context.emit_coding_event(CodingEvent::ToolCallOutputDelta {
                    delta: "started".to_owned(),
                });
                context.cancel().cancelled().await;
                Err(ToolError::cancelled("stopped"))
            }
        },
    );
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .tools([tool])
        .event_sink(sink.clone())
        .build()
        .await
        .expect("builds");
    if boundary == Boundary::Compaction {
        agent.prompt("first").await.result.expect("seed history");
        agent.prompt("second").await.result.expect("seed history");
    }
    let original_history = agent.history();
    let control = agent.control_handle();
    let mut events = agent.subscribe();
    let cancel = CancellationToken::new();
    sink.armed.store(true, Ordering::SeqCst);
    let mut operation = Box::pin(boundary.run(&mut agent, &cancel));
    tokio::select! {
        result = &mut operation => panic!("completed before {boundary:?}: {result:?}"),
        result = timeout(PATIENCE, sink.entered.notified()) => { result.expect("reaches boundary"); }
    }
    let tool_token = if boundary == Boundary::Tool {
        Some(tokens.try_recv().expect("tool is running"))
    } else {
        None
    };
    if boundary == Boundary::Compaction {
        assert_eq!(
            provider.completion_requests().len(),
            1,
            "compaction model is running"
        );
    } else {
        assert_eq!(provider.call_count(), 1, "model request has started");
    }
    if failure == Failure::Cancel {
        cancel.cancel();
    }
    sink.release.notify_one();
    if failure == Failure::Drop {
        drop(operation);
    } else {
        let error = timeout(PATIENCE, operation)
            .await
            .expect("operation stops")
            .expect_err("injected failure");
        match failure {
            Failure::Cancel => assert!(matches!(error, Error::Interrupted(_)), "{error:?}"),
            Failure::Sink => {
                assert!(
                    matches!(&error, Error::EventSink(source) if source.message() == "injected boundary failure"),
                    "{error:?}"
                );
            }
            Failure::Drop => unreachable!(),
        }
    }
    timeout(PATIENCE, control.wait_for_idle())
        .await
        .expect("idle waiters released");
    assert!(!control.is_running());
    assert_eq!(control.is_closed(), failure != Failure::Cancel);
    if let Some(token) = tool_token {
        assert!(token.is_cancelled(), "active tool cancellation propagates");
    }
    if boundary == Boundary::Compaction {
        assert_eq!(
            agent.history(),
            original_history,
            "failed compaction preserves history"
        );
    }
    if failure == Failure::Cancel {
        if boundary == Boundary::Tool {
            let history = agent.history();
            let results: Vec<_> = history
                .turns()
                .iter()
                .filter_map(|turn| match turn {
                    Message::ToolResults { results, .. } => Some(results),
                    _ => None,
                })
                .flatten()
                .collect();
            assert_eq!(
                results.len(),
                1,
                "the interrupted call has exactly one result"
            );
            assert_eq!(results[0].tool_call_id, "call");
            assert!(results[0].is_error);
        }
        timeout(PATIENCE, agent.prompt("recover"))
            .await
            .expect("recovery completes")
            .result
            .expect("agent reusable");
    } else {
        assert!(matches!(
            agent.prompt("again").await.result,
            Err(Error::SessionClosed)
        ));
    }
    let shutdown = timeout(PATIENCE, agent.shutdown(ShutdownReason::Cancelled))
        .await
        .expect("shutdown joins tasks");
    if failure != Failure::Sink {
        shutdown.expect("shutdown succeeds");
    }
    timeout(PATIENCE, async { while events.recv().await.is_ok() {} })
        .await
        .expect("subscribers close");
    let recorded = sink.recorded.lock().expect("sink lock");
    assert!(
        recorded.windows(2).all(|pair| pair[0].seq < pair[1].seq),
        "committed events stay ordered"
    );
    if failure != Failure::Sink {
        assert_eq!(
            recorded
                .iter()
                .filter(|event| matches!(event.event, CodingEvent::SessionEnded))
                .count(),
            1
        );
        if boundary == Boundary::Compaction {
            assert_eq!(
                recorded
                    .iter()
                    .filter(|event| matches!(event.event, CodingEvent::CompactionCancelled { .. }))
                    .count(),
                1
            );
        }
    }
}

#[tokio::test]
async fn cancellation_at_model_output() {
    check_case(Boundary::Model, Failure::Cancel).await;
}
#[tokio::test]
async fn cancellation_during_a_tool() {
    check_case(Boundary::Tool, Failure::Cancel).await;
}
#[tokio::test]
async fn cancellation_during_compaction() {
    check_case(Boundary::Compaction, Failure::Cancel).await;
}
#[tokio::test]
async fn dropped_future_at_model_output() {
    check_case(Boundary::Model, Failure::Drop).await;
}
#[tokio::test]
async fn dropped_future_during_a_tool() {
    check_case(Boundary::Tool, Failure::Drop).await;
}
#[tokio::test]
async fn dropped_future_during_compaction() {
    check_case(Boundary::Compaction, Failure::Drop).await;
}
#[tokio::test]
async fn sink_failure_at_model_output() {
    check_case(Boundary::Model, Failure::Sink).await;
}
#[tokio::test]
async fn sink_failure_during_a_tool() {
    check_case(Boundary::Tool, Failure::Sink).await;
}
#[tokio::test]
async fn sink_failure_during_compaction() {
    check_case(Boundary::Compaction, Failure::Sink).await;
}
