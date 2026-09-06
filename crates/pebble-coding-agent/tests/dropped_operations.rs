//! Future-drop behavior through the APIs an embedding application uses.

use std::future::pending;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pebble_agent::{Agent, AgentError, Tool};
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent, EventSink, EventSinkError};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    scripted_client, text_response, tool_call_response,
};
use pebble_coding_agent::tools::RegisteredTool;
use pebble_coding_agent::{CodingAgent, CompactionOptions, Error, ShutdownReason};
use serde_json::json;
use tokio::sync::{Notify, mpsc};
use tokio::time::timeout;

const PATIENCE: Duration = Duration::from_secs(5);

#[tokio::test]
async fn dropping_a_generic_prompt_closes_it_and_releases_idle_waiters() {
    let (client, provider) = scripted_client(vec![ScriptedCall::PendingOpen]);
    let mut agent = Agent::builder(client, "test/model")
        .build()
        .expect("builds");
    let control = agent.control_handle();
    tokio::select! {
        result = agent.prompt("work") => panic!("unexpected completion: {result:?}"),
        () = provider.wait_for_call() => {}
    }
    assert!(control.is_closed());
    assert!(!control.is_running());
    timeout(PATIENCE, control.wait_for_idle())
        .await
        .expect("idle");
    assert!(matches!(
        agent.prompt("again").await,
        Err(AgentError::Closed)
    ));
}

#[tokio::test]
async fn dropping_a_generic_tool_call_cancels_its_token_and_closes_the_agent() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(tool_call_response(
        "wait",
        "call",
        json!({}),
    ))]);
    let (started, mut received) = mpsc::unbounded_channel();
    let tool = Tool::function(
        "wait",
        "Wait",
        json!({"type":"object"}),
        move |context, _| {
            started
                .send(context.cancellation().clone())
                .expect("receiver lives");
            pending()
        },
    )
    .expect("valid tool");
    let mut agent = Agent::builder(client, "test/model")
        .tools([tool])
        .build()
        .expect("builds");
    let control = agent.control_handle();
    let token = tokio::select! {
        result = agent.prompt("work") => panic!("unexpected completion: {result:?}"),
        token = received.recv() => token.expect("tool starts")
    };
    assert!(token.is_cancelled());
    assert!(control.is_closed());
    assert!(!control.is_running());
}

#[tokio::test]
async fn dropping_a_coding_prompt_closes_it_and_allows_shutdown() {
    let (client, provider) = scripted_client(vec![ScriptedCall::PendingOpen]);
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .build()
        .await
        .expect("builds");
    let control = agent.control_handle();
    tokio::select! {
        result = agent.prompt("work") => panic!("unexpected completion: {result:?}"),
        () = provider.wait_for_call() => {}
    }
    assert!(control.is_closed());
    assert!(!control.is_running());
    timeout(PATIENCE, control.wait_for_idle())
        .await
        .expect("idle");
    assert!(matches!(
        agent.prompt("again").await.result,
        Err(Error::SessionClosed)
    ));
    timeout(PATIENCE, agent.shutdown(ShutdownReason::Cancelled))
        .await
        .expect("shutdown finishes")
        .expect("shutdown succeeds");
}

#[tokio::test]
async fn dropping_a_coding_tool_call_cancels_its_token() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(tool_call_response(
        "wait",
        "call",
        json!({}),
    ))]);
    let (started, mut received) = mpsc::unbounded_channel();
    let tool = RegisteredTool::function(
        "wait",
        "Wait",
        json!({"type":"object"}),
        move |context, _| {
            started
                .send(context.cancel().clone())
                .expect("receiver lives");
            pending()
        },
    );
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .tools([tool])
        .build()
        .await
        .expect("builds");
    let control = agent.control_handle();
    let token = tokio::select! {
        result = agent.prompt("work") => panic!("unexpected completion: {result:?}"),
        token = received.recv() => token.expect("tool starts")
    };
    assert!(token.is_cancelled());
    assert!(control.is_closed());
    assert!(!control.is_running());
    agent
        .shutdown(ShutdownReason::Cancelled)
        .await
        .expect("shutdown succeeds");
}

#[tokio::test]
async fn dropping_manual_compaction_closes_the_agent_and_pairs_its_events() {
    let (client, _) = client_from(
        ScriptedProvider::new(vec![ScriptedCall::response(text_response("done"))])
            .completing(vec![ScriptedCompletion::Pending]),
    );
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .build()
        .await
        .expect("builds");
    agent.prompt("first").await.result.expect("answers");
    agent.prompt("second").await.result.expect("answers");
    let control = agent.control_handle();
    let mut events = agent.subscribe();
    tokio::select! {
        result = agent.compact(CompactionOptions::new().preserve_turns(1)) => panic!("unexpected completion: {result:?}"),
        () = async {
            loop {
                if matches!(events.recv().await.expect("event").event, CodingEvent::CompactionStarted { .. }) { break; }
            }
        } => {}
    }
    assert!(control.is_closed());
    timeout(PATIENCE, control.wait_for_idle())
        .await
        .expect("idle");
    agent
        .shutdown(ShutdownReason::Cancelled)
        .await
        .expect("shutdown succeeds");
    let mut cancelled = 0;
    while let Ok(event) = events.recv().await {
        if matches!(event.event, CodingEvent::CompactionCancelled { .. }) {
            cancelled += 1;
        }
    }
    assert_eq!(cancelled, 1);
}

struct GatedEnd {
    entered: Notify,
    release: Notify,
}

#[async_trait]
impl EventSink for GatedEnd {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        if matches!(event.event, CodingEvent::SessionEnded) {
            self.entered.notify_one();
            self.release.notified().await;
        }
        Ok(())
    }
}

#[tokio::test]
async fn shutdown_can_resume_after_its_future_is_dropped() {
    let sink = Arc::new(GatedEnd {
        entered: Notify::new(),
        release: Notify::new(),
    });
    let (client, _) = scripted_client(vec![]);
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .event_sink(sink.clone())
        .build()
        .await
        .expect("builds");
    let mut events = agent.subscribe();
    tokio::select! {
        result = agent.shutdown(ShutdownReason::Completed) => panic!("unexpected completion: {result:?}"),
        () = sink.entered.notified() => {}
    }
    assert!(agent.control_handle().is_closed());
    sink.release.notify_one();
    timeout(PATIENCE, agent.shutdown(ShutdownReason::Completed))
        .await
        .expect("shutdown finishes")
        .expect("shutdown succeeds");
    let mut ended = 0;
    while let Ok(event) = events.recv().await {
        if matches!(event.event, CodingEvent::SessionEnded) {
            ended += 1;
        }
    }
    assert_eq!(ended, 1);
}
