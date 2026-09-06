//! Application tools declare the same scheduling rules as native tools.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use pebble_agent::ToolScheduling;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, multi_tool_call_response, scripted_client, text_response,
};
use pebble_coding_agent::tools::RegisteredTool;
use pebble_coding_agent::{CodingAgent, ShutdownReason};
use serde_json::json;
use tokio::sync::Barrier;
use tokio::task::yield_now;
use tokio::time::timeout;

#[tokio::test]
async fn sequential_application_calls_preserve_updates_and_model_order() {
    let value = Arc::new(AtomicUsize::new(0));
    let state = value.clone();
    let update = RegisteredTool::function(
        "update",
        "Update state",
        json!({"type":"object"}),
        move |_, _| {
            let state = state.clone();
            async move {
                let previous = state.load(Ordering::SeqCst);
                yield_now().await;
                state.store(previous + 1, Ordering::SeqCst);
                Ok((previous + 1).to_string())
            }
        },
    )
    .with_scheduling(ToolScheduling::Sequential);
    let state = value.clone();
    let read = RegisteredTool::function(
        "read",
        "Read state",
        json!({"type":"object"}),
        move |_, _| {
            let state = state.clone();
            async move {
                assert_eq!(
                    state.load(Ordering::SeqCst),
                    2,
                    "the read follows both writes"
                );
                Ok("2".to_owned())
            }
        },
    );
    run_round(vec![update, read], &["update", "update", "read"]).await;
    assert_eq!(value.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn concurrent_application_calls_can_make_progress_together() {
    let barrier = Arc::new(Barrier::new(2));
    let tool = RegisteredTool::function("read", "Read", json!({"type":"object"}), move |_, _| {
        let barrier = barrier.clone();
        async move {
            barrier.wait().await;
            Ok("done".to_owned())
        }
    })
    .with_scheduling(ToolScheduling::Concurrent);
    run_round(vec![tool], &["read", "read"]).await;
}

#[tokio::test]
async fn human_input_remains_exclusive_in_either_builder_order() {
    for mark_human_first in [false, true] {
        let peer_calls = Arc::new(AtomicUsize::new(0));
        let tool = RegisteredTool::function("ask", "Ask", json!({"type":"object"}), |_, _| async {
            Ok("answer".to_owned())
        });
        let tool = if mark_human_first {
            tool.requires_human_input()
                .with_scheduling(ToolScheduling::Concurrent)
        } else {
            tool.with_scheduling(ToolScheduling::Concurrent)
                .requires_human_input()
        };
        let calls = peer_calls.clone();
        let peer =
            RegisteredTool::function("peer", "Peer", json!({"type":"object"}), move |_, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok("peer".to_owned()) }
            });
        run_round(vec![tool, peer], &["ask", "peer"]).await;
        assert_eq!(peer_calls.load(Ordering::SeqCst), 0);
    }
}

async fn run_round(tools: Vec<RegisteredTool>, names: &[&str]) {
    let ids: Vec<_> = (0..names.len())
        .map(|index| format!("call-{index}"))
        .collect();
    let calls = names
        .iter()
        .zip(&ids)
        .map(|(name, id)| (*name, id.as_str(), json!({})))
        .collect();
    let (client, _) = scripted_client(vec![
        ScriptedCall::response(multi_tool_call_response(calls)),
        ScriptedCall::response(text_response("done")),
    ]);
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .tools(tools)
        .build()
        .await
        .expect("builds");
    timeout(Duration::from_secs(5), agent.prompt("work"))
        .await
        .expect("round completes")
        .result
        .expect("prompt succeeds");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
}
