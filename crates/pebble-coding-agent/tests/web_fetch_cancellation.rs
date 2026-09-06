//! Cancelling a fetch also ends the model call that summarizes its content.
use std::sync::Arc;
use std::time::Duration;

use futures_util::poll;
use pebble_coding_agent::events::{CodingEvent, ToolErrorKind};
use pebble_coding_agent::state::Message;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    text_response, tool_call_response,
};
use pebble_coding_agent::{CodingAgent, Error, ShutdownReason};
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
