//! Caller cancellation reaches lifecycle hooks without closing the agent.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use lithos_llm::types::{ContentPart, Message, Response};
use pebble_agent::{
    Agent, AgentError, AgentLifecycle, AgentState, ConversationUpdate, LifecycleError, Tool,
    ToolCatalog, TurnContext,
};
use pebble_coding_agent::test_support::{
    ScriptedCall, scripted_client, text_response, tool_call_response,
};
use serde_json::json;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    BeforeModel,
    AfterDiscovery,
    AfterModel,
}

struct WaitingLifecycle {
    stage:   Stage,
    waited:  AtomicBool,
    started: mpsc::UnboundedSender<CancellationToken>,
}

impl WaitingLifecycle {
    async fn wait_once(&self, stage: Stage, cancel: &CancellationToken) -> ConversationUpdate {
        if stage == self.stage && !self.waited.swap(true, Ordering::SeqCst) {
            self.started
                .send(cancel.clone())
                .expect("test observes the hook");
            cancel.cancelled().await;
        }
        ConversationUpdate::unchanged()
    }
}

#[async_trait]
impl AgentLifecycle for WaitingLifecycle {
    async fn before_model(
        &self,
        _context: TurnContext<'_>,
        cancel: &CancellationToken,
    ) -> Result<ConversationUpdate, LifecycleError> {
        Ok(self.wait_once(Stage::BeforeModel, cancel).await)
    }
    async fn after_tool_discovery(
        &self,
        _context: TurnContext<'_>,
        _tools: &ToolCatalog,
        cancel: &CancellationToken,
    ) -> Result<ConversationUpdate, LifecycleError> {
        Ok(self.wait_once(Stage::AfterDiscovery, cancel).await)
    }
    async fn after_model(
        &self,
        _context: TurnContext<'_>,
        _response: &Response,
        cancel: &CancellationToken,
    ) -> Result<ConversationUpdate, LifecycleError> {
        Ok(self.wait_once(Stage::AfterModel, cancel).await)
    }
}

#[tokio::test]
async fn caller_cancellation_releases_each_hook_and_keeps_the_agent_reusable() {
    for stage in [Stage::BeforeModel, Stage::AfterDiscovery, Stage::AfterModel] {
        let (client, _) = scripted_client(vec![
            ScriptedCall::response(tool_call_response("inspect", "call_1", json!({}))),
            ScriptedCall::response(text_response("done")),
        ]);
        let (started, mut received) = mpsc::unbounded_channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let tool = Tool::function(
            "inspect",
            "Inspect",
            json!({"type":"object"}),
            move |_, _| {
                observed.fetch_add(1, Ordering::SeqCst);
                async { Ok("result".into()) }
            },
        )
        .expect("valid tool");
        let mut agent = Agent::builder(client, "test/model")
            .tools([tool])
            .lifecycle(Arc::new(WaitingLifecycle {
                stage,
                waited: AtomicBool::new(false),
                started,
            }))
            .build()
            .expect("builds");
        let cancel = CancellationToken::new();
        let mut prompt = Box::pin(agent.prompt_with_cancellation("work", &cancel));
        let hook_cancel = tokio::select! {
            result = &mut prompt => panic!("unexpected completion: {result:?}"),
            token = received.recv() => token.expect("hook starts"),
        };
        cancel.cancel();
        assert!(hook_cancel.is_cancelled());
        assert!(matches!(
            timeout(Duration::from_secs(5), &mut prompt)
                .await
                .expect("cancellation reaches hook"),
            Err(AgentError::Aborted)
        ));
        drop(prompt);
        assert_eq!(agent.state(), AgentState::Idle);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        if stage == Stage::AfterModel {
            let result = agent
                .messages()
                .iter()
                .flat_map(Message::content)
                .find_map(|part| {
                    if let ContentPart::ToolResult(result) = part {
                        Some(result)
                    } else {
                        None
                    }
                })
                .expect("pending call receives a result");
            assert_eq!(result.tool_call_id, "call_1");
            assert!(result.is_error);
        }
        assert_eq!(
            timeout(Duration::from_secs(5), agent.prompt("again"))
                .await
                .expect("second prompt finishes")
                .expect("agent reusable")
                .text(),
            "done"
        );
        agent.shutdown();
    }
}
