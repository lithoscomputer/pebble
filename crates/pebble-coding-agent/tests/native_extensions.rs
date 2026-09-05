//! Native embedding contracts: tool content and per-request application policy.

use std::future::pending;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lithos_llm::types::{ContentPart, ImageContent, MediaSource, Message as LlmMessage, Role};
use pebble_agent::{
    LifecycleError, ToolCallNext, ToolCallRequest, ToolMiddleware, ToolOutcome, ToolSystemError,
};
use pebble_coding_agent::events::{CodingEvent, TokenUsage};
use pebble_coding_agent::extensions::{
    CompactionPolicy, CompactionPreparation, CompactionSummary, ContextPolicy, ContextPreparation,
};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, scripted_client, text_response, tool_call_response,
};
use pebble_coding_agent::tools::{RegisteredTool, ToolArtifact, ToolOutput};
use pebble_coding_agent::{CodingAgent, CompactionOptions, ShutdownReason};
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

struct ObserveDetails(Arc<AtomicUsize>);

#[async_trait]
impl ToolMiddleware for ObserveDetails {
    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> Result<ToolOutcome, ToolSystemError> {
        let outcome = next.run(request).await?;
        assert_eq!(
            outcome.metadata().details,
            Some(json!({"private_annotation":"observer-only"}))
        );
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(outcome)
    }
}

#[tokio::test]
async fn rich_content_reaches_the_model_and_details_reach_middleware_and_durable_events() {
    let image = ContentPart::Image(ImageContent::new(MediaSource::url(
        "https://example.test/chart.png",
    )));
    let expected_image = image.clone();
    let tool = RegisteredTool::rich_function(
        "chart",
        "Draw a chart",
        json!({"type":"object"}),
        move |_, _| {
            let image = image.clone();
            async move {
                Ok(ToolOutput::new(vec![
                    ContentPart::Text {
                        text: "a chart".into(),
                    },
                    image,
                ])
                .with_details(json!({"private_annotation":"observer-only"}))
                .with_artifact(ToolArtifact {
                    reference:   "opaque-chart-ref".into(),
                    label:       "chart".into(),
                    media_type:  "image/png".into(),
                    byte_length: 123,
                }))
            }
        },
    );
    let (client, provider) = scripted_client(vec![
        ScriptedCall::response(tool_call_response("chart", "chart-1", json!({}))),
        ScriptedCall::response(text_response("done")),
    ]);
    let seen = Arc::new(AtomicUsize::new(0));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .tools([tool])
        .tool_middleware(Arc::new(ObserveDetails(Arc::clone(&seen))))
        .build()
        .await
        .expect("builds");
    let mut events = agent.subscribe();
    agent.prompt("draw it").await.expect("answers");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
    let requests = provider.requests();
    let result = requests[1]
        .messages()
        .iter()
        .flat_map(LlmMessage::content)
        .find_map(|part| {
            if let ContentPart::ToolResult(result) = part {
                Some(result)
            } else {
                None
            }
        })
        .expect("tool result");
    assert_eq!(result.content, vec![
        ContentPart::Text {
            text: "a chart".into(),
        },
        expected_image
    ]);
    let wire = serde_json::to_string(&requests[1]).expect("request serializes");
    assert!(!wire.contains("observer-only"));
    assert!(!wire.contains("opaque-chart-ref"));
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    let mut completions = 0;
    while let Ok(event) = events.recv().await {
        if let CodingEvent::ToolCallCompleted { metadata, .. } = event.event {
            assert_eq!(
                metadata.details,
                Some(json!({"private_annotation":"observer-only"}))
            );
            assert_eq!(metadata.artifacts[0].reference, "opaque-chart-ref");
            completions += 1;
        }
    }
    assert_eq!(completions, 1);
    assert!(
        !serde_json::to_string(&agent.to_record())
            .expect("record")
            .contains("observer-only")
    );
}

struct AddContext(AtomicUsize);

#[async_trait]
impl ContextPolicy for AddContext {
    async fn prepare(
        &self,
        context: ContextPreparation<'_>,
        _: &CancellationToken,
    ) -> Result<Option<Vec<LlmMessage>>, LifecycleError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        assert!(!context.session.session_id().as_str().is_empty());
        let mut messages = context.turn.messages().to_vec();
        assert!(
            !serde_json::to_string(&messages)
                .expect("messages")
                .contains("ephemeral project context")
        );
        messages.push(LlmMessage::text(Role::User, "ephemeral project context"));
        Ok(Some(messages))
    }
}

#[tokio::test]
async fn context_is_prepared_for_every_request_without_accumulating_in_history() {
    let (client, provider) = scripted_client(vec![
        ScriptedCall::response(tool_call_response(
            "read_file",
            "read-1",
            json!({"file_path":"/missing"}),
        )),
        ScriptedCall::response(text_response("done")),
        ScriptedCall::response(text_response("done again")),
    ]);
    let policy = Arc::new(AddContext(AtomicUsize::new(0)));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .context_policy(policy.clone())
        .build()
        .await
        .expect("builds");
    agent.prompt("first").await.expect("first answer");
    agent.prompt("second").await.expect("second answer");
    assert_eq!(policy.0.load(Ordering::SeqCst), 3);
    for request in provider.requests() {
        assert_eq!(
            serde_json::to_string(&request)
                .expect("request")
                .matches("ephemeral project context")
                .count(),
            1
        );
    }
    assert!(
        !serde_json::to_string(&agent.to_record())
            .expect("record")
            .contains("ephemeral project context")
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
}

struct RemoveCalls;

#[async_trait]
impl ContextPolicy for RemoveCalls {
    async fn prepare(
        &self,
        context: ContextPreparation<'_>,
        _: &CancellationToken,
    ) -> Result<Option<Vec<LlmMessage>>, LifecycleError> {
        Ok(Some(
            context
                .turn
                .messages()
                .iter()
                .filter(|message| {
                    !message
                        .content()
                        .iter()
                        .any(|part| matches!(part, ContentPart::ToolCall(_)))
                })
                .cloned()
                .collect(),
        ))
    }
}

#[tokio::test]
async fn a_context_policy_cannot_send_orphaned_tool_results() {
    let (client, provider) = scripted_client(vec![ScriptedCall::response(tool_call_response(
        "read_file",
        "read-1",
        json!({"file_path":"/missing"}),
    ))]);
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .context_policy(Arc::new(RemoveCalls))
        .build()
        .await
        .expect("builds");
    assert!(agent.prompt("read").await.is_err());
    assert_eq!(provider.requests().len(), 1);
    assert_eq!(
        agent.history().turns().len(),
        3,
        "committed call and result remain paired"
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
}

struct PendingOnce(Notify, AtomicUsize);

#[async_trait]
impl ContextPolicy for PendingOnce {
    async fn prepare(
        &self,
        _: ContextPreparation<'_>,
        _: &CancellationToken,
    ) -> Result<Option<Vec<LlmMessage>>, LifecycleError> {
        if self.1.fetch_add(1, Ordering::SeqCst) == 0 {
            self.0.notify_one();
            pending().await
        } else {
            Ok(None)
        }
    }
}

#[tokio::test]
async fn cancelling_a_context_hook_leaves_the_agent_reusable() {
    let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
    let policy = Arc::new(PendingOnce(Notify::new(), AtomicUsize::new(0)));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .context_policy(policy.clone())
        .build()
        .await
        .expect("builds");
    let cancel = CancellationToken::new();
    let (result, ()) = tokio::join!(agent.prompt_with_cancellation("first", &cancel), async {
        policy.0.notified().await;
        cancel.cancel();
    });
    assert!(result.is_err());
    assert!(provider.requests().is_empty());
    agent.prompt("second").await.expect("can prompt again");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
}

struct Summarize(Mutex<Vec<usize>>);

#[async_trait]
impl CompactionPolicy for Summarize {
    async fn summarize(
        &self,
        context: CompactionPreparation<'_>,
        _: &CancellationToken,
    ) -> Result<CompactionSummary, LifecycleError> {
        assert!(!context.retained_messages.is_empty());
        assert!(!context.default_request.messages().is_empty());
        self.0.lock().expect("lock").push(context.messages.len());
        Ok(CompactionSummary {
            text:            "application handoff".into(),
            usage:           TokenUsage {
                input: 7,
                output: 3,
                ..TokenUsage::default()
            },
            cost_usd_micros: Some(19),
        })
    }
}

#[tokio::test]
async fn application_compaction_keeps_recent_turns_and_records_usage() {
    let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
    let policy = Arc::new(Summarize(Mutex::new(Vec::new())));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .compaction_policy(policy.clone())
        .build()
        .await
        .expect("builds");
    agent.prompt("first").await.expect("first");
    agent.prompt("second").await.expect("second");
    let before = agent.history();
    let result = agent
        .compact(CompactionOptions::new().preserve_turns(1))
        .await
        .expect("compacts");
    let pebble_coding_agent::CompactionOutcome::Compacted(result) = result else {
        panic!("expected compaction");
    };
    assert!(result.summary().contains("application handoff"));
    assert_eq!(result.usage().input, 7);
    assert_eq!(result.cost_usd_micros(), Some(19));
    assert_eq!(*policy.0.lock().expect("lock"), vec![3]);
    assert_eq!(
        agent.history().to_llm_messages().last(),
        before.to_llm_messages().last()
    );
    assert!(provider.completion_requests().is_empty());
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
}

struct InvalidSummary {
    empty: bool,
}

#[async_trait]
impl CompactionPolicy for InvalidSummary {
    async fn summarize(
        &self,
        _: CompactionPreparation<'_>,
        _: &CancellationToken,
    ) -> Result<CompactionSummary, LifecycleError> {
        if !self.empty {
            return Err(LifecycleError::new("Application summarizer unavailable"));
        }
        Ok(CompactionSummary {
            text:            " \n ".into(),
            usage:           TokenUsage::default(),
            cost_usd_micros: None,
        })
    }
}

#[tokio::test]
async fn failed_or_empty_application_summaries_leave_history_intact() {
    for empty in [false, true] {
        let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .compaction_policy(Arc::new(InvalidSummary { empty }))
            .build()
            .await
            .expect("builds");
        agent.prompt("first").await.expect("first");
        agent.prompt("second").await.expect("second");
        let before = agent.to_record().messages;
        let mut events = agent.subscribe();
        assert!(
            agent
                .compact(CompactionOptions::new().preserve_turns(1))
                .await
                .is_err()
        );
        assert_eq!(agent.to_record().messages, before);
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shutdown");
        let mut failures = 0;
        while let Ok(event) = events.recv().await {
            if matches!(event.event, CodingEvent::CompactionFailed { .. }) {
                failures += 1;
            }
            assert!(!matches!(
                event.event,
                CodingEvent::CompactionCompleted { .. }
            ));
        }
        assert_eq!(failures, 1);
    }
}

#[async_trait]
impl CompactionPolicy for PendingOnce {
    async fn summarize(
        &self,
        _: CompactionPreparation<'_>,
        _: &CancellationToken,
    ) -> Result<CompactionSummary, LifecycleError> {
        self.0.notify_one();
        pending().await
    }
}

#[tokio::test]
async fn cancelling_application_compaction_leaves_history_intact() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
    let policy = Arc::new(PendingOnce(Notify::new(), AtomicUsize::new(0)));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .compaction_policy(policy.clone())
        .build()
        .await
        .expect("builds");
    agent.prompt("first").await.expect("first");
    agent.prompt("second").await.expect("second");
    let before = agent.to_record().messages;
    let cancel = CancellationToken::new();
    let mut events = agent.subscribe();
    let (result, ()) = tokio::join!(
        agent.compact_with_cancellation(CompactionOptions::new().preserve_turns(1), &cancel),
        async {
            policy.0.notified().await;
            cancel.cancel();
        }
    );
    assert!(result.is_err());
    assert_eq!(agent.to_record().messages, before);
    agent.prompt("still usable").await.expect("answers");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
    let mut cancelled = 0;
    while let Ok(event) = events.recv().await {
        if matches!(event.event, CodingEvent::CompactionCancelled { .. }) {
            cancelled += 1;
        }
    }
    assert_eq!(cancelled, 1);
}

#[tokio::test]
async fn automatic_compaction_uses_the_application_summarizer() {
    use pebble_coding_agent::CodingAgentOptions;
    use pebble_coding_agent::test_support::with_input_tokens;
    let (client, provider) = scripted_client(vec![ScriptedCall::response(with_input_tokens(
        text_response("done"),
        180_000,
    ))]);
    let policy = Arc::new(Summarize(Mutex::new(Vec::new())));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .compaction_policy(policy.clone())
        .options(
            CodingAgentOptions::default()
                .with_compaction_threshold_percent(1)
                .with_compaction_preserve_turns(2),
        )
        .build()
        .await
        .expect("builds");
    agent.prompt("first").await.expect("first");
    agent.prompt("second").await.expect("second");
    assert!(!policy.0.lock().expect("lock").is_empty());
    assert!(provider.completion_requests().is_empty());
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
}
