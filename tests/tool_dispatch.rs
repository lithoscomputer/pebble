//! The tool seam, end to end: a registered tool, dispatched against a mock
//! environment, publishing a round's events through a real pipeline.
//!
//! This exercises what the pieces only promise separately — that the registry,
//! the dispatch pipeline, the output budgets, the hooks, and the event pump
//! compose into one round an application can observe.

#![cfg(feature = "test-util")]

use std::sync::{Arc, Mutex, PoisonError};

use pebble::test_support::MockEnvironment;
use pebble::{
    AgentEvent, ContentPart, Emitter, Environment, EventOptions, EventPump, RegisteredTool,
    SessionEvent, SessionOptions, ToolCall, ToolContext, ToolDefinition, ToolDispatch, ToolError,
    ToolErrorKind, ToolHookCallback, ToolHookDecision, ToolRegistry, ToolResult, ToolSource,
};
use serde_json::{Value, json};
use tokio::sync::broadcast::Receiver;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// An event pipeline whose events can be read once the round is over.
struct Events {
    emitter:  Emitter,
    pump:     JoinHandle<pebble::Result<()>>,
    received: Receiver<SessionEvent>,
}

impl Events {
    fn new() -> Self {
        let (emitter, pump) = EventPump::new(EventOptions::default());
        let received = emitter.subscribe();
        Self {
            emitter,
            pump: tokio::spawn(pump.run()),
            received,
        }
    }

    /// Stops the pipeline and returns everything it published.
    async fn drain(self) -> Vec<SessionEvent> {
        let Self {
            emitter,
            pump,
            mut received,
        } = self;
        drop(emitter);
        pump.await
            .expect("the pump task joins")
            .expect("the pump finishes");

        let mut events = Vec::new();
        while let Ok(event) = received.try_recv() {
            events.push(event);
        }
        events
    }
}

/// Records the order the hooks were called in, and what they were told.
#[derive(Debug, Default)]
struct HookLog(Mutex<Vec<String>>);

impl HookLog {
    fn entries(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn push(&self, entry: String) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(entry);
    }
}

#[async_trait::async_trait]
impl ToolHookCallback for HookLog {
    async fn pre_tool_use(&self, tool_name: &str, _tool_input: &Value) -> ToolHookDecision {
        self.push(format!("pre {tool_name}"));
        ToolHookDecision::Proceed
    }

    async fn post_tool_use(&self, tool_name: &str, tool_call_id: &str, _tool_output: &str) {
        self.push(format!("post {tool_name} {tool_call_id}"));
    }

    async fn post_tool_use_failure(
        &self,
        tool_name: &str,
        tool_call_id: &str,
        _error: &str,
        error_kind: ToolErrorKind,
    ) {
        self.push(format!("fail {tool_name} {tool_call_id} {error_kind:?}"));
    }
}

/// A tool that reads a file through the session's environment and reports what
/// it found, so the round exercises the environment seam as well.
fn read_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "read_file",
            "Read a file",
            json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
        ),
        executor:   Arc::new(|arguments: Value, context: ToolContext| {
            Box::pin(async move {
                let path = arguments["path"].as_str().unwrap_or_default().to_owned();
                let content = context.env.read_file_text(&path).await?;
                Ok(content)
            })
        }),
        source:     ToolSource::Native,
    }
}

fn failing_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function("boom", "Always fails", json!({})),
        executor:   Arc::new(|_arguments, _context| {
            Box::pin(async { Err(ToolError::execution("the tool could not finish")) })
        }),
        source:     ToolSource::Native,
    }
}

fn environment() -> Arc<dyn Environment> {
    Arc::new(MockEnvironment {
        files: [(
            "/work/notes.md".to_owned(),
            "hello from the mock".to_owned(),
        )]
        .into_iter()
        .collect(),
        ..MockEnvironment::default()
    })
}

fn text_of(result: &ToolResult) -> &str {
    match result.content.as_slice() {
        [ContentPart::Text { text }] => text.as_str(),
        other => panic!("expected one text part, got {other:?}"),
    }
}

/// The tool-call events for one call, in published order.
fn round_for<'a>(events: &'a [SessionEvent], tool_call_id: &str) -> Vec<&'a AgentEvent> {
    events
        .iter()
        .filter(|event| event.tool_call_id.as_deref() == Some(tool_call_id))
        .map(|event| &event.event)
        .collect()
}

#[tokio::test]
async fn a_round_runs_its_tools_publishes_their_events_and_calls_the_hooks_in_order() {
    let mut registry = ToolRegistry::new();
    registry.register(read_tool());
    registry.register(failing_tool());
    let environment = environment();
    let hooks = Arc::new(HookLog::default());
    let config = SessionOptions {
        tool_hooks: Some(Arc::clone(&hooks) as Arc<dyn ToolHookCallback>),
        ..SessionOptions::default()
    };
    let events = Events::new();
    let calls = [
        ToolCall::function("call_read", "read_file", json!({"path": "/work/notes.md"})),
        ToolCall::function("call_boom", "boom", json!({})),
    ];

    let results = ToolDispatch::new(
        &registry,
        &environment,
        &config,
        &events.emitter,
        "ses_1",
        "ses_1",
    )
    .execute(&calls, true, &CancellationToken::new())
    .await;

    // Every call is answered, in call order.
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].tool_call_id, "call_read");
    assert!(!results[0].is_error);
    assert_eq!(text_of(&results[0]), "hello from the mock");
    assert_eq!(results[1].tool_call_id, "call_boom");
    assert!(results[1].is_error);
    assert_eq!(text_of(&results[1]), "the tool could not finish");

    // Each call publishes started, then its output, then its completion.
    let published = events.drain().await;
    for event in &published {
        assert_eq!(event.session_id, "ses_1");
        assert!(event.seq > 0, "the pump numbers every published event");
    }

    let read_round = round_for(&published, "call_read");
    assert!(matches!(read_round.as_slice(), [
        AgentEvent::ToolCallStarted {
            tool_name,
            tool_call_id,
            ..
        },
        AgentEvent::ToolCallOutputDelta { delta },
        AgentEvent::ToolCallCompleted {
            output,
            is_error: false,
            error_kind: None,
            output_bytes_observed,
            output_bytes_retained,
            output_bytes_omitted: 0,
            ..
        },
    ] if tool_name == "read_file"
        && tool_call_id == "call_read"
        && delta == "hello from the mock"
        && output == &json!("hello from the mock")
        && *output_bytes_observed == "hello from the mock".len()
        && *output_bytes_retained == "hello from the mock".len()));

    let failed_round = round_for(&published, "call_boom");
    assert!(matches!(failed_round.as_slice(), [
        AgentEvent::ToolCallStarted { .. },
        AgentEvent::ToolCallOutputDelta { .. },
        AgentEvent::ToolCallCompleted {
            is_error: true,
            error_kind: Some(ToolErrorKind::Execution),
            ..
        },
    ]));

    // Each call's hooks bracket it: the pre-hook before the tool ran, and
    // exactly one post-hook, chosen by whether the call failed.
    let logged = hooks.entries();
    assert_eq!(logged.len(), 4, "{logged:?}");
    let read_pre = logged
        .iter()
        .position(|entry| entry == "pre read_file")
        .expect("the read tool's pre-hook ran");
    let read_post = logged
        .iter()
        .position(|entry| entry == "post read_file call_read")
        .expect("the read tool's success hook ran");
    assert!(read_pre < read_post);
    assert!(logged.contains(&"pre boom".to_owned()));
    assert!(logged.contains(&"fail boom call_boom Execution".to_owned()));
}

#[tokio::test]
async fn output_past_the_session_budget_is_bounded_before_anything_else_sees_it() {
    let mut registry = ToolRegistry::new();
    registry.register(RegisteredTool {
        definition: ToolDefinition::function("read_file", "Read a file", json!({})),
        executor:   Arc::new(|_arguments, _context| Box::pin(async { Ok("x".repeat(80_000)) })),
        source:     ToolSource::Native,
    });
    let environment = environment();
    let config = SessionOptions {
        tool_output_retention_bytes: 4_096,
        ..SessionOptions::default()
    };
    let events = Events::new();

    let result = ToolDispatch::new(
        &registry,
        &environment,
        &config,
        &events.emitter,
        "ses_1",
        "ses_1",
    )
    .execute_one(
        &ToolCall::function("call_1", "read_file", json!({})),
        CancellationToken::new(),
    )
    .await;

    let published = events.drain().await;
    let AgentEvent::ToolCallCompleted {
        output,
        is_error,
        output_bytes_observed,
        output_bytes_retained,
        output_bytes_omitted,
        ..
    } = published
        .iter()
        .map(|event| &event.event)
        .find(|event| matches!(event, AgentEvent::ToolCallCompleted { .. }))
        .expect("a completion event")
    else {
        unreachable!("the event was matched above")
    };

    // The retention budget bounds what an application observes, and the
    // counters describe what that cost.
    let observed = output.as_str().expect("string output");
    assert!(observed.len() <= config.tool_output_retention_bytes);
    assert!(observed.starts_with("Warning: truncated output"));
    assert!(!is_error, "a bounded output is not a failed call");
    assert_eq!(*output_bytes_observed, 80_000);
    assert!(*output_bytes_retained < config.tool_output_retention_bytes);
    assert_eq!(*output_bytes_omitted, 80_000 - output_bytes_retained);

    // History keeps a smaller copy still: the read tool's own limit.
    assert!(text_of(&result).len() <= observed.len());
    assert!(!result.is_error);
}
