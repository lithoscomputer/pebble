//! Pebble's coding tools, run without a coding agent.
//!
//! A workflow step sometimes needs one tool call answered the way a session
//! would answer it: the same middleware, failure rendering, output budgets,
//! and events. This is the
//! supported route for that, exercised from outside the crate.

use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::types::{ContentPart, ToolCall, ToolDefinition, ToolResult};
use pebble_agent::{ToolCallNext, ToolCallRequest, ToolMiddleware, ToolOutcome, ToolSystemError};
use pebble_coding_agent::environment::Environment;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use pebble_coding_agent::test_support::{DenyTool, FixedPermission, MockEnvironment};
use pebble_coding_agent::tools::{
    ApprovalDecision, CodingToolSet, PermissionMiddleware, RegisteredTool, ToolApprovalService,
    ToolError, ToolErrorKind, ToolPermission, ToolRunner, ToolSource,
};
use pebble_coding_agent::{CodingAgentOptions, SessionId, SessionScope};
use serde_json::json;
use tokio_util::sync::CancellationToken;

struct RejectApproval;

#[async_trait::async_trait]
impl ToolApprovalService for RejectApproval {
    async fn approve(
        &self,
        request: &ToolCallRequest,
    ) -> Result<ApprovalDecision, ToolSystemError> {
        Ok(ApprovalDecision::Deny {
            reason: format!("{} is not allowed by approval", request.call().name),
        })
    }
}

/// Records the order the middleware was called in.
#[derive(Debug, Default)]
struct MiddlewareLog(Mutex<Vec<String>>);

impl MiddlewareLog {
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
impl ToolMiddleware for MiddlewareLog {
    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> Result<ToolOutcome, ToolSystemError> {
        let name = request.call().name.clone();
        let id = request.call().id.clone();
        self.push(format!("pre {name}"));
        let outcome = next.run(request).await?;
        match &outcome {
            ToolOutcome::Success { .. } => self.push(format!("post {name} {id}")),
            ToolOutcome::Failure { kind, .. } => self.push(format!("fail {name} {id} {kind:?}")),
            _ => self.push(format!("fail {name} {id} unknown")),
        }
        Ok(outcome)
    }
}

/// Every event a runner published, in order.
#[derive(Default)]
struct EventLog(Mutex<Vec<CodingAgentEvent>>);

impl EventLog {
    fn events(&self) -> Vec<CodingAgentEvent> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn completion(&self) -> CodingEvent {
        self.events()
            .into_iter()
            .map(|event| event.event)
            .find(|event| matches!(event, CodingEvent::ToolCallCompleted { .. }))
            .expect("a completion event")
    }
}

fn mock_environment() -> Arc<MockEnvironment> {
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

/// A runner over the core tools that logs its events.
fn runner_with(
    environment: Arc<dyn Environment>,
    options: CodingAgentOptions,
) -> (ToolRunner, Arc<EventLog>) {
    let log = Arc::new(EventLog::default());
    let recorder = Arc::clone(&log);
    let runner = ToolRunner::new(CodingToolSet::core(), environment)
        .options(options)
        .session_id(SessionId::new("runner-1"))
        .on_event(move |event| {
            recorder
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event);
        });
    (runner, log)
}

#[test]
fn the_core_set_describes_the_file_and_command_tools() {
    let set = CodingToolSet::core();

    let mut names = set.names();
    names.sort();
    assert_eq!(names, [
        "edit_file",
        "glob",
        "grep",
        "read_file",
        "shell",
        "write_file"
    ]);
    let summaries = set.summaries();
    assert_eq!(summaries.len(), 6);
    assert!(
        summaries
            .iter()
            .all(|tool| tool.source == ToolSource::Native)
    );
    assert!(
        summaries.iter().all(|tool| !tool.description.is_empty()),
        "every tool says what it is for"
    );
    assert_eq!(set.definitions().len(), 6);
}

#[tokio::test]
async fn a_runner_reads_and_writes_through_the_environment_and_reports_each_call() {
    let mock = mock_environment();
    let (runner, log) = runner_with(
        Arc::clone(&mock) as Arc<dyn Environment>,
        CodingAgentOptions::default(),
    );

    let read = runner
        .run(
            &ToolCall::function(
                "call_read",
                "read_file",
                json!({"file_path": "/work/notes.md"}),
            ),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");
    assert!(!read.is_error, "{}", text_of(&read));
    assert!(text_of(&read).contains("hello from the mock"));

    let written = runner
        .run(
            &ToolCall::function(
                "call_write",
                "write_file",
                json!({"file_path": "/work/out.txt", "content": "written by the runner"}),
            ),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");
    assert!(!written.is_error, "{}", text_of(&written));
    assert_eq!(
        mock.written_files
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_slice(),
        [(
            "/work/out.txt".to_owned(),
            "written by the runner".to_owned()
        )]
    );

    // Each call published started, output, and completed, stamped with the
    // runner's session and the call, before `run` returned.
    let events = log.events();
    let for_call = |id: &str| -> Vec<&CodingEvent> {
        events
            .iter()
            .filter(|event| event.tool_call_id.as_deref() == Some(id))
            .map(|event| &event.event)
            .collect()
    };
    assert!(events.iter().all(|event| event.session_id == "runner-1"));
    assert!(matches!(for_call("call_read").as_slice(), [
        CodingEvent::ToolCallStarted { tool_name, .. },
        CodingEvent::ToolCallOutputDelta { .. },
        CodingEvent::ToolCallCompleted {
            is_error: false,
            ..
        },
    ] if tool_name == "read_file"));
    assert!(matches!(for_call("call_write").as_slice(), [
        CodingEvent::ToolCallStarted { .. },
        CodingEvent::ToolCallOutputDelta { .. },
        CodingEvent::ToolCallCompleted {
            is_error: false,
            ..
        },
    ]));
}

#[tokio::test]
async fn a_runner_applies_the_policy_a_session_would() {
    let (runner, log) = runner_with(mock_environment(), CodingAgentOptions::default());
    let runner = runner.tool_middleware(Arc::new(PermissionMiddleware::new(Arc::new(DenyTool(
        "shell",
    )))));

    let result = runner
        .run(
            &ToolCall::function("call_shell", "shell", json!({"command": "rm -rf /"})),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");

    assert!(result.is_error);
    assert_eq!(text_of(&result), "shell denied by tool permission policy");
    assert!(matches!(log.completion(), CodingEvent::ToolCallCompleted {
        is_error: true,
        error_kind: Some(ToolErrorKind::Denied),
        ..
    }));
}

#[tokio::test]
async fn a_runner_calls_middleware_around_each_call() {
    let middleware = Arc::new(MiddlewareLog::default());
    let (runner, _log) = runner_with(mock_environment(), CodingAgentOptions::default());
    let runner = runner.tool_middleware(Arc::clone(&middleware) as Arc<dyn ToolMiddleware>);

    let ok = runner
        .run(
            &ToolCall::function(
                "call_read",
                "read_file",
                json!({"file_path": "/work/notes.md"}),
            ),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");
    let missing = runner
        .run(
            &ToolCall::function(
                "call_miss",
                "read_file",
                json!({"file_path": "/work/absent.md"}),
            ),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");

    assert!(!ok.is_error);
    assert!(missing.is_error);
    assert_eq!(middleware.entries(), [
        "pre read_file",
        "post read_file call_read",
        "pre read_file",
        "fail read_file call_miss Execution",
    ]);
}

#[tokio::test]
async fn an_approval_that_blocks_a_call_answers_it_with_the_reason() {
    let (runner, _log) = runner_with(mock_environment(), CodingAgentOptions::default());
    let runner = runner.tool_middleware(Arc::new(
        PermissionMiddleware::new(Arc::new(FixedPermission(ToolPermission::RequireApproval)))
            .with_approval(Arc::new(RejectApproval)),
    ));

    let result = runner
        .run(
            &ToolCall::function(
                "call_read",
                "read_file",
                json!({"file_path": "/work/notes.md"}),
            ),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");

    assert!(result.is_error);
    assert_eq!(text_of(&result), "read_file is not allowed by approval");
}

#[tokio::test]
async fn a_runner_bounds_output_like_a_session() {
    let log = Arc::new(EventLog::default());
    let recorder = Arc::clone(&log);
    let big = RegisteredTool::new(
        ToolDefinition::function("big", "Answers with a lot", json!({})),
        Arc::new(|_arguments, _context| Box::pin(async { Ok("x".repeat(80_000)) })),
    )
    .with_source(ToolSource::Application);
    let runner = ToolRunner::new(
        CodingToolSet::core().with_tool(big).expect("distinct tool"),
        mock_environment() as Arc<dyn Environment>,
    )
    .options(CodingAgentOptions::default().with_tool_output_retention_bytes(4_096))
    .on_event(move |event| {
        recorder
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    });

    let result = runner
        .run(
            &ToolCall::function("call_big", "big", json!({})),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");

    assert!(!result.is_error, "a bounded output is not a failed call");
    let CodingEvent::ToolCallCompleted {
        output,
        output_bytes_observed,
        output_bytes_retained,
        output_bytes_omitted,
        ..
    } = log.completion()
    else {
        unreachable!("the event was matched above")
    };
    let observed = output.as_str().expect("string output");
    assert!(observed.len() <= 4_096);
    assert!(observed.starts_with("Warning: truncated output"));
    assert_eq!(output_bytes_observed, 80_000);
    assert!(output_bytes_retained < 4_096);
    assert_eq!(output_bytes_omitted, 80_000 - output_bytes_retained);
}

#[tokio::test]
async fn an_unknown_tool_and_a_cancelled_call_are_both_answered() {
    let cancelled_tool = RegisteredTool::new(
        ToolDefinition::function("waits", "Waits until cancelled", json!({})),
        Arc::new(|_arguments, context| {
            Box::pin(async move {
                context.cancel().cancelled().await;
                Err(ToolError::cancelled("Cancelled"))
            })
        }),
    )
    .with_source(ToolSource::Application);
    let runner = ToolRunner::new(
        CodingToolSet::empty()
            .with_tool(cancelled_tool)
            .expect("distinct tool"),
        mock_environment() as Arc<dyn Environment>,
    );

    let unknown = runner
        .run(
            &ToolCall::function("call_nope", "nope", json!({})),
            CancellationToken::new(),
        )
        .await
        .expect("the runner completes");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let cancelled = runner
        .run(&ToolCall::function("call_wait", "waits", json!({})), cancel)
        .await
        .expect("the runner completes");

    assert!(unknown.is_error);
    assert_eq!(text_of(&unknown), "unknown tool `nope`");
    assert!(cancelled.is_error);
    assert_eq!(text_of(&cancelled), "Cancelled");
}

#[tokio::test]
async fn an_event_callback_failure_fails_the_run() {
    let runner = ToolRunner::new(
        CodingToolSet::core(),
        mock_environment() as Arc<dyn Environment>,
    )
    .on_event(|_| panic!("callback failed"));

    let error = runner
        .run(
            &ToolCall::function(
                "call_read",
                "read_file",
                json!({"file_path": "/work/notes.md"}),
            ),
            CancellationToken::new(),
        )
        .await
        .expect_err("a failed callback makes event delivery incomplete");

    assert!(error.message().contains("event pipeline"));
}

#[tokio::test]
async fn a_runner_delivers_session_scope_separately_from_the_tool_call_id() {
    let scope = SessionScope::root(SessionId::new("runner/session"));
    let expected = scope.clone();
    let tool = RegisteredTool::function(
        "inspect_identity",
        "Inspect context",
        json!({"type":"object"}),
        move |context, _| {
            let expected = expected.clone();
            async move {
                assert_eq!(context.session_scope(), Some(&expected));
                assert_eq!(context.session_id(), Some(expected.session_id()));
                assert_eq!(context.root_session_id(), Some(expected.root_session_id()));
                assert_eq!(context.tool_call_id(), Some("model-call"));
                Ok("identified".to_owned())
            }
        },
    );
    let tools = CodingToolSet::empty()
        .with_tool(tool)
        .expect("tool registered");
    let result = ToolRunner::new(tools, mock_environment())
        .session_id(scope.session_id().clone())
        .run(
            &ToolCall::function("model-call", "inspect_identity", json!({})),
            CancellationToken::new(),
        )
        .await
        .expect("call succeeds");
    assert!(!result.is_error);
    assert_eq!(text_of(&result), "identified");
}

#[tokio::test]
async fn application_failures_keep_observer_details_out_of_model_text() {
    use pebble_coding_agent::tools::ToolOutputMetadata;
    let tool = RegisteredTool::function("fail", "Fail", json!({"type":"object"}), |_, _| async {
        Err(
            ToolError::execution("Something failed").with_metadata(ToolOutputMetadata {
                details:   Some(json!({"diagnostic":"observer-only"})),
                artifacts: Vec::new(),
            }),
        )
    });
    let events = Arc::new(EventLog::default());
    let record = Arc::clone(&events);
    let runner = ToolRunner::new(
        CodingToolSet::empty().with_tool(tool).expect("tool"),
        mock_environment(),
    )
    .on_event(move |event| record.0.lock().expect("events").push(event));
    let result = runner
        .run(
            &ToolCall::function("failure", "fail", json!({})),
            CancellationToken::new(),
        )
        .await
        .expect("answers");
    assert!(result.is_error);
    assert!(!text_of(&result).contains("observer-only"));
    let CodingEvent::ToolCallCompleted { metadata, .. } = events.completion() else {
        unreachable!()
    };
    assert_eq!(
        metadata.details,
        Some(json!({"diagnostic":"observer-only"}))
    );
}

#[tokio::test]
async fn mixed_content_and_metadata_cannot_bypass_output_budgets() {
    use lithos_llm::types::{ImageContent, MediaSource};
    use pebble_coding_agent::tools::ToolOutput;
    for oversized_details in [false, true] {
        let tool = RegisteredTool::rich_function(
            "rich",
            "Rich",
            json!({"type":"object"}),
            move |_, _| async move {
                Ok(ToolOutput::new(vec![
                ContentPart::Text { text:"a\"\n".repeat(5000) },
                ContentPart::Image(ImageContent::new(MediaSource::url("https://example.test/image.png"))),
            ]).with_details(json!({"value":if oversized_details { "x".repeat(100_000) } else { "small".into() }})))
            },
        );
        let events = Arc::new(EventLog::default());
        let record = Arc::clone(&events);
        let runner = ToolRunner::new(
            CodingToolSet::empty().with_tool(tool).expect("tool"),
            mock_environment(),
        )
        .options(
            CodingAgentOptions::default()
                .with_tool_output_retention_bytes(512)
                .with_tool_output_serialized_bytes(768),
        )
        .on_event(move |event| record.0.lock().expect("events").push(event));
        let result = runner
            .run(
                &ToolCall::function("rich-1", "rich", json!({})),
                CancellationToken::new(),
            )
            .await
            .expect("answers");
        assert_eq!(result.is_error, oversized_details);
        assert!(serde_json::to_vec(&result.content).expect("content").len() <= 512);
        let CodingEvent::ToolCallCompleted {
            metadata, output, ..
        } = events.completion()
        else {
            unreachable!()
        };
        assert!(
            serde_json::to_vec(&(output, metadata))
                .expect("event data")
                .len()
                <= 768
        );
    }
}
