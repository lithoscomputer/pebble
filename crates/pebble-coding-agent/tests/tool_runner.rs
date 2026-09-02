//! Pebble's coding tools, run without a coding agent.
//!
//! A hook or a workflow step sometimes needs one tool call answered the way a
//! session would answer it: the same policy, the same hooks, the same rendering
//! of a failure, the same output budgets, and the same events. This is the
//! supported route for that, exercised from outside the crate.

use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::types::{ContentPart, ToolCall, ToolDefinition, ToolResult};
use pebble_agent::{
    ToolCallNext, ToolCallRequest, ToolDescriptor, ToolMiddleware, ToolOutcome, ToolSystemError,
};
use pebble_coding_agent::CodingAgentOptions;
use pebble_coding_agent::environment::Environment;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use pebble_coding_agent::test_support::MockEnvironment;
use pebble_coding_agent::tools::{
    ApprovalDecision, CodingToolSet, PermissionMiddleware, RegisteredTool, ToolApprovalService,
    ToolError, ToolErrorKind, ToolPermission, ToolPermissionPolicy, ToolRunner, ToolSource,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// A policy that denies exactly one tool.
struct Denying(&'static str);

impl ToolPermissionPolicy for Denying {
    fn permission(&self, tool: &ToolDescriptor) -> ToolPermission {
        if tool.id().as_str() == self.0 {
            ToolPermission::Deny {
                reason: format!("{} denied by tool permission policy", tool.id()),
            }
        } else {
            ToolPermission::Allow
        }
    }
}

struct ApprovalRequired;

impl ToolPermissionPolicy for ApprovalRequired {
    fn permission(&self, _tool: &ToolDescriptor) -> ToolPermission {
        ToolPermission::RequireApproval
    }
}

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

/// Records the order the hooks were called in.
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
impl ToolMiddleware for HookLog {
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
            ToolOutcome::Success(_) => self.push(format!("post {name} {id}")),
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
        .session_id("hook-1")
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
    assert!(events.iter().all(|event| event.session_id == "hook-1"));
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
    let runner = runner.tool_middleware(Arc::new(PermissionMiddleware::new(Arc::new(Denying(
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
    let hooks = Arc::new(HookLog::default());
    let (runner, _log) = runner_with(mock_environment(), CodingAgentOptions::default());
    let runner = runner.tool_middleware(Arc::clone(&hooks) as Arc<dyn ToolMiddleware>);

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
    assert_eq!(hooks.entries(), [
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
        PermissionMiddleware::new(Arc::new(ApprovalRequired))
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
        CodingToolSet::core().with_tool(big),
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
        CodingToolSet::empty().with_tool(cancelled_tool),
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
