//! Pebble's coding tools, run without a coding agent.
//!
//! A hook or a workflow step sometimes needs one tool call answered the way a
//! session would answer it: the same policy, the same hooks, the same rendering
//! of a failure, the same output budgets, and the same events. This is the
//! supported route for that, exercised from outside the crate.

use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::types::{ContentPart, ToolCall, ToolDefinition, ToolResult};
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use pebble_coding_agent::test_support::MockEnvironment;
use pebble_coding_agent::tools::{
    CodingToolSet, RegisteredTool, ToolAccess, ToolAccessPolicy, ToolApprovalAdapter, ToolError,
    ToolErrorKind, ToolExposureMode, ToolHookCallback, ToolHookDecision, ToolRunner, ToolSource,
};
use pebble_coding_agent::{CodingAgentOptions, Environment};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

/// A policy that denies exactly one tool.
struct Denying(&'static str);

impl ToolAccessPolicy for Denying {
    fn access_for_tool(&self, tool_name: &str) -> ToolAccess {
        if tool_name == self.0 {
            ToolAccess::Denied
        } else {
            ToolAccess::Allowed
        }
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
        .await;
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
        .await;
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
    let (runner, log) = runner_with(mock_environment(), CodingAgentOptions {
        tool_access_policy: Some(Arc::new(Denying("shell"))),
        tool_exposure_mode: ToolExposureMode::IncludeRequiresApproval,
        ..CodingAgentOptions::default()
    });

    let result = runner
        .run(
            &ToolCall::function("call_shell", "shell", json!({"command": "rm -rf /"})),
            CancellationToken::new(),
        )
        .await;

    assert!(result.is_error);
    assert_eq!(text_of(&result), "shell tool denied by tool access policy");
    assert!(matches!(log.completion(), CodingEvent::ToolCallCompleted {
        is_error: true,
        error_kind: Some(ToolErrorKind::Denied),
        ..
    }));
}

#[tokio::test]
async fn a_runner_calls_the_hooks_a_session_would() {
    let hooks = Arc::new(HookLog::default());
    let (runner, _log) = runner_with(mock_environment(), CodingAgentOptions {
        tool_hooks: Some(Arc::clone(&hooks) as Arc<dyn ToolHookCallback>),
        ..CodingAgentOptions::default()
    });

    let ok = runner
        .run(
            &ToolCall::function(
                "call_read",
                "read_file",
                json!({"file_path": "/work/notes.md"}),
            ),
            CancellationToken::new(),
        )
        .await;
    let missing = runner
        .run(
            &ToolCall::function(
                "call_miss",
                "read_file",
                json!({"file_path": "/work/absent.md"}),
            ),
            CancellationToken::new(),
        )
        .await;

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
async fn a_hook_that_blocks_a_call_answers_it_with_the_reason() {
    let (runner, _log) = runner_with(mock_environment(), CodingAgentOptions {
        tool_hooks: Some(Arc::new(ToolApprovalAdapter(Arc::new(
            |name, _arguments| Err(format!("{name} is not allowed from a hook")),
        )))),
        ..CodingAgentOptions::default()
    });

    let result = runner
        .run(
            &ToolCall::function(
                "call_read",
                "read_file",
                json!({"file_path": "/work/notes.md"}),
            ),
            CancellationToken::new(),
        )
        .await;

    assert!(result.is_error);
    assert_eq!(text_of(&result), "read_file is not allowed from a hook");
}

#[tokio::test]
async fn a_runner_bounds_output_like_a_session() {
    let log = Arc::new(EventLog::default());
    let recorder = Arc::clone(&log);
    let big = RegisteredTool {
        definition: ToolDefinition::function("big", "Answers with a lot", json!({})),
        executor:   Arc::new(|_arguments, _context| Box::pin(async { Ok("x".repeat(80_000)) })),
        source:     ToolSource::Application,
    };
    let runner = ToolRunner::new(
        CodingToolSet::core().with_tool(big),
        mock_environment() as Arc<dyn Environment>,
    )
    .options(CodingAgentOptions {
        tool_output_retention_bytes: 4_096,
        ..CodingAgentOptions::default()
    })
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
        .await;

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
    let cancelled_tool = RegisteredTool {
        definition: ToolDefinition::function("waits", "Waits until cancelled", json!({})),
        executor:   Arc::new(|_arguments, context| {
            Box::pin(async move {
                context.cancel.cancelled().await;
                Err(ToolError::cancelled("Cancelled"))
            })
        }),
        source:     ToolSource::Application,
    };
    let runner = ToolRunner::new(
        CodingToolSet::empty().with_tool(cancelled_tool),
        mock_environment() as Arc<dyn Environment>,
    );

    let unknown = runner
        .run(
            &ToolCall::function("call_nope", "nope", json!({})),
            CancellationToken::new(),
        )
        .await;
    let cancel = CancellationToken::new();
    cancel.cancel();
    let cancelled = runner
        .run(&ToolCall::function("call_wait", "waits", json!({})), cancel)
        .await;

    assert!(unknown.is_error);
    assert_eq!(text_of(&unknown), "Unknown tool: nope");
    assert!(cancelled.is_error);
    assert_eq!(text_of(&cancelled), "Cancelled");
}
