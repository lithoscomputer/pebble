//! What the session asks the model for.
//!
//! One request per round, built from the system prompt, the conversation, and
//! the tools the access policy leaves exposed. These tests read the requests
//! the scripted provider captured, which is the only place the session's own
//! decisions about a round are visible.

use lithos_llm::types::{Message as LlmMessage, ReasoningEffort, Request, Role};
use serde_json::{Value, json};

use super::super::testing::wait_for_event;
use super::*;
use crate::config::{ToolAccess, ToolAccessPolicy, ToolApprovalAdapter, ToolExposureMode};
use crate::task_reminder::TASK_REMINDER_TEXT;
use crate::test_support::message_text;

/// A policy that answers from a table and denies whatever it does not know.
struct NamedToolAccessPolicy {
    decisions: Vec<(&'static str, ToolAccess)>,
}

impl NamedToolAccessPolicy {
    /// The policy, ready to install on [`SessionOptions`].
    fn installed(decisions: Vec<(&'static str, ToolAccess)>) -> Arc<dyn ToolAccessPolicy> {
        Arc::new(Self { decisions })
    }
}

impl ToolAccessPolicy for NamedToolAccessPolicy {
    fn access_for_tool(&self, tool_name: &str) -> ToolAccess {
        self.decisions
            .iter()
            .find_map(|(name, access)| (*name == tool_name).then_some(*access))
            .unwrap_or(ToolAccess::Denied)
    }
}

/// The names of the tools the request exposed.
fn tool_names(request: &Request) -> Vec<&str> {
    request
        .tools()
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[tokio::test]
async fn the_system_prompt_carries_the_application_instructions() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .options(SessionOptions {
            user_instructions: Some("Always use TDD".to_owned()),
            ..SessionOptions::default()
        })
        .build();
    session.initialize().await.expect("initialization succeeds");

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    let system = request
        .messages()
        .first()
        .expect("the request is not empty");
    assert_eq!(system.role(), Role::System);
    assert!(
        message_text(system).contains("Always use TDD"),
        "the instructions reach the model: {}",
        message_text(system)
    );
}

#[tokio::test]
async fn a_session_with_no_prompt_sends_no_system_message() {
    // Deliberately not initialized, so the system prompt is still empty.
    let (mut session, provider) = TestSession::answering(answers("captured"));

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    assert!(
        request
            .messages()
            .iter()
            .all(|message| message.role() != Role::System),
        "an empty prompt is left out rather than sent empty"
    );
    assert_eq!(
        request.messages().first().map(LlmMessage::role),
        Some(Role::User),
        "the input is the first thing the model reads"
    );
}

#[tokio::test]
async fn every_registered_tool_is_exposed_when_no_policy_says_otherwise() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .tools([noop_tool("read_file"), noop_tool("write_file")])
        .build();

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    let mut names = tool_names(request);
    names.sort_unstable();
    assert_eq!(names, ["read_file", "write_file"]);
}

#[tokio::test]
async fn a_denied_tool_is_never_advertised() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .tools([noop_tool("read_file"), noop_tool("write_file")])
        .options(SessionOptions {
            tool_access_policy: Some(NamedToolAccessPolicy::installed(vec![
                ("read_file", ToolAccess::Allowed),
                ("write_file", ToolAccess::Denied),
            ])),
            tool_exposure_mode: ToolExposureMode::IncludeRequiresApproval,
            ..SessionOptions::default()
        })
        .build();

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    assert_eq!(tool_names(request), ["read_file"]);
}

#[tokio::test]
async fn an_approval_required_tool_is_advertised_where_the_mode_allows_it() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .tools([noop_tool("read_file"), noop_tool("shell")])
        .options(SessionOptions {
            tool_access_policy: Some(NamedToolAccessPolicy::installed(vec![
                ("read_file", ToolAccess::Allowed),
                ("shell", ToolAccess::RequiresApproval),
            ])),
            tool_exposure_mode: ToolExposureMode::IncludeRequiresApproval,
            ..SessionOptions::default()
        })
        .build();

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    let mut names = tool_names(request);
    names.sort_unstable();
    assert_eq!(names, ["read_file", "shell"]);
}

#[tokio::test]
async fn the_tools_a_session_reports_are_the_tools_it_sends() {
    let (session, _provider) = TestSession::new(answers("captured"))
        .tools([
            noop_tool("read_file"),
            noop_tool("apply_patch"),
            noop_tool("shell"),
        ])
        .options(SessionOptions {
            tool_access_policy: Some(NamedToolAccessPolicy::installed(vec![
                ("read_file", ToolAccess::Allowed),
                ("apply_patch", ToolAccess::RequiresApproval),
                ("shell", ToolAccess::Denied),
            ])),
            tool_exposure_mode: ToolExposureMode::IncludeRequiresApproval,
            ..SessionOptions::default()
        })
        .build();

    let tools = session.effective_tools();

    let mut names: Vec<&str> = tools
        .iter()
        .map(|tool| tool.definition.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["apply_patch", "read_file"]);
    assert!(tools.iter().all(|tool| tool.source == ToolSource::Native));
}

#[tokio::test]
async fn ten_unused_assistant_turns_bring_back_the_task_reminder() {
    let (mut session, provider) = TestSession::new(answers("done"))
        .tools([noop_tool("TaskCreate"), noop_tool("TaskUpdate")])
        .build();

    for index in 0..=10 {
        session
            .prompt(&format!("turn {index}"))
            .await
            .expect("the prompt succeeds");
    }

    let requests = provider.requests();
    let last = requests.last().expect("the last round was requested");
    assert!(
        last.messages().iter().any(|message| {
            message.role() == Role::System && message_text(message) == TASK_REMINDER_TEXT
        }),
        "the eleventh round reminds the model of its task tools"
    );
}

#[tokio::test]
async fn the_reasoning_effort_a_session_is_given_reaches_the_request() {
    // A model whose catalog row says it can be asked to think harder.
    let (mut session, provider) = TestSession::new(answers("captured"))
        .model("test/thinking")
        .build();
    session.set_reasoning_effort(Some(ReasoningEffort::High));

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    assert_eq!(request.reasoning_effort(), Some(ReasoningEffort::High));
}

// --- Approving a call ---

/// The tool round every approval test runs.
fn approval_calls() -> Vec<ScriptedCall> {
    vec![
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_1",
            json!({"text": "hello"}),
        )),
        ScriptedCall::response(text_response("Done")),
    ]
}

#[tokio::test]
async fn a_refused_call_answers_the_model_with_the_reason() {
    let (mut session, _provider) = TestSession::new(approval_calls())
        .tools([echo_tool()])
        .options(SessionOptions {
            tool_hooks: Some(Arc::new(ToolApprovalAdapter(Arc::new(
                |_name, _arguments| Err("denied by policy".to_owned()),
            )))),
            ..SessionOptions::default()
        })
        .build();
    let mut events = session.subscribe();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    assert_eq!(session.state(), SessionState::Idle);
    assert_eq!(session.history().turns().len(), 4);
    let results = tool_results(&session, 2);
    assert!(results[0].is_error);
    assert!(
        result_text(&results[0]).contains("denied by policy"),
        "the model reads why the call was refused"
    );

    let published = settled(&mut session, &mut events).await;
    let completions: Vec<&AgentEvent> = published
        .iter()
        .filter(|event| matches!(event, AgentEvent::ToolCallCompleted { .. }))
        .collect();
    assert_eq!(completions.len(), 1);
    assert!(matches!(completions[0], AgentEvent::ToolCallCompleted {
        is_error: true,
        ..
    }));
}

#[tokio::test]
async fn an_approved_call_runs() {
    let (mut session, _provider) = TestSession::new(approval_calls())
        .tools([echo_tool()])
        .options(SessionOptions {
            tool_hooks: Some(Arc::new(ToolApprovalAdapter(Arc::new(
                |_name, _arguments| Ok(()),
            )))),
            ..SessionOptions::default()
        })
        .build();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert!(!results[0].is_error);
    assert_eq!(result_text(&results[0]), "echo: hello");
}

#[tokio::test]
async fn the_approval_hook_sees_the_call_the_model_asked_for() {
    let captured: Arc<Mutex<Option<(String, Value)>>> = Arc::new(Mutex::new(None));
    let recorder = Arc::clone(&captured);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_1",
            json!({"text": "world"}),
        )),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([echo_tool()])
    .options(SessionOptions {
        tool_hooks: Some(Arc::new(ToolApprovalAdapter(Arc::new(
            move |name, arguments| {
                *recorder.lock().unwrap_or_else(PoisonError::into_inner) =
                    Some((name.to_owned(), arguments.clone()));
                Ok(())
            },
        )))),
        ..SessionOptions::default()
    })
    .build();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let seen = captured.lock().unwrap_or_else(PoisonError::into_inner);
    let (name, arguments) = seen.as_ref().expect("the hook was called");
    assert_eq!(name, "echo");
    assert_eq!(arguments, &json!({"text": "world"}));
}

#[tokio::test]
async fn a_session_with_no_hook_runs_the_call_unchecked() {
    let (mut session, _provider) = TestSession::new(approval_calls())
        .tools([echo_tool()])
        .options(SessionOptions {
            tool_hooks: None,
            ..SessionOptions::default()
        })
        .build();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert!(!results[0].is_error);
    assert_eq!(result_text(&results[0]), "echo: hello");
}

// --- Watching a round from outside ---

#[tokio::test]
async fn a_subscriber_sees_the_round_while_it_runs() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("echo", "call_1", json!({"text": "hi"}))),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([echo_tool()])
    .build();
    let mut events = session.subscribe();
    let watcher = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, AgentEvent::ToolCallStarted { tool_name, .. } if tool_name == "echo")
        })
        .await;
        wait_for_event(&mut events, |event| {
            matches!(event, AgentEvent::ToolCallCompleted { .. })
        })
        .await;
    });

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    watcher.await.expect("the watcher saw the whole call");
}
