//! The coding agent's flow, end to end, without a provider.
//!
//! `examples/coding_agent.rs` shows what embedding pebble looks like: a prompt
//! becomes tool calls that touch a real directory, a prompt is steered while it
//! works, another is interrupted, and the session reports what it used. That
//! example needs credentials and a model that cooperates. These tests drive the
//! same flow against the scripted provider and a `LocalEnvironment` over a
//! temporary directory, so the shape the example demonstrates is checked on
//! every commit.
//!
//! Everything here goes through the public API — the same calls an application
//! makes — and through the harness the catalog chooses for the model, so the
//! tool names are the ones a session really exposes.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{env, fs, process};

use lithos_llm::types::ToolDefinition;
use pebble::advanced::Session;
use pebble::events::{CodingEvent, CodingSessionEvent, ToolSource};
use pebble::resources::Message;
use pebble::test_support::{
    ScriptedCall, ScriptedProvider, client_from, message_text, text_delta_events, text_response,
    tool_call_response, with_cost,
};
use pebble::tools::RegisteredTool;
use pebble::{LocalEnvironment, ShutdownReason};
use serde_json::json;
use tokio::sync::{Notify, broadcast};
use tokio::time::timeout;

/// How long a test waits for a prompt another task has to unblock.
const PATIENCE: Duration = Duration::from_secs(10);

/// The three lines the scripted prompt writes.
const ORIGINAL: &str = "one\ntwo\nthree\n";

/// The same file after the scripted edit.
const EDITED: &str = "one\nedited by pebble\nthree\n";

#[tokio::test]
async fn a_prompt_writes_edits_reads_and_runs_a_command_in_its_workspace() {
    let workspace = Workspace::new("tools");
    let (mut session, provider) = coding_session(&workspace, vec![
        ScriptedCall::response(tool_call_response(
            "write_file",
            "call_write",
            json!({
                "file_path": "greeting.txt",
                "content":   ORIGINAL,
            }),
        )),
        ScriptedCall::response(tool_call_response(
            "edit_file",
            "call_edit",
            json!({
                "file_path":  "greeting.txt",
                "old_string": "two",
                "new_string": "edited by pebble",
            }),
        )),
        ScriptedCall::response(tool_call_response(
            "read_file",
            "call_read",
            json!({
                "file_path": "greeting.txt",
            }),
        )),
        ScriptedCall::response(tool_call_response(
            "shell",
            "call_shell",
            json!({
                "command": "wc -l < greeting.txt",
            }),
        )),
        ScriptedCall::response(text_response("greeting.txt has three lines")),
    ]);
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");
    let answer = timeout(PATIENCE, session.prompt("set up greeting.txt"))
        .await
        .expect("the prompt finishes")
        .expect("the prompt succeeds");

    assert_eq!(answer.as_deref(), Some("greeting.txt has three lines"));
    assert_eq!(
        fs::read_to_string(workspace.join("greeting.txt")).expect("the file was written"),
        EDITED,
        "the model's edit reached the workspace on disk"
    );
    assert_eq!(
        provider.call_count(),
        5,
        "one call per tool round, and one to answer"
    );

    let published = settled(&mut session, &mut events).await;
    assert_eq!(tools_called(&published), [
        "write_file",
        "edit_file",
        "read_file",
        "shell"
    ]);
    assert!(
        !published
            .iter()
            .any(|event| matches!(event, CodingEvent::ToolCallCompleted { is_error: true, .. })),
        "no tool failed: {published:?}"
    );
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::ToolProcessCompleted {
                exit_code: Some(0),
                ..
            })),
        "the command's own exit is reported beside the tool result"
    );
}

#[tokio::test]
async fn every_tool_call_is_answered_before_the_next_round() {
    // The history invariant an application inherits: whatever happens, each
    // round's tool calls are paired with results, because a provider refuses a
    // conversation where one is missing.
    let workspace = Workspace::new("pairing");
    let (mut session, _provider) = coding_session(&workspace, vec![
        ScriptedCall::response(tool_call_response(
            "write_file",
            "call_write",
            json!({
                "file_path": "greeting.txt",
                "content":   ORIGINAL,
            }),
        )),
        ScriptedCall::response(tool_call_response(
            "shell",
            "call_shell",
            json!({
                "command": "cat greeting.txt",
            }),
        )),
        ScriptedCall::response(text_response("done")),
    ]);

    session.initialize().await.expect("initialization succeeds");
    timeout(PATIENCE, session.prompt("write and read it back"))
        .await
        .expect("the prompt finishes")
        .expect("the prompt succeeds");

    let mut asked = 0;
    let mut answered = 0;
    for turn in session.history().turns() {
        match turn {
            Message::Assistant { tool_calls, .. } => asked += tool_calls.len(),
            Message::ToolResults { results, .. } => answered += results.len(),
            _ => {}
        }
    }
    assert_eq!(asked, 2);
    assert_eq!(answered, asked, "every call the model made has a result");
}

#[tokio::test]
async fn a_steer_sent_while_a_tool_runs_arrives_as_the_next_turn() {
    // The example steers from a task watching the event stream. A test cannot
    // rely on that task winning the race to the round boundary, so the steer is
    // sent while a tool call is still open: the round cannot end until the tool
    // answers, which makes the ordering the assertion depends on certain.
    let workspace = Workspace::new("steering");
    let reached = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let (mut session, provider) = session_with(
        &workspace,
        vec![
            ScriptedCall::response(tool_call_response("checkpoint", "call_1", json!({}))),
            ScriptedCall::response(text_response("noted, and done")),
        ],
        vec![checkpoint_tool(Arc::clone(&reached), Arc::clone(&release))],
    );
    let mut events = session.subscribe();
    let control = session.control_handle();

    let steering = tokio::spawn(async move {
        reached.notified().await;
        control.steer("also write notes.md", None);
        release.notify_one();
    });

    session.initialize().await.expect("initialization succeeds");
    let answer = timeout(PATIENCE, session.prompt("start the job"))
        .await
        .expect("the prompt finishes")
        .expect("the prompt succeeds");
    steering.await.expect("the steering task finishes");

    assert_eq!(answer.as_deref(), Some("noted, and done"));
    assert!(
        session.history().turns().iter().any(|turn| matches!(
            turn,
            Message::Steering { content, .. } if content == "also write notes.md"
        )),
        "the steer landed as its own turn: {:?}",
        session.history().turns()
    );
    let asked_again = provider
        .requests()
        .last()
        .expect("the steer sent the round around again")
        .messages()
        .iter()
        .any(|message| message_text(message) == "also write notes.md");
    assert!(asked_again, "the model was told what the steer said");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::SteeringInjected { text, .. } if text == "also write notes.md"
        )),
        "an application watching the stream sees the steer too"
    );
}

#[tokio::test]
async fn an_interrupt_abandons_the_round_and_a_steer_resumes_it() {
    // The second half of the example: the model is part-way through an answer,
    // the operator stops it, and what they say next replaces the abandoned
    // round. The first scripted call never ends, so the interrupt is what moves
    // the prompt on.
    let workspace = Workspace::new("interrupt");
    let (mut session, provider) = coding_session(&workspace, vec![
        ScriptedCall::EventsThenPending(text_delta_events("a long description of")),
        ScriptedCall::response(text_response("DONE")),
    ]);
    let mut events = session.subscribe();
    let mut watched = session.subscribe();
    let control = session.control_handle();

    let controller = tokio::spawn(async move {
        wait_for(&mut watched, |event| {
            matches!(event, CodingEvent::TextDelta { .. })
        })
        .await;
        control.interrupt();
        // Exactly one of these is published per gesture, so waiting for it is
        // waiting for the interrupt to have settled.
        wait_for(&mut watched, |event| {
            matches!(event, CodingEvent::RoundInterrupted { .. })
        })
        .await;
        assert!(control.is_waiting_for_steer(), "a bare interrupt parks");
        control.steer("never mind — just say DONE", None);
    });

    session.initialize().await.expect("initialization succeeds");
    let answer = timeout(PATIENCE, session.prompt("describe every file"))
        .await
        .expect("the interrupt unblocks the hanging stream")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    assert_eq!(answer.as_deref(), Some("DONE"));
    assert_eq!(provider.call_count(), 2, "the round was asked again");
    assert!(
        matches!(
            session.history().turns(),
            [Message::User { .. }, Message::Steering { .. }, Message::Assistant { content, .. }]
                if content == "DONE"
        ),
        "the abandoned turn is not committed: {:?}",
        session.history().turns()
    );

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        published
            .iter()
            .filter(|event| matches!(event, CodingEvent::RoundInterrupted { .. }))
            .count(),
        1,
        "one gesture, one announcement"
    );
    let withdrawn = position(&published, |event| {
        matches!(event, CodingEvent::AssistantOutputReplace { .. })
    })
    .expect("the abandoned output was withdrawn");
    let answered = position(&published, |event| {
        matches!(event, CodingEvent::AssistantMessage { .. })
    })
    .expect("the replacement round answered");
    assert!(
        withdrawn < answered,
        "nothing is shown twice: the withdrawal precedes the answer that replaces it"
    );
}

#[tokio::test]
async fn a_finished_prompt_reports_what_it_used() {
    // What the example prints in its summary comes from these three, so they
    // are pinned here against the scripted response's own numbers.
    let workspace = Workspace::new("usage");
    let (mut session, _provider) = coding_session(&workspace, vec![ScriptedCall::response(
        with_cost(text_response("done"), 12_500),
    )]);

    session.initialize().await.expect("initialization succeeds");
    timeout(PATIENCE, session.prompt("say something"))
        .await
        .expect("the prompt finishes")
        .expect("the prompt succeeds");

    let usage = session.last_prompt_usage();
    assert_eq!(usage.input, 10);
    assert_eq!(usage.output, 5);
    assert_eq!(usage.total(), 15);
    assert_eq!(session.last_prompt_cost_usd_micros(), Some(12_500));
    assert_eq!(
        session.history().turns().len(),
        2,
        "the input and the answer"
    );
}

// --- The harness these tests are assembled from ---

/// A session over a real directory, answering from `calls`.
///
/// The model is the test catalog's `test/model`, whose entry names the
/// Anthropic harness, so the tools are registered under pebble's own names —
/// the names a session would really expose for that model.
fn coding_session(
    workspace: &Workspace,
    calls: Vec<ScriptedCall>,
) -> (Session, Arc<ScriptedProvider>) {
    session_with(workspace, calls, Vec::new())
}

/// The same, with extra tools registered on top of the harness's own.
fn session_with(
    workspace: &Workspace,
    calls: Vec<ScriptedCall>,
    tools: Vec<RegisteredTool>,
) -> (Session, Arc<ScriptedProvider>) {
    let (client, provider) = client_from(ScriptedProvider::new(calls));
    let session = Session::builder(client)
        .model("test/model")
        .environment(Arc::new(LocalEnvironment::new(workspace.path())))
        .tools(tools)
        .build()
        .expect("the session builds");
    (session, provider)
}

/// A tool that stops inside a round until the test lets it go.
fn checkpoint_tool(reached: Arc<Notify>, release: Arc<Notify>) -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "checkpoint",
            "Waits for the test",
            json!({"type": "object"}),
        ),
        executor:   Arc::new(move |_arguments, _context| {
            let reached = Arc::clone(&reached);
            let release = Arc::clone(&release);
            Box::pin(async move {
                reached.notify_one();
                release.notified().await;
                Ok("ready".to_owned())
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Closes the session and reports every event it published.
///
/// Closing first is what makes the list complete: events are published by a
/// task the session owns, and that task stops only when the session tells it
/// to.
async fn settled(
    session: &mut Session,
    events: &mut broadcast::Receiver<CodingSessionEvent>,
) -> Vec<CodingEvent> {
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
    let mut published = Vec::new();
    while let Ok(event) = events.try_recv() {
        published.push(event.event);
    }
    published
}

/// Waits for the first event `wanted` matches.
async fn wait_for(
    events: &mut broadcast::Receiver<CodingSessionEvent>,
    wanted: impl Fn(&CodingEvent) -> bool,
) {
    loop {
        let event = events.recv().await.expect("the event stream stays open");
        if wanted(&event.event) {
            return;
        }
    }
}

/// The tools the session finished calling, in order.
fn tools_called(published: &[CodingEvent]) -> Vec<String> {
    published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::ToolCallCompleted { tool_name, .. } => Some(tool_name.clone()),
            _ => None,
        })
        .collect()
}

/// Where `wanted` first matched, for an assertion about ordering.
fn position(published: &[CodingEvent], wanted: impl Fn(&CodingEvent) -> bool) -> Option<usize> {
    published.iter().position(wanted)
}

/// A directory of its own for one test, removed when the test ends.
struct Workspace {
    path: PathBuf,
}

impl Workspace {
    /// Creates the directory this test works in.
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let path = env::temp_dir().join(format!("pebble-e2e-{name}-{}-{unique}", process::id()));
        fs::create_dir_all(&path).expect("the workspace directory is created");
        Self { path }
    }

    /// Where the session works.
    fn path(&self) -> &Path {
        &self.path
    }

    /// One path inside the workspace.
    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        // A failed cleanup leaves a directory behind in the system temporary
        // directory, which is not worth failing a test over.
        drop(fs::remove_dir_all(&self.path));
    }
}
