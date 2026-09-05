//! Real terminal input, a scripted provider, and durable session recovery.

#![cfg(unix)]

use std::io::{self, Write as _};
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::Duration;
use std::{env, fs};

use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use rustix::process::{Signal, setsid};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};
use twin_openai::config::Config;

#[path = "support/terminal.rs"]
mod terminal;

use terminal::Terminal;

/// Detach from the test runner's controlling terminal before exec. All three
/// standard streams already point at the PTY. Crossterm then uses that input.
#[test]
fn launch_terminal_child() {
    let Ok(args) = env::var("PEBBLE_PTY_ARGS") else {
        return;
    };
    let args: Vec<String> = serde_json::from_str(&args).expect("child arguments");
    setsid().expect("detach from runner terminal");
    writeln!(io::stdout(), "SHELL-HISTORY-MUST-SURVIVE").expect("shell output");
    let error = ProcessCommand::new(env!("CARGO_BIN_EXE_pebble"))
        .args(args)
        .exec();
    panic!("exec Pebble: {error}");
}

async fn provider(scenarios: &str) -> (String, JoinHandle<()>) {
    let mut config = Config::from_lookup(&|_| None).expect("twin config");
    config.scenarios_path = Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(scenarios));
    let app = twin_openai::build_app_with_config(config).expect("twin app");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider");
    let url = format!(
        "http://{}/v1",
        listener.local_addr().expect("provider address")
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve provider");
    });
    (url, task)
}

fn arguments(root: &Path) -> Vec<String> {
    vec![
        "--model".into(),
        "gpt-5.6".into(),
        "--cwd".into(),
        root.display().to_string(),
        "--sessions-dir".into(),
        root.join("sessions").display().to_string(),
    ]
}

fn journal(root: &Path) -> (PathBuf, Vec<CodingAgentEvent>) {
    let session = fs::read_dir(root.join("sessions"))
        .expect("session directory")
        .next()
        .expect("one session")
        .expect("session entry")
        .path();
    let bytes = fs::read_to_string(session.join("events.jsonl")).expect("journal");
    (
        session,
        bytes
            .lines()
            .map(|line| serde_json::from_str(line).expect("event"))
            .collect(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn paste_resize_scrollback_and_resume() {
    let root = tempfile::tempdir().expect("test directory");
    let (url, server) = provider("tests/cmd/scenarios.json").await;
    let args = arguments(root.path());
    let mut terminal = Terminal::start(&args, &url, "exec-answers-without-tools");
    terminal.contains("context ?").await;
    terminal
        .send(b"\x1b[200~say hello\nsecond line\x1b[201~")
        .await;
    terminal.contains("second line").await;
    let (_, events) = journal(root.path());
    assert!(
        !events
            .iter()
            .any(|event| matches!(event.event, CodingEvent::UserInput { .. })),
        "paste must not submit"
    );
    let frames = terminal.screen.frames;
    terminal.resize(60, 12);
    terminal
        .until(|screen| {
            screen.frames > frames
                && screen.live_text().contains("second line")
                && screen.live_text().contains("context ?")
        })
        .await;
    terminal.send(b"\r").await;
    terminal.contains("Hello from the twin.").await;
    terminal.contains("10 in / 5 out").await;
    // Completed output stays in the terminal when a large command response scrolls.
    terminal.send(b"/help\r").await;
    terminal.contains("Ctrl+Z: suspend").await;
    terminal.send(b"/quit\r").await;
    let screen = terminal.finish().await;
    assert!(screen.text().contains("SHELL-HISTORY-MUST-SURVIVE"));
    assert_eq!(screen.text().matches("Hello from the twin.").count(), 1);
    let (session, events) = journal(root.path());
    assert!(events.windows(2).all(|pair| pair[0].seq < pair[1].seq));
    assert!(events.iter().any(|event| matches!(&event.event, CodingEvent::UserInput { text, .. } if text == "say hello\nsecond line")));

    let mut args = args;
    args.extend([
        "--resume".into(),
        session
            .file_name()
            .expect("session id")
            .to_string_lossy()
            .into_owned(),
    ]);
    let mut terminal = Terminal::start(&args, &url, "unused");
    terminal.contains("Hello from the twin.").await;
    terminal.contains("10 in / 5 out").await;
    terminal.send(b"\x1b[A").await;
    terminal
        .until(|screen| screen.live_text().matches("second line").count() >= 2)
        .await;
    terminal.send(b"\x03/quit\r").await;
    terminal.finish().await;
    server.abort();
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn termination_restores_modes_and_saves_the_session() {
    let root = tempfile::tempdir().expect("test directory");
    let mut terminal = Terminal::start(&arguments(root.path()), "http://127.0.0.1:1/v1", "unused");
    terminal.contains("context ?").await;
    terminal.signal(Signal::TERM);
    terminal.finish().await;
    let (session, _) = journal(root.path());
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&fs::read(session.join("checkpoint.json")).expect("checkpoint"))
            .expect("valid checkpoint");
    assert!(checkpoint["record"]["session_id"].is_string());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_preserves_the_draft_and_queued_input() {
    let root = tempfile::tempdir().expect("test directory");
    let (url, server) = provider("tests/tui/scenarios.json").await;
    let mut args = arguments(root.path());
    args.extend(["--permission".into(), "full".into()]);
    let mut terminal = Terminal::start(&args, &url, "tui-cancel");
    terminal.contains("context ?").await;
    terminal.send(b"\x0fslow command\r").await;
    terminal.contains("Running shell_command").await;
    timeout(Duration::from_secs(3), async {
        while !root.path().join("slow-started.txt").exists() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the shell process starts before cancellation");
    terminal.send(b"queued steer\r").await;
    terminal.contains("Queued: 1 steering").await;
    terminal.send(b"\x1b[200~unsent draft\x1b[201~\x1b").await;
    terminal
        .until(|screen| {
            let live = screen.live_text();
            live.contains("unsent draft")
                && live.contains("queued steer")
                && !live.contains("Queued:")
                && !live.contains("Esc to cancel")
        })
        .await;
    terminal.send(b"\x03after cancel\r").await;
    terminal.contains("Ready after cancellation.").await;
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    let (_, events) = journal(root.path());
    assert!(!events.iter().any(|event| matches!(&event.event, CodingEvent::SteeringInjected { text, .. } if text == "queued steer")));
    server.abort();
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn approval_defaults_to_deny_and_can_allow_one_call() {
    let (url, server) = provider("tests/tui/scenarios.json").await;
    let root = tempfile::tempdir().expect("test directory");
    let mut terminal = Terminal::start(&arguments(root.path()), &url, "tui-approval");
    terminal.contains("context ?").await;
    terminal.send(b"request a command\r").await;
    terminal.contains("Approve this tool call?").await;
    assert!(!root.path().join("approval.txt").exists());
    terminal.send(b"\r").await;
    terminal.contains("Permission handled.").await;
    assert!(!root.path().join("approval.txt").exists());
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    server.abort();
    let _ = server.await;

    let (url, server) = provider("tests/tui/scenarios.json").await;
    let mut terminal = Terminal::start(&arguments(root.path()), &url, "tui-approval");
    terminal.contains("context ?").await;
    terminal.send(b"request a command\r").await;
    terminal.contains("Approve this tool call?").await;
    terminal.send(b"y").await;
    terminal.contains("Permission handled.").await;
    assert_eq!(
        fs::read_to_string(root.path().join("approval.txt")).expect("approved tool output"),
        "ran\n"
    );
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    server.abort();
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recovery_keeps_unfinished_output_and_does_not_replay_work() {
    let root = tempfile::tempdir().expect("test directory");
    let args = arguments(root.path());
    let url = "http://127.0.0.1:1/v1";
    let mut terminal = Terminal::start(&args, url, "unused");
    terminal.contains("context ?").await;
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    let (session, events) = journal(root.path());
    let last = events.last().expect("startup events");
    let mut log = fs::OpenOptions::new()
        .append(true)
        .open(session.join("events.jsonl"))
        .expect("append journal");
    let interrupted = CodingAgentEvent::new(
        &last.session_id,
        CodingEvent::TextDelta {
            delta: "Unfinished answer before the crash".into(),
        },
        last.timestamp,
    )
    .with_seq(last.seq + 1);
    writeln!(
        log,
        "{}",
        serde_json::to_string(&interrupted).expect("event JSON")
    )
    .expect("write event");
    write!(log, "{{\"partial\":").expect("write interrupted record");
    drop(log);
    let mut args = args;
    args.extend([
        "--resume".into(),
        session
            .file_name()
            .expect("session id")
            .to_string_lossy()
            .into_owned(),
    ]);
    let mut terminal = Terminal::start(&args, url, "unused");
    terminal
        .contains("Unfinished answer before the crash")
        .await;
    terminal
        .contains("Recovered the last saved checkpoint")
        .await;
    terminal.contains("[Response interrupted]").await;
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    let (_, after) = journal(root.path());
    assert!(after.windows(2).all(|pair| pair[0].seq < pair[1].seq));
    assert!(
        !after
            .iter()
            .any(|event| matches!(event.event, CodingEvent::LlmRequestStarted { .. }))
    );
    assert!(fs::read_dir(session).expect("session files").any(|entry| {
        entry
            .expect("entry")
            .file_name()
            .to_string_lossy()
            .starts_with("events.partial-")
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_editor_handoff_keeps_the_terminal_usable() {
    let root = tempfile::tempdir().expect("test directory");
    fs::write(
        root.path().join("settings.json"),
        serde_json::to_vec(&serde_json::json!({
            "external_editor": "sh -c 'echo edited-draft > \"$1\"' pebble-editor"
        }))
        .expect("settings JSON"),
    )
    .expect("save editor setting");
    let mut terminal = Terminal::start(&arguments(root.path()), "http://127.0.0.1:1/v1", "unused");
    terminal.contains("context ?").await;
    terminal.send(b"original draft\x07").await;
    terminal
        .until(|screen| screen.live_text().contains("edited-draft") && screen.bracketed_paste)
        .await;
    terminal.send(b"\x03/quit\r").await;
    terminal.finish().await;
}

fn without_model(mut args: Vec<String>) -> Vec<String> {
    let index = args
        .iter()
        .position(|arg| arg == "--model")
        .expect("fixture model argument");
    args.drain(index..index + 2);
    args
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_run_saves_a_key_and_model_without_exposing_the_key() {
    let root = tempfile::tempdir().unwrap();
    let (url, server) = provider("tests/cmd/scenarios.json").await;
    let args = without_model(arguments(root.path()));
    let mut terminal = Terminal::start_with_env(&args, &url, None, root.path(), &[]);
    terminal.contains("No configured model is available").await;
    terminal.send(b"openai\r").await;
    terminal.contains("API key for OpenAI (masked)").await;
    terminal
        .send(b"\x1b[200~exec-answers-without-tools\x1b[201~")
        .await;
    terminal
        .until(|screen| screen.live_text().contains("********"))
        .await;
    assert!(!String::from_utf8_lossy(&terminal.output).contains("exec-answers-without-tools"));
    terminal.resize(70, 16);
    terminal.send(b"\r").await;
    terminal.contains("Saved provider credentials").await;
    terminal.send(b"openai/gpt-5.6-sol\r").await;
    terminal.contains("context ?").await;
    terminal.send(b"say hello\r").await;
    terminal.contains("Hello from the twin.").await;
    terminal.contains("10 in / 5 out").await;
    terminal
        .until(|screen| {
            let live = screen.live_text();
            live.contains("10 in / 5 out") && live.lines().any(|line| line.trim() == "Ready")
        })
        .await;
    // Cancelling a later login must not change the key or populate prompt undo.
    terminal.send(b"/login openai\r").await;
    terminal
        .until(|screen| screen.live_text().contains("API key for OpenAI"))
        .await;
    terminal
        .send(b"\x1b[200~CANCELLED-SECRET-MUST-NOT-LEAK\x1b[201~\x1b")
        .await;
    terminal.contains("Login cancelled.").await;
    terminal.send(b"\x1f\x1b[A\x03/export exported.md\r").await;
    terminal.contains("Exported").await;
    terminal.send(b"/quit\r").await;
    terminal.contains("Resume:").await;
    let output = String::from_utf8_lossy(&terminal.output);
    assert!(!output.contains("exec-answers-without-tools"));
    assert!(!output.contains("CANCELLED-SECRET-MUST-NOT-LEAK"));
    terminal.finish().await;
    let settings = fs::read_to_string(root.path().join("settings.json")).unwrap();
    assert!(settings.contains("openai/gpt-5.6-sol"));
    let auth = fs::read_to_string(root.path().join("auth.json")).unwrap();
    assert!(auth.contains("exec-answers-without-tools"));
    assert!(!auth.contains("CANCELLED-SECRET"));
    let (session, _) = journal(root.path());
    for path in [
        session.join("events.jsonl"),
        session.join("checkpoint.json"),
        root.path().join("exported.md"),
        root.path().join("settings.json"),
    ] {
        let content = fs::read_to_string(path).unwrap();
        assert!(!content.contains("exec-answers-without-tools"));
        assert!(!content.contains("CANCELLED-SECRET"));
    }
    server.abort();
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn login_and_logout_report_the_environment_override() {
    let root = tempfile::tempdir().unwrap();
    let mut terminal = Terminal::start(
        &arguments(root.path()),
        "http://127.0.0.1:1/v1",
        "environment-key",
    );
    terminal.contains("context ?").await;
    terminal.send(b"/login openai\r").await;
    terminal.contains("API key for OpenAI").await;
    terminal
        .send(b"\x1b[200~SAVED-SECRET-MUST-NOT-LEAK\x1b[201~\r")
        .await;
    terminal.contains("Active source: environment").await;
    terminal.send(b"/logout openai\r").await;
    terminal
        .contains("Still configured through environment")
        .await;
    terminal.send(b"/quit\r").await;
    terminal.contains("Resume:").await;
    assert!(!String::from_utf8_lossy(&terminal.output).contains("SAVED-SECRET-MUST-NOT-LEAK"));
    terminal.finish().await;
    let auth: serde_json::Value =
        serde_json::from_slice(&fs::read(root.path().join("auth.json")).unwrap()).unwrap();
    assert!(auth["providers"].as_object().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_directory_does_not_move_preferences() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("configuration");
    let mut terminal = Terminal::start_with_env(
        &arguments(root.path()),
        "http://127.0.0.1:1/v1",
        Some("fixture-key"),
        &home,
        &[],
    );
    terminal.contains("context ?").await;
    terminal.send(b"/settings model\r").await;
    terminal.contains("Saved preferences.").await;
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    assert!(home.join("settings.json").exists());
    assert!(!root.path().join("settings.json").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_requests_credentials_for_the_saved_model() {
    let root = tempfile::tempdir().unwrap();
    let url = "http://127.0.0.1:1/v1";
    let mut terminal = Terminal::start(&arguments(root.path()), url, "fixture-key");
    terminal.contains("context ?").await;
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    let (session, _) = journal(root.path());
    fs::write(
        root.path().join("settings.json"),
        r#"{"model":"claude-sonnet-5"}"#,
    )
    .unwrap();
    let mut args = without_model(arguments(root.path()));
    args.extend([
        "--resume".into(),
        session.file_name().unwrap().to_string_lossy().into_owned(),
    ]);
    let mut terminal = Terminal::start_with_env(&args, url, None, root.path(), &[]);
    terminal.contains("Cannot use openai/").await;
    terminal.contains("API key for OpenAI").await;
    terminal
        .send(b"\x1b[200~RESUMED-KEY-MUST-NOT-LEAK\x1b[201~\r")
        .await;
    terminal.contains("context ?").await;
    terminal.send(b"/quit\r").await;
    terminal.contains("Resume:").await;
    assert!(!String::from_utf8_lossy(&terminal.output).contains("RESUMED-KEY-MUST-NOT-LEAK"));
    terminal.finish().await;
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&fs::read(session.join("checkpoint.json")).unwrap()).unwrap();
    assert!(
        checkpoint["metadata"]["model"]
            .as_str()
            .unwrap()
            .starts_with("openai/")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_login_masks_input_and_restores_terminal_modes() {
    let root = tempfile::tempdir().unwrap();
    let args = vec!["auth".into(), "login".into(), "openai".into()];
    let mut terminal =
        Terminal::start_with_env(&args, "http://127.0.0.1:1/v1", None, root.path(), &[]);
    terminal.contains("API key:").await;
    terminal
        .send(b"\x1b[200~CLI-KEY-MUST-NOT-LEAK\x1b[201~")
        .await;
    terminal
        .until(|screen| screen.live_text().contains("*******"))
        .await;
    terminal.send(b"\r").await;
    terminal.contains("Active source:").await;
    assert!(!String::from_utf8_lossy(&terminal.output).contains("CLI-KEY-MUST-NOT-LEAK"));
    terminal.finish_with("Active source:", true).await;
    assert!(
        fs::read_to_string(root.path().join("auth.json"))
            .unwrap()
            .contains("CLI-KEY-MUST-NOT-LEAK")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_signal_cancels_cli_login_and_restores_terminal_modes() {
    let root = tempfile::tempdir().unwrap();
    let args = vec!["auth".into(), "login".into(), "openai".into()];
    let mut terminal =
        Terminal::start_with_env(&args, "http://127.0.0.1:1/v1", None, root.path(), &[]);
    terminal.contains("API key:").await;
    terminal.send(b"partial-key").await;
    terminal
        .until(|screen| screen.live_text().contains("***"))
        .await;
    terminal.signal(Signal::TERM);
    terminal.finish_with("login cancelled", false).await;
    assert!(!root.path().join("auth.json").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_picker_focus_default_and_shortcuts_preserve_the_draft() {
    let root = tempfile::tempdir().unwrap();
    fs::write(
        root.path().join("settings.json"),
        r#"{"model":"openai/gpt-5.6-sol"}"#,
    )
    .unwrap();
    let mut terminal = Terminal::start(&arguments(root.path()), "http://127.0.0.1:1/v1", "unused");
    terminal.contains("Ready").await;
    terminal.send(b"keep my draft\x0c").await;
    terminal
        .until(|screen| screen.cursor_line() == "Search:")
        .await;
    assert!(terminal.screen.live_text().contains("current · default"));
    assert!(terminal.screen.live_text().contains("of "));
    terminal.send(b"terra\x13").await;
    terminal
        .until(|screen| {
            screen.cursor_line() == "› keep my draft"
                && screen.live_text().contains("Ready")
                && screen.live_text().contains("openai/gpt-5.6-terra")
        })
        .await;
    let settings: serde_json::Value =
        serde_json::from_slice(&fs::read(root.path().join("settings.json")).unwrap()).unwrap();
    assert_eq!(settings["model"], "openai/gpt-5.6-terra");
    terminal.send(b"\x10").await;
    terminal
        .until(|screen| {
            screen.cursor_line() == "› keep my draft"
                && screen.live_text().contains("openai/gpt-6-astra · 0 in")
        })
        .await;
    terminal.send(b"\x1b[Z").await;
    terminal
        .until(|screen| {
            screen.cursor_line() == "› keep my draft"
                && screen.live_text().contains("gpt-6-astra · low")
        })
        .await;
    terminal.send(b"\x03/quit\r").await;
    terminal.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_completion_keeps_typing_and_cancellation_in_the_editor() {
    let root = tempfile::tempdir().unwrap();
    assert!(
        ProcessCommand::new("git")
            .args(["init", "--quiet"])
            .arg(root.path())
            .status()
            .unwrap()
            .success()
    );
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(root.path().join("src/main.rs"), "fn main() {}\n").unwrap();
    let mut terminal = Terminal::start(&arguments(root.path()), "http://127.0.0.1:1/v1", "unused");
    terminal.contains("Ready").await;
    terminal.send(b"/hel").await;
    terminal
        .until(|screen| {
            screen.cursor_line() == "› /hel" && screen.live_text().contains("Tab completes")
        })
        .await;
    terminal.send(b"\t").await;
    terminal
        .until(|screen| screen.cursor_line() == "› /help")
        .await;
    terminal.send(b"\x03look at @smr").await;
    terminal
        .until(|screen| {
            screen.cursor_line() == "› look at @smr" && screen.live_text().contains("src/main.rs")
        })
        .await;
    terminal.send(b"\x1b").await;
    terminal
        .until(|screen| {
            screen.cursor_line() == "› look at @smr"
                && !screen.live_text().contains("Tab completes")
        })
        .await;
    terminal.send(b"\t\t").await;
    terminal
        .until(|screen| screen.cursor_line() == "› look at @src/main.rs")
        .await;
    terminal.send(b"\x03/quit\r").await;
    terminal.finish().await;
    let (_, events) = journal(root.path());
    assert!(
        !events
            .iter()
            .any(|event| matches!(event.event, CodingEvent::UserInput { .. }))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn patches_render_colored_diffs_and_export_plain_markdown() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(
        root.path().join("src/lib.rs"),
        "fn hello() {\n    println!(\"old\");\n}\n",
    )
    .unwrap();
    let (url, server) = provider("tests/cmd/scenarios.json").await;
    let mut terminal = Terminal::start_with_env(
        &arguments(root.path()),
        &url,
        Some("exec_patches_a_file_as_gpt56"),
        root.path(),
        &[("PEBBLE_PTY_COLOR", "1")],
    );
    terminal.contains("Ready").await;
    terminal.send(b"patch lib.rs\r").await;
    terminal.contains("Patched lib.rs.").await;
    terminal
        .until(|screen| {
            screen
                .live_text()
                .lines()
                .any(|line| line.trim() == "Ready")
        })
        .await;
    let screen = terminal.screen.text();
    assert!(screen.contains("-    println!(\"old\");"));
    assert!(screen.contains("+    println!(\"new\");"));
    assert!(String::from_utf8_lossy(&terminal.output).contains("\x1b[31m"));
    assert!(String::from_utf8_lossy(&terminal.output).contains("\x1b[32m"));
    terminal.send(b"/tools call_patch\r").await;
    terminal.contains("Tool call call_patch").await;
    terminal.send(b"/export diff.md\r").await;
    terminal.contains("Exported").await;
    terminal.send(b"/quit\r").await;
    terminal.finish().await;
    let markdown = fs::read_to_string(root.path().join("diff.md")).unwrap();
    assert!(markdown.contains("```diff"));
    assert!(markdown.contains("+    println!(\"new\");"));
    assert!(!markdown.contains('\x1b'));
    server.abort();
    let _ = server.await;
}
