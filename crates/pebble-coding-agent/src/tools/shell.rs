//! Running commands, and the pipeline every shell-shaped tool shares.
//!
//! A profile may give its model a different wire schema — another argument
//! name, a working directory, a different description — but what happens once
//! the command is known must not vary: the same environment variables, the
//! same cancellation, the same retention budget, the same rendered result, and
//! the same process event. That is what [`run_shell_command`] and the helpers
//! under it are for.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde_json::Value;
use tokio::task;
use tracing::{debug, warn};

use crate::config::NativeToolOptions;
use crate::environment::{ExecOutcome, ExecRequest};
use crate::tool::{NativeTool, RegisteredTool, ToolContext, ToolError, required_str};
use crate::truncation::{DEFAULT_TOOL_OUTPUT_RETENTION_BYTES, retain_tool_output};
use crate::types::{CodingEvent, CommandTermination, ToolSource};

/// What a shell failure that never produced a process result is called.
///
/// The model reads it as the head of the message, so it can tell a command
/// that ran and failed from an environment that never ran one.
const SHELL_NO_PROCESS_RESULT: &str = "Shell command produced no process result";

/// Runs a command with pebble's default timeouts.
#[must_use]
pub fn make_shell_tool() -> RegisteredTool {
    make_shell_tool_with_options(&NativeToolOptions::default())
}

/// Runs a command with the timeouts `options` names.
///
/// The model may ask for its own timeout; the tool takes the smaller of that
/// and [`max_command_timeout_ms`](NativeToolOptions::max_command_timeout_ms).
#[must_use]
pub fn make_shell_tool_with_options(options: &NativeToolOptions) -> RegisteredTool {
    let default_timeout = options.default_command_timeout_ms;
    let max_timeout = options.max_command_timeout_ms;
    RegisteredTool {
        definition: ToolDefinition::function(
            NativeTool::Shell.canonical_name(),
            "Execute Bash commands for terminal operations, package managers, tests and builds. \
             Use dedicated tools for file reads, file edits, filename searches, and content \
             searches. Provide timeout_ms for long-running commands.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash source to evaluate, run by a non-login Bash shell"},
                    "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds"},
                    "description": {"type": "string", "description": "Description of what this command does"}
                },
                "required": ["command"]
            }),
        ),
        executor:   Arc::new(move |args, ctx| {
            Box::pin(async move {
                let command = required_str(&args, "command")?;
                let timeout_ms = args
                    .get("timeout_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(default_timeout)
                    .min(max_timeout);

                run_shell_command(&ctx, command, timeout_ms, None).await
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Runs one command with the session's environment variables and
/// cancellation.
///
/// A profile's own shell tool calls this rather than the environment, so it
/// cannot accidentally skip the per-call environment resolution, the
/// cancellation token, or the retention cap.
pub(crate) async fn execute_shell_command(
    ctx: &ToolContext,
    command: &str,
    timeout_ms: u64,
    cwd: Option<&str>,
) -> Result<ExecOutcome, ToolError> {
    let tool_env = ctx.resolve_tool_env().await.map_err(no_process_result)?;
    debug!(
        env_var_count = tool_env.as_ref().map_or(0, HashMap::len),
        "Injecting environment variables into a tool call"
    );
    ctx.env
        .exec(ExecRequest {
            timeout_ms: Some(timeout_ms),
            working_dir: cwd,
            env_vars: tool_env.as_ref(),
            cancel_token: Some(ctx.cancel.clone()),
            output_bytes_cap: Some(DEFAULT_TOOL_OUTPUT_RETENTION_BYTES),
            ..ExecRequest::new(command)
        })
        .await
        .map_err(|error| no_process_result(ToolError::from(error)))
}

/// Runs one command, renders it for the model, and publishes what the process
/// did.
///
/// A process that ended any way but a zero exit is a failed tool call whose
/// message *is* the rendered output: the model needs the same exit code,
/// duration and streams whether the command worked or not, and a failure it
/// cannot read is a failure it cannot act on.
pub(crate) async fn run_shell_command(
    ctx: &ToolContext,
    command: &str,
    timeout_ms: u64,
    cwd: Option<&str>,
) -> Result<String, ToolError> {
    let outcome = execute_shell_command(ctx, command, timeout_ms, cwd).await?;
    let text = retain_shell_output(ctx, &outcome, render_shell_result(&outcome));
    let succeeded = outcome.result.is_success();
    let termination = outcome.result.termination;
    emit_shell_process_completed(ctx, outcome).await;

    if succeeded {
        Ok(text)
    } else {
        Err(shell_failure(termination, text))
    }
}

/// Bounds rendered output to the retention budget and reports what that cost.
///
/// The budget is the crate's default rather than the session's configured one:
/// this is the cap the environment already drained the process against, so the
/// two agree on what a command may produce at all. The session's own budget is
/// applied afterwards, to every tool's output alike, by the execution layer.
///
/// What the environment dropped while draining is counted here too, so the
/// numbers describe everything the command wrote rather than everything that
/// reached this point.
pub(crate) fn retain_shell_output(
    ctx: &ToolContext,
    outcome: &ExecOutcome,
    output: String,
) -> String {
    let retained = retain_tool_output(
        output,
        DEFAULT_TOOL_OUTPUT_RETENTION_BYTES,
        outcome.output_capture().omitted_bytes,
    );
    ctx.record_tool_output_stats(retained.stats);
    retained.output
}

/// Publishes what the process did, after the model-facing output has been
/// rendered.
///
/// Takes the outcome by value so redaction does not have to copy output that
/// can be a megabyte long, and runs the redactor on a blocking thread because
/// an application's redactor scans every byte the process wrote.
pub(crate) async fn emit_shell_process_completed(ctx: &ToolContext, outcome: ExecOutcome) {
    if ctx.coding_event_emitter.is_none() {
        return;
    }

    let exit_code = outcome.result.exit_code;
    let termination = outcome.result.termination;
    let duration_ms = outcome.result.duration_ms;
    let streams_separated = outcome.streams_separated;
    let output_stats = outcome.output_capture();
    let result = outcome.result;
    let redactor = Arc::clone(&ctx.redactor);
    let exec_output_tail =
        match task::spawn_blocking(move || result.default_output_tail(redactor.as_ref())).await {
            Ok(exec_output_tail) => exec_output_tail,
            Err(error) => {
                warn!(?error, "Failed to redact the process output tail");
                None
            }
        };

    ctx.emit_coding_event(CodingEvent::ToolProcessCompleted {
        exit_code,
        termination,
        duration_ms,
        streams_separated,
        exec_output_tail,
        output_bytes_observed: output_stats.observed_bytes,
        output_bytes_retained: output_stats.retained_bytes,
        output_bytes_omitted: output_stats.omitted_bytes,
    });
}

/// The model-facing rendering of one command: how it ended, then what it
/// wrote.
///
/// The metadata stays at the head and standard error at the tail, so a
/// head-and-tail truncation of a noisy command keeps both the exit code and
/// the error that explains it.
fn render_shell_result(outcome: &ExecOutcome) -> String {
    let result = &outcome.result;
    let mut output = format!(
        "Termination: {}\nExit code: {}\nDuration: {}ms\n",
        result.termination.as_str(),
        result
            .exit_code
            .map_or_else(|| "none".to_owned(), |code| code.to_string()),
        result.duration_ms,
    );
    if outcome.streams_separated {
        if !result.stdout.is_empty() {
            let _ = write!(output, "stdout:\n{}\n", result.stdout);
        }
        if !result.stderr.is_empty() {
            let _ = write!(output, "stderr:\n{}\n", result.stderr);
        }
    } else if !result.stdout.is_empty() {
        let _ = write!(output, "output (combined):\n{}\n", result.stdout);
    }
    output
}

/// The failure a process that did not exit zero reports.
///
/// The rendered output is the message; only the category is decided here, so
/// an application watching the call's completion event can tell a command that
/// was stopped from one that failed.
fn shell_failure(termination: CommandTermination, output: String) -> ToolError {
    match termination {
        CommandTermination::Cancelled => ToolError::cancelled(output),
        _ => ToolError::execution(output),
    }
}

/// The same failure, said as one that produced no process result.
fn no_process_result(error: ToolError) -> ToolError {
    let message = format!("{SHELL_NO_PROCESS_RESULT}: {}", error.message());
    error.with_message(message)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::env::current_dir;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::json;
    use tokio::sync::broadcast;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::Result as PumpResult;
    use crate::environment::{ExecResult, LocalEnvironment};
    use crate::event::{Emitter, EventOptions, EventPump, SessionBoundEmitter};
    use crate::redact::Redactor;
    use crate::test_support::MockEnvironment;
    use crate::tool::{CodingEventEmitter, StaticEnvProvider, ToolEnvProvider};
    use crate::tools::testing::{context, context_for, schema_of};
    use crate::truncation::{ToolOutputLimits, truncate_tool_output};
    use crate::types::{CodingAgentEvent, ToolErrorKind};

    /// A session-shaped event pipeline: what the tool publishes through, and
    /// what a test reads it back from.
    struct Events {
        emitter:  Emitter,
        receiver: broadcast::Receiver<CodingAgentEvent>,
        pump:     JoinHandle<PumpResult<()>>,
    }

    impl Events {
        fn new() -> Self {
            let (emitter, pump) = EventPump::new(EventOptions::default());
            let receiver = emitter.subscribe();
            Self {
                emitter,
                receiver,
                pump: tokio::spawn(pump.run()),
            }
        }

        fn bound(&self) -> Arc<dyn CodingEventEmitter> {
            Arc::new(SessionBoundEmitter::new(
                self.emitter.clone(),
                "test-session",
                Some("call_1".to_owned()),
            ))
        }

        /// The next published event, which must be the only one the tool
        /// produced.
        ///
        /// A marker is published after it, so "the tool published nothing
        /// else" is decided by what arrives rather than by how long the test
        /// is willing to wait.
        async fn only_event(&mut self) -> CodingEvent {
            self.emitter
                .emit("test-session".to_owned(), CodingEvent::SessionEnded);
            let event = self.receiver.recv().await.expect("an event is published");
            assert_eq!(event.session_id, "test-session");
            assert_eq!(event.tool_call_id.as_deref(), Some("call_1"));
            assert_eq!(
                self.receiver
                    .recv()
                    .await
                    .expect("the marker is published")
                    .event,
                CodingEvent::SessionEnded,
                "the tool published more than one event"
            );
            event.event
        }

        /// Asserts the tool published nothing at all.
        async fn no_events(&mut self) {
            self.emitter
                .emit("test-session".to_owned(), CodingEvent::SessionEnded);
            assert_eq!(
                self.receiver
                    .recv()
                    .await
                    .expect("the marker is published")
                    .event,
                CodingEvent::SessionEnded
            );
        }

        async fn finish(self) {
            let Self { emitter, pump, .. } = self;
            drop(emitter);
            pump.await
                .expect("the pump task joins")
                .expect("the pump finishes");
        }
    }

    fn environment_with(result: ExecResult) -> MockEnvironment {
        MockEnvironment {
            exec_result: result,
            ..MockEnvironment::default()
        }
    }

    fn exited(stdout: &str, stderr: &str, exit_code: i32, duration_ms: u64) -> ExecResult {
        ExecResult {
            stdout: stdout.to_owned(),
            stderr: stderr.to_owned(),
            exit_code: Some(exit_code),
            termination: CommandTermination::Exited,
            duration_ms,
        }
    }

    /// A model that learned `shell({command, timeout_ms, description})`
    /// expects exactly that, so the wire shape is pinned rather than
    /// described.
    #[test]
    fn the_schema_is_unchanged_and_names_bash() {
        let tool = make_shell_tool();

        assert_eq!(tool.definition.name, "shell");
        assert_eq!(
            *schema_of(&tool),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash source to evaluate, run by a non-login Bash shell"},
                    "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds"},
                    "description": {"type": "string", "description": "Description of what this command does"}
                },
                "required": ["command"]
            })
        );
        assert!(
            tool.definition.description.contains("Bash"),
            "the shell tool should identify its interpreter: {}",
            tool.definition.description
        );
    }

    #[tokio::test]
    async fn a_command_that_exits_zero_reports_its_metadata_and_both_streams() {
        let tool = make_shell_tool();

        let output = (tool.executor)(
            json!({"command": "echo hello"}),
            context(environment_with(exited("hello", "a warning", 0, 10))),
        )
        .await
        .expect("exit 0 is a successful tool result");

        assert_eq!(
            output,
            "Termination: exited\nExit code: 0\nDuration: 10ms\nstdout:\nhello\nstderr:\na \
             warning\n"
        );
    }

    #[tokio::test]
    async fn the_command_reaches_the_environment_without_a_redirection_wrapper() {
        let tool = make_shell_tool();
        let environment = Arc::new(environment_with(exited("", "", 0, 1)));

        let _ = (tool.executor)(
            json!({"command": "make test"}),
            context_for(Arc::clone(&environment)),
        )
        .await;

        assert_eq!(
            *environment
                .captured_command
                .lock()
                .expect("captured_command lock is not poisoned"),
            Some("make test".to_owned())
        );
    }

    #[tokio::test]
    async fn the_models_timeout_reaches_the_environment() {
        let tool = make_shell_tool();
        let environment = Arc::new(MockEnvironment::default());

        let _ = (tool.executor)(
            json!({"command": "sleep 1", "timeout_ms": 5000}),
            context_for(Arc::clone(&environment)),
        )
        .await;

        assert_eq!(
            *environment
                .captured_timeout
                .lock()
                .expect("captured_timeout lock is not poisoned"),
            Some(5000)
        );
    }

    #[tokio::test]
    async fn a_timeout_longer_than_the_profile_allows_is_capped() {
        let options = NativeToolOptions {
            default_command_timeout_ms: 10_000,
            max_command_timeout_ms:     30_000,
        };
        let tool = make_shell_tool_with_options(&options);
        let environment = Arc::new(MockEnvironment::default());

        let _ = (tool.executor)(
            json!({"command": "sleep 100", "timeout_ms": 600_000}),
            context_for(Arc::clone(&environment)),
        )
        .await;

        assert_eq!(
            *environment
                .captured_timeout
                .lock()
                .expect("captured_timeout lock is not poisoned"),
            Some(30_000)
        );
    }

    #[tokio::test]
    async fn a_nonzero_exit_is_a_failed_call_carrying_the_same_output() {
        let tool = make_shell_tool();

        let error = (tool.executor)(
            json!({"command": "false"}),
            context(environment_with(exited("error", "", 1, 10))),
        )
        .await
        .expect_err("a nonzero exit is a failed tool result");

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        let output = error.message();
        assert!(output.contains("Termination: exited"), "got: {output}");
        assert!(output.contains("Exit code: 1"), "got: {output}");
        assert!(output.contains("stdout:\nerror"), "got: {output}");
        assert!(!output.contains("stderr:"), "got: {output}");
    }

    #[tokio::test]
    async fn a_timeout_keeps_the_output_the_command_managed() {
        let tool = make_shell_tool();

        let error = (tool.executor)(
            json!({"command": "sleep 100"}),
            context(environment_with(ExecResult {
                stdout:      "partial".to_owned(),
                stderr:      String::new(),
                exit_code:   None,
                termination: CommandTermination::TimedOut,
                duration_ms: 10_000,
            })),
        )
        .await
        .expect_err("a timeout is a failed tool result");

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        let output = error.message();
        assert!(output.contains("Termination: timed_out"), "got: {output}");
        assert!(output.contains("Exit code: none"), "got: {output}");
        assert!(output.contains("stdout:\npartial"), "got: {output}");
    }

    #[tokio::test]
    async fn a_cancelled_command_keeps_its_output_and_says_it_was_stopped() {
        let tool = make_shell_tool();

        let error = (tool.executor)(
            json!({"command": "sleep 100"}),
            context(environment_with(ExecResult {
                stdout:      "partial".to_owned(),
                stderr:      String::new(),
                exit_code:   None,
                termination: CommandTermination::Cancelled,
                duration_ms: 42,
            })),
        )
        .await
        .expect_err("a cancellation is a failed tool result");

        assert_eq!(error.kind(), ToolErrorKind::Cancelled);
        let output = error.message();
        assert!(output.contains("Termination: cancelled"), "got: {output}");
        assert!(output.contains("Exit code: none"), "got: {output}");
        assert!(output.contains("stdout:\npartial"), "got: {output}");
    }

    #[tokio::test]
    async fn an_environment_that_never_ran_the_command_publishes_no_process_event() {
        let tool = make_shell_tool();
        let mut events = Events::new();
        let environment = MockEnvironment {
            exec_error: Some("sandbox transport is down".to_owned()),
            ..MockEnvironment::default()
        };

        let error = (tool.executor)(
            json!({"command": "make test"}),
            context(environment).with_coding_event_emitter(events.bound()),
        )
        .await
        .expect_err("an environment failure is a failed tool result");

        let output = error.message();
        assert!(
            output.contains("Shell command produced no process result"),
            "got: {output}"
        );
        assert!(
            output.contains("sandbox transport is down"),
            "got: {output}"
        );
        assert!(!output.contains("Exit code"), "got: {output}");
        events.no_events().await;
        events.finish().await;
    }

    #[tokio::test]
    async fn the_process_event_carries_the_outcome_and_a_redacted_tail() {
        struct DropKeys;

        impl Redactor for DropKeys {
            fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
                Cow::Owned(text.replace("AKIAYRWQG5EJLPZLBYNP", "[REDACTED]"))
            }
        }

        let tool = make_shell_tool();
        let mut events = Events::new();
        let environment = environment_with(exited("out", "boom key=AKIAYRWQG5EJLPZLBYNP", 7, 12));

        let _ = (tool.executor)(
            json!({"command": "printf out; printf err >&2; exit 7"}),
            context(environment)
                .with_coding_event_emitter(events.bound())
                .with_redactor(Arc::new(DropKeys)),
        )
        .await;

        match events.only_event().await {
            CodingEvent::ToolProcessCompleted {
                exit_code,
                termination,
                duration_ms,
                streams_separated,
                exec_output_tail,
                output_bytes_observed,
                output_bytes_retained,
                output_bytes_omitted,
            } => {
                assert_eq!(exit_code, Some(7));
                assert_eq!(termination, CommandTermination::Exited);
                assert_eq!(duration_ms, 12);
                assert!(streams_separated);
                assert_eq!(output_bytes_observed, output_bytes_retained);
                assert_eq!(output_bytes_omitted, 0);
                let tail = exec_output_tail.expect("output tail");
                assert_eq!(tail.stdout.as_deref(), Some("out"));
                let stderr = tail.stderr.expect("stderr tail");
                assert_eq!(stderr, "boom key=[REDACTED]");
            }
            other => panic!("expected a process event, got {other:?}"),
        }
        events.finish().await;
    }

    #[tokio::test]
    async fn an_environment_that_merges_the_streams_renders_one_section() {
        let tool = make_shell_tool();
        let mut events = Events::new();
        let environment = MockEnvironment {
            exec_result: exited("interleaved", "", 0, 5),
            streams_separated: false,
            ..MockEnvironment::default()
        };

        let output = (tool.executor)(
            json!({"command": "echo interleaved"}),
            context(environment).with_coding_event_emitter(events.bound()),
        )
        .await
        .expect("exit 0 is a successful tool result");

        assert!(
            output.contains("output (combined):\ninterleaved"),
            "got: {output}"
        );
        assert!(!output.contains("stderr:"), "got: {output}");
        match events.only_event().await {
            CodingEvent::ToolProcessCompleted {
                streams_separated, ..
            } => assert!(!streams_separated),
            other => panic!("expected a process event, got {other:?}"),
        }
        events.finish().await;
    }

    /// The rendering puts the metadata first and standard error last for this
    /// reason: history truncates the middle out of a noisy command.
    #[tokio::test]
    async fn truncating_the_output_keeps_the_exit_metadata_and_the_error_tail() {
        let tool = make_shell_tool();
        let stdout = (0..400)
            .map(|line| format!("{line}: {}", "x".repeat(100)))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(stdout.len() > 30_000);

        let error = (tool.executor)(
            json!({"command": "make build"}),
            context(environment_with(exited(
                &stdout,
                "the build failed",
                2,
                900,
            ))),
        )
        .await
        .expect_err("a nonzero exit is a failed tool result");
        let output = error.message();
        let truncated = truncate_tool_output(output, ToolOutputLimits::defaults_for("shell"));

        assert!(truncated.len() < output.len());
        assert!(truncated.starts_with("Warning: truncated output"));
        assert!(truncated.contains("Termination: exited\nExit code: 2\n"));
        assert!(
            truncated.contains("stderr:\nthe build failed"),
            "the standard error tail did not survive truncation"
        );
    }

    #[tokio::test]
    async fn output_beyond_the_retention_budget_is_reported_as_omitted() {
        let tool = make_shell_tool();
        let mut events = Events::new();
        let bound = SessionBoundEmitter::new(
            events.emitter.clone(),
            "test-session",
            Some("call_1".to_owned()),
        );
        let environment = environment_with(exited(
            &"x".repeat(DEFAULT_TOOL_OUTPUT_RETENTION_BYTES + 4_096),
            "",
            0,
            5,
        ));

        let _ = (tool.executor)(
            json!({"command": "cat big"}),
            context(environment).with_coding_event_emitter(Arc::new(bound.clone())),
        )
        .await;

        let stats = bound
            .take_tool_output_stats()
            .expect("the tool reported its output counts");
        // The emitter this test holds is a sender on the event pipeline, and
        // the pump ends only when every sender is gone.
        drop(bound);
        assert!(stats.omitted_bytes > 0, "{stats:?}");
        assert!(
            stats.retained_bytes <= DEFAULT_TOOL_OUTPUT_RETENTION_BYTES,
            "{stats:?}"
        );
        // The event and the model see the same accounting.
        match events.only_event().await {
            CodingEvent::ToolProcessCompleted {
                output_bytes_omitted,
                ..
            } => assert!(output_bytes_omitted > 0),
            other => panic!("expected a process event, got {other:?}"),
        }
        events.finish().await;
    }

    /// End to end against a real process: the local environment separates the
    /// streams and reports the real exit code, and none of it is laundered
    /// into a successful tool result.
    #[tokio::test]
    async fn a_real_process_reports_its_real_outcome() {
        let tool = make_shell_tool();
        let mut events = Events::new();
        let environment = Arc::new(LocalEnvironment::new(
            current_dir().expect("a current directory"),
        ));

        let error = (tool.executor)(
            json!({"command": "printf 'out'; printf 'err' >&2; exit 7"}),
            context_for(environment).with_coding_event_emitter(events.bound()),
        )
        .await
        .expect_err("exit 7 is a failed tool result");

        let output = error.message();
        assert!(output.contains("Termination: exited"), "got: {output}");
        assert!(output.contains("Exit code: 7"), "got: {output}");
        assert!(output.contains("stdout:\nout"), "got: {output}");
        assert!(output.contains("stderr:\nerr"), "got: {output}");

        match events.only_event().await {
            CodingEvent::ToolProcessCompleted {
                exit_code,
                termination,
                streams_separated,
                exec_output_tail,
                ..
            } => {
                assert_eq!(exit_code, Some(7));
                assert_eq!(termination, CommandTermination::Exited);
                assert!(streams_separated);
                let tail = exec_output_tail.expect("output tail");
                assert_eq!(tail.stdout.as_deref(), Some("out"));
                assert_eq!(tail.stderr.as_deref(), Some("err"));
            }
            other => panic!("expected a process event, got {other:?}"),
        }
        events.finish().await;
    }

    #[tokio::test]
    async fn the_calls_environment_variables_reach_the_command() {
        let tool = make_shell_tool();
        let environment = Arc::new(MockEnvironment::default());
        let tool_env = HashMap::from([("MY_KEY".to_owned(), "my_value".to_owned())]);

        let _ = (tool.executor)(
            json!({"command": "echo $MY_KEY"}),
            context_for(Arc::clone(&environment))
                .with_tool_env_provider(Arc::new(StaticEnvProvider(tool_env.clone()))),
        )
        .await;

        assert_eq!(
            *environment
                .captured_env_vars
                .lock()
                .expect("captured_env_vars lock is not poisoned"),
            Some(tool_env)
        );
    }

    #[tokio::test]
    async fn the_environment_variables_are_resolved_again_for_every_call() {
        struct Sequence {
            values: Mutex<Vec<HashMap<String, String>>>,
        }

        #[async_trait]
        impl ToolEnvProvider for Sequence {
            async fn resolve(&self) -> Result<HashMap<String, String>, ToolError> {
                Ok(self
                    .values
                    .lock()
                    .expect("values lock is not poisoned")
                    .remove(0))
            }
        }

        let tool = make_shell_tool();
        let environment = Arc::new(MockEnvironment::default());
        let provider = Arc::new(Sequence {
            values: Mutex::new(vec![
                HashMap::from([("GITHUB_TOKEN".to_owned(), "t1".to_owned())]),
                HashMap::from([("GITHUB_TOKEN".to_owned(), "t2".to_owned())]),
            ]),
        });

        for expected in ["t1", "t2"] {
            let _ = (tool.executor)(
                json!({"command": "echo $GITHUB_TOKEN"}),
                context_for(Arc::clone(&environment))
                    .with_tool_env_provider(Arc::clone(&provider) as Arc<dyn ToolEnvProvider>),
            )
            .await;

            assert_eq!(
                *environment
                    .captured_env_vars
                    .lock()
                    .expect("captured_env_vars lock is not poisoned"),
                Some(HashMap::from([(
                    "GITHUB_TOKEN".to_owned(),
                    expected.to_owned()
                )]))
            );
        }
    }

    #[tokio::test]
    async fn a_provider_that_cannot_resolve_fails_the_call_before_the_command_runs() {
        struct Failing;

        #[async_trait]
        impl ToolEnvProvider for Failing {
            async fn resolve(&self) -> Result<HashMap<String, String>, ToolError> {
                Err(ToolError::execution("GITHUB_TOKEN refresh failed"))
            }
        }

        let tool = make_shell_tool();
        let environment = Arc::new(MockEnvironment::default());

        let error = (tool.executor)(
            json!({"command": "echo $GITHUB_TOKEN"}),
            context_for(Arc::clone(&environment)).with_tool_env_provider(Arc::new(Failing)),
        )
        .await
        .expect_err("the provider fails");

        assert_eq!(
            error.message(),
            "Shell command produced no process result: GITHUB_TOKEN refresh failed"
        );
        assert!(
            environment
                .captured_command
                .lock()
                .expect("captured_command lock is not poisoned")
                .is_none(),
            "no command should have run"
        );
    }

    #[tokio::test]
    async fn a_call_without_a_provider_adds_no_environment_variables() {
        let tool = make_shell_tool();
        let environment = Arc::new(MockEnvironment::default());

        let _ = (tool.executor)(
            json!({"command": "echo hello"}),
            context_for(Arc::clone(&environment)),
        )
        .await;

        assert_eq!(
            *environment
                .captured_env_vars
                .lock()
                .expect("captured_env_vars lock is not poisoned"),
            None
        );
    }

    #[tokio::test]
    async fn a_command_without_one_is_an_argument_error() {
        let tool = make_shell_tool();

        let error = (tool.executor)(json!({}), context(MockEnvironment::default()))
            .await
            .expect_err("the command is required");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "Missing required parameter: command");
    }
}
