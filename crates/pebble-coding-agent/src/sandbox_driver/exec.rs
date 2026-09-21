//! The command execution policy over the sandbox-driver [`Exec`] facet.
//!
//! The vocabulary is the driver's own: an [`ExecSpec`] and [`ExecControls`]
//! go in, an [`ExecResult`] or [`ExecStreamingResult`] comes out. This
//! module adds the policy on the way in and pebble's reading of a result
//! on the way out.
//!
//! A command runs as Bash source under `bash -c` with `BASH_ENV` blanked by
//! the driver whatever the caller passed, and ends in one of three ways:
//!
//! - **timeout**: the spec's timeout fires and the provider runs the stop
//!   ladder the policy asks for: `TERM`, then `KILL` after
//!   [`SandboxExec::stop_grace`]. The result reports [`Termination::TimedOut`].
//! - **cancellation**: the caller's [`CancellationToken`] is the `term` stop;
//!   the provider escalates to `KILL` after the same grace. The result reports
//!   [`Termination::Cancelled`].
//! - **exit**: the process ended on its own.
//!
//! Output is drained regardless of the retention cap and delivered live
//! through the caller's [`sandbox_driver::OutputSink`]. Pebble reads command
//! output as text, so the policy asks the driver for
//! [`OutputSanitization::StripAll`]: terminal escape sequences and stray
//! control characters never reach a result, a sink chunk, or a tail. Secret
//! redaction is the application's ([`Redactor`]) and happens only when a
//! tail is rendered for events or logs ([`redacted_output_tail`]). The
//! explicit environment reaches the provider as the caller composed it: the
//! driver filters credential-shaped names out of the *inherited* host
//! environment itself and treats the spec's own variables as the deliberate
//! channel for secrets, so the policy adds no filter of its own.

use std::collections::HashMap;
use std::time::Duration;

use sandbox_driver::{
    Exec, ExecControls, ExecResult, ExecSpec, ExecStreamingResult, OutputSanitization, Termination,
};
use tokio_util::sync::CancellationToken;

use crate::char_boundary::floor_char_boundary;
use crate::redact::Redactor;
use crate::types::{CommandTermination, ExecOutputTail};

/// Time between `TERM` and `KILL` when the policy stops a command.
pub const DEFAULT_STOP_GRACE: Duration = Duration::from_secs(2);

/// Retention when a caller sets no cap: enough for any build log an
/// application renders, bounded so a runaway command cannot exhaust memory.
pub const DEFAULT_RETAINED_OUTPUT_BYTES: usize = sandbox_driver::DEFAULT_BUFFER_BYTES;

/// The exec policy bound to one driver [`Exec`] facet.
pub struct SandboxExec<'a> {
    exec:        &'a dyn Exec,
    stop_grace:  Duration,
    /// Where a command runs when the caller names no directory. `None`
    /// leaves the choice to the provider's own working directory.
    working_dir: Option<String>,
}

impl<'a> SandboxExec<'a> {
    #[must_use]
    pub fn new(exec: &'a dyn Exec) -> Self {
        Self {
            exec,
            stop_grace: DEFAULT_STOP_GRACE,
            working_dir: None,
        }
    }

    /// The directory commands run in when the caller names none. The
    /// session's working directory can sit below the provider's, so it is
    /// passed explicitly.
    #[must_use]
    pub fn with_working_dir(mut self, working_dir: impl Into<String>) -> Self {
        self.working_dir = Some(working_dir.into());
        self
    }

    /// Time between `TERM` and `KILL` when a command is stopped; the
    /// provider runs the ladder.
    #[must_use]
    pub fn with_stop_grace(mut self, stop_grace: Duration) -> Self {
        self.stop_grace = stop_grace;
        self
    }

    #[must_use]
    pub fn stop_grace(&self) -> Duration {
        self.stop_grace
    }

    /// Runs Bash source to completion and returns its captured output.
    ///
    /// Equivalent to `bash -c <command>` with a clean, non-login shell: no
    /// `errexit`, no `pipefail`, `BASH_ENV` blanked. A caller that wants
    /// different semantics writes them into the command. `None` for
    /// `timeout` runs without a deadline.
    pub async fn run(
        &self,
        command: &str,
        timeout: Option<Duration>,
        working_dir: Option<&str>,
        env_vars: Option<&HashMap<String, String>>,
        cancel_token: Option<CancellationToken>,
    ) -> sandbox_driver::Result<ExecResult> {
        let mut spec = ExecSpec::bash(command).no_timeout();
        if let Some(timeout) = timeout {
            spec = spec.timeout(timeout);
        }
        if let Some(dir) = working_dir {
            spec = spec.working_dir(dir);
        }
        for (key, value) in env_vars.into_iter().flatten() {
            spec = spec.env_var(key, value);
        }
        let controls = ExecControls {
            term: cancel_token,
            ..ExecControls::default()
        };
        Ok(self.run_streaming(spec, controls).await?.result)
    }

    /// Runs `spec` under the policy, delivering output through
    /// `controls.sink` as it arrives.
    ///
    /// The policy fills what the spec leaves open: the stop grace, the
    /// working directory, and the text output policy. The spec's environment
    /// goes to the provider as the caller composed it. The caller's
    /// `controls.term` is the `term` stop; the provider runs the grace and
    /// the `kill` itself. Output beyond `controls.retained_output_limit`
    /// ([`DEFAULT_RETAINED_OUTPUT_BYTES`] when unset) is drained and counted,
    /// not kept.
    pub async fn run_streaming(
        &self,
        spec: ExecSpec,
        mut controls: ExecControls,
    ) -> sandbox_driver::Result<ExecStreamingResult> {
        let spec = self.apply_policy(spec);
        if controls.retained_output_limit.is_none() {
            controls.retained_output_limit = Some(DEFAULT_RETAINED_OUTPUT_BYTES);
        }
        self.exec.run_streaming(&spec, controls).await
    }

    /// Fills what a spec leaves open. The output policy has no "unset"
    /// state: the driver's default is raw, and pebble reads command output
    /// as text, so a spec still at that default gets
    /// [`OutputSanitization::StripAll`]; a caller that chose another policy
    /// keeps it.
    fn apply_policy(&self, mut spec: ExecSpec) -> ExecSpec {
        if spec.stop_grace.is_none() {
            spec.stop_grace = Some(self.stop_grace);
        }
        if spec.working_dir.is_none() {
            spec.working_dir.clone_from(&self.working_dir);
        }
        if spec.output_sanitization == OutputSanitization::default() {
            spec.output_sanitization = OutputSanitization::StripAll;
        }
        spec
    }
}

/// The driver says how the command ended; pebble's event vocabulary has two
/// stops. A timeout is the provider's deadline (the ladder ran for it); a
/// cancelled or killed command was stopped by the caller's token, by a
/// foreign `kill`, or by a provider-side abort: it did not finish and no
/// deadline passed. `Exited`, or a provider that could not tell, is a
/// completed process; nothing asserts success here.
#[must_use]
pub fn command_termination(termination: Termination) -> CommandTermination {
    match termination {
        Termination::TimedOut => CommandTermination::TimedOut,
        Termination::Cancelled | Termination::Killed => CommandTermination::Cancelled,
        _ => CommandTermination::Exited,
    }
}

/// An exit code is only the command's own when it exited on its own. A
/// stopped command may still report the shell's `128 + signal` (143 for a
/// trapped `TERM`), which events must not present as a program result.
#[must_use]
pub fn program_exit_code(termination: Termination, exit_code: Option<i32>) -> Option<i32> {
    // `CommandTermination` is non-exhaustive: only a command that exited on
    // its own owns its exit code.
    match command_termination(termination) {
        CommandTermination::Exited => exit_code,
        _ => None,
    }
}

/// Pebble's reading of a driver [`ExecResult`]: the event-facing numbers.
pub trait ExecResultExt {
    /// The provider's measured run time in whole milliseconds.
    fn duration_ms(&self) -> u64;

    /// The exit code when the command ended on its own; see
    /// [`program_exit_code`].
    fn program_exit_code(&self) -> Option<i32>;
}

impl ExecResultExt for ExecResult {
    fn duration_ms(&self) -> u64 {
        u64::try_from(self.duration.as_millis()).unwrap_or(u64::MAX)
    }

    fn program_exit_code(&self) -> Option<i32> {
        program_exit_code(self.termination, self.exit_code)
    }
}

/// A redacted [`ExecOutputTail`] from stdout/stderr text. Each stream goes
/// through `redactor`, then is capped to its newest `max_bytes_per_stream`.
/// Terminal control sequences are not stripped here: command output reaches
/// the caller with them already removed by the driver under
/// [`SandboxExec`]'s output policy. Pass `""` for either stream that isn't
/// relevant. Returns `None` when both streams are empty.
#[must_use]
pub fn redacted_output_tail(
    stdout: &str,
    stderr: &str,
    max_bytes_per_stream: usize,
    redactor: &dyn Redactor,
) -> Option<ExecOutputTail> {
    let (stdout, stdout_truncated) = redacted_tail(stdout, max_bytes_per_stream, redactor);
    let (stderr, stderr_truncated) = redacted_tail(stderr, max_bytes_per_stream, redactor);
    let tail = ExecOutputTail {
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
    };
    (!tail.is_empty()).then_some(tail)
}

fn redacted_tail(text: &str, max_bytes: usize, redactor: &dyn Redactor) -> (Option<String>, bool) {
    if text.is_empty() || max_bytes == 0 {
        return (None, !text.is_empty());
    }

    let redacted = redactor.redact(text);
    let truncated = redacted.len() > max_bytes;
    let start = if truncated {
        floor_char_boundary(&redacted, redacted.len() - max_bytes)
    } else {
        0
    };
    let tail = redacted[start..].to_string();
    ((!tail.is_empty()).then_some(tail), truncated)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use sandbox_driver::{
        BASH_ENV_VAR, OutputSink, OutputStream, SandboxProvider as _, SandboxSource, SandboxSpec,
        TransportError,
    };
    use sandbox_driver_host::HostProvider;
    use tokio::{fs, time};

    use super::*;
    use crate::environment::DEFAULT_EXEC_OUTPUT_TAIL_BYTES;
    use crate::redact::NoRedaction;
    use crate::sandbox_driver::testing::MaskToken;

    struct HostFixture {
        workspace: tempfile::TempDir,
        _provider: HostProvider,
        sandbox:   Arc<dyn sandbox_driver::Sandbox>,
    }

    impl HostFixture {
        async fn new() -> Self {
            let workspace = tempfile::tempdir().unwrap();
            let provider = HostProvider::new();
            let sandbox = provider
                .create(
                    &SandboxSpec::new(SandboxSource::HostDirectory)
                        .working_directory(workspace.path().display().to_string()),
                    None,
                )
                .await
                .unwrap();
            Self {
                workspace,
                _provider: provider,
                sandbox,
            }
        }

        fn exec(&self) -> SandboxExec<'_> {
            SandboxExec::new(self.sandbox.exec())
        }
    }

    async fn run(fixture: &HostFixture, command: &str) -> ExecResult {
        fixture
            .exec()
            .run(command, Some(Duration::from_secs(10)), None, None, None)
            .await
            .unwrap()
    }

    fn exec_result(stdout: &str, exit_code: Option<i32>, duration_ms: u64) -> ExecResult {
        let mut result = ExecResult::new(
            Termination::Exited,
            exit_code,
            Duration::from_millis(duration_ms),
        );
        result.stdout = stdout.as_bytes().to_vec();
        result
    }

    #[tokio::test]
    async fn runs_bash_source_and_reports_exit_code_and_streams() {
        let fixture = HostFixture::new().await;
        let result = run(&fixture, "echo out; echo err >&2; exit 3").await;
        assert_eq!(result.stdout_lossy(), "out\n");
        assert_eq!(result.stderr_lossy(), "err\n");
        assert_eq!(result.exit_code, Some(3));
        assert_eq!(result.termination, Termination::Exited);
        assert!(!result.success());
        assert!(run(&fixture, "true").await.success());
    }

    #[tokio::test]
    async fn runs_bash_only_syntax_in_a_clean_non_login_shell() {
        let fixture = HostFixture::new().await;
        let result = run(
            &fixture,
            "[[ -n ${BASH_VERSION:-} ]] && shopt -q login_shell && echo login || echo nonlogin; \
             set -o | grep -E '^(errexit|pipefail)' | awk '{print $2}' | sort -u",
        )
        .await;
        assert_eq!(result.stdout_lossy(), "nonlogin\noff\n", "{result:?}");
    }

    #[tokio::test]
    async fn a_caller_supplied_bash_env_never_runs() {
        let fixture = HostFixture::new().await;
        let startup = fixture.workspace.path().join("startup.sh");
        fs::write(&startup, "echo startup-source-loaded\n")
            .await
            .unwrap();
        let env = HashMap::from([(BASH_ENV_VAR.to_string(), startup.display().to_string())]);
        let result = fixture
            .exec()
            .run(
                "echo body",
                Some(Duration::from_secs(10)),
                None,
                Some(&env),
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.stdout_lossy(), "body\n");
    }

    #[tokio::test]
    async fn explicit_variables_reach_the_command_as_composed() {
        let fixture = HostFixture::new().await;
        let env = HashMap::from([
            ("APP_WORKER_TOKEN".to_string(), "deliberate".to_string()),
            ("MY_VAR".to_string(), "ok".to_string()),
        ]);
        let stdout = fixture
            .exec()
            .run("env", Some(Duration::from_secs(10)), None, Some(&env), None)
            .await
            .unwrap()
            .stdout_lossy();
        assert!(stdout.contains("APP_WORKER_TOKEN=deliberate"), "{stdout}");
        assert!(stdout.contains("MY_VAR=ok"), "{stdout}");
    }

    #[tokio::test]
    async fn the_working_directory_applies_when_the_caller_names_none() {
        let fixture = HostFixture::new().await;
        let nested = fixture.workspace.path().join("nested");
        fs::create_dir_all(&nested).await.unwrap();
        let stdout = SandboxExec::new(fixture.sandbox.exec())
            .with_working_dir(nested.display().to_string())
            .run("pwd", Some(Duration::from_secs(10)), None, None, None)
            .await
            .unwrap()
            .stdout_lossy();
        assert_eq!(
            Path::new(stdout.trim()).canonicalize().unwrap(),
            nested.canonicalize().unwrap()
        );
    }

    #[tokio::test]
    async fn timeout_runs_the_ladder_and_reports_timed_out() {
        let fixture = HostFixture::new().await;
        let started = Instant::now();
        let result = fixture
            .exec()
            .run(
                "sleep 10",
                Some(Duration::from_millis(200)),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.termination, Termination::TimedOut);
        assert_eq!(result.program_exit_code(), None);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "sleep honours TERM, so KILL should not have been needed"
        );
    }

    #[tokio::test]
    async fn a_command_that_ignores_term_is_killed_after_the_grace_period() {
        let fixture = HostFixture::new().await;
        let started = Instant::now();
        let result = fixture
            .exec()
            .with_stop_grace(Duration::from_millis(300))
            .run(
                "trap '' TERM; sleep 10",
                Some(Duration::from_millis(100)),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.termination, Termination::TimedOut);
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(400), "{elapsed:?}");
        assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
    }

    #[tokio::test]
    async fn cancellation_reports_cancelled() {
        let fixture = HostFixture::new().await;
        let token = CancellationToken::new();
        let cancel = token.clone();
        tokio::spawn(async move {
            time::sleep(Duration::from_millis(100)).await;
            cancel.cancel();
        });
        let result = fixture
            .exec()
            .run(
                "sleep 10",
                Some(Duration::from_secs(30)),
                None,
                None,
                Some(token),
            )
            .await
            .unwrap();
        assert_eq!(result.termination, Termination::Cancelled);
        assert_eq!(result.program_exit_code(), None);
    }

    #[tokio::test]
    async fn streaming_delivers_live_chunks_and_drains_past_the_retention_cap() {
        let fixture = HostFixture::new().await;
        let seen = Arc::new(Mutex::new(Vec::<u8>::new()));
        let sink_seen = Arc::clone(&seen);
        let sink: OutputSink = Arc::new(move |stream, chunk| {
            let seen = Arc::clone(&sink_seen);
            Box::pin(async move {
                assert_eq!(stream, OutputStream::Stdout);
                seen.lock().unwrap().extend_from_slice(&chunk);
                Ok(())
            })
        });
        let streaming = fixture
            .exec()
            .run_streaming(
                ExecSpec::bash("for i in $(seq 1 200); do echo line-$i; done")
                    .timeout(Duration::from_secs(10)),
                ExecControls {
                    sink: Some(sink),
                    retained_output_limit: Some(64),
                    ..ExecControls::default()
                },
            )
            .await
            .unwrap();
        assert!(streaming.result.success());
        assert!(streaming.live_streaming);
        assert!(streaming.streams_separated);
        let delivered = seen.lock().unwrap().len();
        assert_eq!(streaming.stdout_capture.observed_bytes, delivered);
        assert!(streaming.stdout_capture.omitted_bytes > 0);
        assert!(streaming.result.stdout.len() <= 64);
        assert!(streaming.result.stdout.starts_with(b"line-1\n"));
        assert!(streaming.result.stdout.ends_with(b"line-200\n"));
    }

    #[tokio::test]
    async fn stdin_bytes_are_written_exactly_then_closed() {
        let fixture = HostFixture::new().await;
        let stdin = b"first line\n$(touch must-not-run)\nlast line".to_vec();
        let streaming = fixture
            .exec()
            .run_streaming(
                ExecSpec::bash("cat; test -e must-not-run && echo RAN")
                    .timeout(Duration::from_secs(10))
                    .stdin(stdin.clone()),
                ExecControls::default(),
            )
            .await
            .unwrap();
        assert_eq!(streaming.result.stdout, stdin);
    }

    #[tokio::test]
    async fn a_failing_output_sink_stops_the_command_with_an_error() {
        let fixture = HostFixture::new().await;
        let sink: OutputSink = Arc::new(|_, _| {
            Box::pin(async {
                Err(sandbox_driver::Error::Transport(TransportError::new(
                    "consumer gave up",
                )))
            })
        });
        let error = fixture
            .exec()
            .run_streaming(
                ExecSpec::bash("echo hello; sleep 5").timeout(Duration::from_secs(10)),
                ExecControls {
                    sink: Some(sink),
                    ..ExecControls::default()
                },
            )
            .await
            .map(|streaming| streaming.result.termination);
        // The driver either surfaces the sink failure or reports the command
        // cancelled by it; both keep the consumer's error visible.
        match error {
            Ok(termination) => assert_eq!(termination, Termination::Cancelled),
            Err(error) => assert!(error.to_string().contains("consumer gave up"), "{error}"),
        }
    }

    #[test]
    fn termination_mapping_reads_the_drivers_verdict() {
        assert_eq!(
            command_termination(Termination::TimedOut),
            CommandTermination::TimedOut
        );
        assert_eq!(
            command_termination(Termination::Cancelled),
            CommandTermination::Cancelled
        );
        assert_eq!(
            command_termination(Termination::Killed),
            CommandTermination::Cancelled
        );
        assert_eq!(
            command_termination(Termination::Exited),
            CommandTermination::Exited
        );
    }

    #[test]
    fn program_exit_code_is_the_commands_own_only_when_it_exited() {
        assert_eq!(program_exit_code(Termination::Exited, Some(3)), Some(3));
        assert_eq!(program_exit_code(Termination::TimedOut, Some(143)), None);
        assert_eq!(program_exit_code(Termination::Cancelled, Some(143)), None);
        assert_eq!(program_exit_code(Termination::Killed, Some(137)), None);
        assert_eq!(exec_result("", Some(3), 42).duration_ms(), 42);
    }

    #[test]
    fn output_tail_redacts_before_truncating() {
        let secret = "sk-ant-api03-xK9mZ2vL8nQ5rT1wY4bC7dF0gH3jE6pA";
        let tail = redacted_output_tail(
            &format!("{} {secret} done", "context ".repeat(20)),
            "",
            32,
            &MaskToken(secret),
        )
        .expect("redacted output tail");
        let stdout = tail.stdout.expect("stdout tail");
        assert!(stdout.contains("REDACTED"), "{stdout}");
        assert!(!stdout.contains("F0gH3jE6pA"), "{stdout}");
        assert!(tail.stdout_truncated);
        assert!(redacted_output_tail("", "", 32, &NoRedaction).is_none());
    }

    #[tokio::test]
    async fn command_output_arrives_stripped_of_terminal_control_sequences() {
        let fixture = HostFixture::new().await;
        let result = run(
            &fixture,
            "printf '\\033[31mred\\033[0m \\033]0;window-title\\007shown \\033(Bset \\033Mtwo-byte \
             \\bbackspace'",
        )
        .await;
        assert!(result.success(), "{result:?}");
        assert_eq!(result.stdout_lossy(), "red shown set two-byte backspace");
    }

    #[tokio::test]
    async fn policy_strips_output_unless_the_caller_chose_another_policy() {
        let fixture = HostFixture::new().await;
        let exec = fixture.exec();
        assert_eq!(
            exec.apply_policy(ExecSpec::bash("true"))
                .output_sanitization,
            OutputSanitization::StripAll
        );
        assert_eq!(
            exec.apply_policy(
                ExecSpec::bash("true").output_sanitization(OutputSanitization::StripAnsi)
            )
            .output_sanitization,
            OutputSanitization::StripAnsi
        );
    }

    #[test]
    fn default_output_tail_serialized_budget_stays_below_40_kib() {
        let tail = redacted_output_tail(
            &"o".repeat(DEFAULT_EXEC_OUTPUT_TAIL_BYTES + 128),
            &"e".repeat(DEFAULT_EXEC_OUTPUT_TAIL_BYTES + 128),
            DEFAULT_EXEC_OUTPUT_TAIL_BYTES,
            &NoRedaction,
        )
        .expect("tail present");
        assert_eq!(
            tail.stdout.as_deref().map(str::len),
            Some(DEFAULT_EXEC_OUTPUT_TAIL_BYTES)
        );
        assert_eq!(
            tail.stderr.as_deref().map(str::len),
            Some(DEFAULT_EXEC_OUTPUT_TAIL_BYTES)
        );
        assert!(tail.stdout_truncated);
        assert!(tail.stderr_truncated);
        let serialized = serde_json::to_vec(&tail).expect("serialize tail");
        assert!(
            serialized.len() < 40 * 1024,
            "tail JSON was {} bytes",
            serialized.len()
        );
    }
}
