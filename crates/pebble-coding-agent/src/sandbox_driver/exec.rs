//! The command execution policy over a sandbox-driver [`Sandbox`].
//!
//! The vocabulary going in is the driver's own: an [`ExecSpec`] and
//! [`ExecControls`]. A command runs through the driver's
//! [`run_captured_with`], which keeps a bounded head and tail of each stream
//! and still hands every chunk to the caller's sink. Coming out is pebble's
//! [`ExecResult`] and [`ExecStreamingResult`]. This module adds the policy on
//! the way in and keeps what pebble reads on the way out.
//!
//! A command is shell source run by [`ExecSpec::shell`]: Bash when the
//! sandbox has it (POSIX `sh` otherwise), never a login shell, with
//! `BASH_ENV` unset whatever the caller passed. It ends in one of three ways:
//!
//! - **timeout**: the policy's own deadline passes and the policy runs the stop
//!   ladder: the driver's `term` stop, then its `kill` stop after
//!   [`SandboxExec::stop_grace`]. The result reports [`Termination::TimedOut`].
//! - **cancellation**: the caller's `term` token starts the same ladder. The
//!   result reports the driver's [`Termination::Terminated`], or
//!   [`Termination::Killed`] when the grace ran out.
//! - **exit**: the process ended on its own.
//!
//! Output is drained regardless of the retention cap and delivered live
//! through the caller's [`OutputSink`]. Pebble reads command output as text,
//! so the policy strips terminal escape sequences and stray control
//! characters ([`OutputSanitization::StripAll`]) before a sink chunk, a
//! result, or a tail sees them. Secret redaction is the application's
//! ([`Redactor`]) and happens only when a tail is rendered for events or logs
//! ([`redacted_output_tail`]). The explicit environment reaches the provider
//! as the caller composed it.
//!
//! Parity gaps: the old driver ran the stop grace and the output policy
//! itself (`ExecSpec::stop_grace`, `ExecSpec::output_sanitization`); the new
//! API has neither, so they live here. The new Host provider kills a command
//! outright when the spec's `timeout_ms` passes, so the policy never sets it
//! and keeps the deadline itself, to keep the `TERM` grace on a timeout.
//! `Captured` has no duration, so the policy times the call. The driver keeps
//! output raw, so the retained head and tail are sanitized after capture: the
//! counts are of the raw bytes, and an escape sequence cut by the gap between
//! head and tail is dropped up to its terminator.

use std::collections::HashMap;
use std::future::{self, Future};
use std::io::Cursor;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use sandbox_driver::{
    Captured, CapturedOutput, DEFAULT_CAPTURE_BYTES, Error as DriverError, ExecControls, ExecSpec,
    OutputLoss, OutputSink, OutputStream, Sandbox, Termination, run_captured_with, stop_signal,
};
use tokio::time;
use tokio_util::sync::CancellationToken;

use super::sanitize::{OutputSanitization, OutputSanitizer};
use crate::char_boundary::floor_char_boundary;
use crate::event::OutputCaptureStats;
use crate::redact::Redactor;
use crate::types::{CommandTermination, ExecOutputTail};

/// Time between `TERM` and `KILL` when the policy stops a command.
pub const DEFAULT_STOP_GRACE: Duration = Duration::from_secs(2);

/// Retention per stream when a caller sets no cap: the driver's own default,
/// enough for any build log an application renders, bounded so a runaway
/// command cannot exhaust memory.
pub const DEFAULT_RETAINED_OUTPUT_BYTES: usize = DEFAULT_CAPTURE_BYTES;

/// Bash sources the file this names before a `-c` script; the policy blanks
/// it for every shell script.
pub const BASH_ENV_VAR: &str = "BASH_ENV";

/// The script of a spec [`ExecSpec::shell`] built: its last argument.
#[must_use]
pub fn shell_source(spec: &ExecSpec) -> Option<&str> {
    match spec.argv.as_slice() {
        [program, flag, .., script] if program == "sh" && flag == "-c" && spec.argv.len() == 5 => {
            Some(script)
        }
        _ => None,
    }
}

/// A finished command as pebble reads it: how it ended, and the output it
/// left behind, as retained under the policy's cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecResult {
    /// How the command ended, as the driver reported it, or
    /// [`Termination::TimedOut`] when the policy's deadline stopped it.
    pub termination: Termination,
    /// The process's exit code, when it has one.
    pub exit_code:   Option<i32>,
    /// The signal that ended the process, when the provider reports one.
    pub signal:      Option<i32>,
    /// Retained standard output, after sanitization.
    pub stdout:      Vec<u8>,
    /// Retained standard error, after sanitization.
    pub stderr:      Vec<u8>,
    /// Time from starting the command to its end, measured by the policy.
    pub duration:    Duration,
}

impl ExecResult {
    /// A result with no output.
    #[must_use]
    pub fn new(termination: Termination, exit_code: Option<i32>, duration: Duration) -> Self {
        Self {
            termination,
            exit_code,
            signal: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            duration,
        }
    }

    /// The command exited on its own with code 0.
    #[must_use]
    pub fn success(&self) -> bool {
        self.termination == Termination::Exited && self.exit_code == Some(0)
    }

    /// Standard output as text, invalid UTF-8 replaced.
    #[must_use]
    pub fn stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// Standard error as text, invalid UTF-8 replaced.
    #[must_use]
    pub fn stderr_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }

    /// The run time in whole milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        u64::try_from(self.duration.as_millis()).unwrap_or(u64::MAX)
    }

    /// The exit code when the command ended on its own; see
    /// [`program_exit_code`].
    #[must_use]
    pub fn program_exit_code(&self) -> Option<i32> {
        program_exit_code(self.termination, self.exit_code)
    }
}

/// A streamed command's result with the accounting a streaming caller needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecStreamingResult {
    pub result:            ExecResult,
    /// Bytes standard output produced and how many the cap kept.
    pub stdout_capture:    OutputCaptureStats,
    /// Bytes standard error produced and how many the cap kept.
    pub stderr_capture:    OutputCaptureStats,
    /// Whether stdout and stderr arrived apart. A provider that delivers one
    /// terminal stream has its output in `stdout`.
    pub streams_separated: bool,
    /// Output the provider lost on its own transport; see
    /// [`sandbox_driver::ExecOutcome::output_loss`].
    pub output_loss:       OutputLoss,
}

impl ExecStreamingResult {
    /// A result whose capture counts are the retained output's own.
    #[must_use]
    pub fn new(result: ExecResult) -> Self {
        let stats = |bytes: &[u8]| OutputCaptureStats {
            observed_bytes: bytes.len(),
            retained_bytes: bytes.len(),
            omitted_bytes:  0,
        };
        Self {
            stdout_capture: stats(&result.stdout),
            stderr_capture: stats(&result.stderr),
            result,
            streams_separated: true,
            output_loss: OutputLoss::default(),
        }
    }
}

/// The exec policy bound to one sandbox.
#[derive(Clone)]
pub struct SandboxExec<'a> {
    sandbox:      &'a dyn Sandbox,
    stop_grace:   Duration,
    sanitization: OutputSanitization,
    /// Where a command runs when the caller names no directory. `None`
    /// leaves the choice to the provider's own working directory.
    working_dir:  Option<String>,
}

impl<'a> SandboxExec<'a> {
    #[must_use]
    pub fn new(sandbox: &'a dyn Sandbox) -> Self {
        Self {
            sandbox,
            stop_grace: DEFAULT_STOP_GRACE,
            sanitization: OutputSanitization::StripAll,
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

    /// Time between `TERM` and `KILL` when a command is stopped.
    #[must_use]
    pub fn with_stop_grace(mut self, stop_grace: Duration) -> Self {
        self.stop_grace = stop_grace;
        self
    }

    /// How command output is cleaned before anyone sees it.
    /// [`OutputSanitization::StripAll`] unless set.
    #[must_use]
    pub fn with_output_sanitization(mut self, sanitization: OutputSanitization) -> Self {
        self.sanitization = sanitization;
        self
    }

    #[must_use]
    pub fn stop_grace(&self) -> Duration {
        self.stop_grace
    }

    /// Runs shell source to completion and returns its captured output.
    ///
    /// The script runs as [`ExecSpec::shell`] runs it: under Bash when the
    /// sandbox has it, never a login shell, `BASH_ENV` unset, no `errexit`
    /// or `pipefail`. `None` for `timeout` runs without a deadline.
    pub async fn run(
        &self,
        command: &str,
        timeout: Option<Duration>,
        working_dir: Option<&str>,
        env_vars: Option<&HashMap<String, String>>,
        cancel_token: Option<CancellationToken>,
    ) -> Result<ExecResult, DriverError> {
        let mut spec = ExecSpec::shell(command);
        spec.timeout_ms = timeout.map(duration_ms);
        spec.working_directory = working_dir.map(str::to_string);
        for (key, value) in env_vars.into_iter().flatten() {
            spec.env.insert(key.clone(), value.clone());
        }
        let controls = ExecControls {
            term: cancel_token,
            ..ExecControls::default()
        };
        Ok(self.run_streaming(spec, controls, None).await?.result)
    }

    /// Runs `spec` under the policy, delivering output through
    /// `controls.sink` as it arrives.
    ///
    /// The policy fills what the spec leaves open: the working directory.
    /// `spec.timeout_ms` is the policy's deadline, not the provider's: the
    /// policy runs the ladder for it. The caller's `controls.term` starts the
    /// same ladder; `controls.kill` reaches the provider directly. Output
    /// beyond `retained_output_limit` per stream
    /// ([`DEFAULT_RETAINED_OUTPUT_BYTES`] when `None`) is drained and
    /// counted, not kept.
    pub async fn run_streaming(
        &self,
        spec: ExecSpec,
        controls: ExecControls,
        retained_output_limit: Option<usize>,
    ) -> Result<ExecStreamingResult, DriverError> {
        let (spec, deadline) = self.apply_policy(spec);
        let ExecControls {
            stdin,
            sink,
            term: caller_term,
            kill: caller_kill,
            resize,
        } = controls;
        let live = Arc::new(Mutex::new(LiveOutput::new(self.sanitization)));
        let term = CancellationToken::new();
        let kill = CancellationToken::new();
        let timed_out = AtomicBool::new(false);
        let driver_controls = ExecControls {
            stdin,
            sink: Some(sanitizing_sink(Arc::clone(&live), sink.clone())),
            term: Some(term.clone()),
            kill: Some(kill.clone()),
            resize,
        };
        let ladder = async {
            tokio::select! {
                () = stop_signal(caller_term.as_ref()) => {}
                () = sleep_until(deadline) => timed_out.store(true, Ordering::SeqCst),
            }
            term.cancel();
            time::sleep(self.stop_grace).await;
            kill.cancel();
            future::pending::<()>().await;
        };
        let forward_kill = async {
            stop_signal(caller_kill.as_ref()).await;
            kill.cancel();
            future::pending::<()>().await;
        };
        let started = Instant::now();
        let captured = tokio::select! {
            captured = run_captured_with(
                self.sandbox,
                spec,
                driver_controls,
                retained_output_limit.unwrap_or(DEFAULT_RETAINED_OUTPUT_BYTES),
            ) => captured?,
            () = ladder => unreachable!("the ladder never finishes"),
            () = forward_kill => unreachable!("forwarding never finishes"),
        };
        let duration = started.elapsed();
        let (tail, streams_separated) = {
            let mut live = live.lock().unwrap_or_else(PoisonError::into_inner);
            (live.finish(), live.streams_separated)
        };
        if let Some(sink) = &sink {
            for (stream, bytes) in tail {
                sink(stream, bytes).await?;
            }
        }
        Ok(self.to_result(
            captured,
            timed_out.into_inner(),
            streams_separated,
            duration,
        ))
    }

    /// Fills what a spec leaves open and takes its deadline for the policy.
    fn apply_policy(&self, mut spec: ExecSpec) -> (ExecSpec, Option<Duration>) {
        if spec.working_directory.is_none() {
            spec.working_directory.clone_from(&self.working_dir);
        }
        // Parity gap: `ExecSpec::shell` unsets `BASH_ENV` in its launcher,
        // but Host starts every command under a `/bin/bash -c` sentinel that
        // already sourced a `BASH_ENV` the spec passed. Blanking it keeps a
        // caller's startup file from running ahead of the script anywhere.
        // TODO(parity): drop this once sandbox-driver#100 (Host blanks
        // `BASH_ENV` for every command) merges.
        if shell_source(&spec).is_some() {
            spec.env.insert(BASH_ENV_VAR.to_string(), String::new());
        }
        let deadline = spec.timeout_ms.take().map(Duration::from_millis);
        (spec, deadline)
    }

    fn to_result(
        &self,
        captured: Captured,
        timed_out: bool,
        streams_separated: bool,
        duration: Duration,
    ) -> ExecStreamingResult {
        let Captured {
            outcome,
            stdout,
            stderr,
        } = captured;
        let (stdout, stdout_capture) = self.retained(&stdout);
        let (stderr, stderr_capture) = self.retained(&stderr);
        // The deadline fired, so the ladder stopped the command (or it ended
        // in the same instant): either way it ran out of time.
        let termination = if timed_out {
            Termination::TimedOut
        } else {
            outcome.termination
        };
        ExecStreamingResult {
            result: ExecResult {
                termination,
                exit_code: outcome.exit_code,
                signal: outcome.signal,
                stdout,
                stderr,
                duration,
            },
            stdout_capture,
            stderr_capture,
            streams_separated,
            output_loss: outcome.output_loss,
        }
    }

    /// The driver's retained bytes, sanitized, and their counts.
    fn retained(&self, output: &CapturedOutput) -> (Vec<u8>, OutputCaptureStats) {
        let bytes = self.sanitization.sanitize(&output.bytes);
        let omitted_bytes = usize::try_from(output.omitted_bytes).unwrap_or(usize::MAX);
        let stats = OutputCaptureStats {
            observed_bytes: bytes.len().saturating_add(omitted_bytes),
            retained_bytes: bytes.len(),
            omitted_bytes,
        };
        (bytes, stats)
    }
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

async fn sleep_until(deadline: Option<Duration>) {
    match deadline {
        Some(deadline) => time::sleep(deadline).await,
        None => future::pending().await,
    }
}

/// The sanitizers for the caller's live sink, one per stream, so an escape
/// sequence split across chunks is still removed whole.
struct LiveOutput {
    stdout:            OutputSanitizer,
    stderr:            OutputSanitizer,
    streams_separated: bool,
}

impl LiveOutput {
    fn new(sanitization: OutputSanitization) -> Self {
        Self {
            stdout:            OutputSanitizer::new(sanitization),
            stderr:            OutputSanitizer::new(sanitization),
            streams_separated: true,
        }
    }

    /// Cleans one chunk. A terminal stream is stdout's.
    fn push(&mut self, stream: OutputStream, chunk: &[u8]) -> Vec<u8> {
        match stream {
            OutputStream::Stderr => self.stderr.push(chunk),
            OutputStream::Terminal => {
                self.streams_separated = false;
                self.stdout.push(chunk)
            }
            OutputStream::Stdout => self.stdout.push(chunk),
        }
    }

    /// What the sanitizers still hold, per stream, for the caller's sink.
    fn finish(&mut self) -> Vec<(OutputStream, Vec<u8>)> {
        let stdout_stream = if self.streams_separated {
            OutputStream::Stdout
        } else {
            OutputStream::Terminal
        };
        [
            (stdout_stream, self.stdout.finish()),
            (OutputStream::Stderr, self.stderr.finish()),
        ]
        .into_iter()
        .filter(|(_, bytes)| !bytes.is_empty())
        .collect()
    }
}

type SinkFuture = Pin<Box<dyn Future<Output = Result<(), DriverError>> + Send>>;

/// The sink the driver's capture forwards to: every chunk is cleaned, then
/// handed to the caller's sink, which sees it uncapped.
fn sanitizing_sink(live: Arc<Mutex<LiveOutput>>, sink: Option<OutputSink>) -> OutputSink {
    Arc::new(move |stream, chunk: Vec<u8>| -> SinkFuture {
        let cleaned = live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(stream, &chunk);
        match &sink {
            Some(sink) if !cleaned.is_empty() => sink(stream, cleaned),
            _ => Box::pin(future::ready(Ok(()))),
        }
    })
}

/// A command that ran and failed, kept with the output it left behind.
///
/// Parity gap: the old driver's `Error::Exec(ExecFailure)` carried this, and
/// [`display_for_log`](super::display_for_log) rendered its output tail. The
/// new driver's errors carry no command output, so the adapter's own derived
/// operations (file, search) fail with this instead, and the log rendering
/// reads it from any error chain.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{label} failed ({termination:?}, exit code {exit_code:?})")]
pub struct CommandFailure {
    /// What the command was for, such as `grep`; never the command itself.
    pub label:       String,
    pub termination: Termination,
    pub exit_code:   Option<i32>,
    pub stdout:      Vec<u8>,
    pub stderr:      Vec<u8>,
    pub duration:    Duration,
}

impl CommandFailure {
    /// The failure a finished `result` describes.
    #[must_use]
    pub fn new(label: impl Into<String>, result: ExecResult) -> Self {
        Self {
            label:       label.into(),
            termination: result.termination,
            exit_code:   result.exit_code,
            stdout:      result.stdout,
            stderr:      result.stderr,
            duration:    result.duration,
        }
    }
}

/// `bytes` as the stream [`ExecControls::stdin`] takes.
#[must_use]
pub fn stdin_bytes(bytes: Vec<u8>) -> sandbox_driver::ByteStream {
    Box::pin(Cursor::new(bytes))
}

/// The driver says how the command ended; pebble's event vocabulary has two
/// stops. A timeout is the policy's deadline (the ladder ran for it); a
/// terminated or killed command was stopped by the caller's token, by a
/// foreign `kill`, or by a provider-side abort: it did not finish and no
/// deadline passed. `Exited`, or `Lost` (a provider that could not tell), is
/// a completed process; nothing asserts success here.
#[must_use]
pub fn command_termination(termination: Termination) -> CommandTermination {
    match termination {
        Termination::TimedOut => CommandTermination::TimedOut,
        Termination::Terminated | Termination::Killed => CommandTermination::Cancelled,
        Termination::Exited | Termination::Lost => CommandTermination::Exited,
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

/// A redacted [`ExecOutputTail`] from stdout/stderr text. Each stream goes
/// through `redactor`, then is capped to its newest `max_bytes_per_stream`.
/// Terminal control sequences are not stripped here: command output reaches
/// the caller with them already removed under [`SandboxExec`]'s output
/// policy. Pass `""` for either stream that isn't relevant. Returns `None`
/// when both streams are empty.
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

    use sandbox_driver::Sandboxes;
    use tokio::{fs, time};

    use super::*;
    use crate::environment::DEFAULT_EXEC_OUTPUT_TAIL_BYTES;
    use crate::redact::NoRedaction;
    use crate::sandbox_driver::testing::{MaskToken, host_sandbox};

    struct HostFixture {
        _catalog: Sandboxes,
        sandbox:  Arc<dyn Sandbox>,
    }

    impl HostFixture {
        async fn new() -> Self {
            let (catalog, sandbox) = host_sandbox().await;
            Self {
                _catalog: catalog,
                sandbox,
            }
        }

        fn workspace(&self) -> &Path {
            self.sandbox.working_directory()
        }

        fn exec(&self) -> SandboxExec<'_> {
            SandboxExec::new(&*self.sandbox)
        }
    }

    async fn run(fixture: &HostFixture, command: &str) -> ExecResult {
        fixture
            .exec()
            .run(command, Some(Duration::from_secs(10)), None, None, None)
            .await
            .unwrap()
    }

    fn timed(command: &str, timeout: Duration) -> ExecSpec {
        let mut spec = ExecSpec::shell(command);
        spec.timeout_ms = Some(duration_ms(timeout));
        spec
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
        let startup = fixture.workspace().join("startup.sh");
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
        let nested = fixture.workspace().join("nested");
        fs::create_dir_all(&nested).await.unwrap();
        let stdout = SandboxExec::new(&*fixture.sandbox)
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
    async fn a_timeout_sends_term_first_so_a_trap_can_run() {
        let fixture = HostFixture::new().await;
        let marker = fixture.workspace().join("trapped");
        let result = fixture
            .exec()
            .run(
                &format!(
                    "trap 'echo trapped > {}; exit 0' TERM; sleep 10 & wait",
                    marker.display()
                ),
                Some(Duration::from_millis(200)),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.termination, Termination::TimedOut);
        assert_eq!(fs::read_to_string(&marker).await.unwrap(), "trapped\n");
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
    async fn cancellation_reports_a_stop_without_a_deadline() {
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
        assert_eq!(result.termination, Termination::Terminated);
        assert_eq!(
            command_termination(result.termination),
            CommandTermination::Cancelled
        );
        assert_eq!(result.program_exit_code(), None);
    }

    #[tokio::test]
    async fn streaming_delivers_every_byte_and_retains_a_bounded_head_and_tail() {
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
                timed(
                    "for i in $(seq 1 200); do echo line-$i; done",
                    Duration::from_secs(10),
                ),
                ExecControls {
                    sink: Some(sink),
                    ..ExecControls::default()
                },
                Some(64),
            )
            .await
            .unwrap();
        assert!(streaming.result.success());
        assert!(streaming.streams_separated);
        let delivered = seen.lock().unwrap().clone();
        assert!(delivered.ends_with(b"line-200\n"));
        assert_eq!(streaming.stdout_capture.observed_bytes, delivered.len());
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
                timed(
                    "cat; test -e must-not-run && echo RAN",
                    Duration::from_secs(10),
                ),
                ExecControls {
                    stdin: Some(stdin_bytes(stdin.clone())),
                    ..ExecControls::default()
                },
                None,
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
                Err(DriverError::Transport {
                    message: "consumer gave up".into(),
                    outcome: sandbox_driver::MutationOutcome::Unknown,
                })
            })
        });
        let started = Instant::now();
        let outcome = fixture
            .exec()
            .run_streaming(
                timed("echo hello; sleep 5", Duration::from_secs(10)),
                ExecControls {
                    sink: Some(sink),
                    ..ExecControls::default()
                },
                None,
            )
            .await
            .map(|streaming| streaming.result.termination);
        // The driver either surfaces the sink failure or reports the command
        // stopped by it; both keep the consumer's error visible.
        match outcome {
            Ok(termination) => assert_ne!(termination, Termination::Exited),
            Err(error) => assert!(error.to_string().contains("consumer gave up"), "{error}"),
        }
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn termination_mapping_reads_the_drivers_verdict() {
        assert_eq!(
            command_termination(Termination::TimedOut),
            CommandTermination::TimedOut
        );
        assert_eq!(
            command_termination(Termination::Terminated),
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
        assert_eq!(
            command_termination(Termination::Lost),
            CommandTermination::Exited
        );
    }

    #[test]
    fn program_exit_code_is_the_commands_own_only_when_it_exited() {
        assert_eq!(program_exit_code(Termination::Exited, Some(3)), Some(3));
        assert_eq!(program_exit_code(Termination::TimedOut, Some(143)), None);
        assert_eq!(program_exit_code(Termination::Terminated, Some(143)), None);
        assert_eq!(program_exit_code(Termination::Killed, Some(137)), None);
        let result = ExecResult::new(Termination::Exited, Some(3), Duration::from_millis(42));
        assert_eq!(result.duration_ms(), 42);
    }

    #[test]
    fn a_shell_spec_names_its_source() {
        assert_eq!(shell_source(&ExecSpec::shell("echo hi")), Some("echo hi"));
        assert_eq!(shell_source(&ExecSpec::argv(["echo", "hi"])), None);
        assert_eq!(shell_source(&ExecSpec::argv(["sh", "-c", "echo hi"])), None);
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
    async fn a_caller_can_keep_output_raw() {
        let fixture = HostFixture::new().await;
        let result = fixture
            .exec()
            .with_output_sanitization(OutputSanitization::Raw)
            .run(
                "printf '\\033[31mred'",
                Some(Duration::from_secs(10)),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.stdout, b"\x1b[31mred");
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
