//! Where a session's tools act: files, content search, and commands.
//!
//! [`Environment`] is the one seam between pebble's tools and the machine
//! their work lands on. An application implements it over whatever it has —
//! the local machine ([`LocalEnvironment`]), a container, a remote workspace —
//! and pebble stays out of process isolation, transport, and provisioning.
//!
//! The trait is deliberately small: file access, two search primitives, one
//! command runner, and three cheap accessors that prompt assembly reads. It
//! has no lifecycle methods. An environment is ready when it is handed to a
//! session, and whoever built it owns starting and disposing of it.
//!
//! Everything a tool shows the model comes from here, so two shapes are part
//! of the contract rather than implementation detail:
//! [`format_lines_numbered`] is the exact `read_file` rendering, and
//! [`ExecRequest::command`] is Bash source (see [`Environment::exec`]).

mod capture;
mod glob;
mod local;
#[cfg(any(test, feature = "test-util"))]
pub(crate) mod mock;

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt::Write as _;
use std::io::{self, ErrorKind};
use std::result::Result as StdResult;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

pub use self::local::{CallerEnvPolicy, LocalEnvironment};
use crate::char_boundary::floor_char_boundary;
use crate::event::OutputCaptureStats;
use crate::redact::Redactor;
use crate::types::{CommandTermination, ExecOutputTail};

/// Bytes of each stream kept in an [`ExecOutputTail`] when the caller names no
/// budget.
pub const DEFAULT_EXEC_OUTPUT_TAIL_BYTES: usize = 8 * 1024;

/// Why an [`Environment`] operation failed.
///
/// The kinds a caller branches on, not a mirror of every underlying failure.
/// New kinds may appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EnvironmentErrorKind {
    /// The path does not exist.
    NotFound,
    /// A file was read as text but is not valid UTF-8.
    InvalidUtf8,
    /// An I/O failure: permissions, disk, or transport.
    Io,
    /// A process could not be started, for example a missing interpreter or an
    /// unusable working directory.
    Spawn,
    /// This environment does not offer the operation.
    Unsupported,
    /// The caller's input was unusable, for example a malformed glob pattern.
    InvalidInput,
    /// Anything the kinds above do not describe.
    Other,
}

/// A failure from an [`Environment`] operation.
///
/// The message is written for the model: tools render it into a tool result,
/// so it names the operation and the path and carries no secrets. The
/// underlying failure stays attached as the error's source, which
/// [`detail`](Self::detail) renders for a log or a tool result that wants the
/// whole chain.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct EnvironmentError {
    kind:    EnvironmentErrorKind,
    message: String,
    #[source]
    source:  Option<Box<dyn StdError + Send + Sync + 'static>>,
}

impl EnvironmentError {
    /// Builds an error with no underlying cause.
    #[must_use]
    pub fn new(kind: EnvironmentErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    /// Builds an error that keeps `source` as its cause.
    #[must_use]
    pub fn with_source(
        kind: EnvironmentErrorKind,
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Builds an error from an I/O failure, classifying a missing path as
    /// [`EnvironmentErrorKind::NotFound`] and everything else as
    /// [`EnvironmentErrorKind::Io`].
    #[must_use]
    pub fn io(message: impl Into<String>, source: io::Error) -> Self {
        let kind = if source.kind() == ErrorKind::NotFound {
            EnvironmentErrorKind::NotFound
        } else {
            EnvironmentErrorKind::Io
        };
        Self::with_source(kind, message, source)
    }

    /// The category of this failure.
    #[must_use]
    pub const fn kind(&self) -> EnvironmentErrorKind {
        self.kind
    }

    /// The model-facing message, without its causes.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The message followed by one `"\n  caused by: ..."` line per cause.
    ///
    /// This is the rendering tools hand back to the model, so a failure keeps
    /// the detail that explains it — a missing directory under a failed write,
    /// a transport error under a failed read.
    #[must_use]
    pub fn detail(&self) -> String {
        let mut rendered = self.message.clone();
        let mut current = StdError::source(self);
        while let Some(cause) = current {
            let _ = write!(rendered, "\n  caused by: {cause}");
            current = cause.source();
        }
        rendered
    }
}

/// The result of a fallible [`Environment`] operation.
pub type EnvResult<T> = StdResult<T, EnvironmentError>;

/// What one command produced.
#[derive(Debug, Clone)]
pub struct ExecResult {
    /// Captured standard output, or the combined streams when the environment
    /// cannot separate them.
    pub stdout:      String,
    /// Captured standard error.
    pub stderr:      String,
    /// The process's exit status, absent when it was stopped before exiting.
    pub exit_code:   Option<i32>,
    /// How the process ended.
    pub termination: CommandTermination,
    /// Wall-clock duration of the run.
    pub duration_ms: u64,
}

impl ExecResult {
    /// Whether the process ran to completion and exited zero.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.exit_code == Some(0) && self.termination == CommandTermination::Exited
    }

    /// The exit code for display, using `-1` when the process never reported
    /// one.
    #[must_use]
    pub fn display_exit_code(&self) -> i32 {
        self.exit_code.unwrap_or(-1)
    }

    /// The retained tail of both streams, redacted, or `None` when the process
    /// wrote nothing.
    ///
    /// Each stream keeps its last `max_bytes_per_stream` bytes, cut at a
    /// character boundary, after `redactor` runs and after terminal escape
    /// sequences and other control characters are dropped. This is the shape
    /// an event carries; it is never the shape the model reads.
    #[must_use]
    pub fn output_tail(
        &self,
        redactor: &dyn Redactor,
        max_bytes_per_stream: usize,
    ) -> Option<ExecOutputTail> {
        let (stdout, stdout_truncated) = stream_tail(&self.stdout, redactor, max_bytes_per_stream);
        let (stderr, stderr_truncated) = stream_tail(&self.stderr, redactor, max_bytes_per_stream);
        let tail = ExecOutputTail {
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
        };
        (!tail.is_empty()).then_some(tail)
    }

    /// [`output_tail`](Self::output_tail) with
    /// [`DEFAULT_EXEC_OUTPUT_TAIL_BYTES`] per stream.
    #[must_use]
    pub fn default_output_tail(&self, redactor: &dyn Redactor) -> Option<ExecOutputTail> {
        self.output_tail(redactor, DEFAULT_EXEC_OUTPUT_TAIL_BYTES)
    }
}

fn stream_tail(text: &str, redactor: &dyn Redactor, max_bytes: usize) -> (Option<String>, bool) {
    if text.is_empty() || max_bytes == 0 {
        return (None, !text.is_empty());
    }

    let sanitized = sanitize_exec_output(&redactor.redact(text));
    let truncated = sanitized.len() > max_bytes;
    let start = if truncated {
        floor_char_boundary(&sanitized, sanitized.len() - max_bytes)
    } else {
        0
    };
    let tail = sanitized[start..].to_owned();
    ((!tail.is_empty()).then_some(tail), truncated)
}

/// Drops terminal escape sequences and control characters, keeping tabs and
/// newlines.
fn sanitize_exec_output(text: &str) -> String {
    let mut sanitized = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\u{1b}' {
            match characters.peek().copied() {
                // A control sequence runs to its final byte.
                Some('[') => {
                    characters.next();
                    for next in characters.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                // An operating-system command runs to BEL or ESC-backslash.
                Some(']') => {
                    characters.next();
                    let mut saw_escape = false;
                    for next in characters.by_ref() {
                        if next == '\u{7}' || (saw_escape && next == '\\') {
                            break;
                        }
                        saw_escape = next == '\u{1b}';
                    }
                }
                // A character-set selection takes one more byte.
                Some('(' | ')' | '*' | '+' | '-' | '.' | '/') => {
                    characters.next();
                    characters.next();
                }
                // A two-character escape.
                Some('@'..='_') => {
                    characters.next();
                }
                _ => {}
            }
            continue;
        }
        if character == '\n' || character == '\r' || character == '\t' || !character.is_control() {
            sanitized.push(character);
        }
    }
    sanitized
}

/// One command to run.
///
/// Build it with [`ExecRequest::new`] and struct-update syntax so a field
/// added later keeps compiling:
///
/// ```
/// use pebble::ExecRequest;
///
/// let request = ExecRequest {
///     timeout_ms: Some(30_000),
///     output_bytes_cap: Some(1024 * 1024),
///     ..ExecRequest::new("cargo test")
/// };
/// assert_eq!(request.command, "cargo test");
/// ```
///
/// Implementations should destructure it exhaustively, so a new field is a
/// compile error rather than input they silently ignore.
///
/// The type has no `Debug`: `env_vars` can carry credentials an application
/// injected for one call.
pub struct ExecRequest<'a> {
    /// Bash source, evaluated as `bash -c <command>`. See
    /// [`Environment::exec`] for the interpreter contract.
    pub command:          &'a str,
    /// How long the command may run. `None` means no limit.
    pub timeout_ms:       Option<u64>,
    /// The directory to run in. `None` means
    /// [`Environment::working_directory`].
    pub working_dir:      Option<&'a str>,
    /// Variables layered over whatever environment the implementation
    /// provides. An implementation may still refuse a variable its own policy
    /// forbids.
    pub env_vars:         Option<&'a HashMap<String, String>>,
    /// Cancels the command. The process is stopped and the partial output
    /// captured so far is returned with [`CommandTermination::Cancelled`].
    pub cancel_token:     Option<CancellationToken>,
    /// Bytes retained per stream — half a stable head, half a rolling tail.
    /// Draining always continues past the cap, so a noisy command never
    /// deadlocks against a full pipe. `None` retains everything.
    pub output_bytes_cap: Option<usize>,
}

impl<'a> ExecRequest<'a> {
    /// A request that runs `command` with no timeout, no cancellation, and no
    /// retention cap.
    #[must_use]
    pub const fn new(command: &'a str) -> Self {
        Self {
            command,
            timeout_ms: None,
            working_dir: None,
            env_vars: None,
            cancel_token: None,
            output_bytes_cap: None,
        }
    }
}

/// What one command produced, with the byte accounting the caller needs to
/// report honest totals.
#[derive(Debug, Clone)]
pub struct ExecOutcome {
    /// The command's output and status.
    pub result:            ExecResult,
    /// Whether [`ExecResult::stdout`] and [`ExecResult::stderr`] are really
    /// separate. When `false`, `stdout` carries the combined streams and
    /// `stderr` is empty.
    pub streams_separated: bool,
    /// Bytes standard output produced and how many survived the cap.
    pub stdout_capture:    OutputCaptureStats,
    /// Bytes standard error produced and how many survived the cap.
    pub stderr_capture:    OutputCaptureStats,
}

impl ExecOutcome {
    /// Both streams' capture counts added together.
    #[must_use]
    pub fn output_capture(&self) -> OutputCaptureStats {
        self.stdout_capture.combine(self.stderr_capture)
    }
}

/// One entry of a directory listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    /// The entry's path relative to the listed directory, `/`-joined when the
    /// listing recursed.
    pub name:   String,
    /// Whether the entry is a directory.
    pub is_dir: bool,
    /// The size in bytes, for regular files only.
    pub size:   Option<u64>,
}

/// Controls for [`Environment::grep`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GrepOptions {
    /// Restricts the search to paths matching this glob.
    pub glob_filter:      Option<String>,
    /// Matches without regard to case.
    pub case_insensitive: bool,
    /// Stops after this many matches.
    pub max_results:      Option<usize>,
}

/// Renders file content the way `read_file` shows it to the model.
///
/// Line numbers are 1-based and right-aligned to the width of the last
/// selected line number, and every line ends with a newline:
/// `"{number:>width$} | {line}\n"`. `offset` is the 1-based first line and
/// `limit` the maximum number of lines.
///
/// ```
/// use pebble::format_lines_numbered;
///
/// assert_eq!(
///     format_lines_numbered("hello\nworld", None, None),
///     "1 | hello\n2 | world\n"
/// );
/// ```
#[must_use]
pub fn format_lines_numbered(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    let all_lines: Vec<&str> = content.lines().collect();
    let skip = offset.unwrap_or(1).saturating_sub(1);
    let take = limit.unwrap_or(all_lines.len());
    let selected: Vec<&str> = all_lines.into_iter().skip(skip).take(take).collect();
    let width = (skip + selected.len()).to_string().len().max(1);
    let mut rendered = String::new();
    for (index, line) in selected.iter().enumerate() {
        let line_number = skip + index + 1;
        let _ = writeln!(rendered, "{line_number:>width$} | {line}");
    }
    rendered
}

/// The machine a session's tools work on.
///
/// Paths are strings the implementation interprets: [`LocalEnvironment`]
/// resolves a relative path against
/// [`working_directory`](Self::working_directory) and takes an absolute path as
/// given. Implementations are shared across concurrent tool calls, so every
/// method takes `&self` and must be safe to call from several tasks at once.
///
/// Errors are [`EnvironmentError`]; their messages reach the model, so they
/// name the operation and the path and never carry credentials.
///
/// Three methods have defaults built from the required ones — text reads,
/// numbered reads, and writes to a path the caller knows exists. Override them
/// only to do the same work more cheaply.
#[async_trait]
pub trait Environment: Send + Sync {
    /// The directory commands run in and relative paths resolve against.
    fn working_directory(&self) -> &str;

    /// The platform tag the system prompt shows: `"darwin"`, `"linux"`,
    /// `"windows"`, or `"unknown"`.
    fn platform(&self) -> &str;

    /// The operating system version for display, such as `"darwin 24.0.0"`.
    fn os_version(&self) -> String;

    /// Reads a file's bytes.
    async fn read_file_bytes(&self, path: &str) -> EnvResult<Vec<u8>>;

    /// Reads a file as UTF-8 text.
    ///
    /// Fails with [`EnvironmentErrorKind::InvalidUtf8`] when the bytes are not
    /// text.
    async fn read_file_text(&self, path: &str) -> EnvResult<String> {
        String::from_utf8(self.read_file_bytes(path).await?).map_err(|error| {
            EnvironmentError::with_source(
                EnvironmentErrorKind::InvalidUtf8,
                format!("File is not valid UTF-8: {path}"),
                error,
            )
        })
    }

    /// Reads a file as numbered lines, the shape the model reads.
    ///
    /// `offset` is the 1-based first line and `limit` the maximum number of
    /// lines. See [`format_lines_numbered`].
    async fn read_file(
        &self,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> EnvResult<String> {
        Ok(format_lines_numbered(
            &self.read_file_text(path).await?,
            offset,
            limit,
        ))
    }

    /// Writes a file, creating it and any missing parent directories and
    /// replacing any existing content.
    async fn write_file(&self, path: &str, content: &str) -> EnvResult<()>;

    /// Writes a file the caller has already confirmed exists.
    ///
    /// The default is [`write_file`](Self::write_file). An implementation
    /// overrides it to skip setup that only a new path needs; the observable
    /// result is the same.
    async fn write_existing_file(&self, path: &str, content: &str) -> EnvResult<()> {
        self.write_file(path, content).await
    }

    /// Deletes a file.
    async fn delete_file(&self, path: &str) -> EnvResult<()>;

    /// Whether a path exists.
    async fn file_exists(&self, path: &str) -> EnvResult<bool>;

    /// Lists a directory, descending `depth` levels (1 when absent).
    ///
    /// Entries are sorted by file name within each directory, and a nested
    /// entry's [`name`](DirEntry::name) is its `/`-joined path relative to the
    /// listed directory.
    async fn list_directory(&self, path: &str, depth: Option<usize>) -> EnvResult<Vec<DirEntry>>;

    /// Searches file contents for a regular expression.
    ///
    /// Returns one `"{path}:{line_number}:{line}"` string per match.
    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        options: &GrepOptions,
    ) -> EnvResult<Vec<String>>;

    /// Lists files matching a glob, relative to `path` or to
    /// [`working_directory`](Self::working_directory) when `path` is absent.
    ///
    /// Patterns are relative and use `/`: `*` and `?` stay inside one path
    /// segment, `**` crosses segments, and `[abc]` matches a character class.
    /// An absolute pattern, a `..` segment, or a backslash is
    /// [`EnvironmentErrorKind::InvalidInput`]. Results are the environment's
    /// own paths, sorted by the matched relative path.
    async fn glob(&self, pattern: &str, path: Option<&str>) -> EnvResult<Vec<String>>;

    /// Runs a command to completion and returns its bounded output.
    ///
    /// `request.command` is **Bash source**: it is evaluated as a non-login
    /// Bash program, equivalent to `bash -c <command>`. An implementation
    /// selects the interpreter, not its options — it must not add login mode,
    /// `errexit`, `pipefail`, or any other implicit shell option, must not
    /// wrap the command in redirections, and must never fall back to `sh`. A
    /// caller that wants other semantics writes them into the command itself
    /// (`sh -c ...`, an explicit `set -o pipefail`), which then runs beneath
    /// this Bash boundary.
    ///
    /// A timeout or a cancelled token stops the process and returns what it
    /// wrote, with [`CommandTermination::TimedOut`] or
    /// [`CommandTermination::Cancelled`] and no exit code — those are results,
    /// not errors. An `Err` means the command never ran or its output could
    /// not be collected.
    ///
    /// Output is always drained. With
    /// [`output_bytes_cap`](ExecRequest::output_bytes_cap) set, each stream
    /// keeps a stable head and a rolling tail within the cap and reports what
    /// it dropped in [`ExecOutcome::stdout_capture`] and
    /// [`ExecOutcome::stderr_capture`].
    async fn exec(&self, request: ExecRequest<'_>) -> EnvResult<ExecOutcome>;
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use super::mock::MockEnvironment;
    use super::*;
    use crate::redact::NoRedaction;

    fn exec_result(stdout: &str, stderr: &str) -> ExecResult {
        ExecResult {
            stdout:      stdout.to_owned(),
            stderr:      stderr.to_owned(),
            exit_code:   Some(0),
            termination: CommandTermination::Exited,
            duration_ms: 5,
        }
    }

    #[test]
    fn read_rendering_numbers_lines_from_one() {
        assert_eq!(
            format_lines_numbered("hello\nworld\nfoo", None, None),
            "1 | hello\n2 | world\n3 | foo\n"
        );
    }

    #[test]
    fn read_rendering_right_aligns_wider_line_numbers() {
        let content = (1..=12).fold(String::new(), |mut lines, number| {
            let _ = writeln!(lines, "line {number}");
            lines
        });

        let rendered = format_lines_numbered(content.trim_end(), None, None);

        assert!(rendered.starts_with(" 1 | line 1\n"), "{rendered}");
        assert!(rendered.contains("12 | line 12\n"), "{rendered}");
    }

    #[test]
    fn read_rendering_applies_offset_and_limit() {
        let rendered = format_lines_numbered("a\nb\nc\nd", Some(2), Some(2));

        assert_eq!(rendered, "2 | b\n3 | c\n");
    }

    #[test]
    fn read_rendering_of_empty_content_is_empty() {
        assert_eq!(format_lines_numbered("", None, None), "");
    }

    #[test]
    fn success_requires_a_zero_exit_and_a_normal_end() {
        assert!(exec_result("", "").is_success());
        assert!(
            !ExecResult {
                exit_code: Some(1),
                ..exec_result("", "")
            }
            .is_success()
        );
        assert!(
            !ExecResult {
                exit_code: None,
                termination: CommandTermination::TimedOut,
                ..exec_result("", "")
            }
            .is_success()
        );
    }

    #[test]
    fn a_missing_exit_code_displays_as_minus_one() {
        assert_eq!(
            ExecResult {
                exit_code: None,
                ..exec_result("", "")
            }
            .display_exit_code(),
            -1
        );
        assert_eq!(exec_result("", "").display_exit_code(), 0);
    }

    #[test]
    fn an_error_renders_its_causes_under_the_message() {
        let inner = EnvironmentError::new(EnvironmentErrorKind::Io, "disk went away");
        let error = EnvironmentError::with_source(
            EnvironmentErrorKind::Io,
            "Failed to write /work/out.txt",
            inner,
        );

        assert_eq!(error.kind(), EnvironmentErrorKind::Io);
        assert_eq!(error.message(), "Failed to write /work/out.txt");
        assert_eq!(error.to_string(), "Failed to write /work/out.txt");
        assert_eq!(
            error.detail(),
            "Failed to write /work/out.txt\n  caused by: disk went away"
        );
    }

    #[test]
    fn an_error_without_causes_renders_only_its_message() {
        let error = EnvironmentError::new(EnvironmentErrorKind::NotFound, "File not found: a.txt");

        assert_eq!(error.detail(), "File not found: a.txt");
    }

    #[test]
    fn an_io_error_classifies_a_missing_path_as_not_found() {
        let missing = io::Error::new(ErrorKind::NotFound, "no such file");
        let refused = io::Error::new(ErrorKind::PermissionDenied, "nope");

        assert_eq!(
            EnvironmentError::io("Failed to read a.txt", missing).kind(),
            EnvironmentErrorKind::NotFound
        );
        assert_eq!(
            EnvironmentError::io("Failed to read a.txt", refused).kind(),
            EnvironmentErrorKind::Io
        );
    }

    #[test]
    fn an_output_tail_keeps_the_end_of_each_stream() {
        let result = exec_result("0123456789", "err");

        let tail = result
            .output_tail(&NoRedaction, 4)
            .expect("output was captured");

        assert_eq!(tail.stdout.as_deref(), Some("6789"));
        assert!(tail.stdout_truncated);
        assert_eq!(tail.stderr.as_deref(), Some("err"));
        assert!(!tail.stderr_truncated);
    }

    #[test]
    fn an_output_tail_cuts_at_a_character_boundary() {
        let result = exec_result("aa😀😀zz", "");

        let tail = result
            .output_tail(&NoRedaction, 5)
            .expect("output was captured");

        let stdout = tail.stdout.expect("stdout tail");
        assert!(stdout.ends_with("zz"), "{stdout}");
        assert!(stdout.starts_with('😀'), "{stdout}");
    }

    #[test]
    fn an_output_tail_drops_terminal_escapes_and_control_characters() {
        let result = exec_result("\u{1b}[31mred\u{1b}[0m\u{7}\ttab\nline", "");

        let tail = result
            .output_tail(&NoRedaction, DEFAULT_EXEC_OUTPUT_TAIL_BYTES)
            .expect("output was captured");

        assert_eq!(tail.stdout.as_deref(), Some("red\ttab\nline"));
    }

    #[test]
    fn an_output_tail_runs_the_redactor_before_it_bounds_the_text() {
        struct DropSecrets;

        impl Redactor for DropSecrets {
            fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
                Cow::Owned(text.replace("s3cret", "[REDACTED]"))
            }
        }

        let result = exec_result("token s3cret", "s3cret");

        let tail = result
            .output_tail(&DropSecrets, DEFAULT_EXEC_OUTPUT_TAIL_BYTES)
            .expect("output was captured");

        assert_eq!(tail.stdout.as_deref(), Some("token [REDACTED]"));
        assert_eq!(tail.stderr.as_deref(), Some("[REDACTED]"));
    }

    /// The order matters when the secret is longer than the budget: bounding
    /// first would leave a fragment the redactor no longer recognizes.
    #[test]
    fn a_secret_longer_than_the_budget_is_still_redacted_whole() {
        const SECRET: &str = "sk-live-xK9mZ2vL8nQ5rT1wY4bC7dF0gH3jE6pA";

        struct DropSecrets;

        impl Redactor for DropSecrets {
            fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
                Cow::Owned(text.replace(SECRET, "[REDACTED]"))
            }
        }

        let stdout = format!("{}{SECRET} done", "context ".repeat(20));
        let result = exec_result(&stdout, "");

        let tail = result
            .output_tail(&DropSecrets, 32)
            .expect("output was captured");

        let stdout = tail.stdout.expect("stdout tail");
        assert!(stdout.contains("[REDACTED]"), "{stdout}");
        assert!(!stdout.contains("F0gH3jE6pA"), "{stdout}");
        assert!(tail.stdout_truncated);
    }

    #[test]
    fn a_silent_process_has_no_output_tail() {
        assert!(exec_result("", "").output_tail(&NoRedaction, 16).is_none());
    }

    #[test]
    fn a_zero_byte_budget_keeps_no_tail() {
        // Nothing is retained, so there is nothing to put on an event.
        assert!(
            exec_result("output", "")
                .output_tail(&NoRedaction, 0)
                .is_none()
        );
    }

    #[test]
    fn combined_capture_adds_both_streams() {
        let outcome = ExecOutcome {
            result:            exec_result("", ""),
            streams_separated: true,
            stdout_capture:    OutputCaptureStats {
                observed_bytes: 10,
                retained_bytes: 4,
                omitted_bytes:  6,
            },
            stderr_capture:    OutputCaptureStats::complete(3),
        };

        assert_eq!(outcome.output_capture(), OutputCaptureStats {
            observed_bytes: 13,
            retained_bytes: 7,
            omitted_bytes:  6,
        });
    }

    #[tokio::test]
    async fn an_environment_is_usable_as_a_shared_trait_object() {
        // Tool execution holds one `Arc<dyn Environment>` across concurrent
        // calls, so the trait has to stay object-safe and shareable.
        let environment: Arc<dyn Environment> = Arc::new(MockEnvironment::linux());

        assert_eq!(environment.working_directory(), "/home/test");
        assert!(
            environment
                .exec(ExecRequest::new("echo hello"))
                .await
                .expect("the mock answers")
                .result
                .is_success()
        );
    }

    #[test]
    fn a_request_defaults_everything_but_its_command() {
        let request = ExecRequest::new("echo hello");

        assert_eq!(request.command, "echo hello");
        assert_eq!(request.timeout_ms, None);
        assert_eq!(request.working_dir, None);
        assert!(request.env_vars.is_none());
        assert!(request.cancel_token.is_none());
        assert_eq!(request.output_bytes_cap, None);
    }
}
