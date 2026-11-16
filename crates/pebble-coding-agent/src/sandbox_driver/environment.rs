//! A sandbox-driver handle as the [`Environment`] a session runs in.
//!
//! Pebble's tools speak the `Environment` contract; the driver's new API
//! speaks files (`get`, `put`, `list`), ports, and exec. [`SandboxEnvironment`]
//! is the mapping between the two, and nothing else: every path resolves
//! against the working directory the application named (a relative path
//! against it, which may sit below the provider's own), every command runs
//! through [`SandboxExec`] with its exec policy, and every failure keeps its
//! driver cause.
//!
//! Where the two contracts differ, pebble's wins here because the model reads
//! pebble's: a glob that pebble rejects is rejected before anything runs, a
//! directory listing is in tree order, and a command with no retention cap
//! still drains under the policy's default buffer rather than without bound.
//! Output a provider lost on its own transport
//! ([`ExecStreamingResult::output_loss`]) has no slot in pebble's contract, so
//! it is written where the model already reads: one line at the end of
//! stderr.
//!
//! Parity gaps: the new API has no stat, exists, remove, rename or mkdir
//! (parity plan B2), no search (B3), and no platform information. The adapter
//! runs those as commands in the sandbox: `[ -e ]`, `rm`/`rmdir`, `mkdir -p`
//! and `mv`, the derived search in [`super::search`], and `uname -sr` once at
//! [`SandboxEnvironment::attach`]. So file existence, moves and deletes now
//! need a POSIX shell in the sandbox, which reads and writes do not.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
#[cfg(feature = "mcp")]
use sandbox_driver::PortSupport;
use sandbox_driver::{
    Error as DriverError, ExecControls, ExecSpec, FileKind, OutputLoss, OutputSink, OutputStream,
    Sandbox,
};
use tokio::io::AsyncReadExt as _;
use tokio::sync::OnceCell;
use tracing::warn;

use super::exec::{
    CommandFailure, ExecResult as DriverExecResult, ExecStreamingResult, SandboxExec,
    command_termination, program_exit_code,
};
use super::path::{join_sandbox_path, resolve_path, shell_quote};
use super::search;
use crate::char_boundary::floor_char_boundary;
use crate::environment::support::{compile_glob, tree_order};
use crate::environment::{
    DirEntry, EnvResult, Environment, EnvironmentError, EnvironmentErrorKind, ExecOutcome,
    ExecOutputSink, ExecOutputStream, ExecRequest, ExecResult, GrepOptions,
};
#[cfg(feature = "mcp")]
use crate::mcp::PortRoutes;

/// Deadline for the commands the adapter runs on its own behalf (platform,
/// stat, move, delete).
const FS_TIMEOUT: Duration = Duration::from_secs(60);

/// A sandbox-driver handle working in one directory, as pebble's
/// [`Environment`].
///
/// The handle is a sandbox the application brought to `Running`. The
/// working directory is the session's, which may sit below the handle's
/// own.
pub struct SandboxEnvironment {
    handle:      Arc<dyn Sandbox>,
    working_dir: String,
    platform:    String,
    os_version:  String,
    /// Whether `rg` is in the sandbox, probed on the first search.
    ripgrep:     OnceCell<bool>,
}

impl SandboxEnvironment {
    /// Wraps a running `handle` working in `working_dir`, asking the sandbox
    /// for its platform once.
    ///
    /// Parity gap: the old driver's `platform_info` is not in the new API, so
    /// this runs `uname -sr` in the sandbox: `Linux 6.1.0` becomes platform
    /// `linux` and version `linux 6.1.0`, `Darwin 24.0.0` becomes `darwin`.
    ///
    /// # Errors
    /// The command could not run, or `uname` failed.
    pub async fn attach(
        handle: Arc<dyn Sandbox>,
        working_dir: impl Into<String>,
    ) -> EnvResult<Self> {
        let working_dir = working_dir.into();
        let result = run_fs_command(
            &SandboxExec::new(&*handle).with_working_dir(working_dir.clone()),
            "uname -sr",
            "Failed to read the sandbox platform",
            "uname",
        )
        .await?;
        let uname = result.stdout_lossy();
        let mut words = uname.split_whitespace();
        let platform = words.next().unwrap_or_default().to_ascii_lowercase();
        let version = words.next().unwrap_or_default();
        let os_version = if version.is_empty() {
            platform.clone()
        } else {
            format!("{platform} {version}")
        };
        Ok(Self::with_platform(
            handle,
            working_dir,
            platform,
            os_version,
        ))
    }

    /// Wraps `handle` with a platform already known, so no round trip to
    /// the sandbox is needed before pebble reads it.
    #[must_use]
    pub fn with_platform(
        handle: Arc<dyn Sandbox>,
        working_dir: impl Into<String>,
        platform: impl Into<String>,
        os_version: impl Into<String>,
    ) -> Self {
        Self {
            handle,
            working_dir: working_dir.into(),
            platform: platform.into(),
            os_version: os_version.into(),
            ripgrep: OnceCell::new(),
        }
    }

    /// The driver handle underneath, for what pebble's contract does not
    /// carry.
    #[must_use]
    pub fn handle(&self) -> &Arc<dyn Sandbox> {
        &self.handle
    }

    /// The directory the agent works in.
    #[must_use]
    pub fn working_directory(&self) -> &str {
        &self.working_dir
    }

    /// The exec policy over the handle, working in the agent's directory.
    #[must_use]
    pub fn exec(&self) -> SandboxExec<'_> {
        SandboxExec::new(&*self.handle).with_working_dir(self.working_dir.clone())
    }

    /// Pebble's port routes over the handle's ports, given what the
    /// provider's descriptor says it can expose; see
    /// [`port_routes`](super::port_routes).
    #[cfg(feature = "mcp")]
    #[must_use]
    pub fn port_routes(&self, support: &PortSupport) -> Option<Arc<dyn PortRoutes>> {
        super::ports::port_routes(&self.handle, support)
    }

    /// A caller path as the driver will see it.
    fn resolve(&self, path: &str) -> String {
        resolve_path(path, &self.working_dir)
    }

    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        let mut spec = ExecSpec::shell(format!("[ -e {} ]", shell_quote(&self.resolve(path))));
        spec.timeout_ms = Some(duration_ms(FS_TIMEOUT));
        let result = self
            .exec()
            .run_streaming(spec, ExecControls::default(), None)
            .await
            .map_err(|error| environment_error(&format!("Failed to stat {path}"), error))?
            .result;
        match result.exit_code {
            Some(0) => Ok(true),
            Some(1) if result.termination == sandbox_driver::Termination::Exited => Ok(false),
            _ => Err(command_error(
                &format!("Failed to stat {path}"),
                CommandFailure::new("stat", result),
            )),
        }
    }

    async fn ripgrep(&self) -> bool {
        *self
            .ripgrep
            .get_or_init(|| async { search::ripgrep_available(&self.exec()).await })
            .await
    }

    /// The traversal base a walk starts from. A base at the working
    /// directory walks relative to it so every path component of
    /// `relative_start` is checked against symlinks; any other base is
    /// walked as given.
    fn walk_base(&self, base: &str, relative_start: &str) -> String {
        if base == self.working_dir || base.is_empty() || base == "." {
            if relative_start.is_empty() {
                ".".to_string()
            } else {
                relative_start.to_string()
            }
        } else {
            join_sandbox_path(&self.resolve(base), relative_start)
        }
    }
}

#[async_trait]
impl Environment for SandboxEnvironment {
    fn working_directory(&self) -> &str {
        &self.working_dir
    }

    fn platform(&self) -> &str {
        &self.platform
    }

    fn os_version(&self) -> String {
        self.os_version.clone()
    }

    async fn read_file_bytes(&self, path: &str) -> EnvResult<Vec<u8>> {
        let message = format!("Failed to read {path}");
        let mut stream = self
            .handle
            .files()
            .get(&self.resolve(path), None)
            .await
            .map_err(|error| environment_error(&message, error))?;
        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes).await.map_err(|error| {
            EnvironmentError::with_source(EnvironmentErrorKind::Io, message, error)
        })?;
        Ok(bytes)
    }

    async fn write_file(&self, path: &str, content: &str) -> EnvResult<()> {
        self.handle
            .files()
            .put(
                &self.resolve(path),
                super::exec::stdin_bytes(content.as_bytes().to_vec()),
            )
            .await
            .map_err(|error| environment_error(&format!("Failed to write {path}"), error))
    }

    async fn rename_file(&self, source: &str, destination: &str) -> EnvResult<()> {
        let resolved_source = self.resolve(source);
        let resolved_destination = self.resolve(destination);
        if !self.file_exists(source).await? {
            return Err(EnvironmentError::new(
                EnvironmentErrorKind::NotFound,
                format!("Failed to move {source}: file does not exist"),
            ));
        }
        // The same path spelled twice is a move to itself, which must leave
        // the file where it is. Aliases the sandbox's own filesystem would
        // resolve (a symlinked parent, a hard link) are not checked: there is
        // no remote `realpath`, and `mv a a` is a no-op anyway.
        if normalize(&resolved_source) == normalize(&resolved_destination) {
            return Ok(());
        }
        // The destination's parent is created first, and a parent that is a
        // file fails there, before anything has moved, so the source stays
        // intact as the contract requires.
        let quoted_destination = shell_quote(&resolved_destination);
        let mkdir = parent_directory(&resolved_destination)
            .map(|parent| format!("mkdir -p -- {} && ", shell_quote(parent)))
            .unwrap_or_default();
        run_fs_command(
            &self.exec(),
            &format!(
                "{mkdir}mv -f -- {} {quoted_destination}",
                shell_quote(&resolved_source)
            ),
            &format!("Failed to move {source} to {destination}"),
            "move",
        )
        .await
        .map(drop)
    }

    async fn delete_file(&self, path: &str) -> EnvResult<()> {
        // Pebble's delete is a `remove_file`, which reports a path that is
        // not there.
        if !self.file_exists(path).await? {
            return Err(EnvironmentError::new(
                EnvironmentErrorKind::NotFound,
                format!("Failed to delete {path}: file does not exist"),
            ));
        }
        let quoted = shell_quote(&self.resolve(path));
        run_fs_command(
            &self.exec(),
            &format!("if [ -d {quoted} ]; then rmdir -- {quoted}; else rm -f -- {quoted}; fi"),
            &format!("Failed to delete {path}"),
            "delete",
        )
        .await
        .map(drop)
    }

    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        Self::file_exists(self, path).await
    }

    async fn list_directory(&self, path: &str, depth: Option<usize>) -> EnvResult<Vec<DirEntry>> {
        let resolved = self.resolve(path);
        // Pebble's depth counts the listed level as 1; the driver's counts
        // levels below it, so 0 is the directory's own children.
        let driver_depth = u32::try_from(depth.unwrap_or(1).saturating_sub(1)).unwrap_or(u32::MAX);
        let prefix = format!("{}/", resolved.trim_end_matches('/'));
        let mut entries: Vec<DirEntry> = self
            .handle
            .files()
            .list(&resolved, driver_depth)
            .await
            .map_err(|error| environment_error(&format!("Failed to list {path}"), error))?
            .into_iter()
            .map(|entry| DirEntry {
                is_dir: entry.kind == FileKind::Directory,
                size:   (entry.kind == FileKind::File).then_some(entry.size),
                // The driver names an entry by the listed path joined with
                // the entry's own; pebble names it relative to the listing.
                name:   entry
                    .path
                    .strip_prefix(&prefix)
                    .map_or_else(|| entry.path.clone(), str::to_string),
            })
            .collect();
        // Pebble lists in tree order, where `foo/x` comes before `foo-bar`.
        tree_order(&mut entries);
        Ok(entries)
    }

    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        options: &GrepOptions,
    ) -> EnvResult<Vec<String>> {
        let matches = search::grep(
            &self.exec(),
            self.ripgrep().await,
            pattern,
            &self.resolve(path),
            options,
        )
        .await?;
        Ok(matches
            .into_iter()
            .map(|(path, line_number, line)| format!("{path}:{line_number}:{line}"))
            .collect())
    }

    async fn glob(&self, pattern: &str, path: Option<&str>) -> EnvResult<Vec<String>> {
        // Compiled by pebble's own grammar before anything runs, so the
        // reason reaches the model in pebble's words and the matching is
        // pebble's wherever the files are.
        let glob = compile_glob(pattern)?;
        let base = path.unwrap_or(&self.working_dir);
        let relative_start = glob.traversal_root();
        let walked = search::walk(&self.exec(), &self.walk_base(base, relative_start)).await?;
        let mut relative_paths: Vec<String> = walked
            .into_iter()
            .map(|file| join_sandbox_path(relative_start, &file))
            .filter(|relative_path| glob.is_match(relative_path))
            .collect();
        relative_paths.sort();
        Ok(relative_paths
            .into_iter()
            .map(|relative_path| join_sandbox_path(base, &relative_path))
            .collect())
    }

    async fn exec(&self, request: ExecRequest<'_>) -> EnvResult<ExecOutcome> {
        let ExecRequest {
            command,
            timeout_ms,
            working_dir,
            env_vars,
            cancel_token,
            output_bytes_cap,
            output_sink,
        } = request;
        let mut spec = ExecSpec::shell(command);
        spec.timeout_ms = timeout_ms;
        spec.working_directory = working_dir.map(str::to_string);
        for (key, value) in env_vars.into_iter().flatten() {
            spec.env.insert(key.clone(), value.clone());
        }
        let controls = ExecControls {
            term: cancel_token,
            sink: output_sink.map(adapt_output_sink),
            ..ExecControls::default()
        };
        // `None` asks pebble for no cap at all. The exec policy fills its
        // default buffer when the cap is unset, so a command with no cap
        // drains under that default rather than without bound; the capture
        // counts still say what was dropped.
        let streaming = self
            .exec()
            .run_streaming(spec, controls, output_bytes_cap)
            .await
            .map_err(|error| {
                let kind = match &error {
                    DriverError::Transport { .. }
                    | DriverError::Unavailable { .. }
                    | DriverError::Stale { .. } => EnvironmentErrorKind::Io,
                    DriverError::Unsupported { .. } => EnvironmentErrorKind::Unsupported,
                    _ => EnvironmentErrorKind::Spawn,
                };
                EnvironmentError::with_source(kind, "Failed to run the command", error)
            })?;
        Ok(exec_outcome(streaming, program_name(command)))
    }
}

/// Runs one of the adapter's own commands, failing on anything but success.
async fn run_fs_command(
    exec: &SandboxExec<'_>,
    command: &str,
    message: &str,
    label: &str,
) -> EnvResult<DriverExecResult> {
    let mut spec = ExecSpec::shell(command);
    spec.timeout_ms = Some(duration_ms(FS_TIMEOUT));
    let result = exec
        .run_streaming(spec, ExecControls::default(), None)
        .await
        .map_err(|error| environment_error(message, error))?
        .result;
    if result.success() {
        Ok(result)
    } else {
        Err(command_error(message, CommandFailure::new(label, result)))
    }
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Pebble's outcome for a finished command: the policy's result read the way
/// pebble reads it, plus the provider's own output loss written where the
/// model reads stderr.
///
/// A provider whose transport tore (Daytona's text-only toolbox) completes
/// the command and reports what it discarded in
/// [`ExecStreamingResult::output_loss`] rather than failing it. The frames
/// are gone, the stream they belonged to is unknown, and the counts are of
/// encoded bytes, so they cannot be folded into either stream's capture
/// accounting without guessing; the loss is one line at the end of stderr,
/// where the model and the run log see it, and one log event for the
/// operator.
fn exec_outcome(streaming: ExecStreamingResult, program: &str) -> ExecOutcome {
    let loss = streaming.output_loss;
    let result = streaming.result;
    let mut stderr = result.stderr_lossy();
    if loss.is_lossy() {
        warn!(
            program = %program,
            dropped_frames = loss.dropped_frames,
            dropped_bytes = loss.dropped_bytes,
            "Sandbox provider dropped command output"
        );
        if !stderr.is_empty() && !stderr.ends_with('\n') {
            stderr.push('\n');
        }
        stderr.push_str(&output_loss_line(loss));
    }
    ExecOutcome {
        result:            ExecResult {
            stdout: result.stdout_lossy(),
            stderr,
            exit_code: program_exit_code(result.termination, result.exit_code),
            termination: command_termination(result.termination),
            duration_ms: result.duration_ms(),
        },
        streams_separated: streaming.streams_separated,
        stdout_capture:    streaming.stdout_capture,
        stderr_capture:    streaming.stderr_capture,
    }
}

/// The line stderr ends with when the provider dropped output.
fn output_loss_line(loss: OutputLoss) -> String {
    format!(
        "[sandbox] {} output frame(s), {} bytes dropped by the provider\n",
        loss.dropped_frames, loss.dropped_bytes
    )
}

/// Bytes of a command's first word a log event carries.
const PROGRAM_NAME_BYTES: usize = 64;

/// The word a command starts with, bounded, for a log event that must not
/// carry the command itself.
fn program_name(command: &str) -> &str {
    let word = command.split_whitespace().next().unwrap_or_default();
    &word[..floor_char_boundary(word, PROGRAM_NAME_BYTES)]
}

/// A path with its redundant separators and `.` segments removed, for
/// deciding whether two spellings name the same file.
fn normalize(path: &str) -> String {
    let absolute = path.starts_with('/');
    let joined = path
        .split('/')
        .filter(|segment| !segment.is_empty() && *segment != ".")
        .collect::<Vec<_>>()
        .join("/");
    if absolute {
        format!("/{joined}")
    } else {
        joined
    }
}

/// The directory a path is in, when the path names one.
fn parent_directory(path: &str) -> Option<&str> {
    let trimmed = path.trim_end_matches('/');
    let (parent, _) = trimmed.rsplit_once('/')?;
    if parent.is_empty() {
        return Some("/");
    }
    Some(parent)
}

/// Feeds the driver's asynchronous chunk callback into pebble's synchronous
/// sink. A terminal stream, from a provider that cannot separate the two,
/// is stdout's.
fn adapt_output_sink(sink: ExecOutputSink) -> OutputSink {
    Arc::new(move |stream, chunk: Vec<u8>| {
        let stream = match stream {
            OutputStream::Stderr => ExecOutputStream::Stderr,
            OutputStream::Stdout | OutputStream::Terminal => ExecOutputStream::Stdout,
        };
        sink(stream, &chunk);
        Box::pin(async { Ok(()) })
    })
}

/// A sandbox failure as pebble classifies it, keeping the driver cause.
pub(super) fn environment_error(message: &str, error: DriverError) -> EnvironmentError {
    let kind = match &error {
        DriverError::NotFound { .. } => EnvironmentErrorKind::NotFound,
        DriverError::Unsupported { .. } => EnvironmentErrorKind::Unsupported,
        _ => EnvironmentErrorKind::Io,
    };
    EnvironmentError::with_source(kind, message, error)
}

/// One of the adapter's own commands that ran and failed.
fn command_error(message: &str, failure: CommandFailure) -> EnvironmentError {
    EnvironmentError::with_source(EnvironmentErrorKind::Io, message, failure)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::env::consts::OS;
    use std::path::{Path, PathBuf};

    use sandbox_driver::{ExecOutcome, Files, Ports, SandboxRef, Sandboxes, Status, Termination};
    use tokio::fs;

    use super::*;
    use crate::redact::NoRedaction;
    use crate::sandbox_driver::display_for_log;
    use crate::sandbox_driver::test_support::{MockSandbox, ScriptedSandbox, exec_result};
    use crate::sandbox_driver::testing::host_sandbox;
    use crate::test_support::EnvironmentContract;

    /// The environment over the driver's built-in Host provider.
    async fn host_environment() -> (Sandboxes, SandboxEnvironment) {
        let (catalog, handle) = host_sandbox().await;
        let working_dir = handle.working_directory().display().to_string();
        let sandbox = SandboxEnvironment::attach(handle, working_dir)
            .await
            .expect("the host platform");
        (catalog, sandbox)
    }

    #[tokio::test]
    async fn host_files_satisfy_pebbles_environment_contract() {
        let (_catalog, sandbox) = host_environment().await;
        EnvironmentContract::new(&sandbox, "contract")
            .verify_files()
            .await
            .expect("file contract");
    }

    #[tokio::test]
    async fn host_search_satisfies_pebbles_environment_contract() {
        let (_catalog, sandbox) = host_environment().await;
        EnvironmentContract::new(&sandbox, "contract")
            .verify_search()
            .await
            .expect("search contract");
    }

    #[tokio::test]
    async fn host_commands_satisfy_pebbles_environment_contract() {
        let (_catalog, sandbox) = host_environment().await;
        EnvironmentContract::new(&sandbox, "contract")
            .verify_commands()
            .await
            .expect("command contract");
    }

    #[tokio::test]
    async fn the_platform_is_learned_from_the_sandbox() {
        let (_catalog, sandbox) = host_environment().await;
        let expected = if cfg!(target_os = "macos") {
            "darwin"
        } else {
            OS
        };
        assert_eq!(Environment::platform(&sandbox), expected);
        assert!(sandbox.os_version().starts_with(&format!("{expected} ")));
    }

    #[tokio::test]
    async fn a_directory_listing_is_in_tree_order_and_relative() {
        let (_catalog, sandbox) = host_environment().await;
        for name in ["foo/x.txt", "foo-bar/y.txt", "foo.txt"] {
            Environment::write_file(&sandbox, name, "content")
                .await
                .expect("fixture");
        }
        let entries = Environment::list_directory(&sandbox, ".", Some(2))
            .await
            .expect("listing");
        let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, [
            "foo",
            "foo/x.txt",
            "foo-bar",
            "foo-bar/y.txt",
            "foo.txt"
        ]);
        assert_eq!(entries[0].size, None);
        assert_eq!(entries[1].size, Some(7));
        let top: Vec<String> = Environment::list_directory(&sandbox, "foo", None)
            .await
            .unwrap()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(top, ["x.txt"]);
    }

    #[tokio::test]
    async fn glob_reports_paths_under_the_declared_base_and_skips_symlinks() {
        let (_catalog, sandbox) = host_environment().await;
        let root = PathBuf::from(sandbox.working_directory());
        fs::create_dir_all(root.join(".ai/reports")).await.unwrap();
        fs::create_dir_all(root.join(".ai/target")).await.unwrap();
        fs::write(root.join(".ai/reports/result.md"), "report")
            .await
            .unwrap();
        fs::write(root.join(".ai/reports/empty.md"), "")
            .await
            .unwrap();
        fs::write(root.join(".ai/target/ignored.md"), "ignored")
            .await
            .unwrap();

        let working_dir = sandbox.working_directory().to_string();
        let globbed = Environment::glob(&sandbox, "**/*.md", None).await.unwrap();
        assert_eq!(globbed, vec![
            format!("{working_dir}/.ai/reports/empty.md"),
            format!("{working_dir}/.ai/reports/result.md"),
            format!("{working_dir}/.ai/target/ignored.md"),
        ]);
        let scoped = Environment::glob(&sandbox, "*.md", Some(".ai/reports"))
            .await
            .unwrap();
        assert_eq!(scoped.len(), 2);

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let target = root.join("elsewhere");
            fs::create_dir_all(&target).await.unwrap();
            fs::write(target.join("lib.rs"), "").await.unwrap();
            symlink(&target, root.join("linked")).unwrap();
            let results = Environment::glob(&sandbox, "linked/**/*.rs", None)
                .await
                .unwrap();
            assert!(results.is_empty(), "{results:?}");
        }
    }

    #[tokio::test]
    async fn a_glob_pebble_refuses_never_reaches_the_sandbox() {
        let mock = MockSandbox::linux();
        let error = Environment::glob(&*mock.sandbox(), "docs/", None)
            .await
            .expect_err("a trailing slash names no file");
        assert_eq!(error.kind(), EnvironmentErrorKind::InvalidInput);
        assert!(mock.driver().scripted_exec().recorded().is_empty());
    }

    #[tokio::test]
    async fn grep_returns_path_line_content_triples() {
        let (_catalog, sandbox) = host_environment().await;
        fs::write(
            Path::new(sandbox.working_directory()).join("test.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .await
        .unwrap();
        let results = Environment::grep(&sandbox, "println", "test.rs", &GrepOptions::default())
            .await
            .unwrap();
        let working_dir = sandbox.working_directory();
        assert_eq!(results, [format!(
            "{working_dir}/test.rs:2:    println!(\"hello\");"
        )]);
        let none = Environment::grep(&sandbox, "absent", "test.rs", &GrepOptions::default())
            .await
            .unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn a_failed_adapter_command_keeps_its_output_for_the_log() {
        let mock = MockSandbox {
            exec_result: exec_result("", "mv: permission denied", Some(1), Termination::Exited, 0),
            files: HashMap::from([("a.txt".to_string(), "x".to_string())]),
            ..MockSandbox::linux()
        };
        let error = Environment::rename_file(&*mock.sandbox(), "a.txt", "b/a.txt")
            .await
            .expect_err("the move failed");
        let rendered = display_for_log(&error, &NoRedaction);
        assert!(rendered.contains("mv: permission denied"), "{rendered}");
    }

    #[test]
    fn a_path_spelled_two_ways_is_one_path() {
        assert_eq!(normalize("/work//a/./b.txt"), "/work/a/b.txt");
        assert_eq!(parent_directory("/work/a/b.txt"), Some("/work/a"));
        assert_eq!(parent_directory("/b.txt"), Some("/"));
        assert_eq!(parent_directory("b.txt"), None);
    }

    fn request(command: &str) -> ExecRequest<'_> {
        ExecRequest {
            timeout_ms: Some(10_000),
            ..ExecRequest::new(command)
        }
    }

    fn output_loss(dropped_frames: u64, dropped_bytes: u64) -> OutputLoss {
        let mut loss = OutputLoss::default();
        loss.dropped_frames = dropped_frames;
        loss.dropped_bytes = dropped_bytes;
        loss
    }

    #[tokio::test]
    async fn a_lossless_command_hands_back_stderr_as_the_provider_wrote_it() {
        let mock = MockSandbox {
            exec_result: exec_result(
                "built\n",
                "warning: unused\n",
                Some(0),
                Termination::Exited,
                7,
            ),
            ..MockSandbox::linux()
        };
        let outcome = Environment::exec(&*mock.sandbox(), request("cargo build"))
            .await
            .expect("a scripted command");
        assert_eq!(outcome.result.stdout, "built\n");
        assert_eq!(outcome.result.stderr, "warning: unused\n");
        assert_eq!(outcome.result.exit_code, Some(0));
        assert_eq!(
            outcome.stderr_capture.observed_bytes,
            "warning: unused\n".len()
        );
        // The command ran in the mock's working directory, and the deadline
        // stayed with the policy rather than the provider.
        let spec = mock.driver().scripted_exec().recorded().pop().unwrap();
        assert_eq!(spec.working_directory.as_deref(), Some("/home/test"));
        assert_eq!(spec.timeout_ms, None);
    }

    #[test]
    fn a_provider_output_loss_ends_stderr_with_one_line() {
        let mut streaming = ExecStreamingResult::new(exec_result(
            "built\n",
            "warning: torn",
            Some(1),
            Termination::Exited,
            7,
        ));
        streaming.output_loss = output_loss(2, 4096);

        let outcome = exec_outcome(streaming, "cargo");

        assert_eq!(outcome.result.stdout, "built\n");
        assert_eq!(
            outcome.result.stderr,
            "warning: torn\n[sandbox] 2 output frame(s), 4096 bytes dropped by the provider\n"
        );
        assert_eq!(outcome.result.exit_code, Some(1));
        assert_eq!(outcome.result.duration_ms, 7);
        // The loss is not folded into either stream's accounting.
        assert_eq!(outcome.stdout_capture.observed_bytes, "built\n".len());
        assert_eq!(outcome.stderr_capture.observed_bytes, "warning: torn".len());
    }

    #[test]
    fn a_provider_output_loss_with_no_stderr_is_the_line_alone() {
        let mut streaming =
            ExecStreamingResult::new(exec_result("", "", Some(0), Termination::Exited, 1));
        streaming.output_loss = output_loss(1, 80);
        let outcome = exec_outcome(streaming, "sh");
        assert_eq!(
            outcome.result.stderr,
            "[sandbox] 1 output frame(s), 80 bytes dropped by the provider\n"
        );
    }

    #[test]
    fn a_log_event_names_the_first_word_of_a_command_bounded() {
        assert_eq!(program_name("cargo build --release"), "cargo");
        assert_eq!(program_name("  \n  ls"), "ls");
        assert_eq!(program_name(""), "");
        let long = "x".repeat(PROGRAM_NAME_BYTES + 10);
        assert_eq!(program_name(&long).len(), PROGRAM_NAME_BYTES);
        let multibyte = "é".repeat(PROGRAM_NAME_BYTES);
        assert!(program_name(&multibyte).len() <= PROGRAM_NAME_BYTES);
    }

    /// A handle whose every command reports a provider output loss, as
    /// Daytona does after a torn frame.
    struct LossySandbox {
        inner: Arc<ScriptedSandbox>,
        loss:  OutputLoss,
    }

    #[async_trait]
    impl Sandbox for LossySandbox {
        fn reference(&self) -> SandboxRef {
            self.inner.reference()
        }

        fn working_directory(&self) -> &Path {
            self.inner.working_directory()
        }

        fn env(&self) -> &BTreeMap<String, String> {
            self.inner.env()
        }

        async fn status(&self) -> Result<Status, DriverError> {
            self.inner.status().await
        }

        async fn start(&self) -> Result<(), DriverError> {
            self.inner.start().await
        }

        async fn stop(&self) -> Result<(), DriverError> {
            self.inner.stop().await
        }

        async fn exec(
            &self,
            spec: ExecSpec,
            controls: ExecControls,
        ) -> Result<ExecOutcome, DriverError> {
            let mut outcome = self.inner.exec(spec, controls).await?;
            outcome.output_loss = self.loss;
            Ok(outcome)
        }

        fn files(&self) -> &dyn Files {
            self.inner.files()
        }

        fn ports(&self) -> &dyn Ports {
            self.inner.ports()
        }
    }

    #[tokio::test]
    async fn a_lossy_command_tells_the_model_what_the_provider_dropped() {
        let mock = MockSandbox {
            exec_result: exec_result("built\n", "warning: torn", Some(0), Termination::Exited, 7),
            ..MockSandbox::linux()
        };
        let sandbox = SandboxEnvironment::with_platform(
            Arc::new(LossySandbox {
                inner: mock.driver(),
                loss:  output_loss(3, 512),
            }),
            "/work",
            "linux",
            "linux 6.1.0",
        );

        let outcome = Environment::exec(&sandbox, request("cargo build"))
            .await
            .expect("a lossy command completes rather than fails");

        assert_eq!(outcome.result.stdout, "built\n");
        assert_eq!(
            outcome.result.stderr,
            "warning: torn\n[sandbox] 3 output frame(s), 512 bytes dropped by the provider\n"
        );
        assert_eq!(outcome.result.exit_code, Some(0));
        assert!(outcome.streams_separated);
    }
}
