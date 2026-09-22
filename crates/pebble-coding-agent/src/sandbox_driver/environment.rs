//! A sandbox-driver handle as the [`Environment`] a session runs in.
//!
//! Pebble's tools speak the `Environment` contract; the sandbox driver
//! speaks facets. [`SandboxEnvironment`] is the mapping between the two, and
//! nothing else: every path resolves against the working directory the
//! application named (a relative path against it, which may sit below the
//! provider's own), every command runs through [`SandboxExec`] with its exec
//! policy, and every failure keeps its driver cause.
//!
//! Where the two contracts differ, pebble's wins here because the model reads
//! pebble's: a glob that pebble rejects is rejected before the driver sees it,
//! a directory listing is in tree order, and a command with no retention cap
//! still drains under the driver's default buffer rather than without bound.
//! Output a provider lost on its own transport
//! ([`ExecStreamingResult::output_loss`]) has no slot in pebble's contract,
//! so it is written where the model already reads: one line at the end of
//! stderr.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use sandbox_driver::{
    Error as DriverError, ExecControls, ExecSpec, ExecStreamingResult, FileKind, OutputLoss,
    OutputSink, OutputStream, Sandbox, Search as _, WalkOptions,
};
use tracing::warn;

use super::exec::{ExecResultExt as _, SandboxExec, command_termination, program_exit_code};
use super::path::{join_sandbox_path, resolve_path};
use crate::char_boundary::floor_char_boundary;
use crate::environment::support::{capture_stats, compile_glob, tree_order};
use crate::environment::{
    DirEntry, EnvResult, Environment, EnvironmentError, EnvironmentErrorKind, ExecOutcome,
    ExecOutputSink, ExecOutputStream, ExecRequest, ExecResult, GrepOptions,
};
#[cfg(feature = "mcp")]
use crate::mcp::PortRoutes;

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
}

impl SandboxEnvironment {
    /// Wraps a running `handle` working in `working_dir`, asking the sandbox
    /// for its platform once.
    pub async fn attach(
        handle: Arc<dyn Sandbox>,
        working_dir: impl Into<String>,
    ) -> sandbox_driver::Result<Self> {
        let info = handle.platform_info().await?;
        let platform = platform_name(&info.os).to_string();
        let os_version = if info.version.is_empty() {
            platform.clone()
        } else {
            format!("{platform} {}", info.version)
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
        }
    }

    /// The driver handle underneath, for the facets pebble's contract does
    /// not carry.
    #[must_use]
    pub fn handle(&self) -> &Arc<dyn Sandbox> {
        &self.handle
    }

    /// The directory the agent works in.
    #[must_use]
    pub fn working_directory(&self) -> &str {
        &self.working_dir
    }

    /// The exec policy over the handle's exec facet, working in the agent's
    /// directory.
    #[must_use]
    pub fn exec(&self) -> SandboxExec<'_> {
        SandboxExec::new(self.handle.exec()).with_working_dir(self.working_dir.clone())
    }

    /// Pebble's port routes over the handle's preview URLs, when the
    /// provider has them; see [`port_routes`](super::port_routes).
    #[cfg(feature = "mcp")]
    #[must_use]
    pub fn port_routes(&self) -> Option<Arc<dyn PortRoutes>> {
        super::ports::port_routes(&self.handle)
    }

    /// A caller path as the driver will see it.
    fn resolve(&self, path: &str) -> String {
        resolve_path(path, &self.working_dir)
    }

    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        self.handle
            .fs()
            .exists(&self.resolve(path))
            .await
            .map_err(|error| environment_error(&format!("Failed to stat {path}"), error))
    }

    /// The traversal base the driver walks. A base at the working directory
    /// walks relative to it so every path component of `relative_start` is
    /// checked against symlinks; any other base is walked as given.
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

/// Pebble names the macOS platform `darwin`, as `uname -s` and
/// [`LocalEnvironment`](crate::environment::LocalEnvironment) do.
fn platform_name(os: &str) -> &str {
    match os {
        "macos" => "darwin",
        other => other,
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
        self.handle
            .fs()
            .read(&self.resolve(path))
            .await
            .map_err(|error| environment_error(&format!("Failed to read {path}"), error))
    }

    async fn write_file(&self, path: &str, content: &str) -> EnvResult<()> {
        self.handle
            .fs()
            .write(&self.resolve(path), content.as_bytes())
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
        // no remote `realpath`, and a driver `mv a a` is a no-op anyway.
        if normalize(&resolved_source) == normalize(&resolved_destination) {
            return Ok(());
        }
        // The destination's parent is created first, and a parent that is a
        // file fails here, before anything has moved, so the source stays
        // intact as the contract requires.
        if let Some(parent) = parent_directory(&resolved_destination) {
            self.handle.fs().create_dir(parent).await.map_err(|error| {
                environment_error(
                    &format!("Failed to create the parent directory of {destination}"),
                    error,
                )
            })?;
        }
        self.handle
            .fs()
            .rename(&resolved_source, &resolved_destination)
            .await
            .map_err(|error| {
                environment_error(&format!("Failed to move {source} to {destination}"), error)
            })
    }

    async fn delete_file(&self, path: &str) -> EnvResult<()> {
        // The driver's delete is idempotent; pebble's is a `remove_file`, which
        // reports a path that is not there.
        if !self.file_exists(path).await? {
            return Err(EnvironmentError::new(
                EnvironmentErrorKind::NotFound,
                format!("Failed to delete {path}: file does not exist"),
            ));
        }
        self.handle
            .fs()
            .delete(&self.resolve(path), false)
            .await
            .map_err(|error| environment_error(&format!("Failed to delete {path}"), error))
    }

    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        Self::file_exists(self, path).await
    }

    async fn list_directory(&self, path: &str, depth: Option<usize>) -> EnvResult<Vec<DirEntry>> {
        let mut entries: Vec<DirEntry> = self
            .handle
            .fs()
            .list_dir(&self.resolve(path), depth.unwrap_or(1))
            .await
            .map_err(|error| environment_error(&format!("Failed to list {path}"), error))?
            .into_iter()
            .map(|entry| DirEntry {
                is_dir: entry.kind == FileKind::Directory,
                size:   (entry.kind == FileKind::File)
                    .then_some(entry.size)
                    .flatten(),
                name:   entry.path,
            })
            .collect();
        // The driver lists in flat lexicographic order of the whole relative
        // path, where `foo-bar` sorts between `foo` and `foo/x`. Pebble lists
        // in tree order, and says how.
        tree_order(&mut entries);
        Ok(entries)
    }

    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        options: &GrepOptions,
    ) -> EnvResult<Vec<String>> {
        let search = self.handle.search().ok_or_else(|| {
            EnvironmentError::new(
                EnvironmentErrorKind::Unsupported,
                "Sandbox provider does not support search",
            )
        })?;
        let mut driver_options = sandbox_driver::GrepOptions::default();
        driver_options.case_insensitive = options.case_insensitive;
        driver_options.max_matches = options.max_results;
        driver_options.include = options.glob_filter.clone();
        let matches = search
            .grep(pattern, &self.resolve(path), &driver_options)
            .await
            .map_err(|error| environment_error("Failed to search file contents", error))?;
        Ok(matches
            .into_iter()
            .map(|found| format!("{}:{}:{}", found.path, found.line_number, found.line))
            .collect())
    }

    async fn glob(&self, pattern: &str, path: Option<&str>) -> EnvResult<Vec<String>> {
        // Compiled by pebble's own grammar before the driver sees the
        // pattern, so the reason reaches the model in pebble's words and the
        // matching is pebble's wherever the files are.
        let glob = compile_glob(pattern)?;
        let search = self.handle.search().ok_or_else(|| {
            EnvironmentError::new(
                EnvironmentErrorKind::Unsupported,
                "Sandbox provider does not support search",
            )
        })?;
        let base = path.unwrap_or(&self.working_dir);
        let relative_start = glob.traversal_root();
        let walked = search
            .walk(
                &self.walk_base(base, relative_start),
                &WalkOptions::default(),
            )
            .await
            .map_err(|error| environment_error("Failed to match files", error))?;
        let mut relative_paths: Vec<String> = walked
            .into_iter()
            .map(|file| join_sandbox_path(relative_start, &file.path))
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
        let mut spec = ExecSpec::bash(command).no_timeout();
        if let Some(timeout_ms) = timeout_ms {
            spec = spec.timeout(Duration::from_millis(timeout_ms));
        }
        if let Some(dir) = working_dir {
            spec = spec.working_dir(dir);
        }
        for (key, value) in env_vars.into_iter().flatten() {
            spec = spec.env_var(key, value);
        }
        let controls = ExecControls {
            term: cancel_token,
            sink: output_sink.map(adapt_output_sink),
            // `None` asks pebble for no cap at all. The exec policy fills its
            // default buffer when the cap is unset, so a command with no cap
            // drains under that default rather than without bound; the
            // capture counts still say what was dropped.
            retained_output_limit: output_bytes_cap,
            ..ExecControls::default()
        };
        let streaming = self
            .exec()
            .run_streaming(spec, controls)
            .await
            .map_err(|error| {
                let kind = match &error {
                    DriverError::Transport(_) => EnvironmentErrorKind::Io,
                    DriverError::Unsupported { .. } => EnvironmentErrorKind::Unsupported,
                    _ => EnvironmentErrorKind::Spawn,
                };
                EnvironmentError::with_source(kind, "Failed to run the command", error)
            })?;
        Ok(exec_outcome(
            streaming,
            output_bytes_cap,
            program_name(command),
        ))
    }
}

/// Pebble's outcome for a finished command: the driver's result read the way
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
/// operator. The driver's `truncated` flags on the captures already say the
/// counts undercount.
fn exec_outcome(
    streaming: ExecStreamingResult,
    output_bytes_cap: Option<usize>,
    program: &str,
) -> ExecOutcome {
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
        stdout_capture:    capture_stats(streaming.stdout_capture.observed_bytes, output_bytes_cap),
        stderr_capture:    capture_stats(streaming.stderr_capture.observed_bytes, output_bytes_cap),
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
/// sink.
fn adapt_output_sink(sink: ExecOutputSink) -> OutputSink {
    Arc::new(move |stream, chunk: Vec<u8>| {
        let stream = match stream {
            OutputStream::Stdout => ExecOutputStream::Stdout,
            OutputStream::Stderr => ExecOutputStream::Stderr,
        };
        sink(stream, &chunk);
        Box::pin(async { Ok(()) })
    })
}

/// A sandbox failure as pebble classifies it, keeping the driver cause.
fn environment_error(message: &str, error: DriverError) -> EnvironmentError {
    let kind = match &error {
        DriverError::NotFound { .. } => EnvironmentErrorKind::NotFound,
        DriverError::Unsupported { .. } => EnvironmentErrorKind::Unsupported,
        _ => EnvironmentErrorKind::Io,
    };
    EnvironmentError::with_source(kind, message, error)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::env::consts::OS;

    use sandbox_driver::{
        Capabilities, Exec, Filesystem, PlatformInfo, SandboxId, SandboxProvider as _,
        SandboxSource, SandboxSpec, SandboxStatus, Search, SpawnSpec, StdioProcess, Termination,
    };
    use sandbox_driver_host::HostProvider;
    use sandbox_driver_testing::ScriptedSandbox;
    use tokio::fs;

    use super::*;
    use crate::sandbox_driver::DEFAULT_STOP_GRACE;
    use crate::sandbox_driver::test_support::{MockSandbox, exec_result};
    use crate::test_support::EnvironmentContract;

    /// The environment over the driver's Host provider, in a directory that
    /// goes away with the test.
    async fn host_environment() -> (tempfile::TempDir, HostProvider, SandboxEnvironment) {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let provider = HostProvider::new();
        let handle = provider
            .create(
                &SandboxSpec::new(SandboxSource::HostDirectory)
                    .working_directory(directory.path().display().to_string()),
                None,
            )
            .await
            .expect("a host sandbox");
        let working_dir = handle.working_directory().to_string();
        let sandbox = SandboxEnvironment::attach(handle, working_dir)
            .await
            .expect("the host platform");
        (directory, provider, sandbox)
    }

    #[tokio::test]
    async fn host_files_satisfy_pebbles_environment_contract() {
        let (_directory, _provider, sandbox) = host_environment().await;
        EnvironmentContract::new(&sandbox, "contract")
            .verify_files()
            .await
            .expect("file contract");
    }

    #[tokio::test]
    async fn host_search_satisfies_pebbles_environment_contract() {
        let (_directory, _provider, sandbox) = host_environment().await;
        EnvironmentContract::new(&sandbox, "contract")
            .verify_search()
            .await
            .expect("search contract");
    }

    #[tokio::test]
    async fn host_commands_satisfy_pebbles_environment_contract() {
        let (_directory, _provider, sandbox) = host_environment().await;
        EnvironmentContract::new(&sandbox, "contract")
            .verify_commands()
            .await
            .expect("command contract");
    }

    #[tokio::test]
    async fn the_platform_is_learned_from_the_sandbox() {
        let (_directory, _provider, sandbox) = host_environment().await;
        let expected = if cfg!(target_os = "macos") {
            "darwin"
        } else {
            OS
        };
        assert_eq!(Environment::platform(&sandbox), expected);
        assert!(sandbox.os_version().starts_with(expected));
    }

    #[tokio::test]
    async fn a_directory_listing_is_in_tree_order() {
        let (directory, provider, sandbox) = host_environment().await;
        for name in ["foo/x.txt", "foo-bar/y.txt", "foo.txt"] {
            Environment::write_file(&sandbox, name, "content")
                .await
                .expect("fixture");
        }
        let names: Vec<String> = Environment::list_directory(&sandbox, ".", Some(2))
            .await
            .expect("listing")
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(names, [
            "foo",
            "foo/x.txt",
            "foo-bar",
            "foo-bar/y.txt",
            "foo.txt"
        ]);
        drop((directory, provider));
    }

    #[tokio::test]
    async fn glob_reports_paths_under_the_declared_base_and_skips_symlinks() {
        let (directory, provider, sandbox) = host_environment().await;
        let root = directory.path();
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
        drop((directory, provider));
    }

    #[tokio::test]
    async fn a_glob_pebble_refuses_never_reaches_the_driver() {
        let (_directory, _provider, sandbox) = host_environment().await;
        let error = Environment::glob(&sandbox, "docs/", None)
            .await
            .expect_err("a trailing slash names no file");
        assert_eq!(error.kind(), EnvironmentErrorKind::InvalidInput);
    }

    #[tokio::test]
    async fn grep_returns_path_line_content_triples() {
        let (directory, provider, sandbox) = host_environment().await;
        fs::write(
            directory.path().join("test.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .await
        .unwrap();
        let results = Environment::grep(&sandbox, "println", "test.rs", &GrepOptions::default())
            .await
            .unwrap();
        // The path is resolved against the working directory before the
        // driver sees it, and comes back as the driver reports it.
        let working_dir = sandbox.working_directory();
        assert_eq!(results, [format!(
            "{working_dir}/test.rs:2:    println!(\"hello\");"
        )]);
        drop((directory, provider));
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
            command,
            timeout_ms: Some(10_000),
            working_dir: None,
            env_vars: None,
            cancel_token: None,
            output_bytes_cap: None,
            output_sink: None,
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
        // The command ran in the mock's working directory under the default
        // stop grace.
        let spec = mock.driver().scripted_exec().recorded().pop().unwrap();
        assert_eq!(spec.working_dir.as_deref(), Some("/home/test"));
        assert_eq!(spec.stop_grace, Some(DEFAULT_STOP_GRACE));
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

        let outcome = exec_outcome(streaming, Some(1024), "cargo");

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
        let outcome = exec_outcome(streaming, None, "sh");
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

    /// The driver's scripted sandbox with an exec facet that reports a
    /// provider output loss on every command, as Daytona does after a torn
    /// frame. The scripted double itself has no knob for the loss.
    struct LossySandbox {
        inner: Arc<ScriptedSandbox>,
        exec:  LossyExec,
    }

    struct LossyExec {
        inner: Arc<ScriptedSandbox>,
        loss:  OutputLoss,
    }

    impl LossySandbox {
        fn new(inner: Arc<ScriptedSandbox>, loss: OutputLoss) -> Self {
            Self {
                exec: LossyExec {
                    inner: Arc::clone(&inner),
                    loss,
                },
                inner,
            }
        }
    }

    #[async_trait]
    impl Exec for LossyExec {
        async fn run(&self, spec: &ExecSpec) -> sandbox_driver::Result<sandbox_driver::ExecResult> {
            self.inner.scripted_exec().run(spec).await
        }

        async fn run_streaming(
            &self,
            spec: &ExecSpec,
            controls: ExecControls,
        ) -> sandbox_driver::Result<ExecStreamingResult> {
            let mut streaming = self
                .inner
                .scripted_exec()
                .run_streaming(spec, controls)
                .await?;
            streaming.output_loss = self.loss;
            streaming.stdout_capture.truncated = true;
            streaming.stderr_capture.truncated = true;
            Ok(streaming)
        }

        async fn spawn_stdio(&self, spec: &SpawnSpec) -> sandbox_driver::Result<StdioProcess> {
            self.inner.scripted_exec().spawn_stdio(spec).await
        }
    }

    #[async_trait]
    impl Sandbox for LossySandbox {
        fn id(&self) -> &SandboxId {
            self.inner.id()
        }

        fn capabilities(&self) -> &Capabilities {
            // The scripted sandbox's builder method of the same name shadows
            // the trait's.
            Sandbox::capabilities(&*self.inner)
        }

        async fn describe(&self) -> sandbox_driver::Result<SandboxStatus> {
            self.inner.describe().await
        }

        fn working_directory(&self) -> &str {
            self.inner.working_directory()
        }

        async fn environment(&self) -> sandbox_driver::Result<BTreeMap<String, String>> {
            self.inner.environment().await
        }

        fn runtime_directory(&self) -> Option<&str> {
            Sandbox::runtime_directory(&*self.inner)
        }

        async fn platform_info(&self) -> sandbox_driver::Result<PlatformInfo> {
            self.inner.platform_info().await
        }

        async fn start(&self) -> sandbox_driver::Result<()> {
            self.inner.start().await
        }

        async fn stop(&self) -> sandbox_driver::Result<()> {
            self.inner.stop().await
        }

        async fn delete(&self) -> sandbox_driver::Result<()> {
            self.inner.delete().await
        }

        fn exec(&self) -> &dyn Exec {
            &self.exec
        }

        fn fs(&self) -> &dyn Filesystem {
            self.inner.fs()
        }

        fn provider_search(&self) -> Option<&dyn Search> {
            self.inner.provider_search()
        }
    }

    #[tokio::test]
    async fn a_lossy_command_tells_the_model_what_the_provider_dropped() {
        let scripted =
            Arc::new(
                ScriptedSandbox::with_id_and_working_dir("lossy", "/work")
                    .platform(PlatformInfo::new("linux", "x86_64", "Linux 6.1.0")),
            );
        scripted.scripted_exec().set_default(exec_result(
            "built\n",
            "warning: torn",
            Some(0),
            Termination::Exited,
            7,
        ));
        let sandbox = SandboxEnvironment::with_platform(
            Arc::new(LossySandbox::new(scripted, output_loss(3, 512))),
            "/work",
            "linux",
            "Linux 6.1.0",
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
