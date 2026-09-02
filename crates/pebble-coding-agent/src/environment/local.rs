//! An [`Environment`] that works on the machine pebble itself runs on.

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::path::{MAIN_SEPARATOR, Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::{env, fs as sync_fs, future, process};

use async_trait::async_trait;
#[cfg(unix)]
use rustix::process::{Pid, Signal, kill_process_group};
use tokio::io::{AsyncRead, AsyncReadExt as _};
use tokio::process::{Child, Command};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::{fs, time};
use tokio_util::sync::CancellationToken;

use super::capture::OutputCaptureBuffer;
use super::glob::WorkspaceGlob;
use super::{
    DirEntry, EnvResult, Environment, EnvironmentError, EnvironmentErrorKind, ExecOutcome,
    ExecRequest, ExecResult, GrepOptions,
};
use crate::types::CommandTermination;

/// Shown when the machine has no usable Bash.
const BASH_REMEDIATION: &str = "A local environment runs commands with Bash. Install bash and make it \
     reachable through PATH.";

/// The variable Bash reads a startup file from. A command must never inherit
/// it: ambient configuration would run code before the command itself.
const BASH_ENV_VAR: &str = "BASH_ENV";

/// Marker the Bash probe prints on success.
const BASH_PROBE_MARKER: &str = "pebble-bash-ready";

/// How long the Bash probe may take.
const BASH_PROBE_TIMEOUT_MS: u64 = 10_000;

/// Grace period between SIGTERM and SIGKILL.
const TERMINATION_GRACE: Duration = Duration::from_secs(2);

/// How long a stopped command's output is still collected before the call
/// returns with what was read.
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_millis(500);

/// Bytes read from a pipe at a time.
const PIPE_CHUNK_BYTES: usize = 8192;

/// Proves the interpreter running a command is non-login Bash with no ambient
/// startup source.
///
/// Bash invoked as `sh` still reports `BASH_VERSION` while switching to POSIX
/// behavior, so the probe checks the whole contract rather than the version
/// alone.
const BASH_PROBE_SCRIPT: &str = r#"if [ -n "${BASH_ENV:-}" ]; then
  echo 'interpreter has a BASH_ENV startup source configured' >&2
  exit 1
fi
if [ -z "${BASH_VERSION:-}" ]; then
  echo 'interpreter is not bash' >&2
  exit 1
fi
if shopt -q login_shell; then
  echo 'interpreter is a login shell' >&2
  exit 1
fi
if shopt -qo posix; then
  echo 'interpreter is bash in posix mode' >&2
  exit 1
fi
printf '%s\n' 'pebble-bash-ready'"#;

/// Variables a command keeps regardless of what its name ends with.
const DEFAULT_ENV_SAFELIST: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "TERM",
    "TMPDIR",
    "GOPATH",
    "CARGO_HOME",
    "NVM_DIR",
];

/// Suffixes that mark a variable as sensitive.
const SENSITIVE_SUFFIXES: &[&str] = &["_api_key", "_secret", "_token", "_password", "_credential"];

/// Where a [`LocalEnvironment`] gets its interpreter.
///
/// Only tests name anything but [`ResolveFromPath`](Self::ResolveFromPath): a
/// machine's Bash is the one on its `PATH`, and the other variants exist so the
/// failures around a missing or wrong interpreter can be exercised without
/// rewriting `PATH` for the whole process.
#[derive(Debug)]
enum BashExecutable {
    /// Resolve `bash` through `PATH`.
    ResolveFromPath,
    #[cfg(test)]
    Fixed(PathBuf),
    #[cfg(test)]
    Unavailable,
}

/// What a [`LocalEnvironment`] does with the variables a caller passes in
/// [`ExecRequest::env_vars`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CallerEnvPolicy {
    /// Apply the same name filter the ambient environment gets: a variable
    /// whose name ends in one of a fixed set of credential-like suffixes is
    /// dropped unless it is safelisted.
    ///
    /// The filter reads names, never values, and the set of suffixes is
    /// pebble's rather than exhaustive, so a secret under a name it does not
    /// recognize still reaches the command. It is hygiene against forwarding
    /// the orchestrator's own credentials, not a boundary: a command can read
    /// whatever the process can.
    #[default]
    FilterSensitive,
    /// Pass caller-supplied variables through untouched. Choose this only when
    /// the application deliberately hands secrets to commands.
    Trusted,
}

/// An [`Environment`] that reads, writes, and runs commands on this machine.
///
/// It offers no isolation. Everything a tool asks for happens with the
/// permissions of the process pebble runs in, which makes it the right choice
/// for a local coding agent and the wrong one for untrusted work.
///
/// What it does do is keep the command environment predictable:
///
/// - commands run under Bash resolved through `PATH` (not `/bin/bash`, which
///   some distributions do not have), as `bash -c <command>`;
/// - the ambient environment is cleared and rebuilt, dropping `BASH_ENV` and
///   every variable whose name ends like a credential unless it is safelisted;
/// - each command gets its own process group, so a timeout or a cancellation
///   stops the whole tree rather than the shell alone;
/// - both output streams are drained while the command runs, so a noisy command
///   cannot deadlock on a full pipe, and only the caller's byte cap is
///   retained.
///
/// [`grep`](Environment::grep) shells out to `rg` when it is on `PATH` and to
/// `grep` otherwise, and those two run with the ambient environment rather
/// than the filtered one.
///
/// ```no_run
/// use pebble_coding_agent::environment::{Environment, LocalEnvironment};
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let environment = LocalEnvironment::new("/work/project");
/// environment.prepare().await?;
/// let listing = environment.list_directory(".", None).await?;
/// # let _ = listing;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct LocalEnvironment {
    working_directory: PathBuf,
    env_safelist:      Vec<String>,
    caller_env_policy: CallerEnvPolicy,
    bash_executable:   BashExecutable,
    bash_path:         OnceLock<PathBuf>,
    ripgrep_available: OnceLock<bool>,
    os_version:        OnceLock<String>,
}

impl LocalEnvironment {
    /// An environment rooted at `working_directory`.
    ///
    /// The directory is not touched here; call [`prepare`](Self::prepare) to
    /// create it and check the interpreter before the first command.
    #[must_use]
    pub fn new(working_directory: impl Into<PathBuf>) -> Self {
        Self {
            working_directory: working_directory.into(),
            env_safelist:      DEFAULT_ENV_SAFELIST
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            caller_env_policy: CallerEnvPolicy::default(),
            bash_executable:   BashExecutable::ResolveFromPath,
            bash_path:         OnceLock::new(),
            ripgrep_available: OnceLock::new(),
            os_version:        OnceLock::new(),
        }
    }

    /// An environment that runs commands with a named interpreter.
    #[cfg(test)]
    fn with_bash_executable(
        working_directory: impl Into<PathBuf>,
        bash_executable: BashExecutable,
    ) -> Self {
        Self {
            bash_executable,
            ..Self::new(working_directory)
        }
    }

    /// Replaces the variable names that survive the sensitive-name filter.
    #[must_use]
    pub fn with_env_safelist<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.env_safelist = names.into_iter().map(Into::into).collect();
        self
    }

    /// Chooses what happens to the variables a caller passes per command.
    #[must_use]
    pub fn with_caller_env_policy(mut self, policy: CallerEnvPolicy) -> Self {
        self.caller_env_policy = policy;
        self
    }

    /// Creates the working directory when it is missing and proves commands
    /// will run under non-login Bash.
    ///
    /// Call this once before handing the environment to a session. It is
    /// optional — commands work without it — but it turns "the interpreter is
    /// wrong" into one clear failure instead of a puzzling result from the
    /// first tool call.
    pub async fn prepare(&self) -> EnvResult<()> {
        fs::create_dir_all(&self.working_directory)
            .await
            .map_err(|error| {
                EnvironmentError::io(
                    format!(
                        "Failed to create working directory {}",
                        self.working_directory.display()
                    ),
                    error,
                )
            })?;

        let bash = self.bash()?;
        let outcome = self
            .exec(ExecRequest {
                timeout_ms: Some(BASH_PROBE_TIMEOUT_MS),
                ..ExecRequest::new(BASH_PROBE_SCRIPT)
            })
            .await?;

        if bash_probe_passed(&outcome.result) {
            return Ok(());
        }

        // Only the probe script's own first diagnostic line is quoted, so the
        // message says which part of the contract failed without carrying a
        // command's output into a log.
        Err(EnvironmentError::new(
            EnvironmentErrorKind::Spawn,
            format!(
                "{} is not usable as non-login Bash ({}). {BASH_REMEDIATION}",
                bash.display(),
                first_line(&outcome.result.stderr).unwrap_or("no diagnostic")
            ),
        ))
    }

    /// The Bash executable this environment runs commands with, resolved once.
    fn bash(&self) -> EnvResult<PathBuf> {
        if let Some(path) = self.bash_path.get() {
            return Ok(path.clone());
        }

        let resolved = match &self.bash_executable {
            BashExecutable::ResolveFromPath => binary_path_on_path("bash").ok_or_else(|| {
                EnvironmentError::new(EnvironmentErrorKind::Spawn, BASH_REMEDIATION)
            })?,
            #[cfg(test)]
            BashExecutable::Fixed(path) => path.clone(),
            #[cfg(test)]
            BashExecutable::Unavailable => {
                return Err(EnvironmentError::new(
                    EnvironmentErrorKind::Spawn,
                    BASH_REMEDIATION,
                ));
            }
        };
        Ok(self.bash_path.get_or_init(|| resolved).clone())
    }

    /// Whether a variable is dropped from a command's environment.
    fn filters_env_var(&self, name: &str) -> bool {
        if self
            .env_safelist
            .iter()
            .any(|safelisted| safelisted == name)
        {
            return false;
        }
        let lowercase = name.to_lowercase();
        SENSITIVE_SUFFIXES
            .iter()
            .any(|suffix| lowercase.ends_with(suffix))
    }

    /// The environment one command runs with: the ambient variables that pass
    /// the filter, then the caller's own.
    fn command_env(&self, env_vars: Option<&HashMap<String, String>>) -> Vec<(String, String)> {
        let mut command_env: Vec<(String, String)> = env::vars()
            .filter(|(name, _)| name != BASH_ENV_VAR && !self.filters_env_var(name))
            .collect();

        for (name, value) in env_vars.into_iter().flatten() {
            let trusted = self.caller_env_policy == CallerEnvPolicy::Trusted;
            if name != BASH_ENV_VAR && (trusted || !self.filters_env_var(name)) {
                command_env.push((name.clone(), value.clone()));
            }
        }

        command_env
    }

    /// Resolves a caller's path against the working directory.
    fn resolve_path(&self, path: &str) -> PathBuf {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.working_directory.join(candidate)
        }
    }

    /// Runs `rg` or `grep` and returns its matching lines.
    async fn search(&self, arguments: Vec<String>, binary: &str) -> EnvResult<Vec<String>> {
        let output = Command::new(binary)
            .args(&arguments)
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|error| {
                EnvironmentError::with_source(
                    EnvironmentErrorKind::Spawn,
                    format!("Failed to run {binary}"),
                    error,
                )
            })?;

        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect())
    }
}

#[async_trait]
impl Environment for LocalEnvironment {
    fn working_directory(&self) -> &str {
        // A path that is not UTF-8 is reported as the current directory.
        self.working_directory.to_str().unwrap_or(".")
    }

    fn platform(&self) -> &str {
        if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "windows"
        } else {
            "unknown"
        }
    }

    fn os_version(&self) -> String {
        // `uname` runs at most once per environment, and the accessor is
        // synchronous because prompt assembly reads it outside any task.
        self.os_version
            .get_or_init(|| {
                #[cfg(unix)]
                {
                    match process::Command::new("uname").arg("-r").output() {
                        Ok(output) => {
                            let version = String::from_utf8_lossy(&output.stdout);
                            format!("{} {}", self.platform(), version.trim())
                        }
                        Err(_) => self.platform().to_owned(),
                    }
                }
                #[cfg(not(unix))]
                {
                    self.platform().to_owned()
                }
            })
            .clone()
    }

    async fn read_file_bytes(&self, path: &str) -> EnvResult<Vec<u8>> {
        let full_path = self.resolve_path(path);
        fs::read(&full_path)
            .await
            .map_err(|error| EnvironmentError::io(format!("Failed to read {path}"), error))
    }

    async fn write_file(&self, path: &str, content: &str) -> EnvResult<()> {
        let full_path = self.resolve_path(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).await.map_err(|error| {
                EnvironmentError::io(
                    format!("Failed to create parent directories for {path}"),
                    error,
                )
            })?;
        }
        fs::write(&full_path, content)
            .await
            .map_err(|error| EnvironmentError::io(format!("Failed to write {path}"), error))
    }

    async fn delete_file(&self, path: &str) -> EnvResult<()> {
        let full_path = self.resolve_path(path);
        fs::remove_file(&full_path)
            .await
            .map_err(|error| EnvironmentError::io(format!("Failed to delete {path}"), error))
    }

    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        let full_path = self.resolve_path(path);
        Ok(fs::try_exists(&full_path).await.unwrap_or(false))
    }

    async fn list_directory(&self, path: &str, depth: Option<usize>) -> EnvResult<Vec<DirEntry>> {
        let full_path = self.resolve_path(path);
        let display = path.to_owned();
        let max_depth = depth.unwrap_or(1);

        // The walk is synchronous recursion over `read_dir`, so it runs off
        // the runtime's worker threads.
        spawn_blocking(move || {
            let mut entries = Vec::new();
            list_recursive(&full_path, "", 0, max_depth, &mut entries)?;
            Ok(entries)
        })
        .await
        .map_err(|error| {
            EnvironmentError::with_source(
                EnvironmentErrorKind::Io,
                format!("Failed to list {display}"),
                error,
            )
        })?
    }

    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        options: &GrepOptions,
    ) -> EnvResult<Vec<String>> {
        let full_path = self.resolve_path(path);
        let use_ripgrep = *self
            .ripgrep_available
            .get_or_init(|| binary_path_on_path("rg").is_some());

        let (binary, mut arguments) = if use_ripgrep {
            ("rg", vec!["-n".to_owned()])
        } else {
            ("grep", vec!["-rn".to_owned()])
        };
        if options.case_insensitive {
            arguments.push("-i".to_owned());
        }
        if let Some(glob_filter) = &options.glob_filter {
            arguments.push(if use_ripgrep { "--glob" } else { "--include" }.to_owned());
            arguments.push(glob_filter.clone());
        }
        if let Some(max_results) = options.max_results {
            arguments.push("-m".to_owned());
            arguments.push(max_results.to_string());
        }
        arguments.push(pattern.to_owned());
        arguments.push(full_path.to_string_lossy().into_owned());

        self.search(arguments, binary).await
    }

    async fn glob(&self, pattern: &str, path: Option<&str>) -> EnvResult<Vec<String>> {
        let compiled = WorkspaceGlob::try_new(pattern).map_err(|error| {
            EnvironmentError::with_source(
                EnvironmentErrorKind::InvalidInput,
                format!("Invalid glob pattern: {pattern}"),
                error,
            )
        })?;

        let base = path.map_or_else(
            || self.working_directory.clone(),
            |path| self.resolve_path(path),
        );
        let mut files = walk_files(&base, compiled.traversal_root()).await?;
        files.retain(|file| compiled.is_match(&file.relative_path));
        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

        Ok(files.into_iter().map(|file| file.path).collect())
    }

    async fn exec(&self, request: ExecRequest<'_>) -> EnvResult<ExecOutcome> {
        let ExecRequest {
            command,
            timeout_ms,
            working_dir,
            env_vars,
            cancel_token,
            output_bytes_cap,
        } = request;
        let started = Instant::now();

        let mut builder = Command::new(self.bash()?);
        builder
            .arg("-c")
            .arg(command)
            .current_dir(working_dir.map_or_else(|| self.working_directory.clone(), PathBuf::from))
            .env_clear()
            .envs(self.command_env(env_vars))
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;

            // Its own process group, so stopping the command stops whatever it
            // started.
            builder.as_std_mut().process_group(0);
        }

        let mut child = builder.spawn().map_err(|error| {
            EnvironmentError::with_source(
                EnvironmentErrorKind::Spawn,
                "Failed to start the command",
                error,
            )
        })?;

        // Take the pipes and drain them before waiting. A command that writes
        // more than the pipe buffer holds blocks until someone reads, and a
        // parent blocked on `wait` never would.
        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let drain_stop = CancellationToken::new();
        let stdout_task = tokio::spawn(drain_pipe(
            stdout_pipe,
            output_bytes_cap,
            drain_stop.clone(),
        ));
        let stderr_task = tokio::spawn(drain_pipe(
            stderr_pipe,
            output_bytes_cap,
            drain_stop.clone(),
        ));

        let deadline = optional_timeout(timeout_ms);
        tokio::pin!(deadline);
        let cancelled = cancel_token.unwrap_or_default();

        let (termination, exit_code) = tokio::select! {
            status = child.wait() => {
                let status = status.map_err(|error| EnvironmentError::with_source(
                    EnvironmentErrorKind::Io,
                    "Failed to wait for the command",
                    error,
                ))?;
                (CommandTermination::Exited, status.code())
            }
            () = &mut deadline => {
                terminate(&mut child).await;
                (CommandTermination::TimedOut, None)
            }
            () = cancelled.cancelled() => {
                terminate(&mut child).await;
                (CommandTermination::Cancelled, None)
            }
        };

        let duration_ms = elapsed_ms(started);
        // A command that ended on its own is read to the end, however long its
        // output takes to arrive. A command that had to be stopped is not: what
        // survived the kill may hold the pipes for as long as it likes, and the
        // caller asked for this call to be over.
        let drain_grace =
            (!matches!(termination, CommandTermination::Exited)).then_some(OUTPUT_DRAIN_GRACE);
        let (stdout_buffer, stderr_buffer) =
            join_drains(stdout_task, stderr_task, &drain_stop, drain_grace).await?;
        let (stdout_bytes, stdout_capture) = stdout_buffer.into_parts();
        let (stderr_bytes, stderr_capture) = stderr_buffer.into_parts();

        Ok(ExecOutcome {
            result: ExecResult {
                stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
                stderr: String::from_utf8_lossy(&stderr_bytes).into_owned(),
                exit_code,
                termination,
                duration_ms,
            },
            streams_separated: true,
            stdout_capture,
            stderr_capture,
        })
    }
}

/// One regular file found while walking for a glob.
struct WalkedFile {
    /// The path this environment accepts for the file.
    path:          String,
    /// The `/`-separated path relative to the traversal base.
    relative_path: String,
}

/// Lists a directory recursively, sorted by file name at every level.
fn list_recursive(
    base: &Path,
    prefix: &str,
    current_depth: usize,
    max_depth: usize,
    entries: &mut Vec<DirEntry>,
) -> EnvResult<()> {
    let mut directory_entries: Vec<sync_fs::DirEntry> = sync_fs::read_dir(base)
        .map_err(|error| {
            EnvironmentError::io(
                format!("Failed to read directory {}", base.display()),
                error,
            )
        })?
        .filter_map(Result::ok)
        .collect();
    directory_entries.sort_by_key(sync_fs::DirEntry::file_name);

    for entry in directory_entries {
        let metadata = entry.metadata().map_err(|error| {
            EnvironmentError::io(
                format!("Failed to read metadata for {}", entry.path().display()),
                error,
            )
        })?;
        let name = if prefix.is_empty() {
            entry.file_name().to_string_lossy().into_owned()
        } else {
            format!("{prefix}/{}", entry.file_name().to_string_lossy())
        };
        let is_dir = metadata.is_dir();
        entries.push(DirEntry {
            name: name.clone(),
            is_dir,
            size: metadata.is_file().then_some(metadata.len()),
        });
        if is_dir && current_depth + 1 < max_depth {
            list_recursive(&entry.path(), &name, current_depth + 1, max_depth, entries)?;
        }
    }
    Ok(())
}

/// Collects every regular file below `base`, starting at `relative_start`.
///
/// Symlinked directories are not followed, so a link cannot walk a glob out of
/// the base directory. The base itself is resolved, so an environment rooted
/// at a symlink still works.
async fn walk_files(base: &Path, relative_start: &str) -> EnvResult<Vec<WalkedFile>> {
    let base = base.to_path_buf();
    let display = base.display().to_string();
    let relative_start = relative_start.to_owned();

    // The walk is a synchronous loop over `read_dir`, so it runs off the
    // runtime's worker threads in one dispatch rather than one per entry.
    spawn_blocking(move || {
        let Some((root, root_metadata)) = walk_root(&base, &relative_start)? else {
            return Ok(Vec::new());
        };

        let mut files = Vec::new();
        let mut pending = vec![(root, root_metadata.file_type())];
        while let Some((path, file_type)) = pending.pop() {
            if file_type.is_file() {
                let relative_path = path.strip_prefix(&base).map_err(|error| {
                    EnvironmentError::with_source(
                        EnvironmentErrorKind::Io,
                        format!("Failed to relate {} to {}", path.display(), base.display()),
                        error,
                    )
                })?;
                files.push(WalkedFile {
                    relative_path: relative_path.to_string_lossy().replace(MAIN_SEPARATOR, "/"),
                    path:          path.to_string_lossy().into_owned(),
                });
            } else if file_type.is_dir() {
                let entries = match sync_fs::read_dir(&path) {
                    Ok(entries) => entries,
                    Err(error) if error.kind() == ErrorKind::NotFound => continue,
                    Err(error) => {
                        return Err(EnvironmentError::io(
                            format!("Failed to read directory {}", path.display()),
                            error,
                        ));
                    }
                };
                for entry in entries {
                    let entry = entry.map_err(|error| {
                        EnvironmentError::io(
                            format!("Failed to read directory {}", path.display()),
                            error,
                        )
                    })?;
                    // The type comes from the directory read itself; a stat
                    // happens only where the filesystem left it unknown, and
                    // it never follows a symlink.
                    match entry.file_type() {
                        Ok(file_type) => pending.push((entry.path(), file_type)),
                        Err(error) if error.kind() == ErrorKind::NotFound => {}
                        Err(error) => {
                            return Err(EnvironmentError::io(
                                format!("Failed to inspect {}", entry.path().display()),
                                error,
                            ));
                        }
                    }
                }
            }
        }

        Ok(files)
    })
    .await
    .map_err(|error| {
        EnvironmentError::with_source(
            EnvironmentErrorKind::Io,
            format!("Failed to walk {display}"),
            error,
        )
    })?
}

/// Resolves where a walk starts, or `None` when the path is missing or
/// crosses a symlink.
fn walk_root(base: &Path, relative_start: &str) -> EnvResult<Option<(PathBuf, sync_fs::Metadata)>> {
    fn missing(error: &io::Error) -> bool {
        matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory)
    }

    if relative_start.is_empty() {
        return match sync_fs::metadata(base) {
            Ok(metadata) => Ok(Some((base.to_path_buf(), metadata))),
            Err(error) if missing(&error) => Ok(None),
            Err(error) => Err(EnvironmentError::io(
                format!("Failed to inspect {}", base.display()),
                error,
            )),
        };
    }

    let mut root = base.to_path_buf();
    let mut metadata = None;
    for segment in relative_start.split('/') {
        root.push(segment);
        let segment_metadata = match sync_fs::symlink_metadata(&root) {
            Ok(metadata) => metadata,
            Err(error) if missing(&error) => return Ok(None),
            Err(error) => {
                return Err(EnvironmentError::io(
                    format!("Failed to inspect {}", root.display()),
                    error,
                ));
            }
        };
        if segment_metadata.file_type().is_symlink() {
            return Ok(None);
        }
        metadata = Some(segment_metadata);
    }

    Ok(metadata.map(|metadata| (root, metadata)))
}

/// Reads one pipe to end of file, keeping what the cap allows.
///
/// A read failure ends the capture and keeps what was read: the command itself
/// ran, and its exit status and other stream are worth more to a caller than
/// failing the whole call over a pipe. The failure is logged, and the byte
/// counts then describe what was read rather than what the process wrote.
///
/// `stop` ends the read early and keeps what was read, for the one case where
/// end of file may never come: a descendant that outlived the command holds
/// the same pipe open. `read` is cancel-safe, so nothing already in the pipe
/// is dropped by stopping between reads.
async fn drain_pipe<R>(
    pipe: Option<R>,
    output_bytes_cap: Option<usize>,
    stop: CancellationToken,
) -> OutputCaptureBuffer
where
    R: AsyncRead + Unpin,
{
    let mut captured = OutputCaptureBuffer::new(output_bytes_cap);
    let Some(mut reader) = pipe else {
        return captured;
    };

    let mut chunk = [0_u8; PIPE_CHUNK_BYTES];
    loop {
        let read = tokio::select! {
            biased;
            read = reader.read(&mut chunk) => read,
            () = stop.cancelled() => return captured,
        };
        match read {
            Ok(0) => return captured,
            Ok(read) => captured.push(&chunk[..read]),
            Err(error) => {
                tracing::warn!(%error, "Failed to read command output");
                return captured;
            }
        }
    }
}

/// Joins both drain tasks, and where the command had to be stopped, gives them
/// a bounded window to finish before asking them for what they have.
///
/// A command that was terminated may leave a descendant holding the pipes,
/// which never reach end of file. Without the window, `exec` would wait on
/// that descendant for as long as it lives, past the timeout or cancellation
/// that stopped the command.
async fn join_drains(
    stdout_task: JoinHandle<OutputCaptureBuffer>,
    stderr_task: JoinHandle<OutputCaptureBuffer>,
    stop: &CancellationToken,
    grace: Option<Duration>,
) -> EnvResult<(OutputCaptureBuffer, OutputCaptureBuffer)> {
    let joined = async {
        let stdout_buffer = join_drain(stdout_task, "standard output").await?;
        let stderr_buffer = join_drain(stderr_task, "standard error").await?;
        Ok((stdout_buffer, stderr_buffer))
    };
    tokio::pin!(joined);

    if let Some(grace) = grace {
        tokio::select! {
            joined = &mut joined => return joined,
            () = time::sleep(grace) => stop.cancel(),
        }
    }
    joined.await
}

async fn join_drain(
    task: JoinHandle<OutputCaptureBuffer>,
    stream: &str,
) -> EnvResult<OutputCaptureBuffer> {
    task.await.map_err(|error| {
        EnvironmentError::with_source(
            EnvironmentErrorKind::Io,
            format!("Failed to collect command {stream}"),
            error,
        )
    })
}

/// Resolves after `timeout_ms`, or never when there is no timeout.
async fn optional_timeout(timeout_ms: Option<u64>) {
    match timeout_ms {
        Some(milliseconds) => time::sleep(Duration::from_millis(milliseconds)).await,
        None => future::pending().await,
    }
}

/// Stops a command: SIGTERM to its process group, then SIGKILL if it does not
/// leave within the grace period.
async fn terminate(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child
        .id()
        .and_then(|pid| i32::try_from(pid).ok().and_then(Pid::from_raw))
    {
        // The group, not the process: a shell that started a pipeline leaves
        // orphans behind when only it is signalled.
        let _ = kill_process_group(pid, Signal::TERM);
        if time::timeout(TERMINATION_GRACE, child.wait()).await.is_ok() {
            return;
        }
        // The group again, for the same reason: a descendant that ignored the
        // SIGTERM survives a signal sent to the shell alone, and it holds the
        // pipes this command's output is read through.
        let _ = kill_process_group(pid, Signal::KILL);
    }

    let _ = child.kill().await;
    let _ = child.wait().await;
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Whether the Bash probe proved the interpreter.
///
/// The marker must be the whole of standard output, apart from surrounding
/// whitespace: a shell that printed a banner first, or anything after the
/// marker, is a shell that sources something a command would inherit.
fn bash_probe_passed(result: &ExecResult) -> bool {
    result.is_success() && result.stdout.trim() == BASH_PROBE_MARKER
}

fn first_line(text: &str) -> Option<&str> {
    text.lines().map(str::trim).find(|line| !line.is_empty())
}

/// Finds an executable on `PATH`.
fn binary_path_on_path(binary: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;

    #[cfg(windows)]
    let extensions: Vec<String> = env::var_os("PATHEXT").map_or_else(
        || vec![".exe".to_owned(), ".cmd".to_owned(), ".bat".to_owned()],
        |value| {
            value
                .to_string_lossy()
                .split(';')
                .map(str::to_ascii_lowercase)
                .collect()
        },
    );

    for directory in env::split_paths(&paths) {
        let candidate = directory.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }

        #[cfg(windows)]
        if candidate.extension().is_none() {
            for extension in &extensions {
                let with_extension = directory.join(format!("{binary}{extension}"));
                if with_extension.is_file() {
                    return Some(with_extension);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::symlink;

    use super::*;
    use crate::tools::testing::TempDir;

    fn environment(directory: &TempDir) -> LocalEnvironment {
        LocalEnvironment::new(directory.path())
    }

    /// Bash-only syntax: an array literal and an indexed expansion.
    const BASH_ONLY_COMMAND: &str = "arr=(one two three); [[ ${#arr[@]} -eq 3 ]] && echo ${arr[1]}";

    /// Prints `login` or `nonlogin` for the shell evaluating it.
    const LOGIN_SHELL_REPORT: &str = "shopt -q login_shell && echo login || echo nonlogin";

    #[tokio::test]
    async fn read_file_numbers_its_lines() {
        let directory = TempDir::new("local-env");
        directory.write("test.txt", "hello\nworld\nfoo");

        let read = environment(&directory)
            .read_file("test.txt", None, None)
            .await
            .expect("file is readable");

        assert_eq!(read, "1 | hello\n2 | world\n3 | foo\n");
    }

    #[tokio::test]
    async fn read_file_pads_line_numbers_to_the_widest() {
        let directory = TempDir::new("local-env");
        let content = (1..=12)
            .map(|number| format!("line {number}"))
            .collect::<Vec<_>>()
            .join("\n");
        directory.write("padded.txt", &content);

        let read = environment(&directory)
            .read_file("padded.txt", None, None)
            .await
            .expect("file is readable");

        assert!(read.starts_with(" 1 | line 1\n"), "{read}");
        assert!(read.contains("12 | line 12\n"), "{read}");
    }

    #[tokio::test]
    async fn reading_a_missing_file_reports_not_found() {
        let directory = TempDir::new("local-env");

        let error = environment(&directory)
            .read_file("nonexistent.txt", None, None)
            .await
            .expect_err("the file does not exist");

        assert_eq!(error.kind(), EnvironmentErrorKind::NotFound);
        assert!(error.message().contains("nonexistent.txt"), "{error}");
        assert!(error.detail().contains("caused by"), "{}", error.detail());
    }

    #[tokio::test]
    async fn reading_bytes_that_are_not_text_reports_invalid_utf8() {
        let directory = TempDir::new("local-env");
        sync_fs::write(directory.join("binary.bin"), [0xff_u8, 0xfe]).expect("fixture is writable");

        let error = environment(&directory)
            .read_file_text("binary.bin")
            .await
            .expect_err("the bytes are not text");

        assert_eq!(error.kind(), EnvironmentErrorKind::InvalidUtf8);
    }

    #[tokio::test]
    async fn writing_creates_missing_parent_directories() {
        let directory = TempDir::new("local-env");

        environment(&directory)
            .write_file("sub/dir/test.txt", "content")
            .await
            .expect("the file is writable");

        let written =
            sync_fs::read_to_string(directory.join("sub/dir/test.txt")).expect("file exists");
        assert_eq!(written, "content");
    }

    #[tokio::test]
    async fn writing_an_existing_file_replaces_its_content() {
        let directory = TempDir::new("local-env");
        directory.write("test.txt", "before");

        environment(&directory)
            .write_existing_file("test.txt", "after")
            .await
            .expect("the file is writable");

        assert_eq!(
            sync_fs::read_to_string(directory.join("test.txt")).expect("file exists"),
            "after"
        );
    }

    #[tokio::test]
    async fn deleting_removes_the_file() {
        let directory = TempDir::new("local-env");
        directory.write("gone.txt", "data");
        let environment = environment(&directory);

        environment
            .delete_file("gone.txt")
            .await
            .expect("the file is deletable");

        assert!(!environment.file_exists("gone.txt").await.expect("checked"));
    }

    #[tokio::test]
    async fn existence_reflects_the_filesystem() {
        let directory = TempDir::new("local-env");
        directory.write("exists.txt", "data");
        let environment = environment(&directory);

        assert!(
            environment
                .file_exists("exists.txt")
                .await
                .expect("checked")
        );
        assert!(!environment.file_exists("nope.txt").await.expect("checked"));
    }

    #[tokio::test]
    async fn a_listing_is_sorted_and_marks_directories() {
        let directory = TempDir::new("local-env");
        directory.write("b.txt", "b");
        directory.write("a.txt", "a");
        sync_fs::create_dir(directory.join("c_dir")).expect("directory is creatable");

        let entries = environment(&directory)
            .list_directory(".", None)
            .await
            .expect("the directory is listable");

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "a.txt");
        assert!(!entries[0].is_dir);
        assert!(entries[0].size.is_some());
        assert_eq!(entries[1].name, "b.txt");
        assert_eq!(entries[2].name, "c_dir");
        assert!(entries[2].is_dir);
        assert!(entries[2].size.is_none());
    }

    #[tokio::test]
    async fn a_deeper_listing_joins_nested_names_with_slashes() {
        let directory = TempDir::new("local-env");
        directory.write("nested/inner.txt", "x");

        let entries = environment(&directory)
            .list_directory(".", Some(2))
            .await
            .expect("the directory is listable");

        let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["nested", "nested/inner.txt"]);
    }

    #[test]
    fn the_platform_is_one_pebble_names() {
        let environment = LocalEnvironment::new("/tmp");

        assert!(
            matches!(environment.platform(), "darwin" | "linux" | "windows"),
            "unknown platform: {}",
            environment.platform()
        );
    }

    #[test]
    fn the_os_version_names_the_platform() {
        let environment = LocalEnvironment::new("/tmp");

        assert!(
            environment.os_version().contains(environment.platform()),
            "{}",
            environment.os_version()
        );
    }

    #[test]
    fn the_working_directory_is_reported_as_given() {
        let environment = LocalEnvironment::new("/tmp/test_dir");

        assert_eq!(environment.working_directory(), "/tmp/test_dir");
    }

    #[test]
    fn credential_shaped_variable_names_are_filtered() {
        let environment = LocalEnvironment::new("/tmp");

        for filtered in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DB_PASSWORD",
            "AWS_SECRET",
            "AUTH_TOKEN",
            "MY_CREDENTIAL",
            "SESSION_SECRET",
            "my_api_key",
            "Some_Secret",
        ] {
            assert!(environment.filters_env_var(filtered), "{filtered}");
        }

        for kept in ["PATH", "HOME", "EDITOR", "SECRET_PATH"] {
            assert!(!environment.filters_env_var(kept), "{kept}");
        }
    }

    #[test]
    fn a_custom_safelist_replaces_the_default_one() {
        let environment = LocalEnvironment::new("/tmp").with_env_safelist(["MY_API_KEY"]);

        assert!(!environment.filters_env_var("MY_API_KEY"));
        assert!(environment.filters_env_var("OTHER_API_KEY"));
    }

    /// A shell that prints anything besides the marker sources something a
    /// command would inherit, so only the exact marker proves the interpreter.
    #[test]
    fn the_bash_probe_accepts_only_the_exact_marker() {
        fn probe_output(stdout: &str, exit_code: i32) -> ExecResult {
            ExecResult {
                stdout:      stdout.to_owned(),
                stderr:      String::new(),
                exit_code:   Some(exit_code),
                termination: CommandTermination::Exited,
                duration_ms: 1,
            }
        }

        assert!(bash_probe_passed(&probe_output(
            &format!("  {BASH_PROBE_MARKER}\n"),
            0
        )));
        assert!(!bash_probe_passed(&probe_output(
            &format!("prefix-{BASH_PROBE_MARKER}-suffix"),
            0
        )));
        assert!(!bash_probe_passed(&probe_output(
            &format!("{BASH_PROBE_MARKER}\nunexpected output"),
            0
        )));
        assert!(!bash_probe_passed(&probe_output(BASH_PROBE_MARKER, 1)));
    }

    /// The command still ran, so a pipe that fails part way through keeps what
    /// it read instead of failing the call.
    #[tokio::test]
    async fn a_pipe_that_fails_while_reading_keeps_what_it_read() {
        use std::pin::Pin;
        use std::task::{Context as TaskContext, Poll};

        use tokio::io::ReadBuf;

        struct FailingReader {
            wrote: bool,
        }

        impl AsyncRead for FailingReader {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _context: &mut TaskContext<'_>,
                buffer: &mut ReadBuf<'_>,
            ) -> Poll<io::Result<()>> {
                if self.wrote {
                    return Poll::Ready(Err(io::Error::other("simulated read failure")));
                }
                self.wrote = true;
                buffer.put_slice(b"partial");
                Poll::Ready(Ok(()))
            }
        }

        let captured = drain_pipe(
            Some(FailingReader { wrote: false }),
            None,
            CancellationToken::new(),
        )
        .await;

        let (bytes, stats) = captured.into_parts();
        assert_eq!(
            String::from_utf8(bytes).expect("retained bytes are text"),
            "partial"
        );
        assert_eq!(stats.observed_bytes, 7);
    }

    #[cfg(unix)]
    mod commands {
        use std::collections::HashMap;
        use std::ffi::OsStr;

        use tokio_util::sync::CancellationToken;

        use super::*;
        use crate::event::OutputCaptureStats;

        #[tokio::test]
        async fn a_command_reports_its_output_and_status() {
            let directory = TempDir::new("local-env");

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    ..ExecRequest::new("echo hello")
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.stdout.trim(), "hello");
            assert_eq!(outcome.result.exit_code, Some(0));
            assert_eq!(outcome.result.termination, CommandTermination::Exited);
            assert!(outcome.result.is_success());
            assert!(outcome.streams_separated);
            assert_eq!(
                outcome.output_capture(),
                OutputCaptureStats::complete(outcome.result.stdout.len())
            );
        }

        #[tokio::test]
        async fn a_command_runs_under_bash_not_a_posix_shell() {
            let directory = TempDir::new("local-env");

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    ..ExecRequest::new(BASH_ONLY_COMMAND)
                })
                .await
                .expect("the command runs");

            assert_eq!(
                outcome.result.exit_code,
                Some(0),
                "stderr: {}",
                outcome.result.stderr
            );
            assert_eq!(outcome.result.stdout.trim(), "two");
        }

        #[tokio::test]
        async fn a_command_runs_under_a_non_login_shell_with_no_startup_source() {
            let directory = TempDir::new("local-env");
            directory.write("bash-env", "printf 'startup-source-loaded\\n'\n");
            let env_vars = HashMap::from([(
                BASH_ENV_VAR.to_owned(),
                directory.join("bash-env").to_string_lossy().into_owned(),
            )]);

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    env_vars: Some(&env_vars),
                    ..ExecRequest::new(LOGIN_SHELL_REPORT)
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.stdout.trim(), "nonlogin");
            assert!(
                !outcome.result.stdout.contains("startup-source-loaded"),
                "{}",
                outcome.result.stdout
            );
        }

        #[tokio::test]
        async fn preparing_accepts_the_local_bash() {
            let directory = TempDir::new("local-env");

            environment(&directory)
                .prepare()
                .await
                .expect("bash is usable");
        }

        /// Writes an executable script and answers its path.
        fn write_shim(directory: &TempDir, name: &str, script: &str) -> PathBuf {
            use std::os::unix::fs::PermissionsExt as _;

            directory.write(name, script);
            let path = directory.join(name);
            let mut permissions = sync_fs::metadata(&path)
                .expect("the shim exists")
                .permissions();
            permissions.set_mode(0o755);
            sync_fs::set_permissions(&path, permissions).expect("the shim is executable");
            path
        }

        #[tokio::test]
        async fn preparing_fails_when_the_machine_has_no_bash() {
            let directory = TempDir::new("local-env");
            let environment = LocalEnvironment::with_bash_executable(
                directory.path(),
                BashExecutable::Unavailable,
            );

            let error = environment
                .prepare()
                .await
                .expect_err("a machine without bash cannot be prepared");

            assert_eq!(error.kind(), EnvironmentErrorKind::Spawn);
            assert!(error.to_string().contains("Install bash"), "{error}");
        }

        #[tokio::test]
        async fn preparing_fails_when_the_interpreter_is_not_bash() {
            let directory = TempDir::new("local-env");
            // `sh` is Bash in POSIX mode on macOS and dash on most Linux
            // images. The probe rejects both, for different reasons.
            let shim = write_shim(&directory, "not-bash", "#!/bin/sh\nexec /bin/sh \"$@\"\n");
            let environment = LocalEnvironment::with_bash_executable(
                directory.path(),
                BashExecutable::Fixed(shim),
            );

            let error = environment
                .prepare()
                .await
                .expect_err("sh is not usable as bash");

            assert_eq!(error.kind(), EnvironmentErrorKind::Spawn);
            let message = error.to_string();
            assert!(
                message.contains("not usable as non-login Bash"),
                "{message}"
            );
            assert!(
                message.contains("interpreter is bash in posix mode")
                    || message.contains("interpreter is not bash"),
                "{message}"
            );
        }

        /// The probe's output is a command's output. Only the line naming the
        /// broken part of the contract belongs in an error a caller logs.
        #[tokio::test]
        async fn a_failed_probe_quotes_only_its_first_diagnostic() {
            let directory = TempDir::new("local-env");
            let shim = write_shim(
                &directory,
                "chatty",
                "#!/bin/sh\necho 'first diagnostic' >&2\necho 'second diagnostic' >&2\nexit 1\n",
            );
            let environment = LocalEnvironment::with_bash_executable(
                directory.path(),
                BashExecutable::Fixed(shim),
            );

            let error = environment
                .prepare()
                .await
                .expect_err("a failing probe fails preparation");

            let message = error.to_string();
            assert!(message.contains("first diagnostic"), "{message}");
            assert!(!message.contains("second diagnostic"), "{message}");
        }

        #[tokio::test]
        async fn preparing_creates_a_missing_working_directory() {
            let directory = TempDir::new("local-env");
            let nested = directory.join("created/by/prepare");
            let environment = LocalEnvironment::new(&nested);

            environment.prepare().await.expect("bash is usable");

            assert!(nested.exists());
        }

        #[tokio::test]
        async fn a_failing_command_reports_its_exit_code() {
            let directory = TempDir::new("local-env");

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    ..ExecRequest::new("exit 42")
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.exit_code, Some(42));
            assert_eq!(outcome.result.termination, CommandTermination::Exited);
            assert!(!outcome.result.is_success());
        }

        #[tokio::test]
        async fn standard_error_is_captured_separately() {
            let directory = TempDir::new("local-env");

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    ..ExecRequest::new("echo err >&2")
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.stderr.trim(), "err");
            assert!(outcome.result.stdout.is_empty());
        }

        #[tokio::test]
        async fn a_command_that_overruns_its_timeout_is_stopped() {
            let directory = TempDir::new("local-env");

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(200),
                    ..ExecRequest::new("sleep 10")
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.termination, CommandTermination::TimedOut);
            assert_eq!(outcome.result.exit_code, None);
        }

        #[tokio::test]
        async fn a_cancelled_command_is_stopped() {
            let directory = TempDir::new("local-env");
            let cancel = CancellationToken::new();
            cancel.cancel();

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    cancel_token: Some(cancel),
                    ..ExecRequest::new("sleep 10")
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.termination, CommandTermination::Cancelled);
            assert_eq!(outcome.result.exit_code, None);
        }

        #[tokio::test]
        async fn a_descendant_that_ignores_the_signal_cannot_hold_the_call_open() {
            // The shell and the grandchild both refuse SIGTERM, and the
            // grandchild inherited the pipes. Nothing here reaches end of file
            // on its own, so the call returns only because the group is killed
            // and the drain is bounded.
            let directory = TempDir::new("local-env");

            let outcome = time::timeout(
                Duration::from_secs(20),
                environment(&directory).exec(ExecRequest {
                    timeout_ms: Some(200),
                    ..ExecRequest::new(
                        "trap '' TERM; ( trap '' TERM; sleep 60 ) & echo started; sleep 60",
                    )
                }),
            )
            .await
            .expect("a stopped command does not wait on what outlived it")
            .expect("the command runs");

            assert_eq!(outcome.result.termination, CommandTermination::TimedOut);
            assert_eq!(
                outcome.result.stdout.trim(),
                "started",
                "what was written before the kill is still reported"
            );
        }

        #[tokio::test]
        async fn output_is_drained_past_the_retention_cap() {
            let directory = TempDir::new("local-env");

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    output_bytes_cap: Some(8),
                    ..ExecRequest::new("printf 'abcdefghijklmnopqrst'")
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.stdout, "abcdqrst");
            assert_eq!(outcome.stdout_capture, OutputCaptureStats {
                observed_bytes: 20,
                retained_bytes: 8,
                omitted_bytes:  12,
            });
        }

        #[tokio::test]
        async fn a_command_writing_more_than_a_pipe_holds_still_finishes() {
            let directory = TempDir::new("local-env");

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(30_000),
                    output_bytes_cap: Some(1024),
                    ..ExecRequest::new("head -c 400000 /dev/zero | tr '\\0' 'x'")
                })
                .await
                .expect("the command runs");

            assert_eq!(outcome.result.termination, CommandTermination::Exited);
            assert_eq!(outcome.stdout_capture.observed_bytes, 400_000);
            assert_eq!(outcome.stdout_capture.retained_bytes, 1024);
        }

        #[tokio::test]
        async fn a_command_runs_in_the_working_directory_by_default() {
            let directory = TempDir::new("local-env");
            sync_fs::create_dir(directory.join("elsewhere")).expect("directory is creatable");
            let environment = environment(&directory);

            let default_directory = environment
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    ..ExecRequest::new("pwd")
                })
                .await
                .expect("the command runs");
            let explicit_directory = environment
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    working_dir: Some(directory.join("elsewhere").to_str().expect("path is text")),
                    ..ExecRequest::new("pwd")
                })
                .await
                .expect("the command runs");

            assert!(
                default_directory.result.stdout.trim().ends_with(
                    directory
                        .path()
                        .file_name()
                        .and_then(OsStr::to_str)
                        .expect("directory has a name")
                ),
                "{}",
                default_directory.result.stdout
            );
            assert!(
                explicit_directory
                    .result
                    .stdout
                    .trim()
                    .ends_with("elsewhere"),
                "{}",
                explicit_directory.result.stdout
            );
        }

        #[tokio::test]
        async fn credential_shaped_caller_variables_are_filtered() {
            let directory = TempDir::new("local-env");
            let env_vars = HashMap::from([
                ("PEBBLE_WORKER_TOKEN".to_owned(), "leaked".to_owned()),
                ("MY_VAR".to_owned(), "ok".to_owned()),
            ]);

            let outcome = environment(&directory)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    env_vars: Some(&env_vars),
                    ..ExecRequest::new("env")
                })
                .await
                .expect("the command runs");

            assert!(!outcome.result.stdout.contains("PEBBLE_WORKER_TOKEN=leaked"));
            assert!(outcome.result.stdout.contains("MY_VAR=ok"));
        }

        #[tokio::test]
        async fn a_trusting_environment_passes_caller_variables_through() {
            let directory = TempDir::new("local-env");
            let env_vars = HashMap::from([("PEBBLE_WORKER_TOKEN".to_owned(), "wanted".to_owned())]);

            let outcome = LocalEnvironment::new(directory.path())
                .with_caller_env_policy(CallerEnvPolicy::Trusted)
                .exec(ExecRequest {
                    timeout_ms: Some(5_000),
                    env_vars: Some(&env_vars),
                    ..ExecRequest::new("env")
                })
                .await
                .expect("the command runs");

            assert!(outcome.result.stdout.contains("PEBBLE_WORKER_TOKEN=wanted"));
        }

        #[tokio::test]
        async fn grep_finds_matching_lines() {
            let directory = TempDir::new("local-env");
            directory.write("test.rs", "fn main() {\n    println!(\"hello\");\n}\n");

            let matches = environment(&directory)
                .grep("println", "test.rs", &GrepOptions::default())
                .await
                .expect("the search runs");

            assert_eq!(matches.len(), 1);
            assert!(matches[0].contains("println"), "{:?}", matches[0]);
        }

        #[tokio::test]
        async fn grep_can_ignore_case() {
            let directory = TempDir::new("local-env");
            directory.write("test.txt", "Hello\nhello\nHELLO\n");

            let matches = environment(&directory)
                .grep("hello", "test.txt", &GrepOptions {
                    case_insensitive: true,
                    ..GrepOptions::default()
                })
                .await
                .expect("the search runs");

            assert_eq!(matches.len(), 3);
        }

        #[tokio::test]
        async fn grep_stops_at_the_result_limit() {
            let directory = TempDir::new("local-env");
            directory.write("test.txt", "match1\nmatch2\nmatch3\nmatch4\n");

            let matches = environment(&directory)
                .grep("match", "test.txt", &GrepOptions {
                    max_results: Some(2),
                    ..GrepOptions::default()
                })
                .await
                .expect("the search runs");

            assert_eq!(matches.len(), 2);
        }
    }

    #[tokio::test]
    async fn glob_matches_within_one_segment() {
        let directory = TempDir::new("local-env");
        directory.write("a.rs", "");
        directory.write("b.rs", "");
        directory.write("c.txt", "");

        let matches = environment(&directory)
            .glob("*.rs", None)
            .await
            .expect("the pattern is valid");

        assert_eq!(matches, vec![
            path_string(&directory.join("a.rs")),
            path_string(&directory.join("b.rs")),
        ]);
    }

    #[tokio::test]
    async fn glob_resolves_a_relative_search_path_against_the_working_directory() {
        let directory = TempDir::new("local-env");
        directory.write("src/lib.rs", "");

        let matches = environment(&directory)
            .glob("*.rs", Some("src"))
            .await
            .expect("the pattern is valid");

        assert_eq!(matches, vec![path_string(&directory.join("src/lib.rs"))]);
    }

    #[tokio::test]
    async fn a_recursive_pattern_finds_files_at_any_depth() {
        let directory = TempDir::new("local-env");
        directory.write("a.rs", "");
        directory.write("src/lib.rs", "");
        directory.write("src/nested/main.rs", "");
        directory.write("src/nested/readme.md", "");

        let matches = environment(&directory)
            .glob("**/*.rs", None)
            .await
            .expect("the pattern is valid");

        assert_eq!(matches, vec![
            path_string(&directory.join("a.rs")),
            path_string(&directory.join("src/lib.rs")),
            path_string(&directory.join("src/nested/main.rs")),
        ]);
    }

    #[tokio::test]
    async fn a_single_segment_pattern_finds_files_one_level_down() {
        let directory = TempDir::new("local-env");
        directory.write("skills/SKILL.md", "");
        directory.write("skills/patch/SKILL.md", "");
        directory.write("skills/nested/deeper/SKILL.md", "");
        let skills = directory.join("skills");

        let matches = environment(&directory)
            .glob("*/SKILL.md", Some(skills.to_str().expect("path is text")))
            .await
            .expect("the pattern is valid");

        assert_eq!(matches, vec![path_string(&skills.join("patch/SKILL.md"))]);
    }

    #[tokio::test]
    async fn an_invalid_pattern_is_reported_as_bad_input() {
        let directory = TempDir::new("local-env");

        let error = environment(&directory)
            .glob("/absolute/*.rs", None)
            .await
            .expect_err("the pattern reaches outside the base");

        assert_eq!(error.kind(), EnvironmentErrorKind::InvalidInput);
        assert!(error.detail().contains("must be relative"), "{error}");
    }

    #[tokio::test]
    async fn a_glob_over_a_missing_directory_finds_nothing() {
        let directory = TempDir::new("local-env");

        let matches = environment(&directory)
            .glob("*.rs", Some("absent"))
            .await
            .expect("the pattern is valid");

        assert!(matches.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_glob_does_not_follow_symlinked_directories() {
        let directory = TempDir::new("local-env");
        directory.write("target/lib.rs", "");
        symlink(directory.join("target"), directory.join("linked")).expect("symlink is creatable");

        let matches = environment(&directory)
            .glob("linked/**/*.rs", None)
            .await
            .expect("the pattern is valid");

        assert!(matches.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_glob_follows_only_the_declared_symlinked_root() {
        let directory = TempDir::new("local-env");
        directory.write("workspace/README.md", "");
        directory.write("outside/outside.md", "");
        let search_root = directory.join("workspace-link");
        symlink(
            directory.join("outside"),
            directory.join("workspace/linked"),
        )
        .expect("symlink is creatable");
        symlink(directory.join("workspace"), &search_root).expect("symlink is creatable");

        let matches = LocalEnvironment::new(&search_root)
            .glob("**/*.md", None)
            .await
            .expect("the pattern is valid");

        assert_eq!(matches, vec![path_string(&search_root.join("README.md"))]);
    }

    fn path_string(path: &Path) -> String {
        path.to_string_lossy().into_owned()
    }
}
