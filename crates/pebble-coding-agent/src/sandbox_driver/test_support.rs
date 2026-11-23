//! Test doubles for code that runs over a [`SandboxEnvironment`].
//!
//! [`MockSandbox`] is a configuration over a scripted sandbox-driver handle,
//! [`ScriptedSandbox`]: a test writes down the files, the command answer, and
//! the failures it wants, and takes a [`SandboxEnvironment`] or the bare
//! driver handle from it. What the code under test ran or wrote is read back
//! from the scripted handle through [`MockSandbox::driver`]. Nothing here
//! fakes the adapter's own logic: every call goes through the real
//! [`SandboxEnvironment`] and its exec policy, down to the scripted handle.
//!
//! Parity gap: the old driver's `sandbox-driver-testing` doubles answered a
//! command by a closure, from a queue, or by default, and seeded an in-memory
//! filesystem. The new fake (`FakeProvider`) answers only exact argv it was
//! scripted with and fails every other command, so the scripted handle lives
//! here, implementing the new [`Sandbox`] trait.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Cursor;
use std::num::NonZeroU16;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use sandbox_driver::{
    ByteStream, Error as DriverError, ExecControls, ExecOutcome, ExecSpec, Expose, Exposure,
    FileEntry, FileKind, FileRange, Files, MutationOutcome, NamespaceId, OutputLoss, OutputStream,
    Ports, ProviderName, Resources, Sandbox, SandboxId, SandboxRef, SandboxState, Status,
    Termination,
};
use tokio::io::AsyncReadExt as _;

use super::environment::SandboxEnvironment;
use super::exec::{BASH_ENV_VAR, ExecResult, bash_source};
use super::path::resolve_path;

/// A driver-facing [`ExecResult`] with the given streams, for scripting a
/// mock sandbox's answers.
#[must_use]
pub fn exec_result(
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
    termination: Termination,
    duration_ms: u64,
) -> ExecResult {
    let mut result = ExecResult::new(termination, exit_code, Duration::from_millis(duration_ms));
    result.stdout = stdout.as_bytes().to_vec();
    result.stderr = stderr.as_bytes().to_vec();
    result
}

/// What a test wants its sandbox to be, and what the code under test did
/// with it.
///
/// Build it with a struct literal over [`MockSandbox::default`] (or
/// [`MockSandbox::linux`]), then take the sandbox with
/// [`MockSandbox::sandbox`] or its driver handle with
/// [`MockSandbox::handle`]. Every command answers with `exec_result` unless
/// `exec_error` is set, in which case every command fails as a transport
/// error. Files seed an in-memory filesystem under `working_dir`; absolute
/// paths are kept as given.
pub struct MockSandbox {
    pub files:             HashMap<String, String>,
    pub exec_result:       ExecResult,
    /// Fails every command before any process runs, so callers see a
    /// transport error rather than a result.
    pub exec_error:        Option<String>,
    pub working_dir:       &'static str,
    pub platform_str:      &'static str,
    pub os_version_str:    String,
    /// Lines every grep returns, as `path:line:content`.
    pub grep_results:      Vec<String>,
    /// Set to `false` to model a provider that cannot separate stdout from
    /// stderr: output then arrives as one terminal stream.
    pub streams_separated: bool,
    /// The sandbox once built. Public only so `..Default::default()` works
    /// from other crates; leave it at its default.
    pub built:             OnceLock<Built>,
}

/// The lazily built sandbox and its scripted driver.
pub struct Built {
    sandbox: Arc<SandboxEnvironment>,
    driver:  Arc<ScriptedSandbox>,
}

impl Default for MockSandbox {
    fn default() -> Self {
        Self {
            files:             HashMap::new(),
            exec_result:       exec_result("mock output", "", Some(0), Termination::Exited, 10),
            exec_error:        None,
            working_dir:       "/work",
            platform_str:      "darwin",
            os_version_str:    "Darwin 24.0.0".into(),
            grep_results:      Vec::new(),
            streams_separated: true,
            built:             OnceLock::new(),
        }
    }
}

impl MockSandbox {
    #[must_use]
    pub fn linux() -> Self {
        Self {
            working_dir: "/home/test",
            platform_str: "linux",
            os_version_str: "Linux 6.1.0".into(),
            ..Self::default()
        }
    }

    /// The environment this configuration describes, built once: repeated
    /// calls return the same sandbox over the same recorder.
    pub fn sandbox(&self) -> Arc<SandboxEnvironment> {
        Arc::clone(&self.built().sandbox)
    }

    /// The scripted driver as a bare sandbox handle, for code that takes
    /// `&dyn Sandbox` beside a working directory.
    pub fn handle(&self) -> Arc<dyn Sandbox> {
        Arc::clone(&self.built().driver) as Arc<dyn Sandbox>
    }

    /// The scripted driver behind [`MockSandbox::sandbox`], for scripting
    /// beyond what the fields express.
    pub fn driver(&self) -> Arc<ScriptedSandbox> {
        Arc::clone(&self.built().driver)
    }

    /// Answers commands by their Bash source, ahead of the queue and
    /// `exec_result`: a responder that returns `Some` decides the result,
    /// `None` falls through.
    pub fn respond_with(
        &self,
        responder: impl Fn(&str) -> Option<ExecResult> + Send + Sync + 'static,
    ) -> &Self {
        self.driver().scripted_exec().respond_with(move |spec| {
            let command = spec.argv.last().map(String::as_str).unwrap_or_default();
            responder(command)
        });
        self
    }

    fn built(&self) -> &Built {
        self.built.get_or_init(|| {
            let driver = Arc::new(self.build_driver());
            let sandbox = SandboxEnvironment::with_platform(
                Arc::clone(&driver) as Arc<dyn Sandbox>,
                self.working_dir,
                self.platform_str,
                self.os_version_str.clone(),
            );
            Built {
                sandbox: Arc::new(sandbox),
                driver,
            }
        })
    }

    fn build_driver(&self) -> ScriptedSandbox {
        let driver = ScriptedSandbox::new("mock-sandbox", self.working_dir);
        for (path, content) in &self.files {
            driver.seed_file(path, content.as_bytes());
        }
        let exec = driver.scripted_exec();
        match &self.exec_error {
            Some(message) => exec.fail_by_default(message.clone()),
            None => exec.set_default(self.exec_result.clone()),
        }
        exec.set_streams_separated(self.streams_separated);
        // The adapter greps by running ripgrep or grep, so the mock answers
        // those commands with the lines in grep's NUL-separated form.
        let mut grep_stdout = String::new();
        for line in &self.grep_results {
            let (path, rest) = line.split_once(':').unwrap_or((line, ""));
            grep_stdout.push_str(path);
            grep_stdout.push('\0');
            grep_stdout.push_str(rest);
            grep_stdout.push('\n');
        }
        exec.respond_with(move |spec| {
            let source = bash_source(spec)?;
            if source.starts_with("command -v rg") {
                return Some(exec_result("", "", Some(0), Termination::Exited, 0));
            }
            let searched = source.contains("rg -H --line-number") || source.contains("grep -rnIH");
            searched.then(|| exec_result(&grep_stdout, "", Some(0), Termination::Exited, 0))
        });
        driver
    }

    fn recorded(&self) -> Vec<ExecSpec> {
        self.built
            .get()
            .map(|built| built.driver.scripted_exec().recorded())
            .unwrap_or_default()
    }

    /// The last command's Bash source. Every command, in order, is
    /// `driver().scripted_exec().commands()`.
    pub fn captured_command(&self) -> Option<String> {
        self.recorded()
            .last()
            .and_then(|spec| spec.argv.last().cloned())
    }

    /// The explicit variables of the last command as the caller passed them.
    /// The policy's own `BASH_ENV` blank is not the caller's.
    pub fn captured_env_vars(&self) -> Option<HashMap<String, String>> {
        self.recorded().last().map(|spec| {
            spec.env
                .iter()
                .filter(|(key, value)| key.as_str() != BASH_ENV_VAR || !value.is_empty())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
    }

    /// Every file written so far as `(path, content)`, in order.
    pub fn written_files(&self) -> Vec<(String, String)> {
        self.built
            .get()
            .map(|built| {
                built
                    .driver
                    .writes()
                    .into_iter()
                    .map(|(path, bytes)| (path, String::from_utf8_lossy(&bytes).into_owned()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

type Responder = Box<dyn Fn(&ExecSpec) -> Option<ExecResult> + Send + Sync>;

/// How a [`ScriptedSandbox`] answers commands, and what it was asked.
///
/// A command is answered by the first responder that returns `Some` (newest
/// first), then by the next queued result, then by the default.
#[derive(Default)]
pub struct ScriptedExec {
    state: Mutex<ExecScript>,
}

#[derive(Default)]
struct ExecScript {
    responders:        Vec<Arc<Responder>>,
    queue:             VecDeque<ExecResult>,
    default:           Option<ExecResult>,
    failure:           Option<String>,
    streams_separated: bool,
    recorded:          Vec<ExecSpec>,
}

impl ScriptedExec {
    fn script(&self) -> MutexGuard<'_, ExecScript> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The answer when nothing else applies.
    pub fn set_default(&self, result: ExecResult) {
        let mut script = self.script();
        script.default = Some(result);
        script.failure = None;
    }

    /// Fails every command nothing else answers, as a transport error.
    pub fn fail_by_default(&self, message: impl Into<String>) {
        self.script().failure = Some(message.into());
    }

    /// Queues the answer to one command.
    pub fn push_result(&self, result: ExecResult) {
        self.script().queue.push_back(result);
    }

    /// Answers a command by its spec, ahead of the queue and the default.
    pub fn respond_with(
        &self,
        responder: impl Fn(&ExecSpec) -> Option<ExecResult> + Send + Sync + 'static,
    ) {
        self.script().responders.push(Arc::new(Box::new(responder)));
    }

    /// Whether output arrives as stdout and stderr, or as one terminal
    /// stream.
    pub fn set_streams_separated(&self, separated: bool) {
        self.script().streams_separated = separated;
    }

    /// Every command's spec, in order.
    pub fn recorded(&self) -> Vec<ExecSpec> {
        self.script().recorded.clone()
    }

    /// Every command, in order: a Bash script's source, or the argv joined
    /// by spaces.
    pub fn commands(&self) -> Vec<String> {
        self.recorded()
            .iter()
            .map(|spec| bash_source(spec).map_or_else(|| spec.argv.join(" "), str::to_string))
            .collect()
    }

    fn answer(&self, spec: &ExecSpec) -> Result<(ExecResult, bool), DriverError> {
        let responders = {
            let mut script = self.script();
            script.recorded.push(spec.clone());
            script.responders.clone()
        };
        let separated = self.script().streams_separated;
        for responder in responders.iter().rev() {
            if let Some(result) = responder(spec) {
                return Ok((result, separated));
            }
        }
        let mut script = self.script();
        if let Some(result) = script.queue.pop_front() {
            return Ok((result, separated));
        }
        if let Some(message) = &script.failure {
            return Err(DriverError::Transport {
                message: message.clone(),
                outcome: MutationOutcome::NotStarted,
            });
        }
        Ok((
            script
                .default
                .clone()
                .unwrap_or_else(|| ExecResult::new(Termination::Exited, Some(0), Duration::ZERO)),
            separated,
        ))
    }
}

/// A scripted sandbox-driver handle: an in-memory filesystem, commands
/// answered by [`ScriptedExec`], and ports that refuse.
pub struct ScriptedSandbox {
    reference:   SandboxRef,
    working_dir: PathBuf,
    env:         BTreeMap<String, String>,
    exec:        ScriptedExec,
    files:       Mutex<BTreeMap<String, Vec<u8>>>,
    writes:      Mutex<Vec<(String, Vec<u8>)>>,
}

impl ScriptedSandbox {
    /// A running sandbox `id` working in `working_dir`.
    ///
    /// # Panics
    /// When `id` is not a valid sandbox id.
    #[must_use]
    pub fn new(id: &str, working_dir: &str) -> Self {
        Self {
            reference:   SandboxRef {
                provider:  ProviderName::try_new("mock").expect("constant name"),
                namespace: NamespaceId::try_new("mock:1").expect("constant namespace"),
                id:        SandboxId::try_new(id).expect("a valid sandbox id"),
            },
            working_dir: PathBuf::from(working_dir),
            env:         BTreeMap::new(),
            exec:        ScriptedExec::default(),
            files:       Mutex::new(BTreeMap::new()),
            writes:      Mutex::new(Vec::new()),
        }
    }

    /// How commands are answered, and what ran.
    pub fn scripted_exec(&self) -> &ScriptedExec {
        &self.exec
    }

    /// Puts a file in place without recording it as written.
    pub fn seed_file(&self, path: &str, content: &[u8]) {
        let path = self.resolve(path);
        self.lock_files().insert(path, content.to_vec());
    }

    /// Every file `put` so far as `(path, bytes)`, in order.
    pub fn writes(&self) -> Vec<(String, Vec<u8>)> {
        self.writes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn resolve(&self, path: &str) -> String {
        resolve_path(path, &self.working_dir.to_string_lossy())
    }

    fn lock_files(&self) -> MutexGuard<'_, BTreeMap<String, Vec<u8>>> {
        self.files.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The adapter tests a path with `[ -e '<path>' ]`, followed by
    /// `|| [ -L '<path>' ]` before a delete; answering that from the
    /// in-memory filesystem, which has no links, keeps file existence real
    /// in the mock.
    fn exists_answer(&self, spec: &ExecSpec) -> Option<ExecResult> {
        let source = bash_source(spec)?;
        let test = source.split(" || [ -L ").next()?;
        let quoted = test.strip_prefix("[ -e '")?.strip_suffix("' ]")?;
        let path = quoted.replace("'\\''", "'");
        let prefix = format!("{}/", path.trim_end_matches('/'));
        let files = self.lock_files();
        let exists = files.contains_key(&path) || files.keys().any(|key| key.starts_with(&prefix));
        let code = i32::from(!exists);
        Some(ExecResult::new(
            Termination::Exited,
            Some(code),
            Duration::ZERO,
        ))
    }
}

#[async_trait]
impl Sandbox for ScriptedSandbox {
    fn reference(&self) -> SandboxRef {
        self.reference.clone()
    }

    fn working_directory(&self) -> &Path {
        &self.working_dir
    }

    fn env(&self) -> &BTreeMap<String, String> {
        &self.env
    }

    async fn status(&self) -> Result<Status, DriverError> {
        Ok(Status {
            reference: self.reference(),
            state:     SandboxState::Running,
            name:      None,
            labels:    BTreeMap::new(),
            image:     None,
            snapshot:  None,
            resources: Resources::default(),
            network:   None,
            services:  BTreeMap::new(),
            web_url:   None,
        })
    }

    async fn start(&self) -> Result<(), DriverError> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), DriverError> {
        Ok(())
    }

    async fn exec(
        &self,
        spec: ExecSpec,
        controls: ExecControls,
    ) -> Result<ExecOutcome, DriverError> {
        let (result, separated) = match self.exists_answer(&spec) {
            Some(result) => {
                self.exec.script().recorded.push(spec);
                (result, true)
            }
            None => self.exec.answer(&spec)?,
        };
        if let Some(mut stdin) = controls.stdin {
            let mut ignored = Vec::new();
            let _ = stdin.read_to_end(&mut ignored).await;
        }
        if let Some(sink) = controls.sink {
            if separated {
                for (stream, bytes) in [
                    (OutputStream::Stdout, result.stdout),
                    (OutputStream::Stderr, result.stderr),
                ] {
                    if !bytes.is_empty() {
                        sink(stream, bytes).await?;
                    }
                }
            } else {
                let mut combined = result.stdout;
                combined.extend(result.stderr);
                if !combined.is_empty() {
                    sink(OutputStream::Terminal, combined).await?;
                }
            }
        }
        Ok(ExecOutcome {
            termination: result.termination,
            exit_code:   result.exit_code,
            signal:      result.signal,
            output_loss: OutputLoss::default(),
        })
    }

    fn files(&self) -> &dyn Files {
        self
    }

    fn ports(&self) -> &dyn Ports {
        self
    }
}

#[async_trait]
impl Files for ScriptedSandbox {
    async fn get(&self, path: &str, range: Option<FileRange>) -> Result<ByteStream, DriverError> {
        let path = self.resolve(path);
        let bytes = self
            .lock_files()
            .get(&path)
            .cloned()
            .ok_or_else(|| DriverError::NotFound {
                resource: "file".into(),
                id:       path,
            })?;
        let start = range.map_or(0, |range| {
            usize::try_from(range.offset)
                .unwrap_or(usize::MAX)
                .min(bytes.len())
        });
        let end = range
            .and_then(|range| range.length)
            .map_or(bytes.len(), |length| {
                start
                    .saturating_add(usize::try_from(length).unwrap_or(usize::MAX))
                    .min(bytes.len())
            });
        Ok(Box::pin(Cursor::new(bytes[start..end].to_vec())))
    }

    async fn put(&self, path: &str, mut bytes: ByteStream) -> Result<(), DriverError> {
        let path = self.resolve(path);
        let mut content = Vec::new();
        bytes
            .read_to_end(&mut content)
            .await
            .map_err(|error| DriverError::Transport {
                message: format!("reading the upload failed: {error}"),
                outcome: MutationOutcome::NotStarted,
            })?;
        self.lock_files().insert(path.clone(), content.clone());
        self.writes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((path, content));
        Ok(())
    }

    async fn list(&self, path: &str, depth: u32) -> Result<Vec<FileEntry>, DriverError> {
        let path = self.resolve(path);
        let prefix = format!("{}/", path.trim_end_matches('/'));
        let mut entries = BTreeMap::new();
        for (file, bytes) in self.lock_files().iter() {
            let Some(relative) = file.strip_prefix(&prefix) else {
                continue;
            };
            let parts: Vec<_> = relative.split('/').collect();
            let levels = parts.len().min((depth as usize).saturating_add(1));
            for index in 0..levels {
                let entry_path = format!("{prefix}{}", parts[..=index].join("/"));
                let (kind, size) = if index + 1 == parts.len() {
                    (FileKind::File, bytes.len() as u64)
                } else {
                    (FileKind::Directory, 0)
                };
                entries.insert(entry_path.clone(), FileEntry {
                    path: entry_path,
                    kind,
                    size,
                });
            }
        }
        Ok(entries.into_values().collect())
    }
}

#[async_trait]
impl Ports for ScriptedSandbox {
    async fn expose(&self, _: NonZeroU16, _: Expose) -> Result<Exposure, DriverError> {
        Err(DriverError::Unsupported {
            capability: "ports".into(),
        })
    }

    async fn release(&self, _: NonZeroU16) -> Result<(), DriverError> {
        Err(DriverError::Unsupported {
            capability: "ports".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{Environment, GrepOptions};

    #[tokio::test]
    async fn the_mock_answers_commands_and_records_what_ran() {
        let mock = MockSandbox {
            files: HashMap::from([("README.md".to_string(), "hello".to_string())]),
            ..MockSandbox::default()
        };
        let sandbox = mock.sandbox();
        assert_eq!(Environment::platform(&*sandbox), "darwin");
        assert_eq!(
            Environment::read_file_bytes(&*sandbox, "README.md")
                .await
                .unwrap(),
            b"hello"
        );
        assert!(
            Environment::file_exists(&*sandbox, "README.md")
                .await
                .unwrap()
        );
        assert!(
            !Environment::file_exists(&*sandbox, "missing.md")
                .await
                .unwrap()
        );
        Environment::write_file(&*sandbox, "notes.txt", "written")
            .await
            .unwrap();
        assert_eq!(mock.written_files(), vec![(
            "/work/notes.txt".to_string(),
            "written".to_string()
        )]);
        let env = HashMap::from([("KEY".to_string(), "value".to_string())]);
        let result = sandbox
            .exec()
            .run("echo hi", None, None, Some(&env), None)
            .await
            .unwrap();
        assert_eq!(result.stdout_lossy(), "mock output");
        assert_eq!(mock.captured_command().as_deref(), Some("echo hi"));
        assert_eq!(mock.captured_env_vars(), Some(env));
    }

    #[tokio::test]
    async fn responders_come_before_the_queue_and_the_queue_before_the_default() {
        let mock = MockSandbox {
            exec_error: Some("unexpected command".into()),
            ..MockSandbox::default()
        };
        mock.respond_with(|command| {
            (command == "special")
                .then(|| exec_result("answered", "", Some(0), Termination::Exited, 0))
        });
        mock.driver().scripted_exec().push_result(exec_result(
            "queued",
            "",
            Some(0),
            Termination::Exited,
            0,
        ));
        let exec = mock.sandbox();
        let run = |command: &'static str| {
            let exec = Arc::clone(&exec);
            async move { exec.exec().run(command, None, None, None, None).await }
        };
        assert_eq!(run("special").await.unwrap().stdout_lossy(), "answered");
        assert_eq!(run("first").await.unwrap().stdout_lossy(), "queued");
        assert!(run("second").await.is_err());
        assert_eq!(mock.driver().scripted_exec().commands(), [
            "special", "first", "second"
        ]);
    }

    #[tokio::test]
    async fn grep_results_come_back_through_the_derived_search() {
        let mock = MockSandbox {
            grep_results: vec!["/work/a.rs:3:let x = 1;".to_string()],
            ..MockSandbox::linux()
        };
        let lines = Environment::grep(&*mock.sandbox(), "x", ".", &GrepOptions::default())
            .await
            .unwrap();
        assert_eq!(lines, ["/work/a.rs:3:let x = 1;"]);
    }

    #[tokio::test]
    async fn a_combined_stream_is_stdout() {
        let mock = MockSandbox {
            exec_result: exec_result("out ", "err", Some(0), Termination::Exited, 0),
            streams_separated: false,
            ..MockSandbox::default()
        };
        let streaming = mock
            .sandbox()
            .exec()
            .run_streaming(ExecSpec::bash("x"), ExecControls::default(), None)
            .await
            .unwrap();
        assert!(!streaming.streams_separated);
        assert_eq!(streaming.result.stdout_lossy(), "out err");
        assert!(streaming.result.stderr.is_empty());
    }
}
