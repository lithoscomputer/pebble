//! Test doubles for code that runs over a [`SandboxEnvironment`].
//!
//! [`MockSandbox`] is a configuration over the sandbox driver's scripted
//! double: a test writes down the files, the command answer, and the
//! failures it wants, and takes a [`SandboxEnvironment`] or the bare driver
//! handle from it. What the code under test ran or wrote is read back from
//! the driver double itself, through [`MockSandbox::driver`]; the few
//! accessors here convert what a spec records into the shape an
//! application's tests assert on. Nothing here fakes the adapter's own
//! logic: every call goes through the real [`SandboxEnvironment`] and its
//! exec policy, down to the scripted driver.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use sandbox_driver::{
    BASH_ENV_VAR, ExecResult, ExecSpec, GrepMatch, PlatformInfo, Sandbox, Termination,
};
pub use sandbox_driver_testing::{ScriptedExec, ScriptedProvider, ScriptedSandbox};

use super::environment::SandboxEnvironment;

/// A driver [`ExecResult`] with the given streams, for scripting a mock
/// sandbox's answers.
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
    /// transport error rather than an `ExecResult`.
    pub exec_error:        Option<String>,
    pub working_dir:       &'static str,
    pub platform_str:      &'static str,
    pub os_version_str:    String,
    /// Lines every grep returns, as `path:line:content`.
    pub grep_results:      Vec<String>,
    /// Reported by streaming execution. Set to `false` to model a provider
    /// that cannot separate stdout from stderr.
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

    /// The scripted driver double behind [`MockSandbox::sandbox`], for
    /// scripting beyond what the fields express.
    pub fn driver(&self) -> Arc<ScriptedSandbox> {
        Arc::clone(&self.built().driver)
    }

    /// Answers commands by their Bash source, ahead of the queue and
    /// `exec_result`: a responder that returns `Some` decides the result,
    /// `None` falls through. For tests that interleave different commands
    /// and want each answered by what it is rather than by its position.
    pub fn respond_with(
        &self,
        responder: impl Fn(&str) -> Option<ExecResult> + Send + Sync + 'static,
    ) -> &Self {
        self.driver().scripted_exec().respond_with(move |spec| {
            let command = spec.args.last().map(String::as_str).unwrap_or_default();
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
        let mut driver =
            ScriptedSandbox::with_id_and_working_dir("mock-sandbox", self.working_dir).platform(
                PlatformInfo::new(self.platform_str, "x86_64", self.os_version_str.clone()),
            );
        for (path, content) in &self.files {
            driver = driver.file(path, content);
        }
        let exec = driver.scripted_exec();
        match &self.exec_error {
            Some(message) => exec.fail_by_default(message.clone()),
            None => exec.set_default(self.exec_result.clone()),
        };
        exec.set_streams_separated(self.streams_separated);
        driver.scripted_search().set_grep(
            self.grep_results
                .iter()
                .map(|line| {
                    let mut parts = line.splitn(3, ':');
                    let path = parts.next().unwrap_or_default();
                    let line_number = parts.next().and_then(|n| n.parse().ok()).unwrap_or(0);
                    GrepMatch::new(path, line_number, parts.next().unwrap_or_default())
                })
                .collect(),
        );
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
            .and_then(|spec| spec.args.last().cloned())
    }

    /// The explicit variables of the last command as the caller passed them.
    /// The driver's Bash helper records its own `BASH_ENV` blank on the
    /// spec; that is not the caller's.
    pub fn captured_env_vars(&self) -> Option<HashMap<String, String>> {
        self.recorded().last().map(|spec| {
            spec.env
                .iter()
                .filter(|(key, _)| key.as_str() != BASH_ENV_VAR)
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
                    .memory_fs()
                    .writes()
                    .into_iter()
                    .map(|(path, bytes)| (path, String::from_utf8_lossy(&bytes).into_owned()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;

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
}
