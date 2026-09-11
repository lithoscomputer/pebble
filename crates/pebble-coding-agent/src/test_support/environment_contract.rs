//! Reusable checks of the behavior coding tools require from an environment.

use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::mem::take;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::environment::{
    EnvResult, Environment, EnvironmentError, EnvironmentErrorKind, ExecOutputStream, ExecRequest,
    GrepOptions,
};
use crate::types::CommandTermination;

/// What a contract run's live-output sink records: each chunk with its stream.
type LiveChunks = Arc<Mutex<Vec<(ExecOutputStream, Vec<u8>)>>>;

/// A failed environment contract check.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EnvironmentContractError {
    /// An operation failed before its result could be checked.
    #[error("environment contract: {check}")]
    Operation {
        /// The behavior under test.
        check:  &'static str,
        /// The environment's original failure.
        #[source]
        source: EnvironmentError,
    },
    /// An operation returned a result that violates the contract.
    #[error("environment contract: {check}: expected {expected}, got {actual}")]
    Mismatch {
        /// The behavior under test.
        check:    &'static str,
        /// The expected test value.
        expected: String,
        /// The observed test value.
        actual:   String,
    },
    /// An operation did not finish within the configured limit.
    #[error("environment contract: {check} did not finish within {limit:?}")]
    TimedOut {
        /// The operation under test.
        check: &'static str,
        /// The time allowed for that operation.
        limit: Duration,
    },
}

type ContractResult<T = ()> = Result<T, EnvironmentContractError>;

/// Checks an application's [`Environment`] through its public operations.
///
/// Enable `test-util` in dev dependencies. Each verification method creates a
/// unique child of `scratch_root`. All fixture access goes through the supplied
/// environment, so the checks also work with containers and remote workspaces.
/// The application must dispose of the scratch root after success or failure.
/// The checks leave fixtures in place for inspection and never inspect host
/// paths.
///
/// Run the groups your implementation supports. An unsupported operation fails
/// its group; no check silently skips it. These checks cover observable
/// behavior, not application-specific isolation or access policy.
///
/// ```no_run
/// # async fn check(environment: &dyn pebble_coding_agent::environment::Environment)
/// # -> Result<(), pebble_coding_agent::test_support::EnvironmentContractError> {
/// use pebble_coding_agent::test_support::EnvironmentContract;
///
/// let contract = EnvironmentContract::new(environment, "scratch/environment-tests");
/// contract.verify_files().await?;
/// contract.verify_search().await?;
/// contract.verify_commands().await?;
/// # Ok(())
/// # }
/// ```
pub struct EnvironmentContract<'a> {
    environment:       &'a dyn Environment,
    scratch_root:      String,
    operation_timeout: Duration,
}

impl<'a> EnvironmentContract<'a> {
    /// Uses a caller-owned disposable directory and a ten-second operation
    /// limit.
    #[must_use]
    pub fn new(environment: &'a dyn Environment, scratch_root: impl Into<String>) -> Self {
        Self {
            environment,
            scratch_root: scratch_root.into(),
            operation_timeout: Duration::from_secs(10),
        }
    }

    /// Changes the per-operation deadline, for example for a remote
    /// environment.
    #[must_use]
    pub const fn with_operation_timeout(mut self, limit: Duration) -> Self {
        self.operation_timeout = limit;
        self
    }

    /// Checks file reads, writes, numbered lines, moves, and deletion.
    ///
    /// # Errors
    /// Returns the first failed or timed-out check. Fixtures remain available.
    pub async fn verify_files(&self) -> ContractResult {
        let root = self.fixture_root();
        let source = format!("{root}/nested/source.txt");
        let destination = format!("{root}/moved/destination.txt");
        let env = self.environment;
        equal(
            "new path does not exist",
            &(self
                .step("check missing path", env.file_exists(&source))
                .await?),
            &(false),
        )?;
        self.error_kind(
            "missing reads report NotFound",
            env.read_file_bytes(&source),
            EnvironmentErrorKind::NotFound,
        )
        .await?;
        self.step(
            "write creates parent directories",
            env.write_file(&source, "alpha\nbéta\ngamma\n"),
        )
        .await?;
        equal(
            "written bytes round trip",
            &(self
                .step("read bytes", env.read_file_bytes(&source))
                .await?),
            &("alpha\nbéta\ngamma\n".as_bytes().to_vec()),
        )?;
        equal(
            "numbered reads honor offset and limit",
            &(self
                .step(
                    "read numbered lines",
                    env.read_file(&source, Some(2), Some(1)),
                )
                .await?),
            &("2 | béta\n".to_owned()),
        )?;
        equal(
            "reads beyond EOF are empty",
            &(self
                .step("read beyond EOF", env.read_file(&source, Some(8), None))
                .await?),
            &(String::new()),
        )?;
        self.step(
            "write replaces existing content",
            env.write_file(&source, "replacement"),
        )
        .await?;
        self.expect_text("write truncates old content", &source, "replacement")
            .await?;
        self.step(
            "write existing file",
            env.write_existing_file(&source, "kept"),
        )
        .await?;
        self.step("move to self succeeds", env.rename_file(&source, &source))
            .await?;
        self.expect_text("move to self preserves content", &source, "kept")
            .await?;
        self.step(
            "move creates destination parents",
            env.rename_file(&source, &destination),
        )
        .await?;
        equal(
            "move removes old path",
            &(self
                .step("check moved source", env.file_exists(&source))
                .await?),
            &(false),
        )?;
        self.expect_text("move preserves content", &destination, "kept")
            .await?;
        self.step(
            "create replacement source",
            env.write_file(&source, "short"),
        )
        .await?;
        self.step(
            "move replaces destination",
            env.rename_file(&source, &destination),
        )
        .await?;
        self.expect_text("move replaces all content", &destination, "short")
            .await?;
        // A file cannot be used as a parent directory. The failed move must
        // leave the existing source intact.
        let invalid = format!("{destination}/child.txt");
        let result = self
            .bounded("failed move", env.rename_file(&destination, &invalid))
            .await?;
        equal(
            "invalid destination rejects move",
            &(result.is_err()),
            &(true),
        )?;
        self.expect_text("failed move preserves source", &destination, "short")
            .await?;
        self.step("delete file", env.delete_file(&destination))
            .await?;
        equal(
            "delete removes path",
            &(self
                .step("check deleted path", env.file_exists(&destination))
                .await?),
            &(false),
        )
    }

    /// Checks directory depth, relative names, glob grammar, and grep
    /// rendering.
    ///
    /// # Errors
    /// Returns the first failed or timed-out check, including unsupported
    /// search.
    pub async fn verify_search(&self) -> ContractResult {
        let root = self.fixture_root();
        let env = self.environment;
        for (name, text) in [
            ("z.txt", "last\n"),
            ("nested/b.txt", "needle\n"),
            ("a.txt", "first\nNeedle\nneedle\n"),
        ] {
            self.step(
                "create search fixture",
                env.write_file(&format!("{root}/{name}"), text),
            )
            .await?;
        }
        let entries = self
            .step("list one level", env.list_directory(&root, None))
            .await?;
        equal(
            "directory names are sorted and relative",
            &(entries
                .iter()
                .map(|entry| (entry.name.as_str(), entry.is_dir))
                .collect::<Vec<_>>()),
            &(vec![("a.txt", false), ("nested", true), ("z.txt", false)]),
        )?;
        let entries = self
            .step("list two levels", env.list_directory(&root, Some(2)))
            .await?;
        equal(
            "nested names are relative to listed directory",
            &(entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>()),
            &(vec!["a.txt", "nested", "nested/b.txt", "z.txt"]),
        )?;
        // Glob paths belong to the environment. Match their known suffixes
        // instead of imposing the host's absolute-path representation.
        let matches = self
            .step("recursive glob", env.glob("**/*.txt", Some(&root)))
            .await?;
        equal(
            "glob returns sorted matching paths",
            &(matches
                .iter()
                .zip(["/a.txt", "/nested/b.txt", "/z.txt"])
                .all(|(path, suffix)| path.replace('\\', "/").ends_with(suffix))
                && matches.len() == 3),
            &(true),
        )?;
        for path in &matches {
            self.step("glob paths can be read", env.read_file_text(path))
                .await?;
        }
        let matches = self
            .step("single-segment glob", env.glob("?.txt", Some(&root)))
            .await?;
        equal(
            "star and question mark stay within a segment",
            &(matches.len()),
            &(2),
        )?;
        for pattern in ["/absolute", "../escape", "a\\b", "nested/", "[a/]"] {
            self.error_kind(
                "invalid glob reports InvalidInput",
                env.glob(pattern, Some(&root)),
                EnvironmentErrorKind::InvalidInput,
            )
            .await?;
        }
        let options = GrepOptions {
            case_insensitive: true,
            glob_filter:      Some("a.txt".to_owned()),
            max_results:      Some(1),
        };
        let matches = self
            .step("grep with options", env.grep("needle", &root, &options))
            .await?;
        equal(
            "grep honors case, filter, limit, and line rendering",
            &(matches.len() == 1 && matches[0].replace('\\', "/").ends_with("/a.txt:2:Needle")),
            &(true),
        )
    }

    /// Checks Bash semantics, cwd, environment variables, capture limits,
    /// timeout, active cancellation, and invalid UTF-8 classification.
    ///
    /// Requires file operations and execution of Bash builtins. Cancellation
    /// starts after the command writes a marker through the environment.
    ///
    /// # Errors
    /// Returns the first failed or timed-out check. On timeout or future drop,
    /// the active command's cancellation token is also cancelled.
    pub async fn verify_commands(&self) -> ContractResult {
        let root = self.fixture_root();
        let env = self.environment;
        self.step(
            "create command directory",
            env.write_file(&format!("{root}/fixture"), "ready"),
        )
        .await?;
        let variables = HashMap::from([(
            "PEBBLE_CONTRACT_VALUE".to_owned(),
            "value with spaces".to_owned(),
        )]);
        let cancel = CancellationToken::new();
        let _cancel_on_drop = cancel.clone().drop_guard();
        let outcome = self
            .step(
                "run Bash with cwd and variables",
                env.exec(ExecRequest {
                    working_dir: Some(&root),
                    env_vars: Some(&variables),
                    cancel_token: Some(cancel.clone()),
                    ..ExecRequest::new(
                        r#"[[ -n "$BASH_VERSION" ]] || exit 91
shopt -q login_shell && exit 92
false
false | true
[[ $? == 0 ]] || exit 93
[[ -f fixture ]] || exit 94
printf '%s' "$PEBBLE_CONTRACT_VALUE"
printf 'error-stream' >&2
exit 7"#,
                    )
                }),
            )
            .await?;
        equal(
            "nonzero exit is a result",
            &(outcome.result.exit_code),
            &(Some(7)),
        )?;
        equal(
            "normal termination is Exited",
            &(outcome.result.termination),
            &(CommandTermination::Exited),
        )?;
        if outcome.streams_separated {
            equal(
                "stdout preserves variables",
                &(outcome.result.stdout),
                &("value with spaces".to_owned()),
            )?;
            equal(
                "stderr is separate",
                &(outcome.result.stderr),
                &("error-stream".to_owned()),
            )?;
        } else {
            equal(
                "combined output contains both streams",
                &(outcome.result.stdout.contains("value with spaces")
                    && outcome.result.stdout.contains("error-stream")
                    && outcome.result.stderr.is_empty()),
                &(true),
            )?;
        }
        let outcome = self
            .step(
                "bounded output drains past pipe capacity",
                env.exec(ExecRequest {
                    output_bytes_cap: Some(16),
                    cancel_token: Some(cancel.clone()),
                    ..ExecRequest::new("printf 'HEAD'; printf '%0200000d' 0; printf 'TAIL'")
                }),
            )
            .await?;
        equal(
            "bounded command succeeds",
            &(outcome.result.is_success()),
            &(true),
        )?;
        equal(
            "capture keeps stable head and rolling tail",
            &(outcome.result.stdout.starts_with("HEAD")
                && outcome.result.stdout.ends_with("TAIL")
                && outcome.result.stdout.len() <= 16),
            &(true),
        )?;
        let capture = outcome.output_capture();
        equal(
            "capture counts all bytes",
            &(capture.observed_bytes),
            &(200_008),
        )?;
        equal(
            "capture accounts for omitted bytes",
            &(capture.retained_bytes + capture.omitted_bytes),
            &(capture.observed_bytes),
        )?;
        equal(
            "capture reports retained bytes",
            &(capture.retained_bytes),
            &(outcome.result.stdout.len() + outcome.result.stderr.len()),
        )?;
        let live = LiveChunks::default();
        let recorded = Arc::clone(&live);
        let outcome = self
            .step(
                "live output reaches the sink",
                env.exec(ExecRequest {
                    output_bytes_cap: Some(4),
                    cancel_token: Some(cancel.clone()),
                    output_sink: Some(Arc::new(move |stream, chunk: &[u8]| {
                        recorded
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push((stream, chunk.to_vec()));
                    })),
                    ..ExecRequest::new("printf 'live-stdout'; printf 'live-stderr' >&2")
                }),
            )
            .await?;
        equal(
            "live output command succeeds",
            &(outcome.result.is_success()),
            &(true),
        )?;
        let chunks = take(&mut *live.lock().unwrap_or_else(PoisonError::into_inner));
        let collect = |wanted: ExecOutputStream| -> Vec<u8> {
            chunks
                .iter()
                .filter(|(stream, _)| *stream == wanted)
                .flat_map(|(_, chunk)| chunk.iter().copied())
                .collect()
        };
        equal(
            "sink sees standard output uncapped and in order",
            &(collect(ExecOutputStream::Stdout)),
            &(b"live-stdout".to_vec()),
        )?;
        if outcome.streams_separated {
            equal(
                "sink sees standard error uncapped and in order",
                &(collect(ExecOutputStream::Stderr)),
                &(b"live-stderr".to_vec()),
            )?;
        } else {
            equal(
                "sink sees the merged streams uncapped",
                &(collect(ExecOutputStream::Stdout).len()
                    + collect(ExecOutputStream::Stderr).len()),
                &("live-stdoutlive-stderr".len()),
            )?;
        }
        equal(
            "capped outcome still counts everything the sink saw",
            &(outcome.output_capture().observed_bytes),
            &("live-stdoutlive-stderr".len()),
        )?;
        let outcome = self
            .step(
                "command timeout",
                env.exec(ExecRequest {
                    timeout_ms: Some(20),
                    cancel_token: Some(cancel.clone()),
                    ..ExecRequest::new("while :; do :; done")
                }),
            )
            .await?;
        equal(
            "timeout is a result",
            &(outcome.result.termination, outcome.result.exit_code),
            &(CommandTermination::TimedOut, None),
        )?;
        let marker = format!("{root}/started");
        let run = env.exec(ExecRequest {
            working_dir: Some(&root),
            cancel_token: Some(cancel.clone()),
            ..ExecRequest::new("printf ready > started; while :; do :; done")
        });
        let stop = async {
            loop {
                if env.file_exists(&marker).await? {
                    cancel.cancel();
                    return Ok(());
                }
                sleep(Duration::from_millis(5)).await;
            }
        };
        let (outcome, ()) = tokio::try_join!(
            self.step("cancel active command", run),
            self.step("wait for command start", stop)
        )?;
        equal(
            "cancellation is a result",
            &(outcome.result.termination, outcome.result.exit_code),
            &(CommandTermination::Cancelled, None),
        )?;
        let cancel = CancellationToken::new();
        let _cancel_on_drop = cancel.clone().drop_guard();
        let outcome = self
            .step(
                "create non-UTF-8 file",
                env.exec(ExecRequest {
                    working_dir: Some(&root),
                    cancel_token: Some(cancel),
                    ..ExecRequest::new("printf '\\377' > binary")
                }),
            )
            .await?;
        equal(
            "binary fixture succeeds",
            &(outcome.result.is_success()),
            &(true),
        )?;
        self.error_kind(
            "invalid text reports InvalidUtf8",
            env.read_file_text(&format!("{root}/binary")),
            EnvironmentErrorKind::InvalidUtf8,
        )
        .await
    }

    fn fixture_root(&self) -> String {
        format!(
            "{}/pebble-contract-{}",
            self.scratch_root.trim_end_matches('/'),
            Uuid::new_v4()
        )
    }

    async fn bounded<T>(
        &self,
        check: &'static str,
        operation: impl Future<Output = T>,
    ) -> ContractResult<T> {
        timeout(self.operation_timeout, operation)
            .await
            .map_err(|_| EnvironmentContractError::TimedOut {
                check,
                limit: self.operation_timeout,
            })
    }

    async fn step<T>(
        &self,
        check: &'static str,
        operation: impl Future<Output = EnvResult<T>>,
    ) -> ContractResult<T> {
        self.bounded(check, operation)
            .await?
            .map_err(|source| EnvironmentContractError::Operation { check, source })
    }

    async fn expect_text(&self, check: &'static str, path: &str, expected: &str) -> ContractResult {
        equal(
            check,
            &(self
                .step(check, self.environment.read_file_text(path))
                .await?
                .as_str()),
            &(expected),
        )
    }

    async fn error_kind<T>(
        &self,
        check: &'static str,
        operation: impl Future<Output = EnvResult<T>>,
        expected: EnvironmentErrorKind,
    ) -> ContractResult {
        let actual = self.bounded(check, operation).await?;
        equal(
            check,
            &(actual.as_ref().err().map(EnvironmentError::kind)),
            &(Some(expected)),
        )
    }
}

fn equal<T: PartialEq + Debug>(check: &'static str, actual: &T, expected: &T) -> ContractResult {
    if actual == expected {
        Ok(())
    } else {
        Err(EnvironmentContractError::Mismatch {
            check,
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        })
    }
}
