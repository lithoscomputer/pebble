//! Content search and file walks inside a sandbox, run as commands.
//!
//! Parity gap: the old driver gave every sandbox a search facet, native or
//! derived over exec (`DerivedSearch`). sandbox-driver's new API has no search
//! (parity plan B3), so this is that derived search, moved onto the new exec:
//! ripgrep when the sandbox has it, `grep` otherwise; GNU `find -printf` with
//! a POSIX `find -print0` fallback for walks. A walk refuses a symlinked
//! traversal root.

use std::fmt::Write as _;
use std::time::Duration;

use sandbox_driver::{ExecControls, ExecSpec};

use super::exec::{CommandFailure, ExecResult, SandboxExec};
use super::path::shell_quote;
use super::sanitize::OutputSanitization;
use crate::environment::{EnvResult, EnvironmentError, EnvironmentErrorKind, GrepOptions};

const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const GREP_TIMEOUT: Duration = Duration::from_secs(30);
const WALK_TIMEOUT: Duration = Duration::from_secs(30);

/// Exit code when SIGPIPE ends the producer because `head` already has
/// everything it needs: a successful run truncated at the source.
const SIGPIPE_EXIT: i32 = 141;

/// Whether `rg` is on the sandbox's `PATH`. A failed probe is "no".
pub(super) async fn ripgrep_available(exec: &SandboxExec<'_>) -> bool {
    let mut spec = ExecSpec::shell("command -v rg >/dev/null 2>&1");
    spec.timeout_ms = Some(duration_ms(PROBE_TIMEOUT));
    exec.run_streaming(spec, ExecControls::default(), None)
        .await
        .is_ok_and(|streaming| streaming.result.success())
}

/// Matching lines as `(path, line number, line)`.
pub(super) async fn grep(
    exec: &SandboxExec<'_>,
    ripgrep: bool,
    pattern: &str,
    path: &str,
    options: &GrepOptions,
) -> EnvResult<Vec<(String, u64, String)>> {
    // `--null` separates the path with NUL, so a `:` in a file name cannot
    // corrupt the parse; `-I` keeps binary notices out of grep output
    // (ripgrep skips binary files by default); `-H` forces the file name
    // even for a single-file path, which the parse requires.
    let mut command = if ripgrep {
        String::from("rg -H --line-number --no-heading --no-messages --null")
    } else {
        // `--null` (not `-Z`, which BSD grep reads as zgrep mode) works on
        // both GNU and BSD grep.
        String::from("grep -rnIH --null")
    };
    if options.case_insensitive {
        command.push_str(" -i");
    }
    if let Some(max) = options.max_results {
        let _ = write!(command, " -m {max}");
    }
    if let Some(include) = &options.glob_filter {
        let flag = if ripgrep { " --glob " } else { " --include=" };
        command.push_str(flag);
        command.push_str(&shell_quote(include));
    }
    let _ = write!(
        command,
        " -e {} -- {}",
        shell_quote(pattern),
        shell_quote(path)
    );
    // `-m` bounds matches per file; `head` bounds the total at the source so
    // a broad pattern cannot buffer unbounded output. Under pipefail, `head`
    // closing the pipe surfaces as 141.
    if let Some(max) = options.max_results {
        command = format!("set -o pipefail\n{command} | head -n {max}");
    }

    let result = run(exec, &command, GREP_TIMEOUT).await?;
    let truncated_at_source = result.exit_code == Some(SIGPIPE_EXIT);
    if !result.success() && !truncated_at_source {
        if result.exit_code == Some(1) {
            return Ok(Vec::new());
        }
        return Err(failed("Failed to search file contents", "grep", result));
    }

    let mut matches = Vec::new();
    for line in result.stdout_lossy().lines() {
        let parsed = line.split_once('\0').and_then(|(file, rest)| {
            let (number, content) = rest.split_once(':')?;
            Some((file, number.parse::<u64>().ok()?, content))
        });
        // An unparseable line means wrong results, not skippable noise.
        let Some((file, line_number, content)) = parsed else {
            return Err(failed(
                "Failed to search file contents",
                "grep output parse",
                result,
            ));
        };
        matches.push((file.to_string(), line_number, content.to_string()));
        if options.max_results.is_some_and(|max| matches.len() >= max) {
            break;
        }
    }
    Ok(matches)
}

/// Every regular file under `base`, as paths relative to it. A missing base
/// is an empty walk; a symlinked base is refused the same way.
pub(super) async fn walk(exec: &SandboxExec<'_>, base: &str) -> EnvResult<Vec<String>> {
    // The guard runs in the command so a missing base cannot fail the exec
    // itself. The symlink checks refuse a symlinked traversal root: a
    // sandboxed process could otherwise redirect a walk at an arbitrary
    // tree.
    let guarded = |find: &str| {
        format!(
            "if [ -d {quoted} ]{guard}; then cd {quoted} && {find}; fi",
            quoted = shell_quote(base),
            guard = symlink_guard(base),
        )
    };

    // GNU find first. `-printf` is missing from BSD find, which fails and
    // triggers the portable fallback.
    let result = run(
        exec,
        &guarded("find -H . -type f -printf '%P\\0'"),
        WALK_TIMEOUT,
    )
    .await?;
    if result.success() {
        return Ok(nul_separated(&result.stdout, ""));
    }
    let result = run(exec, &guarded("find -H . -type f -print0"), WALK_TIMEOUT).await?;
    if !result.success() {
        return Err(failed("Failed to match files", "walk", result));
    }
    Ok(nul_separated(&result.stdout, "./"))
}

/// The ` && [ ! -L … ]` checks that refuse a symlinked walk root. Relative
/// bases are checked per accumulated prefix (each lives inside the working
/// directory); absolute bases are checked whole, since their ancestors
/// (macOS `/var` → `/private/var`) are legitimately symlinked.
fn symlink_guard(base: &str) -> String {
    let mut guard = String::new();
    if base.starts_with('/') {
        let _ = write!(guard, " && [ ! -L {} ]", shell_quote(base));
        return guard;
    }
    let mut prefix = String::new();
    for segment in base.split('/').filter(|segment| !segment.is_empty()) {
        if !prefix.is_empty() {
            prefix.push('/');
        }
        prefix.push_str(segment);
        let _ = write!(guard, " && [ ! -L {} ]", shell_quote(&prefix));
    }
    guard
}

fn nul_separated(stdout: &[u8], strip: &str) -> Vec<String> {
    stdout
        .split(|&byte| byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            let text = String::from_utf8_lossy(path);
            text.strip_prefix(strip).unwrap_or(&text).to_string()
        })
        .collect()
}

/// Runs a search command with its output raw: the NUL separators it parses
/// are control characters the policy would otherwise strip.
async fn run(exec: &SandboxExec<'_>, command: &str, timeout: Duration) -> EnvResult<ExecResult> {
    let mut spec = ExecSpec::shell(command);
    spec.timeout_ms = Some(duration_ms(timeout));
    exec.clone()
        .with_output_sanitization(OutputSanitization::Raw)
        .run_streaming(spec, ExecControls::default(), None)
        .await
        .map(|streaming| streaming.result)
        .map_err(|error| super::environment::environment_error("Failed to run a search", error))
}

fn failed(message: &str, label: &str, result: ExecResult) -> EnvironmentError {
    EnvironmentError::with_source(
        EnvironmentErrorKind::Io,
        message,
        CommandFailure::new(label, result),
    )
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_walk_root_is_checked_at_every_prefix() {
        assert_eq!(symlink_guard("a/b"), " && [ ! -L 'a' ] && [ ! -L 'a/b' ]");
        assert_eq!(symlink_guard("/abs/dir"), " && [ ! -L '/abs/dir' ]");
        assert_eq!(symlink_guard("."), " && [ ! -L '.' ]");
    }

    #[test]
    fn walk_output_is_split_on_nul() {
        assert_eq!(nul_separated(b"a.rs\0dir/b.rs\0", ""), ["a.rs", "dir/b.rs"]);
        assert_eq!(nul_separated(b"./a.rs\0./b c.rs\0", "./"), [
            "a.rs", "b c.rs"
        ]);
    }
}
