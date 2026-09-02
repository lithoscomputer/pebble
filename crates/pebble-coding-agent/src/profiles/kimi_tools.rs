//! The Kimi Code tools whose behavior differs from pebble's own.
//!
//! Where a Kimi Code tool behaves the way pebble's already does, the Kimi
//! harness reuses pebble's and only the exposed name changes, which the
//! [vocabulary](crate::tools::ToolVocabulary) does on its own. The five here
//! differ in what their parameters *mean*, so renaming pebble's parameters
//! would advertise behavior pebble does not have:
//!
//! - `Bash` takes `timeout` in **seconds** where pebble takes milliseconds, and
//!   accepts a `cwd`. A rename alone would make every timeout 1000× wrong.
//! - `Read` accepts a **negative** `line_offset`, meaning "read the last N
//!   lines". Pebble's `offset` has no such meaning.
//! - `Write` takes a `mode`, so it can append. Pebble's write always replaces.
//! - `Edit` names the target `path` rather than `file_path`.
//! - `Grep` returns one of three **output shapes** and pages through them.
//!
//! Everything they do reaches the workspace through the same
//! [`Environment`](crate::environment::Environment) methods pebble's own tools
//! use, so environment behavior and path policy are unchanged.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use serde_json::{Value, json};

use super::definition;
use crate::environment::{GrepOptions, format_lines_numbered};
use crate::tool::{NativeTool, RegisteredTool, ToolError, optional_usize_arg, required_str};
use crate::tools::files::DEFAULT_READ_LINES;
use crate::tools::make_edit_file_tool;
use crate::tools::search::{execute_grep, grep_result_path};
use crate::tools::shell::{
    emit_shell_process_completed, execute_shell_command, retain_shell_output,
};
use crate::types::{CommandTermination, ToolSource};

/// How many search results a call returns when the model names no limit.
const DEFAULT_GREP_RESULTS: usize = 250;

/// How many search results a call returns at most.
const MAX_GREP_RESULTS: usize = 2000;

/// How many matches the underlying search is allowed to produce, which bounds
/// paging.
const MAX_GREP_MATCHES_SCANNED: usize = 20_000;

/// `Bash`, taking `timeout` in seconds and an optional `cwd`.
#[must_use]
pub(crate) fn make_kimi_bash_tool(default_timeout_ms: u64, max_timeout_ms: u64) -> RegisteredTool {
    let default_timeout_s = default_timeout_ms / 1000;
    let max_timeout_s = max_timeout_ms / 1000;
    let description = format!(
        "Execute a bash command. Use this for shell semantics — pipes, env, processes, git, \
package managers, build and test runners.

Translate these to a dedicated tool instead:
- `cat` / `head` / `tail` on a known path → Read
- `sed` / `awk` for an in-place edit → Edit
- `echo > file` / heredoc → Write
- `find` or recursive `ls` to locate files by name → Glob (plain `ls <dir>` is fine)
- `grep` / `rg` to search file contents → Grep

The dedicated tools cap their output, so they keep large raw dumps out of the conversation.

Output: stdout and stderr are combined and returned as a string. A non-zero exit appends a \
`Command failed with exit code: N` line.

Guidelines:
- Each call runs in a fresh bash process. Environment variables and `cd` do NOT persist between \
calls — pass `cwd`, or use absolute paths.
- `timeout` is in SECONDS. It defaults to {default_timeout_s} and is capped at {max_timeout_s}.
- A long-running command needs a raised `timeout`, not a retry: a command that timed out once \
will time out again.
- Do not run interactive commands, or commands that never exit.
- Chain genuinely dependent steps with `&&`. Issue independent read-only commands as separate \
parallel calls in one response so their output stays separate.
- Quote paths containing spaces.
- Avoid `..` to reach outside the working directory, and do not modify files outside it unless \
explicitly asked. Never run commands requiring superuser privileges unless explicitly asked."
    );

    RegisteredTool::new(
        definition(
            NativeTool::Shell,
            description,
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute."},
                    "cwd": {
                        "type": "string",
                        "description": "Directory to run the command in. Defaults to the \
            working directory."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": format!(
                            "Timeout in seconds (default {default_timeout_s}, max {max_timeout_s})."
                        )
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of what this command does."
                    }
                },
                "required": ["command"]
            }),
        ),
        Arc::new(move |arguments, context| {
            Box::pin(async move {
                let command = required_str(&arguments, "command")?;
                let cwd = arguments.get("cwd").and_then(Value::as_str);
                // Seconds on the wire, milliseconds in the environment.
                let timeout_ms = match arguments.get("timeout").and_then(Value::as_u64) {
                    Some(seconds) => seconds.saturating_mul(1000).min(max_timeout_ms),
                    None => default_timeout_ms,
                };

                let outcome = execute_shell_command(&context, command, timeout_ms, cwd).await?;
                let result = &outcome.result;

                // Kimi Code's own rendering: how it ended, then one combined
                // stream, then the exit code when it is not zero.
                let mut output = String::new();
                match result.termination {
                    CommandTermination::TimedOut => output.push_str("Command timed out.\n"),
                    CommandTermination::Cancelled => output.push_str("Command cancelled.\n"),
                    CommandTermination::Exited => {}
                }
                output.push_str(&result.stdout);
                if !result.stderr.is_empty() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&result.stderr);
                }
                if let Some(code) = result.exit_code.filter(|code| *code != 0) {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    let _ = write!(output, "Command failed with exit code: {code}");
                }

                let succeeded = result.is_success();
                let termination = result.termination;
                let output = retain_shell_output(&context, &outcome, output);
                emit_shell_process_completed(&context, outcome).await;

                if succeeded {
                    Ok(output)
                } else if termination == CommandTermination::Cancelled {
                    Err(ToolError::cancelled(output))
                } else {
                    Err(ToolError::execution(output))
                }
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// `Read`, where a negative `line_offset` reads from the end of the file.
#[must_use]
pub(crate) fn make_kimi_read_tool() -> RegisteredTool {
    RegisteredTool::new(
        definition(
            NativeTool::ReadFile,
            "Read a text file from the workspace.

- If you have a concrete path, call Read directly. Do not Glob or `ls` first to check that it \
exists — a missing path returns an error you can handle.
- When you need several files, emit multiple Read calls in one response rather than one per turn.
- Returns `<line-number> | <content>` per line. Drop the number and separator when taking text for \
an Edit `old_string`.
- `line_offset` is the 1-based first line to read. A NEGATIVE value reads from the end, so -100 \
returns the last 100 lines.
- `n_lines` defaults to 2000 lines.
- Use Bash or an MCP tool for binary formats; this tool reads text.
- After a successful Edit or Write, do not re-read solely to prove the write landed. When the task \
depends on an exact file, API, or output shape, inspect the final result before finishing.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                    "line_offset": {
                        "type": "integer",
                        "minimum": -2000,
                        "description": "1-based first line to read. Negative reads from the end \
            of the file (-100 reads the last 100 lines); zero is invalid."
                    },
                    "n_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2000,
                        "description": "Number of lines to read (default 2000)."
                    }
                },
                "required": ["path"]
            }),
        ),
        Arc::new(|arguments, context| {
            Box::pin(async move {
                let path = required_str(&arguments, "path")?;
                let n_lines =
                    optional_usize_arg(&arguments, "n_lines")?.unwrap_or(DEFAULT_READ_LINES);
                if n_lines == 0 || n_lines > DEFAULT_READ_LINES {
                    return Err(ToolError::invalid_arguments(format!(
                        "n_lines must be between 1 and {DEFAULT_READ_LINES}"
                    )));
                }
                let line_offset = arguments.get("line_offset").and_then(Value::as_i64);
                if line_offset == Some(0) {
                    return Err(ToolError::invalid_arguments("line_offset must not be zero"));
                }

                match line_offset {
                    // A negative offset counts the file's lines and starts
                    // that many from the end, which is Kimi Code's meaning and
                    // has no counterpart in pebble's own reader.
                    Some(offset) if offset < 0 => {
                        let from_end = usize::try_from(offset.unsigned_abs()).map_err(|_| {
                            ToolError::invalid_arguments("line_offset is too large")
                        })?;
                        if from_end > DEFAULT_READ_LINES {
                            return Err(ToolError::invalid_arguments(format!(
                                "negative line_offset must be at least -{DEFAULT_READ_LINES}"
                            )));
                        }
                        let raw = context.env.read_file_text(path).await?;
                        let total = raw.lines().count();
                        let start = total.saturating_sub(from_end).saturating_add(1);
                        Ok(format_lines_numbered(
                            &raw,
                            Some(start),
                            Some(n_lines.min(from_end)),
                        ))
                    }
                    Some(offset) => {
                        let start = usize::try_from(offset).map_err(|_| {
                            ToolError::invalid_arguments("line_offset must fit in usize")
                        })?;
                        Ok(context
                            .env
                            .read_file(path, Some(start), Some(n_lines))
                            .await?)
                    }
                    None => Ok(context.env.read_file(path, None, Some(n_lines)).await?),
                }
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// What Kimi Code's `Write` does with a file that already exists.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum KimiWriteMode {
    /// Replace the whole file.
    #[default]
    Overwrite,
    /// Add to its end.
    Append,
}

impl KimiWriteMode {
    /// The mode `value` names, or `None` when it names none.
    fn parse(value: &str) -> Option<Self> {
        match value {
            "overwrite" => Some(Self::Overwrite),
            "append" => Some(Self::Append),
            _ => None,
        }
    }
}

/// `Write`, with Kimi Code's `mode` so it can append.
#[must_use]
pub(crate) fn make_kimi_write_tool() -> RegisteredTool {
    RegisteredTool::new(definition(
            NativeTool::WriteFile,
            "Create, append to, or replace a file entirely.

- `mode` defaults to `overwrite`, which replaces the whole file. `append` requires an existing file \
and adds to its end without inserting a newline.
- Write is NOT ALLOWED for incremental changes to existing files, including trivial, one-line, \
quick, or cosmetic edits. Use Edit instead.
- Use Write only when the file does not exist, you intend a complete replacement, or the new \
contents have little continuity with the old contents.
- Read before overwriting an existing file.
- Write ignores the Read/Edit line-number view. NEVER include line prefixes.
- Do not create documentation files that were not asked for.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write."},
                    "content": {"type": "string", "description": "Content to write."},
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Whether to replace the file or append to it (default \
            overwrite)."
                    }
                },
                "required": ["path", "content"]
            }),
        ), Arc::new(|arguments, context| {
            Box::pin(async move {
                let path = required_str(&arguments, "path")?;
                let content = required_str(&arguments, "content")?;
                let mode = arguments
                    .get("mode")
                    .and_then(Value::as_str)
                    .unwrap_or("overwrite");
                let mode = KimiWriteMode::parse(mode).ok_or_else(|| {
                    ToolError::invalid_arguments("Invalid mode (expected overwrite|append)")
                })?;

                match mode {
                    KimiWriteMode::Overwrite => {
                        context.env.write_file(path, content).await?;
                    }
                    // The environment has no append, so read-modify-write
                    // keeps every implementation working and stays inside
                    // whatever path policy it enforces.
                    KimiWriteMode::Append => {
                        let mut existing = context.env.read_file_text(path).await?;
                        existing.push_str(content);
                        context.env.write_file(path, &existing).await?;
                    }
                }
                Ok(format!("Wrote {path}"))
            })
        })).with_source(ToolSource::Native)
}

/// `Edit`, whose target is named `path`.
///
/// Only the adapter field is translated; the exact-match implementation behind
/// it is pebble's own, so what counts as a match and what a failure says are
/// the same in every harness.
#[must_use]
pub(crate) fn make_kimi_edit_tool(description: &str) -> RegisteredTool {
    let shared_executor = make_edit_file_tool().executor;
    RegisteredTool::new(
        definition(
            NativeTool::EditFile,
            description,
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the text file to edit."},
                    "old_string": {"type": "string", "description": "Exact content to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence (default false)."
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        ),
        Arc::new(move |mut arguments, context| {
            let shared_executor = Arc::clone(&shared_executor);
            Box::pin(async move {
                let object = arguments.as_object_mut().ok_or_else(|| {
                    ToolError::invalid_arguments("Edit arguments must be an object")
                })?;
                let path = object.remove("path").ok_or_else(|| {
                    ToolError::invalid_arguments("Missing required parameter: path")
                })?;
                object.insert("file_path".to_owned(), path);
                shared_executor(arguments, context).await
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// The shapes Kimi Code's `Grep` returns its results in.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum GrepOutputMode {
    /// The matching lines themselves.
    Content,
    /// One path per file that matched.
    #[default]
    FilesWithMatches,
    /// One `path:count` line per file that matched.
    CountMatches,
}

impl GrepOutputMode {
    /// The mode `value` names, or `None` when it names none.
    fn parse(value: &str) -> Option<Self> {
        match value {
            "content" => Some(Self::Content),
            "files_with_matches" => Some(Self::FilesWithMatches),
            "count_matches" => Some(Self::CountMatches),
            _ => None,
        }
    }
}

/// `Grep` with Kimi Code's `output_mode`, `head_limit` and `offset`.
///
/// These are all shapes of the result list the environment already returns, so
/// no environment work is needed. Kimi Code's `type`, `multiline` and
/// `include_ignored` are deliberately absent: they would have to reach ripgrep
/// flags through new [`Environment`](crate::environment::Environment) methods,
/// and advertising a parameter that is ignored is worse than omitting it.
#[must_use]
pub(crate) fn make_kimi_grep_tool() -> RegisteredTool {
    RegisteredTool::new(definition(
            NativeTool::Grep,
            "Search file contents with a regular expression.

Use Grep when looking for unknown content or an unknown location. If you already know the path, \
use Read instead. Prefer this over running `grep` or `rg` through Bash: it caps its output, so it \
will not flood the conversation.

- Backed by ripgrep when available and POSIX `grep` otherwise, so keep patterns portable across \
both rather than relying on ripgrep-only syntax.
- `output_mode` selects what comes back: `files_with_matches` (just the paths, the default), \
`content` (matching lines), or `count_matches` (matches per file).
- `head_limit` caps how many results are returned and `offset` skips that many first, so you can \
page through a large result set.
- `glob` limits which files are searched; `-i` folds case.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "Directory or file to search. Defaults to the working directory."},
                    "glob": {"type": "string", "description": "Only search files matching this glob."},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count_matches"],
                        "description": "Shape of the results (default files_with_matches)."
                    },
                    "head_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2000,
                        "description": "Return at most this many results (default 250)."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 20000,
                        "description": "Skip this many results before returning."
                    },
                    "-i": {"type": "boolean", "description": "Perform a case-insensitive search."}
                },
                "required": ["pattern"]
            }),
        ), Arc::new(|arguments, context| {
            Box::pin(async move {
                let pattern = required_str(&arguments, "pattern")?;
                // The environment requires a search root; "." is the working
                // directory.
                let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
                let mode = arguments
                    .get("output_mode")
                    .and_then(Value::as_str)
                    .unwrap_or("files_with_matches");
                let mode = GrepOutputMode::parse(mode).ok_or_else(|| {
                    ToolError::invalid_arguments(
                        "Invalid output_mode (expected content|files_with_matches|count_matches)",
                    )
                })?;
                let head_limit =
                    optional_usize_arg(&arguments, "head_limit")?.unwrap_or(DEFAULT_GREP_RESULTS);
                if head_limit == 0 || head_limit > MAX_GREP_RESULTS {
                    return Err(ToolError::invalid_arguments(format!(
                        "head_limit must be between 1 and {MAX_GREP_RESULTS}"
                    )));
                }
                let offset = optional_usize_arg(&arguments, "offset")?.unwrap_or(0);
                if offset > MAX_GREP_MATCHES_SCANNED {
                    return Err(ToolError::invalid_arguments(format!(
                        "offset must be at most {MAX_GREP_MATCHES_SCANNED}"
                    )));
                }
                if offset.saturating_add(head_limit) > MAX_GREP_MATCHES_SCANNED {
                    return Err(ToolError::invalid_arguments(format!(
                        "offset + head_limit must be at most {MAX_GREP_MATCHES_SCANNED}"
                    )));
                }

                let options = GrepOptions {
                    glob_filter:      arguments
                        .get("glob")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    case_insensitive: arguments
                        .get("-i")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    // A grouped mode counts every match before it groups them,
                    // so it asks for everything the search will produce; a
                    // content mode only needs the page it is about to return.
                    max_results:      match mode {
                        GrepOutputMode::Content => Some(
                            head_limit
                                .saturating_add(offset)
                                .min(MAX_GREP_MATCHES_SCANNED),
                        ),
                        GrepOutputMode::FilesWithMatches | GrepOutputMode::CountMatches => {
                            Some(MAX_GREP_MATCHES_SCANNED)
                        }
                    },
                };

                let lines = execute_grep(&context, pattern, path, &options).await?;
                let results: Vec<String> = shape_grep_results(lines, mode, path)
                    .into_iter()
                    .skip(offset)
                    .take(head_limit)
                    .collect();

                if results.is_empty() {
                    return Ok("No matches found".to_owned());
                }
                Ok(results.join("\n"))
            })
        })).with_source(ToolSource::Native)
}

/// The search results, grouped the way `mode` asks for, in the order they were
/// found.
fn shape_grep_results(lines: Vec<String>, mode: GrepOutputMode, searched: &str) -> Vec<String> {
    match mode {
        GrepOutputMode::Content => lines,
        GrepOutputMode::FilesWithMatches => {
            let mut seen = HashSet::new();
            let mut files = Vec::new();
            for line in lines {
                let file = grep_result_path(&line, searched).to_owned();
                if seen.insert(file.clone()) {
                    files.push(file);
                }
            }
            files
        }
        GrepOutputMode::CountMatches => {
            let mut counts: HashMap<String, usize> = HashMap::new();
            let mut order = Vec::new();
            for line in lines {
                let file = grep_result_path(&line, searched).to_owned();
                if let Some(count) = counts.get_mut(&file) {
                    *count += 1;
                } else {
                    counts.insert(file.clone(), 1);
                    order.push(file);
                }
            }
            order
                .into_iter()
                .map(|file| {
                    let count = counts[&file];
                    format!("{file}:{count}")
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::environment::{Environment as _, ExecResult};
    use crate::test_support::{MockEnvironment, MutableMockEnvironment};
    use crate::tool::StaticEnvProvider;
    use crate::tools::testing::{context, context_for, schema_of};
    use crate::types::ToolErrorKind;

    /// A workspace holding one file.
    fn workspace_with(path: &str, content: &str) -> Arc<MutableMockEnvironment> {
        Arc::new(MutableMockEnvironment::new(HashMap::from([(
            path.to_owned(),
            content.to_owned(),
        )])))
    }

    /// The reason Read is a separate tool: a negative `line_offset` means "the
    /// last N lines", which pebble's `offset` has no notion of.
    #[tokio::test]
    async fn a_negative_line_offset_reads_from_the_end() {
        let lines: Vec<String> = (1..=20).map(|line| format!("line{line}")).collect();
        let environment = workspace_with("/f.txt", &lines.join("\n"));

        let output = (make_kimi_read_tool().executor)(
            json!({"path": "/f.txt", "line_offset": -3}),
            context_for(environment),
        )
        .await
        .expect("the file is read");

        assert!(output.contains("line18"), "{output}");
        assert!(output.contains("line20"), "{output}");
        assert!(
            !output.contains("line1\n"),
            "the head is not included: {output}"
        );
    }

    #[tokio::test]
    async fn a_positive_line_offset_starts_there() {
        let lines: Vec<String> = (1..=20).map(|line| format!("line{line}")).collect();
        let environment = workspace_with("/f.txt", &lines.join("\n"));

        let output = (make_kimi_read_tool().executor)(
            json!({"path": "/f.txt", "line_offset": 5, "n_lines": 2}),
            context_for(environment),
        )
        .await
        .expect("the file is read");

        assert!(output.contains("line5"), "{output}");
        assert!(!output.contains("line8"), "{output}");
    }

    #[tokio::test]
    async fn a_positive_offset_still_applies_the_default_limit() {
        let lines: Vec<String> = (1..=DEFAULT_READ_LINES + 5)
            .map(|line| format!("line{line}"))
            .collect();
        let environment = workspace_with("/f.txt", &lines.join("\n"));

        let output = (make_kimi_read_tool().executor)(
            json!({"path": "/f.txt", "line_offset": 2}),
            context_for(environment),
        )
        .await
        .expect("the file is read");

        assert!(output.contains("2001 | line2001"), "{output}");
        assert!(!output.contains("2002 | line2002"), "{output}");
    }

    #[tokio::test]
    async fn a_zero_line_offset_is_an_argument_error() {
        let environment = workspace_with("/f.txt", "one");

        let error = (make_kimi_read_tool().executor)(
            json!({"path": "/f.txt", "line_offset": 0}),
            context_for(environment),
        )
        .await
        .expect_err("zero names no line");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "line_offset must not be zero");
    }

    /// The reason Write is a separate tool: it has a mode, so it can append.
    #[tokio::test]
    async fn append_mode_keeps_what_the_file_already_held() {
        let environment = workspace_with("/f.txt", "first");

        (make_kimi_write_tool().executor)(
            json!({"path": "/f.txt", "content": "-second", "mode": "append"}),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the file is appended to");

        assert_eq!(
            environment
                .read_file_text("/f.txt")
                .await
                .expect("the file is readable"),
            "first-second"
        );
    }

    #[tokio::test]
    async fn a_write_with_no_mode_replaces_the_file() {
        let environment = workspace_with("/f.txt", "first");

        (make_kimi_write_tool().executor)(
            json!({"path": "/f.txt", "content": "only"}),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the file is written");

        assert_eq!(
            environment
                .read_file_text("/f.txt")
                .await
                .expect("the file is readable"),
            "only"
        );
    }

    #[tokio::test]
    async fn a_mode_the_tool_does_not_have_is_refused_by_name() {
        let environment = workspace_with("/f.txt", "x");

        let error = (make_kimi_write_tool().executor)(
            json!({"path": "/f.txt", "content": "y", "mode": "prepend"}),
            context_for(environment),
        )
        .await
        .expect_err("there is no prepend");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert!(
            error.message().contains("expected overwrite|append"),
            "{}",
            error.message()
        );
    }

    #[tokio::test]
    async fn appending_to_a_file_that_is_not_there_says_so() {
        let environment = Arc::new(MutableMockEnvironment::new(HashMap::new()));

        let error = (make_kimi_write_tool().executor)(
            json!({"path": "/missing.txt", "content": "new", "mode": "append"}),
            context_for(environment),
        )
        .await
        .expect_err("there is nothing to append to");

        assert!(
            error.message().contains("missing.txt"),
            "{}",
            error.message()
        );
    }

    #[tokio::test]
    async fn edit_translates_kimis_path_to_the_shared_executor() {
        let environment = workspace_with("/f.txt", "before");
        let tool = make_kimi_edit_tool("Edit");

        (tool.executor)(
            json!({"path": "/f.txt", "old_string": "before", "new_string": "after"}),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the edit lands");

        assert_eq!(
            environment
                .read_file_text("/f.txt")
                .await
                .expect("the file is readable"),
            "after"
        );
        assert!(schema_of(&tool)["properties"].get("path").is_some());
        assert!(schema_of(&tool)["properties"].get("file_path").is_none());
    }

    /// One search, shaped by `arguments`, over an environment that answers
    /// `lines`.
    async fn grep_with(arguments: Value, lines: Vec<String>) -> Result<String, ToolError> {
        let environment = MockEnvironment {
            grep_results: lines,
            ..MockEnvironment::default()
        };
        (make_kimi_grep_tool().executor)(arguments, context(environment)).await
    }

    #[tokio::test]
    async fn content_mode_returns_the_matching_lines() {
        let output = grep_with(json!({"pattern": "x", "output_mode": "content"}), vec![
            "a.rs:1:x".to_owned(),
            "b.rs:2:x".to_owned(),
        ])
        .await
        .expect("the search runs");

        assert_eq!(output, "a.rs:1:x\nb.rs:2:x");
    }

    #[tokio::test]
    async fn a_search_with_no_mode_answers_with_the_files() {
        let output = grep_with(json!({"pattern": "x"}), vec![
            "a.rs:1:x".to_owned(),
            "a.rs:2:x".to_owned(),
            "b.rs:2:x".to_owned(),
        ])
        .await
        .expect("the search runs");

        assert_eq!(output, "a.rs\nb.rs");
    }

    #[tokio::test]
    async fn files_with_matches_names_each_file_once_in_the_order_found() {
        let output = grep_with(
            json!({"pattern": "x", "output_mode": "files_with_matches"}),
            vec![
                "a.rs:1:x".to_owned(),
                "a.rs:9:x".to_owned(),
                "b.rs:2:x".to_owned(),
            ],
        )
        .await
        .expect("the search runs");

        assert_eq!(output, "a.rs\nb.rs");
    }

    #[tokio::test]
    async fn count_matches_counts_per_file() {
        let output = grep_with(
            json!({"pattern": "x", "output_mode": "count_matches"}),
            vec![
                "a.rs:1:x".to_owned(),
                "a.rs:9:x".to_owned(),
                "b.rs:2:x".to_owned(),
            ],
        )
        .await
        .expect("the search runs");

        assert_eq!(output, "a.rs:2\nb.rs:1");
    }

    #[tokio::test]
    async fn offset_and_head_limit_page_through_the_results() {
        let lines: Vec<String> = (1..=6).map(|file| format!("f{file}.rs:1:x")).collect();

        let output = grep_with(
            json!({
                "pattern": "x",
                "output_mode": "content",
                "offset": 2,
                "head_limit": 2
            }),
            lines,
        )
        .await
        .expect("the search runs");

        assert_eq!(output, "f3.rs:1:x\nf4.rs:1:x");
    }

    #[tokio::test]
    async fn an_output_mode_the_tool_does_not_have_is_refused_by_name() {
        let error = grep_with(json!({"pattern": "x", "output_mode": "json"}), Vec::new())
            .await
            .expect_err("there is no json mode");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert!(
            error
                .message()
                .contains("expected content|files_with_matches|count_matches"),
            "{}",
            error.message()
        );
    }

    #[tokio::test]
    async fn a_search_that_found_nothing_says_so_plainly() {
        let output = grep_with(json!({"pattern": "x"}), Vec::new())
            .await
            .expect("the search runs");

        assert_eq!(output, "No matches found");
    }

    #[test]
    fn the_search_schema_carries_kimi_codes_own_modes_and_flags() {
        let schema = schema_of(&make_kimi_grep_tool()).clone();

        assert_eq!(
            schema["properties"]["output_mode"]["enum"],
            json!(["content", "files_with_matches", "count_matches"])
        );
        assert!(schema["properties"].get("-i").is_some());
        assert!(schema["properties"].get("case_insensitive").is_none());
    }

    /// The reason Bash is a separate tool: `timeout` is seconds, not
    /// milliseconds. A rename would have made every timeout 1000× wrong.
    #[test]
    fn the_shell_schema_states_seconds_and_quotes_the_real_limits() {
        let tool = make_kimi_bash_tool(60_000, 600_000);
        let schema = schema_of(&tool);
        let timeout = schema["properties"]["timeout"]["description"]
            .as_str()
            .expect("the timeout is described");

        assert!(timeout.contains("seconds"), "{timeout}");
        assert!(timeout.contains("60"), "the default is 60s: {timeout}");
        assert!(timeout.contains("600"), "the maximum is 600s: {timeout}");
        assert!(schema["properties"].get("cwd").is_some());
        assert!(
            tool.definition
                .description
                .contains("timeout` is in SECONDS")
        );
        // Pebble has no background shell, so none is promised.
        assert!(!tool.definition.description.contains("run_in_background"));
    }

    #[tokio::test]
    async fn a_command_reaches_the_environment_with_its_seconds_converted() {
        let tool = make_kimi_bash_tool(60_000, 600_000);
        let environment = Arc::new(MockEnvironment {
            exec_result: ExecResult {
                stdout:      String::new(),
                stderr:      String::new(),
                exit_code:   None,
                termination: CommandTermination::TimedOut,
                duration_ms: 7_000,
            },
            ..MockEnvironment::default()
        });
        let tool_env = HashMap::from([("TOKEN".to_owned(), "value".to_owned())]);

        let error = (tool.executor)(
            json!({"command": "echo $TOKEN", "cwd": "/repo", "timeout": 7}),
            context_for(Arc::clone(&environment))
                .with_tool_env_provider(Arc::new(StaticEnvProvider(tool_env.clone()))),
        )
        .await
        .expect_err("a timeout is a failed call");

        assert!(
            error.message().starts_with("Command timed out.\n"),
            "{}",
            error.message()
        );
        assert_eq!(
            *environment
                .captured_timeout
                .lock()
                .expect("the mock records one timeout"),
            Some(7_000)
        );
        assert_eq!(
            environment
                .captured_working_dirs
                .lock()
                .expect("the mock records the working directories")
                .as_slice(),
            &[Some("/repo".to_owned())]
        );
        assert_eq!(
            *environment
                .captured_env_vars
                .lock()
                .expect("the mock records the environment"),
            Some(tool_env)
        );
    }

    #[tokio::test]
    async fn a_command_that_exits_nonzero_appends_kimi_codes_own_line() {
        let tool = make_kimi_bash_tool(60_000, 600_000);
        let environment = MockEnvironment {
            exec_result: ExecResult {
                stdout:      "some output".to_owned(),
                stderr:      "and an error".to_owned(),
                exit_code:   Some(2),
                termination: CommandTermination::Exited,
                duration_ms: 5,
            },
            ..MockEnvironment::default()
        };

        let error = (tool.executor)(json!({"command": "false"}), context(environment))
            .await
            .expect_err("a nonzero exit is a failed call");

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert_eq!(
            error.message(),
            "some output\nand an error\nCommand failed with exit code: 2"
        );
    }
}
