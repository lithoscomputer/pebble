//! Finding things: by content, by name, and by directory.

use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde_json::Value;

use crate::environment::GrepOptions;
use crate::tool::{RegisteredTool, ToolContext, ToolError, optional_usize_arg, required_str};
use crate::types::ToolSource;

/// Searches file contents for a regular expression.
#[must_use]
pub fn make_grep_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "grep",
            "Search file contents with a regex pattern. Use path to choose the search root, \
             glob_filter to limit matching files, case_insensitive for case folding, and \
             max_results to cap output.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Path to search in (default \".\")"},
                    "glob_filter": {"type": "string", "description": "Glob pattern to filter files"},
                    "case_insensitive": {"type": "boolean", "description": "Case insensitive search"},
                    "max_results": {"type": "integer", "description": "Maximum number of results"}
                },
                "required": ["pattern"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let pattern = required_str(&args, "pattern")?;
                let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
                let options = GrepOptions {
                    glob_filter:      args
                        .get("glob_filter")
                        .and_then(Value::as_str)
                        .map(String::from),
                    case_insensitive: args
                        .get("case_insensitive")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    max_results:      optional_usize_arg(&args, "max_results")?,
                };

                let results = execute_grep(&ctx, pattern, path, &options).await?;
                Ok(results.join("\n"))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Runs one content search.
///
/// Shared by the canonical `grep` tool and by the profile search tools that
/// group the same result lines differently, so every one of them searches the
/// same way and reports a failure the same way.
pub(crate) async fn execute_grep(
    ctx: &ToolContext,
    pattern: &str,
    path: &str,
    options: &GrepOptions,
) -> Result<Vec<String>, ToolError> {
    Ok(ctx.env.grep(pattern, path, options).await?)
}

/// The file path in one `"{path}:{line}:{content}"` search result.
///
/// A search of a single file may omit the path, in which case `searched` — the
/// path the caller searched — is the answer. Candidate separators are walked
/// from the left, so a path holding a colon (a Windows drive letter, a file
/// named `a:b`) still parses: the first colon followed by digits and another
/// colon ends the path.
///
/// ```
/// use pebble::grep_result_path;
///
/// assert_eq!(
///     grep_result_path("src/main.rs:42:fn main() {", "src"),
///     "src/main.rs"
/// );
/// assert_eq!(
///     grep_result_path("42:fn main() {", "src/main.rs"),
///     "src/main.rs"
/// );
/// ```
#[must_use]
pub fn grep_result_path<'a>(line: &'a str, searched: &'a str) -> &'a str {
    let mut rest = line;
    let mut consumed = 0_usize;
    while let Some(index) = rest.find(':') {
        let after = &rest[index + 1..];
        let digit_count = after.chars().take_while(char::is_ascii_digit).count();
        if digit_count > 0 && after[digit_count..].starts_with(':') {
            return &line[..consumed + index];
        }
        consumed += index + 1;
        rest = after;
    }
    searched
}

/// Finds files by name.
#[must_use]
pub fn make_glob_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "glob",
            "Find files by search-root-relative path using a glob pattern. Use path to choose the \
             search root. `*` stays within one path segment and `**` searches recursively. Prefer \
             this over shell find or ls when locating repository files.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern relative to the search root"},
                    "path": {"type": "string", "description": "Directory to search in (default: working directory)"}
                },
                "required": ["pattern"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let pattern = required_str(&args, "pattern")?;
                let path = args.get("path").and_then(Value::as_str);

                let results = ctx.env.glob(pattern, path).await?;
                Ok(results.join("\n"))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Lists a directory.
#[must_use]
pub fn make_list_dir_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "list_dir",
            "List directory contents with depth control",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "depth": {"type": "integer", "description": "Depth of listing (default 1)"}
                },
                "required": ["path"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let path = required_str(&args, "path")?;
                let depth = optional_usize_arg(&args, "depth")?;

                let entries = ctx.env.list_directory(path, depth).await?;
                let lines: Vec<String> = entries
                    .iter()
                    .map(|entry| {
                        if entry.is_dir {
                            format!("{}/", entry.name)
                        } else {
                            entry.name.clone()
                        }
                    })
                    .collect();
                Ok(lines.join("\n"))
            })
        }),
        source:     ToolSource::Native,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::environment::DirEntry;
    use crate::test_support::MockEnvironment;
    use crate::tools::testing::{context, context_for};
    use crate::types::ToolErrorKind;

    #[tokio::test]
    async fn grep_returns_one_line_per_match() {
        let tool = make_grep_tool();
        let environment = MockEnvironment {
            grep_results: vec![
                "src/main.rs:10:fn main()".to_owned(),
                "src/lib.rs:5:pub fn".to_owned(),
            ],
            ..MockEnvironment::default()
        };

        let output = (tool.executor)(json!({"pattern": "fn"}), context(environment))
            .await
            .expect("the search runs");

        assert_eq!(output, "src/main.rs:10:fn main()\nsrc/lib.rs:5:pub fn");
    }

    #[tokio::test]
    async fn grep_without_a_pattern_is_an_argument_error() {
        let tool = make_grep_tool();

        let error = (tool.executor)(json!({}), context(MockEnvironment::default()))
            .await
            .expect_err("the pattern is required");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "Missing required parameter: pattern");
    }

    #[tokio::test]
    async fn grep_passes_its_controls_to_the_environment() {
        let tool = make_grep_tool();

        let output = (tool.executor)(
            json!({
                "pattern": "fn",
                "path": "src",
                "glob_filter": "*.rs",
                "case_insensitive": true,
                "max_results": 5
            }),
            context(MockEnvironment::default()),
        )
        .await
        .expect("the search runs");

        // The mock answers with no matches; what matters is that every
        // control parsed rather than failing the call.
        assert_eq!(output, "");
    }

    /// `files_with_matches` and `count` renderings both need the file path,
    /// which a search only prefixes when it scanned a directory.
    #[test]
    fn grep_result_path_handles_both_output_shapes() {
        // A directory scan prefixes the path.
        assert_eq!(
            grep_result_path("src/main.rs:42:fn main() {", "src"),
            "src/main.rs"
        );
        // A colon in the matched line is not the line-number field.
        assert_eq!(
            grep_result_path("src/a.rs:7:let x: u8 = 1;", "src"),
            "src/a.rs"
        );
        // A single-file scan omits the path, so what was searched is the
        // answer.
        assert_eq!(
            grep_result_path("42:fn main() {", "src/main.rs"),
            "src/main.rs"
        );
    }

    #[tokio::test]
    async fn glob_returns_one_line_per_path() {
        let tool = make_glob_tool();
        let environment = MockEnvironment {
            glob_results: vec!["src/main.rs".to_owned(), "src/lib.rs".to_owned()],
            ..MockEnvironment::default()
        };

        let output = (tool.executor)(json!({"pattern": "src/**/*.rs"}), context(environment))
            .await
            .expect("the search runs");

        assert_eq!(output, "src/main.rs\nsrc/lib.rs");
    }

    #[tokio::test]
    async fn glob_without_a_pattern_is_an_argument_error() {
        let tool = make_glob_tool();

        let error = (tool.executor)(json!({}), context(MockEnvironment::default()))
            .await
            .expect_err("the pattern is required");

        assert_eq!(error.message(), "Missing required parameter: pattern");
    }

    #[tokio::test]
    async fn list_dir_marks_directories_with_a_trailing_separator() {
        let tool = make_list_dir_tool();
        let environment = Arc::new(MockEnvironment {
            dir_entries: vec![
                DirEntry {
                    name:   "src".to_owned(),
                    is_dir: true,
                    size:   None,
                },
                DirEntry {
                    name:   "README.md".to_owned(),
                    is_dir: false,
                    size:   Some(12),
                },
            ],
            ..MockEnvironment::default()
        });

        let output = (tool.executor)(
            json!({"path": "/work", "depth": 2}),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the listing runs");

        assert_eq!(output, "src/\nREADME.md");
        assert_eq!(
            *environment
                .captured_listings
                .lock()
                .expect("captured_listings lock is not poisoned"),
            vec![("/work".to_owned(), Some(2))]
        );
    }

    #[tokio::test]
    async fn list_dir_without_a_path_is_an_argument_error() {
        let tool = make_list_dir_tool();

        let error = (tool.executor)(json!({}), context(MockEnvironment::default()))
            .await
            .expect_err("the path is required");

        assert_eq!(error.message(), "Missing required parameter: path");
    }
}
