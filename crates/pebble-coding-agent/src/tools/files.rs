//! Reading and changing files.
//!
//! Every one of these acts through [`Environment`](crate::Environment), so a
//! session working in a container edits the container's files and a session
//! working on this machine edits these.

use std::fmt::Write as _;
use std::sync::Arc;

use futures_util::{StreamExt as _, stream};
use lithos_llm::types::ToolDefinition;
use serde_json::Value;

use crate::tool::{NativeTool, RegisteredTool, ToolError, optional_usize_arg, required_str};
use crate::types::ToolSource;

/// How many lines `read_file` returns when the model names no limit.
pub(crate) const DEFAULT_READ_LINES: usize = 2000;

/// How many files `read_many_files` reads at once.
const MAX_READ_MANY_FILES_CONCURRENCY: usize = 8;

/// Reads a file as numbered lines.
///
/// The default limit is applied here rather than in the environment, so a
/// model that names no limit still gets a bounded read.
#[must_use]
pub fn make_read_file_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            NativeTool::ReadFile.canonical_name(),
            "Read files before editing them. Returns line-numbered text and supports offset/limit \
             for large files. Use this instead of shell cat, head, tail, or sed when inspecting \
             repository files.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "1-based line number to start reading from"},
                    "limit": {"type": "integer", "description": "Number of lines to read (default 2000)"}
                },
                "required": ["file_path"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let file_path = required_str(&args, "file_path")?;
                let offset = optional_usize_arg(&args, "offset")?;
                let limit = optional_usize_arg(&args, "limit")?.or(Some(DEFAULT_READ_LINES));

                Ok(ctx.env.read_file(file_path, offset, limit).await?)
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Writes a whole file.
#[must_use]
pub fn make_write_file_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            NativeTool::WriteFile.canonical_name(),
            "Create new files, or overwrite an existing file only when replacement is explicitly \
             intended. Prefer edit_file for targeted changes to existing files because write_file \
             overwrites the full file content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["file_path", "content"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let file_path = required_str(&args, "file_path")?;
                let content = required_str(&args, "content")?;

                ctx.env.write_file(file_path, content).await?;
                Ok(format!("Successfully wrote to {file_path}"))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Replaces one exact string in a file.
///
/// The file is read as raw text rather than as the numbered lines `read_file`
/// shows, so a match is what the file holds and not what the model was shown.
#[must_use]
pub fn make_edit_file_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            NativeTool::EditFile.canonical_name(),
            "Edit a file by replacing an exact string. The old_string must be an exact match and \
             unique unless replace_all is true; include surrounding context when needed. Read the \
             file first and preserve existing indentation.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "old_string": {"type": "string", "description": "The string to find and replace"},
                    "new_string": {"type": "string", "description": "The replacement string"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"}
                },
                "required": ["file_path", "old_string", "new_string"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let file_path = required_str(&args, "file_path")?;
                let old_string = required_str(&args, "old_string")?;
                let new_string = required_str(&args, "new_string")?;
                let replace_all = args
                    .get("replace_all")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);

                let raw_content = ctx.env.read_file_text(file_path).await?;

                let count = raw_content.matches(old_string).count();
                // The argument is what the model has to change, so both of
                // these are reported as arguments it can fix rather than as
                // the file system refusing the edit.
                if count == 0 {
                    return Err(ToolError::invalid_arguments("old_string not found in file"));
                }
                if count > 1 && !replace_all {
                    return Err(ToolError::invalid_arguments(format!(
                        "old_string is not unique in file (found {count} occurrences). Use \
                         replace_all or provide more context"
                    )));
                }

                let new_content = if replace_all {
                    raw_content.replace(old_string, new_string)
                } else {
                    raw_content.replacen(old_string, new_string, 1)
                };

                ctx.env.write_existing_file(file_path, &new_content).await?;
                Ok(format!("Successfully edited {file_path}"))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Reads several files in one call.
///
/// A file that cannot be read is reported in its own block rather than failing
/// the call, because a model that asked for ten files is better served by the
/// nine that exist.
#[must_use]
pub fn make_read_many_files_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            NativeTool::ReadManyFiles.canonical_name(),
            "Read multiple files at once",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of absolute file paths to read"
                    }
                },
                "required": ["paths"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let paths = read_many_files_paths(&args)?;

                let results = stream::iter(paths)
                    .map(|path| {
                        let env = Arc::clone(&ctx.env);
                        async move {
                            let result = env.read_file(&path, None, None).await;
                            (path, result)
                        }
                    })
                    .buffered(MAX_READ_MANY_FILES_CONCURRENCY)
                    .collect::<Vec<_>>()
                    .await;

                let mut output = String::new();
                for (path, result) in results {
                    match result {
                        Ok(content) => {
                            let _ = write!(output, "=== {path} ===\n{content}\n\n");
                        }
                        Err(error) => {
                            let _ = write!(output, "=== {path} ===\nError: {error}\n\n");
                        }
                    }
                }
                Ok(output)
            })
        }),
        source:     ToolSource::Native,
    }
}

/// The `paths` argument of `read_many_files`, or the error the model is given
/// instead.
fn read_many_files_paths(args: &Value) -> Result<Vec<String>, ToolError> {
    args.get("paths")
        .and_then(Value::as_array)
        .ok_or_else(|| ToolError::invalid_arguments("paths must be an array"))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| ToolError::invalid_arguments("each path must be a string"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::test_support::{MockEnvironment, MutableMockEnvironment};
    use crate::tool::{ToolContext, ToolEnvProvider};
    use crate::tools::testing::{context, context_for};
    use crate::types::ToolErrorKind;

    fn file_context(path: &str, content: &str) -> ToolContext {
        context(MockEnvironment {
            files: HashMap::from([(path.to_owned(), content.to_owned())]),
            ..MockEnvironment::default()
        })
    }

    #[tokio::test]
    async fn read_file_returns_content() {
        let tool = make_read_file_tool();

        let output = (tool.executor)(
            json!({"file_path": "/test.txt"}),
            file_context("/test.txt", "hello\nworld"),
        )
        .await
        .expect("the file is read");

        assert_eq!(output, "1 | hello\n2 | world\n");
    }

    #[tokio::test]
    async fn read_file_applies_the_documented_default_limit() {
        let tool = make_read_file_tool();
        let content = (1..=DEFAULT_READ_LINES + 1)
            .map(|line| format!("line{line}"))
            .collect::<Vec<_>>()
            .join("\n");

        let output = (tool.executor)(
            json!({"file_path": "/test.txt"}),
            file_context("/test.txt", &content),
        )
        .await
        .expect("the file is read");

        assert!(output.contains("2000 | line2000"), "{output}");
        assert!(!output.contains("2001 | line2001"), "{output}");
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let tool = make_read_file_tool();

        let output = (tool.executor)(
            json!({"file_path": "/test.txt", "offset": 2, "limit": 2}),
            file_context("/test.txt", "line1\nline2\nline3\nline4"),
        )
        .await
        .expect("the file is read");

        assert_eq!(output, "2 | line2\n3 | line3\n");
    }

    #[tokio::test]
    async fn read_file_without_a_path_is_an_argument_error() {
        let tool = make_read_file_tool();

        let error = (tool.executor)(json!({}), context(MockEnvironment::default()))
            .await
            .expect_err("the path is required");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "Missing required parameter: file_path");
    }

    #[tokio::test]
    async fn read_file_reports_a_missing_file_as_an_execution_failure() {
        let tool = make_read_file_tool();

        let error = (tool.executor)(
            json!({"file_path": "/missing.txt"}),
            context(MockEnvironment::default()),
        )
        .await
        .expect_err("the file does not exist");

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert_eq!(error.message(), "File not found: /missing.txt");
    }

    /// Only a tool that runs a command resolves the call's environment
    /// variables, so a provider that is failing — an expired credential, a
    /// service that is down — does not stop the model from reading a file.
    #[tokio::test]
    async fn read_file_does_not_resolve_the_calls_environment_variables() {
        struct Failing;

        #[async_trait]
        impl ToolEnvProvider for Failing {
            async fn resolve(&self) -> Result<HashMap<String, String>, ToolError> {
                Err(ToolError::execution("GITHUB_TOKEN refresh failed"))
            }
        }

        let tool = make_read_file_tool();

        let output = (tool.executor)(
            json!({"file_path": "/test.txt"}),
            file_context("/test.txt", "hello").with_tool_env_provider(Arc::new(Failing)),
        )
        .await
        .expect("the file is read");

        assert_eq!(output, "1 | hello\n");
    }

    #[tokio::test]
    async fn write_file_calls_the_environment_create_path() {
        let tool = make_write_file_tool();
        let environment = Arc::new(MockEnvironment::default());

        let output = (tool.executor)(
            json!({"file_path": "/out.txt", "content": "hello"}),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the file is written");

        assert_eq!(output, "Successfully wrote to /out.txt");
        assert_eq!(environment.existing_file_write_count(), 0);
        assert_eq!(
            *environment
                .written_files
                .lock()
                .expect("written_files lock is not poisoned"),
            vec![("/out.txt".to_owned(), "hello".to_owned())]
        );
    }

    #[tokio::test]
    async fn edit_file_replaces_its_match_through_the_existing_file_path() {
        let tool = make_edit_file_tool();
        let environment = Arc::new(MockEnvironment {
            files: HashMap::from([("/f.txt".to_owned(), "hello world".to_owned())]),
            ..MockEnvironment::default()
        });

        let output = (tool.executor)(
            json!({
                "file_path": "/f.txt",
                "old_string": "hello",
                "new_string": "goodbye"
            }),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the file is edited");

        assert_eq!(output, "Successfully edited /f.txt");
        assert_eq!(environment.existing_file_write_count(), 1);
        let written = environment
            .written_files
            .lock()
            .expect("written_files lock is not poisoned");
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].1, "goodbye world");
    }

    #[tokio::test]
    async fn edit_file_reports_a_string_it_could_not_find() {
        let tool = make_edit_file_tool();

        let error = (tool.executor)(
            json!({
                "file_path": "/f.txt",
                "old_string": "missing",
                "new_string": "replacement"
            }),
            file_context("/f.txt", "hello world"),
        )
        .await
        .expect_err("the string is not in the file");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "old_string not found in file");
    }

    #[tokio::test]
    async fn edit_file_refuses_a_string_that_is_not_unique() {
        let tool = make_edit_file_tool();

        let error = (tool.executor)(
            json!({
                "file_path": "/f.txt",
                "old_string": "aa",
                "new_string": "cc"
            }),
            file_context("/f.txt", "aa bb aa"),
        )
        .await
        .expect_err("the string appears twice");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(
            error.message(),
            "old_string is not unique in file (found 2 occurrences). Use replace_all or provide \
             more context"
        );
    }

    #[tokio::test]
    async fn edit_file_replaces_every_occurrence_when_it_is_asked_to() {
        let tool = make_edit_file_tool();
        let environment = Arc::new(MockEnvironment {
            files: HashMap::from([("/f.txt".to_owned(), "aa bb aa".to_owned())]),
            ..MockEnvironment::default()
        });

        let output = (tool.executor)(
            json!({
                "file_path": "/f.txt",
                "old_string": "aa",
                "new_string": "cc",
                "replace_all": true
            }),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the file is edited");

        assert_eq!(output, "Successfully edited /f.txt");
        let written = environment
            .written_files
            .lock()
            .expect("written_files lock is not poisoned");
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].1, "cc bb cc");
    }

    /// The file is edited as it is stored, so text that happens to look like
    /// the numbering `read_file` adds survives the edit.
    #[tokio::test]
    async fn edit_file_preserves_literal_line_number_prefixes() {
        let tool = make_edit_file_tool();
        let environment = Arc::new(MockEnvironment {
            files: HashMap::from([(
                "/f.txt".to_owned(),
                "1 | keep this literal\nhello".to_owned(),
            )]),
            ..MockEnvironment::default()
        });

        let output = (tool.executor)(
            json!({
                "file_path": "/f.txt",
                "old_string": "hello",
                "new_string": "goodbye"
            }),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the file is edited");

        assert_eq!(output, "Successfully edited /f.txt");
        let written = environment
            .written_files
            .lock()
            .expect("written_files lock is not poisoned");
        assert_eq!(written[0].1, "1 | keep this literal\ngoodbye");
    }

    #[tokio::test]
    async fn read_many_files_reads_every_path_it_is_given() {
        let tool = make_read_many_files_tool();
        let environment = MutableMockEnvironment::new(HashMap::from([
            ("/a.txt".to_owned(), "alpha".to_owned()),
            ("/b.txt".to_owned(), "beta".to_owned()),
        ]));

        let output = (tool.executor)(
            json!({"paths": ["/a.txt", "/b.txt"]}),
            context_for(Arc::new(environment)),
        )
        .await
        .expect("the files are read");

        assert_eq!(
            output,
            "=== /a.txt ===\n1 | alpha\n\n\n=== /b.txt ===\n1 | beta\n\n\n"
        );
    }

    #[tokio::test]
    async fn read_many_files_reports_a_missing_file_without_failing_the_call() {
        let tool = make_read_many_files_tool();

        let output = (tool.executor)(
            json!({"paths": ["/a.txt", "/missing.txt"]}),
            file_context("/a.txt", "alpha"),
        )
        .await
        .expect("one unreadable file does not fail the call");

        assert!(output.contains("=== /a.txt ===\n1 | alpha"), "{output}");
        assert!(
            output.contains("=== /missing.txt ===\nError: File not found: /missing.txt"),
            "{output}"
        );
    }

    #[tokio::test]
    async fn read_many_files_refuses_paths_that_are_not_strings() {
        let tool = make_read_many_files_tool();

        let missing = (tool.executor)(json!({}), context(MockEnvironment::default()))
            .await
            .expect_err("paths is required");
        let wrong_element =
            (tool.executor)(json!({"paths": [7]}), context(MockEnvironment::default()))
                .await
                .expect_err("a path must be a string");

        assert_eq!(missing.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(missing.message(), "paths must be an array");
        assert_eq!(wrong_element.message(), "each path must be a string");
    }
}
