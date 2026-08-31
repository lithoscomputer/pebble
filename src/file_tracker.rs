//! What a session has done to which files.
//!
//! Compaction throws away most of a conversation, and the file work is the part
//! a summary most often loses: which files were read, which were written, which
//! were edited. The tracker watches answered tool calls, keeps one line per
//! path, and hands compaction a section it asks the model to copy through
//! verbatim.
//!
//! It records paths, never content, and only for calls that succeeded.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use lithos_llm::types::{ToolCall, ToolResult};
use serde_json::Value;

use crate::tool::{NativeTool, canonical_tool_name, result_text};

/// What a session did to one file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FileOps {
    read:    bool,
    written: bool,
    edited:  bool,
}

/// The files one session has touched, and how.
///
/// Paths are kept in sorted order, so the rendered section is stable between
/// runs that did the same work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileTracker {
    files: BTreeMap<String, FileOps>,
}

impl FileTracker {
    /// Records that a file was read.
    pub fn record_read(&mut self, path: &str) {
        self.files.entry(path.to_owned()).or_default().read = true;
    }

    /// Records that a file was written whole.
    pub fn record_write(&mut self, path: &str) {
        self.files.entry(path.to_owned()).or_default().written = true;
    }

    /// Records that a file was edited in place.
    pub fn record_edit(&mut self, path: &str) {
        self.files.entry(path.to_owned()).or_default().edited = true;
    }

    /// Whether no file has been touched.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// How many distinct files have been touched.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// The tracked files as one Markdown list, one line per path.
    ///
    /// ```
    /// # use pebble::FileTracker;
    /// let mut tracker = FileTracker::default();
    /// tracker.record_read("src/lib.rs");
    /// tracker.record_edit("src/lib.rs");
    /// assert_eq!(tracker.render(), "- src/lib.rs (read, edited)\n");
    /// ```
    #[must_use]
    pub fn render(&self) -> String {
        let mut output = String::new();
        for (path, ops) in &self.files {
            let mut labels = Vec::new();
            if ops.read {
                labels.push("read");
            }
            if ops.written {
                labels.push("written");
            }
            if ops.edited {
                labels.push("edited");
            }
            let _ = writeln!(output, "- {path} ({})", labels.join(", "));
        }
        output
    }

    /// Records the file work in one round of answered tool calls.
    ///
    /// Calls and results are paired in order, which is the order
    /// [`ToolDispatch`](crate::ToolDispatch) answers them in. A call whose
    /// result reports an error is skipped: the file was not touched.
    pub fn record_from_tool_calls(&mut self, tool_calls: &[ToolCall], results: &[ToolResult]) {
        for (call, result) in tool_calls.iter().zip(results) {
            if result.is_error {
                continue;
            }

            match NativeTool::from_canonical_name(canonical_tool_name(&call.name)) {
                Some(NativeTool::ReadFile) => {
                    if let Some(path) = file_path(&call.arguments) {
                        self.record_read(path);
                    }
                }
                Some(NativeTool::WriteFile) => {
                    if let Some(path) = file_path(&call.arguments) {
                        self.record_write(path);
                    }
                }
                Some(NativeTool::EditFile) => {
                    if let Some(path) = file_path(&call.arguments) {
                        self.record_edit(path);
                    }
                }
                Some(NativeTool::ApplyPatch) => self.record_from_patch_summary(result),
                _ => {}
            }
        }
    }

    /// Reads the file list `apply_patch` prints, which names each path by the
    /// change it made: `A` added, `M` modified.
    fn record_from_patch_summary(&mut self, result: &ToolResult) {
        for line in result_text(result).lines() {
            let line = line.trim();
            if let Some(path) = line.strip_prefix("A ") {
                self.record_write(path.trim());
            } else if let Some(path) = line.strip_prefix("M ") {
                self.record_edit(path.trim());
            }
        }
    }
}

/// The path a file tool was given, under either name the vocabularies use.
fn file_path(arguments: &Value) -> Option<&str> {
    arguments
        .get("file_path")
        .or_else(|| arguments.get("path"))
        .and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::ContentPart;
    use serde_json::json;

    use super::*;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall::function("tc1", name, arguments)
    }

    fn success(id: &str, text: &str) -> ToolResult {
        ToolResult {
            tool_call_id: id.to_owned(),
            name:         None,
            content:      vec![ContentPart::Text {
                text: text.to_owned(),
            }],
            is_error:     false,
        }
    }

    fn failure(id: &str, text: &str) -> ToolResult {
        ToolResult {
            is_error: true,
            ..success(id, text)
        }
    }

    #[test]
    fn a_read_renders_the_read_label() {
        let mut tracker = FileTracker::default();

        tracker.record_read("src/main.rs");

        assert_eq!(tracker.render(), "- src/main.rs (read)\n");
    }

    #[test]
    fn every_operation_on_one_file_shares_a_line() {
        let mut tracker = FileTracker::default();

        tracker.record_read("src/lib.rs");
        tracker.record_write("src/lib.rs");
        tracker.record_edit("src/lib.rs");

        assert_eq!(tracker.render(), "- src/lib.rs (read, written, edited)\n");
    }

    #[test]
    fn files_render_in_path_order() {
        let mut tracker = FileTracker::default();

        tracker.record_write("z.rs");
        tracker.record_read("a.rs");

        assert_eq!(tracker.render(), "- a.rs (read)\n- z.rs (written)\n");
    }

    #[test]
    fn a_successful_read_call_is_recorded() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(
            &[call("read_file", json!({ "file_path": "/tmp/foo.rs" }))],
            &[success("tc1", "file contents")],
        );

        assert_eq!(tracker.render(), "- /tmp/foo.rs (read)\n");
    }

    #[test]
    fn a_successful_write_call_is_recorded() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(
            &[call(
                "write_file",
                json!({ "file_path": "/tmp/bar.rs", "content": "hello" }),
            )],
            &[success("tc1", "ok")],
        );

        assert_eq!(tracker.render(), "- /tmp/bar.rs (written)\n");
    }

    #[test]
    fn a_successful_edit_call_is_recorded() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(
            &[call("edit_file", json!({ "file_path": "/tmp/baz.rs" }))],
            &[success("tc1", "ok")],
        );

        assert_eq!(tracker.render(), "- /tmp/baz.rs (edited)\n");
    }

    #[test]
    fn another_vocabularys_names_and_path_argument_are_recorded() {
        let mut tracker = FileTracker::default();
        let calls = [
            ToolCall::function("tc1", "Read", json!({ "path": "/tmp/a.rs" })),
            ToolCall::function(
                "tc2",
                "Write",
                json!({ "path": "/tmp/b.rs", "content": "x" }),
            ),
            ToolCall::function("tc3", "Edit", json!({ "path": "/tmp/c.rs" })),
        ];
        let results: Vec<_> = ["tc1", "tc2", "tc3"]
            .into_iter()
            .map(|id| success(id, "ok"))
            .collect();

        tracker.record_from_tool_calls(&calls, &results);

        assert_eq!(
            tracker.render(),
            "- /tmp/a.rs (read)\n- /tmp/b.rs (written)\n- /tmp/c.rs (edited)\n"
        );
    }

    #[test]
    fn a_failed_call_touched_nothing() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(
            &[call("read_file", json!({ "file_path": "/tmp/missing.rs" }))],
            &[failure("tc1", "File not found")],
        );

        assert!(tracker.is_empty());
    }

    #[test]
    fn a_patch_summary_records_added_and_modified_files() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[call("apply_patch", json!({ "patch": "..." }))], &[
            success(
                "tc1",
                "Success. Updated the following files:\nA src/new.rs\nM src/old.rs\n",
            ),
        ]);

        assert_eq!(
            tracker.render(),
            "- src/new.rs (written)\n- src/old.rs (edited)\n"
        );
    }

    #[test]
    fn an_empty_tracker_counts_nothing() {
        let mut tracker = FileTracker::default();
        assert!(tracker.is_empty());
        assert_eq!(tracker.file_count(), 0);

        tracker.record_read("a.rs");
        tracker.record_write("b.rs");

        assert!(!tracker.is_empty());
        assert_eq!(tracker.file_count(), 2);
    }

    #[test]
    fn a_tool_that_touches_no_file_is_ignored() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[call("shell", json!({ "command": "ls" }))], &[success(
            "tc1",
            "file1\nfile2",
        )]);

        assert!(tracker.is_empty());
    }

    #[test]
    fn a_file_call_without_a_path_argument_is_ignored() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[call("read_file", json!({ "offset": 1 }))], &[success(
            "tc1", "",
        )]);

        assert!(tracker.is_empty());
    }

    #[test]
    fn a_result_without_a_matching_call_is_ignored() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[], &[success("tc1", "A src/new.rs")]);

        assert!(tracker.is_empty());
    }
}
