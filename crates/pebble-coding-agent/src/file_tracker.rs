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
use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::types::{ToolCall, ToolResult};
use serde_json::Value;

use crate::tool::{NativeTool, canonical_tool_name};
use crate::tools::apply_patch::{PatchOperation, parse_apply_patch};

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
/// prompts that did the same work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct FileTracker {
    files: BTreeMap<String, FileOps>,
}

/// The files one prompt wrote or edited, across a session and its children.
///
/// Where [`FileTracker`] outlives prompts and exists for compaction, this is
/// the prompt's own answer to "what did that change?", reset when a prompt
/// begins and read into its report when it ends. It is shared between a
/// session's conversation and the supervisor of its children: a child that
/// finishes a turn adds what it touched, so a parent's report covers the whole
/// tree. Cheap to clone; every clone reads and writes the same list.
#[derive(Debug, Clone, Default)]
pub(crate) struct PromptFiles(Arc<Mutex<PromptFilesState>>);

#[derive(Debug, Default)]
struct PromptFilesState {
    /// Every path touched, in the order first touched, each once.
    touched: Vec<String>,
    /// The path touched most recently.
    last:    Option<String>,
}

impl PromptFiles {
    /// Records paths written or edited, in the order they were touched.
    pub(crate) fn record(&self, paths: impl IntoIterator<Item = String>) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        for path in paths {
            if !state.touched.contains(&path) {
                state.touched.push(path.clone());
            }
            state.last = Some(path);
        }
    }

    /// Forgets everything, at the start of a prompt.
    pub(crate) fn reset(&self) {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        state.touched.clear();
        state.last = None;
    }

    /// The paths touched so far, in touch order, and the most recent one.
    pub(crate) fn snapshot(&self) -> (Vec<String>, Option<String>) {
        let state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        (state.touched.clone(), state.last.clone())
    }
}

impl FileTracker {
    /// Records that a file was read.
    pub(crate) fn record_read(&mut self, path: &str) {
        self.files.entry(path.to_owned()).or_default().read = true;
    }

    /// Records that a file was written whole.
    pub(crate) fn record_write(&mut self, path: &str) {
        self.files.entry(path.to_owned()).or_default().written = true;
    }

    /// Records that a file was edited in place.
    pub(crate) fn record_edit(&mut self, path: &str) {
        self.files.entry(path.to_owned()).or_default().edited = true;
    }

    /// Whether no file has been touched.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// How many distinct files have been touched.
    #[must_use]
    pub(crate) fn file_count(&self) -> usize {
        self.files.len()
    }

    /// The tracked files as one Markdown list, one line per path.
    ///
    /// ```ignore
    /// # use pebble_coding_agent::resources::FileTracker;
    /// let mut tracker = FileTracker::default();
    /// tracker.record_read("src/lib.rs");
    /// tracker.record_edit("src/lib.rs");
    /// assert_eq!(tracker.render(), "- src/lib.rs (read, edited)\n");
    /// ```
    #[must_use]
    pub(crate) fn render(&self) -> String {
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

    /// Records the file work in one round of answered tool calls, and answers
    /// with the paths that round wrote or edited, in call order.
    ///
    /// Calls and results are paired in order, which is the order the agent
    /// loop answers them in. A call whose result reports an error is skipped:
    /// the file was not touched.
    pub(crate) fn record_from_tool_calls(
        &mut self,
        tool_calls: &[ToolCall],
        results: &[ToolResult],
    ) -> Vec<String> {
        let mut written = Vec::new();
        for (call, result) in tool_calls.iter().zip(results) {
            if result.is_error {
                continue;
            }

            let Ok(arguments) = call.input.to_value() else {
                continue;
            };
            match NativeTool::from_canonical_name(canonical_tool_name(&call.name)) {
                Some(NativeTool::ReadFile) => {
                    if let Some(path) = file_path(&arguments) {
                        self.record_read(path);
                    }
                }
                Some(NativeTool::WriteFile) => {
                    if let Some(path) = file_path(&arguments) {
                        self.record_write(path);
                        written.push(path.to_owned());
                    }
                }
                Some(NativeTool::EditFile) => {
                    if let Some(path) = file_path(&arguments) {
                        self.record_edit(path);
                        written.push(path.to_owned());
                    }
                }
                Some(NativeTool::ApplyPatch) => {
                    written.extend(self.record_from_patch_arguments(&arguments));
                }
                _ => {}
            }
        }
        written
    }

    /// Reads the operations out of the patch the call carried.
    ///
    /// Parsed from the arguments rather than from the rendered summary: the
    /// arguments survive per-tool output truncation, which can cut the leading
    /// lines of a very large patch's file list, and the parser is the one the
    /// tool itself ran. A patch that stopped parsing records nothing, matching
    /// the failed call it produced.
    ///
    /// Deletions are deliberately not recorded, as fabro's tracker skipped
    /// them: this list exists so files can be revisited after compaction, and
    /// a deleted file cannot be. An update that moves a file records the
    /// destination, which is where the result lives now.
    fn record_from_patch_arguments(&mut self, arguments: &Value) -> Vec<String> {
        let mut written = Vec::new();
        let Some(patch) = patch_text(arguments) else {
            return written;
        };
        let Ok(operations) = parse_apply_patch(patch) else {
            return written;
        };
        for operation in operations {
            match operation {
                PatchOperation::Add { path, .. } => {
                    self.record_write(&path);
                    written.push(path);
                }
                PatchOperation::Update { path, new_path, .. } => {
                    let path = new_path.unwrap_or(path);
                    self.record_edit(&path);
                    written.push(path);
                }
                PatchOperation::Delete { .. } => {}
            }
        }
        written
    }
}

/// The paths a write, edit, or patch call will touch, read from its
/// arguments alone, for a fold over the event stream that pairs a call's
/// start with its completion. Reads and deletions name nothing.
pub(crate) fn written_paths_from_arguments(tool_name: &str, arguments: &Value) -> Vec<String> {
    match NativeTool::from_canonical_name(canonical_tool_name(tool_name)) {
        Some(NativeTool::WriteFile | NativeTool::EditFile) => file_path(arguments)
            .map(str::to_owned)
            .into_iter()
            .collect(),
        Some(NativeTool::ApplyPatch) => patch_text(arguments)
            .and_then(|patch| parse_apply_patch(patch).ok())
            .map(|operations| {
                operations
                    .into_iter()
                    .filter_map(|operation| match operation {
                        PatchOperation::Add { path, .. } => Some(path),
                        PatchOperation::Update { path, new_path, .. } => {
                            Some(new_path.unwrap_or(path))
                        }
                        PatchOperation::Delete { .. } => None,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// The patch text a call carried: raw text from the grammar tool, or a
/// `patch` member where a harness wraps it in JSON.
fn patch_text(arguments: &Value) -> Option<&str> {
    arguments
        .as_str()
        .or_else(|| arguments.get("patch")?.as_str())
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

    const PATCH: &str = "\
*** Begin Patch
*** Add File: src/new.rs
+fn main() {}
*** Update File: src/old.rs
@@ fn old():
-    pass
+    return 1
*** End Patch";

    #[test]
    fn a_patch_records_added_and_modified_files_from_its_arguments() {
        let mut tracker = FileTracker::default();

        // The grammar tool carries the patch as raw text.
        tracker.record_from_tool_calls(&[call("apply_patch", json!(PATCH))], &[success(
            "tc1", "ok",
        )]);

        assert_eq!(
            tracker.render(),
            "- src/new.rs (written)\n- src/old.rs (edited)\n"
        );
    }

    #[test]
    fn a_json_wrapped_patch_is_read_from_its_patch_member() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[call("apply_patch", json!({ "patch": PATCH }))], &[
            success("tc1", "ok"),
        ]);

        assert_eq!(
            tracker.render(),
            "- src/new.rs (written)\n- src/old.rs (edited)\n"
        );
    }

    /// The arguments survive output truncation, so a summary whose leading
    /// lines were cut still records every file the patch touched.
    #[test]
    fn a_truncated_summary_does_not_lose_the_patchs_files() {
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[call("apply_patch", json!(PATCH))], &[success(
            "tc1",
            "[... output truncated ...]\nM src/other.rs\n",
        )]);

        assert_eq!(
            tracker.render(),
            "- src/new.rs (written)\n- src/old.rs (edited)\n"
        );
    }

    #[test]
    fn a_moved_file_is_recorded_at_its_destination() {
        let patch = "\
*** Begin Patch
*** Update File: src/old.py
*** Move to: src/new.py
@@ def hello():
-    pass
+    return 1
*** End Patch";
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[call("apply_patch", json!(patch))], &[success(
            "tc1", "ok",
        )]);

        assert_eq!(tracker.render(), "- src/new.py (edited)\n");
    }

    #[test]
    fn a_deletion_is_not_recorded() {
        let patch = "\
*** Begin Patch
*** Delete File: src/gone.rs
*** End Patch";
        let mut tracker = FileTracker::default();

        tracker.record_from_tool_calls(&[call("apply_patch", json!(patch))], &[success(
            "tc1", "ok",
        )]);

        assert!(tracker.is_empty());
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
