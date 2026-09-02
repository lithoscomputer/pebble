// Copyright 2026 OpenAI
// SPDX-License-Identifier: Apache-2.0
// Ported from openai/codex codex-rs/apply-patch at 932f72c225.

//! The `apply_patch` tool, in the format Codex-trained models write.
//!
//! A patch is one envelope — `*** Begin Patch` … `*** End Patch` — holding any
//! number of file operations. It is not a unified diff: a hunk names the lines
//! it expects to find rather than where they are, so the model does not have to
//! count line numbers it cannot see.
//!
//! What the model sends is free-form text, not JSON, so the tool is advertised
//! as a custom tool carrying the Lark grammar beside this module. The grammar
//! and the parser here describe the same format, and the parser is the one that
//! decides: a provider that enforces the grammar simply refuses earlier.
//!
//! Matching a hunk is deliberately forgiving, because a model quoting a file
//! back from its context loses trailing spaces and turns quotes into
//! typographic ones. Four passes are tried in order — exact, ignoring trailing
//! whitespace, ignoring all surrounding whitespace, and with typographic
//! characters folded back — and the first that matches wins. A cursor moves
//! forward as each hunk lands, so a patch that changes three identical lines
//! changes them in order.

use std::fmt::Write as _;
use std::slice;
use std::sync::Arc;

use lithos_llm::types::ToolDefinition;

use crate::environment::{Environment, EnvironmentError};
use crate::tool::{NativeTool, RegisteredTool, ToolError};
use crate::types::ToolSource;

/// The grammar the model is given, comments and all.
const APPLY_PATCH_LARK_GRAMMAR: &str = include_str!("apply_patch/apply_patch.lark");

/// The grammar without its `//` comment lines.
///
/// The attribution at the top of the file is for whoever reads the source; a
/// provider compiling the grammar has no use for it, and Lark does not promise
/// to accept it.
fn apply_patch_lark_grammar_definition() -> String {
    APPLY_PATCH_LARK_GRAMMAR
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One line inside a hunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Change {
    /// A line the patch expects to find and takes out.
    Remove(String),
    /// A line the patch puts in.
    Add(String),
    /// A line the patch expects to find and leaves alone.
    Context(String),
}

/// One contiguous change to a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hunk {
    /// What the `@@` line named, or empty for a bare `@@`.
    ///
    /// A non-empty anchor is searched for first, and the hunk is matched below
    /// wherever it was found.
    pub(crate) context_line: String,
    /// The lines this hunk expects, takes out, and puts in.
    pub(crate) changes:      Vec<Change>,
    /// Whether the hunk was marked `*** End of File`, which pins it to the end
    /// of the file rather than the first place it matches.
    pub(crate) end_of_file:  bool,
}

/// One file operation a patch asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PatchOperation {
    /// Write a whole new file. An existing file at the path is overwritten,
    /// which is what Codex does.
    Add {
        /// Where the file goes.
        path:    String,
        /// What it holds.
        content: String,
    },
    /// Remove a file, which must exist.
    Delete {
        /// The file to remove.
        path: String,
    },
    /// Change a file in place, optionally moving it.
    Update {
        /// The file to read.
        path:     String,
        /// Where to write the result, when the patch also moves the file. The
        /// original is removed.
        new_path: Option<String>,
        /// The changes, in the order they are applied.
        hunks:    Vec<Hunk>,
    },
}

/// Whether `line` opens a hunk.
fn is_hunk_start(line: &str) -> bool {
    line == "@@" || line.starts_with("@@ ")
}

/// The anchor a `@@` line names, with the optional trailing `@@` removed.
fn extract_context_line(line: &str) -> String {
    if line == "@@" {
        String::new()
    } else {
        let raw = line.strip_prefix("@@ ").unwrap_or(line);
        raw.strip_suffix(" @@").unwrap_or(raw).trim().to_owned()
    }
}

/// Reads patch text into the operations it asks for.
///
/// Nothing is applied here, so a patch that parses may still fail to match the
/// files it names.
///
/// # Errors
///
/// Returns a [`ToolError`] of kind
/// [`InvalidArguments`](crate::tools::ToolErrorKind::InvalidArguments) when the
/// text is not a patch: the envelope is missing, a hunk holds a line with no
/// `+`, `-`, or space prefix, or an `Add`/`Update` block is empty. The model
/// repairs every one of those by sending a different patch.
pub(crate) fn parse_apply_patch(text: &str) -> Result<Vec<PatchOperation>, ToolError> {
    let lines: Vec<&str> = text.trim().lines().collect();
    let lines = patch_lines_with_valid_boundaries(&lines)?;
    let mut ops = Vec::new();
    let mut i = 0;

    i += 1;

    while i < lines.len() {
        let line = lines[i].trim();

        if line == "*** End Patch" {
            break;
        }

        if let Some(path) = line.strip_prefix("*** Add File: ") {
            let path = path.to_owned();
            i += 1;
            let mut content = String::new();
            let mut add_lines = 0;
            while i < lines.len() {
                let l = lines[i];
                if l.starts_with("*** ") {
                    break;
                }
                if let Some(text_line) = l.strip_prefix('+') {
                    content.push_str(text_line);
                    content.push('\n');
                    add_lines += 1;
                } else {
                    return Err(ToolError::invalid_arguments(format!(
                        "Expected '+' prefix in Add File block, got: {l}"
                    )));
                }
                i += 1;
            }
            if add_lines == 0 {
                return Err(ToolError::invalid_arguments(format!(
                    "Add file hunk for path '{path}' is empty"
                )));
            }
            ops.push(PatchOperation::Add { path, content });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            ops.push(PatchOperation::Delete {
                path: path.to_owned(),
            });
            i += 1;
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = path.to_owned();
            i += 1;

            // Check for *** Move to:
            let new_path = if i < lines.len() {
                if let Some(np) = lines[i].trim().strip_prefix("*** Move to: ") {
                    i += 1;
                    Some(np.to_owned())
                } else {
                    None
                }
            } else {
                None
            };

            let mut hunks = Vec::new();
            while i < lines.len() {
                let l = lines[i];
                if l.starts_with("*** ") && !is_hunk_start(l) {
                    break;
                }
                if is_hunk_start(l) {
                    // Consume stacked @@ lines, keeping the last context
                    let mut context_line = extract_context_line(l);
                    i += 1;
                    while i < lines.len() && is_hunk_start(lines[i]) {
                        context_line = extract_context_line(lines[i]);
                        i += 1;
                    }

                    let mut changes = Vec::new();
                    while i < lines.len() {
                        let cl = lines[i];
                        if cl.starts_with("*** ") || is_hunk_start(cl) {
                            break;
                        }
                        if let Some(removed) = cl.strip_prefix('-') {
                            changes.push(Change::Remove(removed.to_owned()));
                        } else if let Some(added) = cl.strip_prefix('+') {
                            changes.push(Change::Add(added.to_owned()));
                        } else if let Some(ctx) = cl.strip_prefix(' ') {
                            changes.push(Change::Context(ctx.to_owned()));
                        } else if cl.is_empty() {
                            changes.push(Change::Context(String::new()));
                        } else {
                            return Err(ToolError::invalid_arguments(format!(
                                "Unexpected line in hunk (expected +, -, or space prefix): {cl}"
                            )));
                        }
                        i += 1;
                    }

                    // Check for *** End of File marker
                    let end_of_file = if i < lines.len() && lines[i].trim() == "*** End of File" {
                        i += 1;
                        true
                    } else {
                        false
                    };

                    hunks.push(Hunk {
                        context_line,
                        changes,
                        end_of_file,
                    });
                } else {
                    return Err(ToolError::invalid_arguments(format!(
                        "Expected @@ context line, got: {l}"
                    )));
                }
            }
            if hunks.is_empty() {
                return Err(ToolError::invalid_arguments(format!(
                    "Update file hunk for path '{path}' is empty"
                )));
            }
            ops.push(PatchOperation::Update {
                path,
                new_path,
                hunks,
            });
        } else {
            return Err(ToolError::invalid_arguments(format!(
                "Unexpected line in patch: {line}"
            )));
        }
    }

    Ok(ops)
}

/// The patch body, with a heredoc wrapper taken off when there is one.
///
/// A model told to run `apply_patch` from a shell sometimes sends what it would
/// have typed, heredoc and all. Unwrapping it costs one comparison and saves a
/// round trip.
fn patch_lines_with_valid_boundaries<'a>(lines: &'a [&'a str]) -> Result<&'a [&'a str], ToolError> {
    match check_patch_boundaries_strict(lines) {
        Ok(()) => Ok(lines),
        Err(original_error) => {
            if let [first, .., last] = lines
                && (*first == "<<EOF" || *first == "<<'EOF'" || *first == "<<\"EOF\"")
                && last.ends_with("EOF")
                && lines.len() >= 4
            {
                let inner = &lines[1..lines.len() - 1];
                check_patch_boundaries_strict(inner)?;
                return Ok(inner);
            }
            Err(original_error)
        }
    }
}

/// Whether the first and last lines are the patch envelope.
fn check_patch_boundaries_strict(lines: &[&str]) -> Result<(), ToolError> {
    let first_line = lines.first().map(|line| line.trim());
    let last_line = lines.last().map(|line| line.trim());

    match (first_line, last_line) {
        (Some("*** Begin Patch"), Some("*** End Patch")) => Ok(()),
        (Some(first), _) if first != "*** Begin Patch" => Err(ToolError::invalid_arguments(
            "The first line of the patch must be '*** Begin Patch'",
        )),
        _ => Err(ToolError::invalid_arguments(
            "The last line of the patch must be '*** End Patch'",
        )),
    }
}

/// Applies parsed operations to the files `env` holds.
///
/// Operations run in the order the patch listed them, and the first failure
/// stops the prompt — so a patch whose third operation cannot match leaves the
/// first two applied. That is Codex's behavior, and the summary names only what
/// a successful run changed.
///
/// # Errors
///
/// Returns a [`ToolError`] of kind
/// [`InvalidArguments`](crate::tools::ToolErrorKind::InvalidArguments) when the
/// patch cannot be matched against the files it names — an anchor that is not
/// there, lines that are not there, a file to delete that does not exist —
/// because the model repairs those by reading the file and sending a different
/// patch. A failure of the environment itself keeps the kind the environment
/// reported.
pub(crate) async fn apply_patch_operations(
    ops: &[PatchOperation],
    env: &dyn Environment,
) -> Result<String, ToolError> {
    if ops.is_empty() {
        return Err(ToolError::invalid_arguments("No files were modified."));
    }

    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut deleted = Vec::new();

    for op in ops {
        match op {
            PatchOperation::Add { path, content } => {
                env.write_file(path, content)
                    .await
                    .map_err(|error| failure(&format!("Failed to write file {path}"), error))?;
                added.push(path.clone());
            }
            PatchOperation::Delete { path } => {
                if !env
                    .file_exists(path)
                    .await
                    .map_err(|error| failure(&format!("Failed to delete file {path}"), error))?
                {
                    return Err(ToolError::invalid_arguments(format!(
                        "Failed to delete file {path}: file does not exist"
                    )));
                }
                env.delete_file(path)
                    .await
                    .map_err(|error| failure(&format!("Failed to delete file {path}"), error))?;
                deleted.push(path.clone());
            }
            PatchOperation::Update {
                path,
                new_path,
                hunks,
            } => {
                let original = env.read_file_text(path).await.map_err(|error| {
                    failure(&format!("Failed to read file to update {path}"), error)
                })?;
                let updated = apply_hunks(path, &original, hunks)?;
                let dest = new_path.as_deref().unwrap_or(path);
                env.write_file(dest, &updated)
                    .await
                    .map_err(|error| failure(&format!("Failed to write file {dest}"), error))?;
                if new_path.is_some() {
                    env.delete_file(path).await.map_err(|error| {
                        failure(&format!("Failed to remove original {path}"), error)
                    })?;
                }
                modified.push(dest.to_owned());
            }
        }
    }

    Ok(format_summary(&added, &modified, &deleted))
}

/// An environment failure, said the way the patch tool says it.
///
/// The environment's own rendering — its message and its causes — is kept
/// after the prefix, so the model reads why the file operation failed, and the
/// environment error's cause stays attached for logs.
fn failure(prefix: &str, error: EnvironmentError) -> ToolError {
    let message = format!("{prefix}: {}", error.detail());
    ToolError::from(error).with_message(message)
}

/// Folds one typographic character back to the ASCII one it stands in for.
fn normalize_char(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => '-',
        '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
        | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200A}' | '\u{202F}' | '\u{205F}'
        | '\u{3000}' => ' ',
        other => other,
    }
}

/// One line with its typographic characters folded and its edges trimmed.
fn normalize_unicode(s: &str) -> String {
    s.trim().chars().map(normalize_char).collect()
}

/// Where `pattern` sits in `lines`, or `None`.
///
/// Four passes, each more forgiving than the last, so an exact match always
/// wins over a fuzzy one anywhere in the file. `eof` starts the search at the
/// last position the pattern could occupy, which is what `*** End of File`
/// means.
fn seek_sequence(lines: &[String], pattern: &[String], start: usize, eof: bool) -> Option<usize> {
    if pattern.is_empty() {
        return Some(start);
    }
    if pattern.len() > lines.len() {
        return None;
    }

    let search_start = if eof && lines.len() >= pattern.len() {
        lines.len() - pattern.len()
    } else {
        start
    };

    for i in search_start..=lines.len().saturating_sub(pattern.len()) {
        if lines[i..i + pattern.len()] == *pattern {
            return Some(i);
        }
    }
    for i in search_start..=lines.len().saturating_sub(pattern.len()) {
        if pattern
            .iter()
            .enumerate()
            .all(|(offset, pat)| lines[i + offset].trim_end() == pat.trim_end())
        {
            return Some(i);
        }
    }
    for i in search_start..=lines.len().saturating_sub(pattern.len()) {
        if pattern
            .iter()
            .enumerate()
            .all(|(offset, pat)| lines[i + offset].trim() == pat.trim())
        {
            return Some(i);
        }
    }

    (search_start..=lines.len().saturating_sub(pattern.len())).find(|&i| {
        pattern
            .iter()
            .enumerate()
            .all(|(offset, pat)| normalize_unicode(&lines[i + offset]) == normalize_unicode(pat))
    })
}

/// The file `content` with every hunk applied.
///
/// The result always ends with a newline: a file the model patched is a file
/// pebble wrote, and every line of it is a whole line.
fn apply_hunks(path: &str, content: &str, hunks: &[Hunk]) -> Result<String, ToolError> {
    let mut original_lines: Vec<String> = content.split('\n').map(String::from).collect();
    if original_lines.last().is_some_and(String::is_empty) {
        original_lines.pop();
    }

    let replacements = compute_replacements(&original_lines, path, hunks)?;
    let mut new_lines = apply_replacements(original_lines, &replacements);
    if !new_lines.last().is_some_and(String::is_empty) {
        new_lines.push(String::new());
    }
    Ok(new_lines.join("\n"))
}

/// Where each hunk lands, as `(start, lines replaced, lines written)`.
fn compute_replacements(
    original_lines: &[String],
    path: &str,
    hunks: &[Hunk],
) -> Result<Vec<(usize, usize, Vec<String>)>, ToolError> {
    let mut replacements = Vec::new();
    let mut line_index = 0;

    for hunk in hunks {
        if !hunk.context_line.is_empty() {
            if let Some(index) = seek_sequence(
                original_lines,
                slice::from_ref(&hunk.context_line),
                line_index,
                false,
            ) {
                line_index = index + 1;
            } else {
                return Err(ToolError::invalid_arguments(format!(
                    "Failed to find context '{}' in {path}",
                    hunk.context_line
                )));
            }
        }

        let mut old_lines = Vec::new();
        let mut new_lines = Vec::new();
        for change in &hunk.changes {
            match change {
                Change::Remove(line) => old_lines.push(line.clone()),
                Change::Add(line) => new_lines.push(line.clone()),
                Change::Context(line) => {
                    old_lines.push(line.clone());
                    new_lines.push(line.clone());
                }
            }
        }

        if old_lines.is_empty() {
            let insertion_index = original_lines.len();
            replacements.push((insertion_index, 0, new_lines));
            continue;
        }

        let mut pattern: &[String] = &old_lines;
        let mut new_slice: &[String] = &new_lines;
        let mut found = seek_sequence(original_lines, pattern, line_index, hunk.end_of_file);
        if found.is_none() && pattern.last().is_some_and(String::is_empty) {
            pattern = &pattern[..pattern.len() - 1];
            if new_slice.last().is_some_and(String::is_empty) {
                new_slice = &new_slice[..new_slice.len() - 1];
            }
            found = seek_sequence(original_lines, pattern, line_index, hunk.end_of_file);
        }

        if let Some(start_index) = found {
            replacements.push((start_index, pattern.len(), new_slice.to_vec()));
            line_index = start_index + pattern.len();
        } else {
            return Err(ToolError::invalid_arguments(format!(
                "Failed to find expected lines in {path}:\n{}",
                old_lines.join("\n")
            )));
        }
    }

    replacements.sort_by_key(|(start_index, _, _)| *start_index);
    Ok(replacements)
}

/// Every replacement written into `lines`, last one first so the earlier
/// indices stay valid.
fn apply_replacements(
    mut lines: Vec<String>,
    replacements: &[(usize, usize, Vec<String>)],
) -> Vec<String> {
    for (start_index, old_len, new_segment) in replacements.iter().rev() {
        for _ in 0..*old_len {
            if *start_index < lines.len() {
                lines.remove(*start_index);
            }
        }
        for (offset, new_line) in new_segment.iter().enumerate() {
            lines.insert(*start_index + offset, new_line.clone());
        }
    }
    lines
}

/// What the model is told a successful patch did.
fn format_summary(added: &[String], modified: &[String], deleted: &[String]) -> String {
    let mut output = String::from("Success. Updated the following files:\n");
    for path in added {
        let _ = writeln!(output, "A {path}");
    }
    for path in modified {
        let _ = writeln!(output, "M {path}");
    }
    for path in deleted {
        let _ = writeln!(output, "D {path}");
    }
    output
}

/// Applies a patch in the Codex `apply_patch` format.
///
/// A custom tool rather than a function tool: the model sends the patch text
/// itself, and the grammar beside this module is what a provider that can
/// constrain output is given.
#[must_use]
pub fn make_apply_patch_tool() -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::custom(
            NativeTool::ApplyPatch.canonical_name(),
            "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap \
             the patch in JSON.",
            serde_json::json!({
                "type": "grammar",
                "syntax": "lark",
                "definition": apply_patch_lark_grammar_definition(),
            }),
        ),
        Arc::new(|args, ctx| {
            Box::pin(async move {
                let patch_text = args.as_str().ok_or_else(|| {
                    ToolError::invalid_arguments("apply_patch expects raw patch text")
                })?;

                let ops = parse_apply_patch(patch_text)?;
                apply_patch_operations(&ops, ctx.env.as_ref()).await
            })
        }),
    )
    .with_source(ToolSource::Native)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lithos_llm::types::ToolDefinitionKind;
    use serde_json::json;

    use super::*;
    use crate::environment::LocalEnvironment;
    use crate::test_support::MutableMockEnvironment;
    use crate::tools::testing::{TempDir, context_for};
    use crate::types::ToolErrorKind;

    fn environment(files: &[(&str, &str)]) -> Arc<MutableMockEnvironment> {
        Arc::new(MutableMockEnvironment::new(
            files
                .iter()
                .map(|(path, content)| ((*path).to_owned(), (*content).to_owned()))
                .collect::<HashMap<_, _>>(),
        ))
    }

    async fn read(env: &MutableMockEnvironment, path: &str) -> String {
        env.read_file_text(path).await.expect("the file is there")
    }

    // --- parsing ---

    #[test]
    fn parse_apply_patch_add_file() {
        let patch = "\
*** Begin Patch
*** Add File: src/new_file.rs
+fn main() {
+    println!(\"hello\");
+}
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], PatchOperation::Add {
            path:    "src/new_file.rs".into(),
            content: "fn main() {\n    println!(\"hello\");\n}\n".into(),
        });
    }

    #[test]
    fn parse_apply_patch_delete_file() {
        let patch = "\
*** Begin Patch
*** Delete File: src/old_file.rs
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], PatchOperation::Delete {
            path: "src/old_file.rs".into(),
        });
    }

    #[test]
    fn parse_apply_patch_update_file() {
        let patch = "\
*** Begin Patch
*** Update File: src/lib.rs
@@ fn hello() @@
-    println!(\"old\");
+    println!(\"new\");
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOperation::Update {
                path,
                new_path,
                hunks,
            } => {
                assert_eq!(path, "src/lib.rs");
                assert_eq!(*new_path, None);
                assert_eq!(hunks.len(), 1);
                assert_eq!(hunks[0].context_line, "fn hello()");
                assert!(!hunks[0].end_of_file);
                assert_eq!(hunks[0].changes.len(), 2);
                assert_eq!(
                    hunks[0].changes[0],
                    Change::Remove("    println!(\"old\");".into())
                );
                assert_eq!(
                    hunks[0].changes[1],
                    Change::Add("    println!(\"new\");".into())
                );
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[test]
    fn parse_apply_patch_multi_operation() {
        let patch = "\
*** Begin Patch
*** Add File: src/a.rs
+// file a
*** Delete File: src/b.rs
*** Update File: src/c.rs
@@ fn main() @@
-    old_call();
+    new_call();
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 3);
        assert!(matches!(&ops[0], PatchOperation::Add { .. }));
        assert!(matches!(&ops[1], PatchOperation::Delete { .. }));
        assert!(matches!(&ops[2], PatchOperation::Update { .. }));
    }

    #[test]
    fn parse_apply_patch_bare_at_at_hunk() {
        let patch = "\
*** Begin Patch
*** Update File: src/game.py
@@
-from src.cards import Suit
+from src.cards import Card, Suit
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOperation::Update { path, hunks, .. } => {
                assert_eq!(path, "src/game.py");
                assert_eq!(hunks.len(), 1);
                assert_eq!(hunks[0].context_line, "");
                assert_eq!(hunks[0].changes.len(), 2);
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[test]
    fn parse_apply_patch_multiple_bare_at_at_hunks() {
        let patch = "\
*** Begin Patch
*** Update File: src/game.py
@@
-from src.cards import Suit
+from src.cards import Card, Suit
@@
-    stock: list = field(default_factory=list)
+    stock: list[Card] = field(default_factory=list)
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks.len(), 2);
                assert_eq!(hunks[0].context_line, "");
                assert_eq!(hunks[1].context_line, "");
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_patch_bare_at_at_update() {
        let env = environment(&[(
            "src/game.py",
            "from src.cards import Suit\nfrom src.piles import Pile\n\nclass GameState:\n    \
             stock: list = field(default_factory=list)\n    waste: list = \
             field(default_factory=list)",
        )]);

        let ops = vec![PatchOperation::Update {
            path:     "src/game.py".into(),
            new_path: None,
            hunks:    vec![
                Hunk {
                    context_line: String::new(),
                    end_of_file:  false,
                    changes:      vec![
                        Change::Remove("from src.cards import Suit".into()),
                        Change::Add("from src.cards import Card, Suit".into()),
                    ],
                },
                Hunk {
                    context_line: String::new(),
                    end_of_file:  false,
                    changes:      vec![
                        Change::Remove("    stock: list = field(default_factory=list)".into()),
                        Change::Remove("    waste: list = field(default_factory=list)".into()),
                        Change::Add("    stock: list[Card] = field(default_factory=list)".into()),
                        Change::Add("    waste: list[Card] = field(default_factory=list)".into()),
                    ],
                },
            ],
        }];

        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");
        assert!(result.contains("M src/game.py"));

        let content = read(&env, "src/game.py").await;
        assert!(content.contains("from src.cards import Card, Suit"));
        assert!(!content.contains("from src.cards import Suit\n"));
        assert!(content.contains("stock: list[Card]"));
        assert!(content.contains("waste: list[Card]"));
        assert!(content.contains("from src.piles import Pile"));
    }

    #[test]
    fn parse_apply_patch_mixed_bare_and_contextual_hunks() {
        let patch = "\
*** Begin Patch
*** Update File: src/lib.rs
@@ fn setup() @@
-    old_setup();
+    new_setup();
@@
-    old_teardown();
+    new_teardown();
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks.len(), 2);
                assert_eq!(hunks[0].context_line, "fn setup()");
                assert_eq!(hunks[1].context_line, "");
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[test]
    fn parse_apply_patch_bare_at_at_with_context_lines() {
        let patch = "\
*** Begin Patch
*** Update File: src/lib.rs
@@
 fn unchanged() {
-    old_line();
+    new_line();
 }
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks.len(), 1);
                assert_eq!(hunks[0].context_line, "");
                assert_eq!(hunks[0].changes.len(), 4);
                assert_eq!(
                    hunks[0].changes[0],
                    Change::Context("fn unchanged() {".into())
                );
                assert_eq!(
                    hunks[0].changes[1],
                    Change::Remove("    old_line();".into())
                );
                assert_eq!(hunks[0].changes[2], Change::Add("    new_line();".into()));
                assert_eq!(hunks[0].changes[3], Change::Context("}".into()));
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[test]
    fn parse_apply_patch_bare_at_at_add_only_appends_to_file() {
        let patch = "\
*** Begin Patch
*** Update File: src/lib.rs
@@
+new_line();
*** End Patch";

        // Parsing succeeds — the hunk is structurally valid.
        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks[0].context_line, "");
                assert_eq!(hunks[0].changes.len(), 1);
                assert_eq!(hunks[0].changes[0], Change::Add("new_line();".into()));

                let result =
                    apply_hunks("src/lib.rs", "fn main() {}\n", hunks).expect("the hunk applies");
                assert_eq!(result, "fn main() {}\nnew_line();\n");
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_patch_bare_at_at_with_context_lines() {
        let env = environment(&[("src/lib.rs", "fn unchanged() {\n    old_line();\n}")]);

        let ops = vec![PatchOperation::Update {
            path:     "src/lib.rs".into(),
            new_path: None,
            hunks:    vec![Hunk {
                context_line: String::new(),
                end_of_file:  false,
                changes:      vec![
                    Change::Context("fn unchanged() {".into()),
                    Change::Remove("    old_line();".into()),
                    Change::Add("    new_line();".into()),
                    Change::Context("}".into()),
                ],
            }],
        }];

        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");
        assert!(result.contains("M src/lib.rs"));
        assert_eq!(
            read(&env, "src/lib.rs").await,
            "fn unchanged() {\n    new_line();\n}\n"
        );
    }

    #[tokio::test]
    async fn apply_patch_mixed_bare_and_contextual_hunks() {
        let env = environment(&[(
            "src/lib.rs",
            "import foo\nimport bar\n\ndef setup():\n    old_setup()\n\ndef teardown():\n    \
             old_teardown()\n",
        )]);

        let ops = vec![PatchOperation::Update {
            path:     "src/lib.rs".into(),
            new_path: None,
            hunks:    vec![
                Hunk {
                    context_line: "def setup():".into(),
                    end_of_file:  false,
                    changes:      vec![
                        Change::Remove("    old_setup()".into()),
                        Change::Add("    new_setup()".into()),
                    ],
                },
                Hunk {
                    context_line: String::new(),
                    end_of_file:  false,
                    changes:      vec![
                        Change::Remove("    old_teardown()".into()),
                        Change::Add("    new_teardown()".into()),
                    ],
                },
            ],
        }];

        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");
        assert!(result.contains("M src/lib.rs"));

        let content = read(&env, "src/lib.rs").await;
        assert!(content.contains("new_setup()"));
        assert!(content.contains("new_teardown()"));
        assert!(!content.contains("old_setup()"));
        assert!(!content.contains("old_teardown()"));
    }

    #[tokio::test]
    async fn apply_patch_add_file() {
        let env = environment(&[]);
        let ops = vec![PatchOperation::Add {
            path:    "src/new.rs".into(),
            content: "fn new() {}".into(),
        }];

        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");
        assert!(result.contains("A src/new.rs"));
        assert_eq!(read(&env, "src/new.rs").await, "fn new() {}");
    }

    #[tokio::test]
    async fn apply_patch_update_file() {
        let env = environment(&[("src/lib.rs", "fn hello() {\n    println!(\"old\");\n}")]);

        let ops = vec![PatchOperation::Update {
            path:     "src/lib.rs".into(),
            new_path: None,
            hunks:    vec![Hunk {
                context_line: "fn hello() {".into(),
                end_of_file:  false,
                changes:      vec![
                    Change::Remove("    println!(\"old\");".into()),
                    Change::Add("    println!(\"new\");".into()),
                ],
            }],
        }];

        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");
        assert!(result.contains("M src/lib.rs"));

        let content = read(&env, "src/lib.rs").await;
        assert!(content.contains("println!(\"new\")"));
        assert!(!content.contains("println!(\"old\")"));
    }

    /// The environment's own file reader numbers lines; a patch has to match
    /// what the file holds, so the update path reads raw text.
    #[tokio::test]
    async fn apply_patch_updates_raw_local_file_without_line_number_prefixes() {
        let directory = TempDir::new("apply-patch");
        directory.write("src/lib.rs", "fn hello() {\n    println!(\"old\");\n}\n");
        let env = LocalEnvironment::new(directory.path());
        let patch = "\
*** Begin Patch
*** Update File: src/lib.rs
@@
-    println!(\"old\");
+    println!(\"new\");
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let result = apply_patch_operations(&ops, &env)
            .await
            .expect("the patch applies");

        assert_eq!(
            result,
            "Success. Updated the following files:\nM src/lib.rs\n"
        );
        assert_eq!(
            directory.read("src/lib.rs"),
            "fn hello() {\n    println!(\"new\");\n}\n"
        );
    }

    /// `build` is a regular file, so nothing can be created under it. The
    /// model reads the OS's reason, not only that the write failed.
    #[tokio::test]
    async fn adding_a_file_under_a_regular_file_reports_the_os_cause() {
        let directory = TempDir::new("apply-patch");
        directory.write("build", "not a directory");
        let env = LocalEnvironment::new(directory.path());
        let patch = "\
*** Begin Patch
*** Add File: build/out.txt
+hello
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let error = apply_patch_operations(&ops, &env)
            .await
            .expect_err("a regular file cannot hold a file");

        let expected_prefix = format!(
            "Failed to write file build/out.txt: Failed to create parent directories for {}\n  caused \
             by: ",
            directory.join("build").display()
        );
        assert!(
            error.message().starts_with(&expected_prefix),
            "{}",
            error.message()
        );
        let cause = &error.message()[expected_prefix.len()..];
        assert!(
            cause.contains("Not a directory") || cause.contains("File exists"),
            "{cause}"
        );
        assert!(cause.contains("os error"), "{cause}");
        assert_eq!(error.kind(), ToolErrorKind::Execution);
    }

    #[test]
    fn apply_patch_tool_definition_is_custom_freeform() {
        let tool = make_apply_patch_tool();

        assert_eq!(tool.definition.name, "apply_patch");
        assert!(tool.definition.is_custom());
        let ToolDefinitionKind::Custom { format } = &tool.definition.kind else {
            panic!("apply_patch is advertised as a custom tool");
        };
        assert_eq!(format.get("type"), Some(&json!("grammar")));
        assert_eq!(format.get("syntax"), Some(&json!("lark")));
    }

    /// The attribution belongs to whoever reads the file, not to the provider
    /// compiling the grammar.
    #[test]
    fn the_grammar_the_model_is_given_carries_no_comments() {
        let grammar = apply_patch_lark_grammar_definition();

        assert!(!grammar.contains("//"));
        assert!(grammar.starts_with("start: begin_patch hunk+ end_patch"));
        assert!(grammar.contains("%import common.LF"));
        assert!(APPLY_PATCH_LARK_GRAMMAR.contains("SPDX-License-Identifier: Apache-2.0"));
    }

    #[tokio::test]
    async fn apply_patch_tool_executor_accepts_raw_patch_string() {
        let env = environment(&[]);
        let tool = make_apply_patch_tool();
        let patch = "\
*** Begin Patch
*** Add File: hello.txt
+hello
*** End Patch
";

        let output = (tool.executor)(json!(patch), context_for(Arc::clone(&env)))
            .await
            .expect("a raw custom patch applies");

        assert_eq!(
            output,
            "Success. Updated the following files:\nA hello.txt\n"
        );
    }

    #[tokio::test]
    async fn apply_patch_add_overwrites_existing_file_with_codex_summary() {
        let env = environment(&[("duplicate.txt", "old content\n")]);
        let patch = "\
*** Begin Patch
*** Add File: duplicate.txt
+new content
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        assert_eq!(
            result,
            "Success. Updated the following files:\nA duplicate.txt\n"
        );
        assert_eq!(read(&env, "duplicate.txt").await, "new content\n");
    }

    #[test]
    fn parse_update_file_hunk_rejects_empty_update() {
        let patch = "\
*** Begin Patch
*** Update File: empty.txt
*** End Patch";

        let error = parse_apply_patch(patch).expect_err("an empty update hunk is rejected");

        assert!(
            error
                .message()
                .contains("Update file hunk for path 'empty.txt' is empty")
        );
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    #[tokio::test]
    async fn pure_addition_update_hunk_appends_before_final_newline() {
        let env = environment(&[("insert_only.txt", "alpha\nomega\n")]);
        let patch = "\
*** Begin Patch
*** Update File: insert_only.txt
@@
+inserted
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        assert_eq!(
            result,
            "Success. Updated the following files:\nM insert_only.txt\n"
        );
        assert_eq!(
            read(&env, "insert_only.txt").await,
            "alpha\nomega\ninserted\n"
        );
    }

    #[tokio::test]
    async fn pure_addition_update_hunk_uses_raw_local_file_text() {
        let directory = TempDir::new("apply-patch");
        directory.write("insert_only.txt", "alpha\nomega\n");
        let env = LocalEnvironment::new(directory.path());
        let patch = "\
*** Begin Patch
*** Update File: insert_only.txt
@@
+inserted
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let result = apply_patch_operations(&ops, &env)
            .await
            .expect("the patch applies");

        assert_eq!(
            result,
            "Success. Updated the following files:\nM insert_only.txt\n"
        );
        assert_eq!(
            directory.read("insert_only.txt"),
            "alpha\nomega\ninserted\n"
        );
    }

    #[tokio::test]
    async fn update_normalizes_missing_trailing_newline() {
        let env = environment(&[("no_newline.txt", "no newline at end")]);
        let patch = "\
*** Begin Patch
*** Update File: no_newline.txt
@@
-no newline at end
+has newline now
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        assert_eq!(read(&env, "no_newline.txt").await, "has newline now\n");
    }

    #[test]
    fn parse_rejects_text_before_patch_envelope() {
        let patch = "\
please apply this
*** Begin Patch
*** Add File: hello.txt
+hello
*** End Patch";

        let error =
            parse_apply_patch(patch).expect_err("the envelope must start on the first line");

        assert!(
            error
                .message()
                .contains("The first line of the patch must be '*** Begin Patch'")
        );
    }

    #[tokio::test]
    async fn apply_patch_error_reports_failed_context() {
        let env = environment(&[("src/game.py", "def real_fn():\n    pass")]);

        let ops = vec![PatchOperation::Update {
            path:     "src/game.py".into(),
            new_path: None,
            hunks:    vec![Hunk {
                context_line: "def nonexistent():".into(),
                end_of_file:  false,
                changes:      vec![
                    Change::Remove("    old_body()".into()),
                    Change::Add("    new_body()".into()),
                ],
            }],
        }];

        let error = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect_err("the anchor is not in the file");
        assert_eq!(
            error.message(),
            "Failed to find context 'def nonexistent():' in src/game.py"
        );
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    #[tokio::test]
    async fn update_missing_target_file_rejected() {
        let env = environment(&[]);
        let patch = "\
*** Begin Patch
*** Update File: missing.txt
@@
-old
+new
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let error = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect_err("there is no file to update");

        assert!(
            error
                .message()
                .contains("Failed to read file to update missing.txt")
        );
        assert_eq!(error.kind(), ToolErrorKind::Execution);
    }

    #[tokio::test]
    async fn delete_missing_target_file_rejected() {
        let env = environment(&[]);
        let patch = "\
*** Begin Patch
*** Delete File: missing.txt
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let error = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect_err("there is no file to delete");

        assert_eq!(
            error.message(),
            "Failed to delete file missing.txt: file does not exist"
        );
    }

    // --- forward-order hunk application ---

    #[test]
    fn apply_hunks_bare_at_at_searches_forward_from_previous_hunk() {
        let content = "def foo():\n    pass\n\ndef bar():\n    pass";
        let hunks = vec![
            Hunk {
                context_line: String::new(),
                end_of_file:  false,
                changes:      vec![
                    Change::Remove("    pass".into()),
                    Change::Add("    return 1".into()),
                ],
            },
            Hunk {
                context_line: String::new(),
                end_of_file:  false,
                changes:      vec![
                    Change::Remove("    pass".into()),
                    Change::Add("    return 2".into()),
                ],
            },
        ];
        let result = apply_hunks("example.py", content, &hunks).expect("both hunks apply");
        assert!(result.contains("return 1"));
        assert!(result.contains("return 2"));
        assert!(!result.contains("    pass"));
    }

    // --- context without a trailing @@ ---

    #[test]
    fn parse_apply_patch_context_without_trailing_markers() {
        let patch = "\
*** Begin Patch
*** Update File: src/hello.py
@@ def hello():
-    print(\"old\")
+    print(\"new\")
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks[0].context_line, "def hello():");
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    // --- stacked @@ anchors ---

    #[test]
    fn parse_apply_patch_stacked_context_uses_last() {
        let patch = "\
*** Begin Patch
*** Update File: src/foo.py
@@ class Foo:
@@   def bar(self):
-        pass
+        return 42
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks.len(), 1);
                assert_eq!(hunks[0].context_line, "def bar(self):");
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    // --- *** End of File ---

    #[test]
    fn parse_apply_patch_end_of_file_marker() {
        let patch = "\
*** Begin Patch
*** Update File: src/lib.py
@@
-    pass
+    return 1
*** End of File
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks.len(), 1);
                assert!(hunks[0].end_of_file);
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[test]
    fn apply_hunks_end_of_file_searches_backward() {
        // Two functions with an identical "pass" line: End of File matches the
        // last one.
        let content = "def foo():\n    pass\n\ndef bar():\n    pass";
        let hunks = vec![Hunk {
            context_line: String::new(),
            end_of_file:  true,
            changes:      vec![
                Change::Remove("    pass".into()),
                Change::Add("    return 99".into()),
            ],
        }];
        let result = apply_hunks("example.py", content, &hunks).expect("the hunk applies");
        assert_eq!(
            result,
            "def foo():\n    pass\n\ndef bar():\n    return 99\n"
        );
    }

    // --- *** Move to: ---

    #[test]
    fn parse_apply_patch_move_to() {
        let patch = "\
*** Begin Patch
*** Update File: src/old.py
*** Move to: src/new.py
@@ def hello():
-    pass
+    return 1
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        match &ops[0] {
            PatchOperation::Update {
                path,
                new_path,
                hunks,
            } => {
                assert_eq!(path, "src/old.py");
                assert_eq!(*new_path, Some("src/new.py".to_owned()));
                assert_eq!(hunks.len(), 1);
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_patch_move_to_renames_file() {
        let env = environment(&[("src/old.py", "def hello():\n    pass")]);

        let ops = vec![PatchOperation::Update {
            path:     "src/old.py".into(),
            new_path: Some("src/new.py".into()),
            hunks:    vec![Hunk {
                context_line: "def hello():".into(),
                end_of_file:  false,
                changes:      vec![
                    Change::Remove("    pass".into()),
                    Change::Add("    return 1".into()),
                ],
            }],
        }];

        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");
        assert!(result.contains("M src/new.py"));
        assert_eq!(
            read(&env, "src/new.py").await,
            "def hello():\n    return 1\n"
        );
        assert!(env.read_file_text("src/old.py").await.is_err());
    }

    // --- fuzzy matching ---

    #[test]
    fn apply_hunks_prefers_exact_match_over_trimmed() {
        // Line 0 has leading spaces; line 1 is the exact match.
        let content = "  indented\nindented";
        let hunks = vec![Hunk {
            context_line: "indented".into(),
            end_of_file:  false,
            changes:      vec![Change::Add("extra".into())],
        }];
        let result = apply_hunks("example.txt", content, &hunks).expect("the hunk applies");
        assert_eq!(result, "  indented\nindented\nextra\n");
    }

    #[test]
    fn apply_hunks_fuzzy_unicode_normalization() {
        let content = "print(\u{201C}hello\u{201D})";
        let hunks = vec![Hunk {
            context_line: "print(\"hello\")".into(),
            end_of_file:  false,
            changes:      vec![Change::Add("print(\"world\")".into())],
        }];
        let result = apply_hunks("example.py", content, &hunks).expect("the hunk applies");
        assert!(result.contains("print(\u{201C}hello\u{201D})"));
        assert!(result.contains("print(\"world\")"));
    }

    // --- heredoc stripping ---

    #[test]
    fn parse_apply_patch_strips_heredoc_wrapper() {
        let patch = "\
<<'EOF'
*** Begin Patch
*** Update File: src/lib.rs
@@ fn hello():
-    pass
+    return 1
*** End Patch
EOF";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOperation::Update { hunks, .. } => {
                assert_eq!(hunks[0].context_line, "fn hello():");
            }
            other => panic!("expected an update operation, got {other:?}"),
        }
    }

    #[test]
    fn parse_apply_patch_strips_heredoc_unquoted() {
        let patch = "\
<<EOF
*** Begin Patch
*** Add File: src/a.rs
+// hello
*** End Patch
EOF";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], PatchOperation::Add { .. }));
    }

    #[test]
    fn parse_apply_patch_strips_heredoc_double_quoted() {
        let patch = "\
<<\"EOF\"
*** Begin Patch
*** Delete File: src/old.rs
*** End Patch
EOF";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], PatchOperation::Delete { .. }));
    }

    // --- end to end: raw patch text, parsed, applied, read back ---

    #[tokio::test]
    async fn e2e_canonical_format_multi_hunk_update() {
        let env = environment(&[(
            "src/game.py",
            "\
from dataclasses import dataclass, field
from src.cards import Suit
import random

@dataclass
class GameState:
    stock: list = field(default_factory=list)
    waste: list = field(default_factory=list)
    tableau: list = field(default_factory=list)

    def deal(self):
        random.shuffle(self.stock)
        for i in range(7):
            self.tableau.append(self.stock.pop())

    def draw(self):
        if self.stock:
            self.waste.append(self.stock.pop())
",
        )]);

        let patch = "\
*** Begin Patch
*** Update File: src/game.py
@@ from dataclasses import dataclass, field
-from src.cards import Suit
+from src.cards import Card, Suit
@@ class GameState:
-    stock: list = field(default_factory=list)
-    waste: list = field(default_factory=list)
-    tableau: list = field(default_factory=list)
+    stock: list[Card] = field(default_factory=list)
+    waste: list[Card] = field(default_factory=list)
+    tableau: list[Card] = field(default_factory=list)
@@ def draw(self):
-        if self.stock:
-            self.waste.append(self.stock.pop())
+        card = self.stock.pop() if self.stock else None
+        if card:
+            self.waste.append(card)
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        let content = read(&env, "src/game.py").await;
        assert!(content.contains("from src.cards import Card, Suit"));
        assert!(content.contains("stock: list[Card]"));
        assert!(content.contains("waste: list[Card]"));
        assert!(content.contains("tableau: list[Card]"));
        assert!(content.contains("card = self.stock.pop()"));
        assert!(content.contains("self.waste.append(card)"));
        // Untouched lines are preserved.
        assert!(content.contains("from dataclasses import dataclass, field"));
        assert!(content.contains("import random"));
        assert!(content.contains("def deal(self):"));
        assert!(content.contains("random.shuffle(self.stock)"));
    }

    #[tokio::test]
    async fn e2e_multi_operation_add_update_delete() {
        let env = environment(&[
            ("src/old_util.py", "def old_helper():\n    pass\n"),
            (
                "src/main.py",
                "\
from old_util import old_helper

def main():
    old_helper()
    print(\"done\")
",
            ),
        ]);

        let patch = "\
*** Begin Patch
*** Add File: src/new_util.py
+def new_helper():
+    return 42
*** Delete File: src/old_util.py
*** Update File: src/main.py
@@
-from old_util import old_helper
+from new_util import new_helper
@@ def main():
-    old_helper()
+    result = new_helper()
+    print(f\"result: {result}\")
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        assert!(result.contains("A src/new_util.py"));
        assert!(result.contains("D src/old_util.py"));
        assert!(result.contains("M src/main.py"));

        assert_eq!(
            read(&env, "src/new_util.py").await,
            "def new_helper():\n    return 42\n"
        );
        assert!(env.read_file_text("src/old_util.py").await.is_err());

        let main = read(&env, "src/main.py").await;
        assert!(main.contains("from new_util import new_helper"));
        assert!(main.contains("result = new_helper()"));
        assert!(main.contains("print(\"done\")"));
    }

    #[tokio::test]
    async fn e2e_heredoc_stacked_context_end_of_file_and_move() {
        let env = environment(&[(
            "src/models/user.py",
            "\
class User:
    def __init__(self, name):
        self.name = name
        self.active = True

    def greet(self):
        return f\"Hello, {self.name}\"

    def deactivate(self):
        self.active = False
",
        )]);

        // A heredoc-wrapped patch with stacked @@, End of File, and Move to.
        let patch = "\
<<'EOF'
*** Begin Patch
*** Update File: src/models/user.py
*** Move to: src/models/account.py
@@ class User:
@@     def __init__(self, name):
-        self.name = name
-        self.active = True
+        self.name = name
+        self.email = None
+        self.active = True
@@ class User:
@@     def deactivate(self):
-        self.active = False
+        self.active = False
+        self.email = None
*** End of File
*** End Patch
EOF";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        let result = apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        assert!(result.contains("M src/models/account.py"));
        assert!(env.read_file_text("src/models/user.py").await.is_err());

        let content = read(&env, "src/models/account.py").await;
        assert!(content.contains("self.email = None"));
        assert!(content.contains("self.active = True"));
        assert!(content.contains("def greet(self):"));
        // The End of File hunk matched the last deactivate method.
        let deactivate = content
            .rfind("def deactivate")
            .expect("the method is still there");
        assert!(content[deactivate..].contains("self.email = None"));
    }

    #[tokio::test]
    async fn e2e_forward_cursor_with_duplicate_patterns() {
        // Three identical "pass" lines, three anchored hunks: the forward
        // cursor takes them in order.
        let env = environment(&[(
            "src/stubs.py",
            "\
def alpha():
    pass

def beta():
    pass

def gamma():
    pass
",
        )]);

        let patch = "\
*** Begin Patch
*** Update File: src/stubs.py
@@ def alpha():
-    pass
+    return \"a\"
@@ def beta():
-    pass
+    return \"b\"
@@ def gamma():
-    pass
+    return \"c\"
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        let content = read(&env, "src/stubs.py").await;
        assert!(content.contains("return \"a\""));
        assert!(content.contains("return \"b\""));
        assert!(content.contains("return \"c\""));
        assert!(!content.contains("    pass"));
    }

    #[tokio::test]
    async fn e2e_fuzzy_matching_with_trailing_whitespace() {
        // The file has trailing whitespace on its lines; the patch does not.
        let env = environment(&[(
            "src/lib.rs",
            "fn main() {  \n    println!(\"hello\");  \n}\n",
        )]);

        let patch = "\
*** Begin Patch
*** Update File: src/lib.rs
@@ fn main() {
-    println!(\"hello\");
+    println!(\"world\");
*** End Patch";

        let ops = parse_apply_patch(patch).expect("the patch parses");
        apply_patch_operations(&ops, env.as_ref())
            .await
            .expect("the patch applies");

        let content = read(&env, "src/lib.rs").await;
        assert!(content.contains("println!(\"world\")"));
        assert!(!content.contains("println!(\"hello\")"));
    }
}
