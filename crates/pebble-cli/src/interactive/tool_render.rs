//! Tool-specific headings, retained previews, and edit diffs.

use std::path::Path;
use std::time::Duration;

use serde_json::Value;
use similar::TextDiff;

use super::text;
use super::transcript::{Output, ToolRecord};

fn field<'a>(value: &'a Value, names: &[&str]) -> Option<&'a str> {
    names
        .iter()
        .find_map(|name| value.get(name).and_then(Value::as_str))
}

pub(super) fn heading(name: &str, arguments: &str) -> String {
    let value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let path = field(&value, &["file_path", "path", "file", "filename"]);
    let command = field(&value, &["command", "cmd", "script"]);
    let title = if let Some(command) = command {
        format!("$ {command}")
    } else if let Some(path) = path {
        format!("{name} {path}")
    } else if name == "apply_patch" {
        "apply_patch · edit files".into()
    } else {
        format!("{name} {arguments}")
    };
    text::truncate(&title, 180)
}

pub(super) fn result(tool: &ToolRecord, limit: Option<usize>, omitted: usize) -> Vec<Output> {
    let arguments: Value = serde_json::from_str(&tool.arguments).unwrap_or(Value::Null);
    let path = field(&arguments, &["file_path", "path", "file", "filename"]).unwrap_or("");
    let language = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    let mut output = vec![Output::Text(format!(
        "  {} · {}",
        heading(&tool.name, &tool.arguments),
        if tool.failed { "failed" } else { "done" }
    ))];
    let mut body = tool.output.clone();
    let mut style = language.to_owned();
    if !tool.failed {
        if tool.name == "apply_patch" {
            body = format!("{}\n{}", tool.arguments, body);
            style = "diff".into();
        } else if let (Some(old), Some(new)) = (
            field(&arguments, &["old_string", "old_text"]),
            field(&arguments, &["new_string", "new_text"]),
        ) {
            // Preview work is bounded independently of the retained event.
            if old.len() + new.len() <= 64 * 1024 {
                let diff = TextDiff::configure()
                    .timeout(Duration::from_millis(20))
                    .diff_lines(old, new);
                body = format!("{}\n{}", diff.unified_diff().header(path, path), body);
                style = "diff".into();
            }
        } else if let Some(content) = field(&arguments, &["content"]).filter(|_| {
            matches!(
                tool.name.to_ascii_lowercase().as_str(),
                "write" | "write_file"
            )
        }) {
            body = format!("{content}\n{body}");
        } else if let Ok(value) = serde_json::from_str::<Value>(&body) {
            if let Some(stdout) = field(&value, &["stdout", "output", "content", "text"]) {
                stdout.clone_into(&mut body);
                if let Some(stderr) = field(&value, &["stderr"]).filter(|value| !value.is_empty()) {
                    body.push('\n');
                    body.push_str(stderr);
                }
            } else {
                body = serde_json::to_string_pretty(&value).unwrap_or(body);
                style = "json".into();
            }
        }
    }
    let body = text::plain(&body);
    let lines: Vec<_> = body.lines().collect();
    let count = limit.unwrap_or(lines.len()).min(lines.len());
    // Full retained details still come from the journal, not just the live tail.
    let preview = lines[..count].join("\n");
    output.push(Output::Code {
        source:   preview,
        language: style,
    });
    if count < lines.len() || omitted > 0 {
        output.push(Output::Text(format!(
            "    … {} more lines · {omitted} bytes not retained · /tools for details",
            lines.len() - count
        )));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edits_show_actual_removed_and_added_lines_and_errors_do_not_claim_changes() {
        let mut tool = ToolRecord {
            id:        "call".into(),
            session:   "root".into(),
            name:      "edit_file".into(),
            arguments: r#"{"file_path":"src/lib.rs","old_string":"old\n","new_string":"new\n"}"#
                .into(),
            output:    "Updated".into(),
            complete:  true,
            failed:    false,
        };
        let output = result(&tool, Some(10), 0);
        assert!(output.iter().any(|item| matches!(item, Output::Code {source, language} if source.contains("-old") && source.contains("+new") && language == "diff")));
        tool.failed = true;
        assert!(
            !result(&tool, Some(10), 0)
                .iter()
                .any(|item| matches!(item, Output::Code {language, ..} if language == "diff"))
        );
        assert_eq!(
            heading("bash", r#"{"command":"cargo test","timeout":30}"#),
            "$ cargo test"
        );
    }
}
