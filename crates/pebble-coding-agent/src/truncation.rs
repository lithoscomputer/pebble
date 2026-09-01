//! Keeping tool output inside its budgets.
//!
//! A tool's output is bounded twice, for two different reasons.
//!
//! First, **retention**: whatever a tool produced is cut down to what a
//! session is willing to carry — one budget in bytes for the text itself and
//! one for its serialized JSON, because JSON escaping can inflate output far
//! past its own length. This is the form events, hooks, and the model all see,
//! and it is where the "output was truncated" notice is written.
//!
//! Then, **per-tool truncation**: the copy that stays in history is cut again
//! to the character and line limits that tool deserves, because a session
//! re-reads history on every turn and a 50,000-character file listing is worth
//! less than the room it takes.
//!
//! Both budgets come from configuration rather than constants here; the
//! defaults this module names are the values a session starts from.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Result as IoResult, Write};

use lithos_llm::estimate::byte_tokens;
use serde::Serialize;

use crate::char_boundary::{ceil_char_boundary, floor_char_boundary};
use crate::event::OutputCaptureStats;
use crate::tool::NativeTool;

/// Bytes of one tool's output a session retains by default.
pub const DEFAULT_TOOL_OUTPUT_RETENTION_BYTES: usize = 1024 * 1024;

/// Bytes one tool's output may occupy once serialized as JSON, by default.
///
/// Half of the 3 MiB body a fabro run event allows, leaving the other half as
/// headroom for the rest of the envelope. Pebble does not impose that envelope
/// itself; the value is the field-tested default an application can change.
pub const DEFAULT_TOOL_OUTPUT_SERIALIZED_BYTES: usize = 1_572_864;

/// How much of one tool's output a session retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputBudgets {
    /// Bytes of text kept, notice included.
    pub retained_bytes:   usize,
    /// Bytes the kept text may occupy once serialized as JSON.
    pub serialized_bytes: usize,
}

impl OutputBudgets {
    /// Budgets with the given byte counts.
    #[must_use]
    pub const fn new(retained_bytes: usize, serialized_bytes: usize) -> Self {
        Self {
            retained_bytes,
            serialized_bytes,
        }
    }
}

impl Default for OutputBudgets {
    fn default() -> Self {
        Self::new(
            DEFAULT_TOOL_OUTPUT_RETENTION_BYTES,
            DEFAULT_TOOL_OUTPUT_SERIALIZED_BYTES,
        )
    }
}

/// Which end of an over-long output survives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TruncationMode {
    /// Keep the start and the end, dropping the middle. The default, because
    /// most tool output explains itself at both ends.
    #[default]
    HeadTail,
    /// Keep only the end, for output whose value is in its last lines.
    Tail,
}

/// Output kept within a byte budget, with what that cost.
#[derive(Debug)]
pub(crate) struct RetainedToolOutput {
    pub(crate) output: String,
    pub(crate) stats:  OutputCaptureStats,
}

/// The model-facing form of a tool's output. Borrows the input when nothing
/// had to be said about truncation.
#[derive(Debug)]
pub(crate) struct PreviewedToolOutput<'a> {
    pub(crate) output: Cow<'a, str>,
    pub(crate) stats:  OutputCaptureStats,
}

/// The boundaries of an equal-sized head and tail fitting `max_bytes`, or
/// `None` when `output` already fits.
fn split_head_tail(output: &str, max_bytes: usize) -> Option<(usize, usize)> {
    if output.len() <= max_bytes {
        return None;
    }

    let head_budget = max_bytes / 2;
    let tail_budget = max_bytes - head_budget;
    let head_end = floor_char_boundary(output, head_budget);
    let tail_start = ceil_char_boundary(output, output.len() - tail_budget);
    Some((head_end, tail_start))
}

/// Keeps an equal-sized prefix and suffix of `output` within `max_bytes`.
///
/// `previously_omitted_bytes` accounts for output an environment already
/// dropped while draining the process, so the counts describe everything the
/// tool produced rather than everything that reached this point.
#[must_use]
pub(crate) fn retain_tool_output(
    output: String,
    max_bytes: usize,
    previously_omitted_bytes: usize,
) -> RetainedToolOutput {
    let observed_bytes = output.len().saturating_add(previously_omitted_bytes);
    let Some((head_end, tail_start)) = split_head_tail(&output, max_bytes) else {
        return RetainedToolOutput {
            stats: OutputCaptureStats {
                observed_bytes,
                retained_bytes: output.len(),
                omitted_bytes: previously_omitted_bytes,
            },
            output,
        };
    };

    let retained_bytes = head_end + (output.len() - tail_start);
    let mut retained = String::with_capacity(retained_bytes);
    retained.push_str(&output[..head_end]);
    retained.push_str(&output[tail_start..]);

    RetainedToolOutput {
        output: retained,
        stats:  OutputCaptureStats {
            observed_bytes,
            retained_bytes,
            omitted_bytes: observed_bytes.saturating_sub(retained_bytes),
        },
    }
}

/// Builds the model-facing preview of a tool's output, fitting the truncation
/// notice inside both budgets.
///
/// The notice itself costs bytes, so the content budget shrinks until the
/// rendered result fits the retained budget *and* its JSON form fits the
/// serialized budget. Output made almost entirely of characters JSON escapes
/// is why the second check exists.
#[must_use]
pub(crate) fn preview_tool_output(
    output: &str,
    budgets: OutputBudgets,
    previously_omitted_bytes: usize,
) -> PreviewedToolOutput<'_> {
    let observed_bytes = output.len().saturating_add(previously_omitted_bytes);
    let mut content_budget = budgets.retained_bytes;
    loop {
        let (head_end, tail_start, stats) =
            if let Some((head_end, tail_start)) = split_head_tail(output, content_budget) {
                let retained_bytes = head_end + (output.len() - tail_start);
                (head_end, tail_start, OutputCaptureStats {
                    observed_bytes,
                    retained_bytes,
                    omitted_bytes: observed_bytes.saturating_sub(retained_bytes),
                })
            } else {
                // The whole output fits. A notice is still rendered when the
                // stream itself dropped bytes; splitting at the midpoint puts
                // that gap where it happened.
                let middle = floor_char_boundary(output, output.len() / 2);
                (middle, middle, OutputCaptureStats {
                    observed_bytes,
                    retained_bytes: output.len(),
                    omitted_bytes: previously_omitted_bytes,
                })
            };

        let rendered: Cow<'_, str> = if stats.omitted_bytes == 0 {
            Cow::Borrowed(output)
        } else {
            Cow::Owned(render_truncated_segments(
                &output[..head_end],
                &output[tail_start..],
                stats,
                None,
            ))
        };
        let serialized_bytes = serialized_json_bytes(rendered.as_ref());
        if rendered.len() <= budgets.retained_bytes && serialized_bytes <= budgets.serialized_bytes
        {
            return PreviewedToolOutput {
                output: rendered,
                stats,
            };
        }

        let Some(reduced_budget) = content_budget.checked_sub(1) else {
            // The content budget is gone and the notice alone still overflows.
            // Cut the notice itself to fit.
            let output = match split_head_tail(&rendered, budgets.retained_bytes) {
                Some((head_end, tail_start)) => {
                    format!("{}{}", &rendered[..head_end], &rendered[tail_start..])
                }
                None => rendered.into_owned(),
            };
            return PreviewedToolOutput {
                output: Cow::Owned(output),
                stats,
            };
        };

        let mut next_budget = reduced_budget;
        if rendered.len() > budgets.retained_bytes {
            let excess = rendered.len() - budgets.retained_bytes;
            next_budget = next_budget.min(content_budget.saturating_sub(excess));
        }
        if serialized_bytes > budgets.serialized_bytes {
            let scaled_budget = (content_budget as u128)
                .saturating_mul(budgets.serialized_bytes as u128)
                .checked_div(serialized_bytes as u128)
                .and_then(|budget| usize::try_from(budget).ok())
                .unwrap_or(0);
            next_budget = next_budget.min(scaled_budget);
        }
        content_budget = next_budget;
    }
}

/// The size of `value` as JSON, counted without building the string.
pub(crate) fn serialized_json_bytes<T: Serialize + ?Sized>(value: &T) -> usize {
    struct CountingWriter(usize);

    impl Write for CountingWriter {
        fn write(&mut self, buffer: &[u8]) -> IoResult<usize> {
            self.0 += buffer.len();
            Ok(buffer.len())
        }

        fn flush(&mut self) -> IoResult<()> {
            Ok(())
        }
    }

    let mut writer = CountingWriter(0);
    serde_json::to_writer(&mut writer, value).expect("tool output always serializes as JSON");
    writer.0
}

/// The notice that replaces omitted output, with the head and tail that
/// survived around it.
fn render_truncated_segments(
    head: &str,
    tail: &str,
    stats: OutputCaptureStats,
    line_count_omitted: Option<usize>,
) -> String {
    let original_tokens = byte_tokens(stats.observed_bytes);
    let omitted_tokens = byte_tokens(stats.omitted_bytes);
    let middle_marker = line_count_omitted.map_or_else(
        || format!("... approximately {omitted_tokens} tokens truncated ..."),
        |lines| {
            format!(
                "... {lines} lines omitted (approximately {omitted_tokens} tokens truncated) ..."
            )
        },
    );
    format!(
        "Warning: truncated output (original token count: {original_tokens})\n... {} bytes \
         omitted ...\n\n{head}\n\n{middle_marker}\n\n{tail}",
        stats.omitted_bytes
    )
}

/// Cuts `output` to `max_chars` bytes, keeping the end and, in
/// [`TruncationMode::HeadTail`], the start as well.
///
/// Output that already fits is returned unchanged; anything else carries the
/// truncation notice.
#[must_use]
pub fn truncate_output(output: &str, max_chars: usize, mode: TruncationMode) -> String {
    let Some((head_end, tail_start)) = split_head_tail(output, max_chars) else {
        return output.to_owned();
    };

    let (head, tail) = match mode {
        TruncationMode::HeadTail => (&output[..head_end], &output[tail_start..]),
        TruncationMode::Tail => {
            let tail_start = ceil_char_boundary(output, output.len() - max_chars);
            ("", &output[tail_start..])
        }
    };
    let retained_bytes = head.len().saturating_add(tail.len());
    render_truncated_segments(
        head,
        tail,
        OutputCaptureStats {
            observed_bytes: output.len(),
            retained_bytes,
            omitted_bytes: output.len().saturating_sub(retained_bytes),
        },
        None,
    )
}

/// Cuts `output` to `max_lines`, keeping half from each end.
///
/// Output that already fits is returned unchanged.
#[must_use]
pub fn truncate_lines(output: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= max_lines {
        return output.to_owned();
    }

    let head_count = max_lines / 2;
    let tail_count = max_lines.saturating_sub(head_count);
    let head = lines[..head_count].join("\n");
    let tail = lines[lines.len() - tail_count..].join("\n");
    let omitted = lines.len() - max_lines;
    let retained_bytes = head.len().saturating_add(tail.len());

    render_truncated_segments(
        &head,
        &tail,
        OutputCaptureStats {
            observed_bytes: output.len(),
            retained_bytes,
            omitted_bytes: output.len().saturating_sub(retained_bytes),
        },
        Some(omitted),
    )
}

/// How much of one tool's output history keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolOutputLimits {
    /// The byte budget, or `None` to keep every character.
    pub max_chars: Option<usize>,
    /// The line budget, or `None` to keep every line.
    pub max_lines: Option<usize>,
    /// Which end survives the byte budget.
    pub mode:      TruncationMode,
}

impl ToolOutputLimits {
    /// Pebble's built-in limits for a tool, addressed by its canonical name.
    ///
    /// The limits themselves live on [`NativeTool`], where the compiler makes
    /// a new built-in tool state its answer. A tool pebble does not know keeps
    /// its whole output.
    #[must_use]
    pub fn defaults_for(canonical_tool_name: &str) -> Self {
        match NativeTool::from_canonical_name(canonical_tool_name) {
            Some(tool) => tool.default_output_limits(),
            None => Self {
                max_chars: None,
                max_lines: None,
                mode:      TruncationMode::HeadTail,
            },
        }
    }

    /// The limits for one tool, preferring an application's overrides.
    ///
    /// An override is looked up under the name the model calls (`tool_name`,
    /// which a profile may have renamed), then under the canonical name, and
    /// falls back to [`defaults_for`](Self::defaults_for). Configuring
    /// `"shell"` therefore also covers a profile that exposes it as `"Bash"`.
    #[must_use]
    pub fn resolve(
        tool_name: &str,
        canonical_tool_name: &str,
        char_overrides: &HashMap<String, usize>,
        line_overrides: &HashMap<String, usize>,
    ) -> Self {
        fn lookup(
            overrides: &HashMap<String, usize>,
            tool_name: &str,
            canonical_tool_name: &str,
        ) -> Option<usize> {
            overrides
                .get(tool_name)
                .or_else(|| overrides.get(canonical_tool_name))
                .copied()
        }

        let defaults = Self::defaults_for(canonical_tool_name);
        Self {
            max_chars: lookup(char_overrides, tool_name, canonical_tool_name)
                .or(defaults.max_chars),
            max_lines: lookup(line_overrides, tool_name, canonical_tool_name)
                .or(defaults.max_lines),
            mode:      defaults.mode,
        }
    }
}

/// Cuts one tool's output to its limits: characters first, then lines.
#[must_use]
pub fn truncate_tool_output(output: &str, limits: ToolOutputLimits) -> String {
    let after_chars = match limits.max_chars {
        Some(limit) => truncate_output(output, limit, limits.mode),
        None => output.to_owned(),
    };

    match limits.max_lines {
        Some(limit) => truncate_lines(&after_chars, limit),
        None => after_chars,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overrides(entries: &[(&str, usize)]) -> HashMap<String, usize> {
        entries
            .iter()
            .map(|(name, limit)| ((*name).to_owned(), *limit))
            .collect()
    }

    #[test]
    fn retained_output_keeps_an_equal_head_and_tail() {
        let retained = retain_tool_output("abcdefghijkl".to_owned(), 8, 0);

        assert_eq!(retained.output, "abcdijkl");
        assert_eq!(retained.stats.observed_bytes, 12);
        assert_eq!(retained.stats.retained_bytes, 8);
        assert_eq!(retained.stats.omitted_bytes, 4);
    }

    #[test]
    fn retained_output_stays_within_budget_at_character_boundaries() {
        let retained = retain_tool_output("aa😀😀zz".to_owned(), 7, 3);

        assert!(retained.output.len() <= 7, "{}", retained.output.len());
        assert!(retained.output.starts_with("aa"));
        assert!(retained.output.ends_with("zz"));
        assert_eq!(retained.stats.observed_bytes, "aa😀😀zz".len() + 3);
        assert_eq!(
            retained.stats.omitted_bytes,
            retained.stats.observed_bytes - retained.output.len()
        );
    }

    #[test]
    fn output_within_budget_is_retained_whole() {
        let retained = retain_tool_output("short".to_owned(), 64, 0);

        assert_eq!(retained.output, "short");
        assert_eq!(retained.stats, OutputCaptureStats::complete(5));
    }

    #[test]
    fn a_preview_carries_its_notice_inside_the_budget() {
        let output = format!("HEAD{}TAIL", "x".repeat(1_000));

        let preview = preview_tool_output(&output, OutputBudgets::new(512, 1_572_864), 0);

        assert!(preview.output.len() <= 512, "{}", preview.output.len());
        assert!(
            preview
                .output
                .starts_with("Warning: truncated output (original token count: 252)"),
            "{}",
            preview.output
        );
        assert!(preview.output.contains(&format!(
            "... {} bytes omitted ...",
            preview.stats.omitted_bytes
        )));
        assert!(preview.output.contains("approximately"));
        assert!(preview.output.contains("tokens truncated"));
        assert!(preview.output.contains("HEAD"));
        assert!(preview.output.ends_with("TAIL"));
    }

    #[test]
    fn a_preview_reports_bytes_the_stream_dropped_before_it() {
        let preview = preview_tool_output("abcdefgh", OutputBudgets::new(512, 1_572_864), 100);

        assert_eq!(preview.stats.observed_bytes, 108);
        assert_eq!(preview.stats.retained_bytes, 8);
        assert_eq!(preview.stats.omitted_bytes, 100);
        assert!(preview.output.contains("... 100 bytes omitted ..."));
        assert!(preview.output.contains("abcd"));
        assert!(preview.output.ends_with("efgh"));
    }

    #[test]
    fn a_preview_of_output_within_budget_borrows_it_unchanged() {
        let preview = preview_tool_output("all good", OutputBudgets::default(), 0);

        assert!(matches!(preview.output, Cow::Borrowed("all good")));
        assert_eq!(preview.stats, OutputCaptureStats::complete(8));
    }

    #[test]
    fn a_preview_bounds_output_that_json_inflates() {
        let budgets = OutputBudgets::default();
        let output = format!(
            "HEAD{}TAIL",
            "\0".repeat(budgets.retained_bytes - "HEADTAIL".len())
        );
        assert_eq!(output.len(), budgets.retained_bytes);
        assert!(serialized_json_bytes(output.as_str()) > budgets.serialized_bytes);

        let preview = preview_tool_output(&output, budgets, 0);
        let serialized_bytes = serialized_json_bytes(preview.output.as_ref());

        assert!(preview.output.len() <= budgets.retained_bytes);
        assert!(
            serialized_bytes <= budgets.serialized_bytes,
            "serialized preview was {serialized_bytes} bytes"
        );
        assert!(preview.output.starts_with("Warning: truncated output"));
        assert!(preview.output.contains("HEAD"));
        assert!(preview.output.ends_with("TAIL"));
        assert_eq!(preview.stats.observed_bytes, budgets.retained_bytes);
        assert!(preview.stats.retained_bytes < budgets.retained_bytes);
        assert_eq!(
            preview.stats.omitted_bytes,
            preview.stats.observed_bytes - preview.stats.retained_bytes
        );
    }

    #[test]
    fn a_preview_with_no_room_for_its_notice_cuts_the_notice() {
        let output = "x".repeat(200);

        let preview = preview_tool_output(&output, OutputBudgets::new(20, 1_572_864), 0);

        assert!(preview.output.len() <= 20, "{}", preview.output);
        // Only the notice's own head and tail survive a budget this small.
        assert!(preview.output.starts_with("Warning:"), "{}", preview.output);
        assert_eq!(preview.stats.observed_bytes, 200);
        assert!(preview.stats.omitted_bytes > 0);
    }

    #[test]
    fn serialized_size_counts_json_escaping() {
        assert_eq!(serialized_json_bytes("ab"), 4);
        assert_eq!(serialized_json_bytes("\0"), 8);
    }

    #[test]
    fn output_within_the_character_limit_passes_through() {
        let output = "short output";

        assert_eq!(
            truncate_output(output, 100, TruncationMode::HeadTail),
            output
        );
    }

    #[test]
    fn output_at_the_character_limit_passes_through() {
        let output = "x".repeat(100);

        assert_eq!(
            truncate_output(&output, 100, TruncationMode::HeadTail),
            output
        );
    }

    #[test]
    fn head_tail_truncation_keeps_both_ends() {
        let output = "a".repeat(100);

        let truncated = truncate_output(&output, 40, TruncationMode::HeadTail);

        assert!(truncated.contains(&"a".repeat(20)));
        assert!(truncated.starts_with("Warning: truncated output (original token count: 25)"));
        assert!(truncated.contains("... 60 bytes omitted ..."));
        assert!(truncated.contains("approximately 15 tokens truncated"));
    }

    #[test]
    fn tail_truncation_keeps_only_the_end() {
        let output = format!("{}BBB", "A".repeat(100));

        let truncated = truncate_output(&output, 10, TruncationMode::Tail);

        assert!(truncated.starts_with("Warning: truncated output"));
        assert!(truncated.contains("... 93 bytes omitted ..."));
        assert!(truncated.contains("approximately 24 tokens truncated"));
        assert!(truncated.ends_with("AAAAAAABBB"));
    }

    #[test]
    fn multibyte_output_truncates_without_splitting_a_character() {
        let output = "✅".repeat(100);

        let truncated = truncate_output(&output, 10, TruncationMode::HeadTail);

        assert!(truncated.contains("Warning: truncated output"));
    }

    #[test]
    fn output_within_the_line_limit_passes_through() {
        let output = "line1\nline2\nline3";

        assert_eq!(truncate_lines(output, 10), output);
    }

    #[test]
    fn output_at_the_line_limit_passes_through() {
        let output = (1..=10)
            .map(|number| format!("line {number}"))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(truncate_lines(&output, 10), output);
    }

    #[test]
    fn line_truncation_keeps_half_from_each_end() {
        let output = (1..=20)
            .map(|number| format!("line {number}"))
            .collect::<Vec<_>>()
            .join("\n");

        let truncated = truncate_lines(&output, 6);

        assert!(truncated.contains("line 1"));
        assert!(truncated.contains("line 3"));
        assert!(truncated.contains("line 18"));
        assert!(truncated.contains("line 20"));
        assert!(truncated.contains("14 lines omitted"));
        assert!(truncated.contains("tokens truncated"));
    }

    #[test]
    fn characters_are_truncated_before_lines() {
        let long_line = "x".repeat(50_000);
        let output = format!("{long_line}\n{long_line}");

        let truncated = truncate_tool_output(&output, ToolOutputLimits::defaults_for("shell"));

        assert!(truncated.len() < output.len());
    }

    #[test]
    fn an_unknown_tool_keeps_its_whole_output() {
        let output = "x".repeat(200);

        let truncated =
            truncate_tool_output(&output, ToolOutputLimits::defaults_for("unknown_tool"));

        assert_eq!(truncated, output);
    }

    #[test]
    fn a_renamed_tool_uses_its_canonical_limits() {
        let shell_output = "x".repeat(40_000);
        let write_output = "x".repeat(2_000);
        let empty = HashMap::new();

        let shell = truncate_tool_output(
            &shell_output,
            ToolOutputLimits::resolve("Bash", "shell", &empty, &empty),
        );
        let write = truncate_tool_output(
            &write_output,
            ToolOutputLimits::resolve("Write", "write_file", &empty, &empty),
        );

        assert!(shell.len() < shell_output.len());
        assert!(write.len() < write_output.len());
    }

    #[test]
    fn an_override_on_the_canonical_name_reaches_a_renamed_tool() {
        let limits = ToolOutputLimits::resolve(
            "Bash",
            "shell",
            &overrides(&[("shell", 100)]),
            &HashMap::new(),
        );

        let truncated = truncate_tool_output(&"x".repeat(1_000), limits);

        assert_eq!(limits.max_chars, Some(100));
        assert!(truncated.contains("Warning: truncated output"));
    }

    #[test]
    fn an_override_on_the_exposed_name_wins_over_the_canonical_one() {
        let limits = ToolOutputLimits::resolve(
            "Bash",
            "shell",
            &overrides(&[("shell", 100), ("Bash", 50)]),
            &overrides(&[("Bash", 5)]),
        );

        assert_eq!(limits.max_chars, Some(50));
        assert_eq!(limits.max_lines, Some(5));
    }

    #[test]
    fn an_override_gives_an_unknown_tool_limits_it_had_none_of() {
        let output = "x".repeat(5_000);
        let limits = ToolOutputLimits::resolve(
            "my_tool",
            "my_tool",
            &overrides(&[("my_tool", 100)]),
            &HashMap::new(),
        );

        let truncated = truncate_tool_output(&output, limits);

        assert!(truncated.len() < output.len());
        assert!(truncated.contains("Warning: truncated output"));
    }

    #[test]
    fn a_line_override_applies_to_an_unknown_tool() {
        let output = (1..=100)
            .map(|number| format!("line {number}"))
            .collect::<Vec<_>>()
            .join("\n");
        let limits = ToolOutputLimits::resolve(
            "my_tool",
            "my_tool",
            &HashMap::new(),
            &overrides(&[("my_tool", 10)]),
        );

        assert!(truncate_tool_output(&output, limits).contains("lines omitted"));
    }

    #[test]
    fn the_built_in_character_limits_are_what_the_tools_expect() {
        for (tool, limit) in [
            ("read_file", Some(50_000)),
            ("shell", Some(30_000)),
            ("grep", Some(20_000)),
            ("glob", Some(20_000)),
            ("spawn_agent", Some(20_000)),
            ("edit_file", Some(10_000)),
            ("apply_patch", Some(10_000)),
            ("write_file", Some(1_000)),
            ("unknown", None),
        ] {
            assert_eq!(
                ToolOutputLimits::defaults_for(tool).max_chars,
                limit,
                "{tool}"
            );
        }
    }

    #[test]
    fn the_built_in_line_limits_are_what_the_tools_expect() {
        for (tool, limit) in [
            ("shell", Some(256)),
            ("grep", Some(200)),
            ("glob", Some(500)),
            ("unknown", None),
        ] {
            assert_eq!(
                ToolOutputLimits::defaults_for(tool).max_lines,
                limit,
                "{tool}"
            );
        }
    }

    #[test]
    fn tail_mode_belongs_to_the_tools_whose_value_is_at_the_end() {
        for tool in ["grep", "glob", "edit_file", "apply_patch", "write_file"] {
            assert_eq!(
                ToolOutputLimits::defaults_for(tool).mode,
                TruncationMode::Tail,
                "{tool}"
            );
        }
        for tool in ["read_file", "shell", "spawn_agent", "unknown"] {
            assert_eq!(
                ToolOutputLimits::defaults_for(tool).mode,
                TruncationMode::HeadTail,
                "{tool}"
            );
        }
    }

    #[test]
    fn the_default_budgets_are_the_documented_ones() {
        let budgets = OutputBudgets::default();

        assert_eq!(budgets.retained_bytes, 1024 * 1024);
        assert_eq!(budgets.serialized_bytes, 1_572_864);
    }
}
