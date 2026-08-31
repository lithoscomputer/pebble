//! Vocabulary describing a subordinate process a tool ran.

use std::fmt;

use serde::{Deserialize, Serialize};

/// How a subordinate process ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CommandTermination {
    /// The process ran to completion and reported an exit status.
    Exited,
    /// The process exceeded its time budget and was stopped.
    TimedOut,
    /// The process was cancelled before it finished.
    Cancelled,
}

impl CommandTermination {
    /// The wire spelling of this termination.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exited => "exited",
            Self::TimedOut => "timed_out",
            Self::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for CommandTermination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The retained tail of a subordinate process's output streams.
///
/// A tail is bounded and may be redacted before it reaches an event, so both
/// streams carry a truncation flag.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecOutputTail {
    /// The retained tail of standard output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdout:           Option<String>,
    /// The retained tail of standard error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stderr:           Option<String>,
    /// Whether `stdout` dropped earlier output.
    #[serde(default, skip_serializing_if = "is_false")]
    pub stdout_truncated: bool,
    /// Whether `stderr` dropped earlier output.
    #[serde(default, skip_serializing_if = "is_false")]
    pub stderr_truncated: bool,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if predicates receive fields by reference"
)]
fn is_false(value: &bool) -> bool {
    !*value
}

impl ExecOutputTail {
    /// Whether both retained tails are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stdout.as_deref().unwrap_or_default().is_empty()
            && self.stderr.as_deref().unwrap_or_default().is_empty()
    }

    /// The byte length of the retained standard-output tail.
    #[must_use]
    pub fn stdout_len(&self) -> usize {
        self.stdout.as_deref().map_or(0, str::len)
    }

    /// The byte length of the retained standard-error tail.
    #[must_use]
    pub fn stderr_len(&self) -> usize {
        self.stderr.as_deref().map_or(0, str::len)
    }

    /// A flat view for tracing, carrying sizes and flags but never content.
    #[must_use]
    pub fn trace_summary(tail: Option<&Self>) -> ExecOutputTailTrace {
        ExecOutputTailTrace {
            present:          tail.is_some(),
            stdout_bytes:     tail.map_or(0, Self::stdout_len),
            stderr_bytes:     tail.map_or(0, Self::stderr_len),
            stdout_truncated: tail.is_some_and(|tail| tail.stdout_truncated),
            stderr_truncated: tail.is_some_and(|tail| tail.stderr_truncated),
        }
    }
}

/// A flat view of an [`ExecOutputTail`] for tracing field expansion.
///
/// It deliberately carries no output text, so tracing an exec result can never
/// log process output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecOutputTailTrace {
    /// Whether a tail was captured at all.
    pub present:          bool,
    /// The byte length of the retained standard-output tail.
    pub stdout_bytes:     usize,
    /// The byte length of the retained standard-error tail.
    pub stderr_bytes:     usize,
    /// Whether the standard-output tail dropped earlier output.
    pub stdout_truncated: bool,
    /// Whether the standard-error tail dropped earlier output.
    pub stderr_truncated: bool,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn termination_is_snake_case_on_the_wire() {
        assert_eq!(
            serde_json::to_value(CommandTermination::TimedOut).expect("serializes"),
            json!("timed_out")
        );
        assert_eq!(CommandTermination::TimedOut.as_str(), "timed_out");
    }

    #[test]
    fn empty_tail_serializes_to_an_empty_object() {
        let tail = ExecOutputTail::default();
        assert!(tail.is_empty());
        assert_eq!(serde_json::to_value(&tail).expect("serializes"), json!({}));
    }

    #[test]
    fn truncation_flags_are_omitted_when_false() {
        let tail = ExecOutputTail {
            stdout:           Some("out".into()),
            stderr:           None,
            stdout_truncated: true,
            stderr_truncated: false,
        };
        assert_eq!(
            serde_json::to_value(&tail).expect("serializes"),
            json!({"stdout": "out", "stdout_truncated": true})
        );
    }

    #[test]
    fn tail_round_trips() {
        let tail = ExecOutputTail {
            stdout:           Some("out".into()),
            stderr:           Some("err".into()),
            stdout_truncated: true,
            stderr_truncated: true,
        };
        let value = serde_json::to_value(&tail).expect("serializes");
        assert_eq!(
            serde_json::from_value::<ExecOutputTail>(value).expect("parses"),
            tail
        );
    }

    #[test]
    fn trace_summary_reports_sizes_without_content() {
        let tail = ExecOutputTail {
            stdout:           Some("hello".into()),
            stderr:           None,
            stdout_truncated: true,
            stderr_truncated: false,
        };
        let summary = ExecOutputTail::trace_summary(Some(&tail));
        assert_eq!(summary, ExecOutputTailTrace {
            present:          true,
            stdout_bytes:     5,
            stderr_bytes:     0,
            stdout_truncated: true,
            stderr_truncated: false,
        });
        assert!(!format!("{summary:?}").contains("hello"));
    }

    #[test]
    fn trace_summary_of_no_tail_is_all_zero() {
        assert_eq!(ExecOutputTail::trace_summary(None), ExecOutputTailTrace {
            present:          false,
            stdout_bytes:     0,
            stderr_bytes:     0,
            stdout_truncated: false,
            stderr_truncated: false,
        });
    }
}
