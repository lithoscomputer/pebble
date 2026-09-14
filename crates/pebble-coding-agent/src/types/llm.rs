//! Vocabulary describing one model call: token accounting and cost, and the
//! two observed properties of a streaming attempt.
//!
//! Token accounting and cost are lithos-llm's types, re-exported here so an
//! application reads one vocabulary for what a call used and what it cost:
//! [`TokenCounts`] is five disjoint token buckets, [`Cost`] is a price in USD
//! micros with its [`CostSource`], and [`Usage`] pairs the two with the cost
//! optional. Every usage a pebble event, report, or projection carries is a
//! [`Usage`]; a sum is [`Usage::saturating_add`], which keeps a cost only when
//! every part that used tokens is priced.

use std::fmt;

pub use lithos_llm::types::{Cost, CostSource, TokenCounts, Usage};
use serde::{Deserialize, Serialize};

/// Which kind of output a provider produced first for an inference attempt.
///
/// Observed, never inferred: a turn that opens with a tool call emits no text
/// or reasoning delta, so all three variants are required for the first-output
/// edge to fire on every turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LlmOutputKind {
    /// Reasoning arrived first.
    Reasoning,
    /// Assistant text arrived first.
    Text,
    /// A tool call arrived first.
    ToolCall,
}

impl LlmOutputKind {
    /// The wire spelling of this kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::Text => "text",
            Self::ToolCall => "tool_call",
        }
    }
}

impl fmt::Display for LlmOutputKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Which retry loop produced the `attempt` index on a retry event.
///
/// `attempt` is a 0-based counter fed by two independent loops: the retry
/// policy that reopens a stream before any output is visible (`Open`), and the
/// stream-consume loop that replays a turn whose stream broke after visible
/// output (`Consume`). Without this discriminator a reader cannot tell which
/// counter an index belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LlmRetryPhase {
    /// The attempt to open or reopen the stream.
    Open,
    /// The replay of a turn whose stream broke while being consumed.
    Consume,
}

impl LlmRetryPhase {
    /// The wire spelling of this phase.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Consume => "consume",
        }
    }
}

impl fmt::Display for LlmRetryPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn output_kind_and_retry_phase_render_their_wire_spelling() {
        assert_eq!(LlmOutputKind::ToolCall.to_string(), "tool_call");
        assert_eq!(
            serde_json::to_value(LlmOutputKind::ToolCall).expect("serializes"),
            json!("tool_call")
        );
        assert_eq!(LlmRetryPhase::Consume.to_string(), "consume");
        assert_eq!(
            serde_json::to_value(LlmRetryPhase::Consume).expect("serializes"),
            json!("consume")
        );
    }
}
