//! Vocabulary describing one model call: token accounting, cost provenance,
//! and the two observed properties of a streaming attempt.

use std::fmt;

use lithos_llm::types::{CostSource as LlmCostSource, TokenCounts};
use serde::{Deserialize, Serialize};

/// Token accounting for one model response.
///
/// The five buckets are **disjoint**: every token is counted in exactly one of
/// them, so [`TokenUsage::total`] is their plain sum. The field names mirror
/// [`lithos_llm::types::TokenCounts`], which normalizes the inclusive counters
/// providers report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt tokens that were neither read from nor written to a cache.
    #[serde(default)]
    pub input:       u64,
    /// Completion tokens that are not reasoning tokens.
    #[serde(default)]
    pub output:      u64,
    /// Completion tokens spent on reasoning, billed at the output rate.
    #[serde(default)]
    pub reasoning:   u64,
    /// Prompt tokens served from a provider cache.
    #[serde(default)]
    pub cache_read:  u64,
    /// Prompt tokens written into a provider cache.
    #[serde(default)]
    pub cache_write: u64,
}

impl TokenUsage {
    /// Adds every bucket without wrapping a counter that reached its limit.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self {
            input:       self.input.saturating_add(other.input),
            output:      self.output.saturating_add(other.output),
            reasoning:   self.reasoning.saturating_add(other.reasoning),
            cache_read:  self.cache_read.saturating_add(other.cache_read),
            cache_write: self.cache_write.saturating_add(other.cache_write),
        }
    }

    /// The sum of all five disjoint buckets.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.input
            .saturating_add(self.output)
            .saturating_add(self.reasoning)
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }

    /// Tokens billed at the output rate: `output + reasoning`.
    #[must_use]
    pub const fn billable_output(self) -> u64 {
        self.output.saturating_add(self.reasoning)
    }

    /// Tokens that occupied the prompt: `input + cache_read + cache_write`.
    #[must_use]
    pub const fn prompt(self) -> u64 {
        self.input
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }
}

impl From<TokenCounts> for TokenUsage {
    fn from(counts: TokenCounts) -> Self {
        Self {
            input:       counts.input,
            output:      counts.output,
            reasoning:   counts.reasoning,
            cache_read:  counts.cache_read,
            cache_write: counts.cache_write,
        }
    }
}

impl From<TokenUsage> for TokenCounts {
    fn from(usage: TokenUsage) -> Self {
        Self {
            input:       usage.input,
            output:      usage.output,
            reasoning:   usage.reasoning,
            cache_read:  usage.cache_read,
            cache_write: usage.cache_write,
        }
    }
}

/// Where a reported cost came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CostSource {
    /// Priced locally from the model catalog.
    Catalog,
    /// Reported by the provider itself.
    Provider,
    /// Supplied by the embedding application.
    Application,
}

impl From<LlmCostSource> for CostSource {
    /// Maps a lithos-llm cost source onto pebble's.
    ///
    /// `lithos_llm::types::CostSource` is `#[non_exhaustive]`. A source pebble
    /// does not yet know is reported as [`CostSource::Application`], the
    /// weakest claim of the three.
    fn from(source: LlmCostSource) -> Self {
        match source {
            LlmCostSource::Catalog => Self::Catalog,
            LlmCostSource::Provider => Self::Provider,
            _ => Self::Application,
        }
    }
}

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
    fn token_usage_round_trips_through_lithos_counts() {
        let usage = TokenUsage {
            input:       100,
            output:      50,
            reasoning:   20,
            cache_read:  80,
            cache_write: 10,
        };
        let counts = TokenCounts::from(usage);
        assert_eq!(counts.input, 100);
        assert_eq!(counts.cache_write, 10);
        assert_eq!(TokenUsage::from(counts), usage);
    }

    #[test]
    fn token_usage_sums_disjoint_buckets() {
        let usage = TokenUsage {
            input:       100,
            output:      50,
            reasoning:   20,
            cache_read:  80,
            cache_write: 10,
        };
        assert_eq!(usage.total(), 260);
        assert_eq!(usage.billable_output(), 70);
        assert_eq!(usage.prompt(), 190);
    }

    #[test]
    fn token_usage_addition_saturates_each_bucket() {
        let almost_full = TokenUsage {
            input: u64::MAX,
            output: 2,
            ..TokenUsage::default()
        };
        let added = almost_full.saturating_add(TokenUsage {
            input: 1,
            output: 3,
            cache_read: 4,
            ..TokenUsage::default()
        });

        assert_eq!(added.input, u64::MAX);
        assert_eq!(added.output, 5);
        assert_eq!(added.cache_read, 4);
    }

    #[test]
    fn token_usage_defaults_every_missing_bucket() {
        let usage: TokenUsage = serde_json::from_value(json!({"input": 7})).expect("parses");
        assert_eq!(usage, TokenUsage {
            input: 7,
            ..TokenUsage::default()
        });
    }

    #[test]
    fn cost_source_is_snake_case_on_the_wire() {
        assert_eq!(
            serde_json::to_value(CostSource::Application).expect("serializes"),
            json!("application")
        );
    }

    #[test]
    fn cost_source_maps_from_lithos() {
        assert_eq!(
            CostSource::from(LlmCostSource::Catalog),
            CostSource::Catalog
        );
        assert_eq!(
            CostSource::from(LlmCostSource::Provider),
            CostSource::Provider
        );
        assert_eq!(
            CostSource::from(LlmCostSource::Application),
            CostSource::Application
        );
    }

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
