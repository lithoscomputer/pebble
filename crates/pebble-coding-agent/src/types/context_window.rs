//! A snapshot of how much of the model's context window a session is using.

use std::fmt;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Which part of the prompt a token count belongs to.
///
/// The declaration order is the order a breakdown appears on the wire: the
/// context-window accounting keeps its counts in a `BTreeMap` keyed by this
/// enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContextWindowCategory {
    /// The assembled system prompt.
    SystemPrompt,
    /// Native tool definitions.
    Tools,
    /// Tool definitions proxied from MCP servers.
    McpTools,
    /// Skill material.
    Skills,
    /// Loaded memory files.
    Memory,
    /// The conversation itself.
    Conversation,
    /// Anything the accounting does not attribute.
    Other,
}

impl ContextWindowCategory {
    /// The wire spelling of this category.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SystemPrompt => "system_prompt",
            Self::Tools => "tools",
            Self::McpTools => "mcp_tools",
            Self::Skills => "skills",
            Self::Memory => "memory",
            Self::Conversation => "conversation",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for ContextWindowCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How the token counts in a snapshot were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContextWindowCountMethod {
    /// A provider token-count call, scaled across the local breakdown.
    ProviderApiScaledBreakdown,
    /// The usage a response reported, scaled across the local breakdown.
    ResponseUsageScaledBreakdown,
    /// A local estimate only.
    LocalEstimate,
}

/// How current a snapshot is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContextWindowStaleness {
    /// Measured for the turn that carries it.
    Live,
    /// Carried forward from an earlier turn.
    Stored,
    /// No measurement was available.
    Unavailable,
}

/// A caveat attached to a snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextWindowWarning {
    /// A stable machine-readable code.
    pub code:    String,
    /// The caveat rendered for a human.
    pub message: String,
}

/// One category's share of the prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextWindowBreakdownItem {
    /// The part of the prompt this line accounts for.
    pub category:      ContextWindowCategory,
    /// Tokens attributed to the category.
    pub tokens:        u64,
    /// The category's share of the context window, as a percentage.
    pub usage_percent: f64,
}

/// How much of the model's context window a session is using.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextWindowSnapshot {
    /// The provider the measurement belongs to.
    pub provider:              String,
    /// The model the measurement belongs to.
    pub model:                 String,
    /// The model's context window, in tokens.
    pub context_window_tokens: u64,
    /// Prompt tokens the session is currently occupying.
    pub input_tokens:          u64,
    /// `input_tokens` as a percentage of `context_window_tokens`.
    pub usage_percent:         f64,
    /// How the counts were obtained.
    pub count_method:          ContextWindowCountMethod,
    /// How current the measurement is.
    pub staleness:             ContextWindowStaleness,
    /// When the measurement was taken.
    #[serde(with = "crate::types::rfc3339_millis")]
    pub generated_at:          SystemTime,
    /// The coding event sequence that produced a stored snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_seq:             Option<u64>,
    /// The per-category breakdown, in category order.
    #[serde(default)]
    pub breakdown:             Vec<ContextWindowBreakdownItem>,
    /// Caveats about the measurement.
    #[serde(default)]
    pub warnings:              Vec<ContextWindowWarning>,
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use serde_json::json;

    use super::*;

    fn snapshot() -> ContextWindowSnapshot {
        ContextWindowSnapshot {
            provider:              "anthropic".into(),
            model:                 "claude-sonnet-5".into(),
            context_window_tokens: 200_000,
            input_tokens:          50_000,
            usage_percent:         25.0,
            count_method:          ContextWindowCountMethod::LocalEstimate,
            staleness:             ContextWindowStaleness::Live,
            generated_at:          UNIX_EPOCH + Duration::from_millis(1_767_225_600_500),
            event_seq:             None,
            breakdown:             Vec::new(),
            warnings:              Vec::new(),
        }
    }

    #[test]
    fn snapshot_serializes_its_timestamp_as_rfc3339_millis() {
        let value = serde_json::to_value(snapshot()).expect("serializes");
        assert_eq!(value["generated_at"], json!("2026-01-01T00:00:00.500Z"));
    }

    #[test]
    fn snapshot_round_trips_with_a_breakdown() {
        let mut original = snapshot();
        original.event_seq = Some(41);
        original.breakdown = vec![
            ContextWindowBreakdownItem {
                category:      ContextWindowCategory::SystemPrompt,
                tokens:        1_000,
                usage_percent: 0.5,
            },
            ContextWindowBreakdownItem {
                category:      ContextWindowCategory::Conversation,
                tokens:        49_000,
                usage_percent: 24.5,
            },
        ];
        original.warnings = vec![ContextWindowWarning {
            code:    "estimate".into(),
            message: "counts are estimated locally".into(),
        }];

        let json = serde_json::to_string(&original).expect("serializes");
        let restored: ContextWindowSnapshot = serde_json::from_str(&json).expect("parses");
        assert_eq!(restored, original);
    }

    #[test]
    fn absent_breakdown_warnings_and_sequence_parse_as_empty() {
        let restored: ContextWindowSnapshot = serde_json::from_value(json!({
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "context_window_tokens": 200_000,
            "input_tokens": 50_000,
            "usage_percent": 25.0,
            "count_method": "local_estimate",
            "staleness": "live",
            "generated_at": "2026-01-01T00:00:00.500Z",
        }))
        .expect("parses");
        assert_eq!(restored, snapshot());
    }

    #[test]
    fn category_order_is_the_wire_order_of_a_breakdown() {
        let mut categories = vec![
            ContextWindowCategory::Other,
            ContextWindowCategory::Conversation,
            ContextWindowCategory::SystemPrompt,
            ContextWindowCategory::Tools,
        ];
        categories.sort_unstable();
        assert_eq!(categories, vec![
            ContextWindowCategory::SystemPrompt,
            ContextWindowCategory::Tools,
            ContextWindowCategory::Conversation,
            ContextWindowCategory::Other,
        ]);
    }

    #[test]
    fn count_method_and_staleness_are_snake_case_on_the_wire() {
        assert_eq!(
            serde_json::to_value(ContextWindowCountMethod::ProviderApiScaledBreakdown)
                .expect("serializes"),
            json!("provider_api_scaled_breakdown")
        );
        assert_eq!(
            serde_json::to_value(ContextWindowStaleness::Unavailable).expect("serializes"),
            json!("unavailable")
        );
        assert_eq!(
            serde_json::to_value(ContextWindowCategory::McpTools).expect("serializes"),
            json!("mcp_tools")
        );
    }
}
