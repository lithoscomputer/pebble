//! Readable model reasoning, normalized into a provider-neutral shape.
//!
//! Providers expose reasoning through several unrelated channels: OpenAI
//! Responses reasoning items, OpenAI-compatible `reasoning_details`, and
//! flattened reasoning text. [`ReasoningOutput`] reduces all of them to the
//! two capabilities consumers actually care about, so the durable event
//! contract does not change shape when a provider dialect does.
//!
//! Normalization is deliberately tolerant: provider payloads are read as
//! [`serde_json::Value`] with optional lookups, so unknown detail variants,
//! missing members, extra members, and unexpected member types are ignored
//! rather than surfaced as errors.

use std::result::Result as StdResult;

use lithos_llm::types::ContentPart;
use serde::{Deserialize, Serialize, de};

/// The opaque content kind holding an OpenAI Responses reasoning item.
const OPENAI_REASONING_KIND: &str = "openai.reasoning";

/// The opaque content kind holding OpenAI-compatible `reasoning_details`.
const OPENAI_COMPAT_REASONING_DETAILS_KIND: &str = "openai_compatible.reasoning_details";

/// Separator between distinct complete reasoning blocks. Fragments of one
/// logical block are coalesced by the streaming codecs before they reach this
/// module.
const BLOCK_SEPARATOR: &str = "\n\n";

/// Readable model reasoning normalized into a provider-neutral shape.
///
/// Both fields may be populated for the same response. An emitted object
/// always carries at least one of them; opaque provider material
/// (signatures, IDs, encrypted or redacted payloads) never appears here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReasoningOutput {
    /// Model-authored summary of its reasoning, safe to show to users.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    /// Verbatim readable reasoning text, when the provider returns it in
    /// addition to (or instead of) a summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trace:   Option<String>,
}

impl ReasoningOutput {
    /// Creates reasoning output with both a model-authored summary and a
    /// verbatim trace.
    #[must_use]
    pub fn new(summary: impl Into<String>, trace: impl Into<String>) -> Self {
        Self {
            summary: Some(summary.into()),
            trace:   Some(trace.into()),
        }
    }

    /// Creates reasoning output containing only a model-authored summary.
    #[must_use]
    pub fn from_summary(summary: impl Into<String>) -> Self {
        Self {
            summary: Some(summary.into()),
            trace:   None,
        }
    }

    /// Creates reasoning output containing only a verbatim trace.
    #[must_use]
    pub fn from_trace(trace: impl Into<String>) -> Self {
        Self {
            summary: None,
            trace:   Some(trace.into()),
        }
    }

    /// Normalizes the content parts of a final response into readable
    /// reasoning.
    ///
    /// Returns `None` when the response carries no readable reasoning, so an
    /// event without reasoning keeps its serialized shape.
    #[must_use]
    pub fn from_content(content: &[ContentPart]) -> Option<Self> {
        let mut blocks = Blocks::default();
        for part in content {
            match part {
                ContentPart::Reasoning(reasoning) if !reasoning.redacted => {
                    push_block(&mut blocks.fallback_trace, &reasoning.text);
                }
                ContentPart::Opaque { kind, data } if kind == OPENAI_REASONING_KIND => {
                    collect_openai_reasoning_item(data, &mut blocks);
                }
                ContentPart::Opaque { kind, data }
                    if kind == OPENAI_COMPAT_REASONING_DETAILS_KIND =>
                {
                    collect_reasoning_details(data, &mut blocks);
                }
                _ => {}
            }
        }
        blocks.into_output()
    }

    /// Returns the model-authored summary, when present.
    #[must_use]
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Returns the verbatim readable reasoning trace, when present.
    #[must_use]
    pub fn trace(&self) -> Option<&str> {
        self.trace.as_deref()
    }
}

impl<'de> Deserialize<'de> for ReasoningOutput {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            #[serde(default)]
            summary: Option<String>,
            #[serde(default)]
            trace:   Option<String>,
        }

        let Fields { summary, trace } = Fields::deserialize(deserializer)?;
        match (summary, trace) {
            (Some(summary), Some(trace)) => Ok(Self::new(summary, trace)),
            (Some(summary), None) => Ok(Self::from_summary(summary)),
            (None, Some(trace)) => Ok(Self::from_trace(trace)),
            (None, None) => Err(de::Error::custom(
                "reasoning output requires a summary or trace",
            )),
        }
    }
}

/// Readable blocks collected per normalized field.
///
/// Explicit blocks come from a channel with documented reasoning semantics.
/// Fallback blocks come from flattened provider reasoning text, which
/// aggregators commonly duplicate alongside a structured channel. They only
/// fill a trace that no explicit trace produced.
#[derive(Default)]
struct Blocks<'a> {
    explicit_summary: Vec<&'a str>,
    explicit_trace:   Vec<&'a str>,
    fallback_trace:   Vec<&'a str>,
}

impl Blocks<'_> {
    fn into_output(self) -> Option<ReasoningOutput> {
        let summary = join_blocks(&self.explicit_summary);
        let trace = join_blocks(&self.explicit_trace)
            .or_else(|| join_blocks(&self.fallback_trace))
            .filter(|trace| summary.as_ref() != Some(trace));

        match (summary, trace) {
            (Some(summary), Some(trace)) => Some(ReasoningOutput::new(summary, trace)),
            (Some(summary), None) => Some(ReasoningOutput::from_summary(summary)),
            (None, Some(trace)) => Some(ReasoningOutput::from_trace(trace)),
            (None, None) => None,
        }
    }
}

/// Joins retained complete blocks in provider order. Text is never trimmed or
/// rewritten.
fn join_blocks(blocks: &[&str]) -> Option<String> {
    (!blocks.is_empty()).then(|| blocks.join(BLOCK_SEPARATOR))
}

fn push_block<'a>(blocks: &mut Vec<&'a str>, block: &'a str) {
    if !block.trim().is_empty() {
        blocks.push(block);
    }
}

/// Reads a text-bearing member with the provider's documented semantics.
fn readable_member<'a>(entry: &'a serde_json::Value, member: &str) -> Option<&'a str> {
    entry.get(member).and_then(serde_json::Value::as_str)
}

/// Extracts readable text from an OpenAI Responses `reasoning` output item.
///
/// `summary[].text` is the model-authored summary; `content[]` entries typed
/// `reasoning_text` are the verbatim trace. `encrypted_content`, `id`, and
/// `status` are opaque and ignored.
fn collect_openai_reasoning_item<'a>(item: &'a serde_json::Value, blocks: &mut Blocks<'a>) {
    if let Some(entries) = item.get("summary").and_then(serde_json::Value::as_array) {
        for entry in entries {
            if let Some(text) = entry.as_str() {
                push_block(&mut blocks.explicit_summary, text);
            } else if let Some(text) = readable_member(entry, "text") {
                push_block(&mut blocks.explicit_summary, text);
            }
        }
    }
    if let Some(entries) = item.get("content").and_then(serde_json::Value::as_array) {
        for entry in entries {
            let Some(text) = readable_member(entry, "text") else {
                continue;
            };
            if readable_member(entry, "type").unwrap_or_default() == "reasoning_text" {
                push_block(&mut blocks.explicit_trace, text);
            }
        }
    }
}

/// Extracts readable text from OpenAI-compatible `reasoning_details` entries.
fn collect_reasoning_details<'a>(details: &'a serde_json::Value, blocks: &mut Blocks<'a>) {
    let Some(entries) = details.as_array() else {
        return;
    };
    for entry in entries {
        match readable_member(entry, "type").unwrap_or_default() {
            "reasoning.text" => {
                if let Some(text) = readable_member(entry, "text") {
                    push_block(&mut blocks.explicit_trace, text);
                }
            }
            "reasoning.summary" => {
                if let Some(text) = readable_member(entry, "summary") {
                    push_block(&mut blocks.explicit_summary, text);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::ReasoningContent;
    use serde_json::json;

    use super::*;

    fn reasoning(text: &str) -> ContentPart {
        ContentPart::Reasoning(ReasoningContent {
            text:             text.to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         false,
        })
    }

    fn openai_reasoning(item: serde_json::Value) -> ContentPart {
        ContentPart::opaque(OPENAI_REASONING_KIND, item)
    }

    fn reasoning_details(details: serde_json::Value) -> ContentPart {
        ContentPart::opaque(OPENAI_COMPAT_REASONING_DETAILS_KIND, details)
    }

    // --- Type shape ---

    #[test]
    fn summary_only_round_trips_without_trace_member() {
        let output = ReasoningOutput::from_summary("checked the parser first");
        let value = serde_json::to_value(&output).expect("serializes");
        assert_eq!(value, json!({"summary": "checked the parser first"}));
        assert_eq!(
            serde_json::from_value::<ReasoningOutput>(value).expect("parses"),
            output
        );
    }

    #[test]
    fn trace_only_round_trips_without_summary_member() {
        let output = ReasoningOutput::from_trace("step one, step two");
        let value = serde_json::to_value(&output).expect("serializes");
        assert_eq!(value, json!({"trace": "step one, step two"}));
        assert_eq!(
            serde_json::from_value::<ReasoningOutput>(value).expect("parses"),
            output
        );
    }

    #[test]
    fn both_fields_round_trip() {
        let output = ReasoningOutput::new("summary", "trace");
        let value = serde_json::to_value(&output).expect("serializes");
        assert_eq!(value, json!({"summary": "summary", "trace": "trace"}));
        assert_eq!(
            serde_json::from_value::<ReasoningOutput>(value).expect("parses"),
            output
        );
    }

    #[test]
    fn empty_object_is_rejected() {
        let error = serde_json::from_value::<ReasoningOutput>(json!({}))
            .expect_err("an empty object carries no reasoning");
        assert!(error.to_string().contains("requires a summary or trace"));
    }

    #[test]
    fn explicit_nulls_are_rejected() {
        let error =
            serde_json::from_value::<ReasoningOutput>(json!({"summary": null, "trace": null}))
                .expect_err("explicit nulls carry no reasoning");
        assert!(error.to_string().contains("requires a summary or trace"));
    }

    // --- Normalization ---

    #[test]
    fn non_redacted_reasoning_becomes_a_trace() {
        let output = ReasoningOutput::from_content(&[reasoning("weighing the options")])
            .expect("readable reasoning");
        assert!(output.summary().is_none());
        assert_eq!(output.trace(), Some("weighing the options"));
    }

    #[test]
    fn redacted_reasoning_yields_no_readable_reasoning() {
        let redacted = ContentPart::Reasoning(ReasoningContent {
            text:             "AAAAopaque".to_owned(),
            signature:        Some("sig".to_owned()),
            signature_origin: Some("anthropic".to_owned()),
            redacted:         true,
        });
        assert!(ReasoningOutput::from_content(&[redacted]).is_none());
    }

    #[test]
    fn responses_item_with_summary_and_reasoning_text_produces_both_fields() {
        let output = ReasoningOutput::from_content(&[openai_reasoning(json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "gAAAAA",
            "summary": [{"type": "summary_text", "text": "inspect first"}],
            "content": [{"type": "reasoning_text", "text": "step one"}],
        }))])
        .expect("readable reasoning");
        assert_eq!(output.summary(), Some("inspect first"));
        assert_eq!(output.trace(), Some("step one"));
    }

    #[test]
    fn responses_blocks_join_in_provider_order() {
        let output = ReasoningOutput::from_content(&[openai_reasoning(json!({
            "summary": [
                {"type": "summary_text", "text": "first"},
                {"type": "summary_text", "text": "second"},
            ],
        }))])
        .expect("readable reasoning");
        assert_eq!(output.summary(), Some("first\n\nsecond"));
    }

    #[test]
    fn responses_summary_fallback_part_is_deduplicated_against_the_summary() {
        // What the lithos OpenAI Responses codec decodes for a `reasoning`
        // item with two `summary_text` blocks and no `content[]`: a
        // `Reasoning` part holding the summaries joined by a blank line, then
        // the opaque item. The fallback matches the joined summary exactly and
        // so never becomes a trace.
        let output = ReasoningOutput::from_content(&[
            reasoning("A\n\nB"),
            openai_reasoning(json!({
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "gAAAAA",
                "summary": [
                    {"type": "summary_text", "text": "A"},
                    {"type": "summary_text", "text": "B"},
                ],
            })),
        ])
        .expect("readable reasoning");
        assert_eq!(output, ReasoningOutput::from_summary("A\n\nB"));
        assert_eq!(
            serde_json::to_value(&output).expect("serializes"),
            json!({"summary": "A\n\nB"})
        );
    }

    #[test]
    fn responses_summary_fallback_joined_differently_survives_as_a_trace() {
        // Deduplication is by equality alone: a fallback joined with "" (the
        // codec behavior lithos-llm PR #2 replaced) is reported as a trace.
        let output = ReasoningOutput::from_content(&[
            reasoning("AB"),
            openai_reasoning(json!({
                "summary": [
                    {"type": "summary_text", "text": "A"},
                    {"type": "summary_text", "text": "B"},
                ],
            })),
        ])
        .expect("readable reasoning");
        assert_eq!(output, ReasoningOutput::new("A\n\nB", "AB"));
    }

    #[test]
    fn responses_item_with_reasoning_text_is_not_duplicated_by_its_reasoning_part() {
        // The codec pairs a `Reasoning` part holding exactly the `content[]`
        // text with the opaque item that also carries it.
        let output = ReasoningOutput::from_content(&[
            reasoning("step one"),
            openai_reasoning(json!({
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "A"}],
                "content": [{"type": "reasoning_text", "text": "step one"}],
            })),
        ])
        .expect("readable reasoning");
        assert_eq!(output, ReasoningOutput::new("A", "step one"));
    }

    #[test]
    fn unknown_responses_content_types_remain_opaque() {
        assert!(
            ReasoningOutput::from_content(&[openai_reasoning(json!({
                "content": [{"type": "reasoning_future", "text": "not classified"}],
            }))])
            .is_none()
        );
    }

    #[test]
    fn structured_details_produce_summary_and_trace() {
        let output = ReasoningOutput::from_content(&[reasoning_details(json!([
            {"type": "reasoning.summary", "summary": "checked the parser"},
            {"type": "reasoning.text", "text": "read convert.rs", "signature": "sig"},
        ]))])
        .expect("readable reasoning");
        assert_eq!(output.summary(), Some("checked the parser"));
        assert_eq!(output.trace(), Some("read convert.rs"));
    }

    #[test]
    fn encrypted_details_are_excluded() {
        let output = ReasoningOutput::from_content(&[reasoning_details(json!([
            {"type": "reasoning.encrypted", "data": "gAAAAAsecret", "format": "openai-responses-v1"},
            {"type": "reasoning.summary", "summary": "visible"},
        ]))])
        .expect("readable reasoning");
        assert_eq!(output.summary(), Some("visible"));
        assert!(output.trace().is_none());
    }

    #[test]
    fn encrypted_only_details_produce_no_reasoning() {
        assert!(
            ReasoningOutput::from_content(&[reasoning_details(json!([
                {"type": "reasoning.encrypted", "data": "gAAAAAsecret"},
            ]))])
            .is_none()
        );
    }

    #[test]
    fn unknown_detail_variants_remain_opaque() {
        assert!(
            ReasoningOutput::from_content(&[reasoning_details(json!([
                {"type": "reasoning.future", "text": "new channel"},
            ]))])
            .is_none()
        );
    }

    #[test]
    fn malformed_details_are_ignored_without_failing() {
        assert!(
            ReasoningOutput::from_content(&[reasoning_details(json!("not-an-array"))]).is_none()
        );
        assert!(
            ReasoningOutput::from_content(&[reasoning_details(json!([
                42,
                {"type": "reasoning.summary", "summary": 7},
                {"no_type": true},
            ]))])
            .is_none()
        );
    }

    #[test]
    fn structured_details_suppress_a_duplicate_flattened_value() {
        let output = ReasoningOutput::from_content(&[
            reasoning_details(json!([
                {"type": "reasoning.summary", "summary": "checked the parser"},
            ])),
            reasoning("checked the parser"),
        ])
        .expect("readable reasoning");
        assert_eq!(output.summary(), Some("checked the parser"));
        assert!(output.trace().is_none());
    }

    #[test]
    fn structured_trace_takes_precedence_over_flattened_trace() {
        let output = ReasoningOutput::from_content(&[
            reasoning_details(json!([{"type": "reasoning.text", "text": "verbatim"}])),
            reasoning("flattened"),
        ])
        .expect("readable reasoning");
        assert!(output.summary().is_none());
        assert_eq!(output.trace(), Some("verbatim"));
    }

    #[test]
    fn structured_summary_keeps_a_distinct_flattened_trace() {
        let output = ReasoningOutput::from_content(&[
            reasoning_details(json!([
                {"type": "reasoning.summary", "summary": "short summary"},
            ])),
            reasoning("full verbatim trace"),
        ])
        .expect("readable reasoning");
        assert_eq!(output.summary(), Some("short summary"));
        assert_eq!(output.trace(), Some("full verbatim trace"));
    }

    #[test]
    fn whitespace_only_fragments_do_not_create_reasoning() {
        assert!(ReasoningOutput::from_content(&[reasoning("   \n ")]).is_none());
    }

    #[test]
    fn non_empty_text_is_preserved_verbatim() {
        let output = ReasoningOutput::from_content(&[reasoning("  indented thought\n")])
            .expect("readable reasoning");
        assert_eq!(output.trace(), Some("  indented thought\n"));
    }

    #[test]
    fn unrelated_content_parts_are_ignored() {
        let parts = vec![
            ContentPart::Text {
                text: "answer".to_owned(),
            },
            ContentPart::opaque(
                "openai.message",
                json!({"type": "message", "content": [{"text": "answer"}]}),
            ),
        ];
        assert!(ReasoningOutput::from_content(&parts).is_none());
    }
}
