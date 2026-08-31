//! Durable conversation history: the turns a session replays to the model.
//!
//! History keeps pebble's own [`Message`] turns rather than wire messages,
//! because a turn carries usage, a timestamp, and the steering distinction the
//! wire form drops. [`History::to_llm_messages`] is the single translation to
//! the wire form.
//!
//! Compaction replaces the older turns with a summary. It is deliberately
//! conservative about where it cuts: a preserved tool result whose tool call
//! would be discarded is a broken conversation that providers reject, so the
//! cut only ever moves earlier, never between a call and its result.

use std::collections::HashSet;
use std::mem;
use std::time::SystemTime;

use lithos_llm::types::{ContentPart, Message as LlmMessage};

use crate::record::StoredMessage;
use crate::types::{Message, TokenUsage};

/// How many tokens of discarded user input compaction carries forward.
const COMPACTION_USER_MESSAGE_TOKEN_BUDGET: usize = 20_000;

/// The rough characters-per-token ratio the compaction budget assumes.
const CHARS_PER_TOKEN: usize = 4;

/// Opaque parts that stop being replayable once compaction rewrites the turns
/// around them.
///
/// An OpenAI reasoning item names the output item that must follow it, so once
/// a summary replaces that surrounding context the pair is unreplayable and
/// the provider rejects it. Reasoning blocks that carry their own text and
/// signature — Anthropic thinking, and the OpenAI-compatible reasoning details
/// — stay, because they remain valid on their own.
const STALE_AFTER_COMPACTION: [&str; 4] = [
    "openai.reasoning",
    "openai.message",
    // The spellings the reference implementation persisted, which lithos-llm's
    // OpenAI codec still claims.
    "openai_reasoning",
    "openai_message",
];

/// The turns a session will replay to the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    turns: Vec<Message>,
}

impl History {
    /// Restores history from a stored record's messages.
    #[must_use]
    pub fn from_stored_messages(messages: &[StoredMessage]) -> Self {
        Self {
            turns: messages.iter().map(Message::from_stored_message).collect(),
        }
    }

    /// Appends one turn.
    pub fn push(&mut self, turn: Message) {
        self.turns.push(turn);
    }

    /// The turns, oldest first.
    #[must_use]
    pub fn turns(&self) -> &[Message] {
        &self.turns
    }

    /// Whether there are no turns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// How many turns there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Projects the turns into their stored form.
    #[must_use]
    pub fn to_stored_messages(&self) -> Vec<StoredMessage> {
        self.turns.iter().map(Message::to_stored_message).collect()
    }

    /// Projects the turns into the wire messages sent to the provider.
    #[must_use]
    pub fn to_llm_messages(&self) -> Vec<LlmMessage> {
        self.turns.iter().map(Message::to_llm_message).collect()
    }

    /// Replaces everything but the trailing `preserve_count` turns with a
    /// summary.
    ///
    /// The cut moves earlier when it would separate a tool call from its
    /// result, and does nothing at all when honoring that would preserve the
    /// whole history. Recent user input from the discarded turns is carried
    /// forward after the summary, within a fixed token budget.
    ///
    /// Preserved assistant turns have their usage cleared: the numbers
    /// described a prompt that no longer exists, and a later context-window
    /// estimate must not read them as its baseline. The authoritative
    /// accounting is on the emitted events.
    pub fn compact(&mut self, preserve_count: usize, summary: String) {
        if self.turns.len() <= preserve_count {
            return;
        }
        let preserve_start = self.compact_preserve_start(preserve_count);
        self.compact_from(preserve_start, summary);
    }

    /// The earliest turn index that can be preserved without separating a tool
    /// call from its result.
    #[must_use]
    pub(crate) fn compact_preserve_start(&self, preserve_count: usize) -> usize {
        compact_preserve_start(&self.turns, preserve_count)
    }

    /// Compacts at an index a caller already chose.
    ///
    /// Does nothing when the index would preserve everything (`0`) or lies
    /// past the end.
    pub(crate) fn compact_from(&mut self, preserve_start: usize, summary: String) {
        if preserve_start == 0 || preserve_start > self.turns.len() {
            return;
        }
        let mut preserved = self.turns.split_off(preserve_start);
        invalidate_preserved_usage(&mut preserved);
        let discarded = mem::take(&mut self.turns);
        let carried_forward =
            extract_recent_user_messages(discarded, COMPACTION_USER_MESSAGE_TOKEN_BUDGET);
        self.turns.push(Message::System {
            content:   summary,
            timestamp: SystemTime::now(),
        });
        self.turns.extend(carried_forward);
        self.turns.extend(preserved);
        self.strip_stale_provider_parts();
    }

    /// Drops provider parts that compaction has invalidated.
    fn strip_stale_provider_parts(&mut self) {
        for turn in &mut self.turns {
            if let Message::Assistant { provider_parts, .. } = turn {
                provider_parts.retain(|part| !is_stale_after_compaction(part));
            }
        }
    }
}

/// Clears the reported usage on every preserved assistant turn.
fn invalidate_preserved_usage(preserved: &mut [Message]) {
    for turn in preserved {
        if let Message::Assistant { usage, .. } = turn {
            *usage = TokenUsage::default();
        }
    }
}

/// Whether a provider part is no longer replayable after compaction.
fn is_stale_after_compaction(part: &ContentPart) -> bool {
    matches!(
        part,
        ContentPart::Opaque { kind, .. } if STALE_AFTER_COMPACTION.contains(&kind.as_str())
    )
}

/// Collects the most recent user turns from the discarded head, newest-first
/// until the budget is spent, and returns them in conversation order.
fn extract_recent_user_messages(discarded: Vec<Message>, token_budget: usize) -> Vec<Message> {
    let char_budget = token_budget.saturating_mul(CHARS_PER_TOKEN);
    let mut spent = 0;
    let mut first_kept = discarded.len();

    for (index, turn) in discarded.iter().enumerate().rev() {
        if let Message::User { content, .. } = turn {
            if spent + content.len() > char_budget {
                break;
            }
            spent += content.len();
            first_kept = index;
        }
    }

    discarded
        .into_iter()
        .skip(first_kept)
        .filter(|turn| matches!(turn, Message::User { .. }))
        .collect()
}

/// Walks the preserve boundary earlier until every preserved tool result has
/// its tool call preserved too.
fn compact_preserve_start(turns: &[Message], preserve_count: usize) -> usize {
    let mut start = turns.len().saturating_sub(preserve_count);
    let mut required_call_ids = HashSet::new();
    add_tool_result_call_ids(&turns[start..], &mut required_call_ids);

    loop {
        let Some(call_index) = turns[..start].iter().rposition(|turn| {
            let Message::Assistant { tool_calls, .. } = turn else {
                return false;
            };
            tool_calls
                .iter()
                .any(|tool_call| required_call_ids.contains(tool_call.id.as_str()))
        }) else {
            return start;
        };

        // Moving the boundary earlier can pull in more results, whose calls
        // may sit earlier still.
        add_tool_result_call_ids(&turns[call_index..start], &mut required_call_ids);
        start = call_index;
    }
}

fn add_tool_result_call_ids<'a>(turns: &'a [Message], call_ids: &mut HashSet<&'a str>) {
    for turn in turns {
        if let Message::ToolResults { results, .. } = turn {
            call_ids.extend(results.iter().map(|result| result.tool_call_id.as_str()));
        }
    }
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::{ReasoningContent, Role, ToolCall, ToolResult};
    use serde_json::json;

    use super::*;

    fn now() -> SystemTime {
        SystemTime::now()
    }

    fn user(content: &str) -> Message {
        Message::User {
            content:   content.to_owned(),
            timestamp: now(),
        }
    }

    fn assistant(content: &str, response_id: &str) -> Message {
        Message::Assistant {
            content:        content.to_owned(),
            tool_calls:     Vec::new(),
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    response_id.to_owned(),
            timestamp:      now(),
        }
    }

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall::function(id, name, arguments)
    }

    fn tool_result(call_id: &str, output: &str) -> ToolResult {
        ToolResult {
            tool_call_id: call_id.to_owned(),
            name:         None,
            content:      vec![ContentPart::Text {
                text: output.to_owned(),
            }],
            is_error:     false,
        }
    }

    fn openai_reasoning_part() -> ContentPart {
        ContentPart::opaque(
            "openai.reasoning",
            json!({ "type": "reasoning", "id": "rs_1" }),
        )
    }

    fn thinking_part(text: &str, signature: Option<&str>) -> ContentPart {
        ContentPart::Reasoning(ReasoningContent {
            text:             text.to_owned(),
            signature:        signature.map(ToOwned::to_owned),
            signature_origin: signature.map(|_| "anthropic".to_owned()),
            redacted:         false,
        })
    }

    fn text_of(message: &LlmMessage) -> String {
        message
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn compact_replaces_old_turns_with_summary() {
        let mut history = History::default();
        for index in 0..8 {
            history.push(user(&format!("msg {index}")));
        }

        history.compact(4, "Summary of old conversation".into());

        // One summary, four carried-forward user turns, four preserved.
        assert_eq!(history.len(), 9);
    }

    #[test]
    fn compact_noop_when_fewer_turns_than_preserve() {
        let mut history = History::default();
        for index in 0..3 {
            history.push(user(&format!("msg {index}")));
        }

        history.compact(6, "Summary".into());

        assert_eq!(history.len(), 3);
    }

    #[test]
    fn compact_preserves_recent_turns() {
        let mut history = History::default();
        for index in 0..8 {
            history.push(user(&format!("msg {index}")));
        }

        history.compact(4, "Summary".into());

        let turns = history.turns();
        assert!(matches!(&turns[0], Message::System { .. }));
        for (offset, index) in (0..8).enumerate() {
            assert!(
                matches!(&turns[offset + 1], Message::User { content, .. } if content == &format!("msg {index}")),
                "turn {} is not `msg {index}`",
                offset + 1
            );
        }
    }

    #[test]
    fn compact_preserves_matching_tool_calls_for_preserved_tool_results() {
        let mut history = History::default();
        history.push(user("old msg"));
        for index in 0..3 {
            let call_id = format!("call_{index}");
            history.push(Message::Assistant {
                content:        String::new(),
                tool_calls:     vec![tool_call(
                    &call_id,
                    "read_file",
                    json!({ "file_path": format!("{index}.txt") }),
                )],
                provider_parts: Vec::new(),
                usage:          TokenUsage::default(),
                response_id:    format!("resp_{index}"),
                timestamp:      now(),
            });
            history.push(Message::ToolResults {
                results:   vec![tool_result(&call_id, "ok")],
                timestamp: now(),
            });
        }
        history.push(Message::Assistant {
            content:        String::new(),
            tool_calls:     vec![tool_call(
                "call_3",
                "read_file",
                json!({ "file_path": "3.txt" }),
            )],
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp_3".into(),
            timestamp:      now(),
        });

        history.compact(6, "Summary".into());

        let mut seen_tool_calls: Vec<String> = Vec::new();
        for message in history.to_llm_messages() {
            for part in message.content() {
                match part {
                    ContentPart::ToolCall(call) => seen_tool_calls.push(call.id.clone()),
                    ContentPart::ToolResult(result) => assert!(
                        seen_tool_calls.contains(&result.tool_call_id),
                        "tool result {} has no preserved tool call",
                        result.tool_call_id
                    ),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn compact_noops_when_preserved_tool_result_requires_first_turn() {
        let mut history = History::default();
        history.push(Message::Assistant {
            content:        String::new(),
            tool_calls:     vec![tool_call("call_1", "read_file", json!({}))],
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });
        history.push(Message::ToolResults {
            results:   vec![tool_result("call_1", "ok")],
            timestamp: now(),
        });

        history.compact(1, "Summary".into());

        assert_eq!(history.len(), 2);
        assert!(!matches!(history.turns()[0], Message::System { .. }));
    }

    #[test]
    fn compact_summary_maps_to_system_message() {
        let mut history = History::default();
        for index in 0..6 {
            history.push(user(&format!("msg {index}")));
        }

        history.compact(2, "[Context Summary]\nThis is a summary".into());

        let messages = history.to_llm_messages();
        assert_eq!(messages[0].role(), Role::System);
        assert!(text_of(&messages[0]).contains("[Context Summary]"));
    }

    #[test]
    fn empty_history_produces_empty_messages() {
        let history = History::default();

        assert!(history.to_llm_messages().is_empty());
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn user_turn_maps_to_user_message() {
        let mut history = History::default();
        history.push(user("Hello"));

        let messages = history.to_llm_messages();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), Role::User);
        assert_eq!(text_of(&messages[0]), "Hello");
    }

    #[test]
    fn assistant_turn_maps_to_assistant_message() {
        let mut history = History::default();
        history.push(assistant("Hi there", "resp_1"));

        let messages = history.to_llm_messages();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), Role::Assistant);
        assert_eq!(text_of(&messages[0]), "Hi there");
    }

    #[test]
    fn assistant_turn_with_tool_calls() {
        let mut history = History::default();
        history.push(Message::Assistant {
            content:        "Let me read that".into(),
            tool_calls:     vec![tool_call(
                "call_1",
                "read_file",
                json!({ "path": "foo.rs" }),
            )],
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp_2".into(),
            timestamp:      now(),
        });

        let messages = history.to_llm_messages();

        assert_eq!(messages[0].role(), Role::Assistant);
        let calls = messages[0]
            .content()
            .iter()
            .filter(|part| matches!(part, ContentPart::ToolCall(_)))
            .count();
        assert_eq!(calls, 1);
    }

    #[test]
    fn assistant_turn_with_reasoning_in_provider_parts() {
        let mut history = History::default();
        history.push(Message::Assistant {
            content:        "The answer is 42".into(),
            tool_calls:     Vec::new(),
            provider_parts: vec![thinking_part("Let me think about this...", None)],
            usage:          TokenUsage::default(),
            response_id:    "resp_3".into(),
            timestamp:      now(),
        });

        let messages = history.to_llm_messages();

        let reasoning = messages[0]
            .content()
            .iter()
            .filter(|part| matches!(part, ContentPart::Reasoning(_)))
            .count();
        assert_eq!(reasoning, 1);
    }

    #[test]
    fn reasoning_signature_is_preserved_through_provider_parts() {
        let mut history = History::default();
        history.push(Message::Assistant {
            content:        "The answer".into(),
            tool_calls:     Vec::new(),
            provider_parts: vec![thinking_part("Let me think...", Some("sig_abc123"))],
            usage:          TokenUsage::default(),
            response_id:    "resp_4".into(),
            timestamp:      now(),
        });

        let messages = history.to_llm_messages();

        let signatures: Vec<_> = messages[0]
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::Reasoning(reasoning) => Some(reasoning.signature.as_deref()),
                _ => None,
            })
            .collect();
        assert_eq!(signatures, vec![Some("sig_abc123")]);
    }

    #[test]
    fn assistant_turn_puts_provider_parts_before_tool_calls() {
        let mut history = History::default();
        history.push(Message::Assistant {
            content:        String::new(),
            tool_calls:     vec![tool_call("call_1", "search", json!({}))],
            provider_parts: vec![openai_reasoning_part()],
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });

        let messages = history.to_llm_messages();

        assert_eq!(messages.len(), 1);
        assert!(matches!(
            &messages[0].content()[0],
            ContentPart::Opaque { kind, .. } if kind == "openai.reasoning"
        ));
        assert!(matches!(
            &messages[0].content()[1],
            ContentPart::ToolCall(_)
        ));
    }

    #[test]
    fn tool_results_turn_maps_to_tool_message() {
        let mut history = History::default();
        history.push(Message::ToolResults {
            results:   vec![tool_result("call_1", "file contents here")],
            timestamp: now(),
        });

        let messages = history.to_llm_messages();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), Role::Tool);
        assert_eq!(messages[0].tool_call_id(), Some("call_1"));
    }

    #[test]
    fn system_turn_maps_to_system_message() {
        let mut history = History::default();
        history.push(Message::System {
            content:   "You are a coding assistant".into(),
            timestamp: now(),
        });

        let messages = history.to_llm_messages();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), Role::System);
        assert_eq!(text_of(&messages[0]), "You are a coding assistant");
    }

    #[test]
    fn steering_turn_maps_to_user_message() {
        let mut history = History::default();
        history.push(Message::Steering {
            content:   "Focus on the main task".into(),
            timestamp: now(),
        });

        let messages = history.to_llm_messages();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), Role::User);
        assert_eq!(text_of(&messages[0]), "Focus on the main task");
    }

    #[test]
    fn stored_round_trip_preserves_runtime_history() {
        let mut history = History::default();
        history.push(user("Read a file"));
        history.push(Message::Assistant {
            content:        "Reading".into(),
            tool_calls:     vec![tool_call("call_1", "read_file", json!({ "path": "a.rs" }))],
            provider_parts: vec![thinking_part("weighing it", Some("sig_1"))],
            usage:          TokenUsage {
                input: 10,
                output: 3,
                ..TokenUsage::default()
            },
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });
        history.push(Message::ToolResults {
            results:   vec![tool_result("call_1", "ok")],
            timestamp: now(),
        });

        let restored = History::from_stored_messages(&history.to_stored_messages());

        assert_eq!(restored, history);
        assert_eq!(restored.len(), 3);
        assert!(matches!(
            &restored.turns()[1],
            Message::Assistant { usage, .. } if usage.input == 10 && usage.output == 3
        ));
    }

    #[test]
    fn turns_len_matches_push_count() {
        let mut history = History::default();
        assert_eq!(history.len(), 0);

        history.push(user("First"));
        assert_eq!(history.len(), 1);

        history.push(assistant("Second", "resp_1"));
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn round_trip_preserves_content() {
        let mut history = History::default();
        history.push(user("Hello"));
        history.push(Message::Assistant {
            content:        "Hi".into(),
            tool_calls:     vec![tool_call("c1", "shell", json!({ "cmd": "ls" }))],
            provider_parts: vec![thinking_part("thinking...", None)],
            usage:          TokenUsage {
                input: 10,
                output: 5,
                ..TokenUsage::default()
            },
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });
        history.push(Message::ToolResults {
            results:   vec![tool_result("c1", "file1.rs\nfile2.rs")],
            timestamp: now(),
        });

        let messages = history.to_llm_messages();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role(), Role::User);
        assert_eq!(messages[1].role(), Role::Assistant);
        assert_eq!(messages[2].role(), Role::Tool);
    }

    #[test]
    fn compact_strips_openai_reasoning_from_preserved_turns() {
        let mut history = History::default();
        history.push(user("old msg"));
        history.push(user("recent msg"));
        history.push(Message::Assistant {
            content:        "response".into(),
            tool_calls:     vec![tool_call("call_1", "search", json!({}))],
            provider_parts: vec![openai_reasoning_part()],
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });

        history.compact(2, "Summary".into());

        // Summary, carried-forward "old msg", preserved "recent msg", preserved
        // assistant.
        let Message::Assistant {
            provider_parts,
            tool_calls,
            content,
            ..
        } = &history.turns()[3]
        else {
            panic!("expected an assistant turn");
        };
        assert!(
            provider_parts.is_empty(),
            "reasoning items should be stripped"
        );
        assert_eq!(tool_calls.len(), 1, "tool calls should be preserved");
        assert_eq!(content, "response", "text should be preserved");
    }

    #[test]
    fn compact_strips_the_legacy_openai_reasoning_spelling() {
        let mut history = History::default();
        history.push(user("old msg"));
        history.push(Message::Assistant {
            content:        "response".into(),
            tool_calls:     Vec::new(),
            provider_parts: vec![
                ContentPart::opaque("openai_reasoning", json!({ "id": "rs_1" })),
                ContentPart::opaque("openai_message", json!({ "id": "msg_1" })),
            ],
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });

        history.compact(1, "Summary".into());

        let Message::Assistant { provider_parts, .. } = &history.turns()[2] else {
            panic!("expected an assistant turn");
        };
        assert!(provider_parts.is_empty());
    }

    #[test]
    fn compact_preserves_reasoning_blocks() {
        let mut history = History::default();
        history.push(user("old msg"));
        history.push(user("recent msg"));
        history.push(Message::Assistant {
            content:        "answer".into(),
            tool_calls:     Vec::new(),
            provider_parts: vec![thinking_part("deep thought", Some("sig_xyz"))],
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });

        history.compact(2, "Summary".into());

        let Message::Assistant { provider_parts, .. } = &history.turns()[3] else {
            panic!("expected an assistant turn");
        };
        assert_eq!(
            provider_parts.len(),
            1,
            "reasoning blocks should be preserved"
        );
        assert!(matches!(&provider_parts[0], ContentPart::Reasoning(_)));
    }

    #[test]
    fn compact_preserves_openai_compatible_reasoning_details() {
        let mut history = History::default();
        history.push(user("old msg"));
        history.push(Message::Assistant {
            content:        "answer".into(),
            tool_calls:     Vec::new(),
            provider_parts: vec![ContentPart::opaque(
                "openai_compat_reasoning_details",
                json!([{ "type": "reasoning.text", "text": "kept" }]),
            )],
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });

        history.compact(1, "Summary".into());

        let Message::Assistant { provider_parts, .. } = &history.turns()[2] else {
            panic!("expected an assistant turn");
        };
        assert_eq!(provider_parts.len(), 1);
    }

    #[test]
    fn compact_preserves_assistant_data_but_resets_usage() {
        let mut history = History::default();
        history.push(user("old msg"));
        let call = tool_call("call_1", "search", json!({ "query": "pebble" }));
        let thinking = thinking_part("deep thought", Some("sig_xyz"));
        history.push(Message::Assistant {
            content:        "answer".into(),
            tool_calls:     vec![call.clone()],
            provider_parts: vec![thinking.clone()],
            usage:          TokenUsage {
                input:       10,
                output:      20,
                reasoning:   30,
                cache_read:  40,
                cache_write: 50,
            },
            response_id:    "resp_1".into(),
            timestamp:      now(),
        });

        history.compact(1, "Summary".into());

        let preserved = history
            .turns()
            .iter()
            .find(|turn| matches!(turn, Message::Assistant { .. }))
            .expect("the assistant turn is preserved");
        let Message::Assistant {
            content,
            tool_calls,
            provider_parts,
            usage,
            response_id,
            ..
        } = preserved
        else {
            panic!("expected an assistant turn");
        };
        assert_eq!(content, "answer");
        assert_eq!(tool_calls, &[call]);
        assert_eq!(provider_parts, &[thinking]);
        assert_eq!(response_id, "resp_1");
        assert_eq!(*usage, TokenUsage::default());
    }

    #[test]
    fn compact_strips_reasoning_from_every_preserved_assistant_turn() {
        let mut history = History::default();
        history.push(user("old msg"));
        for index in 0..2 {
            history.push(Message::Assistant {
                content:        format!("response {index}"),
                tool_calls:     Vec::new(),
                provider_parts: vec![ContentPart::opaque(
                    "openai.reasoning",
                    json!({ "type": "reasoning", "id": format!("rs_{index}") }),
                )],
                usage:          TokenUsage::default(),
                response_id:    format!("resp_{index}"),
                timestamp:      now(),
            });
        }

        history.compact(2, "Summary".into());

        for turn in history.turns() {
            if let Message::Assistant { provider_parts, .. } = turn {
                assert!(
                    provider_parts.is_empty(),
                    "every preserved assistant turn should lose its reasoning items"
                );
            }
        }
    }

    #[test]
    fn extract_recent_user_messages_collects_in_conversation_order() {
        let turns = vec![user("first"), assistant("reply", "r1"), user("second")];

        let extracted = extract_recent_user_messages(turns, 20_000);

        assert_eq!(extracted.len(), 2);
        assert!(matches!(&extracted[0], Message::User { content, .. } if content == "first"));
        assert!(matches!(&extracted[1], Message::User { content, .. } if content == "second"));
    }

    #[test]
    fn extract_recent_user_messages_respects_the_token_budget() {
        let turns = vec![user(&"a".repeat(100)), user(&"b".repeat(100))];

        // 30 tokens is 120 characters: the second turn fits, the first does not.
        let extracted = extract_recent_user_messages(turns, 30);

        assert_eq!(extracted.len(), 1);
        assert!(matches!(&extracted[0], Message::User { content, .. } if content.starts_with('b')));
    }

    #[test]
    fn compact_carries_forward_only_user_turns() {
        let mut history = History::default();
        history.push(user("user msg"));
        history.push(assistant("assistant msg", "r1"));
        history.push(user("preserved"));

        history.compact(1, "Summary".into());

        assert_eq!(history.len(), 3);
        assert!(matches!(&history.turns()[0], Message::System { .. }));
        assert!(
            matches!(&history.turns()[1], Message::User { content, .. } if content == "user msg")
        );
        assert!(
            matches!(&history.turns()[2], Message::User { content, .. } if content == "preserved")
        );
    }

    #[test]
    fn compact_from_ignores_an_out_of_range_boundary() {
        let mut history = History::default();
        history.push(user("only"));

        history.compact_from(0, "Summary".into());
        history.compact_from(9, "Summary".into());

        assert_eq!(history.len(), 1);
    }
}
