//! Noticing that a session is repeating itself.
//!
//! A model that has stopped making progress usually says so by asking for the
//! same tool calls again: the same command, the same file, the same search,
//! round after round. [`detect_loop`] reduces each assistant turn that
//! requested tools to one signature and looks for a short cycle in the recent
//! ones.
//!
//! It is deliberately cheap and deliberately conservative. Only assistant turns
//! that requested tools count, so ordinary conversation never triggers it, and
//! a cycle has to repeat completely — every group in the window must match —
//! before the session is told about it.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde_json::Value;

use crate::history::History;
use crate::types::Message;

/// The longest cycle [`detect_loop`] looks for, in turns.
const MAX_PATTERN_LEN: usize = 3;

/// Whether the last `window_size` tool-calling turns repeat a short cycle.
///
/// Walks back over the turns that asked for tools — at most `window_size` of
/// them — and answers `true` when they are a cycle of one, two, or three turns
/// repeated to the end of the window. A tool call's signature is its name and
/// its arguments together, so the same tool with different arguments is
/// progress rather than a loop.
///
/// Fewer than two tool-calling turns is never a loop.
///
/// ```ignore
/// # use pebble_coding_agent::resources::{History, detect_loop};
/// let history = History::default();
/// assert!(!detect_loop(&history, 10));
/// ```
#[must_use]
pub(crate) fn detect_loop(history: &History, window_size: usize) -> bool {
    let signatures = recent_turn_signatures(history, window_size);

    if signatures.len() < 2 {
        return false;
    }

    (1..=MAX_PATTERN_LEN).any(|pattern_len| is_repeating_pattern(&signatures, pattern_len))
}

/// The signatures of the most recent tool-calling turns, oldest first.
fn recent_turn_signatures(history: &History, window_size: usize) -> Vec<u64> {
    let mut signatures = Vec::new();

    for turn in history.turns().iter().rev() {
        if signatures.len() >= window_size {
            break;
        }
        if let Some(signature) = turn_signature(turn) {
            signatures.push(signature);
        }
    }

    signatures.reverse();
    signatures
}

/// One assistant turn's tool calls reduced to a single signature.
///
/// Turns that requested no tools have none, so they are skipped rather than
/// counted as a repeat of each other.
fn turn_signature(turn: &Message) -> Option<u64> {
    let Message::Assistant { tool_calls, .. } = turn else {
        return None;
    };
    if tool_calls.is_empty() {
        return None;
    }

    let mut hasher = DefaultHasher::new();
    for call in tool_calls {
        tool_call_signature(&call.name, &call.arguments).hash(&mut hasher);
    }
    Some(hasher.finish())
}

/// One tool call reduced to a signature over its name and arguments.
fn tool_call_signature(name: &str, arguments: &Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    arguments.to_string().hash(&mut hasher);
    hasher.finish()
}

/// Whether every complete group of `pattern_len` signatures in the window
/// equals the last one.
fn is_repeating_pattern(signatures: &[u64], pattern_len: usize) -> bool {
    if signatures.len() < pattern_len * 2 {
        return false;
    }

    let pattern = &signatures[signatures.len() - pattern_len..];
    let group_count = signatures.len() / pattern_len;
    let groups_start = signatures.len() - group_count * pattern_len;

    signatures[groups_start..]
        .chunks_exact(pattern_len)
        .all(|group| group == pattern)
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use lithos_llm::types::ToolCall;
    use serde_json::json;

    use super::*;
    use crate::runtime::testing::history_from;
    use crate::types::TokenUsage;

    fn assistant_with_tool(name: &str, arguments: Value) -> Message {
        Message::Assistant {
            content:        String::new(),
            tool_calls:     vec![ToolCall::function("call_1", name, arguments)],
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp".into(),
            timestamp:      SystemTime::now(),
        }
    }

    #[test]
    fn an_empty_history_is_not_a_loop() {
        assert!(!detect_loop(&History::default(), 10));
    }

    #[test]
    fn one_tool_calling_turn_is_not_a_loop() {
        let history = history_from(vec![assistant_with_tool("shell", json!({ "cmd": "ls" }))]);

        assert!(!detect_loop(&history, 10));
    }

    #[test]
    fn one_repeated_turn_is_a_loop() {
        let history = history_from(
            (0..3)
                .map(|_| assistant_with_tool("shell", json!({ "cmd": "ls" })))
                .collect(),
        );

        assert!(detect_loop(&history, 10));
    }

    #[test]
    fn a_two_turn_cycle_is_a_loop() {
        let history = history_from(vec![
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
            assistant_with_tool("read_file", json!({ "path": "foo.rs" })),
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
            assistant_with_tool("read_file", json!({ "path": "foo.rs" })),
        ]);

        assert!(detect_loop(&history, 10));
    }

    #[test]
    fn a_three_turn_cycle_is_a_loop() {
        let history = history_from(vec![
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
            assistant_with_tool("read_file", json!({ "path": "a.rs" })),
            assistant_with_tool("grep", json!({ "pattern": "fn" })),
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
            assistant_with_tool("read_file", json!({ "path": "a.rs" })),
            assistant_with_tool("grep", json!({ "pattern": "fn" })),
        ]);

        assert!(detect_loop(&history, 10));
    }

    #[test]
    fn turns_that_do_not_repeat_are_not_a_loop() {
        let history = history_from(vec![
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
            assistant_with_tool("read_file", json!({ "path": "a.rs" })),
            assistant_with_tool("grep", json!({ "pattern": "fn" })),
            assistant_with_tool("shell", json!({ "cmd": "cat" })),
        ]);

        assert!(!detect_loop(&history, 10));
    }

    #[test]
    fn the_same_tool_with_different_arguments_is_progress() {
        let history = history_from(vec![
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
            assistant_with_tool("shell", json!({ "cmd": "pwd" })),
            assistant_with_tool("shell", json!({ "cmd": "cat" })),
        ]);

        assert!(!detect_loop(&history, 10));
    }

    #[test]
    fn the_same_call_signs_the_same_way() {
        assert_eq!(
            tool_call_signature("shell", &json!({ "cmd": "ls" })),
            tool_call_signature("shell", &json!({ "cmd": "ls" }))
        );
    }

    #[test]
    fn a_different_name_signs_differently() {
        assert_ne!(
            tool_call_signature("shell", &json!({ "cmd": "ls" })),
            tool_call_signature("read_file", &json!({ "cmd": "ls" }))
        );
    }

    #[test]
    fn different_arguments_sign_differently() {
        assert_ne!(
            tool_call_signature("shell", &json!({ "cmd": "ls" })),
            tool_call_signature("shell", &json!({ "cmd": "pwd" }))
        );
    }

    #[test]
    fn turns_without_tool_calls_are_ignored() {
        let history = history_from(
            (0..3)
                .map(|_| Message::User {
                    content:   "hello".into(),
                    timestamp: SystemTime::now(),
                })
                .collect(),
        );

        assert!(!detect_loop(&history, 10));
    }

    #[test]
    fn the_window_bounds_how_far_back_the_detector_looks() {
        let history = history_from(vec![
            assistant_with_tool("shell", json!({ "cmd": "unique1" })),
            assistant_with_tool("shell", json!({ "cmd": "unique2" })),
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
            assistant_with_tool("shell", json!({ "cmd": "ls" })),
        ]);

        // The last two turns repeat; the two before them do not.
        assert!(detect_loop(&history, 2));
        assert!(!detect_loop(&history, 4));
    }

    #[test]
    fn a_partial_cycle_is_not_a_loop() {
        // A-B-C-A-B: the trailing pair repeats, but the window as a whole is
        // not a completed cycle at any length.
        let history = history_from(vec![
            assistant_with_tool("shell", json!({ "cmd": "a" })),
            assistant_with_tool("shell", json!({ "cmd": "b" })),
            assistant_with_tool("shell", json!({ "cmd": "c" })),
            assistant_with_tool("shell", json!({ "cmd": "a" })),
            assistant_with_tool("shell", json!({ "cmd": "b" })),
        ]);

        assert!(!detect_loop(&history, 10));
    }
}
