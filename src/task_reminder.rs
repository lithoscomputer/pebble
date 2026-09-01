//! Reminding a model that it has task tools it has stopped using.
//!
//! A session that tracks its work in tasks stays legible to the person watching
//! it, but a model drifts away from the task tools over a long prompt. The
//! reminder is one system turn, injected only when the session actually has
//! both task tools, only after ten assistant turns without either, and never
//! more often than once every ten assistant turns.
//!
//! The turn is staged rather than committed: the loop builds it into the
//! request and keeps it only when the round survives, so an interrupted round
//! does not leave a reminder behind in history.

use crate::history::History;
use crate::tool::{NativeTool, canonical_tool_name};
use crate::types::Message;

/// How many assistant turns without a task tool trigger the reminder, and how
/// many must pass before it is sent again.
const TURN_THRESHOLD: usize = 10;

/// The reminder pebble injects, exactly as history records it.
///
/// [`maybe_task_reminder`] recognizes an earlier reminder by comparing a system
/// turn's trimmed text to this, so the text is part of the module's behavior
/// rather than decoration.
pub const TASK_REMINDER_TEXT: &str = "\
<system-reminder>
TaskCreate and TaskUpdate are available but have not been used in the last 10 assistant turns. For multi-step work, create tasks with TaskCreate and keep progress current with TaskUpdate.
</system-reminder>";

/// The reminder to inject before the next turn, when one is due.
///
/// `available_tool_names` are the tools the session currently exposes, named as
/// the model sees them. Both task tools must be there: a reminder to use a tool
/// that is not registered would send the model after something it cannot call.
///
/// ```
/// # use pebble::resources::{History, maybe_task_reminder};
/// let history = History::default();
/// assert!(maybe_task_reminder(&history, &["TaskCreate", "TaskUpdate"]).is_none());
/// ```
#[must_use]
pub fn maybe_task_reminder(history: &History, available_tool_names: &[&str]) -> Option<String> {
    if !task_tools_available(available_tool_names) {
        return None;
    }

    let counts = turn_counts(history);
    (counts.since_task_tool >= TURN_THRESHOLD && counts.since_reminder >= TURN_THRESHOLD)
        .then(|| TASK_REMINDER_TEXT.to_owned())
}

/// Whether both task tools are exposed, under any vocabulary's spelling.
fn task_tools_available(tool_names: &[&str]) -> bool {
    [NativeTool::TaskCreate, NativeTool::TaskUpdate]
        .into_iter()
        .all(|tool| {
            tool_names
                .iter()
                .any(|name| canonical_tool_name(name) == tool.canonical_name())
        })
}

/// Whether a tool call is one of the two task tools.
fn is_task_tool(name: &str) -> bool {
    matches!(
        NativeTool::from_any_name(name),
        Some(NativeTool::TaskCreate | NativeTool::TaskUpdate)
    )
}

/// How long the session has gone without each of the two things the reminder
/// waits on.
#[derive(Debug, Clone, Copy, Default)]
struct TurnCounts {
    /// Assistant turns since the last task-tool call.
    since_task_tool: usize,
    /// Assistant turns since the last reminder.
    since_reminder:  usize,
}

/// Counts assistant turns back to the last task-tool call and the last
/// reminder, stopping as soon as both are found.
fn turn_counts(history: &History) -> TurnCounts {
    let mut found_task_tool = false;
    let mut found_reminder = false;
    let mut counts = TurnCounts::default();

    for turn in history.turns().iter().rev() {
        match turn {
            Message::Assistant { tool_calls, .. } => {
                if !found_task_tool && tool_calls.iter().any(|call| is_task_tool(&call.name)) {
                    found_task_tool = true;
                }

                if !found_task_tool {
                    counts.since_task_tool += 1;
                }
                if !found_reminder {
                    counts.since_reminder += 1;
                }
            }
            Message::System { content, .. } if !found_reminder && is_reminder(content) => {
                found_reminder = true;
            }
            _ => {}
        }

        if found_task_tool && found_reminder {
            break;
        }
    }

    counts
}

/// Whether a system turn is a reminder pebble injected earlier.
fn is_reminder(content: &str) -> bool {
    content.trim() == TASK_REMINDER_TEXT
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use lithos_llm::types::ToolCall;
    use serde_json::json;

    use super::*;
    use crate::session::testing::history_from;
    use crate::types::TokenUsage;

    fn assistant(tool_name: Option<&str>) -> Message {
        Message::Assistant {
            content:        String::new(),
            tool_calls:     tool_name
                .map(|name| vec![ToolCall::function("call_1", name, json!({}))])
                .unwrap_or_default(),
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp".into(),
            timestamp:      SystemTime::now(),
        }
    }

    fn system(content: &str) -> Message {
        Message::System {
            content:   content.into(),
            timestamp: SystemTime::now(),
        }
    }

    const BOTH_TOOLS: [&str; 2] = ["TaskCreate", "TaskUpdate"];

    #[test]
    fn ten_assistant_turns_without_a_task_tool_earn_a_reminder() {
        let history = history_from((0..10).map(|_| assistant(None)).collect());

        assert_eq!(
            maybe_task_reminder(&history, &BOTH_TOOLS).as_deref(),
            Some(TASK_REMINDER_TEXT)
        );
    }

    #[test]
    fn nine_turns_are_not_enough() {
        let history = history_from((0..9).map(|_| assistant(None)).collect());

        assert!(maybe_task_reminder(&history, &BOTH_TOOLS).is_none());
    }

    #[test]
    fn a_reminder_starts_its_own_ten_turn_cooldown() {
        let mut turns = vec![system(TASK_REMINDER_TEXT)];
        turns.extend((0..9).map(|_| assistant(None)));
        assert!(maybe_task_reminder(&history_from(turns), &BOTH_TOOLS).is_none());

        let mut turns = vec![system(TASK_REMINDER_TEXT)];
        turns.extend((0..10).map(|_| assistant(None)));
        assert!(maybe_task_reminder(&history_from(turns), &BOTH_TOOLS).is_some());
    }

    #[test]
    fn a_session_without_both_task_tools_is_never_reminded() {
        let history = history_from((0..10).map(|_| assistant(None)).collect());

        assert!(maybe_task_reminder(&history, &["TaskCreate"]).is_none());
        assert!(maybe_task_reminder(&history, &["TaskUpdate"]).is_none());
        assert!(maybe_task_reminder(&history, &["TaskList", "TaskGet"]).is_none());
        assert!(maybe_task_reminder(&history, &[]).is_none());
    }

    #[test]
    fn either_task_tool_resets_the_count() {
        for tool_name in ["TaskCreate", "TaskUpdate"] {
            let mut turns: Vec<Message> = (0..10).map(|_| assistant(None)).collect();
            turns.push(assistant(Some(tool_name)));
            turns.extend((0..9).map(|_| assistant(None)));

            assert!(
                maybe_task_reminder(&history_from(turns), &BOTH_TOOLS).is_none(),
                "{tool_name} should reset the reminder count"
            );
        }
    }

    #[test]
    fn a_system_turn_that_is_not_a_reminder_does_not_reset_the_cooldown() {
        let mut turns = vec![system("some other injected note")];
        turns.extend((0..10).map(|_| assistant(None)));

        assert!(maybe_task_reminder(&history_from(turns), &BOTH_TOOLS).is_some());
    }
}
