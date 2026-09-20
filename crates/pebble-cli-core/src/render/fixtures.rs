//! Events as the stream tells them, for the render tests.

use std::time::SystemTime;

use pebble_coding_agent::events::{
    CodingAgentEvent, CodingEvent, Cost, CostSource, ErrorData, ErrorKind, InputSource,
    LlmRetryPhase, ReasoningOutput, TokenCounts, Usage,
};
use serde_json::{Value, json};

pub(super) fn root(event: CodingEvent) -> CodingAgentEvent {
    CodingAgentEvent::new("ses_root", event, SystemTime::UNIX_EPOCH)
}

pub(super) fn child(event: CodingEvent) -> CodingAgentEvent {
    CodingAgentEvent::new("ses_child", event, SystemTime::UNIX_EPOCH)
        .with_parent_session_id("ses_root")
}

/// `input` and `output` tokens, priced from the catalog when `cost` is
/// given.
pub(super) fn priced(input: u64, output: u64, cost: Option<u64>) -> Usage {
    Usage {
        tokens: TokenCounts {
            input,
            output,
            ..TokenCounts::default()
        },
        cost:   cost.map(|usd_micros| Cost {
            usd_micros,
            source: CostSource::Catalog,
        }),
    }
}

pub(super) fn message(input: u64, output: u64, cost: Option<u64>) -> CodingEvent {
    CodingEvent::AssistantMessage {
        text:            "ok".into(),
        model:           "model".into(),
        usage:           priced(input, output, cost),
        tool_call_count: 0,
        context_window:  None,
        reasoning:       None,
    }
}

pub(super) fn tool_started(tool_name: &str, tool_call_id: &str) -> CodingEvent {
    CodingEvent::ToolCallStarted {
        tool_name:    tool_name.into(),
        tool_call_id: tool_call_id.into(),
        arguments:    json!({"path": "/w/a.txt"}),
    }
}

pub(super) fn tool_completed(tool_name: &str, tool_call_id: &str, is_error: bool) -> CodingEvent {
    tool_answered(tool_name, tool_call_id, json!("done"), is_error, 0)
}

pub(super) fn tool_answered(
    tool_name: &str,
    tool_call_id: &str,
    output: Value,
    is_error: bool,
    output_bytes_omitted: usize,
) -> CodingEvent {
    CodingEvent::ToolCallCompleted {
        tool_name: tool_name.into(),
        tool_call_id: tool_call_id.into(),
        output,
        metadata: pebble_agent::ToolOutputMetadata::default(),
        is_error,
        error_kind: None,
        output_bytes_observed: 0,
        output_bytes_retained: 0,
        output_bytes_omitted,
    }
}

pub(super) fn ask() -> CodingEvent {
    CodingEvent::LlmRequestStarted {
        requested_model: "model".into(),
    }
}

pub(super) fn delta(text: &str) -> CodingEvent {
    CodingEvent::TextDelta { delta: text.into() }
}

/// A committed turn with `text`, and its reasoning when the route carried
/// one.
pub(super) fn turn(text: &str, reasoning: Option<ReasoningOutput>) -> CodingEvent {
    CodingEvent::AssistantMessage {
        text: text.into(),
        model: "model".into(),
        usage: priced(10, 5, None),
        tool_call_count: 0,
        context_window: None,
        reasoning,
    }
}

pub(super) fn retry() -> CodingEvent {
    CodingEvent::LlmRetry {
        provider:   "test".into(),
        model:      "model".into(),
        attempt:    0,
        delay_secs: 0.1,
        error:      ErrorData::new(ErrorKind::Llm, "slow down"),
        phase:      LlmRetryPhase::Open,
    }
}

pub(super) fn prompt(text: &str) -> CodingEvent {
    CodingEvent::UserInput {
        text:    text.into(),
        content: None,
        source:  InputSource::Prompt,
    }
}

/// One prompt as the stream tells it: the root answers twice with a retry
/// before the first, a child answers once and fails a tool call, and two of
/// the root's tool calls succeed.
pub(super) fn scripted_prompt() -> Vec<CodingAgentEvent> {
    vec![
        root(prompt("go")),
        root(tool_started("read_file", "r1")),
        root(tool_completed("read_file", "r1", false)),
        root(retry()),
        root(message(10, 5, Some(100))),
        root(CodingEvent::SubAgentSpawned {
            agent_id:   "a1".into(),
            depth:      1,
            task:       "look".into(),
            generation: 1,
        }),
        child(tool_started("read_file", "r2")),
        child(tool_completed("read_file", "r2", true)),
        child(message(7, 3, Some(400))),
        root(CodingEvent::SubAgentCompleted {
            agent_id:   "a1".into(),
            depth:      1,
            generation: 1,
            success:    true,
            turns_used: 1,
        }),
        root(tool_started("edit_file", "e1")),
        root(tool_completed("edit_file", "e1", false)),
        root(message(20, 5, Some(200))),
        root(CodingEvent::ProcessingEnd),
        root(CodingEvent::SessionEnded),
    ]
}

/// A tool round as the stream tells it: the model asks for a tool while
/// answering, the tool answers, and the model answers again.
pub(super) fn scripted_tool_round() -> Vec<CodingAgentEvent> {
    vec![
        root(ask()),
        root(delta("Reading.")),
        root(turn("Reading.", None)),
        root(tool_started("read_file", "r1")),
        root(tool_answered(
            "read_file",
            "r1",
            json!("one\ntwo\n"),
            false,
            0,
        )),
        root(tool_started("edit_file", "e1")),
        root(tool_answered(
            "edit_file",
            "e1",
            json!("no such anchor"),
            true,
            0,
        )),
        root(ask()),
        root(delta("Done.")),
        root(turn("Done.", None)),
        root(CodingEvent::ProcessingEnd),
        root(CodingEvent::SessionEnded),
    ]
}
