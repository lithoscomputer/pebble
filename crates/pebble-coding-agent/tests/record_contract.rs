//! Contract tests for the stored session record.
//!
//! The snapshot and fixture pin the current record format.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lithos_llm::types::{
    ContentPart, Message as LlmMessage, ReasoningContent, Role, ToolArguments, ToolCall, ToolInput,
    ToolResult,
};
use pebble_agent::{SessionId, SessionScope};
use pebble_coding_agent::events::{CompactionReason, TokenUsage};
use pebble_coding_agent::state::{
    History, Message, SESSION_RECORD_FORMAT_VERSION, SessionRecord, StoredMessage,
};
use serde_json::json;

/// A record with complete ancestry and typed tool input.
const SAMPLE_RECORD: &str = include_str!("fixtures/session_record_v4.json");

fn moment() -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(1_767_225_600_500)
}

fn usage() -> TokenUsage {
    TokenUsage {
        input:       1_200,
        output:      340,
        reasoning:   96,
        cache_read:  800,
        cache_write: 64,
    }
}

fn tool_call() -> ToolCall {
    ToolCall {
        id:                "call_1".into(),
        name:              "read_file".into(),
        input:             ToolInput::Function(ToolArguments::from_json(
            json!({ "path": "src/lib.rs" }),
        )),
        provider_metadata: BTreeMap::from([("openai".to_owned(), json!({ "item_id": "fc_1" }))]),
    }
}

fn every_turn() -> Vec<Message> {
    vec![
        Message::User {
            content:   "read the crate root".into(),
            timestamp: moment(),
        },
        Message::Assistant {
            content:        "Reading it now.".into(),
            tool_calls:     vec![tool_call()],
            provider_parts: vec![
                ContentPart::Reasoning(ReasoningContent {
                    text:             "the root is a facade".into(),
                    signature:        Some("sig_1".into()),
                    signature_origin: Some("anthropic".into()),
                    redacted:         false,
                }),
                ContentPart::opaque("openai.reasoning", json!({ "id": "rs_1" })),
            ],
            usage:          usage(),
            response_id:    "resp_1".into(),
            timestamp:      moment(),
        },
        Message::ToolResults {
            results:   vec![ToolResult {
                tool_call_id: "call_1".into(),
                name:         Some("read_file".into()),
                content:      vec![ContentPart::Text {
                    text: "//! Pebble is a coding-agent loop library.".into(),
                }],
                is_error:     false,
            }],
            timestamp: moment(),
        },
        Message::Compaction {
            summary:                 "[Context Summary]\nThe session read the crate root.".into(),
            reason:                  CompactionReason::Manual,
            original_turn_count:     8,
            preserved_turn_count:    2,
            estimated_tokens_before: 12_000,
            summary_token_estimate:  24,
            tracked_file_count:      1,
            summary_truncated:       false,
            usage:                   usage(),
            cost_usd_micros:         Some(1_250),
            timestamp:               moment(),
        },
        Message::Steering {
            content:   "also update the changelog".into(),
            timestamp: moment(),
        },
    ]
}

fn sample_record() -> SessionRecord {
    let mut record = SessionRecord::new(SessionScope::root(SessionId::new("ses_root")));
    record.provider = Some("anthropic".into());
    record.model = Some("claude-sonnet-5".into());
    record.created_at = moment();
    record.updated_at = moment();
    record.last_event_seq = 41;
    record.messages = every_turn()
        .iter()
        .map(Message::to_stored_message)
        .collect();
    record
}

#[test]
fn the_session_record_keeps_its_serialized_shape() {
    let rendered = serde_json::to_string_pretty(&sample_record()).expect("record serializes");
    insta::assert_snapshot!("session_record", rendered);
}

#[test]
fn the_stored_record_restores_history_and_accounting() {
    let record: SessionRecord =
        serde_json::from_str(SAMPLE_RECORD).expect("the current record parses");

    assert_eq!(record.format_version, SESSION_RECORD_FORMAT_VERSION);
    assert!(record.is_supported());
    assert_eq!(record.scope.session_id().as_str(), "ses_root");
    assert_eq!(record.last_event_seq, 41);

    let history = History::from_stored_messages(&record.messages);
    assert_eq!(history.len(), 5);

    // The assistant turn restores its exact accounting, not a default.
    let Message::Assistant {
        usage: restored_usage,
        tool_calls,
        provider_parts,
        response_id,
        ..
    } = &history.turns()[1]
    else {
        panic!("expected an assistant turn");
    };
    assert_eq!(*restored_usage, usage());
    assert_eq!(tool_calls, &[tool_call()]);
    assert_eq!(provider_parts.len(), 2);
    assert_eq!(response_id, "resp_1");

    // The restored conversation still replays as a valid exchange.
    let messages = history.to_llm_messages();
    let roles: Vec<Role> = messages.iter().map(LlmMessage::role).collect();
    assert_eq!(roles, vec![
        Role::User,
        Role::Assistant,
        Role::Tool,
        Role::System,
        Role::User,
    ]);
    assert_eq!(messages[2].tool_call_id(), Some("call_1"));
}

#[test]
fn a_record_with_unknown_members_still_parses() {
    let mut document: serde_json::Value =
        serde_json::from_str(SAMPLE_RECORD).expect("the fixture parses as JSON");
    let object = document.as_object_mut().expect("a record is an object");
    object.insert("workspace".to_owned(), json!("/work/pebble"));
    object.insert("future_field".to_owned(), json!({ "nested": [1, 2, 3] }));
    let messages = object
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
        .expect("a record has messages");
    for message in messages.iter_mut() {
        let member = message.as_object_mut().expect("a message is an object");
        member.insert("token_estimate".to_owned(), json!(42));
    }

    let record: SessionRecord =
        serde_json::from_value(document).expect("unknown members are ignored");
    let expected: SessionRecord =
        serde_json::from_str(SAMPLE_RECORD).expect("the fixture still parses");

    assert_eq!(record, expected);
}

#[test]
fn a_record_missing_its_optional_members_still_parses() {
    let record: SessionRecord = serde_json::from_value(json!({
        "format_version": SESSION_RECORD_FORMAT_VERSION,
        "scope": SessionScope::root(SessionId::new("ses_root")),
        "created_at": "2026-01-01T00:00:00.500Z",
        "updated_at": "2026-01-01T00:00:00.500Z",
    }))
    .expect("a minimal record parses");

    assert_eq!(record.format_version, SESSION_RECORD_FORMAT_VERSION);
    assert_eq!(record.last_event_seq, 0);
    assert!(record.messages.is_empty());
    assert!(History::from_stored_messages(&record.messages).is_empty());
}

#[test]
fn a_record_whose_members_are_null_still_resumes() {
    let record: SessionRecord = serde_json::from_value(json!({
        "format_version": SESSION_RECORD_FORMAT_VERSION,
        "scope": SessionScope::root(SessionId::new("ses_root")),
        "created_at": "2026-01-01T00:00:00.500Z",
        "updated_at": "2026-01-01T00:00:00.500Z",
        "last_event_seq": null,
        "messages": [
            {
                "kind": "assistant",
                "content": "hello",
                "tool_calls": null,
                "provider_parts": null,
                "usage": null,
                "response_id": "resp_1",
                "timestamp": "2026-01-01T00:00:00.500Z",
            },
        ],
    }))
    .expect("null members read as their defaults");

    assert_eq!(record.last_event_seq, 0);
    assert_eq!(record.messages, vec![StoredMessage::Assistant {
        content:        "hello".into(),
        tool_calls:     Vec::new(),
        provider_parts: Vec::new(),
        usage:          TokenUsage::default(),
        response_id:    "resp_1".into(),
        timestamp:      moment(),
    }]);
}

#[test]
fn a_record_newer_than_this_build_is_refused() {
    let mut record = sample_record();
    record.format_version = SESSION_RECORD_FORMAT_VERSION + 1;

    let json = serde_json::to_string(&record).expect("record serializes");
    let restored: SessionRecord = serde_json::from_str(&json).expect("record parses");

    assert!(
        !restored.is_supported(),
        "a record from a newer format must not be treated as resumable"
    );
}
