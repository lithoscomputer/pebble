//! Round-trip tests for the two translations a history turn goes through.
//!
//! A [`Message`] is translated twice: into the wire form a provider receives
//! ([`Message::to_llm_message`]) and into the stored form a record keeps
//! ([`Message::to_stored_message`]). Both run on the loop's commit path, so
//! each variant is covered here, with the richest content pebble ever
//! preserves — reasoning that carries a signature, provider-namespaced opaque
//! parts, and a tool call whose raw argument text must survive verbatim.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lithos_llm::types::{
    ContentPart, Message as LlmMessage, ReasoningContent, Role, ToolCall, ToolCallKind, ToolResult,
};
use pebble_coding_agent::events::TokenUsage;
use pebble_coding_agent::state::{History, Message};
use serde_json::json;

fn moment() -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(1_767_225_600_500)
}

/// The name of a turn's variant.
///
/// `Message` is non-exhaustive, so this arm is what a reader outside pebble
/// has to write. A turn kind this file does not know reaches it and fails the
/// run, which is the reminder to cover the new kind in `every_variant`.
fn variant_of(turn: &Message) -> &'static str {
    match turn {
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
        Message::ToolResults { .. } => "tool_results",
        Message::System { .. } => "system",
        Message::Steering { .. } => "steering",
        _ => panic!("a turn kind this test does not know: add it to `every_variant`"),
    }
}

fn tool_call() -> ToolCall {
    ToolCall {
        id:                "call_1".into(),
        name:              "shell".into(),
        arguments:         json!({ "command": "cargo test" }),
        kind:              ToolCallKind::Function,
        // Codecs replay this text rather than re-serializing `arguments`, so a
        // lost or reordered copy breaks provider prompt caching.
        raw_arguments:     Some("{\"command\":\"cargo test\"}".into()),
        provider_metadata: BTreeMap::from([
            ("openai".to_owned(), json!({ "id": "fc_1" })),
            ("anthropic".to_owned(), json!({ "id": "toolu_1" })),
        ]),
    }
}

fn every_variant() -> Vec<Message> {
    vec![
        Message::User {
            content:   "fix the failing test".into(),
            timestamp: moment(),
        },
        Message::Assistant {
            content:        "Running the suite.".into(),
            tool_calls:     vec![tool_call()],
            provider_parts: vec![
                ContentPart::Reasoning(ReasoningContent {
                    text:             "the assertion is the place to start".into(),
                    signature:        Some("sig_1".into()),
                    signature_origin: Some("anthropic".into()),
                    redacted:         false,
                }),
                ContentPart::opaque("openai.reasoning", json!({ "id": "rs_1" })),
                ContentPart::opaque("openai.message", json!({ "id": "msg_1" })),
            ],
            usage:          TokenUsage {
                input:       1_200,
                output:      340,
                reasoning:   96,
                cache_read:  800,
                cache_write: 64,
            },
            response_id:    "resp_1".into(),
            timestamp:      moment(),
        },
        Message::ToolResults {
            results:   vec![
                ToolResult {
                    tool_call_id: "call_1".into(),
                    name:         Some("shell".into()),
                    content:      vec![ContentPart::Text {
                        text: "test result: FAILED".into(),
                    }],
                    is_error:     true,
                },
                ToolResult {
                    tool_call_id: "call_2".into(),
                    name:         None,
                    content:      vec![ContentPart::Json {
                        value: json!({ "files": ["src/lib.rs"] }),
                    }],
                    is_error:     false,
                },
            ],
            timestamp: moment(),
        },
        Message::System {
            content:   "[Context Summary]\nThe suite was failing.".into(),
            timestamp: moment(),
        },
        Message::Steering {
            content:   "also update the changelog".into(),
            timestamp: moment(),
        },
    ]
}

#[test]
fn every_turn_kind_is_covered() {
    let covered: Vec<&str> = every_variant().iter().map(variant_of).collect();

    assert_eq!(covered, vec![
        "user",
        "assistant",
        "tool_results",
        "system",
        "steering",
    ]);
}

#[test]
fn every_turn_survives_the_wire_translation() {
    for turn in every_variant() {
        let message = turn.to_llm_message();

        let json = serde_json::to_string(&message).expect("a wire message serializes");
        let restored: LlmMessage = serde_json::from_str(&json).expect("a wire message parses");

        assert_eq!(
            restored,
            message,
            "the wire form of a {} turn did not survive a round trip",
            variant_of(&turn)
        );
    }
}

#[test]
fn every_turn_survives_the_stored_translation() {
    for turn in every_variant() {
        let stored = turn.to_stored_message();

        let json = serde_json::to_string(&stored).expect("a stored turn serializes");
        let restored = Message::from_stored_message(
            &serde_json::from_str(&json).expect("a stored turn parses"),
        );

        assert_eq!(
            restored,
            turn,
            "a {} turn did not survive the stored round trip",
            variant_of(&turn)
        );
    }
}

#[test]
fn a_stored_turn_produces_the_same_wire_message_as_the_turn_it_came_from() {
    for turn in every_variant() {
        let restored = Message::from_stored_message(&turn.to_stored_message());

        assert_eq!(
            restored.to_llm_message(),
            turn.to_llm_message(),
            "a resumed {} turn would be sent differently than the original",
            variant_of(&turn)
        );
    }
}

#[test]
fn an_assistant_turn_replays_its_parts_in_provider_order() {
    let history = History::from_stored_messages(
        &every_variant()
            .iter()
            .map(Message::to_stored_message)
            .collect::<Vec<_>>(),
    );

    let messages = history.to_llm_messages();
    let assistant = &messages[1];

    assert_eq!(assistant.role(), Role::Assistant);
    // Provider-native parts first, then the text, then the tool calls: any
    // other order breaks replay for the reasoning-carrying providers.
    assert!(matches!(assistant.content(), [
        ContentPart::Reasoning(_),
        ContentPart::Opaque { .. },
        ContentPart::Opaque { .. },
        ContentPart::Text { .. },
        ContentPart::ToolCall(_),
    ]));

    let ContentPart::ToolCall(call) = &assistant.content()[4] else {
        panic!("expected the tool call last");
    };
    assert_eq!(
        call.raw_arguments.as_deref(),
        Some("{\"command\":\"cargo test\"}")
    );
    assert_eq!(call.provider_metadata.len(), 2);

    let ContentPart::Reasoning(reasoning) = &assistant.content()[0] else {
        panic!("expected reasoning first");
    };
    assert_eq!(reasoning.signature.as_deref(), Some("sig_1"));
    assert_eq!(reasoning.signature_origin.as_deref(), Some("anthropic"));
}

#[test]
fn a_tool_results_turn_keeps_every_result_and_its_error_flag() {
    let turn = &every_variant()[2];

    let message = turn.to_llm_message();

    assert_eq!(message.role(), Role::Tool);
    assert_eq!(message.tool_call_id(), Some("call_1"));
    let flags: Vec<bool> = message
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result.is_error),
            _ => None,
        })
        .collect();
    assert_eq!(flags, vec![true, false]);
}

#[test]
fn steering_reaches_the_model_as_user_input_but_stays_its_own_turn() {
    let steering = &every_variant()[4];

    assert_eq!(steering.to_llm_message().role(), Role::User);
    assert!(
        matches!(
            Message::from_stored_message(&steering.to_stored_message()),
            Message::Steering { .. }
        ),
        "steering must not resume as ordinary user input"
    );
}
