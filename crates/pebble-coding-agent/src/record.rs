//! The stored form of a session: what pebble writes so a conversation can be
//! resumed later.
//!
//! A [`SessionRecord`] is the whole durable state of one session — its
//! identity, the model it ran against, where its event stream left off, and
//! its conversation as [`StoredMessage`] values. The application decides where
//! records live; pebble only owns their shape.
//!
//! # Stability
//!
//! The serialized form is public API and carries a
//! [`format_version`](SessionRecord::format_version). This build requires the
//! current format. Ancestry must be explicit; older identities are not
//! inferred.

use std::time::SystemTime;

mod tool_call;

use lithos_llm::types::{ContentPart, ToolCall, ToolResult};
use serde::{Deserialize, Deserializer, Serialize};

use crate::SessionScope;
use crate::compaction::CompactionReason;
use crate::types::{InputContent, TokenUsage, rfc3339_millis};

/// The record format version this build writes.
///
/// Version 4 stores the canonical session scope, including parent and depth.
pub const SESSION_RECORD_FORMAT_VERSION: u32 = 4;

/// One session, stored.
///
/// Build one with [`SessionRecord::new`] and set the fields that apply; the
/// struct is `#[non_exhaustive]` because later phases store more resume state
/// on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionRecord {
    /// The format this record was written in.
    pub format_version: u32,

    /// The identity and ancestry a resumed session keeps.
    pub scope: SessionScope,

    /// The provider the session last resolved to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,

    /// The catalog identifier of the model the session last resolved to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// When the session was created.
    #[serde(with = "rfc3339_millis")]
    pub created_at: SystemTime,

    /// When the record was last written.
    #[serde(with = "rfc3339_millis")]
    pub updated_at: SystemTime,

    /// The highest sequence number committed by the session tree's event
    /// pipeline.
    ///
    /// A resumed root session continues numbering from here, so one tree's
    /// events stay uniquely numbered across restarts. With a durable sink,
    /// commitment means the sink accepted the event. Events still queued when
    /// the record is taken are not counted.
    #[serde(default, deserialize_with = "null_as_default")]
    pub last_event_seq: u64,

    /// The conversation, oldest turn first.
    #[serde(default, deserialize_with = "null_as_default")]
    pub messages: Vec<StoredMessage>,
}

impl SessionRecord {
    /// Starts a record for `scope`, stamped with the current time.
    #[must_use]
    pub fn new(scope: SessionScope) -> Self {
        let now = SystemTime::now();
        Self {
            format_version: SESSION_RECORD_FORMAT_VERSION,
            scope,
            provider: None,
            model: None,
            created_at: now,
            updated_at: now,
            last_event_seq: 0,
            messages: Vec::new(),
        }
    }

    /// Whether this build can restore the record.
    ///
    /// Only the current format is supported.
    #[must_use]
    pub const fn is_supported(&self) -> bool {
        self.format_version == SESSION_RECORD_FORMAT_VERSION
    }

    /// Resumes this record past a durable log whose head is `log_head`: the
    /// cursor moves up to the head, never back, so the resumed session's first
    /// event is numbered above every event the log already holds.
    ///
    /// This is the rule for a record and an event log that were not saved in
    /// one transaction: a crash between the log's last write and the record's
    /// leaves the log ahead, and a resume that numbered from the record would
    /// reuse a sequence number the log has. Call it with the log's last
    /// sequence before [`CodingAgent::resume`](crate::CodingAgent::resume).
    pub fn resume_after(&mut self, log_head: u64) {
        self.last_event_seq = self.last_event_seq.max(log_head);
    }

    /// Advances the event cursor to include events already held by the sink:
    /// [`resume_after`](Self::resume_after) under its older name.
    pub fn advance_event_cursor(&mut self, committed_seq: u64) {
        self.resume_after(committed_seq);
    }

    /// The exact route the session last ran on, as a `provider/model` selector
    /// the client's resolver restores without guessing.
    ///
    /// `None` when the record names no provider or no model, which a resume
    /// on the recorded model refuses rather than reinterpreting.
    #[must_use]
    pub fn recorded_route(&self) -> Option<String> {
        match (&self.provider, &self.model) {
            (Some(provider), Some(model)) => Some(format!("{provider}/{model}")),
            _ => None,
        }
    }
}

/// One stored conversation turn.
///
/// The mirror of [`crate::state::Message`]. The two forms hold the same
/// facts: [`crate::state::Message::to_stored_message`] writes one and
/// [`crate::state::Message::from_stored_message`] reads it back, keeping
/// every field. Timestamps are the one place the serialized form is coarser
/// than the turn it came from, because they are written with millisecond
/// precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StoredMessage {
    /// Input from the person or system driving the session.
    User {
        /// The input content.
        content:   InputContent,
        /// When the turn was recorded.
        #[serde(with = "rfc3339_millis")]
        timestamp: SystemTime,
    },
    /// A committed assistant turn.
    Assistant {
        /// The assistant's text.
        content:        String,
        /// The tool calls the turn requested.
        #[serde(default, deserialize_with = "tool_call::deserialize")]
        tool_calls:     Vec<ToolCall>,
        /// Provider-native parts preserved for lossless replay.
        #[serde(default, deserialize_with = "null_as_default")]
        provider_parts: Vec<ContentPart>,
        /// The token accounting the provider reported for this turn.
        ///
        /// Stored as typed counts, so restoring a record returns exactly the
        /// numbers that were saved.
        #[serde(default, deserialize_with = "null_as_default")]
        usage:          TokenUsage,
        /// The provider's identifier for the response.
        #[serde(default)]
        response_id:    String,
        /// When the turn was committed.
        #[serde(with = "rfc3339_millis")]
        timestamp:      SystemTime,
    },
    /// The results of the tool calls a preceding assistant turn requested.
    ToolResults {
        /// One result per tool call, in call order.
        #[serde(default, deserialize_with = "null_as_default")]
        results:   Vec<ToolResult>,
        /// When the results were recorded.
        #[serde(with = "rfc3339_millis")]
        timestamp: SystemTime,
    },
    /// Injected content sent to the model with the system role.
    System {
        /// The injected text.
        content:   String,
        /// When the turn was recorded.
        #[serde(with = "rfc3339_millis")]
        timestamp: SystemTime,
    },
    /// A handoff summary that replaced older turns.
    Compaction {
        /// The model-visible summary.
        summary:                 String,
        /// Why the compaction ran.
        #[serde(default)]
        reason:                  CompactionReason,
        /// Turns present before compaction.
        #[serde(default)]
        original_turn_count:     usize,
        /// Turns preserved verbatim.
        #[serde(default)]
        preserved_turn_count:    usize,
        /// Estimated context tokens before compaction.
        #[serde(default)]
        estimated_tokens_before: usize,
        /// Estimated tokens in the summary.
        #[serde(default)]
        summary_token_estimate:  usize,
        /// Files represented in the compaction prompt.
        #[serde(default)]
        tracked_file_count:      usize,
        /// Whether Pebble truncated the generated summary.
        #[serde(default)]
        summary_truncated:       bool,
        /// Usage from the summarization call.
        #[serde(default, deserialize_with = "null_as_default")]
        usage:                   TokenUsage,
        /// Cost of the summarization call in USD micros.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost_usd_micros:         Option<u64>,
        /// When the summary was recorded.
        #[serde(with = "rfc3339_millis")]
        timestamp:               SystemTime,
    },
    /// Injected steering sent to the model with the user role.
    Steering {
        /// The steering content.
        content:   InputContent,
        /// When the turn was recorded.
        #[serde(with = "rfc3339_millis")]
        timestamp: SystemTime,
    },
}

impl StoredMessage {
    /// When this turn was recorded.
    #[must_use]
    pub fn timestamp(&self) -> SystemTime {
        match self {
            Self::User { timestamp, .. }
            | Self::Assistant { timestamp, .. }
            | Self::ToolResults { timestamp, .. }
            | Self::System { timestamp, .. }
            | Self::Compaction { timestamp, .. }
            | Self::Steering { timestamp, .. } => *timestamp,
        }
    }
}

/// Reads a member that may be absent or `null` as its default.
///
/// Serde's `default` covers an absent member only. Records in the wild carry
/// explicit nulls — a writer that failed to serialize a value, a hand-edited
/// document — and a null must not be the difference between a session that
/// resumes and one that does not.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{Duration, UNIX_EPOCH};

    use lithos_llm::types::{ReasoningContent, ToolArguments, ToolInput};
    use serde_json::json;

    use super::*;
    use crate::SessionId;
    use crate::types::Message;

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
            provider_metadata: BTreeMap::from([(
                "openai".to_owned(),
                json!({ "item_id": "fc_1" }),
            )]),
        }
    }

    fn tool_result() -> ToolResult {
        ToolResult {
            tool_call_id: "call_1".into(),
            name:         Some("read_file".into()),
            content:      vec![ContentPart::Text {
                text: "//! Pebble".into(),
            }],
            is_error:     false,
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
                        text:             "the root is small".into(),
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
                results:   vec![tool_result()],
                timestamp: moment(),
            },
            Message::System {
                content:   "[Context Summary]".into(),
                timestamp: moment(),
            },
            Message::Steering {
                content:   "also update the changelog".into(),
                timestamp: moment(),
            },
        ]
    }

    #[test]
    fn every_turn_survives_a_stored_round_trip() {
        for turn in every_turn() {
            let restored = Message::from_stored_message(&turn.to_stored_message());
            assert_eq!(restored, turn, "turn did not survive the round trip");
        }
    }

    #[test]
    fn every_turn_survives_a_json_round_trip() {
        let stored: Vec<StoredMessage> = every_turn()
            .iter()
            .map(Message::to_stored_message)
            .collect();

        let json = serde_json::to_string(&stored).expect("stored messages serialize");
        let restored: Vec<StoredMessage> =
            serde_json::from_str(&json).expect("stored messages parse");

        assert_eq!(restored, stored);
    }

    #[test]
    fn usage_round_trips_exactly() {
        let turn = Message::Assistant {
            content:        String::new(),
            tool_calls:     Vec::new(),
            provider_parts: Vec::new(),
            usage:          usage(),
            response_id:    "resp_1".into(),
            timestamp:      moment(),
        };

        let json = serde_json::to_string(&turn.to_stored_message()).expect("turn serializes");
        let stored: StoredMessage = serde_json::from_str(&json).expect("turn parses");

        let Message::Assistant {
            usage: restored, ..
        } = Message::from_stored_message(&stored)
        else {
            panic!("expected an assistant turn");
        };
        assert_eq!(restored, usage());
    }

    #[test]
    fn a_null_usage_restores_as_no_usage() {
        let stored: StoredMessage = serde_json::from_value(json!({
            "kind": "assistant",
            "content": "hello",
            "tool_calls": null,
            "provider_parts": null,
            "usage": null,
            "response_id": "resp_1",
            "timestamp": "2026-01-01T00:00:00.500Z",
        }))
        .expect("a record with null members still restores");

        let Message::Assistant {
            usage,
            tool_calls,
            provider_parts,
            ..
        } = Message::from_stored_message(&stored)
        else {
            panic!("expected an assistant turn");
        };
        assert_eq!(usage, TokenUsage::default());
        assert!(tool_calls.is_empty());
        assert!(provider_parts.is_empty());
    }

    #[test]
    fn an_assistant_turn_restores_from_its_required_members_alone() {
        let stored: StoredMessage = serde_json::from_value(json!({
            "kind": "assistant",
            "content": "hello",
            "timestamp": "2026-01-01T00:00:00.500Z",
        }))
        .expect("optional members may be absent");

        assert_eq!(stored, StoredMessage::Assistant {
            content:        "hello".into(),
            tool_calls:     Vec::new(),
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    String::new(),
            timestamp:      moment(),
        });
    }

    #[test]
    fn unknown_members_are_ignored() {
        let stored: StoredMessage = serde_json::from_value(json!({
            "kind": "user",
            "content": "hello",
            "timestamp": "2026-01-01T00:00:00.500Z",
            "sentiment": "curious",
        }))
        .expect("an unknown member is ignored");

        assert_eq!(stored, StoredMessage::User {
            content:   "hello".into(),
            timestamp: moment(),
        });
    }

    #[test]
    fn a_record_requires_explicit_scope_and_version() {
        assert!(
            serde_json::from_value::<SessionRecord>(json!({
                "session_id": "ses_1",
                "created_at": "2026-01-01T00:00:00.500Z",
                "updated_at": "2026-01-01T00:00:00.500Z",
            }))
            .is_err()
        );
    }

    #[test]
    fn a_newer_format_version_is_not_supported() {
        let mut record = SessionRecord::new(SessionScope::root(SessionId::new("ses_1")));
        assert!(record.is_supported());

        record.format_version = SESSION_RECORD_FORMAT_VERSION + 1;

        assert!(!record.is_supported());
    }

    #[test]
    fn reconciling_an_event_cursor_only_moves_it_forward() {
        let mut record = SessionRecord::new(SessionScope::root(SessionId::new("ses_1")));
        record.last_event_seq = 41;

        record.advance_event_cursor(45);
        assert_eq!(record.last_event_seq, 45);

        record.advance_event_cursor(42);
        assert_eq!(record.last_event_seq, 45);
    }

    #[test]
    fn a_new_record_names_the_current_format() {
        let record = SessionRecord::new(SessionScope::root(SessionId::new("ses_1")));

        assert_eq!(record.format_version, SESSION_RECORD_FORMAT_VERSION);
        assert_eq!(record.scope.session_id().as_str(), "ses_1");
        assert_eq!(record.created_at, record.updated_at);
    }

    #[test]
    fn a_whole_record_survives_a_json_round_trip() {
        let mut record = SessionRecord::new(SessionScope::root(SessionId::new("ses_1")));
        record.scope =
            SessionScope::root(SessionId::new("ses_root")).child(SessionId::new("ses_1"));
        record.provider = Some("anthropic".into());
        record.model = Some("claude-sonnet-5".into());
        record.created_at = moment();
        record.updated_at = moment();
        record.last_event_seq = 41;
        record.messages = every_turn()
            .iter()
            .map(Message::to_stored_message)
            .collect();

        let json = serde_json::to_string(&record).expect("record serializes");
        let restored: SessionRecord = serde_json::from_str(&json).expect("record parses");

        assert_eq!(restored, record);
    }

    #[test]
    fn a_stored_turn_reports_its_timestamp() {
        for turn in every_turn() {
            assert_eq!(turn.to_stored_message().timestamp(), moment());
        }
    }
}
