//! Contract tests for the serialized event stream.
//!
//! The JSON form of [`SessionEvent`] and [`AgentEvent`] is public API, so
//! these snapshots are reviewed like a specification. Evolution is additive:
//! a new variant or a new optional field may appear, but an existing line may
//! not change or disappear without a major version and a migration.
//!
//! Values are serialized directly from the typed value, never through
//! `serde_json::Value`, so a snapshot pins what a consumer actually receives.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lithos_llm::catalog::ProviderId;
use lithos_llm::types::{Error as LlmError, ErrorKind as LlmErrorKind, RetryClassification};
use pebble::{
    Actor, AgentEvent, AgentProfileKind, CommandTermination, ContextWindowBreakdownItem,
    ContextWindowCategory, ContextWindowCountMethod, ContextWindowSnapshot, ContextWindowStaleness,
    ContextWindowWarning, CostSource, Error, ErrorData, ErrorKind, EventSinkError, ExecOutputTail,
    InterruptReason, LlmOutputKind, LlmRetryPhase, MemoryFileSummary, PermissionLevel,
    ReasoningOutput, SessionEvent, SkillActivationSource, SkillSummary, TodoCreatedProps,
    TodoDeletedProps, TodoListKind, TodoStatus, TodoUpdatedProps, TokenUsage, ToolCategory,
    ToolErrorKind, ToolSource, ToolSummary,
};
use serde::Serialize;
use serde_json::json;

fn render<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).expect("value serializes")
}

/// A model-layer failure carrying every member pebble projects from one.
fn rate_limited() -> LlmError {
    LlmError::new(LlmErrorKind::RateLimit, "slow down")
        .with_provider(ProviderId::from("openai"))
        .with_status(429)
        .with_provider_code("rate_limit_exceeded")
        .with_retry(RetryClassification::after(Duration::from_secs(2)))
        .with_provider_retry_after(Duration::from_secs(2))
}

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

fn context_window() -> ContextWindowSnapshot {
    ContextWindowSnapshot {
        provider:              "anthropic".into(),
        model:                 "claude-sonnet-5".into(),
        context_window_tokens: 200_000,
        input_tokens:          52_000,
        usage_percent:         26.0,
        count_method:          ContextWindowCountMethod::ResponseUsageScaledBreakdown,
        staleness:             ContextWindowStaleness::Live,
        generated_at:          moment(),
        event_seq:             Some(41),
        breakdown:             vec![
            ContextWindowBreakdownItem {
                category:      ContextWindowCategory::SystemPrompt,
                tokens:        2_000,
                usage_percent: 1.0,
            },
            ContextWindowBreakdownItem {
                category:      ContextWindowCategory::Conversation,
                tokens:        50_000,
                usage_percent: 25.0,
            },
        ],
        warnings:              vec![ContextWindowWarning {
            code:    "scaled_estimate".into(),
            message: "per-category counts are scaled from the reported total".into(),
        }],
    }
}

fn every_variant() -> Vec<AgentEvent> {
    vec![
        AgentEvent::SessionStarted {
            provider: Some("anthropic".into()),
            model:    Some("claude-sonnet-5".into()),
        },
        AgentEvent::SessionEnded,
        AgentEvent::ProcessingEnd,
        AgentEvent::UserInput {
            text: "fix the failing test".into(),
        },
        AgentEvent::LlmRequestStarted {
            requested_model: "claude-sonnet-5".into(),
        },
        AgentEvent::LlmFirstOutput {
            kind: LlmOutputKind::Reasoning,
        },
        AgentEvent::AssistantOutputReplace {
            text:      String::new(),
            reasoning: None,
        },
        AgentEvent::AssistantMessage {
            text:            "I updated the parser.".into(),
            model:           "claude-sonnet-5".into(),
            usage:           usage(),
            cost_usd_micros: Some(12_500),
            cost_source:     Some(CostSource::Catalog),
            tool_call_count: 2,
            context_window:  Some(context_window()),
            reasoning:       Some(ReasoningOutput::new("checked the parser", "step one")),
        },
        AgentEvent::TextDelta {
            delta: "I up".into(),
        },
        AgentEvent::ReasoningDelta {
            delta: "weighing".into(),
        },
        AgentEvent::ToolCallStarted {
            tool_name:    "shell".into(),
            tool_call_id: "call_1".into(),
            arguments:    serde_json::json!({"command": "cargo test"}),
        },
        AgentEvent::ToolCallOutputDelta {
            delta: "running 1 test".into(),
        },
        AgentEvent::ToolCallCompleted {
            tool_name:             "shell".into(),
            tool_call_id:          "call_1".into(),
            output:                serde_json::json!("test result: ok"),
            is_error:              false,
            error_kind:            None,
            output_bytes_observed: 4_096,
            output_bytes_retained: 1_024,
            output_bytes_omitted:  3_072,
        },
        AgentEvent::ToolCallCompleted {
            tool_name:             "write_file".into(),
            tool_call_id:          "call_2".into(),
            output:                serde_json::json!("permission denied"),
            is_error:              true,
            error_kind:            Some(ToolErrorKind::Denied),
            output_bytes_observed: 17,
            output_bytes_retained: 17,
            output_bytes_omitted:  0,
        },
        AgentEvent::ToolProcessCompleted {
            exit_code:             Some(101),
            termination:           CommandTermination::Exited,
            duration_ms:           1_432,
            streams_separated:     true,
            exec_output_tail:      Some(ExecOutputTail {
                stdout:           Some("test result: FAILED".into()),
                stderr:           None,
                stdout_truncated: true,
                stderr_truncated: false,
            }),
            output_bytes_observed: 4_096,
            output_bytes_retained: 1_024,
            output_bytes_omitted:  3_072,
        },
        AgentEvent::Error {
            error: ErrorData::from(&Error::Interrupted(InterruptReason::WallClockTimeout)),
        },
        AgentEvent::Warning {
            kind:    "context_budget".into(),
            message: "the prompt is close to the context window".into(),
            details: serde_json::json!({"usage_percent": 92.5}),
        },
        AgentEvent::LoopDetected,
        AgentEvent::SteeringInjected {
            text:  "also update the changelog".into(),
            actor: Some(Actor::User {
                id:           Some("u_1".into()),
                display_name: Some("Ada".into()),
            }),
        },
        AgentEvent::RoundInterrupted { generation: 3 },
        AgentEvent::CompactionStarted {
            estimated_tokens:    180_000,
            context_window_size: 200_000,
        },
        AgentEvent::CompactionCompleted {
            original_turn_count:    64,
            preserved_turn_count:   12,
            summary_token_estimate: 900,
            tracked_file_count:     7,
        },
        AgentEvent::LlmRetry {
            provider:   "openai".into(),
            model:      "gpt-5".into(),
            attempt:    0,
            delay_secs: 2.0,
            error:      ErrorData::new(ErrorKind::Llm, "rate limited")
                .with_provider("openai")
                .with_model("gpt-5"),
            phase:      LlmRetryPhase::Open,
        },
        AgentEvent::SubAgentSpawned {
            agent_id:   "sa_1".into(),
            depth:      1,
            task:       "survey the test suite".into(),
            generation: 1,
        },
        AgentEvent::SubAgentTurnStarted {
            agent_id:   "sa_1".into(),
            depth:      1,
            task:       "survey the test suite".into(),
            generation: 2,
        },
        AgentEvent::SubAgentCompleted {
            agent_id:   "sa_1".into(),
            depth:      1,
            generation: 2,
            success:    true,
            turns_used: 4,
        },
        AgentEvent::SubAgentFailed {
            agent_id:   "sa_2".into(),
            depth:      1,
            generation: 1,
            error:      ErrorData::from(&Error::SessionClosed),
        },
        AgentEvent::SubAgentClosed {
            agent_id:   "sa_1".into(),
            depth:      1,
            generation: 2,
        },
        AgentEvent::MemoryLoaded {
            profile:            AgentProfileKind::Claude5.as_str().to_owned(),
            files:              vec![MemoryFileSummary {
                path:         "/work/AGENTS.md".into(),
                byte_count:   4_096,
                loaded_bytes: 2_048,
                truncated:    true,
            }],
            total_loaded_bytes: 2_048,
            budget_bytes:       8_192,
        },
        AgentEvent::SkillsDiscovered {
            profile:     AgentProfileKind::Claude5.as_str().to_owned(),
            source_dirs: vec!["/work/.skills".into()],
            skills:      vec![SkillSummary {
                name:        "review".into(),
                description: "Review a diff".into(),
            }],
        },
        AgentEvent::SkillActivated {
            skill_name: "review".into(),
            source:     SkillActivationSource::Slash,
        },
        AgentEvent::TodoCreated(TodoCreatedProps {
            list_id:     "openai_plan:ses_1".into(),
            list_kind:   TodoListKind::OpenAiPlan,
            todo_id:     "todo_1".into(),
            status:      TodoStatus::Pending,
            order:       0,
            subject:     "read the failing test".into(),
            description: "start from the assertion".into(),
            active_form: Some("reading the failing test".into()),
            owner:       Some("agent".into()),
            blocks:      vec!["todo_2".into()],
            blocked_by:  Vec::new(),
            metadata:    BTreeMap::from([("origin".to_owned(), serde_json::json!("plan"))]),
        }),
        AgentEvent::TodoUpdated(TodoUpdatedProps {
            status: Some(TodoStatus::InProgress),
            active_form: Some(None),
            owner: Some(Some("agent".into())),
            metadata_patch: BTreeMap::from([("origin".to_owned(), serde_json::Value::Null)]),
            ..TodoUpdatedProps::new("openai_plan:ses_1", TodoListKind::OpenAiPlan, "todo_1")
        }),
        AgentEvent::TodoDeleted(TodoDeletedProps {
            list_id:   "openai_plan:ses_1".into(),
            list_kind: TodoListKind::OpenAiPlan,
            todo_id:   "todo_1".into(),
        }),
    ]
}

#[test]
fn every_agent_event_variant_keeps_its_serialized_shape() {
    insta::assert_snapshot!("agent_event_variants", render(&every_variant()));
}

#[test]
fn the_session_event_envelope_keeps_its_serialized_shape() {
    let events = vec![
        SessionEvent::new("ses_root", AgentEvent::SessionEnded, moment()).with_seq(1),
        SessionEvent::new(
            "ses_child",
            AgentEvent::ToolCallOutputDelta {
                delta: "running".into(),
            },
            moment(),
        )
        .with_seq(42)
        .with_parent_session_id("ses_root")
        .with_tool_call_id("call_1"),
    ];
    insta::assert_snapshot!("session_event_envelope", render(&events));
}

#[test]
fn the_supporting_vocabulary_keeps_its_serialized_shape() {
    let summaries = vec![
        ToolSummary {
            name:        "read_file".into(),
            description: "Read a file from the workspace".into(),
            source:      ToolSource::Native,
            category:    ToolCategory::Read,
            invoked:     false,
        },
        ToolSummary {
            name:        "inspect".into(),
            description: "Inspect an application value".into(),
            source:      ToolSource::Application,
            category:    ToolCategory::Other,
            invoked:     true,
        },
        ToolSummary {
            name:        "mcp__filesystem__read_text_file".into(),
            description: "Read a text file".into(),
            source:      ToolSource::Mcp {
                server_name:   "filesystem".into(),
                original_name: "read_text_file".into(),
            },
            category:    ToolCategory::Read,
            invoked:     true,
        },
        ToolSummary {
            name:        "use_skill".into(),
            description: "Activate a skill".into(),
            source:      ToolSource::Skill,
            category:    ToolCategory::Other,
            invoked:     false,
        },
    ];
    insta::assert_snapshot!("tool_summaries", render(&summaries));
    insta::assert_snapshot!(
        "permission_levels",
        render(&[
            PermissionLevel::ReadOnly,
            PermissionLevel::ReadWrite,
            PermissionLevel::Full,
        ])
    );
    insta::assert_snapshot!("agent_profile_kinds", render(&AgentProfileKind::ALL));
}

#[test]
fn error_projections_keep_their_serialized_shape() {
    let projections = vec![
        ErrorData::from(&Error::SessionClosed),
        ErrorData::from(&Error::InvalidState("no active round".into())),
        ErrorData::from(&Error::ToolExecution("shell exited 1".into())),
        ErrorData::from(&Error::Interrupted(InterruptReason::Cancelled)),
        ErrorData::from(&Error::EventSink(EventSinkError::new(
            "the event log is not writable",
        ))),
        ErrorData::new(ErrorKind::Compaction, "context compaction failed")
            .with_provider("anthropic")
            .with_model("claude-sonnet-5"),
        // The model-layer members are re-exported lithos-llm types, so their
        // spellings are pinned here too: a change in that crate must not
        // change pebble's event stream unnoticed.
        ErrorData::from(&rate_limited()).with_model("gpt-5"),
    ];
    insta::assert_snapshot!("error_projections", render(&projections));
}

#[test]
fn every_variant_survives_a_round_trip_through_its_snapshot_shape() {
    let events = every_variant();
    let json = serde_json::to_string(&events).expect("events serialize");
    let restored: Vec<AgentEvent> = serde_json::from_str(&json).expect("events parse");
    assert_eq!(restored, events);
}

#[test]
fn an_event_with_unknown_members_still_parses() {
    let envelope: SessionEvent = serde_json::from_value(json!({
        "seq": 7,
        "event": {
            "UserInput": {
                "text": "fix the failing test",
                "locale": "en-GB",
            },
        },
        "timestamp": "2026-01-01T00:00:00.500Z",
        "session_id": "ses_root",
        "trace_id": "01JC",
    }))
    .expect("unknown members are ignored");

    assert_eq!(envelope.seq, 7);
    assert_eq!(envelope.session_id, "ses_root");
    assert_eq!(envelope.parent_session_id, None);
    assert!(matches!(
        envelope.event,
        AgentEvent::UserInput { text } if text == "fix the failing test"
    ));
}

#[test]
fn an_event_variant_this_build_does_not_know_is_read_through_the_envelope() {
    // The lenient-read pattern the crate docs give a consumer: an additive
    // variant fails the typed parse, so the envelope is read first and the
    // payload is recognized second. Without it, one unknown variant would cost
    // the reader the sequence number and the session identity too.
    #[derive(serde::Deserialize)]
    struct RawEnvelope {
        seq:        u64,
        session_id: String,
        event:      serde_json::Value,
    }

    let line = json!({
        "seq": 9,
        "event": { "SandboxEscaped": { "detail": "from a newer pebble" } },
        "timestamp": "2026-01-01T00:00:00.500Z",
        "session_id": "ses_root",
    });

    serde_json::from_value::<SessionEvent>(line.clone())
        .expect_err("the typed envelope refuses a variant this build does not know");

    let envelope: RawEnvelope =
        serde_json::from_value(line).expect("the envelope parses without the payload");
    assert_eq!(envelope.seq, 9);
    assert_eq!(envelope.session_id, "ses_root");
    serde_json::from_value::<AgentEvent>(envelope.event)
        .expect_err("the payload is what this build cannot read");
}

#[test]
fn an_event_without_its_optional_members_still_parses() {
    let envelope: SessionEvent = serde_json::from_value(json!({
        "event": "SessionEnded",
        "timestamp": "2026-01-01T00:00:00.500Z",
        "session_id": "ses_root",
    }))
    .expect("optional members may be absent");

    assert_eq!(envelope.seq, 0, "an unsequenced event reads as sequence 0");
    assert_eq!(envelope.tool_call_id, None);
    assert!(matches!(envelope.event, AgentEvent::SessionEnded));
}
