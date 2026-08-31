//! Pebble is a coding-agent loop library built on lithos-llm. It runs the
//! turn loop that a coding agent needs — model calls, tool execution, session
//! state, and the events an application observes — and leaves transport,
//! storage, and process isolation to the embedding application.
//!
//! # Stability
//!
//! The serialized form of [`SessionEvent`] and [`AgentEvent`] is public API.
//! Evolution is additive — new variants and new optional fields — and renames
//! or removals are breaking changes with a documented migration. Consumers
//! should ignore members they do not know and tolerate variants they do not
//! know.
//!
//! [`SessionRecord`] is the same kind of contract, with one addition: it
//! carries a [`format_version`](SessionRecord::format_version), so a record
//! written by an older pebble is still readable by a newer one.
//!
//! Both promises are about what an application *reads*. Pebble builds the
//! events and their payloads; an application receives them. So a new optional
//! field on a payload — [`ContextWindowSnapshot`], [`ToolSummary`],
//! [`MemoryFileSummary`], the todo props — is additive on the wire and
//! additive for code that reads it, and it is not covered for code outside
//! pebble that builds one with a struct literal naming every field. Where a
//! payload has a constructor or a `Default`, such as
//! [`TodoUpdatedProps::new`], build through it and fill the rest with `..`,
//! which a later field cannot break. [`SessionEvent`] is the envelope
//! everything travels in and the likeliest to grow, so it is
//! `#[non_exhaustive]`: build it with [`SessionEvent::new`] and the `with_*`
//! methods.

mod error;
mod event;
mod history;
mod reasoning;
mod record;
mod types;

/// The model-layer error category carried by [`ErrorData::llm_kind`].
#[doc(inline)]
pub use lithos_llm::types::ErrorKind as LlmErrorKind;
/// Whether repeating a failed model call is safe, carried by
/// [`ErrorData::retry`].
#[doc(inline)]
pub use lithos_llm::types::RetryClassification;

pub use self::error::{CompactionError, Error, ErrorData, ErrorKind, InterruptReason, Result};
pub use self::event::{
    DEFAULT_EVENT_CAPACITY, Emitter, EventCapacity, EventOptions, EventPump, EventSink,
    EventSinkError, OutputCaptureStats, SessionBoundEmitter,
};
pub use self::history::History;
pub use self::reasoning::ReasoningOutput;
pub use self::record::{SESSION_RECORD_FORMAT_VERSION, SessionRecord, StoredMessage};
pub use self::types::{
    Actor, AgentEvent, AgentProfileKind, CommandTermination, ContextWindowBreakdownItem,
    ContextWindowCategory, ContextWindowCountMethod, ContextWindowSnapshot, ContextWindowStaleness,
    ContextWindowWarning, CostSource, ExecOutputTail, ExecOutputTailTrace,
    INITIAL_SUBAGENT_GENERATION, LlmOutputKind, LlmRetryPhase, MemoryFileSummary, Message,
    PermissionLevel, SessionEvent, SessionState, SkillActivationSource, SkillSummary,
    TodoCreatedProps, TodoDeletedProps, TodoListKind, TodoListProjection, TodoPatch,
    TodoProjection, TodoStatus, TodoUpdatedProps, TokenUsage, ToolCategory, ToolErrorKind,
    ToolSource, ToolSummary,
};
