//! Pebble is a coding-agent loop library built on lithos-llm. It runs the
//! turn loop that a coding agent needs — model calls, tool execution, session
//! state, and the events an application observes — and leaves transport,
//! storage, and process isolation to the embedding application.
//!
//! # Where the work lands
//!
//! Tools act through one seam, [`Environment`]: file access, content search,
//! and running commands. [`LocalEnvironment`] does that on this machine;
//! an application with a container or a remote workspace implements the trait
//! over that instead. Test doubles live in `pebble::test_support`, behind the
//! `test-util` feature.
//!
//! # What a session may call
//!
//! [`ToolRegistry`] holds the tools a session exposes. Pebble's own are named
//! by [`NativeTool`], and a [`ToolVocabulary`] decides which spelling of those
//! names one model family sees, so the same tool can be `read_file` to one
//! model and `Read` to another without anything else in the crate noticing.
//!
//! An [`AgentProfile`] decides which tools a session starts with, which
//! vocabulary they are named in, and what the system prompt says, because a
//! model trained inside a coding harness expects that harness back. Pebble
//! selects one from the catalog metadata of the model a session resolved to.
//!
//! A round of calls runs through [`ToolDispatch`], which answers every call it
//! is given — including the ones it refuses — publishes what happened, and
//! bounds what a tool produced before the model or an application sees it.
//!
//! An application decides what may run. A [`ToolAccessPolicy`] answers by name
//! and filters the tools a session even advertises; a [`ToolHookCallback`]
//! sees each call's arguments and may block it. Pebble installs neither, and
//! without them every registered tool is exposed. [`PermissionLevel`] and its
//! [`table`](PermissionLevel::auto_approves) are there to build a policy out
//! of, not a policy pebble applies.
//!
//! # Asking a person
//!
//! A session asks the person a question only where the application gave it a
//! [`HumanInputProvider`]. Without one no question tool is registered, so a
//! model cannot block a run waiting for an answer that will never come.
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

mod char_boundary;
mod config;
mod environment;
mod error;
mod event;
mod history;
mod human_input;
mod profile;
mod reasoning;
mod record;
mod redact;
mod tool;
mod truncation;
mod types;

#[cfg(any(test, feature = "test-util"))]
pub mod test_support;

/// One part of a message or of a tool's result.
#[doc(inline)]
pub use lithos_llm::types::ContentPart;
/// The model-layer error category carried by [`ErrorData::llm_kind`].
#[doc(inline)]
pub use lithos_llm::types::ErrorKind as LlmErrorKind;
/// How hard the model should think, carried by
/// [`SessionOptions::reasoning_effort`].
#[doc(inline)]
pub use lithos_llm::types::ReasoningEffort;
/// Whether repeating a failed model call is safe, carried by
/// [`ErrorData::retry`].
#[doc(inline)]
pub use lithos_llm::types::RetryClassification;
/// The latency or cost tier to ask for, carried by
/// [`SessionOptions::speed`].
#[doc(inline)]
pub use lithos_llm::types::Speed;
/// One answered tool call, as history and the model read it.
#[doc(inline)]
pub use lithos_llm::types::ToolResult;
/// A tool invocation the model asked for, answered by [`ToolDispatch`].
#[doc(inline)]
pub use lithos_llm::types::{ToolCall, ToolCallKind};
/// What the model is told about one tool, carried by [`RegisteredTool`].
#[doc(inline)]
pub use lithos_llm::types::{ToolDefinition, ToolDefinitionKind};

pub use self::config::{
    NativeToolOptions, SessionOptions, ToolAccess, ToolAccessPolicy, ToolApprovalAdapter,
    ToolApprovalFn, ToolExposureMode, ToolHookCallback, ToolHookDecision,
};
pub use self::environment::{
    CallerEnvPolicy, DEFAULT_EXEC_OUTPUT_TAIL_BYTES, DirEntry, EnvResult, Environment,
    EnvironmentError, EnvironmentErrorKind, ExecOutcome, ExecRequest, ExecResult, GrepOptions,
    LocalEnvironment, format_lines_numbered,
};
pub use self::error::{CompactionError, Error, ErrorData, ErrorKind, InterruptReason, Result};
pub use self::event::{
    DEFAULT_EVENT_CAPACITY, Emitter, EventCapacity, EventOptions, EventPump, EventSink,
    EventSinkError, OutputCaptureStats, SessionBoundEmitter,
};
pub use self::history::History;
pub use self::human_input::{
    Answer, AnswerStatus, HumanInputError, HumanInputProvider, Question, QuestionKind,
    QuestionOption, is_question_tool,
};
pub use self::profile::{
    AgentProfile, DEFAULT_CONTEXT_WINDOW_TOKENS, EnvContext, ModelFacts, Skill, SubagentSupport,
};
pub use self::reasoning::ReasoningOutput;
pub use self::record::{SESSION_RECORD_FORMAT_VERSION, SessionRecord, StoredMessage};
pub use self::redact::{NoRedaction, Redactor};
pub use self::tool::{
    AgentEventEmitter, NativeTool, RegisteredTool, StaticEnvProvider, ToolContext,
    ToolDefinitionWithSource, ToolDispatch, ToolEnvProvider, ToolError, ToolExecutor, ToolRegistry,
    ToolVocabulary, canonical_tool_name, known_tool_category, tool_category, validate_tool_args,
};
pub use self::truncation::{
    DEFAULT_TOOL_OUTPUT_RETENTION_BYTES, DEFAULT_TOOL_OUTPUT_SERIALIZED_BYTES, OutputBudgets,
    ToolOutputLimits, TruncationMode, truncate_lines, truncate_output, truncate_tool_output,
};
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
