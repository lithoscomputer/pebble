//! Durable coding events, the sink that records them, and retry observation.
//!
//! Every event a coding agent publishes is a [`CodingAgentEvent`]: a numbered
//! envelope around one [`CodingEvent`]. One root agent and all its descendants
//! share one ordered stream. An application records it without loss through an
//! [`EventSink`] and watches it live, with bounded lag, through
//! [`CodingAgent::subscribe`](crate::CodingAgent::subscribe). The serialized
//! form of both event types is public API.

pub use lithos_llm::types::ReasoningOutput;

pub use crate::compaction::CompactionReason;
pub use crate::error::{ErrorData, ErrorKind};
pub use crate::event::{
    DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_SINK_TIMEOUT, EventCapacity, EventSink, EventSinkError,
    EventSinkTimeout,
};
pub use crate::runtime::RetryEventObserver;
pub use crate::types::{
    Actor, AgentProfileKind, CodingAgentEvent, CodingAgentState, CodingEvent, CommandTermination,
    ContextWindowBreakdownItem, ContextWindowCategory, ContextWindowCountMethod,
    ContextWindowSnapshot, ContextWindowStaleness, ContextWindowWarning, CostSource,
    ExecOutputTail, ExecOutputTailTrace, INITIAL_SUBAGENT_GENERATION, InputContent, InputSource,
    LlmOutputKind, LlmRetryPhase, MemoryFileSummary, PermissionLevel, SkillActivationSource,
    SkillSummary, SkippedSkill, SkippedSkillReason, TodoCreatedProps, TodoDeletedProps,
    TodoListKind, TodoListProjection, TodoProjection, TodoStatus, TodoUpdatedProps, TokenUsage,
    ToolCategory, ToolErrorKind, ToolSource, ToolSummary,
};
