//! Durable coding events, the sink that records them, and retry observation.
//!
//! Every event a coding agent publishes is a [`CodingAgentEvent`]: a numbered
//! envelope around one [`CodingEvent`]. An application records them through an
//! [`EventSink`] and watches them live through
//! [`CodingAgent::subscribe`](crate::CodingAgent::subscribe). The serialized
//! form of both is public API.

pub use crate::error::{ErrorData, ErrorKind};
pub use crate::event::{
    DEFAULT_EVENT_CAPACITY, DEFAULT_EVENT_SINK_TIMEOUT, EventCapacity, EventSink, EventSinkError,
    EventSinkTimeout,
};
pub use crate::reasoning::ReasoningOutput;
pub use crate::runtime::RetryEventObserver;
pub use crate::types::{
    Actor, AgentProfileKind, CodingAgentEvent, CodingAgentState, CodingEvent, CommandTermination,
    ContextWindowBreakdownItem, ContextWindowCategory, ContextWindowCountMethod,
    ContextWindowSnapshot, ContextWindowStaleness, ContextWindowWarning, CostSource,
    ExecOutputTail, ExecOutputTailTrace, INITIAL_SUBAGENT_GENERATION, LlmOutputKind, LlmRetryPhase,
    MemoryFileSummary, PermissionLevel, SkillActivationSource, SkillSummary, TodoCreatedProps,
    TodoDeletedProps, TodoListKind, TodoListProjection, TodoProjection, TodoStatus,
    TodoUpdatedProps, TokenUsage, ToolCategory, ToolErrorKind, ToolSource, ToolSummary,
};
