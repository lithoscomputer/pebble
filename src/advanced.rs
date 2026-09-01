//! Lower-level coding runtime and orchestration contracts.
//!
//! Most applications should use [`crate::CodingSession`].

/// The attribute used by Pebble's async extension traits.
pub use async_trait::async_trait;
/// The model layer used by Pebble's public contracts.
pub use lithos_llm as llm;
/// The provider-neutral agent layer below Pebble's coding facade.
pub use pebble_agent as agent;
/// The cancellation signal used by Pebble's extension traits.
pub use tokio_util::sync::CancellationToken;

pub use crate::error::CompactionError;
pub use crate::human_input::{
    Answer, AnswerStatus, HumanInputError, HumanInputProvider, Question, QuestionKind,
    QuestionOption, is_question_tool,
};
pub use crate::redact::{NoRedaction, Redactor};
pub use crate::runtime::{
    CodingRuntime, CodingRuntimeBuildError, CodingRuntimeBuilder, CompletionCoordinator,
    InterruptReasonHandle, PromptTiming, SessionControlHandle, SteeringItem, SteeringMessage,
};
pub use crate::search::{
    SearchError, SearchErrorKind, SearchProvider, SearchRequest, SearchResult,
};
pub use crate::subagent::{ChildSessionSpec, SessionFactory, SubagentLimits};
