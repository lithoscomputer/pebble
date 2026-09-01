//! Lower-level coding runtime and orchestration contracts.
//!
//! Most applications should use [`crate::CodingSession`].

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
