//! Lower-level session construction and orchestration contracts.
//!
//! Most applications should use [`crate::CodingSession`].

pub use crate::error::CompactionError;
pub use crate::human_input::{
    Answer, AnswerStatus, HumanInputError, HumanInputProvider, Question, QuestionKind,
    QuestionOption, is_question_tool,
};
pub use crate::redact::{NoRedaction, Redactor};
pub use crate::search::{
    SearchError, SearchErrorKind, SearchProvider, SearchRequest, SearchResult,
};
pub use crate::session::{
    CompletionCoordinator, InterruptReasonHandle, PromptTiming, Session, SessionBuildError,
    SessionBuilder, SessionControlHandle, SteeringItem, SteeringMessage,
};
pub use crate::subagent::{ChildSessionSpec, SessionFactory, SubagentLimits};
