//! Optional services supplied by an embedding application.

pub use crate::human_input::{
    Answer, AnswerStatus, HumanInputError, HumanInputProvider, Question, QuestionKind,
    QuestionOption,
};
pub use crate::profile::EnvContext;
pub use crate::prompt_transform::{
    SystemPromptContext, SystemPromptDecision, SystemPromptTransform,
};
pub use crate::redact::{NoRedaction, Redactor};
pub use crate::search::{
    SearchError, SearchErrorKind, SearchProvider, SearchRequest, SearchResult,
};
