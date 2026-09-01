//! Errors returned by the agent boundary.

use std::result::Result as StdResult;

use lithos_llm::types::{Error as LlmError, RequestBuildError};
use thiserror::Error;

use crate::turn::TurnBoundaryError;

/// A result returned while an agent processes a prompt.
pub type Result<T> = StdResult<T, AgentError>;

/// Why an agent could not be built.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentBuildError {
    /// The model selector was blank.
    #[error("the model selector must not be blank")]
    EmptyModel,
    /// The event buffer cannot hold an event.
    #[error("the event capacity must be greater than zero")]
    ZeroEventCapacity,
    /// Two tools use the same model-visible name.
    #[error("tool `{name}` was registered more than once")]
    DuplicateTool {
        /// The duplicated name.
        name: String,
    },
}

/// Why an agent operation failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentError {
    /// The agent has been shut down.
    #[error("the agent is closed")]
    Closed,
    /// The input carried no content.
    #[error("the user message must contain at least one content part")]
    EmptyInput,
    /// Two tools resolved to the same model-visible name for one turn.
    #[error("tool `{name}` was resolved more than once for one turn")]
    DuplicateTool {
        /// The duplicated name.
        name: String,
    },
    /// The current prompt was aborted.
    #[error("the agent prompt was aborted")]
    Aborted,
    /// The request could not be built from the current conversation.
    #[error("building the model request")]
    Request {
        /// The request validation failure.
        #[source]
        source: RequestBuildError,
    },
    /// The model call failed.
    #[error("calling the model")]
    Model {
        /// The model-layer failure.
        #[source]
        source: LlmError,
    },
    /// A configured turn-boundary hook failed.
    #[error("processing a model-turn boundary")]
    TurnBoundary {
        /// The hook failure.
        #[source]
        source: TurnBoundaryError,
    },
}
