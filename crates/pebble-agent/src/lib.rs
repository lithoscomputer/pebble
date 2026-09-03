//! A provider-neutral agent loop built on `lithos-llm`.
//!
//! This crate owns one active conversation: model turns, tool execution,
//! steering, follow-up input, cancellation, history, and lifecycle events. It
//! does not know about files, shells, coding profiles, memory, skills,
//! subagents, persistence, credentials, or provider construction.
//!
//! Specialized layers supply a [`ToolService`] and compose policy through
//! [`ToolMiddleware`]. [`AgentLifecycle`] owns compaction and
//! background-result boundaries. [`ConversationProjection`] receives explicit
//! conversation commits, while [`EventProjection`] maps observable lifecycle
//! events into an application event model.
//!
//! # One agent
//!
//! ```no_run
//! use pebble_agent::{Agent, Tool};
//! use serde_json::json;
//!
//! # async fn example(client: lithos_llm::Client) -> Result<(), Box<dyn std::error::Error>> {
//! let inspect = Tool::function(
//!     "inspect",
//!     "Inspect one named value",
//!     json!({
//!         "type": "object",
//!         "properties": { "name": { "type": "string" } },
//!         "required": ["name"]
//!     }),
//!     |_context, arguments| async move {
//!         Ok(format!("inspected {}", arguments["name"]).into())
//!     },
//! )?;
//!
//! let mut agent = Agent::builder(client, "provider/model")
//!     .system_prompt("Use tools when they help.")
//!     .tools([inspect])
//!     .build()?;
//!
//! let outcome = agent.prompt("Inspect the parser").await?;
//! println!("{}", outcome.text());
//! # Ok(())
//! # }
//! ```

mod agent;
mod control;
mod conversation;
mod error;
mod event;
mod model;
mod stream;
mod tool;
mod turn;
mod validation;

pub use self::agent::{
    Agent, AgentBuilder, AgentConfig, AgentSnapshot, AgentState, PromptOutcome, UserMessage,
};
pub use self::control::{
    AgentControlHandle, AgentControlSnapshot, AgentPendingInput, CompletionLease, QueueOutcome,
};
pub use self::conversation::ConversationProjection;
pub use self::error::{AgentBuildError, AgentError, Result};
pub use self::event::{AgentEvent, EventProjection, FirstOutputKind};
pub use self::model::ModelService;
pub use self::tool::{
    Tool, ToolCallNext, ToolCallRequest, ToolCatalog, ToolContext, ToolDescriptor,
    ToolDiscoveryContext, ToolDiscoveryNext, ToolError, ToolErrorKind, ToolExecutor, ToolId,
    ToolIdError, ToolMiddleware, ToolOutcome, ToolOutput, ToolOutputStats, ToolScheduling,
    ToolService, ToolSystem, ToolSystemError,
};
pub use self::turn::{
    AfterAnswerAction, AgentLifecycle, ConversationUpdate, LifecycleError, TurnContext,
};

/// The small specialization interface a layer built on this crate uses.
///
/// `pebble-coding-agent` is that layer. It observes the response stream as it
/// arrives and validates tool arguments the way the generic loop does.
/// Nothing here is needed to run an [`Agent`] directly.
pub mod integration {
    pub use crate::stream::{StreamObserver, StreamOutcome, stream_response};
    pub use crate::validation::{ToolArgumentsError, validate_tool_arguments};
}
