//! A provider-neutral agent loop built on `lithos-llm`.
//!
//! This crate owns one active conversation: model turns, tool execution,
//! steering, follow-up input, cancellation, history, and lifecycle events. It
//! does not know about files, shells, coding profiles, memory, skills,
//! subagents, persistence, credentials, or provider construction.
//!
//! Specialized layers can resolve and filter tools for each turn with
//! [`ToolProvider`] and [`ToolAccessPolicy`]. [`ToolCallHooks`] surround tool
//! execution, [`TurnBoundaryHooks`] own compaction and background-result
//! boundaries, and [`EventProjection`] maps the generic lifecycle into a
//! durable application event model.
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
//! );
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
mod error;
mod event;
mod model;
mod stream;
mod tool;
mod turn;
mod validation;

pub use self::agent::{
    Agent, AgentBuilder, AgentConfig, AgentSnapshot, AgentState, PromptOutcome, ToolExecution,
    UserMessage,
};
pub use self::control::AgentControlHandle;
pub use self::error::{AgentBuildError, AgentError, Result};
pub use self::event::{AgentEvent, EventProjection, FirstOutputKind};
pub use self::model::ModelService;
pub use self::tool::{
    BeforeToolCall, Tool, ToolAccess, ToolAccessContext, ToolAccessPolicy, ToolCallContext,
    ToolCallHooks, ToolCallOutcome, ToolContext, ToolError, ToolErrorKind, ToolExecutor,
    ToolOutput, ToolProvider,
};
pub use self::turn::{
    TurnBoundaryAction, TurnBoundaryContext, TurnBoundaryError, TurnBoundaryHooks, TurnContext,
};

/// Lower-level turn primitives for specialized agent layers.
pub mod advanced {
    pub use crate::stream::{StreamObserver, StreamOutcome, stream_response};
    pub use crate::tool::{ToolRoundContext, ToolRoundExecutor};
    pub use crate::validation::{ToolArgumentsError, validate_tool_arguments};
}
