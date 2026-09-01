//! A provider-neutral agent loop built on `lithos-llm`.
//!
//! This crate owns one active conversation: model turns, tool execution,
//! steering, follow-up input, cancellation, history, and lifecycle events. It
//! does not know about files, shells, coding profiles, memory, skills,
//! subagents, persistence, credentials, or provider construction.
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
mod context;
mod control;
mod error;
mod event;
mod model;
mod stream;
mod tool;
mod validation;

pub use self::agent::{
    Agent, AgentBuilder, AgentConfig, AgentSnapshot, AgentState, PromptOutcome, ToolExecution,
    UserMessage,
};
pub use self::context::{ContextTransform, ContextTransformError, TransformContext};
pub use self::control::AgentControlHandle;
pub use self::error::{AgentBuildError, AgentError, Result};
pub use self::event::{AgentEvent, FirstOutputKind};
pub use self::model::ModelService;
pub use self::tool::{Tool, ToolContext, ToolError, ToolExecutor, ToolOutput};

/// Lower-level turn primitives for specialized agent layers.
pub mod advanced {
    pub use crate::stream::{StreamObserver, StreamOutcome, stream_response};
    pub use crate::validation::{ToolArgumentsError, validate_tool_arguments};
}

/// The model layer used by this crate.
///
/// Re-exported under one namespace so applications can use the exact request,
/// response, and tool types in this crate's public contracts.
pub use lithos_llm as llm;
