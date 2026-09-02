//! The machinery a session dispatches tools through.
//!
//! [`ToolRegistry`] holds what a session can call. [`NativeTool`] names the
//! tools pebble implements itself, and [`ToolVocabulary`] decides which
//! spelling of those names one model family sees. [`ToolContext`] is what a
//! running tool is handed, and [`ToolError`] is what it reports when the call
//! does not produce output. [`ToolDispatch`] runs a round of calls against all
//! of it.

mod error;
mod execution;
mod native;
mod permissions;
mod registry;
mod runner;

pub use self::error::ToolError;
pub(crate) use self::execution::result_text;
pub use self::execution::{ToolDispatch, validate_tool_args};
pub use self::native::{NativeTool, ToolVocabulary};
pub use self::permissions::{canonical_tool_name, known_tool_category, tool_category};
pub use self::registry::{
    CodingEventEmitter, RegisteredTool, StaticEnvProvider, ToolContext, ToolDefinitionWithSource,
    ToolEnvProvider, ToolExecutor, ToolRegistry,
};
pub(crate) use self::registry::{optional_usize_arg, required_str};
pub use self::runner::{CodingToolSet, ToolEventCallback, ToolRunner};
