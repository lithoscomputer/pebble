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
pub(crate) use self::execution::{ToolDispatch, result_text};
pub(crate) use self::native::{NativeTool, ToolVocabulary};
pub use self::permissions::canonical_tool_name;
#[cfg(test)]
pub(crate) use self::permissions::{known_tool_category, tool_category};
pub use self::registry::{
    CodingEventEmitter, RegisteredTool, StaticEnvProvider, ToolContext, ToolEnvProvider,
    ToolExecutor,
};
pub(crate) use self::registry::{
    ToolDefinitionWithSource, ToolRegistry, optional_integer_arg, optional_usize_arg, required_str,
    whole_number,
};
pub use self::runner::{CodingToolSet, ToolEventCallback, ToolRunner};
