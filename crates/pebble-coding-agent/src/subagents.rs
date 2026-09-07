//! Subagent configuration and tree limits.
//!
//! Pebble builds its own children. An application turns subagents on with
//! [`SubagentOptions`] and bounds the tree with [`SubagentLimits`]; what a
//! child inherits is decided here, not by the application:
//!
//! - a child acts through its parent's environment;
//! - a child runs under its parent's tool middleware, and cannot widen either;
//! - an application tool is root-only unless it was marked
//!   [`allow_in_subagents`](crate::tools::RegisteredTool::allow_in_subagents),
//!   and a tool marked
//!   [`requires_human_input`](crate::tools::RegisteredTool::requires_human_input)
//!   never reaches a child however it is marked;
//! - a child runs on its parent's model;
//! - a child writes events to the session tree's shared event stream;
//! - a child has nobody to ask a question, so no question tool;
//! - a child is given a task, not a project briefing: it loads no memory files
//!   and discovers no skills unless the application asks for either with
//!   [`SubagentOptions::with_inherited_memory`] and
//!   [`SubagentOptions::with_inherited_skills`].

pub use crate::subagent::{SubagentLimits, SubagentOptions};
