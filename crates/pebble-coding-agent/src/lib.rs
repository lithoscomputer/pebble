//! Pebble is a coding agent as a library. It builds coding profiles,
//! environment-backed tools, memory, skills, compaction, subagents, and durable
//! events on the provider-neutral `pebble-agent` loop.
//!
//! Start with [`CodingAgent`]. Use [`CodingAgent::builder`] to configure an
//! agent, [`CodingAgent::prompt`] to process one prompt, and
//! [`CodingAgent::shutdown`] to close its owned work.
//!
//! Pebble keeps its root API small. Durable event types are in [`events`]. Tool
//! contracts and built-in tools are in [`tools`]. Durable state is in
//! [`resources`]. Optional application services are in [`extensions`], and
//! subagent construction is in [`subagents`].

mod char_boundary;
mod coding_agent;
mod compaction;
mod config;
mod context_window;
mod environment;
mod error;
mod event;
pub mod events;
pub mod extensions;
mod file_tracker;
mod history;
mod human_input;
mod loop_detection;
mod memory;
mod profile;
mod profiles;
mod reasoning;
mod record;
mod redact;
pub mod resources;
mod runtime;
mod search;
mod skills;
mod subagent;
pub mod subagents;
mod task_reminder;
mod template;
mod tool;
pub mod tools;
mod truncation;
mod types;

#[cfg(any(test, feature = "test-util"))]
pub mod test_support;

// The README's coding agent is the snippet most readers start from, so the
// verification gate compiles it: `cargo test --doc` builds every Rust block in
// the file, and the one there is marked `no_run` because it would call a
// provider. Nothing else compiles this module and it appears in no
// documentation — rustdoc defines `doctest` only while it collects them.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
mod readme {}

pub use self::coding_agent::{
    CodingAgent, CodingAgentBuilder, CodingAgentControlHandle, PromptOutcome,
};
pub use self::config::CodingAgentOptions;
pub use self::environment::{
    CallerEnvPolicy, DEFAULT_EXEC_OUTPUT_TAIL_BYTES, DirEntry, EnvResult, Environment,
    EnvironmentError, EnvironmentErrorKind, ExecOutcome, ExecRequest, ExecResult, GrepOptions,
    LocalEnvironment, format_lines_numbered,
};
pub use self::error::{Error, InterruptReason, Result};
pub use self::runtime::{CodingAgentBuildError, PromptTiming, ShutdownReason};
