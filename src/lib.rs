//! Pebble is a coding agent as a library. It builds coding profiles,
//! environment-backed tools, memory, skills, compaction, subagents, and durable
//! events on the provider-neutral [`agent`] loop.
//!
//! Start with [`CodingSession`]. Use [`CodingSession::builder`] to configure a
//! session, [`CodingSession::prompt`] to process one prompt, and
//! [`CodingSession::shutdown`] to close its owned work.
//!
//! Pebble keeps its root API small. Durable event types are in [`events`]. Tool
//! contracts and built-in tools are in [`tools`]. History and other session
//! resources are in [`resources`]. The lower-level session API is in
//! [`advanced`].

pub mod advanced;
mod char_boundary;
mod coding_session;
mod compaction;
mod config;
mod context_window;
mod environment;
mod error;
mod event;
pub mod events;
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
mod search;
mod session;
mod skills;
mod subagent;
mod task_reminder;
mod template;
mod tool;
pub mod tools;
mod truncation;
mod types;

#[cfg(any(test, feature = "test-util"))]
pub mod test_support;

// The README's session is the snippet most readers start from, so the
// verification gate compiles it: `cargo test --doc` builds every Rust block in
// the file, and the one there is marked `no_run` because it would call a
// provider. Nothing else compiles this module and it appears in no
// documentation — rustdoc defines `doctest` only while it collects them.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme {}

/// The attribute used by Pebble's async extension traits.
pub use async_trait::async_trait;
/// The model layer used by Pebble's public contracts.
///
/// Model requests, responses, tools, catalogs, credentials, and client
/// middleware live under this one namespace rather than being mixed into the
/// coding-agent root.
pub use lithos_llm as llm;
/// The provider-neutral agent layer used below Pebble's coding facade.
pub use pebble_agent as agent;
/// The cancellation signal used by Pebble's extension traits.
pub use tokio_util::sync::CancellationToken;

pub use self::coding_session::{
    CodingSession, CodingSessionBuildError, CodingSessionBuilder, CodingSessionControlHandle,
    PromptOutcome,
};
pub use self::config::CodingSessionOptions;
pub use self::environment::{
    CallerEnvPolicy, DEFAULT_EXEC_OUTPUT_TAIL_BYTES, DirEntry, EnvResult, Environment,
    EnvironmentError, EnvironmentErrorKind, ExecOutcome, ExecRequest, ExecResult, GrepOptions,
    LocalEnvironment, format_lines_numbered,
};
pub use self::error::{Error, InterruptReason, Result};
pub use self::session::ShutdownReason;
