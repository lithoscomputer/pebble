//! Pebble is a coding agent as a library. It builds coding profiles,
//! environment-backed tools, memory, skills, compaction, subagents, and durable
//! events on the provider-neutral `pebble-agent` loop.
//!
//! Start with [`CodingAgent`]. Use [`CodingAgent::builder`] to configure an
//! agent, [`CodingAgent::prompt`] to process one prompt, and
//! [`CodingAgent::shutdown`] to close its owned work.
//!
//! Pebble keeps its root API small, with one public path for each supported
//! concept. The environment a session acts through is in [`environment`].
//! Durable event types and the sink that records them are in [`events`]. Tool
//! contracts, built-in tools, and the standalone tool runner are in [`tools`].
//! Tool permissions are ordinary `pebble-agent` middleware, with the coding
//! policies and approval service contracts also in [`tools`].
//! Durable state is in [`state`]. Optional application services — human input,
//! search, redaction, the prompt transform — are in [`extensions`], and
//! subagent configuration is in [`subagents`]. The runtime underneath is not
//! public.

mod char_boundary;
mod coding_agent;
mod compaction;
mod config;
mod context_window;
pub mod environment;
mod error;
mod event;
pub mod events;
pub mod extensions;
mod file_tracker;
mod history;
mod human_input;
mod loop_detection;
mod memory;
mod output;
mod policy;
mod profile;
mod profiles;
mod prompt_transform;
mod reasoning;
mod record;
mod redact;
mod runtime;
mod search;
mod session;
mod skills;
pub mod state;
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
    CodingAgent, CodingAgentBuildError, CodingAgentBuilder, CodingAgentControlHandle,
    CodingAgentExport, CodingAgentObservation, CodingAgentSnapshot, CodingInput, ControlSnapshot,
    PendingInput, PromptOutcome, PromptTiming, ResumeMode, ShutdownReason, SteeringMessage,
    SteeringOutcome,
};
pub use self::compaction::{
    CompactionOptions, CompactionOutcome, CompactionReason, CompactionResult,
};
pub use self::config::{CodingAgentOptions, CodingAgentOptionsError};
pub use self::error::{CompactionError, Error, InterruptReason, Result, TaskKind};
pub use self::runtime::SteeringLease;
pub use self::session::{SessionId, SessionScope};
pub use self::skills::SkillExpansionError;
pub use self::types::{InputContent, InputSource};
