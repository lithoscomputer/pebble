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
//! subagent configuration is in [`subagents`]. [`ProjectMemory`] is the loader
//! an agent reads its memory files with, for an application that gives the
//! same instructions to a model call of its own. The runtime underneath is
//! not public.

mod char_boundary;
mod coding_agent;
mod compaction;
mod config;
mod context_window;
mod discovery;
pub mod environment;
mod error;
mod event;
pub mod events;
pub mod extensions;
mod file_tracker;
mod history;
mod human_input;
mod loop_detection;
#[cfg(feature = "mcp")]
pub mod mcp;
mod memory;
mod policy;
mod profile;
mod profiles;
mod prompt_transform;
mod record;
mod redact;
mod runtime;
mod search;
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

pub use pebble_agent::{SessionId, SessionScope};

pub use self::coding_agent::{
    CodingAgent, CodingAgentBuildError, CodingAgentBuilder, CodingAgentControlHandle,
    CodingAgentExport, CodingAgentObservation, CodingAgentSnapshot, CodingInput, ControlSnapshot,
    FallbackRoute, PendingInput, PromptOutput, PromptReport, PromptTiming, ResumeMode,
    ShutdownReason, SteeringMessage, SteeringOutcome,
};
pub use self::compaction::{
    CompactionAccount, CompactionOptions, CompactionOutcome, CompactionReason, CompactionResult,
};
pub use self::config::{CodingAgentOptions, CodingAgentOptionsError};
pub use self::discovery::{
    MemoryDiscovery, MemoryRoot, ResolvedSkillDirs, SkillDiscovery, SkillSearch, SkillSearchBase,
};
pub use self::error::{CompactionError, Error, InterruptReason, Result, TaskKind};
pub use self::memory::{MemoryDocument, ProjectMemory};
pub use self::runtime::SteeringLease;
pub use self::skills::SkillExpansionError;
pub use self::types::{InputContent, InputSource};
