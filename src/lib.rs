//! Pebble is a coding-agent loop library built on lithos-llm. It runs the
//! turn loop that a coding agent needs — model calls, tool execution, session
//! state, and the events an application observes — and leaves transport,
//! storage, and process isolation to the embedding application.
//!
//! # Running one
//!
//! [`Session`] is the whole loop. [`Session::builder`] resolves a model through
//! the client's catalog, picks the harness that model expects, and freezes the
//! tools the session may call; [`Session::initialize`] loads what the session
//! was told and captures where it is working; [`Session::run`] answers one
//! input, however many rounds of model calls and tool calls that takes. A
//! [`SessionControlHandle`] steers or interrupts a run already in progress, and
//! [`Session::shutdown`] closes the session and joins everything it owns.
//!
//! [`Session::subscribe`] is how an application watches all of that happen. The
//! stream it hands out is bounded and lossy for a reader that falls behind, and
//! it ends when the session is shut down rather than when the session value is
//! dropped: a reader looping until `RecvError::Closed` finishes once
//! [`Session::shutdown`] has returned, so it can be joined before the session
//! goes. An application that must see every event installs an [`EventSink`]
//! instead.
//!
//! # Embedding pebble
//!
//! An application needs one dependency for all of this: `pebble`. What a
//! session is built from that pebble does not define is re-exported here — the
//! model layer as [`lithos_llm`], whose [`Client`](lithos_llm::Client),
//! catalog and credentials build the client a session talks through; the
//! [`async_trait`](macro@async_trait) attribute every seam trait is written
//! with; and [`CancellationToken`], which those seams take and
//! [`Session::cancel_token`] hands out. Naming lithos-llm as a dependency of
//! its own works too, but a version that resolves apart from pebble's produces
//! a `Client` a session will not take.
//!
//! # Repeating a failed call
//!
//! Pebble replays a turn only when it must: a stream that fails after the model
//! has already shown output cannot be reconnected underneath a reader without
//! showing that output twice, so the session withdraws it and replays. Every
//! other retry belongs to the client, which means the application has to
//! install one when it builds the client:
//!
//! ```no_run
//! # use lithos_llm::Client;
//! use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
//! use pebble::RetryEventObserver;
//!
//! # fn build(catalog: lithos_llm::catalog::Catalog) -> Result<(), Box<dyn std::error::Error>> {
//! let policy = RetryPolicy::exponential().max_attempts(4);
//! let build = Client::builder()
//!     .catalog(catalog)
//!     .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
//!     .build()?;
//! # let _ = build;
//! # Ok(())
//! # }
//! ```
//!
//! [`RetryEventObserver`] is what puts the client's own retries on the
//! session's event stream; without it a session still runs correctly and simply
//! never reports one. Give
//! [`SessionOptions::retry_policy`](SessionOptions::retry_policy) the same
//! policy, so one failure is spaced the same way whichever layer handles it.
//!
//! # Where the work lands
//!
//! Tools act through one seam, [`Environment`]: file access, content search,
//! and running commands. [`LocalEnvironment`] does that on this machine;
//! an application with a container or a remote workspace implements the trait
//! over that instead. Test doubles live in `pebble::test_support`, behind the
//! `test-util` feature.
//!
//! # What a session may call
//!
//! [`ToolRegistry`] holds the tools a session exposes. Pebble's own are named
//! by [`NativeTool`], and a [`ToolVocabulary`] decides which spelling of those
//! names one model family sees, so the same tool can be `read_file` to one
//! model and `Read` to another without anything else in the crate noticing.
//!
//! An [`AgentProfile`] decides which tools a session starts with, which
//! vocabulary they are named in, and what the system prompt says, because a
//! model trained inside a coding harness expects that harness back. Pebble
//! selects one from the catalog metadata of the model a session resolved to,
//! from the six it ships — one per [`AgentProfileKind`], covering the Claude,
//! Claude 5, Gemini CLI, OpenAI, Codex and Kimi Code families. An application
//! picks a harness by picking a model: the built-in ones are not otherwise
//! nameable, and a session takes no profile of an application's own.
//!
//! Pebble's own are the ones a coding agent cannot work without: reading,
//! writing and editing files ([`make_read_file_tool`],
//! [`make_write_file_tool`], [`make_edit_file_tool`],
//! [`make_read_many_files_tool`]), searching by content and by name
//! ([`make_grep_tool`], [`make_glob_tool`], [`make_list_dir_tool`]), running
//! commands ([`make_shell_tool`]), applying a patch
//! ([`make_apply_patch_tool`]), and fetching a page
//! ([`make_web_fetch_tool`]). Beside them are the tools a harness expects
//! rather than needs: keeping a plan ([`TodoRuntime`] and the tools around
//! it), asking a person a question ([`make_question_tool`]), loading a skill
//! ([`make_use_skill_tool`]), and searching the web
//! ([`make_web_search_tool`]). Every one of them acts through
//! the [`Environment`], and what they answer with — schemas, descriptions,
//! rendered output, failure messages — is part of pebble's contract, because a
//! model reads all of it.
//!
//! A round of calls runs through [`ToolDispatch`], which answers every call it
//! is given — including the ones it refuses — publishes what happened, and
//! bounds what a tool produced before the model or an application sees it.
//!
//! Some of those tools are given something the session owns rather than the
//! profile. A shell command's output tail reaches the event stream through the
//! [`Redactor`] an application installed with
//! [`SessionBuilder::redactor`] — pebble ships no secret detector, so without
//! one the tail is what the process wrote. `web_fetch` answers a prompt
//! about a page by asking the model named with
//! [`SessionBuilder::web_fetch_summarizer`]; without one it returns the page
//! and says the summary was unavailable. `web_search` goes to the
//! [`SearchProvider`] installed with [`SessionBuilder::search_provider`], and
//! is advertised only when there is one, because a search with no engine has
//! no answer worth giving. A question goes to the [`HumanInputProvider`] the
//! root session was given, and a child has none: a subagent reports back to
//! its parent instead of interrupting a person.
//!
//! An application decides what may run. A [`ToolAccessPolicy`] answers by name
//! and filters the tools a session even advertises; a [`ToolHookCallback`]
//! sees each call's arguments and may block it. Pebble installs neither, and
//! without them every registered tool is exposed. [`PermissionLevel`] and its
//! [`table`](PermissionLevel::auto_approves) are there to build a policy out
//! of, not a policy pebble applies.
//!
//! # Asking a person
//!
//! A session asks the person a question only where the application gave it a
//! [`HumanInputProvider`]. Without one no question tool is registered, so a
//! model cannot block a run waiting for an answer that will never come.
//!
//! # What the session is told
//!
//! Two kinds of written instruction reach a session, and pebble finds neither
//! by convention. [`load_memory`] reads the project instructions an application
//! names, within a fixed byte budget; [`discover_skills`] searches the
//! directories it is given for `SKILL.md` files. Name nothing and a session
//! carries neither, because guessing at a filename or walking up to a
//! repository root is the application's decision, not the library's.
//!
//! # Staying inside the window
//!
//! A long run outgrows the model's context window.
//! [`build_local_snapshot`] measures where a request stands, attributing tokens
//! to the prompt, the tools, memory, skills, and the conversation, and
//! [`check_context_usage`] says when the session is close enough to the edge to
//! act. [`compact_context`] then spends one call summarizing the older turns
//! and replaces them with the summary, keeping the file work a
//! [`FileTracker`] recorded. [`detect_loop`] catches the other way a run stops
//! progressing: the same tool calls, round after round.
//!
//! # Stability
//!
//! The serialized form of [`SessionEvent`] and [`AgentEvent`] is public API.
//! Evolution is additive — new variants and new optional fields — and renames
//! or removals are breaking changes with a documented migration. Consumers
//! should ignore members they do not know and tolerate variants they do not
//! know.
//!
//! Ignoring an unknown member is free: [`AgentEvent`] and its payloads skip
//! members they were not built to read. Tolerating an unknown *variant* is the
//! reader's own work, because a variant this build has never heard of fails
//! the whole envelope with it. A consumer that reads a stream a newer pebble
//! may have written keeps the payload as JSON until it has recognized it:
//!
//! ```
//! use pebble::AgentEvent;
//! use serde::Deserialize;
//!
//! /// The envelope, with the event left unread.
//! #[derive(Deserialize)]
//! struct Envelope {
//!     seq:   u64,
//!     event: serde_json::Value,
//! }
//!
//! # fn read(line: &str) -> Result<(), serde_json::Error> {
//! let envelope: Envelope = serde_json::from_str(line)?;
//! match serde_json::from_value::<AgentEvent>(envelope.event) {
//!     Ok(event) => println!("{}: {event:?}", envelope.seq),
//!     // Newer than this build: keep the sequence number, skip the payload.
//!     Err(_) => println!("{}: an event this build does not know", envelope.seq),
//! }
//! # Ok(())
//! # }
//! # read(r#"{"seq":7,"event":{"SomethingNewer":{}},"session_id":"ses_1"}"#).unwrap();
//! ```
//!
//! [`SessionRecord`] is the same kind of contract, with one addition: it
//! carries a [`format_version`](SessionRecord::format_version), so a record
//! written by an older pebble is still readable by a newer one.
//!
//! Both promises are about what an application *reads*. Pebble builds the
//! events and their payloads; an application receives them. So a new optional
//! field on a payload — [`ContextWindowSnapshot`], [`ToolSummary`],
//! [`MemoryFileSummary`], the todo props — is additive on the wire and
//! additive for code that reads it, and it is not covered for code outside
//! pebble that builds one with a struct literal naming every field. Where a
//! payload has a constructor or a `Default`, such as
//! [`TodoUpdatedProps::new`], build through it and fill the rest with `..`,
//! which a later field cannot break. [`SessionEvent`] is the envelope
//! everything travels in and the likeliest to grow, so it is
//! `#[non_exhaustive]`: build it with [`SessionEvent::new`] and the `with_*`
//! methods.

mod char_boundary;
mod compaction;
mod config;
mod context_window;
mod environment;
mod error;
mod event;
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
mod search;
mod session;
mod skills;
mod subagent;
mod task_reminder;
mod template;
mod tool;
mod tools;
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

/// The attribute pebble's own async traits are written with, so an
/// implementation of [`Environment`], [`EventSink`], [`HumanInputProvider`],
/// [`SearchProvider`] or [`ToolExecutor`] can be written without naming the
/// desugared lifetimes.
pub use async_trait::async_trait;
/// The model layer pebble is built on, re-exported whole because a session
/// cannot be built without a [`Client`](lithos_llm::Client) and an application
/// resolving its own copy could resolve a different version.
pub use lithos_llm;
/// How a failed model call is spaced before it is tried again, carried by
/// [`SessionOptions::retry_policy`].
///
/// The same type the client's
/// [`RetryMiddleware`](lithos_llm::middleware::RetryMiddleware) is built with,
/// so one policy can be given to both.
#[doc(inline)]
pub use lithos_llm::middleware::RetryPolicy;
/// One part of a message or of a tool's result.
#[doc(inline)]
pub use lithos_llm::types::ContentPart;
/// The model-layer error category carried by [`ErrorData::llm_kind`].
#[doc(inline)]
pub use lithos_llm::types::ErrorKind as LlmErrorKind;
/// How hard the model should think, carried by
/// [`SessionOptions::reasoning_effort`].
#[doc(inline)]
pub use lithos_llm::types::ReasoningEffort;
/// Whether repeating a failed model call is safe, carried by
/// [`ErrorData::retry`].
#[doc(inline)]
pub use lithos_llm::types::RetryClassification;
/// The latency or cost tier to ask for, carried by
/// [`SessionOptions::speed`].
#[doc(inline)]
pub use lithos_llm::types::Speed;
/// One answered tool call, as history and the model read it.
#[doc(inline)]
pub use lithos_llm::types::ToolResult;
/// A tool invocation the model asked for, answered by [`ToolDispatch`].
#[doc(inline)]
pub use lithos_llm::types::{ToolCall, ToolCallKind};
/// What the model is told about one tool, carried by [`RegisteredTool`].
#[doc(inline)]
pub use lithos_llm::types::{ToolDefinition, ToolDefinitionKind};
/// The signal a session passes into the work it starts, so an application's
/// [`HumanInputProvider`] or [`ToolExecutor`] can stop when the round it is
/// running in does. [`Session::cancel_token`] hands out the session's own.
pub use tokio_util::sync::CancellationToken;

pub use self::compaction::{
    CompactionRequest, ContextEstimate, ContextEstimateMethod, check_context_usage,
    compact_context, estimate_active_context_usage, render_turns_for_summary,
};
pub use self::config::{
    NativeToolOptions, SessionOptions, ToolAccess, ToolAccessPolicy, ToolApprovalAdapter,
    ToolApprovalFn, ToolExposureMode, ToolHookCallback, ToolHookDecision,
};
pub use self::context_window::{
    ACTIVATED_SKILL_WARNING, ContextWindowInput, build_local_snapshot,
    context_window_from_response_usage, scaled_snapshot,
};
pub use self::environment::{
    CallerEnvPolicy, DEFAULT_EXEC_OUTPUT_TAIL_BYTES, DirEntry, EnvResult, Environment,
    EnvironmentError, EnvironmentErrorKind, ExecOutcome, ExecRequest, ExecResult, GrepOptions,
    LocalEnvironment, format_lines_numbered,
};
pub use self::error::{CompactionError, Error, ErrorData, ErrorKind, InterruptReason, Result};
pub use self::event::{
    DEFAULT_EVENT_CAPACITY, Emitter, EventCapacity, EventOptions, EventPump, EventSink,
    EventSinkError, OutputCaptureStats, SessionBoundEmitter,
};
pub use self::file_tracker::FileTracker;
pub use self::history::History;
pub use self::human_input::{
    Answer, AnswerStatus, HumanInputError, HumanInputProvider, Question, QuestionKind,
    QuestionOption, is_question_tool,
};
pub use self::loop_detection::detect_loop;
pub use self::memory::{MEMORY_BUDGET_BYTES, MemoryDocument, load_memory};
pub use self::profile::{
    AgentProfile, DEFAULT_CONTEXT_WINDOW_TOKENS, EnvContext, ModelFacts, SubagentSupport,
};
pub use self::reasoning::ReasoningOutput;
pub use self::record::{SESSION_RECORD_FORMAT_VERSION, SessionRecord, StoredMessage};
pub use self::redact::{NoRedaction, Redactor};
pub use self::search::{SearchError, SearchErrorKind, SearchProvider, SearchRequest, SearchResult};
pub use self::session::{
    CompletionCoordinator, InterruptReasonHandle, RetryEventObserver, RunOptions, RunTiming,
    Session, SessionBuildError, SessionBuilder, SessionControlHandle, ShutdownReason, SteeringItem,
    SteeringMessage,
};
pub use self::skills::{
    ExpandedInput, Skill, SkillExpansion, SkillExpansionError, SkillParseError, discover_skills,
    expand_skill, format_skills_prompt_section, parse_skill,
};
pub use self::subagent::{ChildSessionSpec, SessionFactory, SubagentLimits};
pub use self::task_reminder::{TASK_REMINDER_TEXT, maybe_task_reminder};
pub use self::tool::{
    AgentEventEmitter, NativeTool, RegisteredTool, StaticEnvProvider, ToolContext,
    ToolDefinitionWithSource, ToolDispatch, ToolEnvProvider, ToolError, ToolExecutor, ToolRegistry,
    ToolVocabulary, canonical_tool_name, known_tool_category, tool_category, validate_tool_args,
};
pub use self::tools::{
    Change, Hunk, PatchOperation, TodoRuntime, WebFetchSummarizer, apply_patch_operations,
    grep_result_path, make_anthropic_question_tool, make_apply_patch_tool,
    make_claude5_question_tool, make_edit_file_tool, make_glob_tool, make_grep_tool,
    make_list_dir_tool, make_openai_question_tool, make_question_tool, make_read_file_tool,
    make_read_many_files_tool, make_shell_tool, make_shell_tool_with_options,
    make_task_create_tool, make_task_get_tool, make_task_list_tool, make_task_update_tool,
    make_todo_list_tool, make_update_plan_tool, make_use_skill_tool,
    make_use_skill_tool_for_vocabulary, make_web_fetch_tool, make_web_search_tool,
    make_write_file_tool, parse_apply_patch,
};
pub use self::truncation::{
    DEFAULT_TOOL_OUTPUT_RETENTION_BYTES, DEFAULT_TOOL_OUTPUT_SERIALIZED_BYTES, OutputBudgets,
    ToolOutputLimits, TruncationMode, truncate_lines, truncate_output, truncate_tool_output,
};
pub use self::types::{
    Actor, AgentEvent, AgentProfileKind, CommandTermination, ContextWindowBreakdownItem,
    ContextWindowCategory, ContextWindowCountMethod, ContextWindowSnapshot, ContextWindowStaleness,
    ContextWindowWarning, CostSource, ExecOutputTail, ExecOutputTailTrace,
    INITIAL_SUBAGENT_GENERATION, LlmOutputKind, LlmRetryPhase, MemoryFileSummary, Message,
    PermissionLevel, SessionEvent, SessionState, SkillActivationSource, SkillSummary,
    TodoCreatedProps, TodoDeletedProps, TodoListKind, TodoListProjection, TodoPatch,
    TodoProjection, TodoStatus, TodoUpdatedProps, TokenUsage, ToolCategory, ToolErrorKind,
    ToolSource, ToolSummary,
};
