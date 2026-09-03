//! Internal coding-agent state and adapters.
//!
//! [`CodingRuntime`] holds the coding-specific state around the
//! provider-neutral [`Agent`]. The public [`CodingAgent`](crate::CodingAgent)
//! facade owns its lifecycle and is the only application entry point.

mod control;
#[cfg(test)]
mod loop_tests;
mod retry;
#[cfg(test)]
pub(crate) mod testing;
mod turn;

use std::collections::HashMap;
use std::fmt;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::SystemTime;

use lithos_llm::Client;
use lithos_llm::catalog::{Metadata, ModelHandle};
use lithos_llm::resolver::ResolvedRoute;
use lithos_llm::types::{
    Error as LlmError, ErrorKind as LlmErrorKind, ReasoningEffort, Request, Speed,
};
use pebble_agent::{Agent, AgentControlHandle, ToolMiddleware};
use serde::Deserialize;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

pub use self::control::SteeringLease;
pub(crate) use self::control::{actor_from_attribution, input_message, steering_message};
pub use self::retry::RetryEventObserver;
use self::turn::{CodingAgentBridge, ConversationState};
pub(crate) use crate::coding_agent::{
    CodingAgentBuildError, CodingInput, PromptTiming, ResumeMode, ShutdownReason,
};
use crate::compaction::{
    CompactionControl, CompactionOptions, CompactionOutcome, CompactionReason, CompactionRequest,
    compact_context, estimate_active_context_usage,
};
use crate::config::CodingAgentOptions;
use crate::context_window::{memory_prompt_tokens, skills_prompt_tokens};
use crate::environment::{Environment, ExecRequest};
use crate::error::{Error, ErrorData, ErrorKind, InterruptReason, Result, TaskKind};
use crate::event::{Emitter, EventCapacity, EventOptions, EventPump, EventSink, EventSinkTimeout};
use crate::file_tracker::FileTracker;
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::memory::{MEMORY_BUDGET_BYTES, MemoryDocument, load_memory};
use crate::profile::{AgentProfile, EnvContext, ModelFacts, SubagentSupport, builtin_profile};
use crate::profiles::{FileEditToolKind, ProfileDeps};
use crate::prompt_transform::{SystemPromptContext, SystemPromptTransform};
use crate::record::{RecordMigrationError, SessionRecord};
use crate::redact::{NoRedaction, Redactor};
use crate::search::SearchProvider;
use crate::skills::{Skill, SkillExpansion, discover_skills};
use crate::subagent::{
    ChildDeps, ChildIdentity, ChildObserver, OpenSessions, SubagentEventCallback, SubagentLimits,
    SubagentOptions, SubagentSupervisor,
};
use crate::tool::{
    NativeTool, RegisteredTool, StaticEnvProvider, ToolDefinitionWithSource, ToolEnvProvider,
    ToolRegistry,
};
use crate::tools::skill::make_use_skill_tool_for_vocabulary;
use crate::tools::{WebFetchSummarizer, make_question_tool, make_web_search_tool};
#[cfg(test)]
use crate::types::PermissionLevel;
use crate::types::{
    AgentProfileKind, CodingAgentEvent, CodingAgentState, CodingEvent, ContextWindowSnapshot,
    MemoryFileSummary, Message, SkillSummary, TokenUsage, ToolSummary, rfc3339_millis,
};

/// The catalog metadata namespace pebble reads.
const METADATA_NAMESPACE: &str = "pebble";

/// How long a probe run inside the environment may take.
const PROBE_TIMEOUT_MS: u64 = 5_000;

/// A live session's state, warm, for a successor in the same process.
///
/// The record is the durable part. The rest is what
/// [`CodingRuntime::initialize`] derives from the environment and the options,
/// carried so the successor does not derive it again.
#[derive(Clone, Debug)]
pub(crate) struct WarmState {
    pub(crate) record: SessionRecord,
    pub(crate) system_prompt: String,
    pub(crate) skills: Vec<Skill>,
    pub(crate) memory_summaries: Vec<MemoryFileSummary>,
    pub(crate) memory_tokens: u64,
    pub(crate) skills_tokens: u64,
    pub(crate) file_tracker: FileTracker,
    pub(crate) activated_skill_context_observed: bool,
    pub(crate) context_window: Option<ContextWindowSnapshot>,
}

/// What one prompt accumulated across every input it processed.
#[derive(Clone, Copy, Debug, Default)]
struct PromptTotals {
    timing:          PromptTiming,
    usage:           TokenUsage,
    cost_usd_micros: Option<u64>,
}

/// The `pebble` namespace of a catalog entry.
///
/// Unknown keys are ignored, because the namespace grows and an older pebble
/// must keep reading a catalog a newer one wrote.
#[derive(Debug, Default, Deserialize)]
struct PebbleMetadata {
    /// Which harness the model expects.
    #[serde(default)]
    profile:              Option<String>,
    /// How the model's training data is dated, as a person would write it.
    #[serde(default)]
    knowledge_cutoff:     Option<String>,
    /// Whether the model reasons without being asked to, where the capabilities
    /// alone do not say.
    #[serde(default)]
    reasoning_by_default: Option<bool>,
}

/// Collects everything a session needs and builds it.
///
/// The builder owns the tool registry: it asks the profile for its tools,
/// merges whatever the application registered, and freezes the result into the
/// session. Nothing mutates a registry afterwards.
#[must_use = "a builder does nothing until `build` is called"]
pub(crate) struct CodingRuntimeBuilder {
    client:               Client,
    model:                Option<String>,
    environment:          Option<Arc<dyn Environment>>,
    tools:                Vec<RegisteredTool>,
    tool_middleware:      Vec<Arc<dyn ToolMiddleware>>,
    human_input:          Option<Arc<dyn HumanInputProvider>>,
    tool_env_provider:    Option<Arc<dyn ToolEnvProvider>>,
    redactor:             Arc<dyn Redactor>,
    web_fetch_summarizer: Option<String>,
    search_provider:      Option<Arc<dyn SearchProvider>>,
    options:              CodingAgentOptions,
    events:               EventOptions,
    profile:              Option<Arc<dyn AgentProfile>>,
    prompt_transform:     Option<Arc<dyn SystemPromptTransform>>,
    subagents_enabled:    bool,
    child_observer:       Option<ChildObserver>,
    subagent_limits:      SubagentLimits,
    child:                Option<ChildIdentity>,
}

impl CodingRuntimeBuilder {
    /// Starts a session that talks to the model through `client`.
    fn new(client: Client) -> Self {
        Self {
            client,
            model: None,
            environment: None,
            tools: Vec::new(),
            tool_middleware: Vec::new(),
            human_input: None,
            tool_env_provider: None,
            redactor: Arc::new(NoRedaction),
            web_fetch_summarizer: None,
            search_provider: None,
            options: CodingAgentOptions::default(),
            events: EventOptions::default(),
            profile: None,
            prompt_transform: None,
            subagents_enabled: false,
            child_observer: None,
            subagent_limits: SubagentLimits::default(),
            child: None,
        }
    }

    /// Lets the application adjust the system prompt the profile writes.
    ///
    /// The transform sees the default prompt and the context it was written
    /// from, and answers with the default, an addition, or a replacement. It
    /// applies to this root session only: a child runs its parent's profile and
    /// prompt.
    pub(crate) fn system_prompt_transform(
        mut self,
        transform: Arc<dyn SystemPromptTransform>,
    ) -> Self {
        self.prompt_transform = Some(transform);
        self
    }

    /// Names the model, as the client's catalog spells it.
    ///
    /// Anything the catalog resolver accepts works — a model id, an alias, a
    /// `provider/model` pair, or `default`. The session pins whatever it
    /// resolves to, so every round of the prompt reaches the same model.
    pub(crate) fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets where the session's tools act.
    pub(crate) fn environment(mut self, environment: Arc<dyn Environment>) -> Self {
        self.environment = Some(environment);
        self
    }

    /// Whether a model has been named on this builder.
    pub(crate) const fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Adds tools on top of the ones the profile contributes.
    ///
    /// The registry renames pebble's own tools into the profile's vocabulary as
    /// they arrive, so a built-in registered here still reaches the model under
    /// the name that model expects.
    pub(crate) fn tools(mut self, tools: impl IntoIterator<Item = RegisteredTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Adds one tool middleware to this session and its descendants.
    pub(crate) fn tool_middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.tool_middleware.push(middleware);
        self
    }

    /// Sets where the session asks a person a question.
    ///
    /// Without one, no question tool is registered, so the model cannot park a
    /// prompt waiting for an answer nobody will give.
    pub(crate) fn human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Sets where a tool call's extra environment variables come from.
    ///
    /// Resolved once per tool round, so a credential that expires mid-prompt is
    /// fetched again rather than reused.
    pub(crate) fn tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env_provider = Some(provider);
        self
    }

    /// Sets fixed extra environment variables for every tool call.
    pub(crate) fn tool_env(self, env: HashMap<String, String>) -> Self {
        self.tool_env_provider(Arc::new(StaticEnvProvider(env)))
    }

    /// Sets what strips secrets out of the process output the session
    /// publishes.
    ///
    /// Pebble ships no secret detector, so without one the tail of a command's
    /// output reaches the event stream exactly as the command wrote it. What
    /// the model reads is never redacted: it is the same text the person at a
    /// terminal would have seen.
    pub(crate) fn redactor(mut self, redactor: Arc<dyn Redactor>) -> Self {
        self.redactor = redactor;
        self
    }

    /// Lets `web_fetch` answer a prompt about a page by asking `model`.
    ///
    /// The selector is resolved through this session's client, so the
    /// summarizing model can be smaller and cheaper than the one running the
    /// session. Without one, a `web_fetch` call carrying a prompt returns the
    /// page and says the summary was unavailable.
    ///
    /// The built-in profiles' fetch tool captures the summarizer when the
    /// profile is constructed; an application registering
    /// [`make_web_fetch_tool`](crate::tools::make_web_fetch_tool) itself passes
    /// a [`WebFetchSummarizer`](crate::tools::WebFetchSummarizer) directly
    /// instead.
    pub(crate) fn web_fetch_summarizer(mut self, model: impl Into<String>) -> Self {
        self.web_fetch_summarizer = Some(model.into());
        self
    }

    /// Sets where the session's web searches go.
    ///
    /// Pebble talks to no search engine of its own: the application implements
    /// [`SearchProvider`] over whatever it has. Registering one is the whole
    /// switch — the session advertises `web_search`, built on pebble's schema
    /// and answering in pebble's format whichever engine is underneath — and a
    /// session without one advertises no search tool, so a model is never told
    /// it can search and then refused. A child inherits its parent's provider.
    ///
    /// A profile whose model family expects a different search tool registers
    /// its own, which replaces this one.
    pub(crate) fn search_provider(mut self, provider: Arc<dyn SearchProvider>) -> Self {
        self.search_provider = Some(provider);
        self
    }

    /// Sets how the session behaves.
    pub(crate) fn options(mut self, options: CodingAgentOptions) -> Self {
        self.options = options;
        self
    }

    /// Records every event durably before any subscriber sees it.
    ///
    /// A sink that refuses an event stops the prompt and closes the session: a
    /// session that cannot record what it did is worse than one that stops. The
    /// prompt that noticed reports [`Error::EventSink`]; every call after it
    /// reports [`Error::SessionClosed`].
    pub(crate) fn event_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.events.sink = Some(sink);
        self
    }

    /// Sets the pending event queue and live subscription capacity.
    pub(crate) fn event_capacity(mut self, capacity: impl Into<EventCapacity>) -> Self {
        self.events.capacity = capacity.into();
        self
    }

    /// Sets the longest one durable sink write may take.
    pub(crate) fn event_sink_timeout(mut self, timeout: impl Into<EventSinkTimeout>) -> Self {
        self.events.sink_timeout = timeout.into();
        self
    }

    /// Lets this session spawn children, built by `factory`.
    ///
    /// Registering a factory is the whole switch: the profile's subagent tools
    /// are registered only when there is one, and a session without one answers
    /// every spawn with a tool error. The builder also wires the supervisor to
    /// this session's event pipeline, so a child's events cannot be lost by an
    /// application that forgot to connect them.
    ///
    /// Pebble builds the [`ChildAgentSpec`](crate::subagent::ChildAgentSpec)
    /// each call receives from this session, so a child inherits the
    /// environment, the inheritable tools, the
    /// tool middleware its parent had, and never a
    /// [`HumanInputProvider`]: a child cannot ask a person a question.
    pub(crate) fn subagents(mut self, options: SubagentOptions) -> Self {
        self.subagents_enabled = options.is_enabled();
        self.subagent_limits = options.limits();
        self
    }

    /// Sees each child this session's tree builds, for the crate's own tests.
    #[cfg(test)]
    pub(crate) fn observe_children(mut self, observer: ChildObserver) -> Self {
        self.subagents_enabled = true;
        self.child_observer = Some(observer);
        self
    }

    /// Builds this session as a child of another.
    ///
    /// Crate-internal: the identity, the depth, and the tree's open-session
    /// budget come from the spawning session, never from an application.
    pub(crate) fn child_of(mut self, child: ChildIdentity) -> Self {
        self.child = Some(child);
        self
    }

    /// Overrides the profile the catalog would select.
    ///
    /// Crate-internal on purpose: an application picks a harness by picking a
    /// model, never by naming one, so this exists for pebble's own tests and
    /// for a child session, which is built with the harness its parent already
    /// resolved.
    pub(crate) fn with_profile(mut self, profile: Arc<dyn AgentProfile>) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Builds the session.
    ///
    /// # Errors
    ///
    /// Returns [`CodingAgentBuildError`] when a dependency is missing, the
    /// model selector resolves to nothing, or the resolved model's catalog
    /// entry names no harness pebble can run.
    pub(crate) fn build(self) -> StdResult<CodingRuntime, CodingAgentBuildError> {
        self.build_with_id(new_session_id(), SystemTime::now())
    }

    fn build_with_id(
        self,
        id: String,
        created_at: SystemTime,
    ) -> StdResult<CodingRuntime, CodingAgentBuildError> {
        let selector = self.model.ok_or(CodingAgentBuildError::MissingModel)?;
        let environment = self
            .environment
            .ok_or(CodingAgentBuildError::MissingEnvironment)?;

        let route = resolve_route(&self.client, &selector)?;
        let handle = route.handle();
        let metadata = effective_metadata(&route, &handle)?;
        // The catalog's capabilities say what the model can do; the `pebble`
        // namespace is where a row says what it does by default, for the
        // always-reasoning models whose capabilities cannot tell.
        let mut facts = ModelFacts::from_catalog_model(route.model());
        if let Some(reasons_by_default) = metadata.reasoning_by_default {
            facts = facts.with_reasons_by_default(reasons_by_default);
        }
        // The catalog decides which harness the model expects whether or not
        // an implementation was injected, so a model that names none is
        // refused the same way either way.
        let kind = profile_kind(&metadata, &handle)?;
        // Built here rather than inside the profile: the tool is the
        // application's answer — someone to ask — crossed with the harness's,
        // and a harness with no question tool of its own, Gemini, answers
        // `None` however the session was configured. The prompt reads it back
        // out of the registry rather than being told, because a child session
        // runs this same profile with nobody to ask.
        let question_tool = self
            .human_input
            .as_ref()
            .and_then(|_| make_question_tool(kind));
        // Built before the profile, because the profile's `web_fetch` tool
        // captures it at construction the way a search tool captures its
        // engine.
        let web_fetch_summarizer = self
            .web_fetch_summarizer
            .map(|model| Arc::new(WebFetchSummarizer::new(self.client.clone(), model)));
        let deps = ProfileDeps {
            provider_display_name: route.provider().display_name().to_owned(),
            file_edit_tool: FileEditToolKind::for_codec(route.provider().codec()),
            search_provider: self.search_provider.clone(),
            web_fetch_summarizer,
        };
        let profile = self.profile.unwrap_or_else(|| builtin_profile(kind, &deps));

        let mut registry = ToolRegistry::with_vocabulary(profile.tool_vocabulary());
        let profile_tools = profile.base_tools();
        // Built-in profiles contribute the search shape their models expect.
        // An injected profile that contributes no search tool gets the
        // canonical one, preserving the builder's public search-provider
        // contract without replacing a profile's own definition.
        if let Some(provider) = &self.search_provider
            && !profile_tools
                .iter()
                .any(|tool| tool.definition.name == NativeTool::WebSearch.canonical_name())
        {
            registry.register(make_web_search_tool(Arc::clone(provider)));
        }
        for tool in profile_tools {
            registry.register(tool);
        }
        // Root-only, and only where the application named somewhere to ask: a
        // child reports back to its parent rather than interrupting a person,
        // and a spec carries no `HumanInputProvider` for exactly that reason.
        if let Some(tool) = question_tool {
            registry.register(tool);
        }
        for tool in &self.tools {
            registry.register(tool.clone());
        }

        // A child was placed in its tree by whoever spawned it; a root names
        // itself and starts the tree's budget.
        let (parent_session_id, root_session_id, depth, open_sessions, observer, inherited_emitter) =
            match self.child {
                Some(child) => (
                    Some(child.parent_session_id),
                    child.root_session_id,
                    child.depth,
                    child.open_sessions,
                    child.observer,
                    Some(child.event_emitter),
                ),
                None => (
                    None,
                    id.clone(),
                    0,
                    OpenSessions::root(self.subagent_limits),
                    self.child_observer,
                    None,
                ),
            };

        let (emitter, pump) = if let Some(emitter) = inherited_emitter {
            (emitter, None)
        } else {
            let (emitter, pump) = EventPump::new(self.events);
            let emitter = emitter.in_stream(root_session_id.clone());
            (emitter, Some(tokio::spawn(pump.run())))
        };

        let supervisor = self.subagents_enabled.then(|| {
            SubagentSupervisor::new(Arc::new(ChildDeps {
                client: self.client.clone(),
                model_selector: handle.to_string(),
                profile: Arc::clone(&profile),
                environment: Arc::clone(&environment),
                tools: self.tools,
                tool_middleware: self.tool_middleware.clone(),
                options: child_options(&self.options),
                tool_env_provider: self.tool_env_provider.clone(),
                redactor: Arc::clone(&self.redactor),
                search_provider: self.search_provider.clone(),
                event_emitter: emitter.clone(),
                observer,
                open_sessions,
                depth,
            }))
        });
        // Asked for whether or not there is a supervisor: a profile answers
        // with no tools when subagents are off, and this is the only place a
        // profile's subagent family reaches the registry.
        for tool in profile.subagent_tools(&SubagentSupport::new(depth, supervisor.clone())) {
            registry.register(tool);
        }

        let state = StateMachine::new(emitter.clone(), id.clone());

        let session = CodingRuntime {
            root_session_id,
            parent_session_id,
            id,
            created_at,
            config: self.options,
            conversation: Arc::new(Mutex::new(ConversationState::new(History::default()))),
            emitter,
            pump,
            state,
            ended: false,
            client: self.client,
            profile,
            provider: handle.provider().as_str().to_owned(),
            model: handle.model().as_str().to_owned(),
            model_selector: handle.to_string(),
            facts,
            knowledge_cutoff: metadata.knowledge_cutoff.unwrap_or_default(),
            registry,
            tool_middleware: self.tool_middleware,
            env: environment,
            human_input: self.human_input,
            tool_env_provider: self.tool_env_provider,
            redactor: self.redactor,
            agent_control: AgentControlHandle::detached(),
            cancel_token: CancellationToken::new(),
            interrupt_reason: Arc::new(Mutex::new(None)),
            compaction: CompactionControl::default(),
            skills: Vec::new(),
            memory_summaries: Vec::new(),
            memory_tokens: 0,
            skills_tokens: 0,
            system_prompt: String::new(),
            subagents: supervisor,
            prompt_transform: self.prompt_transform,
            coding_agent: None,
            coding_bridge: None,
        };

        // Wired here rather than by the application: a supervisor with no
        // callback loses every child event, silently.
        if let Some(supervisor) = session.subagents.as_ref() {
            supervisor.set_event_callback(session.sub_agent_event_callback());
        }

        Ok(session)
    }
}

/// What a child session inherits from its parent's options.
///
/// Everything that bounds or governs the child comes across unchanged — the
/// tool middleware, the permission level, the output budgets, the
/// wall-clock budget — so a factory cannot be handed anything wider than the
/// parent had. What does not come across is what the root loads once: the
/// memory files and the skill directories. A child is given a task, not a
/// project briefing, and paying for the briefing again in every child is how a
/// tree of agents spends a context window on nothing.
fn child_options(parent: &CodingAgentOptions) -> CodingAgentOptions {
    CodingAgentOptions {
        memory_files: Vec::new(),
        skill_dirs: Vec::new(),
        ..parent.clone()
    }
}

/// Resolves the selector the way the session's own calls will.
fn resolve_route(
    client: &Client,
    selector: &str,
) -> StdResult<ResolvedRoute, CodingAgentBuildError> {
    let probe = Request::builder()
        .model(selector)
        .user("probe")
        .build()
        .map_err(|source| CodingAgentBuildError::Selector {
            selector: selector.to_owned(),
            source,
        })?;
    client
        .resolve_route(&probe)
        .map_err(|source| CodingAgentBuildError::ModelSelection {
            selector: selector.to_owned(),
            source,
        })
}

/// The `pebble` namespace for a route, with the model's answers taking
/// precedence over the provider's.
///
/// Precedence is per member, not per namespace: a model that carries a
/// `pebble` block naming only its knowledge cutoff still takes its profile from
/// the provider.
fn effective_metadata(
    route: &ResolvedRoute,
    handle: &ModelHandle,
) -> StdResult<PebbleMetadata, CodingAgentBuildError> {
    let model = read_metadata(route.model().metadata(), handle)?;
    let provider = read_metadata(route.provider().metadata(), handle)?;
    Ok(PebbleMetadata {
        profile:              model.profile.or(provider.profile),
        knowledge_cutoff:     model.knowledge_cutoff.or(provider.knowledge_cutoff),
        reasoning_by_default: model.reasoning_by_default.or(provider.reasoning_by_default),
    })
}

fn read_metadata(
    metadata: &Metadata,
    handle: &ModelHandle,
) -> StdResult<PebbleMetadata, CodingAgentBuildError> {
    metadata
        .namespace::<PebbleMetadata>(METADATA_NAMESPACE)
        .map(Option::unwrap_or_default)
        .map_err(|source| CodingAgentBuildError::InvalidProfileMetadata {
            model: handle.to_string(),
            source,
        })
}

/// The harness the catalog says this model expects.
fn profile_kind(
    metadata: &PebbleMetadata,
    handle: &ModelHandle,
) -> StdResult<AgentProfileKind, CodingAgentBuildError> {
    let named = metadata.profile.as_deref().ok_or_else(|| {
        CodingAgentBuildError::MissingProfileMetadata {
            model: handle.to_string(),
        }
    })?;
    AgentProfileKind::ALL
        .iter()
        .copied()
        .find(|kind| kind.as_str() == named)
        .ok_or_else(|| CodingAgentBuildError::UnknownProfile {
            model:   handle.to_string(),
            profile: named.to_owned(),
        })
}

/// A fresh session identifier.
fn new_session_id() -> String {
    format!("ses_{}", uuid::Uuid::new_v4())
}

/// Where an outside task names why it is about to cancel a prompt.
///
/// [`CodingRuntime::interrupt_reason_handle`] hands one out before the prompt
/// starts, so a watchdog holding it can say what it is doing — a budget spent,
/// a deadline passed — and the prompt then ends with that reason rather than a
/// plain cancellation. Cloning is cheap and every clone names the same session.
///
/// First writer wins: the reason a prompt reports is the first one recorded,
/// and [`record`](Self::record) is the only way to write, so a later task
/// cannot overwrite what already explains the interrupt.
#[derive(Clone, Debug)]
pub(crate) struct InterruptReasonHandle(Arc<Mutex<Option<InterruptReason>>>);

impl InterruptReasonHandle {
    /// Records `reason` unless one is already recorded.
    ///
    /// Answers whether this call is the one the prompt will report.
    pub(crate) fn record(&self, reason: InterruptReason) -> bool {
        let mut recorded = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if recorded.is_some() {
            return false;
        }
        *recorded = Some(reason);
        true
    }

    /// The reason recorded so far, if any.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn reason(&self) -> Option<InterruptReason> {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One conversation with one model.
///
/// A session is used from one place at a time: [`CodingRuntime::prompt`]
/// borrows it for the length of a prompt. Everything that has to reach a live
/// prompt — steering, interrupts, follow-up input, the event stream — comes
/// from a handle taken before the prompt starts.
#[must_use = "call `shutdown` to stop the session and join what it owns"]
pub(crate) struct CodingRuntime {
    id:                String,
    /// The root of this session's tree. A root session names itself; a child
    /// inherits its parent's root, which is how root-scoped tools — one shared
    /// todo list across a tree of agents — know where they belong.
    root_session_id:   String,
    /// The session that spawned this one, for a child. A root has none.
    parent_session_id: Option<String>,
    created_at:        SystemTime,
    config:            CodingAgentOptions,
    /// The conversation and what the current prompt has accumulated, shared
    /// with the bridge that writes it.
    conversation:      Arc<Mutex<ConversationState>>,
    emitter:           Emitter,
    /// The event pump, until [`CodingRuntime::shutdown`] joins it.
    pump:              Option<JoinHandle<Result<()>>>,
    /// What the session is doing, shared with the bridge that runs the tool
    /// round.
    state:             StateMachine,
    ended:             bool,
    client:            Client,
    profile:           Arc<dyn AgentProfile>,
    provider:          String,
    model:             String,
    /// What every request names, which is the resolved `provider/model` pair
    /// rather than the selector the application gave, so no round can drift to
    /// a different model than the one whose harness the session is running.
    model_selector:    String,
    facts:             ModelFacts,
    knowledge_cutoff:  String,
    registry:          ToolRegistry,
    tool_middleware:   Vec<Arc<dyn ToolMiddleware>>,
    env:               Arc<dyn Environment>,
    human_input:       Option<Arc<dyn HumanInputProvider>>,
    tool_env_provider: Option<Arc<dyn ToolEnvProvider>>,
    /// What strips secrets out of the process output this session publishes.
    redactor:          Arc<dyn Redactor>,
    /// The control the generic agent is bound to when it is built, so a
    /// handle can be given out before the first prompt creates the agent.
    agent_control:     AgentControlHandle,
    /// Ends the whole prompt. Distinct from the round token, which ends one
    /// turn.
    cancel_token:      CancellationToken,
    interrupt_reason:  Arc<Mutex<Option<InterruptReason>>>,
    compaction:        CompactionControl,
    skills:            Vec<Skill>,
    memory_summaries:  Vec<MemoryFileSummary>,
    /// What the memory files and the skills section contribute to the system
    /// prompt, measured once at initialization: both are fixed for the
    /// session's life, and every round's context snapshot reads them.
    memory_tokens:     u64,
    skills_tokens:     u64,
    system_prompt:     String,
    subagents:         Option<SubagentSupervisor>,
    /// The application's adjustment to the system prompt, applied once when
    /// the session initializes.
    prompt_transform:  Option<Arc<dyn SystemPromptTransform>>,
    /// The provider-neutral conversation loop, created after initialization on
    /// the first prompt and retained for the rest of the session.
    coding_agent:      Option<Agent>,
    /// Coding state and durable projection shared with `coding_agent`.
    coding_bridge:     Option<Arc<CodingAgentBridge>>,
}

impl fmt::Debug for CodingRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodingRuntime")
            .field("id", &self.id)
            .field("root_session_id", &self.root_session_id)
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("profile", &self.profile.profile_kind())
            .field("state", &self.state.current())
            .field("ended", &self.ended)
            .field("turns", &self.conversation().history.len())
            .finish_non_exhaustive()
    }
}

impl CodingRuntime {
    /// Starts building a session that talks to the model through `client`.
    pub(crate) fn builder(client: Client) -> CodingRuntimeBuilder {
        CodingRuntimeBuilder::new(client)
    }

    /// Rebuilds a stored session, ready to carry on where it left off.
    ///
    /// The record supplies the identity, the conversation, and where the event
    /// stream had got to; `deps` supplies everything a record cannot hold — the
    /// client, the environment, the tools, the options. `mode` decides the
    /// model: the exact route the record names, restored without guessing, or
    /// a selector the caller chose for failover. Event numbering continues from
    /// the record, so the root session tree's events stay uniquely numbered
    /// across restarts.
    ///
    /// # Errors
    ///
    /// Returns [`CodingAgentBuildError`] for the same reasons
    /// [`CodingRuntimeBuilder::build`] does;
    /// [`UnsupportedRecord`](CodingAgentBuildError::UnsupportedRecord) for a
    /// format version this build does not read; and, when the recorded model is
    /// asked for,
    /// [`RecordedRouteMissing`](CodingAgentBuildError::RecordedRouteMissing)
    /// for a record that names no route,
    /// [`RecordedRouteUnavailable`](CodingAgentBuildError::RecordedRouteUnavailable)
    /// for one the client cannot reach, and
    /// [`RecordedRouteMismatch`](CodingAgentBuildError::RecordedRouteMismatch)
    /// if the resolver answered with a different model.
    pub(crate) fn from_record(
        record: SessionRecord,
        mode: &ResumeMode,
        deps: CodingRuntimeBuilder,
    ) -> StdResult<Self, CodingAgentBuildError> {
        let record = record.migrate().map_err(|error| match error {
            RecordMigrationError::UnsupportedVersion { version, supported } => {
                CodingAgentBuildError::UnsupportedRecord { version, supported }
            }
        })?;

        let recorded = match mode {
            ResumeMode::RecordedModel => {
                let route = record.recorded_route().ok_or_else(|| {
                    CodingAgentBuildError::RecordedRouteMissing {
                        session_id: record.session_id.clone(),
                    }
                })?;
                Some(route)
            }
            ResumeMode::UseModel(_) => None,
        };
        let selector = match mode {
            ResumeMode::RecordedModel => recorded.clone().unwrap_or_default(),
            ResumeMode::UseModel(selector) => selector.clone(),
        };

        let mut deps = deps;
        deps.model = Some(selector);
        deps.events.resume_after_seq = record.last_event_seq;
        let built = deps.build_with_id(record.session_id.clone(), record.created_at);
        let mut session = match (built, &recorded) {
            // An exact route the client cannot reach is its own failure, and
            // never a reason to run the conversation somewhere else.
            (Err(CodingAgentBuildError::ModelSelection { source, .. }), Some(_)) => {
                return Err(CodingAgentBuildError::RecordedRouteUnavailable {
                    provider: record.provider.clone().unwrap_or_default(),
                    model: record.model.clone().unwrap_or_default(),
                    source,
                });
            }
            (built, _) => built?,
        };
        if let Some(recorded) = recorded {
            let resolved = format!("{}/{}", session.provider, session.model);
            if resolved != recorded {
                return Err(CodingAgentBuildError::RecordedRouteMismatch { recorded, resolved });
            }
        }

        session.conversation().history = History::from_stored_messages(&record.messages);
        // The parentage the record carries is restored, so storing a resumed
        // child again says the same thing. The tree itself is not: a resumed
        // child has no supervisor above it, and rebuilding one is the
        // application's to do.
        session.parent_session_id = record.parent_session_id;
        Ok(session)
    }

    /// Rebuilds a live session's successor from its warm state, on the same
    /// route, without initializing again.
    ///
    /// Everything [`initialize`](Self::initialize) would compute is taken from
    /// the state instead: the system prompt, the discovered skills and the tool
    /// that loads one, the token counts the context snapshot reads, and the
    /// files the session touched. Call
    /// [`start_from_warm_state`](Self::start_from_warm_state) afterwards in
    /// place of `initialize`.
    ///
    /// # Errors
    ///
    /// As [`from_record`](Self::from_record) on the recorded model.
    pub(crate) fn from_warm_state(
        state: WarmState,
        deps: CodingRuntimeBuilder,
    ) -> StdResult<Self, CodingAgentBuildError> {
        let mut session = Self::from_record(state.record, &ResumeMode::RecordedModel, deps)?;
        if !state.skills.is_empty() {
            let vocabulary = session.registry.vocabulary();
            session
                .registry
                .register(make_use_skill_tool_for_vocabulary(
                    Arc::from(state.skills.clone()),
                    vocabulary,
                ));
        }
        session.skills = state.skills;
        session.memory_summaries = state.memory_summaries;
        session.system_prompt = state.system_prompt;
        session.memory_tokens = state.memory_tokens;
        session.skills_tokens = state.skills_tokens;
        {
            let mut conversation = session.conversation();
            conversation.file_tracker = state.file_tracker;
            conversation.activated_skill_context_observed = state.activated_skill_context_observed;
            conversation.context_window = state.context_window;
        }
        Ok(session)
    }

    /// Opens the event stream of a session built from warm state.
    ///
    /// Publishes [`SessionStarted`](CodingEvent::SessionStarted) and nothing
    /// else: no memory was loaded and no skills were discovered, because the
    /// warm state already carried what they produce.
    pub(crate) async fn start_from_warm_state(&mut self) -> Result<()> {
        self.emit(CodingEvent::SessionStarted {
            provider: Some(self.provider.clone()),
            model:    Some(self.model.clone()),
        });
        self.flush_events().await.map(|_| ())
    }

    /// Everything a successor in this process needs to carry on without
    /// initializing again: the durable record plus the state initialization
    /// derived from it.
    pub(crate) fn warm_state(&self) -> WarmState {
        let record = self.to_record();
        let conversation = self.conversation();
        WarmState {
            record,
            system_prompt: self.system_prompt.clone(),
            skills: self.skills.clone(),
            memory_summaries: self.memory_summaries.clone(),
            memory_tokens: self.memory_tokens,
            skills_tokens: self.skills_tokens,
            file_tracker: conversation.file_tracker.clone(),
            activated_skill_context_observed: conversation.activated_skill_context_observed,
            context_window: conversation.context_window.clone(),
        }
    }

    /// The session as it should be stored.
    ///
    /// Everything a resumed session needs and nothing an application could not
    /// supply again. A child records which session spawned it, so a stored tree
    /// can be read back in shape; the tree itself is the application's to
    /// rebuild, because a child's supervisor is not stored.
    ///
    /// The record stores the last sequence committed by the event pipeline.
    /// Public callers take records between prompts, after the prompt's event
    /// barrier has committed its complete history.
    pub(crate) fn to_record(&self) -> SessionRecord {
        let mut record = SessionRecord::new(self.id.clone());
        record.parent_session_id.clone_from(&self.parent_session_id);
        record.provider = Some(self.provider.clone());
        record.model = Some(self.model.clone());
        record.created_at = self.created_at;
        record.last_event_seq = self.emitter.committed_seq();
        record.messages = self.conversation().history.to_stored_messages();
        record
    }

    /// Loads what the session was told, and captures where it is working.
    ///
    /// Call this once, before the first prompt. It publishes
    /// [`SessionStarted`](CodingEvent::SessionStarted), loads the memory files
    /// and skill directories the options name, probes the environment for the
    /// prompt's sake, and asks the profile for the system prompt the whole
    /// session will use. Naming no memory files and no skill directories is
    /// normal: pebble looks in no conventional location and guesses no
    /// filename.
    ///
    /// Discovering any skill also registers the tool that loads one, in the
    /// profile's own vocabulary. That is the only tool a session adds to
    /// itself: what it loads is not known until the directories have been
    /// read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Interrupted`] when the session is cancelled while it is
    /// initializing, which is checked around every read and every probe.
    #[tracing::instrument(
        name = "coding_session_initialize",
        skip_all,
        fields(session_id = %self.id, provider = %self.provider, model = %self.model)
    )]
    pub(crate) async fn initialize(&mut self) -> Result<()> {
        let cancel = self.cancel_token.clone();

        self.emit(CodingEvent::SessionStarted {
            provider: Some(self.provider.clone()),
            model:    Some(self.model.clone()),
        });
        if cancel.is_cancelled() {
            return Err(Error::Interrupted(InterruptReason::Cancelled));
        }

        let profile = self.profile.profile_kind().as_str().to_owned();

        // Independent reads of the environment, overlapped; the events they
        // feed stay in their documented order below.
        let (memory, skills) = tokio::join!(
            load_memory(self.env.as_ref(), &self.config.memory_files, &cancel),
            discover_skills(self.env.as_ref(), &self.config.skill_dirs, &cancel),
        );

        let memory = memory?;
        // The files are described, never quoted: the durable stream must not
        // carry the bytes of a project's own instructions. The same
        // descriptions are what a prompt transform is shown.
        let memory_summaries: Vec<_> = memory.iter().map(MemoryDocument::to_summary).collect();
        self.memory_summaries.clone_from(&memory_summaries);
        self.emit(CodingEvent::MemoryLoaded {
            profile:            profile.clone(),
            files:              memory_summaries.clone(),
            total_loaded_bytes: memory.iter().map(|document| document.loaded_bytes).sum(),
            budget_bytes:       MEMORY_BUDGET_BYTES,
        });

        self.skills = skills?;
        self.emit(CodingEvent::SkillsDiscovered {
            profile,
            source_dirs: self.config.skill_dirs.clone(),
            skills: self.skills.iter().map(Skill::to_summary).collect(),
        });
        // The one tool that cannot be built by the builder: what it loads is
        // discovered here, and a session that discovered no skills advertises
        // no way to load one.
        if !self.skills.is_empty() {
            let vocabulary = self.registry.vocabulary();
            self.registry.register(make_use_skill_tool_for_vocabulary(
                Arc::from(self.skills.clone()),
                vocabulary,
            ));
        }

        // Measured once: memory and skills never change again, and the context
        // snapshot built every round reads these numbers instead of
        // re-tokenizing the same text.
        self.memory_tokens = memory_prompt_tokens(&memory);
        self.skills_tokens = skills_prompt_tokens(&self.skills, self.registry.vocabulary());

        let env_context = self.build_env_context(&cancel).await?;
        debug!(
            is_git_repo = env_context.is_git_repo,
            model = env_context.model.as_str(),
            "Environment context built"
        );

        // Built once and fixed for the session's life. Only the loaded text
        // reaches the profile; the file metadata is already on the stream.
        let memory: Vec<String> = memory
            .into_iter()
            .map(|document| document.content)
            .collect();
        // The registry is complete by now — the builder froze it and the skill
        // tool above is the last addition — so a profile that gates a prompt
        // section on a tool reads the session's real answer.
        let default_prompt = self.profile.build_system_prompt(
            &self.registry,
            &env_context,
            &memory,
            self.config.user_instructions.as_deref(),
            &self.skills,
        );
        // The application's one chance to adjust the words the model reads
        // first. It sees the prompt as written and the session as the prompt
        // describes it. Tool summaries are the registered starting set;
        // per-turn middleware can narrow what the model sees later.
        self.system_prompt = match &self.prompt_transform {
            Some(transform) => {
                let tools: Vec<_> = self
                    .registered_tools()
                    .iter()
                    .map(ToolDefinitionWithSource::to_tool_summary)
                    .collect();
                let skills: Vec<_> = self.skills.iter().map(Skill::to_summary).collect();
                let context = SystemPromptContext::new(
                    &default_prompt,
                    &env_context,
                    &tools,
                    &memory_summaries,
                    &skills,
                );
                transform.transform(context).apply(default_prompt)
            }
            None => default_prompt,
        };

        self.flush_events().await.map(|_| ())
    }

    /// Gathers what the system prompt says about where the session is working.
    ///
    /// The three git answers come from running git in the session's own
    /// environment, so a session working in a container describes that
    /// container's checkout rather than the machine pebble runs on. A probe
    /// that fails is simply left out: an environment without git is a working
    /// environment.
    async fn build_env_context(&self, cancel: &CancellationToken) -> Result<EnvContext> {
        stop_if_cancelled(cancel)?;
        let (current_date, git_branch) = tokio::join!(
            self.probe_date(cancel),
            self.probe(cancel, "git rev-parse --abbrev-ref HEAD"),
        );
        let is_git_repo = git_branch.is_some();

        stop_if_cancelled(cancel)?;
        let (git_status_short, git_recent_commits) = if is_git_repo {
            tokio::join!(
                self.probe(cancel, "git status --short"),
                self.probe(cancel, "git log --oneline -10"),
            )
        } else {
            (None, None)
        };

        stop_if_cancelled(cancel)?;
        Ok(EnvContext {
            is_git_repo,
            git_branch,
            git_status_short,
            git_recent_commits,
            current_date,
            model: self.model.clone(),
            knowledge_cutoff: self.knowledge_cutoff.clone(),
            ..EnvContext::from_environment(self.env.as_ref())
        })
    }

    /// Runs one short command in the environment, answering with its trimmed
    /// output when it succeeded and produced any.
    async fn probe(&self, cancel: &CancellationToken, command: &str) -> Option<String> {
        let outcome = self
            .env
            .exec(ExecRequest {
                timeout_ms: Some(PROBE_TIMEOUT_MS),
                cancel_token: Some(cancel.child_token()),
                ..ExecRequest::new(command)
            })
            .await
            .ok()?;
        if !outcome.result.is_success() {
            return None;
        }
        let output = outcome.result.stdout.trim();
        (!output.is_empty()).then(|| output.to_owned())
    }

    /// Today's date where the session is working.
    ///
    /// Asked of the environment, because a session working somewhere else
    /// should date its prompt from there. An environment that cannot answer —
    /// no shell, no `date` — falls back to the UTC date on this machine, which
    /// is never wrong by more than a day.
    async fn probe_date(&self, cancel: &CancellationToken) -> String {
        let reported = self.probe(cancel, "date +%Y-%m-%d").await;
        match reported {
            Some(date) if is_iso_date(&date) => date,
            _ => rfc3339_millis::format(SystemTime::now())
                .get(..10)
                .unwrap_or_default()
                .to_owned(),
        }
    }

    /// This session's identifier.
    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    /// The root of this session's tree, which a root session answers with its
    /// own [`id`](Self::id).
    ///
    /// Read-only for the same reason [`CodingRuntimeBuilder`]'s `child_of` is
    /// crate-internal: a session's place in its tree is settled when it is
    /// built, by whoever spawned it. Root-scoped tools, the shared event
    /// stream, and stored records all key on this, so a session that could
    /// be re-rooted afterwards could be detached from the tree that owns
    /// it.
    pub(crate) fn root_session_id(&self) -> &str {
        &self.root_session_id
    }

    /// Which harness this session runs.
    pub(crate) fn profile_kind(&self) -> AgentProfileKind {
        self.profile.profile_kind()
    }

    /// Descriptions of the memory files loaded into the system prompt.
    pub(crate) fn memory_summaries(&self) -> &[MemoryFileSummary] {
        &self.memory_summaries
    }

    /// Descriptions of the skills available to this session.
    pub(crate) fn skill_summaries(&self) -> Vec<SkillSummary> {
        self.skills.iter().map(Skill::to_summary).collect()
    }

    /// Descriptions of registered tools in stable name order.
    pub(crate) fn tool_summaries(&self) -> Vec<ToolSummary> {
        let mut tools: Vec<_> = self
            .registered_tools()
            .iter()
            .map(ToolDefinitionWithSource::to_tool_summary)
            .collect();
        tools.sort_by(|left, right| left.name.cmp(&right.name));
        tools
    }

    /// The provider the session resolved to.
    pub(crate) fn provider(&self) -> &str {
        &self.provider
    }

    /// The catalog identifier of the model the session resolved to.
    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    /// What the model is told about its context window and output budget.
    #[cfg(test)]
    pub(crate) fn model_facts(&self) -> ModelFacts {
        self.facts
    }

    /// Where this session's tools act, for the crate's own tests.
    #[cfg(test)]
    pub(crate) const fn environment(&self) -> &Arc<dyn Environment> {
        &self.env
    }

    /// How this session was configured, for the crate's own tests.
    #[cfg(test)]
    pub(crate) const fn config(&self) -> &CodingAgentOptions {
        &self.config
    }

    /// The permission level the application recorded for this session.
    #[cfg(test)]
    pub(crate) fn permission_level(&self) -> Option<PermissionLevel> {
        self.config.permission_level
    }

    /// What the session is doing right now.
    pub(crate) fn state(&self) -> CodingAgentState {
        self.state.current()
    }

    /// The state machine itself, for a test that reads it while a prompt has
    /// the session borrowed.
    pub(crate) fn state_machine(&self) -> StateMachine {
        self.state.clone()
    }

    /// Shared cancellation for the session's current compaction.
    pub(crate) fn compaction_control(&self) -> CompactionControl {
        self.compaction.clone()
    }

    /// A snapshot of the conversation so far.
    pub(crate) fn history(&self) -> History {
        self.conversation().history.clone()
    }

    /// The latest context-window measurement, when a model has answered.
    pub(crate) fn context_window(&self) -> Option<ContextWindowSnapshot> {
        self.conversation().context_window.clone()
    }

    /// The most recent assistant message, without cloning the full history.
    pub(crate) fn final_assistant_message(&self) -> Option<Message> {
        self.conversation()
            .history
            .turns()
            .iter()
            .rev()
            .find(|message| matches!(message, Message::Assistant { .. }))
            .cloned()
    }

    /// A snapshot of the files this session has read and changed.
    #[cfg(test)]
    pub(crate) fn file_tracker(&self) -> FileTracker {
        self.conversation().file_tracker.clone()
    }

    /// Where the last prompt spent its time.
    pub(crate) fn last_prompt_timing(&self) -> PromptTiming {
        self.conversation().totals.timing
    }

    /// What the last prompt cost in tokens, summed over every response.
    pub(crate) fn last_prompt_usage(&self) -> TokenUsage {
        self.conversation().totals.usage
    }

    /// What the last prompt cost in USD micros, where the catalog or the
    /// provider priced it.
    pub(crate) fn last_prompt_cost_usd_micros(&self) -> Option<u64> {
        self.conversation().totals.cost_usd_micros
    }

    /// The shared conversation, locked.
    fn conversation(&self) -> MutexGuard<'_, ConversationState> {
        self.conversation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The tools registered for this session before middleware filters them.
    pub(crate) fn registered_tools(&self) -> Vec<ToolDefinitionWithSource> {
        self.registry.definitions_with_source()
    }

    /// Watches the session's events from here on.
    ///
    /// The stream is lossy for a subscriber that falls behind; an application
    /// that must see everything configures
    /// [`CodingRuntimeBuilder::event_sink`] instead.
    ///
    /// The stream ends with the session, not with the value: once
    /// [`CodingRuntime::shutdown`] has returned, the receiver reads out
    /// whatever it still holds and then observes `RecvError::Closed`, with
    /// the session still alive. So a reader that loops until the stream
    /// closes can be joined before the session is dropped, which is the
    /// order an application wants — the renderer's summary is the last
    /// thing printed.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<CodingAgentEvent> {
        self.emitter.subscribe()
    }

    /// A handle that steers and interrupts this session from elsewhere.
    pub(crate) fn control_handle(&self) -> AgentControlHandle {
        self.agent_control.clone()
    }

    /// Queues guidance for the next round.
    #[cfg(test)]
    pub(crate) fn steer(&self, text: impl Into<String>) {
        let _ = self.agent_control.enqueue_steering(text.into());
    }

    /// Hands out a steering lease that parks natural completion while an
    /// external steering source is attached.
    #[cfg(test)]
    pub(crate) fn steering_lease(&self) -> SteeringLease {
        SteeringLease::acquire(&self.agent_control)
    }

    /// Queues more input to process once the current input is finished.
    #[cfg(test)]
    pub(crate) fn follow_up(&self, message: impl Into<String>) {
        let _ = self.agent_control.follow_up(message.into());
    }

    /// Ends the prompt.
    ///
    /// The loop unwinds through its own checkpoints — every tool call still
    /// gets its result recorded — and then closes the session. This is the
    /// terminal gesture; [`AgentControlHandle::interrupt`] is the one that
    /// only abandons a round.
    #[cfg(test)]
    pub(crate) fn interrupt(&self) {
        self.set_interrupt_reason(InterruptReason::Cancelled);
        self.cancel_token.cancel();
    }

    /// The token that ends this session's prompt.
    pub(crate) fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Where this session's supervisor publishes child-lifecycle facts.
    ///
    /// A child's own events already use the tree's shared pipeline. Lifecycle
    /// facts are the parent's news, so this callback publishes them under the
    /// parent's session identity on that same pipeline.
    pub(crate) fn sub_agent_event_callback(&self) -> SubagentEventCallback {
        let emitter = self.emitter.clone();
        let parent_session_id = self.id.clone();
        Arc::new(move |event| {
            emitter.emit(parent_session_id.clone(), event);
        })
    }

    /// This session's children, for the crate's own tests.
    #[cfg(test)]
    pub(crate) const fn subagent_supervisor(&self) -> Option<&SubagentSupervisor> {
        self.subagents.as_ref()
    }

    /// Whether this session can ask a person a question, for the crate's own
    /// tests: a child never can.
    #[cfg(test)]
    pub(crate) const fn has_human_input(&self) -> bool {
        self.human_input.is_some()
    }

    /// The reason slot an outside task can fill before cancelling.
    ///
    /// First writer wins, so a watchdog that names its own reason before
    /// cancelling gets that reason reported instead of a plain cancellation.
    #[must_use]
    pub(crate) fn interrupt_reason_handle(&self) -> InterruptReasonHandle {
        InterruptReasonHandle(Arc::clone(&self.interrupt_reason))
    }

    /// Changes how hard the model is asked to think, from the next round on.
    pub(crate) fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.config.reasoning_effort = effort;
        if let Some(agent) = &mut self.coding_agent {
            agent.set_reasoning_effort(effort);
        }
    }

    /// Changes which latency or cost tier the session asks for, from the next
    /// round on.
    pub(crate) fn set_speed(&mut self, speed: Option<Speed>) {
        self.config.speed = speed;
        if let Some(agent) = &mut self.coding_agent {
            agent.set_speed(speed);
        }
    }

    /// Changes where a tool call's extra environment variables come from.
    #[cfg(test)]
    pub(crate) fn set_tool_env_provider(&mut self, provider: Arc<dyn ToolEnvProvider>) {
        self.tool_env_provider = Some(Arc::clone(&provider));
        if let Some(bridge) = &self.coding_bridge {
            bridge.set_tool_env_provider(provider);
        }
    }

    /// Processes one input to completion, answering with the assistant's final
    /// text when it ended with any.
    ///
    /// Questions go to the provider the session was built with.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SessionClosed`] for a session that has ended,
    /// [`Error::Interrupted`] when the prompt was cancelled or ran out of
    /// wall-clock time, [`Error::Llm`] when the model call failed for good, and
    /// [`Error::EventSink`] when the configured sink refused an event.
    pub(crate) async fn prompt(&mut self, input: impl Into<CodingInput>) -> Result<Option<String>> {
        self.prompt_with_cancellation(input, None).await
    }

    /// Compacts older history while the session is idle.
    pub(crate) async fn compact(
        &mut self,
        options: CompactionOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<CompactionOutcome> {
        match self.state.current() {
            CodingAgentState::Closed => return Err(Error::SessionClosed),
            CodingAgentState::Idle => {}
            state => {
                return Err(Error::InvalidState(format!(
                    "cannot compact while the session is {state:?}"
                )));
            }
        }

        let (mut history, file_tracker) = {
            let state = self.conversation();
            (state.history.clone(), state.file_tracker.clone())
        };
        let preserve_turns = options
            .preserve_turns_value()
            .unwrap_or(self.config.compaction_preserve_turns)
            .max(1);
        if history.compact_preserve_start(preserve_turns) == 0 {
            return Ok(CompactionOutcome::Unchanged);
        }

        let estimate = estimate_active_context_usage(&self.system_prompt, &history);
        let compaction_cancel = self.cancel_token.child_token();
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            compaction_cancel.cancel();
        }
        let caller_link = cancel.map(|caller| link_cancellation(caller, &compaction_cancel));
        let operation = self.compaction.begin(&compaction_cancel);
        self.state.transition(CodingAgentState::Compacting);
        let request = CompactionRequest {
            model: &self.model_selector,
            facts: self.facts,
            preserve_turns,
            estimate,
            reason: CompactionReason::Manual,
            instructions: options.instructions_ref(),
            cancel: operation.token(),
        };
        let result = compact_context(
            &mut history,
            &self.client,
            &file_tracker,
            request,
            &self.emitter,
            &self.id,
        )
        .await;
        drop(operation);
        if let Some(link) = caller_link {
            link.abort();
        }

        if result.is_ok() {
            self.conversation().history = history;
        }
        if self.state.current() == CodingAgentState::Closed {
            self.shutdown(ShutdownReason::Cancelled).await?;
        } else {
            self.state.transition(CodingAgentState::Idle);
            if let Err(error) = self.flush_events().await {
                let _ = self.shutdown(ShutdownReason::Error).await;
                return Err(error);
            }
        }
        result
    }

    /// Processes one input until it completes or `cancel` fires.
    ///
    /// Cancelling `cancel` ends this prompt alone: the loop unwinds through its
    /// checkpoints so every tool call still gets its result, the prompt reports
    /// [`Error::Interrupted`], and the session returns to
    /// [`Idle`](CodingAgentState::Idle) ready for its next prompt. A call that
    /// is running is cancelled through its token and keeps the result it
    /// returns; a call the model asked for that has not started yet — the
    /// cancellation landed while the assistant turn was being committed or
    /// compacted — is answered `Cancelled` without running, so history stays
    /// paired. Only close or shutdown closes the session.
    ///
    /// # Errors
    ///
    /// As [`prompt`](Self::prompt).
    #[tracing::instrument(
        name = "coding_session_prompt",
        skip_all,
        fields(
            session_id = %self.id,
            provider = %self.provider,
            model = %self.model
        )
    )]
    pub(crate) async fn prompt_with_cancellation(
        &mut self,
        input: impl Into<CodingInput>,
        cancel: Option<&CancellationToken>,
    ) -> Result<Option<String>> {
        let input = input.into();
        self.conversation().totals = PromptTotals::default();
        if self.state.current() == CodingAgentState::Closed {
            return Err(Error::SessionClosed);
        }

        // A child of the terminal token, so a shutdown ends the prompt too.
        // Caller cancellation and a failed durable stream are joined in by
        // tasks because a cancellation token has only one parent.
        let prompt_cancel = self.cancel_token.child_token();
        let caller_link = cancel.map(|caller| link_cancellation(caller, &prompt_cancel));
        let event_failure = self.emitter.failure_token();
        let event_link = link_cancellation(&event_failure, &prompt_cancel);

        let timer = self.start_wall_clock_timer(&prompt_cancel);
        let result = self
            .process_input(input, SkillExpansion::Apply, &prompt_cancel)
            .await;

        if let Some(link) = caller_link {
            link.abort();
        }
        event_link.abort();
        let mut task_failure = stop_wall_clock_timer(timer).await;
        // The reason has been reported by now. Clearing it here rather than at
        // the start of the next prompt keeps a reason a watchdog records just
        // before it cancels, and still stops one prompt's reason reaching the
        // next.
        if self.state.current() != CodingAgentState::Closed {
            *self
                .interrupt_reason
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = None;
        }

        if self.state.current() == CodingAgentState::Closed {
            let reason = if self.cancel_token.is_cancelled() {
                ShutdownReason::Cancelled
            } else {
                ShutdownReason::Error
            };
            if let Err(error) = self.shutdown(reason).await {
                remember_failure(&mut task_failure, error);
            }
        } else {
            self.state.transition(CodingAgentState::Idle);
            // `ProcessingEnd` is the prompt's durability barrier. The prompt
            // cannot finish before it and every earlier event reach the sink.
            if let Err(error) = self.flush_events().await {
                remember_failure(&mut task_failure, error);
            }
        }

        // A failure found by the final barrier moved the session to `Closed`
        // after the branch above began. Finish the shutdown now so children and
        // the pump do not outlive the prompt that reports it.
        if self.state.current() == CodingAgentState::Closed
            && !self.ended
            && let Err(error) = self.shutdown(ShutdownReason::Error).await
        {
            remember_failure(&mut task_failure, error);
        }

        // The durable stream is the record of every other outcome. If it is
        // incomplete, that is the failure the caller must act on even when the
        // model or a cleanup task also failed.
        if task_failure
            .as_ref()
            .is_some_and(|error| error.kind() == ErrorKind::EventStream)
        {
            return Err(task_failure.expect("the failure was present"));
        }

        match (result, task_failure) {
            // The prompt's own failure is the story; a task that also failed on
            // the way out is reported rather than returned.
            (Err(error), Some(task)) => {
                warn!(error = ?task, "A session task failed while the prompt was already failing");
                Err(error)
            }
            (Err(error), None) => Err(error),
            (Ok(_), Some(task)) => Err(task),
            (Ok(output), None) => Ok(output),
        }
    }

    /// Starts the task that ends a prompt which has taken too long.
    ///
    /// The timer cancels the prompt rather than dropping it, so the loop
    /// unwinds through its own checkpoints and every tool call still has its
    /// result recorded: a running call is cancelled and keeps its own result,
    /// and a call not yet started is answered `Cancelled` without running. The
    /// session stays open: running out of time is the prompt's failure, and
    /// the next prompt gets a fresh budget.
    fn start_wall_clock_timer(&self, prompt_cancel: &CancellationToken) -> Option<WallClockTimer> {
        let duration = self.config.wall_clock_timeout?;
        let stop = CancellationToken::new();
        let cancel = prompt_cancel.clone();
        let reason = self.interrupt_reason_handle();
        let watched = stop.clone();
        let task = tokio::spawn(async move {
            tokio::select! {
                () = watched.cancelled() => {}
                () = sleep(duration) => {
                    reason.record(InterruptReason::WallClockTimeout);
                    cancel.cancel();
                }
            }
        });
        Some(WallClockTimer { stop, task })
    }

    /// Closes the session, joining everything it owns.
    ///
    /// Only the first call does anything, which is what lets the loop close a
    /// cancelled session at whichever checkpoint reaches it first. Children are
    /// closed before this session publishes its own end, so a reader sees a
    /// tree unwind from the leaves.
    ///
    /// Every stream [`CodingRuntime::subscribe`] handed out ends by the time
    /// this returns: joining the pump is what closes them, so a reader
    /// looping until `RecvError::Closed` finishes without the session
    /// having to be dropped first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EventSink`] when the configured sink had refused an
    /// event, and [`Error::Task`] when a task the session owned failed
    /// outright. The session is closed either way.
    #[tracing::instrument(
        name = "coding_session_shutdown",
        skip_all,
        fields(session_id = %self.id, reason = ?reason)
    )]
    pub(crate) async fn shutdown(&mut self, reason: ShutdownReason) -> Result<bool> {
        if self.ended {
            return Ok(false);
        }
        if reason == ShutdownReason::Cancelled {
            self.set_interrupt_reason(InterruptReason::Cancelled);
            self.cancel_token.cancel();
        }
        self.state.transition(CodingAgentState::Closed);
        if let Some(supervisor) = &self.subagents {
            supervisor.shutdown_all().await;
        }
        if let Some(agent) = &mut self.coding_agent {
            let _ = agent.shutdown();
        }
        // A session shut down before its first prompt never built its agent,
        // so the control it handed out is closed here for it to read.
        let _ = self.agent_control.close();
        self.ended = true;
        self.emit(CodingEvent::SessionEnded);
        let flushed = self.flush_events().await;
        let joined = self.join_pump().await;
        flushed?;
        joined?;
        Ok(true)
    }

    /// Waits until every event currently queued has reached the durable sink.
    pub(crate) async fn flush_events(&mut self) -> Result<u64> {
        match self.emitter.flush().await {
            Ok(seq) => Ok(seq),
            Err(failure) => {
                self.state.transition(CodingAgentState::Closed);
                Err(failure.into_runtime_error())
            }
        }
    }

    /// The highest event sequence committed by the event pipeline.
    ///
    /// With a durable sink, commitment means the sink accepted the event.
    pub(crate) fn committed_event_seq(&self) -> u64 {
        self.emitter.committed_seq()
    }

    /// Publishes everything queued, then joins the pump.
    async fn join_pump(&mut self) -> Result<()> {
        let Some(pump) = self.pump.take() else {
            return Ok(());
        };
        let _ = self.emitter.close().await;
        match pump.await {
            Ok(result) => result,
            Err(source) => Err(Error::Task {
                task: TaskKind::EventPump,
                source,
            }),
        }
    }

    /// Reports a pump that has already stopped, which only a refusing sink
    /// does while a prompt is in progress.
    ///
    /// Checked at round boundaries so a prompt ends promptly once its events
    /// stop being recorded, rather than working on against a stream nobody
    /// has.
    ///
    /// A sink failure closes the session as well as ending the prompt. The
    /// pipeline stops for good when the pump does — nothing is recorded, and no
    /// subscriber is served — so a session that kept answering would be working
    /// where nobody could see it. The application gets the failure from the
    /// prompt that found it, and [`Error::SessionClosed`] from every call
    /// after.
    async fn check_pump(&mut self) -> Result<()> {
        if let Some(failure) = self.emitter.failure() {
            self.state.transition(CodingAgentState::Closed);
            return Err(failure.into_runtime_error());
        }
        let Some(pump) = self.pump.as_ref() else {
            return Ok(());
        };
        if !pump.is_finished() {
            return Ok(());
        }
        let outcome = self.join_pump().await;
        if outcome.is_err() {
            self.state.transition(CodingAgentState::Closed);
        }
        outcome
    }

    /// Answers for a prompt the agent loop aborted.
    ///
    /// A terminal cancellation closes the session; anything else — the
    /// caller's prompt token, or the wall clock — ends only this prompt, and
    /// the session is left open for the next one.
    pub(super) async fn prompt_aborted(&mut self) -> Error {
        if self.cancel_token.is_cancelled() {
            self.close_cancelled().await
        } else {
            self.interrupted_error()
        }
    }

    /// Closes a cancelled session and answers with the error the prompt ends
    /// on.
    async fn close_cancelled(&mut self) -> Error {
        let interrupted = self.interrupted_error();
        match self.shutdown(ShutdownReason::Cancelled).await {
            Ok(_) => interrupted,
            Err(error) => error,
        }
    }

    fn set_interrupt_reason(&self, reason: InterruptReason) {
        self.interrupt_reason_handle().record(reason);
    }

    /// The error an interrupted prompt ends with, naming whatever reason was
    /// recorded first.
    fn interrupted_error(&self) -> Error {
        let reason = *self
            .interrupt_reason
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .unwrap_or(&InterruptReason::Cancelled);
        Error::Interrupted(reason)
    }

    /// Publishes one event on this session's stream.
    fn emit(&self, event: CodingEvent) {
        self.emitter.emit(self.id.clone(), event);
    }

    /// Publishes a model failure, closing the session when the credential is
    /// the problem.
    ///
    /// An authentication failure will not fix itself between rounds, so the
    /// session stops rather than spending the rest of the prompt failing the
    /// same way.
    fn emit_llm_error(&mut self, error: LlmError) -> Error {
        let credential_failure = is_auth_error(&error);
        // Projected from the error the caller receives, so what a reader sees
        // and what the prompt returns say the same thing.
        let error = Error::Llm(error);
        self.emit(CodingEvent::Error {
            error: ErrorData::from(&error),
        });
        if credential_failure {
            self.state.transition(CodingAgentState::Closed);
        }
        error
    }
}

/// The session's state, and the one place it moves.
///
/// Shared between the runtime, which moves it at the edges of a prompt, and
/// the bridge, which moves it into `Executing` around each tool round while
/// the runtime is borrowed by the prompt in progress. Cheap to clone; every
/// clone reads and moves the same state.
#[derive(Clone, Debug)]
pub(crate) struct StateMachine {
    state:      Arc<Mutex<CodingAgentState>>,
    emitter:    Emitter,
    session_id: String,
}

impl StateMachine {
    /// A session that starts idle.
    fn new(emitter: Emitter, session_id: String) -> Self {
        Self {
            state: Arc::new(Mutex::new(CodingAgentState::Idle)),
            emitter,
            session_id,
        }
    }

    /// What the session is doing right now.
    pub(crate) fn current(&self) -> CodingAgentState {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Moves the session's state machine, publishing the end of a processing
    /// cycle where one ends.
    ///
    /// Valid moves: Idle or Executing to Thinking, Thinking to Executing or
    /// Idle, Idle to Compacting, Compacting to Idle, and anything to Closed.
    /// Ending the session belongs to
    /// [`CodingRuntime::shutdown`], never here.
    pub(super) fn transition(&self, to: CodingAgentState) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let from = *state;
        if from == to {
            return;
        }

        debug_assert!(
            matches!(
                (from, to),
                (
                    CodingAgentState::Idle | CodingAgentState::Executing,
                    CodingAgentState::Thinking
                ) | (
                    CodingAgentState::Thinking,
                    CodingAgentState::Executing | CodingAgentState::Idle
                ) | (CodingAgentState::Idle, CodingAgentState::Compacting)
                    | (CodingAgentState::Compacting, CodingAgentState::Idle)
                    | (_, CodingAgentState::Closed)
            ),
            "invalid session state transition: {from:?} -> {to:?}"
        );

        *state = to;
        drop(state);
        if matches!(
            from,
            CodingAgentState::Thinking | CodingAgentState::Executing | CodingAgentState::Compacting
        ) && to == CodingAgentState::Idle
        {
            self.emitter
                .emit(self.session_id.clone(), CodingEvent::ProcessingEnd);
        }
    }
}

/// Cancels `prompt_cancel` when `caller` fires.
///
/// A token has one parent, and the prompt token's is the terminal token, so
/// the caller's is joined in by a task instead. The task ends on its own once
/// the prompt token fires for any reason, and the prompt aborts it when it
/// finishes.
fn link_cancellation(
    caller: &CancellationToken,
    prompt_cancel: &CancellationToken,
) -> JoinHandle<()> {
    let caller = caller.clone();
    let prompt_cancel = prompt_cancel.clone();
    tokio::spawn(async move {
        tokio::select! {
            () = caller.cancelled() => prompt_cancel.cancel(),
            () = prompt_cancel.cancelled() => {}
        }
    })
}

/// Keeps the event-stream failure when cleanup finds more than one failure.
fn remember_failure(stored: &mut Option<Error>, failure: Error) {
    match stored.as_ref().map(Error::kind) {
        None => *stored = Some(failure),
        Some(ErrorKind::EventStream) if failure.kind() == ErrorKind::EventStream => {}
        Some(_) if failure.kind() == ErrorKind::EventStream => *stored = Some(failure),
        Some(_) => {
            warn!(error = ?failure, "A session task also failed while the prompt was ending");
        }
    }
}

/// The task watching one prompt's wall-clock budget.
struct WallClockTimer {
    stop: CancellationToken,
    task: JoinHandle<()>,
}

/// Stops the timer and joins it, reporting a task that failed outright.
async fn stop_wall_clock_timer(timer: Option<WallClockTimer>) -> Option<Error> {
    let timer = timer?;
    timer.stop.cancel();
    timer.task.await.err().map(|source| Error::Task {
        task: TaskKind::WallClockTimer,
        source,
    })
}

/// Stops a step that a cancelled session should not take.
fn stop_if_cancelled(cancel: &CancellationToken) -> Result<()> {
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }
    Ok(())
}

/// Whether a model failure means the credential, rather than the call.
fn is_auth_error(error: &LlmError) -> bool {
    matches!(
        error.kind(),
        LlmErrorKind::Authentication | LlmErrorKind::AccessDenied
    )
}

/// Whether a probe's answer looks like the `YYYY-MM-DD` it was asked for.
fn is_iso_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes.iter().enumerate().all(|(index, byte)| match index {
            4 | 7 => *byte == b'-',
            _ => byte.is_ascii_digit(),
        })
}

#[cfg(test)]
mod tests {
    use std::result::Result as StdResult;
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::Notify;
    use tokio::sync::broadcast::error::RecvError;
    use tokio::task::yield_now;
    use tokio::time::timeout;

    use super::testing::{TestProfile, TestSession, builder, event_names, settled};
    use super::*;
    use crate::error::ErrorKind;
    use crate::event::EventSinkError;
    use crate::record::SESSION_RECORD_FORMAT_VERSION;
    use crate::test_support::{
        MockEnvironment, ScriptedCall, ScriptedFailure, scripted_client, text_delta_events,
        text_response,
    };
    use crate::types::Message;

    /// A client that answers every round with the same text.
    fn client() -> Client {
        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        client
    }

    /// A session that answers every round with the same text.
    fn session() -> CodingRuntime {
        let (session, _provider) =
            TestSession::answering(vec![ScriptedCall::response(text_response("done"))]);
        session
    }

    // --- Building ---

    #[tokio::test]
    async fn a_session_needs_a_model() {
        let error = CodingRuntime::builder(client())
            .environment(Arc::new(MockEnvironment::linux()))
            .build()
            .expect_err("no model was named");

        assert!(matches!(error, CodingAgentBuildError::MissingModel));
    }

    #[tokio::test]
    async fn a_session_needs_an_environment() {
        let error = CodingRuntime::builder(client())
            .model("test/model")
            .build()
            .expect_err("no environment was given");

        assert!(matches!(error, CodingAgentBuildError::MissingEnvironment));
    }

    #[tokio::test]
    async fn a_selector_that_names_nothing_is_refused() {
        let error = CodingRuntime::builder(client())
            .model("no-such-model")
            .environment(Arc::new(MockEnvironment::linux()))
            .build()
            .expect_err("the selector names nothing");

        assert!(matches!(
            error,
            CodingAgentBuildError::ModelSelection { ref selector, .. } if selector == "no-such-model"
        ));
    }

    #[tokio::test]
    async fn a_blank_selector_is_refused() {
        let error = CodingRuntime::builder(client())
            .model("   ")
            .environment(Arc::new(MockEnvironment::linux()))
            .build()
            .expect_err("a blank selector names nothing");

        assert!(matches!(error, CodingAgentBuildError::Selector { .. }));
    }

    #[tokio::test]
    async fn a_model_that_names_no_profile_is_refused() {
        let error = CodingRuntime::builder(client())
            .model("bare/plain")
            .environment(Arc::new(MockEnvironment::linux()))
            .with_profile(TestProfile::shared())
            .build()
            .expect_err("nothing names a harness");

        assert!(matches!(
            error,
            CodingAgentBuildError::MissingProfileMetadata { ref model } if model == "bare/plain"
        ));
    }

    #[tokio::test]
    async fn a_profile_pebble_does_not_know_is_refused() {
        let error = CodingRuntime::builder(client())
            .model("test/strange")
            .environment(Arc::new(MockEnvironment::linux()))
            .build()
            .expect_err("`nonesuch` is not a pebble profile");

        assert!(matches!(
            error,
            CodingAgentBuildError::UnknownProfile { ref profile, .. } if profile == "nonesuch"
        ));
    }

    #[tokio::test]
    async fn a_models_profile_beats_its_providers() {
        let resolved = |selector: &str| {
            CodingRuntime::builder(client())
                .model(selector)
                .environment(Arc::new(MockEnvironment::linux()))
                .build()
                .expect("the session builds")
                .profile_kind()
        };

        // The model row names `anthropic`; its provider row names `openai`.
        assert_eq!(resolved("test/model"), AgentProfileKind::Anthropic);
        // This row names nothing, so the provider's answer stands.
        assert_eq!(resolved("test/inherited"), AgentProfileKind::OpenAi);
    }

    #[tokio::test]
    async fn a_built_session_pins_what_it_resolved() {
        let session = session();

        assert_eq!(session.provider(), "test");
        assert_eq!(session.model(), "model");
        assert_eq!(session.model_selector, "test/model");
        assert_eq!(session.profile_kind(), AgentProfileKind::Anthropic);
        assert_eq!(session.model_facts().context_window_tokens, 200_000);
        assert_eq!(session.state(), CodingAgentState::Idle);
        assert!(session.id().starts_with("ses_"));
        assert_eq!(session.root_session_id(), session.id());
    }

    #[tokio::test]
    async fn the_catalog_says_which_models_reason_without_being_asked() {
        let facts = |selector: &str| {
            CodingRuntime::builder(client())
                .model(selector)
                .environment(Arc::new(MockEnvironment::linux()))
                .with_profile(TestProfile::shared())
                .build()
                .expect("the session builds")
                .model_facts()
                .reasons_by_default
        };

        assert!(!facts("test/model"), "a model that cannot reason");
        assert!(facts("test/thinking"), "a model that takes an effort level");
        assert!(
            facts("test/always-thinking"),
            "a row that says so itself, where the capabilities cannot"
        );
    }

    // --- Initializing ---

    #[tokio::test]
    async fn initializing_reports_what_it_loaded_and_where_it_is_working() {
        let mut session = session();
        let mut events = session.subscribe();
        let env_context = session
            .build_env_context(&CancellationToken::new())
            .await
            .expect("the environment is described");

        session.initialize().await.expect("initialization succeeds");

        let published = settled(&mut session, &mut events).await;
        assert_eq!(event_names(&published), [
            "started", "memory", "skills", "ended"
        ]);
        assert!(matches!(
            &published[1],
            CodingEvent::MemoryLoaded {
                files,
                total_loaded_bytes: 0,
                budget_bytes: MEMORY_BUDGET_BYTES,
                ..
            } if files.is_empty()
        ));
        assert!(
            session
                .system_prompt
                .contains("test assistant working in /home/test")
        );
        assert_eq!(env_context.knowledge_cutoff, "May 2026");
        assert_eq!(env_context.model, "model");
        assert_eq!(env_context.platform, "linux");
        assert_eq!(
            env_context.current_date.len(),
            10,
            "an environment that cannot date itself still dates the prompt"
        );
    }

    #[tokio::test]
    async fn initializing_a_cancelled_session_stops() {
        let mut session = session();
        session.interrupt();

        let error = session
            .initialize()
            .await
            .expect_err("a cancelled session initializes nothing");

        assert!(matches!(
            error,
            Error::Interrupted(InterruptReason::Cancelled)
        ));
    }

    // --- Running ---

    #[tokio::test]
    async fn a_text_answer_ends_the_prompt() {
        let (mut session, _provider) =
            TestSession::answering(vec![ScriptedCall::response(text_response("all done"))]);
        let mut events = session.subscribe();
        session.initialize().await.expect("initialization succeeds");

        let answer = session
            .prompt("fix the test")
            .await
            .expect("the prompt succeeds");

        assert_eq!(answer.as_deref(), Some("all done"));
        assert_eq!(session.history().len(), 2, "the input and the answer");
        assert_eq!(session.state(), CodingAgentState::Idle);
        assert_eq!(event_names(&settled(&mut session, &mut events).await), [
            "started",
            "memory",
            "skills",
            "input",
            "request",
            "first_output",
            "delta",
            "message",
            "processing_end",
            "ended",
        ]);
    }

    #[tokio::test]
    async fn an_interrupt_is_announced_once_and_its_steer_follows_it() {
        let (mut session, provider) = TestSession::answering(vec![
            ScriptedCall::PendingOpen,
            ScriptedCall::response(text_response("done")),
        ]);
        let mut events = session.subscribe();
        // Two gestures against one hanging round: one round to settle, two
        // generations to announce.
        let handle = session.control_handle();
        let controller = tokio::spawn(async move {
            provider.wait_for_call().await;
            assert!(handle.interrupt());
            handle.steer("do this instead");
        });

        timeout(Duration::from_secs(5), session.prompt("do a thing"))
            .await
            .expect("the steer unblocks the hanging call")
            .expect("the prompt succeeds");
        controller.await.expect("the controller finishes");

        let published = settled(&mut session, &mut events).await;
        let interrupts: Vec<u64> = published
            .iter()
            .filter_map(|event| match event {
                CodingEvent::RoundInterrupted { generation } => Some(*generation),
                _ => None,
            })
            .collect();
        assert_eq!(interrupts, [1, 2], "one announcement per gesture");
        let position = |matcher: fn(&CodingEvent) -> bool| {
            published
                .iter()
                .position(&matcher)
                .expect("the event was published")
        };
        assert!(
            position(|event| matches!(event, CodingEvent::RoundInterrupted { generation: 2 }))
                < position(|event| matches!(event, CodingEvent::SteeringInjected { .. })),
            "the interrupt settles before its steer is delivered"
        );
        assert!(matches!(
            session.history().turns()[1],
            Message::Steering { .. }
        ));
    }

    #[tokio::test]
    async fn a_closed_session_refuses_input() {
        let mut session = session();
        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the shutdown succeeds");

        let error = session
            .prompt("anything")
            .await
            .expect_err("the session ended");

        assert!(matches!(error, Error::SessionClosed));
    }

    // --- Shutting down ---

    #[tokio::test]
    async fn only_the_first_shutdown_does_anything() {
        let mut session = session();
        let mut events = session.subscribe();

        assert!(
            session
                .shutdown(ShutdownReason::Completed)
                .await
                .expect("the shutdown succeeds")
        );
        assert!(
            !session
                .shutdown(ShutdownReason::Completed)
                .await
                .expect("a second shutdown does nothing")
        );

        let mut published = Vec::new();
        while let Ok(event) = events.try_recv() {
            published.push(event.event);
        }
        assert_eq!(
            published
                .iter()
                .filter(|event| matches!(event, CodingEvent::SessionEnded))
                .count(),
            1
        );
        assert_eq!(session.state(), CodingAgentState::Closed);
    }

    #[tokio::test]
    async fn shutting_down_ends_the_streams_the_session_handed_out() {
        let mut session = session();
        // The renderer an application writes: read the stream until it ends,
        // then report. Nothing tells it to stop except the stream itself.
        let mut events = session.subscribe();
        let renderer = tokio::spawn(async move {
            let mut seen = 0_usize;
            loop {
                match events.recv().await {
                    Ok(_) => seen += 1,
                    Err(RecvError::Lagged(_)) => {}
                    Err(RecvError::Closed) => break,
                }
            }
            seen
        });
        session
            .prompt("do a thing")
            .await
            .expect("the prompt succeeds");

        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the shutdown succeeds");

        let seen = timeout(Duration::from_secs(5), renderer)
            .await
            .expect("the stream ends when the session is shut down, not when it is dropped")
            .expect("the renderer finishes");
        assert!(seen > 0, "the renderer read the prompt it was watching");
        // Read after the join on purpose: the session is still alive here,
        // which is the order an application works in — wait for the renderer,
        // then let the session go.
        assert_eq!(session.state(), CodingAgentState::Closed);
        assert!(
            matches!(session.subscribe().recv().await, Err(RecvError::Closed)),
            "subscribing to a closed session answers with a stream that has ended"
        );
    }

    /// A sink that records nothing, so the pump stops on the first event.
    struct RefusingSink;

    #[async_trait]
    impl EventSink for RefusingSink {
        async fn record(&self, _event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
            Err(EventSinkError::new("the disk is full"))
        }
    }

    /// Holds the user event until a test lets the next model request begin.
    #[derive(Debug, Default)]
    struct UserInputGate {
        reached: Notify,
        release: Notify,
    }

    #[async_trait]
    impl EventSink for UserInputGate {
        async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
            if matches!(event.event, CodingEvent::UserInput { .. }) {
                self.reached.notify_one();
                self.release.notified().await;
            }
            Ok(())
        }
    }

    /// Accepts setup events, then breaks while a model stream is still open.
    struct RefuseTextDeltaSink;

    #[async_trait]
    impl EventSink for RefuseTextDeltaSink {
        async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
            if matches!(event.event, CodingEvent::TextDelta { .. }) {
                return Err(EventSinkError::new("the event store disconnected"));
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_model_request_waits_until_its_input_is_durable() {
        let sink = Arc::new(UserInputGate::default());
        let (client, provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut session = builder(client)
            .event_sink(Arc::clone(&sink) as Arc<dyn EventSink>)
            .build()
            .expect("the session builds");
        let prompt = session.prompt("do a thing");
        tokio::pin!(prompt);

        tokio::select! {
            () = sink.reached.notified() => {}
            result = &mut prompt => panic!("the prompt passed its durability boundary: {result:?}"),
        }
        assert_eq!(
            provider.call_count(),
            0,
            "the model is not called before its input reaches the sink"
        );

        sink.release.notify_one();
        prompt.await.expect("the prompt continues after the commit");
    }

    #[tokio::test]
    async fn a_mid_stream_sink_failure_cancels_the_model_and_keeps_its_error() {
        let (client, _provider) = scripted_client(vec![ScriptedCall::EventsThenPending(
            text_delta_events("partial"),
        )]);
        let mut session = builder(client)
            .event_sink(Arc::new(RefuseTextDeltaSink))
            .build()
            .expect("the session builds");

        let failure = timeout(Duration::from_secs(5), session.prompt("do a thing"))
            .await
            .expect("the failed stream cancels a model response that never ends")
            .expect_err("the prompt reports the sink failure");

        assert_eq!(failure.kind(), ErrorKind::EventStream);
        assert!(
            ErrorData::from(&failure)
                .message
                .contains("the event store disconnected")
        );
        assert_eq!(session.state(), CodingAgentState::Closed);
        assert!(session.ended, "the failing prompt completed its shutdown");
        assert!(session.pump.is_none(), "the failing prompt joined its pump");
    }

    #[tokio::test]
    async fn a_refusing_sink_stops_the_prompt() {
        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut session = builder(client)
            .event_sink(Arc::new(RefusingSink))
            .build()
            .expect("the session builds");

        let failure = session
            .prompt("do a thing")
            .await
            .expect_err("the prompt waits for its sink failure");

        assert_eq!(failure.kind(), ErrorKind::EventStream);
        assert!(
            ErrorData::from(&failure)
                .message
                .contains("the disk is full")
        );
    }

    #[tokio::test]
    async fn a_session_whose_sink_refused_takes_no_further_input() {
        let (client, provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut session = builder(client)
            .event_sink(Arc::new(RefusingSink))
            .build()
            .expect("the session builds");
        let mut events = session.subscribe();

        let failure = session
            .prompt("do a thing")
            .await
            .expect_err("the current prompt reports the sink failure");
        let calls_before = provider.call_count();

        assert_eq!(failure.kind(), ErrorKind::EventStream);
        assert_eq!(
            session.state(),
            CodingAgentState::Closed,
            "a session whose events go nowhere stops"
        );
        assert!(
            matches!(
                session.prompt("and another").await,
                Err(Error::SessionClosed)
            ),
            "the next prompt is refused rather than run blind"
        );
        assert_eq!(
            provider.call_count(),
            calls_before,
            "the refused prompt asks the model nothing"
        );
        assert!(
            events.try_recv().is_err(),
            "nothing reached a subscriber after the sink refused"
        );
    }

    // --- The state machine ---

    #[tokio::test]
    async fn a_tool_round_moves_the_state_through_executing_and_back() {
        // The moves the bridge makes around a tool round, in the order the loop
        // makes them. A move the table does not allow panics in a debug build,
        // so reaching the end is the assertion that the table allows them all.
        let (mut session, _provider) = TestSession::answering(vec![]);
        let mut events = session.subscribe();
        let machine = session.state_machine();

        machine.transition(CodingAgentState::Thinking);
        machine.transition(CodingAgentState::Executing);
        assert_eq!(
            session.state(),
            CodingAgentState::Executing,
            "the session reads the state the bridge moved"
        );
        machine.transition(CodingAgentState::Thinking);
        assert_eq!(session.state(), CodingAgentState::Thinking);
        machine.transition(CodingAgentState::Idle);
        assert_eq!(session.state(), CodingAgentState::Idle);

        let published = settled(&mut session, &mut events).await;
        assert_eq!(
            published
                .iter()
                .filter(|event| matches!(event, CodingEvent::ProcessingEnd))
                .count(),
            1,
            "returning to idle ends one processing cycle, and executing ends none"
        );
    }

    // --- Storing and resuming ---

    #[tokio::test]
    async fn a_session_round_trips_through_its_record() {
        let mut session = session();
        session
            .prompt("do a thing")
            .await
            .expect("the prompt succeeds");
        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the shutdown succeeds");
        let record = session.to_record();

        let resumed = CodingRuntime::from_record(
            record.clone(),
            &ResumeMode::RecordedModel,
            builder(client()),
        )
        .expect("the record restores");

        assert_eq!(resumed.id(), session.id());
        assert_eq!(resumed.root_session_id(), session.id());
        assert_eq!(resumed.history().turns(), session.history().turns());
        assert_eq!(record.provider.as_deref(), Some("test"));
        assert_eq!(record.model.as_deref(), Some("model"));
        assert!(record.last_event_seq > 0);
        assert_eq!(
            resumed.state(),
            CodingAgentState::Idle,
            "a resumed session is idle whatever ended the last one"
        );
    }

    #[tokio::test]
    async fn a_resumed_session_keeps_numbering_where_it_left_off() {
        let mut session = session();
        session
            .prompt("do a thing")
            .await
            .expect("the prompt succeeds");
        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the shutdown succeeds");
        let record = session.to_record();
        let last_seq = record.last_event_seq;

        let resumed = CodingRuntime::from_record(
            record.clone(),
            &ResumeMode::RecordedModel,
            builder(client()),
        )
        .expect("the record restores");
        let mut events = resumed.subscribe();
        resumed.emit(CodingEvent::LoopDetected);

        assert_eq!(
            events.recv().await.expect("the event is published").seq,
            last_seq + 1
        );
    }

    #[tokio::test]
    async fn a_record_taken_mid_life_covers_the_events_still_queued() {
        // The checkpoint case: a record is stored while the session runs on,
        // and the numbers a resumed session would issue must start above every
        // event this one has already emitted, published or not.
        let mut session = session();
        let mut events = session.subscribe();
        session
            .prompt("do a thing")
            .await
            .expect("the prompt succeeds");

        let record = session.to_record();

        let mut published = Vec::new();
        for _ in 0..8 {
            while let Ok(event) = events.try_recv() {
                published.push(event.seq);
            }
            yield_now().await;
        }
        assert!(
            !published.is_empty(),
            "the prompt published events for the record to cover"
        );
        assert!(
            published.iter().all(|seq| *seq <= record.last_event_seq),
            "a record taken while the pipeline is behind still covers what the \
             prompt emitted: {published:?} against {}",
            record.last_event_seq
        );
        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the shutdown succeeds");
    }

    #[tokio::test]
    async fn a_record_from_a_newer_pebble_is_refused() {
        let mut record = SessionRecord::new("ses_1");
        record.format_version = SESSION_RECORD_FORMAT_VERSION + 1;

        let error = CodingRuntime::from_record(
            record.clone(),
            &ResumeMode::RecordedModel,
            builder(client()),
        )
        .expect_err("this build is too old for the record");

        assert!(matches!(
            error,
            CodingAgentBuildError::UnsupportedRecord { version, supported }
                if version == SESSION_RECORD_FORMAT_VERSION + 1
                    && supported == SESSION_RECORD_FORMAT_VERSION
        ));
    }

    // --- Small parts ---

    #[test]
    fn a_date_is_recognized_by_its_shape() {
        assert!(is_iso_date("2026-08-31"));
        assert!(!is_iso_date("mock output"));
        assert!(!is_iso_date("2026-08-3"));
        assert!(!is_iso_date("2026/08/31"));
        assert!(!is_iso_date(""));
    }

    #[test]
    fn only_a_credential_failure_closes_a_session() {
        for kind in [LlmErrorKind::Authentication, LlmErrorKind::AccessDenied] {
            assert!(is_auth_error(&LlmError::new(kind, "no")));
        }
        for kind in [
            LlmErrorKind::RateLimit,
            LlmErrorKind::Server,
            LlmErrorKind::Network,
        ] {
            assert!(!is_auth_error(&LlmError::new(kind, "no")));
        }
    }

    #[tokio::test]
    async fn a_scripted_failure_reaches_the_session_as_the_error_it_names() {
        let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::Failure(
            ScriptedFailure::terminal(LlmErrorKind::Server, "the provider is down"),
        )]);

        let error = session
            .prompt("anything")
            .await
            .expect_err("the call failed");

        assert!(
            matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::Server),
            "{error:?}"
        );
    }
}
