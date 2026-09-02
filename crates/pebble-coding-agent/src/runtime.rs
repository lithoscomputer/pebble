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

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::SystemTime;

use lithos_llm::Client;
use lithos_llm::catalog::{Metadata, ModelHandle};
use lithos_llm::resolver::ResolvedRoute;
use lithos_llm::types::{
    Error as LlmError, ErrorKind as LlmErrorKind, ReasoningEffort, Request, Speed,
};
use pebble_agent::{Agent, AgentControlHandle};
use serde::Deserialize;
use tokio::sync::{Notify, broadcast};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use self::control::ControlState;
pub use self::control::SteeringLease;
pub(crate) use self::control::{SessionControlHandle, SteeringItem};
pub use self::retry::RetryEventObserver;
use self::turn::CodingAgentBridge;
pub(crate) use crate::coding_agent::{CodingAgentBuildError, PromptTiming, ShutdownReason};
use crate::config::CodingAgentOptions;
use crate::context_window::{memory_prompt_tokens, skills_prompt_tokens};
use crate::environment::{Environment, ExecRequest};
use crate::error::{Error, ErrorData, InterruptReason, Result};
use crate::event::{Emitter, EventCapacity, EventOptions, EventPump, EventSink};
use crate::file_tracker::FileTracker;
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::memory::{MEMORY_BUDGET_BYTES, MemoryDocument, load_memory};
use crate::profile::{AgentProfile, EnvContext, ModelFacts, SubagentSupport, builtin_profile};
use crate::profiles::{FileEditToolKind, ProfileDeps};
use crate::record::{SESSION_RECORD_FORMAT_VERSION, SessionRecord};
use crate::redact::{NoRedaction, Redactor};
use crate::search::SearchProvider;
use crate::skills::{Skill, SkillExpansion, discover_skills};
use crate::subagent::{
    ChildAgentFactory, ChildDeps, ChildIdentity, OpenSessions, SubagentCallbackEvent,
    SubagentEventCallback, SubagentLimits, SubagentSupervisor,
};
use crate::tool::{
    NativeTool, RegisteredTool, StaticEnvProvider, ToolDefinitionWithSource, ToolEnvProvider,
    ToolRegistry,
};
use crate::tools::{
    WebFetchSummarizer, make_question_tool, make_use_skill_tool_for_vocabulary,
    make_web_search_tool,
};
use crate::types::{
    AgentProfileKind, CodingAgentEvent, CodingAgentState, CodingEvent, PermissionLevel, TokenUsage,
    rfc3339_millis,
};

/// The catalog metadata namespace pebble reads.
const METADATA_NAMESPACE: &str = "pebble";

/// How long a probe run inside the environment may take.
const PROBE_TIMEOUT_MS: u64 = 5_000;

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
    human_input:          Option<Arc<dyn HumanInputProvider>>,
    tool_env_provider:    Option<Arc<dyn ToolEnvProvider>>,
    redactor:             Arc<dyn Redactor>,
    web_fetch_summarizer: Option<String>,
    search_provider:      Option<Arc<dyn SearchProvider>>,
    options:              CodingAgentOptions,
    events:               EventOptions,
    profile:              Option<Arc<dyn AgentProfile>>,
    subagents:            Option<ChildAgentFactory>,
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
            human_input: None,
            tool_env_provider: None,
            redactor: Arc::new(NoRedaction),
            web_fetch_summarizer: None,
            search_provider: None,
            options: CodingAgentOptions::default(),
            events: EventOptions::default(),
            profile: None,
            subagents: None,
            subagent_limits: SubagentLimits::default(),
            child: None,
        }
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

    /// Adds tools on top of the ones the profile contributes.
    ///
    /// The registry renames pebble's own tools into the profile's vocabulary as
    /// they arrive, so a built-in registered here still reaches the model under
    /// the name that model expects.
    pub(crate) fn tools(mut self, tools: impl IntoIterator<Item = RegisteredTool>) -> Self {
        self.tools.extend(tools);
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

    /// Sets how many events the live stream buffers for each subscriber.
    pub(crate) fn event_capacity(mut self, capacity: impl Into<EventCapacity>) -> Self {
        self.events.capacity = capacity.into();
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
    /// Pebble builds the
    /// [`ChildAgentSpec`](crate::subagents::ChildAgentSpec) each
    /// call receives from this
    /// session, so a child inherits the environment, the tools, the access
    /// policy, and the hooks its parent had, and never a
    /// [`HumanInputProvider`]: a child cannot ask a person a question.
    pub(crate) fn subagents(mut self, factory: ChildAgentFactory) -> Self {
        self.subagents = Some(factory);
        self
    }

    /// Sets how many sessions this tree may hold open at once.
    ///
    /// Set on the root; children inherit the root's counter, so the limit
    /// covers the whole tree however deep it goes. Without a
    /// [`ChildAgentFactory`](Self::subagents) it does nothing.
    pub(crate) const fn subagent_limits(mut self, limits: SubagentLimits) -> Self {
        self.subagent_limits = limits;
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
        let (parent_session_id, root_session_id, depth, open_sessions, built_from) =
            match self.child {
                Some(child) => (
                    Some(child.parent_session_id),
                    child.root_session_id,
                    child.depth,
                    child.open_sessions,
                    Some(child.built_from),
                ),
                None => (
                    None,
                    id.clone(),
                    0,
                    OpenSessions::root(self.subagent_limits),
                    None,
                ),
            };

        let supervisor = self.subagents.map(|factory| {
            SubagentSupervisor::new(Arc::new(ChildDeps {
                client: self.client.clone(),
                model_selector: handle.to_string(),
                profile: Arc::clone(&profile),
                environment: Arc::clone(&environment),
                tools: self.tools,
                options: child_options(&self.options),
                tool_env_provider: self.tool_env_provider.clone(),
                redactor: Arc::clone(&self.redactor),
                search_provider: self.search_provider.clone(),
                event_capacity: self.events.capacity,
                factory,
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

        let (emitter, pump) = EventPump::new(self.events);
        let pump = tokio::spawn(pump.run());

        let session = CodingRuntime {
            root_session_id,
            parent_session_id,
            built_from,
            id,
            created_at,
            config: self.options,
            history: History::default(),
            emitter,
            pump: Some(pump),
            state: CodingAgentState::Idle,
            ended: false,
            client: self.client,
            profile,
            provider: handle.provider().as_str().to_owned(),
            model: handle.model().as_str().to_owned(),
            model_selector: handle.to_string(),
            facts,
            knowledge_cutoff: metadata.knowledge_cutoff.unwrap_or_default(),
            registry,
            env: environment,
            human_input: self.human_input,
            tool_env_provider: self.tool_env_provider,
            redactor: self.redactor,
            control_state: Arc::new(Mutex::new(ControlState::default())),
            control_notify: Arc::new(Notify::new()),
            active_agent_control: Arc::new(Mutex::new(None)),
            followup_queue: Arc::new(Mutex::new(VecDeque::new())),
            cancel_token: CancellationToken::new(),
            interrupt_reason: Arc::new(Mutex::new(None)),
            skills: Vec::new(),
            memory_tokens: 0,
            skills_tokens: 0,
            system_prompt: String::new(),
            activated_skill_context_observed: false,
            file_tracker: FileTracker::default(),
            subagents: supervisor,
            last_prompt: PromptTotals::default(),
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
/// access policy, the hooks, the permission level, the output budgets, the
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
    id: String,
    /// The root of this session's tree. A root session names itself; a child
    /// inherits its parent's root, which is how root-scoped tools — one shared
    /// todo list across a tree of agents — know where they belong.
    root_session_id: String,
    /// The session that spawned this one, for a child. A root has none.
    parent_session_id: Option<String>,
    /// What this child was built from, for the supervisor to recognize its own
    /// specification in the session a factory answered with. A root has none.
    built_from: Option<Arc<ChildDeps>>,
    created_at: SystemTime,
    config: CodingAgentOptions,
    history: History,
    emitter: Emitter,
    /// The event pump, until [`CodingRuntime::shutdown`] joins it.
    pump: Option<JoinHandle<Result<()>>>,
    state: CodingAgentState,
    ended: bool,
    client: Client,
    profile: Arc<dyn AgentProfile>,
    provider: String,
    model: String,
    /// What every request names, which is the resolved `provider/model` pair
    /// rather than the selector the application gave, so no round can drift to
    /// a different model than the one whose harness the session is running.
    model_selector: String,
    facts: ModelFacts,
    knowledge_cutoff: String,
    registry: ToolRegistry,
    env: Arc<dyn Environment>,
    human_input: Option<Arc<dyn HumanInputProvider>>,
    tool_env_provider: Option<Arc<dyn ToolEnvProvider>>,
    /// What strips secrets out of the process output this session publishes.
    redactor: Arc<dyn Redactor>,
    control_state: Arc<Mutex<ControlState>>,
    control_notify: Arc<Notify>,
    active_agent_control: Arc<Mutex<Option<AgentControlHandle>>>,
    followup_queue: Arc<Mutex<VecDeque<String>>>,
    /// Ends the whole prompt. Distinct from the round token, which ends one
    /// turn.
    cancel_token: CancellationToken,
    interrupt_reason: Arc<Mutex<Option<InterruptReason>>>,
    skills: Vec<Skill>,
    /// What the memory files and the skills section contribute to the system
    /// prompt, measured once at initialization: both are fixed for the
    /// session's life, and every round's context snapshot reads them.
    memory_tokens: u64,
    skills_tokens: u64,
    system_prompt: String,
    activated_skill_context_observed: bool,
    file_tracker: FileTracker,
    subagents: Option<SubagentSupervisor>,
    last_prompt: PromptTotals,
    /// The provider-neutral conversation loop, created after initialization on
    /// the first prompt and retained for the rest of the session.
    coding_agent: Option<Agent>,
    /// Coding state and durable projection shared with `coding_agent`.
    coding_bridge: Option<Arc<CodingAgentBridge>>,
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
            .field("state", &self.state)
            .field("ended", &self.ended)
            .field("turns", &self.history.len())
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
    /// client, the environment, the tools, the options. Event numbering
    /// continues from the record, so one session's events stay uniquely
    /// numbered across restarts.
    ///
    /// # Errors
    ///
    /// Returns [`CodingAgentBuildError`] for the same reasons
    /// [`CodingRuntimeBuilder::build`] does, and
    /// [`UnsupportedRecord`](CodingAgentBuildError::UnsupportedRecord) for a
    /// record this build is too old to read.
    pub(crate) fn from_record(
        record: &SessionRecord,
        deps: CodingRuntimeBuilder,
    ) -> StdResult<Self, CodingAgentBuildError> {
        if !record.is_supported() {
            return Err(CodingAgentBuildError::UnsupportedRecord {
                version:   record.format_version,
                supported: SESSION_RECORD_FORMAT_VERSION,
            });
        }

        let mut deps = deps;
        if let Some(model) = &record.model {
            deps.model = Some(model.clone());
        }
        deps.events.resume_after_seq = record.last_event_seq;
        let mut session = deps.build_with_id(record.session_id.clone(), record.created_at)?;
        session.history = History::from_stored_messages(&record.messages);
        // The parentage the record carries is restored, so storing a resumed
        // child again says the same thing. The tree itself is not: a resumed
        // child has no supervisor above it, and rebuilding one is the
        // application's to do.
        session
            .parent_session_id
            .clone_from(&record.parent_session_id);
        Ok(session)
    }

    /// The session as it should be stored.
    ///
    /// Everything a resumed session needs and nothing an application could not
    /// supply again. A child records which session spawned it, so a stored tree
    /// can be read back in shape; the tree itself is the application's to
    /// rebuild, because a child's supervisor is not stored.
    ///
    /// A record can be taken at any time, including mid-prompt as a crash
    /// checkpoint: the event numbering it stores covers every event emitted
    /// before the call, whether or not the pipeline has published it yet, so a
    /// session resumed from the record never reuses a number.
    pub(crate) fn to_record(&self) -> SessionRecord {
        let mut record = SessionRecord::new(self.id.clone());
        record.parent_session_id.clone_from(&self.parent_session_id);
        record.provider = Some(self.provider.clone());
        record.model = Some(self.model.clone());
        record.created_at = self.created_at;
        record.last_event_seq = self.emitter.last_seq();
        record.messages = self.history.to_stored_messages();
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
        // carry the bytes of a project's own instructions.
        self.emit(CodingEvent::MemoryLoaded {
            profile:            profile.clone(),
            files:              memory.iter().map(MemoryDocument::to_summary).collect(),
            total_loaded_bytes: memory.iter().map(|document| document.loaded_bytes).sum(),
            budget_bytes:       MEMORY_BUDGET_BYTES,
        });

        self.skills = skills?;
        debug!(skill_count = self.skills.len(), "Skills discovered");
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
        self.system_prompt = self.profile.build_system_prompt(
            &self.registry,
            &env_context,
            &memory,
            self.config.user_instructions.as_deref(),
            &self.skills,
        );

        Ok(())
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
    /// built, by whoever spawned it. Root-scoped tools, forwarded events and
    /// stored records all key on this, so a session that could be re-rooted
    /// afterwards could be detached from the tree that owns it.
    pub(crate) fn root_session_id(&self) -> &str {
        &self.root_session_id
    }

    /// Which harness this session runs.
    #[cfg(test)]
    pub(crate) fn profile_kind(&self) -> AgentProfileKind {
        self.profile.profile_kind()
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

    /// The permission level the application recorded for this session.
    pub(crate) fn permission_level(&self) -> Option<PermissionLevel> {
        self.config.permission_level
    }

    /// What the session is doing right now.
    pub(crate) const fn state(&self) -> CodingAgentState {
        self.state
    }

    /// The conversation so far.
    pub(crate) const fn history(&self) -> &History {
        &self.history
    }

    /// The files this session has read and changed.
    #[cfg(test)]
    pub(crate) const fn file_tracker(&self) -> &FileTracker {
        &self.file_tracker
    }

    /// Where the last prompt spent its time.
    pub(crate) const fn last_prompt_timing(&self) -> PromptTiming {
        self.last_prompt.timing
    }

    /// What the last prompt cost in tokens, summed over every response.
    pub(crate) const fn last_prompt_usage(&self) -> TokenUsage {
        self.last_prompt.usage
    }

    /// What the last prompt cost in USD micros, where the catalog or the
    /// provider priced it.
    pub(crate) const fn last_prompt_cost_usd_micros(&self) -> Option<u64> {
        self.last_prompt.cost_usd_micros
    }

    /// The tools the model is actually shown, after the access policy.
    ///
    /// The same filter the session builds its requests with, so what an
    /// application reads here is what the model was told.
    pub(crate) fn effective_tools(&self) -> Vec<ToolDefinitionWithSource> {
        self.registry.definitions_with_source_for_policy(
            self.config.tool_access_policy.as_deref(),
            self.config.tool_exposure_mode,
        )
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
    pub(crate) fn control_handle(&self) -> SessionControlHandle {
        SessionControlHandle::attached(
            Arc::clone(&self.control_state),
            Arc::clone(&self.control_notify),
            Arc::clone(&self.active_agent_control),
        )
    }

    /// Queues guidance for the next round.
    #[cfg(test)]
    pub(crate) fn steer(&self, text: impl Into<String>) {
        self.control_handle().steer(text, None);
    }

    /// Hands out a steering lease that parks natural completion while an
    /// external steering source is attached.
    #[cfg(test)]
    pub(crate) fn steering_lease(&self) -> SteeringLease {
        SteeringLease::acquire(self.control_handle())
    }

    /// Queues more input to process once the current input is finished.
    #[cfg(test)]
    pub(crate) fn follow_up(&self, message: impl Into<String>) {
        self.followup_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back(message.into());
    }

    /// The follow-up queue, for a producer outside the session.
    pub(crate) fn followup_queue_handle(&self) -> Arc<Mutex<VecDeque<String>>> {
        Arc::clone(&self.followup_queue)
    }

    /// Ends the prompt.
    ///
    /// The loop unwinds through its own checkpoints — every tool call still
    /// gets its result recorded — and then closes the session. This is the
    /// terminal gesture; [`SessionControlHandle::interrupt`] is the one that
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

    /// Where this session's supervisor sends what its children produce.
    ///
    /// Lifecycle facts about a child are the parent's own news, so they are
    /// published under the parent's identity. A child's own event is forwarded
    /// as it stands — its `session_id` untouched — and only gains a
    /// `parent_session_id` when it does not already have one, so a grandchild's
    /// event keeps naming its real parent. Forwarded events go through this
    /// session's pump, which is what gives them parent-stream sequence numbers.
    pub(crate) fn sub_agent_event_callback(&self) -> SubagentEventCallback {
        let emitter = self.emitter.clone();
        let parent_session_id = self.id.clone();
        Arc::new(move |event| match event {
            SubagentCallbackEvent::Lifecycle(event) => {
                emitter.emit(parent_session_id.clone(), event);
            }
            SubagentCallbackEvent::Forwarded(mut event) => {
                if event.parent_session_id.is_none() {
                    event.parent_session_id = Some(parent_session_id.clone());
                }
                emitter.forward(event);
            }
        })
    }

    /// Whether this session was built from `deps`, which is how a supervisor
    /// recognizes the child it specified in the session a factory answered
    /// with.
    pub(crate) fn was_built_from(&self, deps: &Arc<ChildDeps>) -> bool {
        self.built_from
            .as_ref()
            .is_some_and(|built_from| Arc::ptr_eq(built_from, deps))
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
    pub(crate) fn set_tool_env_provider(&mut self, provider: Arc<dyn ToolEnvProvider>) {
        self.tool_env_provider = Some(Arc::clone(&provider));
        if let Some(bridge) = &self.coding_bridge {
            bridge.set_tool_env_provider(provider);
        }
    }

    /// Sets fixed extra environment variables for every tool call.
    pub(crate) fn set_tool_env(&mut self, env: HashMap<String, String>) {
        self.set_tool_env_provider(Arc::new(StaticEnvProvider(env)));
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
    pub(crate) async fn prompt(&mut self, input: &str) -> Result<Option<String>> {
        self.prompt_with_cancellation(input, None).await
    }

    /// Processes one input until it completes or `cancel` fires.
    ///
    /// Cancelling `cancel` ends this prompt alone: the loop unwinds through its
    /// checkpoints so every tool call still gets its result, the prompt reports
    /// [`Error::Interrupted`], and the session returns to
    /// [`Idle`](CodingAgentState::Idle) ready for its next prompt. Only a
    /// shutdown closes the session.
    ///
    /// # Errors
    ///
    /// As [`prompt`](Self::prompt).
    pub(crate) async fn prompt_with_cancellation(
        &mut self,
        input: &str,
        cancel: Option<&CancellationToken>,
    ) -> Result<Option<String>> {
        self.last_prompt = PromptTotals::default();
        if self.state == CodingAgentState::Closed {
            return Err(Error::SessionClosed);
        }

        // A child of the terminal token, so a shutdown ends the prompt too, and
        // linked to the caller's token where one was given.
        let prompt_cancel = self.cancel_token.child_token();
        let link = cancel.map(|caller| link_cancellation(caller, &prompt_cancel));

        let timer = self.start_wall_clock_timer(&prompt_cancel);
        let result = self
            .process_input(input, SkillExpansion::Apply, &prompt_cancel)
            .await;

        if let Some(link) = link {
            link.abort();
        }
        let mut task_failure = stop_wall_clock_timer(timer).await;
        // The reason has been reported by now. Clearing it here rather than at
        // the start of the next prompt keeps a reason a watchdog records just
        // before it cancels, and still stops one prompt's reason reaching the
        // next.
        if self.state != CodingAgentState::Closed {
            *self
                .interrupt_reason
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = None;
        }

        if self.state == CodingAgentState::Closed {
            let reason = if self.cancel_token.is_cancelled() {
                ShutdownReason::Cancelled
            } else {
                ShutdownReason::Error
            };
            if let Err(error) = self.shutdown(reason).await {
                task_failure = task_failure.or(Some(error));
            }
        } else {
            self.transition(CodingAgentState::Idle);
            // A sink that refused an event during this prompt is this prompt's
            // failure, even where the loop got to a boundary too late to
            // notice it.
            if let Err(error) = self.check_pump().await {
                task_failure = task_failure.or(Some(error));
            }
        }

        match (result, task_failure) {
            // The prompt's own failure is the story; a task that also failed on
            // the way out is reported rather than returned.
            (Err(error), Some(task)) => {
                warn!(%task, "A session task failed while the prompt was already failing");
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
    /// result recorded. The session stays open: running out of time is the
    /// prompt's failure, and the next prompt gets a fresh budget.
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
    /// event, and [`Error::InvalidState`] when a task the session owned failed
    /// outright. The session is closed either way.
    pub(crate) async fn shutdown(&mut self, reason: ShutdownReason) -> Result<bool> {
        if self.ended {
            return Ok(false);
        }
        if reason == ShutdownReason::Cancelled {
            self.set_interrupt_reason(InterruptReason::Cancelled);
            self.cancel_token.cancel();
        }
        self.transition(CodingAgentState::Closed);
        if let Some(supervisor) = &self.subagents {
            supervisor.shutdown_all().await;
        }
        if let Some(agent) = &mut self.coding_agent {
            let _ = agent.shutdown();
        }
        self.ended = true;
        self.emit(CodingEvent::SessionEnded);
        self.join_pump().await?;
        Ok(true)
    }

    /// Publishes everything queued, then joins the pump.
    async fn join_pump(&mut self) -> Result<()> {
        let Some(pump) = self.pump.take() else {
            return Ok(());
        };
        self.emitter.close();
        match pump.await {
            Ok(result) => result,
            Err(error) => Err(Error::InvalidState(format!(
                "the event pump task failed: {error}"
            ))),
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
        let Some(pump) = self.pump.as_ref() else {
            return Ok(());
        };
        if !pump.is_finished() {
            return Ok(());
        }
        let outcome = self.join_pump().await;
        if outcome.is_err() {
            self.transition(CodingAgentState::Closed);
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
            self.transition(CodingAgentState::Closed);
        }
        error
    }

    /// Moves the session's state machine, publishing the end of a processing
    /// cycle where one ends.
    ///
    /// Valid moves: Idle or Executing to Thinking, Thinking to Executing or
    /// Idle, anything to Closed. Ending the session belongs to
    /// [`CodingRuntime::shutdown`], never here.
    fn transition(&mut self, to: CodingAgentState) {
        let from = self.state;
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
                ) | (_, CodingAgentState::Closed)
            ),
            "invalid session state transition: {from:?} -> {to:?}"
        );

        if matches!(
            from,
            CodingAgentState::Thinking | CodingAgentState::Executing
        ) && to == CodingAgentState::Idle
        {
            self.emit(CodingEvent::ProcessingEnd);
        }

        self.state = to;
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

/// The task watching one prompt's wall-clock budget.
struct WallClockTimer {
    stop: CancellationToken,
    task: JoinHandle<()>,
}

/// Stops the timer and joins it, reporting a task that failed outright.
async fn stop_wall_clock_timer(timer: Option<WallClockTimer>) -> Option<Error> {
    let timer = timer?;
    timer.stop.cancel();
    timer
        .task
        .await
        .err()
        .map(|error| Error::InvalidState(format!("the wall-clock timer task failed: {error}")))
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
    use tokio::sync::broadcast::error::RecvError;
    use tokio::task::yield_now;
    use tokio::time::timeout;

    use super::testing::{TestProfile, TestSession, builder, event_names, settled};
    use super::*;
    use crate::error::ErrorKind;
    use crate::event::EventSinkError;
    use crate::test_support::{
        MockEnvironment, ScriptedCall, ScriptedFailure, scripted_client, text_response,
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
        let mut session = session();
        let mut events = session.subscribe();
        // Two gestures before the loop ever runs: one round to settle, two
        // generations to announce.
        let handle = session.control_handle();
        handle.interrupt();
        handle.interrupt_then_steer("do this instead", None);

        session
            .prompt("do a thing")
            .await
            .expect("the prompt succeeds");

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

    #[tokio::test]
    async fn a_refusing_sink_stops_the_prompt() {
        let (client, _provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut session = builder(client)
            .event_sink(Arc::new(RefusingSink))
            .build()
            .expect("the session builds");

        // The pump stops on the first event it is given, which the loop
        // notices at its next round boundary.
        let first = session.prompt("do a thing").await;
        yield_now().await;
        let second = session.prompt("do another thing").await;
        let ((Err(failure), _) | (Ok(_), Err(failure))) = (first, second) else {
            panic!("a refusing sink must stop the prompt");
        };

        assert_eq!(failure.kind(), ErrorKind::EventSink);
        assert!(failure.to_string().contains("the disk is full"));
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

        // The failure surfaces at whichever checkpoint first finds the pump
        // stopped, which is the end of the first prompt or the start of the
        // second.
        let mut failure = None;
        for _ in 0..2 {
            match session.prompt("do a thing").await {
                Ok(_) => yield_now().await,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        let calls_before = provider.call_count();

        let failure = failure.expect("a refusing sink stops a prompt");
        assert_eq!(failure.kind(), ErrorKind::EventSink);
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

        let resumed =
            CodingRuntime::from_record(&record, builder(client())).expect("the record restores");

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

        let resumed =
            CodingRuntime::from_record(&record, builder(client())).expect("the record restores");
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

        let error = CodingRuntime::from_record(&record, builder(client()))
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
