//! What a session can call, and what a tool is handed when it runs.

use std::collections::HashMap;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lithos_llm::types::ToolDefinition;
use pebble_agent::{ToolOutput, ToolScheduling};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::error::ToolError;
use super::native::{NativeTool, ToolVocabulary};
use super::permissions::known_tool_category;
use crate::environment::Environment;
use crate::event::{OutputCaptureStats, SessionBoundEmitter};
use crate::human_input::HumanInputProvider;
use crate::output::ToolOutputStore;
use crate::redact::{NoRedaction, Redactor};
use crate::types::{CodingEvent, ToolCategory, ToolSource, ToolSummary};
use crate::{SessionId, SessionScope};

/// The narrow handle a running tool publishes events through.
///
/// A tool that changes session-visible state — the todo list, a subprocess it
/// ran — says so through this rather than by returning it, because the output
/// the model reads and the events an application observes are different
/// things.
///
/// Implementations must stamp emitted events with the session identity the
/// owning session is using, so a child session's events stay attributable.
pub trait CodingEventEmitter: Send + Sync {
    /// Publishes one event on the owning session's stream.
    fn emit(&self, event: CodingEvent);

    /// Reports how many bytes of model-facing output the running tool
    /// produced.
    ///
    /// A side channel, not an event: the execution layer drains it once the
    /// tool returns and folds it into the call's byte counters. An emitter
    /// with no tool-execution owner may ignore it, which is what the default
    /// does.
    fn record_tool_output_stats(&self, _stats: OutputCaptureStats) {}
}

impl CodingEventEmitter for SessionBoundEmitter {
    fn emit(&self, event: CodingEvent) {
        Self::emit(self, event);
    }

    fn record_tool_output_stats(&self, stats: OutputCaptureStats) {
        Self::record_tool_output_stats(self, stats);
    }
}

/// Extra environment variables one tool call runs with.
///
/// An application implements this when a command's environment is decided per
/// call rather than per session — prompt-scoped credentials, a token that has
/// to be fetched, a value that changes between rounds. The shell tool resolves
/// it immediately before each call.
#[async_trait]
pub trait ToolEnvProvider: Send + Sync {
    /// Produces the variables to add to this call's environment.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] when the variables cannot be produced; the call
    /// fails with its message, so the message is written for the model and
    /// carries no secret.
    async fn resolve(&self) -> Result<HashMap<String, String>, ToolError>;
}

/// A [`ToolEnvProvider`] that hands out the same variables every time.
pub struct StaticEnvProvider(pub HashMap<String, String>);

#[async_trait]
impl ToolEnvProvider for StaticEnvProvider {
    async fn resolve(&self) -> Result<HashMap<String, String>, ToolError> {
        Ok(self.0.clone())
    }
}

/// Everything a tool is given besides its arguments.
///
/// Built once per call by the execution layer. Outside a session — an
/// application exercising one tool directly, a test — the session scope is
/// absent, and tools that need it say so rather than assuming a session.
///
/// New members appear here as tools gain capabilities, so build a context with
/// [`new`](Self::new) and the `with_*` methods rather than a struct literal,
/// and read it through its accessors.
pub struct ToolContext {
    /// Where the tool's work lands.
    pub(crate) env:                  Arc<dyn Environment>,
    /// Fires when this call should stop. Composed from the session's terminal
    /// cancellation and the current model turn's interrupt, so a tool that
    /// watches it observes both.
    pub(crate) cancel:               CancellationToken,
    /// Application-owned storage for complete output.
    pub(crate) output_store:         Option<Arc<dyn ToolOutputStore>>,
    /// References collected before the command's model-facing result is built.
    pub(crate) output_artifacts:     Arc<Mutex<Vec<pebble_agent::ToolArtifact>>>,
    /// Extra environment variables for a command this call runs.
    pub(crate) tool_env_provider:    Option<Arc<dyn ToolEnvProvider>>,
    /// The calling session and the root shared by its tree.
    session_scope:                   Option<SessionScope>,
    /// The model-native identifier of this call.
    pub(crate) tool_call_id:         Option<String>,
    /// Where the tool publishes events.
    pub(crate) coding_event_emitter: Option<Arc<dyn CodingEventEmitter>>,
    /// Where the tool asks the person a question.
    pub(crate) human_input:          Option<Arc<dyn HumanInputProvider>>,
    /// What strips secrets out of text the tool publishes.
    pub(crate) redactor:             Arc<dyn Redactor>,
}

impl ToolContext {
    /// Where the tool's work lands.
    #[must_use]
    pub const fn env(&self) -> &Arc<dyn Environment> {
        &self.env
    }

    /// Fires when this call should stop.
    ///
    /// Composed from the session's terminal cancellation and the current model
    /// turn's interrupt, so a tool that watches it observes both. Watching it
    /// is the tool's own responsibility, and the session waits for the answer
    /// either way: a cancelled call is never dropped, because a call with no
    /// result is a conversation the provider will refuse. A tool that ignores
    /// this token therefore holds its turn — and the prompt ending it — open
    /// until it returns, so long work must watch it and answer.
    #[must_use]
    pub const fn cancel(&self) -> &CancellationToken {
        &self.cancel
    }

    /// Where a command this call runs takes extra environment variables from.
    #[must_use]
    pub const fn tool_env_provider(&self) -> Option<&Arc<dyn ToolEnvProvider>> {
        self.tool_env_provider.as_ref()
    }

    /// The session that called the tool, when the call runs inside one.
    #[must_use]
    pub fn session_id(&self) -> Option<&SessionId> {
        self.session_scope.as_ref().map(SessionScope::session_id)
    }

    /// The root of the session tree this call belongs to. Equal to
    /// [`session_id`](Self::session_id) in a root session; a child inherits
    /// its parent's root.
    #[must_use]
    pub fn root_session_id(&self) -> Option<&SessionId> {
        self.session_scope
            .as_ref()
            .map(SessionScope::root_session_id)
    }

    /// The model-native identifier of this call.
    #[must_use]
    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    /// Where the tool asks the person a question. Absent in child sessions and
    /// wherever the application installed no provider.
    #[must_use]
    pub const fn human_input(&self) -> Option<&Arc<dyn HumanInputProvider>> {
        self.human_input.as_ref()
    }

    /// What strips secrets out of text the tool publishes.
    ///
    /// Only output leaving the session through an event goes through it — the
    /// process tail a shell tool publishes — never what the model is shown,
    /// which is the same text the model would have read from the terminal.
    /// [`NoRedaction`](crate::extensions::NoRedaction) unless the application
    /// installed one.
    #[must_use]
    pub const fn redactor(&self) -> &Arc<dyn Redactor> {
        &self.redactor
    }

    /// A context that has an environment and nothing else.
    #[must_use]
    pub fn new(env: Arc<dyn Environment>) -> Self {
        Self {
            env,
            cancel: CancellationToken::new(),
            output_store: None,
            output_artifacts: Arc::default(),
            tool_env_provider: None,
            session_scope: None,
            tool_call_id: None,
            coding_event_emitter: None,
            human_input: None,
            redactor: Arc::new(NoRedaction),
        }
    }

    /// Sets the token that stops this call.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Sets the calling session and the root of its session tree.
    #[must_use]
    pub fn with_session(mut self, session_scope: SessionScope) -> Self {
        self.session_scope = Some(session_scope);
        self
    }

    /// The calling session and its tree, absent for a call outside a session.
    #[must_use]
    pub const fn session_scope(&self) -> Option<&SessionScope> {
        self.session_scope.as_ref()
    }

    /// Sets the model-native identifier of this call.
    #[must_use]
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Sets storage for complete command output. The store must authorize reads
    /// using the supplied session identity and clean up abandoned captures.
    #[must_use]
    pub fn with_output_store(mut self, store: Arc<dyn ToolOutputStore>) -> Self {
        self.output_store = Some(store);
        self
    }

    /// Sets where extra environment variables come from.
    #[must_use]
    pub fn with_tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env_provider = Some(provider);
        self
    }

    /// Sets where the tool's events go.
    #[must_use]
    pub fn with_coding_event_emitter(mut self, emitter: Arc<dyn CodingEventEmitter>) -> Self {
        self.coding_event_emitter = Some(emitter);
        self
    }

    /// Sets where the tool's questions go.
    #[must_use]
    pub fn with_human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Sets what strips secrets out of text the tool publishes. The
    /// dispatch layer runs the same redactor over the message of every failed
    /// call.
    #[must_use]
    pub fn with_redactor(mut self, redactor: Arc<dyn Redactor>) -> Self {
        self.redactor = redactor;
        self
    }

    /// Resolves the extra environment variables for this call, or `None` when
    /// the session has no provider.
    ///
    /// # Errors
    ///
    /// Returns whatever the provider reported.
    pub async fn resolve_tool_env(&self) -> Result<Option<HashMap<String, String>>, ToolError> {
        match &self.tool_env_provider {
            Some(provider) => provider.resolve().await.map(Some),
            None => Ok(None),
        }
    }

    /// Publishes an event, or does nothing when the context has no emitter.
    pub fn emit_coding_event(&self, event: CodingEvent) {
        if let Some(emitter) = self.coding_event_emitter.as_ref() {
            emitter.emit(event);
        }
    }

    /// Reports this call's model-facing output byte counts, or does nothing
    /// when the context has no emitter.
    pub fn record_tool_output_stats(&self, stats: OutputCaptureStats) {
        if let Some(emitter) = self.coding_event_emitter.as_ref() {
            emitter.record_tool_output_stats(stats);
        }
    }

    /// Whether this call is running in the root of its session tree.
    ///
    /// False outside a session, because a tool that is root-only needs a
    /// session to be root of.
    #[must_use]
    pub fn is_root_session(&self) -> bool {
        self.session_scope
            .as_ref()
            .is_some_and(SessionScope::is_root)
    }
}

/// What a tool does when the model calls it.
///
/// The arguments arrive as the JSON the model produced, already validated
/// against the tool's schema when it has one. Returning `Ok` gives the model
/// the string as the call's output; returning [`ToolError`] gives it the
/// error's message and reports the kind on the call's completion event.
pub type ToolExecutor = Arc<
    dyn Fn(Value, ToolContext) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send>>
        + Send
        + Sync,
>;

/// Executes a tool that returns content parts and observer-only information.
pub type RichToolExecutor = Arc<
    dyn Fn(
            Value,
            ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send>>
        + Send
        + Sync,
>;

/// A tool registration that would make dispatch ambiguous.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ToolRegistrationError {
    /// The tool name cannot serve as an identity.
    #[error("tool `{name}` has an invalid identity")]
    InvalidIdentity {
        /// The invalid name.
        name: String,
    },
    /// Two registrations expose the same name to the model.
    #[error("tool name `{name}` was registered more than once")]
    DuplicateName {
        /// The duplicated visible name.
        name: String,
    },
    /// Two registrations claim the same stable identity.
    #[error("tool identity `{id}` was registered more than once")]
    DuplicateIdentity {
        /// The duplicated stable identity.
        id: String,
    },
    /// An explicit replacement names no registered tool.
    #[error("cannot replace unregistered tool identity `{id}`")]
    UnknownReplacement {
        /// The requested stable identity.
        id: String,
    },
}

/// A tool a session can call: what the model is told, and what runs.
///
/// Build one with [`function`](Self::function) or [`new`](Self::new), then say
/// where it came from and how far it reaches. An application tool is root-only
/// unless [`allow_in_subagents`](Self::allow_in_subagents) says otherwise, and
/// one that [`requires_human_input`](Self::requires_human_input) never reaches
/// a child however it is marked, because a child has nobody to ask.
#[derive(Clone)]
pub struct RegisteredTool {
    /// What the model is told about the tool. The registry may rename it on
    /// insert; see [`ToolRegistry::register`].
    pub(crate) definition: ToolDefinition,
    /// What runs when the model calls it.
    pub(crate) executor:   RichToolExecutor,
    /// Where the tool came from.
    pub(crate) source:     ToolSource,
    /// Whether a child session may be given this tool.
    inheritable:           bool,
    /// Whether the tool parks a prompt on a person's answer.
    human_input:           bool,
    scheduling:            ToolScheduling,
}

impl RegisteredTool {
    /// What the model is told about the tool.
    #[must_use]
    pub const fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    /// Where the tool came from.
    #[must_use]
    pub const fn source(&self) -> &ToolSource {
        &self.source
    }

    /// Pairs a definition with what runs, as an application tool.
    ///
    /// The tool is root-only until
    /// [`allow_in_subagents`](Self::allow_in_subagents) says otherwise.
    /// Name another origin with [`with_source`](Self::with_source).
    #[must_use]
    pub fn new(definition: ToolDefinition, executor: ToolExecutor) -> Self {
        Self::new_rich(
            definition,
            Arc::new(move |arguments, context| {
                let result = executor(arguments, context);
                Box::pin(async move { result.await.map(ToolOutput::from) })
            }),
        )
    }

    /// Pairs a definition with an executor returning rich content.
    #[must_use]
    pub fn new_rich(definition: ToolDefinition, executor: RichToolExecutor) -> Self {
        Self {
            definition,
            executor,
            source: ToolSource::Application,
            inheritable: false,
            human_input: false,
            scheduling: ToolScheduling::Concurrent,
        }
    }

    /// Defines an application tool with an asynchronous closure.
    ///
    /// Arguments have already been validated against `input_schema` when the
    /// closure runs. Returning an error sends its safe message back to the
    /// model and records its structured kind on the completion event.
    pub fn function<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(ToolContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, ToolError>> + Send + 'static,
    {
        Self::new(
            ToolDefinition::function(name, description, input_schema),
            Arc::new(move |arguments, context| Box::pin(execute(context, arguments))),
        )
    }

    /// Defines an application tool returning content parts, details, and
    /// artifacts.
    ///
    /// Only content parts are sent to the model. Details and artifact metadata
    /// travel through middleware and the durable completion event.
    pub fn rich_function<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(ToolContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, ToolError>> + Send + 'static,
    {
        Self::new_rich(
            ToolDefinition::function(name, description, input_schema),
            Arc::new(move |arguments, context| Box::pin(execute(context, arguments))),
        )
    }

    /// Names where the tool came from.
    ///
    /// A built-in registered as [`Native`](ToolSource::Native) is renamed into
    /// the session's vocabulary and categorized by the permission table.
    #[must_use]
    pub fn with_source(mut self, source: ToolSource) -> Self {
        self.source = source;
        self
    }

    /// Declares how the generic loop may schedule this tool within a round.
    ///
    /// Use `Sequential` when a call changes state that other calls may read or
    /// write. This does not synchronize separate agents sharing an environment.
    /// A tool requiring human input remains exclusive regardless of this
    /// setting.
    #[must_use]
    pub const fn with_scheduling(mut self, scheduling: ToolScheduling) -> Self {
        self.scheduling = scheduling;
        self
    }

    /// The effective scheduling rule, including the human-input guarantee.
    #[must_use]
    pub const fn scheduling(&self) -> ToolScheduling {
        if self.human_input {
            ToolScheduling::ExclusiveRound
        } else {
            self.scheduling
        }
    }

    /// Lets a child session be given this tool.
    ///
    /// Application tools are root-only by default: a child is given a task,
    /// not the application's integrations, unless the application says so per
    /// tool. Pebble's built-in tools are always inheritable. A tool that
    /// [`requires_human_input`](Self::requires_human_input) is withheld from
    /// children whatever this says.
    #[must_use]
    pub const fn allow_in_subagents(mut self) -> Self {
        self.inheritable = true;
        self
    }

    /// Declares that the tool parks a prompt until a person answers.
    ///
    /// A child session has nobody to ask, so such a tool never reaches one.
    #[must_use]
    pub const fn requires_human_input(mut self) -> Self {
        self.human_input = true;
        self
    }

    /// Whether a child session may be given this tool.
    ///
    /// Built-in tools always may; an application tool only when marked. A tool
    /// that needs a person is never given to a child.
    #[must_use]
    pub fn is_inheritable(&self) -> bool {
        if self.human_input {
            return false;
        }
        self.inheritable || matches!(self.source, ToolSource::Native | ToolSource::Skill)
    }

    /// Whether the tool parks a prompt on a person's answer.
    #[must_use]
    pub const fn needs_human_input(&self) -> bool {
        self.human_input
    }
}

/// One request's tool, borrowed from the request and the registry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AdvertisedTool<'a> {
    /// What the model was told about the tool.
    pub(crate) definition: &'a ToolDefinition,
    /// Where the tool came from.
    pub(crate) source:     &'a ToolSource,
}

/// One registered tool's advertised half.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ToolDefinitionWithSource {
    /// What the model is told about the tool.
    pub(crate) definition: ToolDefinition,
    /// Where the tool came from.
    pub(crate) source:     ToolSource,
}

impl ToolDefinitionWithSource {
    /// Projects this tool into the summary an observer of the session reads.
    ///
    /// The parameter schema is dropped: it is for the model, is often large,
    /// and belongs to no observer. `invoked` is `false`, because registration
    /// is not a call; a consumer reducing the event stream flips it.
    #[must_use]
    pub(crate) fn to_tool_summary(&self) -> ToolSummary {
        ToolSummary {
            name:        self.definition.name.clone(),
            description: self.definition.description.clone(),
            source:      self.source.clone(),
            category:    known_tool_category(&self.definition.name).unwrap_or(ToolCategory::Other),
            invoked:     false,
        }
    }
}

/// The tools one session exposes.
///
/// A session builder fills a registry and hands it to the session, which adds
/// one last tool while it initializes — the skill tool, whose skills are
/// discovered rather than configured — and only reads it afterwards.
/// Registration is therefore a setup activity: nothing is added or removed
/// while a prompt is in flight.
#[derive(Clone)]
pub(crate) struct ToolRegistry {
    tools:      HashMap<String, RegisteredTool>,
    identities: HashMap<pebble_agent::ToolId, String>,
    /// The naming scheme applied to built-in tools as they are registered.
    ///
    /// Held by the registry rather than applied as a pass over a finished set,
    /// so a tool registered late — a subagent tool, a skill — cannot miss it
    /// and leave the model with a mixed-vocabulary tool set.
    vocabulary: ToolVocabulary,
}

impl ToolRegistry {
    /// An empty registry speaking pebble's own tool names.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::with_vocabulary(ToolVocabulary::Canonical)
    }

    /// An empty registry that exposes built-in tools under `vocabulary`.
    #[must_use]
    pub(crate) fn with_vocabulary(vocabulary: ToolVocabulary) -> Self {
        Self {
            tools: HashMap::new(),
            identities: HashMap::new(),
            vocabulary,
        }
    }

    /// The naming scheme this registry applies.
    #[must_use]
    pub(crate) fn vocabulary(&self) -> ToolVocabulary {
        self.vocabulary
    }

    /// Adds a tool, renaming a built-in one into the registry's vocabulary.
    ///
    /// A tool is built-in when it is registered as [`ToolSource::Native`]
    /// under a canonical pebble name, or as [`ToolSource::Skill`] under the
    /// skill tool's own name. Only canonical names count, so an unrelated
    /// extension registered as `Read` keeps that name instead of being
    /// mistaken for pebble's file reader.
    ///
    /// Both the translated visible name and stable identity must be unique.
    pub(crate) fn register(
        &mut self,
        mut tool: RegisteredTool,
    ) -> StdResult<(), ToolRegistrationError> {
        let native = match &tool.source {
            ToolSource::Native => NativeTool::from_canonical_name(&tool.definition.name),
            ToolSource::Skill if tool.definition.name == NativeTool::UseSkill.canonical_name() => {
                Some(NativeTool::UseSkill)
            }
            // Matched exhaustively so a new kind of tool has to state whether
            // the vocabulary applies to it.
            ToolSource::Application | ToolSource::Skill | ToolSource::Mcp { .. } => None,
        };
        let identity = match native {
            Some(native) => native.canonical_name(),
            None => tool.definition.name.as_str(),
        };
        let id = pebble_agent::ToolId::try_new(identity).map_err(|_| {
            ToolRegistrationError::InvalidIdentity {
                name: identity.to_owned(),
            }
        })?;
        if let Some(native) = native {
            native
                .name(self.vocabulary)
                .clone_into(&mut tool.definition.name);
        }
        let name = tool.definition.name.clone();
        if self.tools.contains_key(&name) {
            return Err(ToolRegistrationError::DuplicateName { name });
        }
        if self.identities.contains_key(&id) {
            return Err(ToolRegistrationError::DuplicateIdentity {
                id: id.as_str().to_owned(),
            });
        }
        self.identities.insert(id, name.clone());
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Replaces an explicitly selected tool while preserving its name and
    /// identity.
    pub(crate) fn replace(
        &mut self,
        id: &str,
        mut tool: RegisteredTool,
    ) -> StdResult<(), ToolRegistrationError> {
        let identity = pebble_agent::ToolId::try_new(id).map_err(|_| {
            ToolRegistrationError::InvalidIdentity {
                name: id.to_owned(),
            }
        })?;
        let name = self
            .identities
            .get(&identity)
            .ok_or_else(|| ToolRegistrationError::UnknownReplacement { id: id.to_owned() })?;
        tool.definition.name.clone_from(name);
        self.tools.insert(name.clone(), tool);
        Ok(())
    }

    pub(crate) fn tools_with_ids(
        &self,
    ) -> impl Iterator<Item = (&pebble_agent::ToolId, &RegisteredTool)> {
        self.identities
            .iter()
            .map(|(id, name)| (id, &self.tools[name]))
    }

    /// The tool exposed under `name`.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    /// The tool with this stable identity, independent of visible vocabulary.
    #[must_use]
    pub(crate) fn get_by_id(&self, id: &pebble_agent::ToolId) -> Option<&RegisteredTool> {
        self.identities
            .get(id)
            .and_then(|name| self.tools.get(name))
    }

    /// A built-in tool by identity, whatever vocabulary it is exposed under.
    #[must_use]
    pub(crate) fn get_native(&self, tool: NativeTool) -> Option<&RegisteredTool> {
        let id = pebble_agent::ToolId::try_new(tool.canonical_name())
            .expect("native identities are valid");
        self.get_by_id(&id)
    }

    /// Every registered tool's definition, in no particular order.
    #[must_use]
    pub(crate) fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    /// Every registered tool's definition and origin, in no particular order.
    #[must_use]
    pub(crate) fn definitions_with_source(&self) -> Vec<ToolDefinitionWithSource> {
        self.tools
            .values()
            .map(|tool| ToolDefinitionWithSource {
                definition: tool.definition.clone(),
                source:     tool.source.clone(),
            })
            .collect()
    }

    /// The request's tool definitions paired with their registered origins.
    #[must_use]
    pub(crate) fn sources_for<'a>(
        &'a self,
        definitions: &'a [ToolDefinition],
    ) -> Vec<AdvertisedTool<'a>> {
        definitions
            .iter()
            .filter_map(|definition| {
                self.tools.get(&definition.name).map(|tool| AdvertisedTool {
                    definition,
                    source: &tool.source,
                })
            })
            .collect()
    }

    /// The names every registered tool is exposed under, in no particular
    /// order.
    #[must_use]
    pub(crate) fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// One required string argument, or the error the model is given instead.
///
/// Pebble's structural schema check accepts a call whose argument is missing or
/// wrongly typed only where the schema could not say otherwise, so a tool still
/// asks for what it needs rather than assuming.
///
/// # Errors
///
/// Returns a [`ToolError`] of kind
/// [`InvalidArguments`](crate::tools::ToolErrorKind::InvalidArguments) when
/// `key` is absent or is not a string.
pub(crate) fn required_str<'a>(arguments: &'a Value, key: &str) -> StdResult<&'a str, ToolError> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::invalid_arguments(format!("Missing required parameter: {key}")))
}

/// One optional counting argument, or the error the model is given instead.
///
/// An argument that is absent, negative, or not a whole number is `None`, so
/// a tool falls back to its own default rather than refusing the call. A
/// number too large for this machine is refused, because silently clamping a
/// line offset would answer a question the model did not ask.
///
/// # Errors
///
/// Returns a [`ToolError`] of kind
/// [`InvalidArguments`](crate::tools::ToolErrorKind::InvalidArguments) when
/// `key` holds a number this platform cannot hold.
pub(crate) fn optional_usize_arg(
    arguments: &Value,
    key: &str,
) -> StdResult<Option<usize>, ToolError> {
    optional_integer_arg::<i128>(arguments, key)
        .filter(|value| *value >= 0)
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                ToolError::invalid_arguments(format!("Parameter {key} is too large: {value}"))
            })
        })
        .transpose()
}

/// One optional integer argument as `T`, or `None`.
///
/// `None` when `key` is absent, is not a whole number, or names a whole
/// number `T` cannot hold. See [`whole_number`] for what counts as a whole
/// number.
pub(crate) fn optional_integer_arg<T: TryFrom<i128>>(arguments: &Value, key: &str) -> Option<T> {
    arguments.get(key).and_then(whole_number)
}

/// The integer a JSON number carries, as `T`, or `None`.
///
/// JSON has one number type, and some providers spell an integer as `2000.0`.
/// A number with no fractional part is the integer it names, however it was
/// spelled, so `2000`, `2000.0` and `-3.0` all read; `2.5`, text and anything
/// else are `None`. A whole number `T` cannot hold — a negative count for an
/// unsigned `T`, or a number past `T`'s range — is `None` too, so a caller
/// that wants to refuse one rather than fall back reads an `i128` and
/// narrows it itself.
pub(crate) fn whole_number<T: TryFrom<i128>>(value: &Value) -> Option<T> {
    let wide: i128 = if let Some(number) = value.as_i64() {
        number.into()
    } else if let Some(number) = value.as_u64() {
        number.into()
    } else {
        let number = value.as_f64()?;
        // Every whole float inside this range is an integer `i128` holds
        // exactly, and the range check is what keeps the cast from
        // saturating. NaN and infinity are outside every range.
        if number.fract() != 0.0 || !I128_RANGE_AS_F64.contains(&number) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a whole float inside the i128 range converts exactly"
        )]
        {
            number as i128
        }
    };
    T::try_from(wide).ok()
}

/// The floats that name an `i128`: from `i128::MIN`, which a float holds
/// exactly, up to but not including 2^127, the first magnitude past it.
const I128_RANGE_AS_F64: Range<f64> = -170_141_183_460_469_231_731_687_303_715_884_105_728.0
    ..170_141_183_460_469_231_731_687_303_715_884_105_728.0;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde_json::json;

    use super::*;
    use crate::event::{EventOptions, EventPump};
    use crate::human_input::{Answer, HumanInputError, Question};
    use crate::test_support::MockEnvironment;
    use crate::types::ToolErrorKind;

    fn make_tool(name: &str) -> RegisteredTool {
        RegisteredTool::new(
            ToolDefinition::function(name, format!("Tool {name}"), json!({"type": "object"})),
            Arc::new(|_args, _ctx| Box::pin(async { Ok("ok".to_owned()) })),
        )
        .with_source(ToolSource::Native)
    }

    fn context() -> ToolContext {
        ToolContext::new(Arc::new(MockEnvironment::default()))
    }

    #[tokio::test]
    async fn application_tool_function_pairs_definition_and_executor() {
        let tool = RegisteredTool::function(
            "inspect",
            "Inspect a value",
            json!({"type": "object"}),
            |_context, arguments| async move { Ok(format!("{}", arguments["name"])) },
        );

        let output = (tool.executor)(json!({"name": "parser"}), context())
            .await
            .map(|output| output.text())
            .expect("the tool succeeds");

        assert_eq!(tool.definition.name, "inspect");
        assert_eq!(tool.source, ToolSource::Application);
        assert_eq!(output, "\"parser\"");
    }

    #[test]
    fn register_and_get() {
        let mut registry = ToolRegistry::new();
        registry
            .register(make_tool("read_file"))
            .expect("tool registration is unique");

        let tool = registry.get("read_file").expect("registered");
        assert_eq!(tool.definition.name, "read_file");
    }

    #[test]
    fn kimi_registry_renames_canonical_native_tools_only() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::KimiCode);
        registry
            .register(make_tool("read_file"))
            .expect("tool registration is unique");
        assert!(matches!(
            registry.register(make_tool("Read")),
            Err(ToolRegistrationError::DuplicateName { .. })
        ));

        assert!(registry.get("Read").is_some());
        assert!(registry.get("read_file").is_none());
    }

    #[test]
    fn registry_does_not_reinterpret_mcp_names_as_native_tools() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::KimiCode);
        let mut tool = make_tool("read_file");
        tool.source = ToolSource::Mcp {
            server_name:   "files".to_owned(),
            original_name: "read_file".to_owned(),
        };

        registry
            .register(tool)
            .expect("tool registration is unique");

        assert!(registry.get("read_file").is_some());
        assert!(registry.get("Read").is_none());
    }

    #[test]
    fn the_skill_tool_is_renamed_but_other_skill_tools_are_not() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::KimiCode);
        let mut skill_tool = make_tool("use_skill");
        skill_tool.source = ToolSource::Skill;
        let mut other = make_tool("read_file");
        other.source = ToolSource::Skill;

        registry
            .register(skill_tool)
            .expect("tool registration is unique");
        registry
            .register(other)
            .expect("tool registration is unique");

        assert!(registry.get("Skill").is_some());
        assert!(registry.get("use_skill").is_none());
        assert!(registry.get("read_file").is_some());
    }

    #[test]
    fn get_missing_returns_none() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn get_native_resolves_the_exposed_vocabulary() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::Claude5);
        registry
            .register(make_tool("read_file"))
            .expect("tool registration is unique");

        let tool = registry
            .get_native(NativeTool::ReadFile)
            .expect("registered under the Claude 5 name");
        assert_eq!(tool.definition.name, "Read");
        assert!(registry.get_native(NativeTool::Shell).is_none());
    }

    #[test]
    fn explicit_replacement_preserves_identity() {
        let mut registry = ToolRegistry::new();
        registry
            .register(
                RegisteredTool::new(
                    ToolDefinition::function("tool_a", "version 1", json!({})),
                    Arc::new(|_args, _ctx| Box::pin(async { Ok("v1".to_owned()) })),
                )
                .with_source(ToolSource::Native),
            )
            .expect("tool registration is unique");
        registry
            .replace(
                "tool_a",
                RegisteredTool::new(
                    ToolDefinition::function("replacement_name", "version 2", json!({})),
                    Arc::new(|_args, _ctx| Box::pin(async { Ok("v2".to_owned()) })),
                )
                .with_source(ToolSource::Native),
            )
            .expect("tool registration is unique");

        let tool = registry.get("tool_a").expect("registered");
        assert_eq!(tool.definition.description, "version 2");
        assert_eq!(tool.definition.name, "tool_a");
        assert!(registry.get("replacement_name").is_none());
    }

    #[test]
    fn definitions_returns_all() {
        let mut registry = ToolRegistry::new();
        registry
            .register(make_tool("tool_a"))
            .expect("tool registration is unique");
        registry
            .register(make_tool("tool_b"))
            .expect("tool registration is unique");

        let definitions = registry.definitions();

        assert_eq!(definitions.len(), 2);
        let names: Vec<&str> = definitions
            .iter()
            .map(|definition| definition.name.as_str())
            .collect();
        assert!(names.contains(&"tool_a"));
        assert!(names.contains(&"tool_b"));
    }

    #[test]
    fn sources_for_keeps_only_the_definitions_in_the_request() {
        let mut registry = ToolRegistry::new();
        registry
            .register(make_tool("visible"))
            .expect("tool registration is unique");
        registry
            .register(make_tool("hidden"))
            .expect("tool registration is unique");
        let advertised = [ToolDefinition::function(
            "visible",
            "Filtered view",
            json!({}),
        )];

        let tools = registry.sources_for(&advertised);

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition.name, "visible");
        assert_eq!(tools[0].definition.description, "Filtered view");
        assert_eq!(*tools[0].source, ToolSource::Native);
    }

    #[test]
    fn names_returns_all() {
        let mut registry = ToolRegistry::new();
        registry
            .register(make_tool("tool_x"))
            .expect("tool registration is unique");
        registry
            .register(make_tool("tool_y"))
            .expect("tool registration is unique");

        let names = registry.names();

        assert_eq!(names.len(), 2);
        assert!(names.contains(&"tool_x".to_owned()));
        assert!(names.contains(&"tool_y".to_owned()));
    }

    #[tokio::test]
    async fn executor_can_be_called() {
        let mut registry = ToolRegistry::new();
        registry
            .register(make_tool("echo"))
            .expect("tool registration is unique");
        let tool = registry.get("echo").expect("registered");

        let result = (tool.executor)(json!({}), context())
            .await
            .map(|output| output.text());

        assert_eq!(result.expect("the tool succeeds"), "ok");
    }

    #[tokio::test]
    async fn an_executor_reports_a_typed_failure() {
        let mut registry = ToolRegistry::new();
        registry
            .register(
                RegisteredTool::new(
                    ToolDefinition::function("boom", "Fails", json!({})),
                    Arc::new(|_args, _ctx| {
                        Box::pin(async { Err(ToolError::invalid_arguments("path is required")) })
                    }),
                )
                .with_source(ToolSource::Native),
            )
            .expect("tool registration is unique");
        let tool = registry.get("boom").expect("registered");

        let error = (tool.executor)(json!({}), context())
            .await
            .map(|output| output.text())
            .expect_err("the tool fails");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "path is required");
    }

    #[test]
    fn default_creates_empty_registry() {
        let registry = ToolRegistry::default();
        assert!(registry.names().is_empty());
        assert!(registry.definitions().is_empty());
        assert_eq!(registry.vocabulary(), ToolVocabulary::Canonical);
    }

    fn tool_with_source(name: &str, source: ToolSource) -> ToolDefinitionWithSource {
        ToolDefinitionWithSource {
            definition: ToolDefinition::function(
                name,
                format!("{name} description"),
                json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                }),
            ),
            source,
        }
    }

    #[test]
    fn to_tool_summary_maps_known_native_categories_and_drops_parameters() {
        let cases = [
            ("apply_patch", ToolCategory::Write),
            ("grep", ToolCategory::Read),
            ("glob", ToolCategory::Read),
            ("spawn_agent", ToolCategory::Subagent),
            ("shell", ToolCategory::Shell),
            ("unknown_native", ToolCategory::Other),
        ];
        for (name, expected) in cases {
            let summary = tool_with_source(name, ToolSource::Native).to_tool_summary();
            assert_eq!(summary.name, name);
            assert_eq!(summary.description, format!("{name} description"));
            assert_eq!(summary.source, ToolSource::Native);
            assert_eq!(summary.category, expected);
            assert!(!summary.invoked);

            let json = serde_json::to_value(&summary).expect("serializes");
            assert!(
                json.as_object()
                    .expect("an object")
                    .get("parameters")
                    .is_none(),
                "tool summaries must not include parameter schemas"
            );
        }
    }

    #[test]
    fn to_tool_summary_carries_mcp_original_name_from_source() {
        let summary = tool_with_source("mcp__filesystem__read_file", ToolSource::Mcp {
            server_name:   "filesystem".to_owned(),
            original_name: "read_file".to_owned(),
        })
        .to_tool_summary();

        assert_eq!(summary.source, ToolSource::Mcp {
            server_name:   "filesystem".to_owned(),
            original_name: "read_file".to_owned(),
        });
        assert_eq!(summary.category, ToolCategory::Other);
    }

    #[tokio::test]
    async fn a_context_without_a_provider_resolves_no_environment() {
        assert!(
            context()
                .resolve_tool_env()
                .await
                .expect("no provider is not a failure")
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_static_provider_resolves_its_variables() {
        let provider = StaticEnvProvider(HashMap::from([("TOKEN".to_owned(), "abc".to_owned())]));
        let context = context().with_tool_env_provider(Arc::new(provider));

        let resolved = context
            .resolve_tool_env()
            .await
            .expect("the provider succeeds")
            .expect("a provider is installed");

        assert_eq!(resolved.get("TOKEN").map(String::as_str), Some("abc"));
    }

    #[tokio::test]
    async fn a_failing_env_provider_fails_the_call() {
        struct Failing;

        #[async_trait]
        impl ToolEnvProvider for Failing {
            async fn resolve(&self) -> Result<HashMap<String, String>, ToolError> {
                Err(ToolError::execution(
                    "Could not read the prompt's credentials",
                ))
            }
        }

        let context = context().with_tool_env_provider(Arc::new(Failing));

        let error = context
            .resolve_tool_env()
            .await
            .expect_err("the provider fails");

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert_eq!(error.message(), "Could not read the prompt's credentials");
    }

    #[test]
    fn a_context_outside_a_session_is_not_a_root_session() {
        assert!(!context().is_root_session());
        assert!(
            context()
                .with_session(
                    crate::SessionScope::root(crate::SessionId::new("ses_1"))
                        .child(crate::SessionId::new("ses_1"))
                )
                .is_root_session()
        );
        assert!(
            !context()
                .with_session(
                    crate::SessionScope::root(crate::SessionId::new("ses_1"))
                        .child(crate::SessionId::new("ses_2"))
                )
                .is_root_session()
        );
    }

    #[test]
    fn a_context_without_an_emitter_drops_events() {
        let context = context();
        // No emitter is installed: neither call reaches anything, and neither
        // panics.
        context.emit_coding_event(CodingEvent::SessionEnded);
        context.record_tool_output_stats(OutputCaptureStats::complete(8));
    }

    #[tokio::test]
    async fn a_bound_emitter_serves_as_the_context_emitter() {
        let (emitter, pump) = EventPump::new(EventOptions::default());
        let mut events = emitter.subscribe();
        let pump = tokio::spawn(pump.run());
        let bound = SessionBoundEmitter::new(emitter, "ses_1", Some("call_1".to_owned()));
        let context = context().with_coding_event_emitter(Arc::new(bound.clone()));

        context.emit_coding_event(CodingEvent::SessionEnded);
        context.record_tool_output_stats(OutputCaptureStats::complete(8));

        let event = events.recv().await.expect("an event is published");
        assert_eq!(event.session_id, "ses_1");
        assert_eq!(event.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(event.event, CodingEvent::SessionEnded);
        assert_eq!(
            bound.take_tool_output_stats(),
            Some(OutputCaptureStats::complete(8))
        );

        drop(context);
        drop(bound);
        pump.await
            .expect("the pump task joins")
            .expect("the pump finishes");
    }

    #[test]
    fn a_human_input_provider_rides_the_context() {
        struct Silent;

        #[async_trait]
        impl HumanInputProvider for Silent {
            async fn ask_questions(
                &self,
                _tool_call_id: &str,
                _questions: Vec<Question>,
                _cancel_token: CancellationToken,
            ) -> Result<Vec<Answer>, HumanInputError> {
                Ok(Vec::new())
            }
        }

        let context = context().with_human_input(Arc::new(Silent));
        assert!(context.human_input.is_some());
    }

    #[test]
    fn a_context_redacts_nothing_until_it_is_given_a_redactor() {
        struct MaskAll;

        impl Redactor for MaskAll {
            fn redact<'a>(&self, _text: &'a str) -> Cow<'a, str> {
                Cow::Borrowed("[REDACTED]")
            }
        }

        assert_eq!(context().redactor.redact("token abc"), "token abc");
        assert_eq!(
            context()
                .with_redactor(Arc::new(MaskAll))
                .redactor
                .redact("token abc"),
            "[REDACTED]"
        );
    }

    #[test]
    fn a_required_string_argument_is_asked_for_by_name() {
        let arguments = json!({"file_path": "/a.txt", "limit": 3});

        assert_eq!(
            required_str(&arguments, "file_path").expect("the argument is there"),
            "/a.txt"
        );
        let missing = required_str(&arguments, "old_string").expect_err("the argument is absent");
        assert_eq!(missing.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(missing.message(), "Missing required parameter: old_string");
        // A value of the wrong type is missing as far as the tool is
        // concerned: it cannot use it either way.
        assert_eq!(
            required_str(&arguments, "limit")
                .expect_err("the argument is not a string")
                .message(),
            "Missing required parameter: limit"
        );
    }

    #[test]
    fn an_optional_count_falls_back_rather_than_refusing_the_call() {
        let arguments = json!({
            "limit": 12,
            "spelled_as_a_float": 12.0,
            "negative": -1,
            "negative_float": -1.0,
            "fractional": 1.5,
            "text": "12",
        });

        assert_eq!(
            optional_usize_arg(&arguments, "limit").expect("a whole number"),
            Some(12)
        );
        assert_eq!(
            optional_usize_arg(&arguments, "spelled_as_a_float").expect("a whole number"),
            Some(12)
        );
        for key in ["absent", "negative", "negative_float", "fractional", "text"] {
            assert_eq!(
                optional_usize_arg(&arguments, key).expect("nothing to refuse"),
                None,
                "{key} is not a count, so the tool uses its own default"
            );
        }
    }

    #[test]
    fn an_optional_count_past_this_machine_is_refused() {
        let arguments = json!({"limit": 1e30});

        let error = optional_usize_arg(&arguments, "limit").expect_err("1e30 is past usize");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(
            error.message(),
            "Parameter limit is too large: 1000000000000000019884624838656"
        );
    }

    #[test]
    fn a_whole_number_reads_however_json_spelled_it() {
        assert_eq!(whole_number::<u64>(&json!(2000)), Some(2000));
        assert_eq!(whole_number::<u64>(&json!(2000.0)), Some(2000));
        assert_eq!(whole_number::<i64>(&json!(-3)), Some(-3));
        assert_eq!(whole_number::<i64>(&json!(-3.0)), Some(-3));
        assert_eq!(whole_number::<u64>(&json!(u64::MAX)), Some(u64::MAX));
        assert_eq!(
            whole_number::<i128>(&json!(1e30)),
            Some(1_000_000_000_000_000_019_884_624_838_656)
        );
    }

    #[test]
    fn a_whole_number_is_none_when_it_is_not_one_or_does_not_fit() {
        // Not a whole number.
        assert_eq!(whole_number::<i128>(&json!(2.5)), None);
        assert_eq!(whole_number::<i128>(&json!("2000")), None);
        assert_eq!(whole_number::<i128>(&json!(null)), None);
        assert_eq!(whole_number::<i128>(&json!(true)), None);
        // Negative into unsigned.
        assert_eq!(whole_number::<u64>(&json!(-1)), None);
        assert_eq!(whole_number::<u64>(&json!(-1.0)), None);
        assert_eq!(whole_number::<usize>(&json!(-1)), None);
        // Past the target type.
        assert_eq!(whole_number::<u8>(&json!(256)), None);
        assert_eq!(whole_number::<u8>(&json!(256.0)), None);
        assert_eq!(whole_number::<i64>(&json!(u64::MAX)), None);
        assert_eq!(whole_number::<u64>(&json!(1e30)), None);
        // Past i128 itself, either way.
        assert_eq!(whole_number::<i128>(&json!(1e40)), None);
        assert_eq!(whole_number::<i128>(&json!(-1e40)), None);
    }

    #[test]
    fn an_optional_integer_argument_is_read_by_key() {
        let arguments = json!({"timeout_ms": 30000.0, "offset": -2});

        assert_eq!(
            optional_integer_arg::<u64>(&arguments, "timeout_ms"),
            Some(30_000)
        );
        assert_eq!(optional_integer_arg::<i64>(&arguments, "offset"), Some(-2));
        assert_eq!(optional_integer_arg::<u64>(&arguments, "offset"), None);
        assert_eq!(optional_integer_arg::<u64>(&arguments, "absent"), None);
    }
}
