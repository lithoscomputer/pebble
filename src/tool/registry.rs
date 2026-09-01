//! What a session can call, and what a tool is handed when it runs.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::ToolDefinition;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::error::ToolError;
use super::native::{NativeTool, ToolVocabulary};
use super::permissions::known_tool_category;
use crate::config::{ToolAccessPolicy, ToolExposureMode};
use crate::environment::Environment;
use crate::event::{OutputCaptureStats, SessionBoundEmitter};
use crate::human_input::HumanInputProvider;
use crate::redact::{NoRedaction, Redactor};
use crate::types::{AgentEvent, ToolCategory, ToolSource, ToolSummary};

/// The narrow handle a running tool publishes events through.
///
/// A tool that changes session-visible state — the todo list, a subprocess it
/// ran — says so through this rather than by returning it, because the output
/// the model reads and the events an application observes are different
/// things.
///
/// Implementations must stamp emitted events with the session identity the
/// owning session is using, so a child session's events stay attributable.
pub trait AgentEventEmitter: Send + Sync {
    /// Publishes one event on the owning session's stream.
    fn emit(&self, event: AgentEvent);

    /// Reports how many bytes of model-facing output the running tool
    /// produced.
    ///
    /// A side channel, not an event: the execution layer drains it once the
    /// tool returns and folds it into the call's byte counters. An emitter
    /// with no tool-execution owner may ignore it, which is what the default
    /// does.
    fn record_tool_output_stats(&self, _stats: OutputCaptureStats) {}
}

impl AgentEventEmitter for SessionBoundEmitter {
    fn emit(&self, event: AgentEvent) {
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
/// application exercising one tool directly, a test — the identity members are
/// absent, and tools that need them say so rather than assuming a session.
///
/// New members appear here as tools gain capabilities, so build a context with
/// [`new`](Self::new) and the `with_*` methods rather than a struct literal.
#[non_exhaustive]
pub struct ToolContext {
    /// Where the tool's work lands.
    pub env:                 Arc<dyn Environment>,
    /// Fires when this call should stop. Composed from the session's terminal
    /// cancellation and the current round's interrupt, so a tool that watches
    /// it observes both.
    ///
    /// Watching it is the tool's own responsibility, and the session waits for
    /// the answer either way: a cancelled call is never dropped, because a call
    /// with no result is a conversation the provider will refuse. A tool that
    /// ignores this token therefore holds its round — and the prompt ending it
    /// — open until it returns, so long work must watch it and answer.
    pub cancel:              CancellationToken,
    /// Extra environment variables for a command this call runs.
    pub tool_env_provider:   Option<Arc<dyn ToolEnvProvider>>,
    /// The session that called the tool.
    pub session_id:          Option<String>,
    /// The root of the session tree this call belongs to. Equal to
    /// [`session_id`](Self::session_id) in a root session; a child inherits
    /// its parent's root.
    pub root_session_id:     Option<String>,
    /// The model-native identifier of this call.
    pub tool_call_id:        Option<String>,
    /// Where the tool publishes events.
    pub agent_event_emitter: Option<Arc<dyn AgentEventEmitter>>,
    /// Where the tool asks the person a question. Absent in child sessions and
    /// wherever the application installed no provider.
    pub human_input:         Option<Arc<dyn HumanInputProvider>>,
    /// What strips secrets out of text the tool publishes.
    ///
    /// Only output leaving the session through an event goes through it — the
    /// process tail a shell tool publishes — never what the model is shown,
    /// which is the same text the model would have read from the terminal.
    /// [`NoRedaction`](crate::NoRedaction) unless the application installed
    /// one.
    pub redactor:            Arc<dyn Redactor>,
}

impl ToolContext {
    /// A context that has an environment and nothing else.
    #[must_use]
    pub fn new(env: Arc<dyn Environment>) -> Self {
        Self {
            env,
            cancel: CancellationToken::new(),
            tool_env_provider: None,
            session_id: None,
            root_session_id: None,
            tool_call_id: None,
            agent_event_emitter: None,
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
    pub fn with_session(
        mut self,
        session_id: impl Into<String>,
        root_session_id: impl Into<String>,
    ) -> Self {
        self.session_id = Some(session_id.into());
        self.root_session_id = Some(root_session_id.into());
        self
    }

    /// Sets the model-native identifier of this call.
    #[must_use]
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
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
    pub fn with_event_emitter(mut self, emitter: Arc<dyn AgentEventEmitter>) -> Self {
        self.agent_event_emitter = Some(emitter);
        self
    }

    /// Sets where the tool's questions go.
    #[must_use]
    pub fn with_human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Sets what strips secrets out of text the tool publishes.
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
    pub fn emit_agent_event(&self, event: AgentEvent) {
        if let Some(emitter) = self.agent_event_emitter.as_ref() {
            emitter.emit(event);
        }
    }

    /// Reports this call's model-facing output byte counts, or does nothing
    /// when the context has no emitter.
    pub fn record_tool_output_stats(&self, stats: OutputCaptureStats) {
        if let Some(emitter) = self.agent_event_emitter.as_ref() {
            emitter.record_tool_output_stats(stats);
        }
    }

    /// Whether this call is running in the root of its session tree.
    ///
    /// False outside a session, because a tool that is root-only needs a
    /// session to be root of.
    #[must_use]
    pub fn is_root_session(&self) -> bool {
        match (&self.session_id, &self.root_session_id) {
            (Some(session_id), Some(root_session_id)) => session_id == root_session_id,
            _ => false,
        }
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

/// A tool a session can call: what the model is told, and what runs.
#[derive(Clone)]
pub struct RegisteredTool {
    /// What the model is told about the tool. The registry may rename it on
    /// insert; see [`ToolRegistry::register`].
    pub definition: ToolDefinition,
    /// What runs when the model calls it.
    pub executor:   ToolExecutor,
    /// Where the tool came from.
    pub source:     ToolSource,
}

impl RegisteredTool {
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
        Self {
            definition: ToolDefinition::function(name, description, input_schema),
            executor:   Arc::new(move |arguments, context| Box::pin(execute(context, arguments))),
            source:     ToolSource::Application,
        }
    }
}

/// One registered tool's advertised half.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinitionWithSource {
    /// What the model is told about the tool.
    pub definition: ToolDefinition,
    /// Where the tool came from.
    pub source:     ToolSource,
}

impl ToolDefinitionWithSource {
    /// Projects this tool into the summary an observer of the session reads.
    ///
    /// The parameter schema is dropped: it is for the model, is often large,
    /// and belongs to no observer. `invoked` is `false`, because registration
    /// is not a call; a consumer reducing the event stream flips it.
    #[must_use]
    pub fn to_tool_summary(&self) -> ToolSummary {
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
pub struct ToolRegistry {
    tools:      HashMap<String, RegisteredTool>,
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
    pub fn new() -> Self {
        Self::with_vocabulary(ToolVocabulary::Canonical)
    }

    /// An empty registry that exposes built-in tools under `vocabulary`.
    #[must_use]
    pub fn with_vocabulary(vocabulary: ToolVocabulary) -> Self {
        Self {
            tools: HashMap::new(),
            vocabulary,
        }
    }

    /// The naming scheme this registry applies.
    #[must_use]
    pub fn vocabulary(&self) -> ToolVocabulary {
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
    /// The tool is keyed by the name it ends up exposed under, and the last
    /// registration of a name wins.
    pub fn register(&mut self, mut tool: RegisteredTool) {
        let native = match &tool.source {
            ToolSource::Native => NativeTool::from_canonical_name(&tool.definition.name),
            ToolSource::Skill if tool.definition.name == NativeTool::UseSkill.canonical_name() => {
                Some(NativeTool::UseSkill)
            }
            // Matched exhaustively so a new kind of tool has to state whether
            // the vocabulary applies to it.
            ToolSource::Application | ToolSource::Skill | ToolSource::Mcp { .. } => None,
        };
        if let Some(native) = native {
            native
                .name(self.vocabulary)
                .clone_into(&mut tool.definition.name);
        }
        self.tools.insert(tool.definition.name.clone(), tool);
    }

    /// The tool exposed under `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    /// A built-in tool by identity, whatever vocabulary it is exposed under.
    #[must_use]
    pub fn get_native(&self, tool: NativeTool) -> Option<&RegisteredTool> {
        self.tools.get(tool.name(self.vocabulary))
    }

    /// Every registered tool's definition, in no particular order.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    /// Every registered tool's definition and origin, in no particular order.
    #[must_use]
    pub fn definitions_with_source(&self) -> Vec<ToolDefinitionWithSource> {
        // With no policy the exposure mode is never consulted.
        self.definitions_with_source_for_policy(None, ToolExposureMode::AutoApprovedOnly)
    }

    /// The definitions `policy` allows a session to advertise.
    ///
    /// No policy exposes everything.
    #[must_use]
    pub fn definitions_for_policy(
        &self,
        policy: Option<&dyn ToolAccessPolicy>,
        exposure_mode: ToolExposureMode,
    ) -> Vec<ToolDefinition> {
        self.definitions_with_source_for_policy(policy, exposure_mode)
            .into_iter()
            .map(|tool| tool.definition)
            .collect()
    }

    /// The definitions and origins `policy` allows a session to advertise.
    ///
    /// The policy is asked about the name the model would call — the exposed
    /// name, after any vocabulary rename — so a policy written against pebble's
    /// canonical names resolves them itself. See
    /// [`canonical_tool_name`](super::permissions::canonical_tool_name).
    #[must_use]
    pub fn definitions_with_source_for_policy(
        &self,
        policy: Option<&dyn ToolAccessPolicy>,
        exposure_mode: ToolExposureMode,
    ) -> Vec<ToolDefinitionWithSource> {
        self.tools
            .values()
            .filter(|tool| {
                policy.is_none_or(|policy| {
                    policy
                        .access_for_tool(&tool.definition.name)
                        .is_exposed(exposure_mode)
                })
            })
            .map(|tool| ToolDefinitionWithSource {
                definition: tool.definition.clone(),
                source:     tool.source.clone(),
            })
            .collect()
    }

    /// The names every registered tool is exposed under, in no particular
    /// order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
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
/// [`InvalidArguments`](crate::ToolErrorKind::InvalidArguments) when `key` is
/// absent or is not a string.
pub(crate) fn required_str<'a>(arguments: &'a Value, key: &str) -> StdResult<&'a str, ToolError> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::invalid_arguments(format!("Missing required parameter: {key}")))
}

/// One optional counting argument, or the error the model is given instead.
///
/// An argument that is absent, negative, or not a number at all is `None`, so
/// a tool falls back to its own default rather than refusing the call. A
/// number too large for this machine is refused, because silently clamping a
/// line offset would answer a question the model did not ask.
///
/// # Errors
///
/// Returns a [`ToolError`] of kind
/// [`InvalidArguments`](crate::ToolErrorKind::InvalidArguments) when `key`
/// holds a number this platform cannot hold.
pub(crate) fn optional_usize_arg(
    arguments: &Value,
    key: &str,
) -> StdResult<Option<usize>, ToolError> {
    arguments
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                ToolError::invalid_arguments(format!("Parameter {key} is too large: {value}"))
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde_json::json;

    use super::*;
    use crate::config::ToolAccess;
    use crate::event::{EventOptions, EventPump};
    use crate::human_input::{Answer, HumanInputError, Question};
    use crate::test_support::MockEnvironment;
    use crate::types::ToolErrorKind;

    struct NamedPolicy {
        decisions: HashMap<String, ToolAccess>,
    }

    impl NamedPolicy {
        fn new(decisions: impl IntoIterator<Item = (&'static str, ToolAccess)>) -> Self {
            Self {
                decisions: decisions
                    .into_iter()
                    .map(|(name, access)| (name.to_owned(), access))
                    .collect(),
            }
        }
    }

    impl ToolAccessPolicy for NamedPolicy {
        fn access_for_tool(&self, tool_name: &str) -> ToolAccess {
            self.decisions
                .get(tool_name)
                .copied()
                .unwrap_or(ToolAccess::Denied)
        }
    }

    fn make_tool(name: &str) -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition::function(
                name,
                format!("Tool {name}"),
                json!({"type": "object"}),
            ),
            executor:   Arc::new(|_args, _ctx| Box::pin(async { Ok("ok".to_owned()) })),
            source:     ToolSource::Native,
        }
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
            .expect("the tool succeeds");

        assert_eq!(tool.definition.name, "inspect");
        assert_eq!(tool.source, ToolSource::Application);
        assert_eq!(output, "\"parser\"");
    }

    #[test]
    fn register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file"));

        let tool = registry.get("read_file").expect("registered");
        assert_eq!(tool.definition.name, "read_file");
    }

    #[test]
    fn kimi_registry_renames_canonical_native_tools_only() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::KimiCode);
        registry.register(make_tool("read_file"));
        registry.register(make_tool("Read"));

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

        registry.register(tool);

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

        registry.register(skill_tool);
        registry.register(other);

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
        registry.register(make_tool("read_file"));

        let tool = registry
            .get_native(NativeTool::ReadFile)
            .expect("registered under the Claude 5 name");
        assert_eq!(tool.definition.name, "Read");
        assert!(registry.get_native(NativeTool::Shell).is_none());
    }

    #[test]
    fn name_collision_overrides() {
        let mut registry = ToolRegistry::new();
        registry.register(RegisteredTool {
            definition: ToolDefinition::function("tool_a", "version 1", json!({})),
            executor:   Arc::new(|_args, _ctx| Box::pin(async { Ok("v1".to_owned()) })),
            source:     ToolSource::Native,
        });
        registry.register(RegisteredTool {
            definition: ToolDefinition::function("tool_a", "version 2", json!({})),
            executor:   Arc::new(|_args, _ctx| Box::pin(async { Ok("v2".to_owned()) })),
            source:     ToolSource::Native,
        });

        let tool = registry.get("tool_a").expect("registered");
        assert_eq!(tool.definition.description, "version 2");
    }

    #[test]
    fn definitions_returns_all() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("tool_a"));
        registry.register(make_tool("tool_b"));

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
    fn definitions_with_no_policy_returns_all_registered_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("allowed"));
        registry.register(make_tool("denied"));

        let definitions = registry.definitions_for_policy(None, ToolExposureMode::AutoApprovedOnly);

        let names: Vec<&str> = definitions
            .iter()
            .map(|definition| definition.name.as_str())
            .collect();
        assert_eq!(definitions.len(), 2);
        assert!(names.contains(&"allowed"));
        assert!(names.contains(&"denied"));
    }

    #[test]
    fn definitions_for_policy_omits_denied_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file"));
        registry.register(make_tool("write_file"));
        let policy = NamedPolicy::new([
            ("read_file", ToolAccess::Allowed),
            ("write_file", ToolAccess::Denied),
        ]);

        let definitions = registry
            .definitions_for_policy(Some(&policy), ToolExposureMode::IncludeRequiresApproval);

        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "read_file");
    }

    #[test]
    fn definitions_for_policy_exposes_approval_tools_only_when_enabled() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file"));
        registry.register(make_tool("shell"));
        let policy = NamedPolicy::new([
            ("read_file", ToolAccess::Allowed),
            ("shell", ToolAccess::RequiresApproval),
        ]);

        let auto_only =
            registry.definitions_for_policy(Some(&policy), ToolExposureMode::AutoApprovedOnly);
        let with_approval = registry
            .definitions_for_policy(Some(&policy), ToolExposureMode::IncludeRequiresApproval);

        assert_eq!(
            auto_only
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["read_file"]
        );
        let with_approval_names: Vec<&str> = with_approval
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        assert_eq!(with_approval_names.len(), 2);
        assert!(with_approval_names.contains(&"read_file"));
        assert!(with_approval_names.contains(&"shell"));
    }

    #[test]
    fn policy_sees_the_exposed_name_after_a_vocabulary_rename() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::KimiCode);
        registry.register(make_tool("read_file"));
        let policy = NamedPolicy::new([("Read", ToolAccess::Allowed)]);

        let definitions =
            registry.definitions_for_policy(Some(&policy), ToolExposureMode::AutoApprovedOnly);

        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "Read");
    }

    #[test]
    fn names_returns_all() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("tool_x"));
        registry.register(make_tool("tool_y"));

        let names = registry.names();

        assert_eq!(names.len(), 2);
        assert!(names.contains(&"tool_x".to_owned()));
        assert!(names.contains(&"tool_y".to_owned()));
    }

    #[tokio::test]
    async fn executor_can_be_called() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("echo"));
        let tool = registry.get("echo").expect("registered");

        let result = (tool.executor)(json!({}), context()).await;

        assert_eq!(result.expect("the tool succeeds"), "ok");
    }

    #[tokio::test]
    async fn an_executor_reports_a_typed_failure() {
        let mut registry = ToolRegistry::new();
        registry.register(RegisteredTool {
            definition: ToolDefinition::function("boom", "Fails", json!({})),
            executor:   Arc::new(|_args, _ctx| {
                Box::pin(async { Err(ToolError::invalid_arguments("path is required")) })
            }),
            source:     ToolSource::Native,
        });
        let tool = registry.get("boom").expect("registered");

        let error = (tool.executor)(json!({}), context())
            .await
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
        assert!(context().with_session("ses_1", "ses_1").is_root_session());
        assert!(!context().with_session("ses_2", "ses_1").is_root_session());
    }

    #[test]
    fn a_context_without_an_emitter_drops_events() {
        let context = context();
        // No emitter is installed: neither call reaches anything, and neither
        // panics.
        context.emit_agent_event(AgentEvent::SessionEnded);
        context.record_tool_output_stats(OutputCaptureStats::complete(8));
    }

    #[tokio::test]
    async fn a_bound_emitter_serves_as_the_context_emitter() {
        let (emitter, pump) = EventPump::new(EventOptions::default());
        let mut events = emitter.subscribe();
        let pump = tokio::spawn(pump.run());
        let bound = SessionBoundEmitter::new(emitter, "ses_1", Some("call_1".to_owned()));
        let context = context().with_event_emitter(Arc::new(bound.clone()));

        context.emit_agent_event(AgentEvent::SessionEnded);
        context.record_tool_output_stats(OutputCaptureStats::complete(8));

        let event = events.recv().await.expect("an event is published");
        assert_eq!(event.session_id, "ses_1");
        assert_eq!(event.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(event.event, AgentEvent::SessionEnded);
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
            "negative": -1,
            "fractional": 1.5,
            "text": "12",
        });

        assert_eq!(
            optional_usize_arg(&arguments, "limit").expect("a whole number"),
            Some(12)
        );
        for key in ["absent", "negative", "fractional", "text"] {
            assert_eq!(
                optional_usize_arg(&arguments, key).expect("nothing to refuse"),
                None,
                "{key} is not a count, so the tool uses its own default"
            );
        }
    }
}
