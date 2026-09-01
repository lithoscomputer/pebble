//! What one model family's harness expects of a session.
//!
//! A model trained inside a coding harness expects that harness back: the tools
//! it was shown, the names it was shown them under, and a system prompt written
//! the way it was trained to read one. An [`AgentProfile`] is that expectation,
//! and pebble picks one from the catalog metadata of the model a session
//! resolved to.
//!
//! A profile *contributes*; it does not own. The session builder holds the
//! registry and merges what every contributor hands it — the profile's own
//! tools, the subagent tools, whatever the application registered — and then
//! freezes it. So a profile returns tools rather than mutating a registry
//! something else also holds a handle on.
//!
//! Prompt assembly reads [`EnvContext`], not the environment: the session
//! gathers what the prompt needs once, and a profile turns it into text without
//! running anything.

use std::fmt;
use std::sync::Arc;

use lithos_llm::catalog::CatalogModel;

use crate::environment::Environment;
use crate::profiles::{
    AnthropicProfile, Claude5Profile, GeminiProfile, Gpt56Profile, KimiProfile, OpenAiProfile,
    ProfileDeps,
};
use crate::skills::Skill;
use crate::subagent::{SubagentSupervisor, subagent_tools};
use crate::tool::{RegisteredTool, ToolRegistry, ToolVocabulary};
use crate::types::AgentProfileKind;

/// The context window pebble assumes for a model the catalog says nothing
/// about.
pub const DEFAULT_CONTEXT_WINDOW_TOKENS: usize = 200_000;

/// What a system prompt says about where the session is working.
///
/// The session gathers this once, before the first turn: the first three
/// members come from the environment, the git members from commands run in it,
/// and the rest from the resolved model. A profile reads it and writes text.
///
/// Every member has a default, so build one with `..Default::default()` and
/// fill in what is known.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnvContext {
    /// Where the session's tools act.
    pub working_directory:  String,
    /// The operating system family, as
    /// [`Environment::platform`](crate::Environment::platform) names it:
    /// `darwin`, `linux`, `windows`, or `unknown`.
    pub platform:           String,
    /// The operating system version, as the host reports it.
    pub os_version:         String,
    /// Whether the working directory is inside a git repository.
    pub is_git_repo:        bool,
    /// The checked-out branch, when there is one.
    pub git_branch:         Option<String>,
    /// The short-format working-tree status, when it is not empty.
    pub git_status_short:   Option<String>,
    /// The most recent commits, one per line, when there are any.
    pub git_recent_commits: Option<String>,
    /// Today's date where the session is running, as `YYYY-MM-DD`.
    pub current_date:       String,
    /// The catalog identifier of the model answering.
    pub model:              String,
    /// How the model's training data is dated, when the catalog says.
    pub knowledge_cutoff:   String,
}

impl EnvContext {
    /// A context carrying what the environment reports about itself.
    ///
    /// The git members and the model members are left at their defaults,
    /// because neither is the environment's to answer.
    #[must_use]
    pub fn from_environment(env: &dyn Environment) -> Self {
        Self {
            working_directory: env.working_directory().to_owned(),
            platform: env.platform().to_owned(),
            os_version: env.os_version(),
            ..Self::default()
        }
    }
}

/// What a session's subagent support hands a profile.
///
/// Subagent tools differ by profile — pebble's own set spawns and waits on
/// agents, while the Claude 5 harness expects a background-agent family
/// instead — so a profile builds them rather than a builder choosing for it.
///
/// It carries the session's place in the tree, and — when the application
/// configured a [`SessionFactory`](crate::SessionFactory) — the supervisor the
/// tools drive. The supervisor is a shared handle, so this type deliberately
/// derives neither `Copy` nor `Eq`: growing it must not have to remove a derive
/// that callers depend on.
///
/// A profile outside pebble reads [`depth`](Self::depth) and contributes
/// whatever tools it likes; pebble's own four are reached through
/// [`AgentProfile::subagent_tools`]'s default implementation, which is what a
/// profile with no subagent story of its own inherits.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct SubagentSupport {
    /// How deep in the session tree the session being built sits, counting the
    /// root as zero.
    pub depth:             usize,
    /// The supervisor this session's children run under, when the application
    /// configured subagents at all.
    pub(crate) supervisor: Option<SubagentSupervisor>,
}

impl fmt::Debug for SubagentSupport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubagentSupport")
            .field("depth", &self.depth)
            .field("supervisor", &self.supervisor)
            .finish_non_exhaustive()
    }
}

impl SubagentSupport {
    /// The support a session with a configured factory hands its profile.
    pub(crate) const fn new(depth: usize, supervisor: Option<SubagentSupervisor>) -> Self {
        Self { depth, supervisor }
    }

    /// Whether this session may spawn children at all.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.supervisor.is_some()
    }

    /// Pebble's own four subagent tools: spawn, send input, wait, close.
    ///
    /// Empty when the application configured no session factory, which is how
    /// omitting one disables subagents without any profile having to check.
    #[must_use]
    pub(crate) fn pebble_tools(&self) -> Vec<RegisteredTool> {
        self.supervisor
            .as_ref()
            .map(subagent_tools)
            .unwrap_or_default()
    }
}

/// What the machinery around a session needs to know about its model.
///
/// Fabro's profiles answered these from their own catalog; pebble reads them
/// from the lithos-llm catalog the session resolved its model through, so a
/// profile carries no model facts at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelFacts {
    /// How many tokens the model's context window holds.
    pub context_window_tokens: usize,
    /// The most tokens the model may produce in one response, when the catalog
    /// says.
    pub max_output_tokens:     Option<u64>,
    /// Whether the model spends output tokens on reasoning without being asked
    /// to.
    ///
    /// A provider's output limit covers reasoning and visible text together, so
    /// a budget sized for the text alone can be spent entirely on thinking and
    /// return an empty response. Compaction adds headroom where this is set.
    pub reasons_by_default:    bool,
}

impl ModelFacts {
    /// The facts pebble assumes for a model nothing is known about.
    ///
    /// The same as [`ModelFacts::default`], and the starting point for building
    /// a set by hand: name what is known with the `with_*` methods and leave
    /// the rest.
    ///
    /// ```
    /// # use pebble::ModelFacts;
    /// let facts = ModelFacts::new()
    ///     .with_context_window_tokens(128_000)
    ///     .with_max_output_tokens(Some(8_192));
    /// # let _ = facts;
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The facts the catalog records about one model.
    ///
    /// A model the catalog describes without limits — a passthrough entry, say
    /// — falls back to [`DEFAULT_CONTEXT_WINDOW_TOKENS`] and no output limit,
    /// which is what fabro assumed for a model it could not look up.
    ///
    /// `reasons_by_default` is read from the catalog the way fabro read its
    /// own: a model that takes a named reasoning effort level reasons unless it
    /// is told not to, and one that only takes a thinking budget does not. A
    /// catalog row that knows better says so with
    /// `metadata.pebble.reasoning_by_default`, which
    /// [`SessionBuilder`](crate::SessionBuilder) applies on top of this.
    #[must_use]
    pub fn from_catalog_model(model: &CatalogModel) -> Self {
        let capabilities = model.capabilities();
        let reasons_by_default = capabilities.reasoning && capabilities.reasoning_effort_levels;
        model.limits().map_or(
            Self {
                reasons_by_default,
                ..Self::default()
            },
            |limits| Self {
                context_window_tokens: usize::try_from(limits.context_tokens).unwrap_or(usize::MAX),
                max_output_tokens: Some(limits.max_output_tokens),
                reasons_by_default,
            },
        )
    }

    /// The same facts, with the context window set.
    #[must_use]
    pub fn with_context_window_tokens(mut self, tokens: usize) -> Self {
        self.context_window_tokens = tokens;
        self
    }

    /// The same facts, with the output limit set.
    #[must_use]
    pub fn with_max_output_tokens(mut self, tokens: Option<u64>) -> Self {
        self.max_output_tokens = tokens;
        self
    }

    /// The same facts, saying whether the model reasons without being asked to.
    #[must_use]
    pub fn with_reasons_by_default(mut self, reasons_by_default: bool) -> Self {
        self.reasons_by_default = reasons_by_default;
        self
    }
}

impl Default for ModelFacts {
    fn default() -> Self {
        Self {
            context_window_tokens: DEFAULT_CONTEXT_WINDOW_TOKENS,
            max_output_tokens:     None,
            reasons_by_default:    false,
        }
    }
}

/// One model family's harness: its tools, their names, and its prompt.
///
/// Pebble ships a profile for each family it supports and selects one from the
/// catalog metadata of the model a session resolved to. A session cannot be
/// given a profile written outside the crate: the trait is public because it
/// describes what a harness is and is named by pebble's own types, not because
/// it is an installation point. Opening one is additive, so it stays a later
/// decision rather than a promise made now.
///
/// Everything here is asked once, while a session is being built. A profile is
/// then shared and read-only, so implementations must be cheap to call and must
/// answer the same way every time.
pub trait AgentProfile: Send + Sync {
    /// Which harness this profile implements.
    fn profile_kind(&self) -> AgentProfileKind;

    /// The names this profile's model expects its built-in tools under.
    ///
    /// The session's registry is created with this and renames built-in tools
    /// as they arrive, so a tool registered late cannot end up with the wrong
    /// spelling.
    fn tool_vocabulary(&self) -> ToolVocabulary;

    /// The tools this profile gives the session.
    ///
    /// Returned rather than registered: the session builder owns the registry
    /// and merges these with whatever else the session is given.
    fn base_tools(&self) -> Vec<RegisteredTool>;

    /// The system prompt this profile's model starts a session with.
    ///
    /// `registry` is the session's own, filled: what this session actually
    /// holds, which is not always what this profile contributed. A profile is
    /// shared — a child session runs its parent's — so a prompt section about a
    /// tool a child does not inherit has to be conditioned on the registry
    /// rather than on anything the profile was built with. Asking a person a
    /// question is the one such tool today: a child has nobody to ask.
    ///
    /// `memory` holds the files the application asked the session to load,
    /// already read; `user_instructions` is the application's own addition; and
    /// `skills` are the skills discovered for this session, which a profile
    /// normally summarizes rather than expands.
    fn build_system_prompt(
        &self,
        registry: &ToolRegistry,
        env_context: &EnvContext,
        memory: &[String],
        user_instructions: Option<&str>,
        skills: &[Skill],
    ) -> String;

    /// The subagent tools this profile gives the session.
    ///
    /// The default is pebble's own four — spawn, send input, wait, close —
    /// which is nothing at all when the application configured no session
    /// factory. A profile whose model family expects a different family of
    /// subagent tools overrides this and builds those instead.
    fn subagent_tools(&self, subagents: &SubagentSupport) -> Vec<RegisteredTool> {
        subagents.pebble_tools()
    }
}

/// The profile pebble ships for `kind`.
///
/// This is the one place a session turns a catalog profile identifier into a
/// profile. It is crate-internal on purpose: an application selects a profile
/// by choosing a model, never by naming one, so `deps` could grow again — the
/// harnesses read what the session is, and a later one may read more — without
/// costing anything outside the crate.
///
/// Every one of the six identifiers pebble knows is answered here; a catalog
/// row naming something else is refused earlier, when the identifier is parsed.
pub(crate) fn builtin_profile(kind: AgentProfileKind, deps: &ProfileDeps) -> Arc<dyn AgentProfile> {
    match kind {
        AgentProfileKind::Anthropic => Arc::new(AnthropicProfile::new(deps)),
        AgentProfileKind::Claude5 => Arc::new(Claude5Profile::new(deps)),
        AgentProfileKind::Gemini => Arc::new(GeminiProfile::new(deps)),
        AgentProfileKind::Gpt56 => Arc::new(Gpt56Profile::new(deps)),
        AgentProfileKind::Kimi => Arc::new(KimiProfile::new(deps)),
        AgentProfileKind::OpenAi => Arc::new(OpenAiProfile::new(deps)),
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use lithos_llm::catalog::Catalog;
    use lithos_llm::types::ToolDefinition;
    use serde_json::json;

    use super::*;
    use crate::test_support::MockEnvironment;
    use crate::types::ToolSource;

    struct TestProfile {
        tools: Vec<RegisteredTool>,
    }

    impl TestProfile {
        fn new() -> Self {
            Self { tools: Vec::new() }
        }

        fn with_tools(tools: Vec<RegisteredTool>) -> Self {
            Self { tools }
        }
    }

    impl AgentProfile for TestProfile {
        fn profile_kind(&self) -> AgentProfileKind {
            AgentProfileKind::Anthropic
        }

        fn tool_vocabulary(&self) -> ToolVocabulary {
            ToolVocabulary::Canonical
        }

        fn base_tools(&self) -> Vec<RegisteredTool> {
            self.tools.clone()
        }

        fn build_system_prompt(
            &self,
            _registry: &ToolRegistry,
            env_context: &EnvContext,
            memory: &[String],
            user_instructions: Option<&str>,
            skills: &[Skill],
        ) -> String {
            let mut prompt = format!(
                "You are a test assistant.\n\nWorking directory: {}\nPlatform: {}\n",
                env_context.working_directory, env_context.platform
            );
            for document in memory {
                prompt.push_str(document);
                prompt.push('\n');
            }
            for skill in skills {
                let _ = writeln!(prompt, "/{}: {}", skill.name, skill.description);
            }
            if let Some(instructions) = user_instructions {
                prompt.push_str("\n# User Instructions\n");
                prompt.push_str(instructions);
            }
            prompt
        }
    }

    fn tool(name: &str) -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition::function(name, format!("Tool {name}"), json!({})),
            executor:   Arc::new(|_arguments, _context| Box::pin(async { Ok("ok".to_owned()) })),
            source:     ToolSource::Native,
        }
    }

    #[test]
    fn a_profile_names_its_harness_and_its_vocabulary() {
        let profile = TestProfile::new();

        assert_eq!(profile.profile_kind(), AgentProfileKind::Anthropic);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::Canonical);
    }

    #[test]
    fn a_profile_contributes_tools_rather_than_owning_a_registry() {
        let profile = TestProfile::with_tools(vec![tool("read_file"), tool("shell")]);

        let contributed = profile.base_tools();

        assert_eq!(
            contributed
                .iter()
                .map(|tool| tool.definition.name.as_str())
                .collect::<Vec<_>>(),
            ["read_file", "shell"]
        );
    }

    #[test]
    fn a_profile_contributes_no_subagent_tools_by_default() {
        let profile = TestProfile::new();

        assert!(
            profile
                .subagent_tools(&SubagentSupport::default())
                .is_empty()
        );
    }

    #[test]
    fn a_prompt_is_built_from_the_gathered_context() {
        let profile = TestProfile::new();
        let env_context = EnvContext {
            working_directory: "/work".to_owned(),
            platform: "linux".to_owned(),
            ..EnvContext::default()
        };

        let prompt =
            profile.build_system_prompt(&ToolRegistry::new(), &env_context, &[], None, &[]);

        assert!(prompt.contains("test assistant"));
        assert!(prompt.contains("Working directory: /work"));
        assert!(prompt.contains("Platform: linux"));
    }

    #[test]
    fn a_prompt_carries_memory_skills_and_user_instructions() {
        let profile = TestProfile::new();
        let skills = [Skill {
            name:        "review".to_owned(),
            description: "Review a change".to_owned(),
            template:    "Read the diff.".to_owned(),
        }];

        let prompt = profile.build_system_prompt(
            &ToolRegistry::new(),
            &EnvContext::default(),
            &["README.md contents".to_owned()],
            Some("Always use TDD"),
            &skills,
        );

        assert!(prompt.contains("README.md contents"));
        assert!(prompt.contains("/review: Review a change"));
        assert!(prompt.contains("Always use TDD"));
    }

    #[test]
    fn an_environment_answers_the_three_members_it_owns() {
        let env = MockEnvironment::linux();

        let context = EnvContext::from_environment(&env);

        assert_eq!(context.working_directory, "/home/test");
        assert_eq!(context.platform, "linux");
        assert_eq!(context.os_version, "Linux 6.1.0");
        assert!(!context.is_git_repo);
        assert!(context.git_branch.is_none());
        assert!(context.model.is_empty());
    }

    #[test]
    fn every_harness_pebble_ships_answers_for_the_one_it_was_asked_for() {
        for kind in AgentProfileKind::ALL {
            let profile = builtin_profile(*kind, &ProfileDeps::default());

            assert_eq!(
                profile.profile_kind(),
                *kind,
                "{kind} was answered with another harness"
            );
        }
    }

    /// Every harness gives its model a shell, whatever else it withholds. The
    /// GPT-5.6 one has nothing but a shell and a patch tool, and Claude 5
    /// drives its own searches through one, so this is the only tool all
    /// six share.
    #[test]
    fn every_harness_pebble_ships_gives_its_model_a_way_to_run_a_command() {
        for kind in AgentProfileKind::ALL {
            let profile = builtin_profile(*kind, &ProfileDeps::default());

            assert!(
                profile
                    .base_tools()
                    .iter()
                    .any(|tool| tool.definition.name == "shell"),
                "{kind} offers no shell"
            );
        }
    }

    /// A built-in harness contributes built-in tools, under their canonical
    /// names. That is what lets the registry rename them into the harness's own
    /// vocabulary, and what keeps an application's permission gate able to
    /// categorize what it is being asked to approve.
    #[test]
    fn every_tool_a_built_in_harness_contributes_is_one_pebble_knows_by_name() {
        use crate::tool::NativeTool;

        for kind in AgentProfileKind::ALL {
            let profile = builtin_profile(*kind, &ProfileDeps::default());

            for tool in profile.base_tools() {
                assert_eq!(tool.source, ToolSource::Native, "{kind}");
                assert!(
                    NativeTool::from_canonical_name(&tool.definition.name).is_some(),
                    "{kind} contributes `{}`, which pebble cannot rename or categorize",
                    tool.definition.name
                );
            }
        }
    }

    #[test]
    fn a_model_the_catalog_says_nothing_about_gets_the_default_window() {
        let facts = ModelFacts::default();

        assert_eq!(facts.context_window_tokens, DEFAULT_CONTEXT_WINDOW_TOKENS);
        assert_eq!(facts.max_output_tokens, None);
        assert!(!facts.reasons_by_default);
    }

    #[test]
    fn catalog_limits_become_model_facts() {
        let catalog = Catalog::builder()
            .toml_layer(
                "test",
                r#"
                schema_version = 1

                [providers.mock]
                display_name = "Mock"
                adapter = "openai"
                codec = "openai-chat"
                base_url = "https://example.invalid"
                auth = { type = "none" }

                [providers.mock.models.described]
                display_name = "Described"
                api_model = "described"
                limits = { context_tokens = 128000, max_output_tokens = 8192 }
                capabilities = { text = true, reasoning = true, reasoning_effort_levels = true }

                [providers.mock.models.undescribed]
                display_name = "Undescribed"
                api_model = "undescribed"
                "#,
            )
            .expect("the layer parses")
            .build()
            .expect("the catalog validates");

        let described = catalog.model("mock", "described").expect("a known model");
        let undescribed = catalog.model("mock", "undescribed").expect("a known model");

        assert_eq!(
            ModelFacts::from_catalog_model(described),
            ModelFacts::new()
                .with_context_window_tokens(128_000)
                .with_max_output_tokens(Some(8192))
                .with_reasons_by_default(true)
        );
        assert_eq!(
            ModelFacts::from_catalog_model(undescribed),
            ModelFacts::default()
        );
    }

    /// A catalog naming one model, with whatever capabilities the case needs.
    fn catalog_with(capabilities: &str) -> Catalog {
        Catalog::builder()
            .toml_layer(
                "test",
                &format!(
                    r#"
                    schema_version = 1

                    [providers.mock]
                    display_name = "Mock"
                    adapter = "openai"
                    codec = "openai-chat"
                    base_url = "https://example.invalid"
                    auth = {{ type = "none" }}

                    [providers.mock.models.plain]
                    display_name = "Plain"
                    api_model = "plain"
                    limits = {{ context_tokens = 8000, max_output_tokens = 1024 }}
                    capabilities = {{ {capabilities} }}
                    "#
                ),
            )
            .expect("the layer parses")
            .build()
            .expect("the catalog validates")
    }

    /// Whether the one model of a catalog built from `capabilities` reasons
    /// without being asked to.
    fn reasons_by_default(capabilities: &str) -> bool {
        let catalog = catalog_with(capabilities);
        let model = catalog.model("mock", "plain").expect("a known model");
        ModelFacts::from_catalog_model(model).reasons_by_default
    }

    #[test]
    fn a_model_that_cannot_reason_says_so() {
        assert!(!reasons_by_default("text = true"));
    }

    #[test]
    fn only_a_model_that_takes_an_effort_level_reasons_by_default() {
        // Fabro's own rule: a model the request can set an effort level on
        // reasons unless it is told not to, while one that only takes a
        // thinking budget reasons when it is asked to. The second case is the
        // one a `reasoning` capability alone gets wrong.
        assert!(reasons_by_default(
            "text = true, reasoning = true, reasoning_effort_levels = true"
        ));
        assert!(!reasons_by_default("text = true, reasoning = true"));
        assert!(
            !reasons_by_default("text = true, reasoning_effort_levels = true"),
            "an effort level means nothing without reasoning behind it"
        );
    }

    #[test]
    fn facts_can_be_built_by_hand() {
        let facts = ModelFacts::new()
            .with_context_window_tokens(64_000)
            .with_max_output_tokens(Some(4_096))
            .with_reasons_by_default(true);

        assert_eq!(facts.context_window_tokens, 64_000);
        assert_eq!(facts.max_output_tokens, Some(4_096));
        assert!(facts.reasons_by_default);
    }
}
