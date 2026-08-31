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

use lithos_llm::catalog::CatalogModel;

use crate::environment::Environment;
use crate::tool::{RegisteredTool, ToolVocabulary};
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

/// A prompt a person wrote for the model to follow on demand.
///
/// A skill is discovered as a file, named in conversation, and expanded into
/// the model's context by the skill tool. Pebble carries the three parts every
/// profile's prompt assembly needs: what to call it, when to reach for it, and
/// what it says.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Skill {
    /// What the skill is named, and what a person types to invoke it.
    pub name:        String,
    /// When to use the skill, written for the model.
    pub description: String,
    /// The prompt the skill expands into.
    pub template:    String,
}

/// What a session's subagent support hands a profile.
///
/// Subagent tools differ by profile — pebble's own set spawns and waits on
/// agents, while the Claude 5 harness expects a background-agent family
/// instead — so a profile builds them rather than a builder choosing for it.
///
/// It carries only the session's place in the tree today; the supervisor and
/// the session factory arrive here when subagents land. Both are shared
/// handles, so this type deliberately derives neither `Copy` nor `Eq`: growing
/// it must not have to remove a derive that callers depend on.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SubagentSupport {
    /// How deep in the session tree the session being built sits, counting the
    /// root as zero.
    pub depth: usize,
}

/// The token budgets one model works within.
///
/// Fabro's profiles answered these from their own catalog; pebble reads them
/// from the lithos-llm catalog the session resolved its model through, so a
/// profile carries no model facts at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFacts {
    /// How many tokens the model's context window holds.
    pub context_window_tokens: usize,
    /// The most tokens the model may produce in one response, when the catalog
    /// says.
    pub max_output_tokens:     Option<u64>,
}

impl ModelFacts {
    /// The facts the catalog records about one model.
    ///
    /// A model the catalog describes without limits — a passthrough entry, say
    /// — falls back to [`DEFAULT_CONTEXT_WINDOW_TOKENS`] and no output limit,
    /// which is what fabro assumed for a model it could not look up.
    #[must_use]
    pub fn from_catalog_model(model: &CatalogModel) -> Self {
        model.limits().map_or_else(Self::default, |limits| Self {
            context_window_tokens: usize::try_from(limits.context_tokens).unwrap_or(usize::MAX),
            max_output_tokens:     Some(limits.max_output_tokens),
        })
    }
}

impl Default for ModelFacts {
    fn default() -> Self {
        Self {
            context_window_tokens: DEFAULT_CONTEXT_WINDOW_TOKENS,
            max_output_tokens:     None,
        }
    }
}

/// One model family's harness: its tools, their names, and its prompt.
///
/// Pebble ships a profile for each family it supports and selects one from the
/// catalog. An application implements this only to run a model pebble does not
/// know, or to change what a known family is given.
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
    /// `memory` holds the files the application asked the session to load,
    /// already read; `user_instructions` is the application's own addition; and
    /// `skills` are the skills discovered for this session, which a profile
    /// normally summarizes rather than expands.
    fn build_system_prompt(
        &self,
        env_context: &EnvContext,
        memory: &[String],
        user_instructions: Option<&str>,
        skills: &[Skill],
    ) -> String;

    /// The subagent tools this profile gives the session.
    ///
    /// Answered only when the application configured subagents. The default is
    /// no tools, so a profile that has no subagent story says nothing.
    fn subagent_tools(&self, _subagents: &SubagentSupport) -> Vec<RegisteredTool> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;
    use std::sync::Arc;

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

        let prompt = profile.build_system_prompt(&env_context, &[], None, &[]);

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
    fn a_model_the_catalog_says_nothing_about_gets_the_default_window() {
        let facts = ModelFacts::default();

        assert_eq!(facts.context_window_tokens, DEFAULT_CONTEXT_WINDOW_TOKENS);
        assert_eq!(facts.max_output_tokens, None);
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

        assert_eq!(ModelFacts::from_catalog_model(described), ModelFacts {
            context_window_tokens: 128_000,
            max_output_tokens:     Some(8192),
        });
        assert_eq!(
            ModelFacts::from_catalog_model(undescribed),
            ModelFacts::default()
        );
    }
}
