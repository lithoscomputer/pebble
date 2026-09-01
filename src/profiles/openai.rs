//! The harness OpenAI's coding models expect, and every model reached through
//! an OpenAI-compatible gateway.

use std::sync::Arc;

use super::{EmbeddedPrompt, FileEditToolKind, ProfileDeps, assemble_system_prompt, core_tools};
use crate::config::NativeToolOptions;
use crate::profile::{AgentProfile, EnvContext};
use crate::skills::Skill;
use crate::tool::{RegisteredTool, ToolRegistry, ToolVocabulary};
use crate::tools::{TodoRuntime, make_update_plan_tool};
use crate::types::AgentProfileKind;

/// The system prompt this harness starts from.
const CORE_PROMPT: &str = include_str!("prompts/openai.md.j2");

/// Pebble's own tool names, Codex's whole-plan `update_plan`, and whichever
/// file editor the route's wire codec can carry.
///
/// This is the one harness whose tool set depends on how the model is reached
/// rather than on which model it is: `apply_patch` is a freeform grammar that
/// only the OpenAI Responses codec carries, so a model served through an
/// OpenAI-compatible gateway is offered `edit_file` instead. The prompt names
/// the editor it was given, in three places, so guidance cannot describe a tool
/// the session does not have.
pub(crate) struct OpenAiProfile {
    tools:                 Vec<RegisteredTool>,
    provider_display_name: String,
    file_edit_tool:        FileEditToolKind,
    has_web_search:        bool,
}

impl OpenAiProfile {
    /// The harness for a session built from `deps`.
    pub(crate) fn new(deps: &ProfileDeps) -> Self {
        let options = NativeToolOptions::for_profile(AgentProfileKind::OpenAi);
        let mut tools = core_tools(&options);
        tools.push(deps.file_edit_tool.tool());
        // Codex's `update_plan` replaces a whole plan at once, so the list
        // behind it belongs to this session rather than to the tree.
        tools.push(make_update_plan_tool(Arc::new(TodoRuntime::new())));

        Self {
            tools,
            provider_display_name: deps.provider_display_name.clone(),
            file_edit_tool: deps.file_edit_tool,
            has_web_search: deps.has_web_search(),
        }
    }
}

impl AgentProfile for OpenAiProfile {
    fn profile_kind(&self) -> AgentProfileKind {
        AgentProfileKind::OpenAi
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
        let template = EmbeddedPrompt::new("openai.md.j2", CORE_PROMPT)
            .with_string("provider_name", self.provider_display_name.clone())
            .with_string("file_edit_tool", self.file_edit_tool.as_str())
            .with_bool("has_web_search", self.has_web_search);

        assemble_system_prompt(
            template,
            self.tool_vocabulary(),
            env_context,
            memory,
            user_instructions,
            skills,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::tests::{
        advertises, search_provider, shell_timeout_ms, snapshot_context, system_prompt, tool_names,
        web_search_name,
    };

    /// The harness a session on `file_edit_tool` with `has_web_search` gets,
    /// answering for `provider_display_name`.
    fn profile(
        file_edit_tool: FileEditToolKind,
        has_web_search: bool,
        provider_display_name: &str,
    ) -> OpenAiProfile {
        OpenAiProfile::new(&ProfileDeps {
            provider_display_name: provider_display_name.to_owned(),
            file_edit_tool,
            search_provider: search_provider(has_web_search),
            ..ProfileDeps::default()
        })
    }

    /// The harness a model reached over the OpenAI Responses codec gets.
    fn patching(has_web_search: bool) -> OpenAiProfile {
        profile(FileEditToolKind::ApplyPatch, has_web_search, "OpenAI")
    }

    /// The harness a model reached over any other codec gets.
    fn editing(has_web_search: bool) -> OpenAiProfile {
        profile(FileEditToolKind::EditFile, has_web_search, "Moonshot AI")
    }

    #[test]
    fn the_profile_names_its_harness_and_pebbles_own_vocabulary() {
        let profile = patching(false);

        assert_eq!(profile.profile_kind(), AgentProfileKind::OpenAi);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::Canonical);
    }

    #[test]
    fn the_harness_offers_eight_tools() {
        assert_eq!(tool_names(&patching(false)), [
            "apply_patch",
            "glob",
            "grep",
            "read_file",
            "shell",
            "update_plan",
            "web_fetch",
            "write_file",
        ]);
    }

    #[test]
    fn a_route_that_cannot_carry_a_patch_grammar_gets_the_json_editor_instead() {
        let profile = editing(false);

        assert!(advertises(&profile, "edit_file"));
        assert!(!advertises(&profile, "apply_patch"));
        assert_eq!(tool_names(&profile).len(), 8);
    }

    #[test]
    fn the_patch_tool_is_the_one_definition_that_is_not_a_function() {
        let patch = patching(false)
            .base_tools()
            .into_iter()
            .find(|tool| tool.definition.name == "apply_patch")
            .expect("the harness offers a patch tool");

        assert!(patch.definition.is_custom());
        // Every other definition on either route is an object-shaped function,
        // which is what a codec that refuses freeform tools requires.
        for profile in [patching(false), editing(false)] {
            for tool in profile.base_tools() {
                if tool.definition.name == "apply_patch" {
                    continue;
                }
                assert!(
                    !tool.definition.is_custom(),
                    "{} must be a function",
                    tool.definition.name
                );
            }
        }
    }

    #[test]
    fn the_harness_leaves_out_the_tools_other_harnesses_own() {
        let profile = patching(false);

        assert!(!advertises(&profile, "TaskCreate"));
        assert!(!advertises(&profile, "TaskUpdate"));
        assert!(!advertises(&profile, "TaskList"));
        assert!(!advertises(&profile, "read_many_files"));
        assert!(!advertises(&profile, "list_dir"));
        assert!(!advertises(&profile, "web_search"));
    }

    #[tokio::test]
    async fn a_command_with_no_timeout_gets_the_ten_seconds_the_prompt_promises() {
        assert_eq!(shell_timeout_ms(&patching(false)).await, 10_000);
        assert!(system_prompt(&patching(false)).contains("Default timeout is 10 seconds"));
    }

    #[test]
    fn the_prompt_names_the_provider_the_catalog_names() {
        for (display_name, expected) in [
            ("OpenAI", "powered by OpenAI"),
            ("Moonshot AI", "powered by Moonshot AI"),
            ("Z.ai", "powered by Z.ai"),
            ("MiniMax", "powered by MiniMax"),
            ("Inception", "powered by Inception"),
        ] {
            let prompt = system_prompt(&profile(FileEditToolKind::EditFile, false, display_name));
            assert!(prompt.contains(expected), "{prompt}");
        }
    }

    #[test]
    fn the_prompt_says_where_the_session_is_working() {
        let prompt = system_prompt(&patching(false));

        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("Working directory: /home/test"));
        assert!(prompt.contains("Platform: linux"));
    }

    #[test]
    fn the_prompt_documents_the_editor_the_route_can_carry() {
        let patch = system_prompt(&patching(false));
        let edit = system_prompt(&editing(false));

        assert!(patch.contains("## apply_patch"));
        assert!(patch.contains("freeform tool"));
        assert!(patch.contains("*** Begin Patch"));
        assert!(patch.contains("When apply_patch fails"));
        assert!(patch.contains("For modifications, prefer apply_patch."));
        assert!(!patch.contains("## edit_file"));

        assert!(edit.contains("## edit_file"));
        assert!(edit.contains("When edit_file fails"));
        assert!(edit.contains("For modifications, prefer edit_file."));
        assert!(!edit.contains("## apply_patch"));
        assert!(!edit.contains("freeform tool"));
        assert!(!edit.contains("*** Begin Patch"));
    }

    #[test]
    fn the_prompt_documents_the_tools_it_tells_the_model_how_to_steer() {
        let prompt = system_prompt(&patching(false));

        for tool in [
            "read_file",
            "apply_patch",
            "write_file",
            "shell",
            "grep",
            "glob",
            "web_fetch",
        ] {
            assert!(
                prompt.contains(&format!("## {tool}")),
                "the prompt lost {tool}"
            );
        }
        assert!(prompt.contains("timeout_ms"));
    }

    #[test]
    fn the_prompt_keeps_codexs_own_guidance() {
        let prompt = system_prompt(&patching(false));

        assert!(prompt.contains(
            "update item statuses incrementally as each item is completed rather than marking \
             every item done only at the end"
        ));
        assert!(prompt.contains("clean, maintainable code"));
        assert!(prompt.contains("existing code conventions"));
        assert!(prompt.contains("NEVER add copyright or license headers"));
        assert!(prompt.contains("# AGENTS.md"));
    }

    #[test]
    fn the_prompt_mentions_a_search_tool_exactly_when_the_session_has_one() {
        let without = system_prompt(&patching(false));
        let with = system_prompt(&patching(true));

        assert!(!without.contains(web_search_name(&patching(false))));
        assert!(with.contains("## web_search"));
        assert!(with.contains("Search the web. Returns titles, URLs, and descriptions."));
    }

    #[test]
    fn the_prompt_carries_memory_and_user_instructions() {
        let prompt = patching(false).build_system_prompt(
            &ToolRegistry::new(),
            &snapshot_context(),
            &["# Project README".to_owned()],
            Some("Always write tests first"),
            &[],
        );

        assert!(prompt.contains("# Project README"));
        assert!(prompt.contains("# User Instructions\nAlways write tests first"));
    }

    #[test]
    fn apply_patch_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&patching(false)));
    }

    #[test]
    fn apply_patch_and_web_search_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&patching(true)));
    }

    #[test]
    fn edit_file_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&editing(false)));
    }

    #[test]
    fn edit_file_and_web_search_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&editing(true)));
    }
}
