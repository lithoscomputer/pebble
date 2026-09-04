//! The harness GPT-6 models expect.
//!
//! GPT-6 Astra uses Codex's narrow tool contract, like the GPT-5.6 family, but
//! OpenAI gives it a distinct prompt. Its prompt calibrates the model's
//! initiative, sensitivity to instruction files, response style, subagent
//! use, and test scope. The tools stay in [`super::codex_tools`] so the
//! two Codex-trained families cannot drift apart at the protocol boundary.

use super::codex_tools::codex_tools;
use super::{EmbeddedPrompt, FileEditToolKind, ProfileDeps, assemble_system_prompt};
use crate::profile::{AgentProfile, EnvContext};
use crate::skills::Skill;
use crate::tool::{NativeTool, RegisteredTool, ToolRegistry, ToolVocabulary};
use crate::types::AgentProfileKind;

/// The system prompt this harness starts from.
const CORE_PROMPT: &str = include_str!("prompts/gpt6.md.j2");

/// Codex's direct-call tools with GPT-6-specific instructions.
pub(crate) struct Gpt6Profile {
    tools:                 Vec<RegisteredTool>,
    provider_display_name: String,
    file_edit_tool:        FileEditToolKind,
}

impl Gpt6Profile {
    /// The harness for a session built from `deps`.
    pub(crate) fn new(deps: &ProfileDeps) -> Self {
        Self {
            tools:                 codex_tools(AgentProfileKind::Gpt6, deps),
            provider_display_name: deps.provider_display_name.clone(),
            file_edit_tool:        deps.file_edit_tool,
        }
    }
}

impl AgentProfile for Gpt6Profile {
    fn profile_kind(&self) -> AgentProfileKind {
        AgentProfileKind::Gpt6
    }

    fn tool_vocabulary(&self) -> ToolVocabulary {
        ToolVocabulary::Codex
    }

    fn base_tools(&self) -> Vec<RegisteredTool> {
        self.tools.clone()
    }

    fn build_system_prompt(
        &self,
        registry: &ToolRegistry,
        env_context: &EnvContext,
        memory: &[String],
        user_instructions: Option<&str>,
        skills: &[Skill],
    ) -> String {
        let template = EmbeddedPrompt::new("gpt6.md.j2", CORE_PROMPT)
            .with_string("provider_name", self.provider_display_name.clone())
            .with_string("file_edit_tool", self.file_edit_tool.as_str())
            .with_bool(
                "has_web_search",
                registry.get_native(NativeTool::WebSearch).is_some(),
            )
            .with_bool(
                "has_subagents",
                registry.get_native(NativeTool::SpawnAgent).is_some(),
            )
            .with_bool(
                "has_questions",
                registry.get_native(NativeTool::RequestUserInput).is_some(),
            );

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
        native_marker, registry_of, search_provider, snapshot_context, system_prompt,
        system_prompt_with_tools, tool_names,
    };

    /// The GPT-6 harness reached by `file_edit_tool`, with search when asked.
    fn profile(file_edit_tool: FileEditToolKind, has_web_search: bool) -> Gpt6Profile {
        Gpt6Profile::new(&ProfileDeps {
            provider_display_name: "OpenAI".to_owned(),
            file_edit_tool,
            search_provider: search_provider(has_web_search),
            ..ProfileDeps::default()
        })
    }

    /// GPT-6 reached over OpenAI Responses, which carries the patch grammar.
    fn patching(has_web_search: bool) -> Gpt6Profile {
        profile(FileEditToolKind::ApplyPatch, has_web_search)
    }

    /// GPT-6 reached through a compatible gateway, which does not.
    fn editing(has_web_search: bool) -> Gpt6Profile {
        profile(FileEditToolKind::EditFile, has_web_search)
    }

    #[test]
    fn the_profile_uses_gpt6s_identity_and_codexs_tool_contract() {
        let profile = patching(false);

        assert_eq!(profile.profile_kind(), AgentProfileKind::Gpt6);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::Codex);
        assert_eq!(tool_names(&profile), [
            "apply_patch",
            "shell",
            "update_plan"
        ]);
        assert!(system_prompt(&profile).starts_with("You are a coding agent based on GPT-6"));
    }

    #[test]
    fn the_prompt_carries_astras_behavior_guidance() {
        let prompt = system_prompt(&patching(false));

        for expected in [
            "bias towards action",
            "The user's instructions take precedence",
            "Use lists only when the information is genuinely parallel",
            "Do not write tests for reversible, low-impact changes",
            "Run tests appropriate to the change",
        ] {
            assert!(prompt.contains(expected), "the prompt lost `{expected}`");
        }
    }

    #[test]
    fn optional_guidance_names_only_tools_the_session_has() {
        let profile = patching(true);
        let without = system_prompt(&profile);
        assert!(!without.contains("call `request_user_input`"));
        assert!(!without.contains("use `spawn_agent`"));
        assert!(without.contains("use `web_search` rather than guessing"));

        let with = system_prompt_with_tools(&profile, vec![
            native_marker(NativeTool::RequestUserInput),
            native_marker(NativeTool::SpawnAgent),
        ]);
        assert!(with.contains("call `request_user_input`"));
        assert!(with.contains("use `spawn_agent`"));
    }

    #[test]
    fn the_prompt_describes_the_editor_the_route_carries() {
        let patching = system_prompt(&patching(false));
        assert!(patching.contains("Use `apply_patch` for local file edits"));
        assert!(patching.contains("*** Begin Patch"));
        assert!(!patching.contains("edit_file"));

        let editing = system_prompt(&editing(false));
        assert!(editing.contains("Use `edit_file` for local file edits"));
        assert!(!editing.contains("apply_patch"));
    }

    #[test]
    fn the_prompt_carries_memory_and_user_instructions() {
        let profile = patching(false);
        let prompt = profile.build_system_prompt(
            &registry_of(&profile, Vec::new()),
            &snapshot_context(),
            &["# Project README".to_owned()],
            Some("Always write tests first"),
            &[],
        );

        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("# Project README"));
        assert!(prompt.ends_with("# User Instructions\nAlways write tests first"));
    }

    #[test]
    fn the_attribution_never_reaches_the_model() {
        let prompt = system_prompt(&patching(false));

        assert!(!prompt.contains("Apache-2.0"));
        assert!(CORE_PROMPT.contains("Adapted from openai/codex (Apache-2.0)"));
    }

    #[test]
    fn default_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&patching(false)));
    }

    #[test]
    fn fully_equipped_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt_with_tools(&patching(true), vec![
            native_marker(NativeTool::RequestUserInput),
            native_marker(NativeTool::SpawnAgent),
        ],));
    }
}
