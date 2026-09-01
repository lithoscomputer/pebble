//! The harness Anthropic's coding models expect.

use std::sync::Arc;

use super::{EmbeddedPrompt, ProfileDeps, assemble_system_prompt, core_tools};
use crate::config::NativeToolOptions;
use crate::profile::{AgentProfile, EnvContext};
use crate::skills::Skill;
use crate::tool::{NativeTool, RegisteredTool, ToolRegistry, ToolVocabulary};
use crate::tools::{
    TodoRuntime, make_edit_file_tool, make_task_create_tool, make_task_get_tool,
    make_task_list_tool, make_task_update_tool,
};
use crate::types::AgentProfileKind;

/// The system prompt this harness starts from.
const CORE_PROMPT: &str = include_str!("prompts/anthropic.md.j2");

/// Pebble's own tool names, an exact-string file editor, and the four
/// incremental task tools.
///
/// The task tools are what distinguishes this harness from the OpenAI one:
/// they add and update one item at a time against a list scoped to the root
/// session, rather than replacing a whole plan. Every session in one tree runs
/// the same profile, so a child's tasks land on the same list as its parent's.
pub(crate) struct AnthropicProfile {
    tools: Vec<RegisteredTool>,
}

impl AnthropicProfile {
    /// The harness for a session built from `deps`.
    pub(crate) fn new(deps: &ProfileDeps) -> Self {
        let options = NativeToolOptions::for_profile(AgentProfileKind::Anthropic);
        let mut tools = core_tools(
            &options,
            deps.search_provider.clone(),
            deps.web_fetch_summarizer.clone(),
        );
        tools.push(make_edit_file_tool());
        // One runtime behind all four, so a task created through one tool is
        // the task the others read.
        let todo_runtime = Arc::new(TodoRuntime::new());
        tools.push(make_task_create_tool(Arc::clone(&todo_runtime)));
        tools.push(make_task_update_tool(Arc::clone(&todo_runtime)));
        tools.push(make_task_get_tool(Arc::clone(&todo_runtime)));
        tools.push(make_task_list_tool(todo_runtime));

        Self { tools }
    }
}

impl AgentProfile for AnthropicProfile {
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
        registry: &ToolRegistry,
        env_context: &EnvContext,
        memory: &[String],
        user_instructions: Option<&str>,
        skills: &[Skill],
    ) -> String {
        let template = EmbeddedPrompt::new("anthropic.md.j2", CORE_PROMPT)
            .with_bool(
                "has_spawn_agent",
                registry.get_native(NativeTool::SpawnAgent).is_some(),
            )
            .with_bool(
                "has_web_search",
                registry.get_native(NativeTool::WebSearch).is_some(),
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
        advertises, native_marker, search_provider, shell_timeout_ms, snapshot_context,
        system_prompt_with_tools, tool_names, web_search_name,
    };

    /// The harness a session with `has_web_search` gets.
    fn profile(has_web_search: bool) -> AnthropicProfile {
        AnthropicProfile::new(&ProfileDeps {
            search_provider: search_provider(has_web_search),
            ..ProfileDeps::default()
        })
    }

    /// The prompt from a completed registry with these optional capabilities.
    fn prompt(has_web_search: bool, has_subagents: bool) -> String {
        let profile = profile(has_web_search);
        let extra = has_subagents
            .then(|| native_marker(NativeTool::SpawnAgent))
            .into_iter()
            .collect();
        system_prompt_with_tools(&profile, extra)
    }

    #[test]
    fn the_profile_names_its_harness_and_pebbles_own_vocabulary() {
        let profile = profile(false);

        assert_eq!(profile.profile_kind(), AgentProfileKind::Anthropic);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::Canonical);
    }

    #[test]
    fn the_harness_offers_eleven_tools() {
        assert_eq!(tool_names(&profile(false)), [
            "TaskCreate",
            "TaskGet",
            "TaskList",
            "TaskUpdate",
            "edit_file",
            "glob",
            "grep",
            "read_file",
            "shell",
            "web_fetch",
            "write_file",
        ]);
    }

    #[test]
    fn the_harness_leaves_out_the_tools_other_harnesses_own() {
        let profile = profile(false);

        // `update_plan` is Codex's whole-plan surface, and this harness keeps
        // an incremental list instead.
        assert!(!advertises(&profile, "update_plan"));
        // The search tool is the session builder's to add, and only when the
        // application gave the session an engine.
        assert!(!advertises(&profile, "web_search"));
        // Reading many files and listing a directory are Gemini's.
        assert!(!advertises(&profile, "read_many_files"));
        assert!(!advertises(&profile, "list_dir"));
    }

    #[tokio::test]
    async fn a_command_with_no_timeout_gets_this_harnesses_own() {
        // Anthropic's harness documents two minutes, where pebble's own
        // default is ten seconds.
        assert_eq!(shell_timeout_ms(&profile(false)).await, 120_000);
    }

    #[test]
    fn the_prompt_says_who_the_model_is_and_where_it_is_working() {
        let prompt = prompt(false, false);

        assert!(prompt.contains("You are Claude, an AI coding assistant made by Anthropic"));
        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("Working directory: /home/test"));
        assert!(prompt.contains("Platform: linux"));
    }

    #[test]
    fn the_prompt_keeps_the_claude_code_style_sections() {
        let prompt = prompt(false, false);

        for heading in [
            "# System",
            "# Doing tasks",
            "# Executing actions with care",
            "# Using your tools",
            "# Communicating with the user",
            "# Tone and style",
            "# Coding Best Practices",
        ] {
            assert!(prompt.contains(heading), "the prompt lost {heading}");
        }
    }

    #[test]
    fn the_prompt_drills_the_habits_this_harness_was_trained_on() {
        let prompt = prompt(false, false);

        assert!(prompt.contains(
            "Do NOT use the shell tool to run commands when a relevant dedicated tool is provided"
        ));
        assert!(prompt.contains("Break down and manage your work with the TaskCreate tool"));
        assert!(prompt.contains("Use TaskUpdate to keep task status current"));
        assert!(prompt.contains("Mark each task as completed as soon as you are done"));
        assert!(
            prompt.contains("Before your first tool call, briefly state what you're about to do")
        );
        assert!(prompt.contains("Do not expose internal deliberation"));
        assert!(prompt.contains("Do not create planning documents unless the user asks"));
        assert!(prompt.contains("read or inspect it first"));
        assert!(prompt.contains("Report outcomes faithfully"));
        assert!(prompt.contains("Write clean, maintainable code"));
        assert!(
            !prompt.contains("## read_file"),
            "per-tool detail belongs in the tool descriptions"
        );
    }

    #[test]
    fn the_prompt_mentions_a_search_tool_exactly_when_the_session_has_one() {
        let without = prompt(false, false);
        let with = prompt(true, false);

        assert!(!without.contains(web_search_name(&profile(false))));
        assert!(without.contains("To inspect a specific URL use web_fetch."));
        assert!(
            with.contains("To search the internet use web_search, and to inspect a specific URL")
        );
    }

    #[test]
    fn the_prompt_mentions_subagents_exactly_when_the_session_can_spawn_one() {
        let without = prompt(false, false);
        let with = prompt(false, true);

        assert!(!without.contains("Subagents are valuable for independent work"));
        assert!(with.contains("Subagents are valuable for independent work"));
        assert!(with.contains("avoid duplicating work"));
        assert!(with.contains("wait for their results and synthesize them"));
    }

    #[test]
    fn the_prompt_carries_memory_and_user_instructions() {
        let profile = profile(false);

        let prompt = profile.build_system_prompt(
            &ToolRegistry::new(),
            &snapshot_context(),
            &[
                "# Project README".to_owned(),
                "# CONTRIBUTING guide".to_owned(),
            ],
            Some("Always write tests first"),
            &[],
        );

        assert!(prompt.contains("# Project README"));
        assert!(prompt.contains("# CONTRIBUTING guide"));
        assert!(prompt.contains("# User Instructions\nAlways write tests first"));
    }

    #[test]
    fn the_prompt_carries_what_the_session_gathered_about_its_environment() {
        let context = EnvContext {
            is_git_repo: true,
            git_branch: Some("feature-branch".to_owned()),
            current_date: "2026-02-20".to_owned(),
            model: "claude-opus-4-6".to_owned(),
            knowledge_cutoff: "May 2025".to_owned(),
            ..snapshot_context()
        };

        let prompt =
            profile(false).build_system_prompt(&ToolRegistry::new(), &context, &[], None, &[]);

        assert!(prompt.contains("Git branch: feature-branch"));
        assert!(prompt.contains("Is git repository: true"));
        assert!(prompt.contains("Today's date: 2026-02-20"));
        assert!(prompt.contains("Model: claude-opus-4-6"));
        assert!(prompt.contains("Knowledge cutoff: May 2025"));
    }

    #[test]
    fn default_prompt_snapshot() {
        insta::assert_snapshot!(prompt(false, false));
    }

    #[test]
    fn web_search_prompt_snapshot() {
        insta::assert_snapshot!(prompt(true, false));
    }

    #[test]
    fn subagents_prompt_snapshot() {
        insta::assert_snapshot!(prompt(false, true));
    }

    #[test]
    fn web_search_and_subagents_prompt_snapshot() {
        insta::assert_snapshot!(prompt(true, true));
    }
}
