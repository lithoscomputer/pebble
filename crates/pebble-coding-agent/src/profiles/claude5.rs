//! The harness Claude Fable 5, Opus 5 and Sonnet 5 expect.

use std::sync::Arc;

use super::{EmbeddedPrompt, ProfileDeps, assemble_system_prompt, claude5_tools};
use crate::config::NativeToolOptions;
use crate::profile::{AgentProfile, EnvContext, SubagentSupport};
use crate::skills::Skill;
use crate::tool::{NativeTool, RegisteredTool, ToolRegistry, ToolVocabulary};
use crate::tools::{
    TodoRuntime, make_task_create_tool, make_task_get_tool, make_task_list_tool,
    make_task_update_tool,
};
use crate::types::AgentProfileKind;

/// The system prompt this harness starts from.
const CORE_PROMPT: &str = include_str!("prompts/claude5.md.j2");

/// Claude 5's own tool names, strict schemas, and a background-agent family in
/// place of pebble's spawn-and-wait one.
///
/// Three things set this harness apart. Its schemas are strict objects, so a
/// field the model invented is refused rather than quietly dropped. It has no
/// `grep` or `glob`: this model family was trained to drive `rg` through the
/// shell, and offering a second way to search costs a choice at every turn.
/// And its subagents are background agents that announce themselves when they
/// finish, rather than handles the model has to remember to wait on.
pub(crate) struct Claude5Profile {
    tools: Vec<RegisteredTool>,
}

impl Claude5Profile {
    /// The harness for a session built from `deps`.
    pub(crate) fn new(deps: &ProfileDeps) -> Self {
        let options = NativeToolOptions::for_profile(AgentProfileKind::Claude5);
        let mut tools = vec![
            claude5_tools::make_read_tool(),
            claude5_tools::make_write_tool(),
            claude5_tools::make_edit_tool(),
            claude5_tools::make_bash_tool(&options),
            claude5_tools::make_claude5_web_fetch_tool(deps.web_fetch_summarizer.clone()),
        ];
        // The one harness that gives its model a search tool of a different
        // shape, so it replaces the canonical one the builder registered.
        if let Some(provider) = &deps.search_provider {
            tools.push(claude5_tools::make_claude5_web_search_tool(Arc::clone(
                provider,
            )));
        }

        // One runtime behind all four, so a task created through one tool is
        // the task the others read.
        let todo_runtime = Arc::new(TodoRuntime::new());
        for task_tool in [
            make_task_create_tool(Arc::clone(&todo_runtime)),
            make_task_update_tool(Arc::clone(&todo_runtime)),
            make_task_get_tool(Arc::clone(&todo_runtime)),
            make_task_list_tool(todo_runtime),
        ] {
            tools.push(claude5_tools::strict_object_tool(task_tool));
        }

        Self { tools }
    }
}

impl AgentProfile for Claude5Profile {
    fn profile_kind(&self) -> AgentProfileKind {
        AgentProfileKind::Claude5
    }

    fn tool_vocabulary(&self) -> ToolVocabulary {
        ToolVocabulary::Claude5
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
        let template = EmbeddedPrompt::new("claude5.md.j2", CORE_PROMPT)
            // A prompt section for a tool the session does not have costs a
            // wasted call, so every conditional comes from what this session
            // actually holds.
            .with_bool(
                "has_agent",
                registry.get_native(NativeTool::BackgroundAgent).is_some(),
            )
            .with_bool(
                "has_ask_user_question",
                registry.get_native(NativeTool::AskUserQuestion).is_some(),
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

    /// Claude 5's background-agent family, not pebble's own four.
    ///
    /// The trait's default is the spawn-and-wait family every other harness
    /// gets; overriding it here is what keeps this model family from being
    /// shown tools it was never trained on.
    fn subagent_tools(&self, subagents: &SubagentSupport) -> Vec<RegisteredTool> {
        subagents
            .supervisor
            .as_ref()
            .map(claude5_tools::background_agent_tools)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::tests::{
        advertises, native_marker, registry_of, search_provider, shell_timeout_ms,
        snapshot_context, system_prompt_with_tools, tool_names, web_search_name,
    };
    use crate::tools::make_question_tool;

    /// The harness a session with `has_web_search` gets.
    fn profile(has_web_search: bool) -> Claude5Profile {
        Claude5Profile::new(&ProfileDeps {
            search_provider: search_provider(has_web_search),
            ..ProfileDeps::default()
        })
    }

    /// What a session with someone to ask holds beyond the harness's own tools.
    ///
    /// The builder registers this one, not the profile: the harness offers a
    /// question tool, and the application says whether there is anybody to ask.
    fn asked(has_question_tool: bool) -> Vec<RegisteredTool> {
        has_question_tool
            .then(|| make_question_tool(AgentProfileKind::Claude5).expect("this harness can ask"))
            .into_iter()
            .collect()
    }

    /// The prompt from a completed registry with these optional capabilities.
    fn prompt(has_web_search: bool, has_subagents: bool, has_question: bool) -> String {
        let profile = profile(has_web_search);
        let mut extra = asked(has_question);
        if has_subagents {
            extra.push(native_marker(NativeTool::BackgroundAgent));
        }
        system_prompt_with_tools(&profile, extra)
    }

    #[test]
    fn the_profile_names_its_harness_and_claude_5s_own_vocabulary() {
        let profile = profile(false);

        assert_eq!(profile.profile_kind(), AgentProfileKind::Claude5);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::Claude5);
    }

    #[test]
    fn the_harness_offers_nine_tools() {
        assert_eq!(tool_names(&profile(false)), [
            "TaskCreate",
            "TaskGet",
            "TaskList",
            "TaskUpdate",
            "edit_file",
            "read_file",
            "shell",
            "web_fetch",
            "write_file",
        ]);
    }

    /// The names above are canonical because a profile contributes definitions
    /// and the registry renames them; what the model reads is the vocabulary's
    /// spelling.
    #[test]
    fn the_registry_renames_the_contributed_tools_into_claude_5s_spelling() {
        use crate::tool::ToolRegistry;

        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::Claude5);
        for tool in profile(false).base_tools() {
            registry.register(tool);
        }
        let mut names = registry.names();
        names.sort();

        assert_eq!(names, [
            "Bash",
            "Edit",
            "Read",
            "TaskCreate",
            "TaskGet",
            "TaskList",
            "TaskUpdate",
            "WebFetch",
            "Write",
        ]);
    }

    /// Deliberately absent: this harness drives `rg` through `Bash`, and a
    /// second way to search is a decision the model has to make every turn.
    #[test]
    fn the_harness_leaves_out_the_search_tools_it_drives_through_bash() {
        let profile = profile(false);

        assert!(!advertises(&profile, "grep"));
        assert!(!advertises(&profile, "glob"));
        assert!(!advertises(&profile, "read_many_files"));
        assert!(!advertises(&profile, "list_dir"));
        assert!(!advertises(&profile, "update_plan"));
    }

    #[test]
    fn every_tool_this_harness_offers_refuses_a_field_it_does_not_name() {
        use crate::tools::testing::schema_of;

        for tool in profile(true).base_tools() {
            assert_eq!(
                schema_of(&tool)["additionalProperties"],
                serde_json::Value::Bool(false),
                "{} accepts fields it does not name",
                tool.definition.name
            );
        }
    }

    /// The search tool this harness gives its model takes a query and nothing
    /// else, which is what replaces the canonical one the builder registered.
    #[test]
    fn the_search_tool_is_this_harnesss_own_narrower_one() {
        use crate::tools::testing::schema_of;

        let searching = profile(true);
        assert!(advertises(&searching, "web_search"));

        let search = searching
            .base_tools()
            .into_iter()
            .find(|tool| tool.definition.name == "web_search")
            .expect("the harness offers a search tool");
        assert!(
            schema_of(&search)["properties"]
                .get("max_results")
                .is_none()
        );
        assert!(!advertises(&profile(false), "web_search"));
    }

    #[tokio::test]
    async fn a_command_with_no_timeout_gets_this_harnesses_own() {
        assert_eq!(shell_timeout_ms(&profile(false)).await, 120_000);
    }

    #[test]
    fn the_prompt_says_who_the_model_is_and_where_it_is_working() {
        let prompt = prompt(false, false, false);

        assert!(prompt.contains("You are Claude, a software engineering agent running in Pebble."));
        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("Working directory: /home/test"));
        assert!(prompt.contains("Platform: linux"));
    }

    #[test]
    fn the_prompt_keeps_the_sections_this_harness_was_trained_on() {
        let prompt = prompt(false, false, false);

        for heading in [
            "# Harness",
            "# Delivering work",
            "# Working in the codebase",
            "# Tool use",
            "# Communicating with the user",
            "# Context management",
        ] {
            assert!(prompt.contains(heading), "the prompt lost {heading}");
        }
        assert!(prompt.contains("This workspace tracks reads before writes"));
        assert!(prompt.contains("`timeout` is measured in milliseconds"));
    }

    /// Every conditional section, against every combination of the three
    /// capabilities that gate them. Two snapshots pin the wording; this pins
    /// that each section appears exactly when its tool is registered.
    ///
    /// The question section is gated on the registry rather than on how the
    /// harness was built, which is what keeps a child — it runs its parent's
    /// profile and has nobody to ask — from being told to ask.
    #[test]
    fn the_prompt_sections_track_the_tools_the_session_has() {
        for has_web_search in [false, true] {
            for has_subagents in [false, true] {
                for has_question in [false, true] {
                    let prompt = prompt(has_web_search, has_subagents, has_question);
                    let label = format!(
                        "web_search={has_web_search} subagents={has_subagents} \
                         question={has_question}"
                    );

                    assert_eq!(
                        prompt.contains("Use `WebSearch`"),
                        has_web_search,
                        "{label}"
                    );
                    assert_eq!(
                        prompt.contains("# Background agents"),
                        has_subagents,
                        "{label}"
                    );
                    assert_eq!(
                        prompt.contains("# Asking the user"),
                        has_question,
                        "{label}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_prompt_names_the_search_tool_the_way_the_registry_will() {
        let searching = profile(true);

        assert!(prompt(true, false, false).contains(web_search_name(&searching)));
    }

    #[test]
    fn the_prompt_carries_memory_and_user_instructions() {
        let profile = profile(false);
        let prompt = profile.build_system_prompt(
            &registry_of(&profile, Vec::new()),
            &snapshot_context(),
            &["# Project README".to_owned()],
            Some("Always write tests first"),
            &[],
        );

        assert!(prompt.contains("# Project README"));
        assert!(prompt.contains("# User Instructions\nAlways write tests first"));
    }

    #[test]
    fn a_session_with_no_factory_is_given_no_background_agent_tools() {
        assert!(
            profile(false)
                .subagent_tools(&SubagentSupport::default())
                .is_empty()
        );
    }

    #[test]
    fn default_prompt_snapshot() {
        insta::assert_snapshot!(prompt(false, false, false));
    }

    #[test]
    fn all_conditionals_prompt_snapshot() {
        insta::assert_snapshot!(prompt(true, true, true));
    }
}
