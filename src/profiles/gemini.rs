//! The harness Google's Gemini coding models expect.

use super::{EmbeddedPrompt, ProfileDeps, assemble_system_prompt, core_tools};
use crate::config::NativeToolOptions;
use crate::profile::{AgentProfile, EnvContext};
use crate::skills::Skill;
use crate::tool::{RegisteredTool, ToolRegistry, ToolVocabulary};
use crate::tools::{make_edit_file_tool, make_list_dir_tool, make_read_many_files_tool};
use crate::types::AgentProfileKind;

/// The system prompt this harness starts from.
const CORE_PROMPT: &str = include_str!("prompts/gemini.md.j2");

/// Pebble's own tool names, an exact-string file editor, and the two tools
/// only this harness is given.
///
/// Gemini CLI reads a batch of files in one call and lists a directory
/// outright, and its prompt tells the model to prefer both over reading files
/// one at a time. It also keeps no plan of its own and never asks a person a
/// question: the session builder registers no question tool for this harness,
/// because Gemini CLI has none.
pub(crate) struct GeminiProfile {
    tools:          Vec<RegisteredTool>,
    has_web_search: bool,
}

impl GeminiProfile {
    /// The harness for a session built from `deps`.
    pub(crate) fn new(deps: &ProfileDeps) -> Self {
        let options = NativeToolOptions::for_profile(AgentProfileKind::Gemini);
        let mut tools = core_tools(&options, deps.web_fetch_summarizer.clone());
        tools.push(make_edit_file_tool());
        tools.push(make_read_many_files_tool());
        tools.push(make_list_dir_tool());

        Self {
            tools,
            has_web_search: deps.has_web_search(),
        }
    }
}

impl AgentProfile for GeminiProfile {
    fn profile_kind(&self) -> AgentProfileKind {
        AgentProfileKind::Gemini
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
        let template = EmbeddedPrompt::new("gemini.md.j2", CORE_PROMPT)
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
    use crate::tools::make_question_tool;

    /// The harness a session with `has_web_search` gets.
    fn profile(has_web_search: bool) -> GeminiProfile {
        GeminiProfile::new(&ProfileDeps {
            search_provider: search_provider(has_web_search),
            ..ProfileDeps::default()
        })
    }

    #[test]
    fn the_profile_names_its_harness_and_pebbles_own_vocabulary() {
        let profile = profile(false);

        assert_eq!(profile.profile_kind(), AgentProfileKind::Gemini);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::Canonical);
    }

    #[test]
    fn the_harness_offers_nine_tools() {
        assert_eq!(tool_names(&profile(false)), [
            "edit_file",
            "glob",
            "grep",
            "list_dir",
            "read_file",
            "read_many_files",
            "shell",
            "web_fetch",
            "write_file",
        ]);
    }

    #[test]
    fn this_is_the_only_harness_that_reads_many_files_and_lists_a_directory() {
        let profile = profile(false);

        assert!(advertises(&profile, "read_many_files"));
        assert!(advertises(&profile, "list_dir"));
        assert!(!advertises(&profile, "update_plan"));
        assert!(!advertises(&profile, "TaskCreate"));
        assert!(!advertises(&profile, "apply_patch"));
        assert!(!advertises(&profile, "web_search"));
    }

    #[test]
    fn this_harness_has_no_question_tool_at_all() {
        // Gemini CLI has no way to ask a person a question, so a session on
        // this harness registers none however it was configured.
        assert!(make_question_tool(AgentProfileKind::Gemini).is_none());
    }

    #[tokio::test]
    async fn a_command_with_no_timeout_gets_the_ten_seconds_the_prompt_promises() {
        assert_eq!(shell_timeout_ms(&profile(false)).await, 10_000);
        assert!(system_prompt(&profile(false)).contains("Default timeout is 10 seconds"));
    }

    #[test]
    fn the_prompt_says_who_the_model_is_and_what_it_is_for() {
        let prompt = system_prompt(&profile(false));

        assert!(prompt.contains("You are Gemini CLI"));
        assert!(prompt.contains("solving bugs"));
        assert!(prompt.contains("adding new functionality"));
        assert!(prompt.contains("refactoring code"));
        assert!(prompt.contains("explaining code"));
        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("Working directory: /home/test"));
        assert!(prompt.contains("Platform: linux"));
    }

    #[test]
    fn the_prompt_documents_every_tool_this_harness_offers() {
        let prompt = system_prompt(&profile(false));

        for tool in tool_names(&profile(false)) {
            assert!(
                prompt.contains(&format!("## {tool}")),
                "the prompt lost {tool}"
            );
        }
    }

    #[test]
    fn the_prompt_keeps_geminis_own_conventions() {
        let prompt = system_prompt(&profile(false));

        assert!(prompt.contains("GEMINI.md"));
        assert!(prompt.contains("AGENTS.md"));
        assert!(prompt.contains("clean, maintainable code"));
        assert!(prompt.contains("Handle errors appropriately"));
        assert!(prompt.contains("existing code conventions"));
        assert!(prompt.contains("Research -> Strategy -> Execution"));
    }

    #[test]
    fn the_prompt_mentions_a_search_tool_exactly_when_the_session_has_one() {
        let without = system_prompt(&profile(false));
        let with = system_prompt(&profile(true));

        assert!(!without.contains(web_search_name(&profile(false))));
        assert!(with.contains("## web_search"));
        assert!(with.contains("Search the web for information."));
    }

    #[test]
    fn the_prompt_carries_memory_and_user_instructions() {
        let prompt = profile(false).build_system_prompt(
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
    fn default_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&profile(false)));
    }

    #[test]
    fn web_search_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&profile(true)));
    }
}
