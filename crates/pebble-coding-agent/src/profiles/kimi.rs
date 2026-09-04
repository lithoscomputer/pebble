//! The harness Kimi Code's models expect.

use std::sync::Arc;

use super::{
    EmbeddedPrompt, ProfileDeps, assemble_system_prompt, discovery_and_web_tools, kimi_tools,
};
use crate::config::NativeToolOptions;
use crate::profile::{AgentProfile, EnvContext};
use crate::skills::Skill;
use crate::tool::{NativeTool, RegisteredTool, ToolRegistry, ToolVocabulary};
use crate::tools::{TodoRuntime, make_todo_list_tool};
use crate::types::AgentProfileKind;

/// The system prompt this harness starts from.
const CORE_PROMPT: &str = include_str!("prompts/kimi.md.j2");

/// Kimi models repeatedly reconstruct `old_string` from memory rather than from
/// a fresh read: across two observed K3 implementation stages, 32 of 35 tool
/// failures were edits against a file the model had not read, or `old_string`
/// values recalled from an earlier version. Kimi Code carries this guidance in
/// its tool descriptions and nowhere in its system prompt, so this profile does
/// the same — the rule lands in the description of the tool being called.
const EDIT_FILE_DESCRIPTION: &str = "Perform exact replacements in existing files.

- Edit is mandatory for every incremental change, especially small edits. DO NOT use Write or \
Bash `sed`.
- Read the target file before every Edit. DO NOT call Edit from memory, stale context, or a \
guessed `old_string`.
- Take `old_string` and `new_string` from the Read output view, dropping the line-number prefix \
and separator; match only file content.
- `old_string` must be unique unless `replace_all` is set. If it is ambiguous, add surrounding \
context. Use `replace_all` only when every occurrence should change — for example, renaming a \
symbol throughout the file.
- DO NOT issue consecutive Edit calls on the same file. A previous Edit can invalidate a later \
Edit's `old_string`, causing `old_string not found`. Read the file again before the next Edit.
- If an Edit fails with `old_string not found`, re-read the file and take the exact text from the \
fresh output rather than guessing again.
- Preserve existing indentation.";

/// Kimi Code's own `Glob` wording, which is longer and more prescriptive than
/// pebble's because this model family reaches for `find` otherwise.
const GLOB_DESCRIPTION: &str = "Find files by search-root-relative path using a glob pattern. \
Results are sorted lexicographically by relative path.

Use this instead of `find` or recursive `ls` through Bash. Prefer patterns with a literal anchor \
— an extension or a subdirectory — over bare wildcards.

Good patterns:
- `*.rs` — direct children of the search root
- `**/*.rs` — files at any depth below the search root
- `src/*.rs` — directly inside `src/`, not recursive
- `src/**/*.rs` — recursive walk under a subdirectory
- `src/[lm]ib.rs` — a bracket expression matches one character

Avoid recursing into dependency or build output (`node_modules/**`, `target/**`): those produce \
thousands of matches and waste context. Narrow to a specific subpath instead. Results are files, \
so to locate a directory, glob for something inside it. Patterns must use `/`, be relative, and \
cannot contain a `..` segment.";

/// Kimi Code's tool names, its five differently shaped tools, and a
/// replace-whole-list todo surface.
///
/// The prompt has no conditionals at all: Kimi Code keeps read-before-edit
/// mechanics and search guidance in the tool descriptions, and this profile
/// follows that split. What the session can do therefore reaches the model
/// through the tool list alone.
pub(crate) struct KimiProfile {
    tools: Vec<RegisteredTool>,
}

impl KimiProfile {
    /// The harness for a session built from `deps`.
    pub(crate) fn new(deps: &ProfileDeps) -> Self {
        let options = NativeToolOptions::for_profile(AgentProfileKind::Kimi);

        // Finding files by name and reading one off the web have the same
        // contract in both vocabularies, so the rename is all they need. The
        // rest are adapters, reusing pebble's execution where the behavior
        // agrees.
        let mut tools = discovery_and_web_tools(
            deps.search_provider.clone(),
            deps.web_fetch_summarizer.clone(),
        );
        for tool in &mut tools {
            if tool.definition.name == NativeTool::Glob.canonical_name() {
                GLOB_DESCRIPTION.clone_into(&mut tool.definition.description);
            }
        }

        tools.push(kimi_tools::make_kimi_read_tool());
        tools.push(kimi_tools::make_kimi_write_tool());
        tools.push(kimi_tools::make_kimi_edit_tool(EDIT_FILE_DESCRIPTION));
        tools.push(kimi_tools::make_kimi_grep_tool());
        tools.push(kimi_tools::make_kimi_bash_tool(
            options.default_command_timeout_ms,
            options.max_command_timeout_ms,
        ));

        // Kimi Code drives todos with one replace-whole-list call. The
        // incremental task tools model the opposite interaction — mutation
        // against tracked ids — so they are the wrong surface here even though
        // both persist through the same runtime.
        tools.push(make_todo_list_tool(Arc::new(TodoRuntime::new())));

        Self { tools }
    }
}

impl AgentProfile for KimiProfile {
    fn profile_kind(&self) -> AgentProfileKind {
        AgentProfileKind::Kimi
    }

    fn tool_vocabulary(&self) -> ToolVocabulary {
        ToolVocabulary::KimiCode
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
        assemble_system_prompt(
            EmbeddedPrompt::new("kimi.md.j2", CORE_PROMPT),
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
    };
    use crate::tool::{ToolRegistry, known_tool_category, tool_category};
    use crate::types::ToolCategory;

    /// The harness a session with `has_web_search` gets.
    fn profile(has_web_search: bool) -> KimiProfile {
        KimiProfile::new(&ProfileDeps {
            search_provider: search_provider(has_web_search),
            ..ProfileDeps::default()
        })
    }

    /// The registry a session on this harness builds, so the names under test
    /// are the ones the model reads.
    fn registry(profile: &KimiProfile) -> ToolRegistry {
        let mut registry = ToolRegistry::with_vocabulary(profile.tool_vocabulary());
        for tool in profile.base_tools() {
            registry
                .register(tool)
                .expect("tool registration is unique");
        }
        registry
    }

    /// What `profile` says about the tool exposed as `name`.
    fn describe(profile: &KimiProfile, name: &str) -> String {
        registry(profile)
            .get(name)
            .unwrap_or_else(|| panic!("the harness offers {name}"))
            .definition
            .description
            .clone()
    }

    #[test]
    fn the_profile_names_its_harness_and_kimi_codes_own_vocabulary() {
        let profile = profile(false);

        assert_eq!(profile.profile_kind(), AgentProfileKind::Kimi);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::KimiCode);
    }

    #[test]
    fn the_harness_offers_eight_tools() {
        assert_eq!(tool_names(&profile(false)), [
            "TodoList",
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
    fn the_tools_reach_the_model_under_kimi_codes_names() {
        let mut names = registry(&profile(false)).names();
        names.sort();

        assert_eq!(names, [
            "Bash", "Edit", "FetchURL", "Glob", "Grep", "Read", "TodoList", "Write",
        ]);
        for canonical in [
            "read_file",
            "write_file",
            "edit_file",
            "shell",
            "grep",
            "glob",
            "web_fetch",
        ] {
            assert!(
                !names.contains(&canonical.to_owned()),
                "{canonical} should have been renamed"
            );
        }
    }

    /// A tool registered after the profile contributed its own — the skill
    /// tool, the subagent tools — must land in the same vocabulary, or the
    /// model sees a mixed-case tool set.
    #[test]
    fn a_tool_registered_later_also_gets_a_kimi_code_name() {
        use crate::skills::Skill;
        use crate::tools::skill::make_use_skill_tool_for_vocabulary;

        let mut registry = registry(&profile(false));
        registry
            .register(make_use_skill_tool_for_vocabulary(
                Arc::from(vec![Skill {
                    name:        "demo".to_owned(),
                    description: "d".to_owned(),
                    template:    "t".to_owned(),
                }]),
                ToolVocabulary::KimiCode,
            ))
            .expect("tool registration is unique");

        let names = registry.names();
        assert!(names.contains(&"Skill".to_owned()), "{names:?}");
        assert!(!names.contains(&"use_skill".to_owned()), "{names:?}");
    }

    /// The rename must not change what a tool is allowed to do. An exposed
    /// name that fails to resolve would fall back to `Shell` in an
    /// application's gate, silently demanding approval for reads.
    #[test]
    fn a_renamed_tool_keeps_its_permission_category() {
        for name in registry(&profile(true)).names() {
            let tool = NativeTool::from_any_name(&name).unwrap_or_else(|| {
                panic!("the harness offers a tool pebble does not know: {name}")
            });
            assert_eq!(
                known_tool_category(&name),
                tool.category(),
                "the exposed name `{name}` must categorize as its canonical identity"
            );
        }
        // The specific regression: reads stay reads, not shell commands.
        assert_eq!(tool_category("Read"), ToolCategory::Read);
        assert_eq!(tool_category("Bash"), ToolCategory::Shell);
    }

    #[test]
    fn the_harness_leaves_out_the_tools_other_harnesses_own() {
        let profile = profile(false);

        assert!(!advertises(&profile, "TaskCreate"));
        assert!(!advertises(&profile, "update_plan"));
        assert!(!advertises(&profile, "apply_patch"));
        assert!(!advertises(&profile, "read_many_files"));
        assert!(!advertises(&profile, "list_dir"));
        assert!(!advertises(&profile, "web_search"));
    }

    #[tokio::test]
    async fn a_command_with_no_timeout_gets_the_minute_kimi_code_documents() {
        assert_eq!(shell_timeout_ms(&profile(false)).await, 60_000);
    }

    /// Kimi Code carries read-before-edit guidance in the tool descriptions and
    /// nowhere in its system prompt, so this is where it has to land.
    #[test]
    fn the_edit_and_write_descriptions_drill_reading_first() {
        let profile = profile(false);

        for name in ["Edit", "Write"] {
            assert!(
                describe(&profile, name).contains("Read"),
                "{name} should steer the model to read the file first"
            );
        }
        assert!(
            describe(&profile, "Edit")
                .contains("DO NOT call Edit from memory, stale context, or a guessed")
        );
        assert!(
            describe(&profile, "Edit")
                .contains("DO NOT issue consecutive Edit calls on the same file")
        );
        assert!(describe(&profile, "Write").contains("Read before overwriting an existing file"));
        // Re-reading only to confirm a write landed is waste, not diligence.
        assert!(
            describe(&profile, "Read").contains("do not re-read solely to prove the write landed")
        );
        // Read tells the model how to turn its output into an Edit old_string.
        assert!(describe(&profile, "Read").contains("Drop the number and separator"));
    }

    #[test]
    fn the_shell_description_maps_every_command_to_the_tool_that_replaces_it() {
        let bash = describe(&profile(false), "Bash");

        for expected in ["→ Read", "→ Edit", "→ Write", "→ Glob", "→ Grep"] {
            assert!(bash.contains(expected), "Bash should map {expected}");
        }
        // Bash takes SECONDS, unlike pebble's millisecond built-in. The
        // seconds value is quoted and the raw millisecond value is not, which
        // is what a unit bug would look like.
        let options = NativeToolOptions::for_profile(AgentProfileKind::Kimi);
        let seconds = (options.default_command_timeout_ms / 1000).to_string();
        assert!(
            bash.contains(&seconds),
            "Bash should quote {seconds}s: {bash}"
        );
        assert!(
            !bash.contains(&options.default_command_timeout_ms.to_string()),
            "Bash quotes milliseconds, so the unit conversion is wrong: {bash}"
        );
        assert!(bash.contains("SECONDS"), "{bash}");
        // Pebble has no background shell; promising one would be a lie.
        assert!(!bash.contains("run_in_background"), "{bash}");
    }

    #[test]
    fn the_search_descriptions_promise_only_what_the_environment_delivers() {
        let profile = profile(false);

        // Grep must not promise ripgrep syntax: an environment may fall back
        // to POSIX grep.
        assert!(describe(&profile, "Grep").contains("POSIX"));
        assert!(describe(&profile, "Glob").contains("sorted lexicographically"));
        assert!(describe(&profile, "Glob").contains("`*.rs` — direct children"));
    }

    #[test]
    fn the_edit_schema_names_its_target_the_way_kimi_code_does() {
        use crate::tools::testing::schema_of;

        let registry = registry(&profile(false));
        let edit = registry.get("Edit").expect("the harness offers an editor");
        let schema = schema_of(edit);

        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("file_path").is_none());
        assert_eq!(
            schema["required"],
            serde_json::json!(["path", "old_string", "new_string"])
        );
    }

    #[test]
    fn the_prompt_says_who_the_model_is_and_where_it_is_working() {
        let prompt = system_prompt(&profile(false));

        assert!(prompt.contains("You are Kimi"));
        assert!(prompt.contains("# Tracking Multi-Step Work"));
        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("Working directory: /home/test"));
        assert!(prompt.contains("Platform: linux"));
    }

    /// This harness's prompt has no conditionals at all, so a session that can
    /// search reads exactly the same prompt as one that cannot; the tool list
    /// is what tells it.
    #[test]
    fn the_prompt_never_mentions_a_search_tool_either_way() {
        let without = system_prompt(&profile(false));
        let with = system_prompt(&profile(true));

        assert_eq!(without, with);
        assert!(!with.contains("WebSearch"));
        // Read-before-edit mechanics stay in the tool descriptions.
        assert!(!with.contains("Reading Before Writing"));
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
}
