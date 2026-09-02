//! The harness the GPT-5.6 models (Sol, Terra, Luna) expect.
//!
//! These models were trained against Codex, whose core tool set is far
//! narrower than what pebble offers the other OpenAI models: a shell,
//! `apply_patch`, and `update_plan`, plus web search when the application has
//! an engine. Codex has no dedicated file-read, file-write, grep, glob, or
//! fetch tool — reading and searching local files go through the shell, and
//! writes go through the patch tool. OpenAI-compatible gateways cannot carry
//! that freeform grammar, so those routes receive pebble's JSON-schema
//! `edit_file` instead.
//!
//! One deliberate difference from Codex: Codex drives 5.6 in *code mode*,
//! exposing a single `exec` tool that takes JavaScript and reaching every other
//! tool through a `tools` object inside a V8 isolate. Pebble calls tools
//! directly, so this profile matches Codex's tool *contract* — names,
//! parameters, and guidance — without that indirection.

use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde_json::{Value, json};

use super::{EmbeddedPrompt, FileEditToolKind, ProfileDeps, assemble_system_prompt};
use crate::config::NativeToolOptions;
use crate::profile::{AgentProfile, EnvContext};
use crate::skills::Skill;
use crate::tool::{NativeTool, RegisteredTool, ToolRegistry, ToolVocabulary, required_str};
use crate::tools::shell::run_shell_command;
use crate::tools::{TodoRuntime, make_update_plan_tool, make_web_search_tool};
use crate::types::{AgentProfileKind, ToolSource};

/// The system prompt this harness starts from.
const CORE_PROMPT: &str = include_str!("prompts/gpt56.md.j2");

/// Codex's three tools, and nothing else.
pub(crate) struct Gpt56Profile {
    tools:                 Vec<RegisteredTool>,
    provider_display_name: String,
    file_edit_tool:        FileEditToolKind,
}

impl Gpt56Profile {
    /// The harness for a session built from `deps`.
    pub(crate) fn new(deps: &ProfileDeps) -> Self {
        let options = NativeToolOptions::for_profile(AgentProfileKind::Gpt56);
        let mut tools = vec![
            make_shell_command_tool(&options, deps.file_edit_tool),
            deps.file_edit_tool.tool(),
            // Codex's `update_plan` replaces a whole plan at once, so the list
            // behind it belongs to this session rather than to the tree.
            make_update_plan_tool(Arc::new(TodoRuntime::new())),
        ];
        if let Some(provider) = &deps.search_provider {
            tools.push(make_web_search_tool(Arc::clone(provider)));
        }

        Self {
            tools,
            provider_display_name: deps.provider_display_name.clone(),
            file_edit_tool: deps.file_edit_tool,
        }
    }
}

/// Codex's `shell_command`: a shell script plus an explicit `workdir`.
///
/// Pebble's own `shell` tool has no `workdir`, and its description steers the
/// model toward the dedicated read and search tools. Neither fits here: 5.6 has
/// no dedicated tools to steer toward, and Codex tells it to set `workdir`
/// rather than to `cd`.
fn shell_command_description(
    default_timeout_ms: u64,
    max_timeout_ms: u64,
    file_edit_tool: FileEditToolKind,
) -> String {
    let file_edit_tool = file_edit_tool.as_str();
    format!(
        "Runs a shell command and returns its output.
- Always set the `workdir` param rather than using `cd`.
- Reading and searching files goes through this tool: prefer `rg` and \
`rg --files`, which are much faster than alternatives like `grep` and `find`.
- Use `{file_edit_tool}` to edit files, not `cat`, heredocs, or other shell write tricks.
- `timeout_ms` defaults to {default_timeout_ms} ms and is capped at {max_timeout_ms} ms. A command \
that timed out once will time out again, so raise the timeout rather than retrying."
    )
}

/// The shell this harness offers, describing the editor it was built beside.
///
/// The two arguments have to agree: the description points at the file editor
/// by name, so a shell built for one route and a patch tool from another would
/// name a tool the model was never given. Fabro registered a shell, swapped
/// the editor, and then re-described the shell to catch up; here both come from
/// one answer.
fn make_shell_command_tool(
    options: &NativeToolOptions,
    file_edit_tool: FileEditToolKind,
) -> RegisteredTool {
    let default_timeout_ms = options.default_command_timeout_ms;
    let max_timeout_ms = options.max_command_timeout_ms;

    RegisteredTool::new(ToolDefinition::function(
            // The canonical identity; the registry renames it to
            // `shell_command` for the Codex vocabulary.
            NativeTool::Shell.canonical_name(),
            shell_command_description(default_timeout_ms, max_timeout_ms, file_edit_tool),
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash source to evaluate, run by a non-login Bash shell."
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory for the command. Defaults to the turn cwd."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": format!(
                            "Maximum command runtime. Defaults to {default_timeout_ms} ms."
                        )
                    }
                },
                "required": ["command"]
            }),
        ), Arc::new(move |arguments, context| {
            Box::pin(async move {
                let command = required_str(&arguments, "command")?;
                let workdir = arguments.get("workdir").and_then(Value::as_str);
                let timeout_ms = arguments
                    .get("timeout_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(default_timeout_ms)
                    .min(max_timeout_ms);

                run_shell_command(&context, command, timeout_ms, workdir).await
            })
        })).with_source(ToolSource::Native)
}

impl AgentProfile for Gpt56Profile {
    fn profile_kind(&self) -> AgentProfileKind {
        AgentProfileKind::Gpt56
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
        let template = EmbeddedPrompt::new("gpt56.md.j2", CORE_PROMPT)
            .with_string("provider_name", self.provider_display_name.clone())
            .with_string("file_edit_tool", self.file_edit_tool.as_str())
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
        advertises, search_provider, shell_timeout_ms, snapshot_context, system_prompt, tool_names,
        web_search_name,
    };
    use crate::tool::ToolRegistry;
    use crate::tools::testing::schema_of;

    /// The harness a session on `file_edit_tool` with `has_web_search` gets,
    /// answering for `provider_display_name`.
    fn profile(
        file_edit_tool: FileEditToolKind,
        has_web_search: bool,
        provider_display_name: &str,
    ) -> Gpt56Profile {
        Gpt56Profile::new(&ProfileDeps {
            provider_display_name: provider_display_name.to_owned(),
            file_edit_tool,
            search_provider: search_provider(has_web_search),
            ..ProfileDeps::default()
        })
    }

    /// 5.6 reached over the OpenAI Responses codec, which carries the patch
    /// grammar.
    fn patching(has_web_search: bool) -> Gpt56Profile {
        profile(FileEditToolKind::ApplyPatch, has_web_search, "OpenAI")
    }

    /// 5.6 reached through a gateway, which cannot.
    fn editing(has_web_search: bool) -> Gpt56Profile {
        profile(FileEditToolKind::EditFile, has_web_search, "OpenRouter")
    }

    /// What `profile` calls its shell, and what it says about it.
    fn shell_description(profile: &Gpt56Profile) -> String {
        let mut registry = ToolRegistry::with_vocabulary(profile.tool_vocabulary());
        for tool in profile.base_tools() {
            registry.register(tool);
        }
        registry
            .get("shell_command")
            .expect("the harness offers Codex's shell")
            .definition
            .description
            .clone()
    }

    #[test]
    fn the_profile_names_its_harness_and_codexs_own_vocabulary() {
        let profile = patching(false);

        assert_eq!(profile.profile_kind(), AgentProfileKind::Gpt56);
        assert_eq!(profile.tool_vocabulary(), ToolVocabulary::Codex);
    }

    /// The whole point of the harness: 5.6 sees Codex's tools and nothing else.
    #[test]
    fn the_harness_offers_only_codexs_three_tools() {
        assert_eq!(tool_names(&patching(false)), [
            "apply_patch",
            "shell",
            "update_plan",
        ]);
    }

    #[test]
    fn the_shell_reaches_the_model_under_codexs_name() {
        let mut registry = ToolRegistry::with_vocabulary(ToolVocabulary::Codex);
        for tool in patching(false).base_tools() {
            registry.register(tool);
        }
        let mut names = registry.names();
        names.sort();

        assert_eq!(names, ["apply_patch", "shell_command", "update_plan"]);
    }

    #[test]
    fn the_harness_leaves_out_everything_codex_does_not_have() {
        let profile = patching(false);

        for absent in [
            "read_file",
            "write_file",
            "edit_file",
            "grep",
            "glob",
            "web_fetch",
            "list_dir",
            "read_many_files",
            "TaskCreate",
            "web_search",
        ] {
            assert!(
                !advertises(&profile, absent),
                "5.6 should not be given {absent}"
            );
        }
    }

    /// The `openai_compatible` codec rejects a custom tool definition outright,
    /// so a freeform patch tool on that route fails every request. 5.6 is
    /// served through OpenRouter, which uses exactly that codec.
    #[test]
    fn a_route_that_cannot_carry_a_patch_grammar_gets_the_json_editor_instead() {
        let profile = editing(false);

        assert!(advertises(&profile, "edit_file"));
        assert!(!advertises(&profile, "apply_patch"));
        for tool in profile.base_tools() {
            assert!(
                !tool.definition.is_custom(),
                "{} must not be a custom definition on a gateway route",
                tool.definition.name
            );
        }
    }

    #[test]
    fn the_patch_tool_stays_a_freeform_grammar() {
        let patch = patching(false)
            .base_tools()
            .into_iter()
            .find(|tool| tool.definition.name == "apply_patch")
            .expect("the harness offers a patch tool");

        assert!(patch.definition.is_custom());
    }

    #[test]
    fn the_shell_takes_a_workdir_because_each_call_is_its_own_process() {
        let shell = patching(false)
            .base_tools()
            .into_iter()
            .find(|tool| tool.definition.name == "shell")
            .expect("the harness offers a shell");
        let schema = schema_of(&shell);

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["workdir"].is_object());
        assert_eq!(schema["required"], json!(["command"]));
        assert_eq!(
            schema["properties"]["command"]["description"],
            "Bash source to evaluate, run by a non-login Bash shell."
        );
    }

    /// The shell names the file editor, so it has to follow the route or it
    /// points 5.6 at a tool it was never given.
    #[test]
    fn the_shell_description_names_the_editor_the_route_actually_carries() {
        let patching = shell_description(&patching(false));
        assert!(patching.contains("`apply_patch`"));
        assert!(!patching.contains("`edit_file`"));

        let editing = shell_description(&editing(false));
        assert!(editing.contains("`edit_file`"));
        assert!(!editing.contains("`apply_patch`"));
    }

    #[tokio::test]
    async fn a_command_with_no_timeout_gets_the_ten_seconds_codex_documents() {
        assert_eq!(shell_timeout_ms(&patching(false)).await, 10_000);
    }

    #[test]
    fn the_prompt_names_the_provider_the_catalog_names() {
        assert!(system_prompt(&patching(false)).contains("powered by OpenAI"));
        assert!(system_prompt(&editing(false)).contains("powered by OpenRouter"));
    }

    #[test]
    fn the_prompt_describes_the_editor_the_route_actually_carries() {
        let patching = system_prompt(&patching(false));
        assert!(patching.contains("Use `apply_patch` for local file edits"));
        assert!(patching.contains("*** Begin Patch"));
        assert!(!patching.contains("edit_file"));

        let editing = system_prompt(&editing(false));
        assert!(editing.contains("Use `edit_file` for local file edits"));
        assert!(!editing.contains("apply_patch"));
        assert!(!editing.contains("*** Begin Patch"));
    }

    #[test]
    fn the_prompt_names_the_shell_tool_as_codex_does() {
        let prompt = system_prompt(&patching(false));

        assert!(prompt.contains("shell_command"));
        // `grep`, `find` and `glob` are deliberately absent from this list:
        // the prompt names them as shell programs, which is what Codex does,
        // rather than as tools pebble registers.
        for absent in ["read_file", "write_file", "edit_file", "web_fetch"] {
            assert!(
                !prompt.contains(absent),
                "the prompt should not mention {absent}"
            );
        }
    }

    #[test]
    fn the_prompt_keeps_codexs_own_guidance() {
        let prompt = system_prompt(&patching(false));

        for heading in [
            "# Personality",
            "# Working with the user",
            "# AGENTS.md",
            "# Rules for getting work done",
            "# Destructive actions",
        ] {
            assert!(prompt.contains(heading), "the prompt lost {heading}");
        }
        assert!(prompt.contains("Set the `workdir` parameter on `shell_command`"));
        assert!(prompt.contains("Never repurpose `$HOME`, `$home`, or `$PEBBLE_HOME`"));
    }

    /// The attribution comment renders as nothing, so what the model reads
    /// starts at the identity line.
    #[test]
    fn the_apache_attribution_never_reaches_the_model() {
        let prompt = system_prompt(&patching(false));

        assert!(prompt.starts_with("You are a coding agent powered by OpenAI"));
        assert!(!prompt.contains("Apache-2.0"));
        assert!(CORE_PROMPT.contains("Adapted from openai/codex (Apache-2.0)"));
    }

    #[test]
    fn the_prompt_mentions_a_search_tool_exactly_when_the_session_has_one() {
        let without = system_prompt(&patching(false));
        let with = system_prompt(&patching(true));

        assert!(!without.contains(web_search_name(&patching(false))));
        assert!(with.contains("use `web_search` rather than guessing"));
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

        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("Platform: linux"));
        assert!(prompt.contains("# Project README"));
        assert!(prompt.contains("# User Instructions\nAlways write tests first"));
    }

    #[test]
    fn default_prompt_snapshot() {
        insta::assert_snapshot!(system_prompt(&patching(false)));
    }

    #[test]
    fn web_search_prompt_snapshot() {
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
