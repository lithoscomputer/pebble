//! The harnesses pebble ships.
//!
//! Each module here implements [`AgentProfile`](crate::state::AgentProfile)
//! for one model family: the tools that family was trained to call, the names
//! it expects them under, and a checked-in system prompt written the way it was
//! trained to read one. A session never names a profile — it names a model, and
//! [`builtin_profile`](crate::profile::builtin_profile) turns the catalog's
//! `metadata.pebble.profile` into the harness that model expects.
//!
//! What is shared sits here: the prompt assembly every profile ends with, the
//! environment block every prompt embeds, the tools every profile starts from,
//! and the one route-dependent choice a profile cannot make on its own — which
//! file editor the wire codec can carry.
//!
//! # Contributing, not owning
//!
//! Fabro's profiles owned a registry and mutated it; whoever wanted a tool
//! registered reached in. Pebble's builder owns the registry and a profile
//! contributes to it, so what a profile can say about the session is exactly
//! what it returns. Optional prompt sections read the completed registry. This
//! keeps a prompt from advertising a search, subagent, or question tool the
//! session does not have — including in a child, which shares its parent's
//! profile but has its own registry.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use lithos_llm::catalog::{CodecId, codec_ids};
use lithos_llm::types::ToolDefinition;
use serde_json::Value;

use crate::config::NativeToolOptions;
use crate::profile::EnvContext;
use crate::search::SearchProvider;
use crate::skills::{Skill, format_skills_prompt_section};
use crate::template::{TemplateContext, TemplateValue, render_named};
use crate::tool::{NativeTool, RegisteredTool, ToolVocabulary};
use crate::tools::{
    WebFetchSummarizer, make_apply_patch_tool, make_edit_file_tool, make_glob_tool, make_grep_tool,
    make_read_file_tool, make_shell_tool_with_options, make_web_fetch_tool, make_web_search_tool,
    make_write_file_tool,
};

pub(crate) mod anthropic;
pub(crate) mod claude5;
pub(crate) mod claude5_tools;
mod codex_tools;
pub(crate) mod gemini;
pub(crate) mod gpt56;
pub(crate) mod gpt6;
pub(crate) mod kimi;
pub(crate) mod kimi_tools;
pub(crate) mod openai;

pub(crate) use self::anthropic::AnthropicProfile;
pub(crate) use self::claude5::Claude5Profile;
pub(crate) use self::gemini::GeminiProfile;
pub(crate) use self::gpt6::Gpt6Profile;
pub(crate) use self::gpt56::Gpt56Profile;
pub(crate) use self::kimi::KimiProfile;
pub(crate) use self::openai::OpenAiProfile;

/// What a built-in profile is constructed from.
///
/// Everything here is decided by the session being built rather than by the
/// profile: what the route says about itself, and which of the optional
/// capabilities the application configured. A profile reads them once, at
/// construction, because a profile is shared and read-only afterwards —
/// including by every child session, which runs the same harness.
#[derive(Clone, Default)]
pub(crate) struct ProfileDeps {
    /// How the catalog names the provider answering, for a prompt that says
    /// who the model is from.
    pub(crate) provider_display_name: String,
    /// Which file editor this route's wire codec can carry.
    pub(crate) file_edit_tool:        FileEditToolKind,
    /// The engine a search tool runs through, when the application gave the
    /// session one.
    ///
    /// Carried rather than reduced to a flag because each profile contributes
    /// the search tool its model expects. Claude 5's `WebSearch`, for example,
    /// takes a query and nothing else.
    pub(crate) search_provider:       Option<Arc<dyn SearchProvider>>,
    /// The model that answers a `web_fetch` prompt about a page, when the
    /// application named one.
    ///
    /// Carried here so a profile's fetch tool captures it at construction, the
    /// way a search tool captures its engine — including by every child
    /// session, which runs the same profile.
    pub(crate) web_fetch_summarizer:  Option<Arc<WebFetchSummarizer>>,
}

impl fmt::Debug for ProfileDeps {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProfileDeps")
            .field("provider_display_name", &self.provider_display_name)
            .field("file_edit_tool", &self.file_edit_tool)
            .field(
                "search_provider",
                &self.search_provider.as_ref().map(|_| "<provider>"),
            )
            .field(
                "web_fetch_summarizer",
                &self.web_fetch_summarizer.as_ref().map(|_| "<summarizer>"),
            )
            .finish()
    }
}

/// Which file-editing tool a profile exposes.
///
/// `apply_patch` is a freeform grammar tool, and only the OpenAI Responses
/// codec can carry one: every other codec rejects a custom tool definition
/// outright. A model reached through a gateway such as OpenRouter therefore has
/// to be offered the JSON-schema `edit_file` instead, or every request it makes
/// fails.
///
/// Fabro keyed this on its own `CodecKind`; pebble reads the lithos catalog's
/// provider codec, which is where the same fact lives. The two spell the
/// identifier differently — fabro's `openai_responses`, lithos's
/// `openai-responses` — and lithos has no per-model codec, which is
/// behaviour-compatible because no fabro catalog row ever set one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum FileEditToolKind {
    /// Codex's freeform patch grammar.
    ApplyPatch,
    /// Pebble's JSON-schema exact-string editor, which every codec carries and
    /// which is therefore the answer for a route nothing is known about.
    #[default]
    EditFile,
}

impl FileEditToolKind {
    /// The editor a route speaking `codec` can carry.
    pub(crate) fn for_codec(codec: &CodecId) -> Self {
        if codec.as_str() == codec_ids::OPENAI_RESPONSES {
            Self::ApplyPatch
        } else {
            Self::EditFile
        }
    }

    /// What the model calls this editor.
    ///
    /// A template input as well as a tool name: `openai.md.j2` and
    /// `gpt56.md.j2` and `gpt6.md.j2` compare it against a string literal, so
    /// these two spellings are part of the prompt contract.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ApplyPatch => "apply_patch",
            Self::EditFile => "edit_file",
        }
    }

    /// The tool itself.
    pub(crate) fn tool(self) -> RegisteredTool {
        match self {
            Self::ApplyPatch => make_apply_patch_tool(),
            Self::EditFile => make_edit_file_tool(),
        }
    }
}

/// The tools every built-in profile starts from.
///
/// Fabro called this `register_core_tools`. Search is optional: the application
/// supplies the engine and the profile contributes the shape its model expects.
pub(crate) fn core_tools(
    options: &NativeToolOptions,
    search_provider: Option<Arc<dyn SearchProvider>>,
    web_fetch_summarizer: Option<Arc<WebFetchSummarizer>>,
) -> Vec<RegisteredTool> {
    let mut tools = vec![
        make_read_file_tool(),
        make_write_file_tool(),
        make_shell_tool_with_options(options),
        make_grep_tool(),
    ];
    tools.extend(discovery_and_web_tools(
        search_provider,
        web_fetch_summarizer,
    ));
    tools
}

/// Finding things by name, and reading one off the web.
///
/// Split out because the Kimi harness takes these and replaces the rest.
pub(crate) fn discovery_and_web_tools(
    search_provider: Option<Arc<dyn SearchProvider>>,
    web_fetch_summarizer: Option<Arc<WebFetchSummarizer>>,
) -> Vec<RegisteredTool> {
    let mut tools = vec![make_glob_tool(), make_web_fetch_tool(web_fetch_summarizer)];
    if let Some(provider) = search_provider {
        tools.push(make_web_search_tool(provider));
    }
    tools
}

/// A tool definition under `tool`'s canonical name, which the registry renames
/// into the session's vocabulary.
///
/// The harness modules that build a family's own tools go through this so the
/// name a tool is registered under always comes from [`NativeTool`]'s table
/// rather than from a literal beside it.
pub(crate) fn definition(
    tool: NativeTool,
    description: impl Into<String>,
    parameters: Value,
) -> ToolDefinition {
    ToolDefinition::function(tool.canonical_name(), description, parameters)
}

/// A checked-in system-prompt template and the inputs it reads.
///
/// The environment block is supplied by [`assemble_system_prompt`] and cannot
/// be set by a caller: a prompt that describes a different working directory
/// than the session's would be worse than one with no block at all.
pub(crate) struct EmbeddedPrompt {
    name:   &'static str,
    source: &'static str,
    inputs: BTreeMap<String, TemplateValue>,
}

impl EmbeddedPrompt {
    /// The template `source`, called `name` when it fails to render.
    pub(crate) fn new(name: &'static str, source: &'static str) -> Self {
        Self {
            name,
            source,
            inputs: BTreeMap::new(),
        }
    }

    /// Gives the template a text input.
    pub(crate) fn with_string(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.inputs
            .insert(name.to_owned(), TemplateValue::Text(value.into()));
        self
    }

    /// Gives the template a condition.
    pub(crate) fn with_bool(mut self, name: &'static str, value: bool) -> Self {
        self.inputs
            .insert(name.to_owned(), TemplateValue::Flag(value));
        self
    }

    /// The rendered template, with `env_block` as its last input.
    ///
    /// # Panics
    ///
    /// Panics when a checked-in template fails to render, which means it is
    /// malformed or names an input its profile does not supply. Every
    /// conditional variant of every template is rendered under test, so this is
    /// a programmer error rather than a runtime condition.
    fn render(mut self, env_block: String) -> String {
        self.inputs
            .insert("env_block".to_owned(), TemplateValue::Text(env_block));
        let context = TemplateContext::new().with_inputs(self.inputs);
        render_named(self.name, self.source, &context).unwrap_or_else(|error| {
            panic!("the embedded prompt template failed to render: {error}")
        })
    }
}

/// The memory text every profile appends to its system prompt.
pub(crate) fn memory_prompt_suffix<'a>(memory: impl IntoIterator<Item = &'a str>) -> String {
    let mut memory = memory.into_iter();
    let Some(first) = memory.next() else {
        return String::new();
    };

    let mut section = String::from("\n\n");
    section.push_str(first);
    for document in memory {
        section.push_str("\n\n");
        section.push_str(document);
    }
    section
}

/// The skills text every profile appends to its system prompt.
pub(crate) fn skills_prompt_suffix(skills: &[Skill], vocabulary: ToolVocabulary) -> String {
    let section = format_skills_prompt_section(skills, vocabulary);
    if section.is_empty() {
        String::new()
    } else {
        memory_prompt_suffix([section.as_str()])
    }
}

/// One complete system prompt: the profile's template, then the sections every
/// profile ends with.
///
/// Each trailing section is separated by a blank line and left out when it is
/// empty, so a session that loaded no memory and discovered no skills gets the
/// template and nothing more.
///
/// `vocabulary` is the profile's own — every caller passes
/// [`AgentProfile::tool_vocabulary`](crate::state::AgentProfile::tool_vocabulary) — so
/// the skills section names the skill tool the way the registry will register
/// it. Fabro set this on the template instead, where it could disagree with the
/// registry the same profile built.
pub(crate) fn assemble_system_prompt(
    template: EmbeddedPrompt,
    vocabulary: ToolVocabulary,
    env_context: &EnvContext,
    memory: &[String],
    user_instructions: Option<&str>,
    skills: &[Skill],
) -> String {
    let env_block = build_env_context_block(env_context);
    let prompt = template.render(env_block);

    let memory_section = memory_prompt_suffix(memory.iter().map(String::as_str));
    let skills_section = skills_prompt_suffix(skills, vocabulary);
    let user_section = match user_instructions {
        Some(instructions) => format!("\n\n# User Instructions\n{instructions}"),
        None => String::new(),
    };

    format!("{prompt}{memory_section}{skills_section}{user_section}")
}

/// What the prompt says about where the session is working.
///
/// Line order is part of the contract — the prompt snapshots pin it — and the
/// members that are not always known are simply left out rather than rendered
/// empty.
pub(crate) fn build_env_context_block(context: &EnvContext) -> String {
    let mut lines = vec![
        "<environment>".to_owned(),
        format!("Working directory: {}", context.working_directory),
        format!("Is git repository: {}", context.is_git_repo),
    ];

    if let Some(branch) = &context.git_branch {
        lines.push(format!("Git branch: {branch}"));
    }

    lines.push(format!("Platform: {}", context.platform));
    lines.push(format!("OS version: {}", context.os_version));

    if !context.current_date.is_empty() {
        lines.push(format!("Today's date: {}", context.current_date));
    }
    if !context.model.is_empty() {
        lines.push(format!("Model: {}", context.model));
    }
    if !context.knowledge_cutoff.is_empty() {
        lines.push(format!("Knowledge cutoff: {}", context.knowledge_cutoff));
    }

    if let Some(status) = &context.git_status_short {
        lines.push(format!("Git status:\n{status}"));
    }
    if let Some(commits) = &context.git_recent_commits {
        lines.push(format!("Recent commits:\n{commits}"));
    }

    lines.push("</environment>".to_owned());
    lines.join("\n")
}

#[cfg(test)]
pub(crate) mod tests {
    use std::result::Result as StdResult;
    use std::sync::Arc;

    use async_trait::async_trait;
    use lithos_llm::catalog::CodecId;
    use serde_json::json;

    use super::*;
    use crate::environment::Environment;
    use crate::profile::AgentProfile;
    use crate::search::{SearchError, SearchRequest, SearchResult};
    use crate::test_support::MockEnvironment;
    use crate::tool::{NativeTool, ToolContext, ToolRegistry};
    use crate::types::ToolSource;

    /// A search engine no prompt test ever asks anything: what these tests are
    /// about is whether the tool is offered at all, and under what name.
    pub(crate) struct UnusedSearch;

    #[async_trait]
    impl SearchProvider for UnusedSearch {
        async fn search(
            &self,
            _request: SearchRequest,
        ) -> StdResult<Vec<SearchResult>, SearchError> {
            Ok(Vec::new())
        }
    }

    /// The engine a harness built with `configured` is given.
    pub(crate) fn search_provider(configured: bool) -> Option<Arc<dyn SearchProvider>> {
        configured.then(|| Arc::new(UnusedSearch) as Arc<dyn SearchProvider>)
    }

    /// A built-in identity that only marks a prompt test's completed registry.
    pub(crate) fn native_marker(tool: NativeTool) -> RegisteredTool {
        RegisteredTool::new(
            ToolDefinition::function(
                tool.canonical_name(),
                "Prompt capability marker",
                json!({ "type": "object" }),
            ),
            Arc::new(|_arguments, _context| Box::pin(async { Ok(String::new()) })),
        )
        .with_source(ToolSource::Native)
    }

    /// The environment every prompt test renders against, which is also what
    /// fabro's own prompt snapshots were taken in.
    pub(crate) fn snapshot_context() -> EnvContext {
        EnvContext::from_environment(&MockEnvironment::linux())
    }

    /// The prompt `profile` builds for a session that loaded nothing and holds
    /// exactly the tools the profile contributed.
    pub(crate) fn system_prompt(profile: &dyn AgentProfile) -> String {
        system_prompt_with_tools(profile, Vec::new())
    }

    /// The same, for a session whose registry also holds `extra` — the tools
    /// the builder adds that no profile contributes.
    pub(crate) fn system_prompt_with_tools(
        profile: &dyn AgentProfile,
        extra: Vec<RegisteredTool>,
    ) -> String {
        profile.build_system_prompt(
            &registry_of(profile, extra),
            &snapshot_context(),
            &[],
            None,
            &[],
        )
    }

    /// The registry a session built from `profile` would hold, plus `extra`.
    pub(crate) fn registry_of(
        profile: &dyn AgentProfile,
        extra: Vec<RegisteredTool>,
    ) -> ToolRegistry {
        let mut registry = ToolRegistry::with_vocabulary(profile.tool_vocabulary());
        for tool in profile.base_tools().into_iter().chain(extra) {
            registry.register(tool);
        }
        registry
    }

    /// The name `profile` would advertise a web search under.
    pub(crate) fn web_search_name(profile: &dyn AgentProfile) -> &'static str {
        NativeTool::WebSearch.name(profile.tool_vocabulary())
    }

    /// Whether `profile` contributes a tool called `name`.
    pub(crate) fn advertises(profile: &dyn AgentProfile, name: &str) -> bool {
        profile
            .base_tools()
            .iter()
            .any(|tool| tool.definition.name == name)
    }

    /// What `profile` says about the tool it contributes as `name`, when it
    /// contributes one at all.
    ///
    /// The name is the canonical one, because a profile contributes definitions
    /// and the registry renames them on the way in.
    fn describe(profile: &dyn AgentProfile, name: &str) -> Option<String> {
        profile
            .base_tools()
            .into_iter()
            .find(|tool| tool.definition.name == name)
            .map(|tool| tool.definition.description)
    }

    /// The tools built from shared factories, which is where a reworded
    /// description would leak from one harness into another.
    const SHARED_TOOLS: [&str; 6] = [
        "read_file",
        "write_file",
        "edit_file",
        "shell",
        "grep",
        "glob",
    ];

    /// The names `profile` contributes, sorted.
    pub(crate) fn tool_names(profile: &dyn AgentProfile) -> Vec<String> {
        let mut names: Vec<String> = profile
            .base_tools()
            .iter()
            .map(|tool| tool.definition.name.clone())
            .collect();
        names.sort();
        names
    }

    /// How long a command with no timeout of its own runs under `profile`.
    ///
    /// Asked of the tool rather than of the options it was built from, because
    /// what matters is that the profile actually handed them over.
    pub(crate) async fn shell_timeout_ms(profile: &dyn AgentProfile) -> u64 {
        let shell = profile
            .base_tools()
            .into_iter()
            .find(|tool| tool.definition.name == "shell")
            .expect("every built-in harness offers a shell");
        let environment = Arc::new(MockEnvironment::linux());
        let context = ToolContext::new(Arc::clone(&environment) as Arc<dyn Environment>);

        (shell.executor)(json!({"command": "true"}), context)
            .await
            .expect("the mock command succeeds");

        environment
            .captured_timeout
            .lock()
            .expect("the mock records one timeout")
            .expect("the shell tool always names a timeout")
    }

    #[test]
    fn the_responses_codec_is_the_only_one_that_carries_a_patch_grammar() {
        assert_eq!(
            FileEditToolKind::for_codec(&CodecId::from("openai-responses")),
            FileEditToolKind::ApplyPatch
        );
        for codec in [
            "openai-chat",
            "anthropic-messages",
            "gemini-generate",
            "bedrock-converse",
            "something-new",
        ] {
            assert_eq!(
                FileEditToolKind::for_codec(&CodecId::from(codec)),
                FileEditToolKind::EditFile,
                "{codec} cannot carry a freeform tool"
            );
        }
    }

    #[test]
    fn the_editor_names_are_the_ones_the_prompts_compare_against() {
        assert_eq!(FileEditToolKind::ApplyPatch.as_str(), "apply_patch");
        assert_eq!(FileEditToolKind::EditFile.as_str(), "edit_file");
        assert_eq!(
            FileEditToolKind::ApplyPatch.tool().definition.name,
            "apply_patch"
        );
        assert_eq!(
            FileEditToolKind::EditFile.tool().definition.name,
            "edit_file"
        );
    }

    #[test]
    fn a_patch_tool_is_the_one_definition_that_is_not_a_function() {
        assert!(FileEditToolKind::ApplyPatch.tool().definition.is_custom());
        assert!(!FileEditToolKind::EditFile.tool().definition.is_custom());
    }

    /// Every harness that speaks pebble's own tool names gets the same `shell`
    /// definition, so the Bash contract cannot drift between model families.
    /// Only the timeouts behind it differ, and those are not in the definition.
    #[test]
    fn the_harnesses_on_pebbles_own_names_advertise_one_shell_tool() {
        let profiles: [Arc<dyn AgentProfile>; 3] = [
            Arc::new(AnthropicProfile::new(&ProfileDeps::default())),
            Arc::new(GeminiProfile::new(&ProfileDeps::default())),
            Arc::new(OpenAiProfile::new(&ProfileDeps::default())),
        ];

        let definitions: Vec<_> = profiles
            .iter()
            .map(|profile| {
                profile
                    .base_tools()
                    .into_iter()
                    .find(|tool| tool.definition.name == "shell")
                    .expect("every harness offers a shell")
                    .definition
            })
            .collect();

        for definition in &definitions {
            assert_eq!(definition, &definitions[0]);
        }
        assert!(
            definitions[0].description.contains("Bash"),
            "{}",
            definitions[0].description
        );
    }

    /// The file tools are shared factories, so a harness that reworded one
    /// would reword it for every model family. Only Kimi is meant to, and it
    /// replaces the tools rather than editing them.
    #[test]
    fn the_harnesses_on_pebbles_own_names_share_their_tool_descriptions() {
        let anthropic: Arc<dyn AgentProfile> =
            Arc::new(AnthropicProfile::new(&ProfileDeps::default()));
        let others: [Arc<dyn AgentProfile>; 2] = [
            Arc::new(GeminiProfile::new(&ProfileDeps::default())),
            Arc::new(OpenAiProfile::new(&ProfileDeps::default())),
        ];

        for name in SHARED_TOOLS {
            let Some(expected) = describe(anthropic.as_ref(), name) else {
                continue;
            };
            for profile in &others {
                let Some(other) = describe(profile.as_ref(), name) else {
                    continue;
                };
                assert_eq!(
                    other,
                    expected,
                    "{:?} reworded {name}",
                    profile.profile_kind()
                );
            }
        }
    }

    /// The other half of the same rule: Kimi's rewordings must reach Kimi's
    /// model and nobody else's.
    ///
    /// Kimi Code keeps its read-before-edit drilling and its search guidance in
    /// the tool descriptions rather than in its prompt, so this harness rewords
    /// six tools that every other harness shares. Asserting only that the
    /// wording is absent elsewhere goes quietly vacuous the next time it is
    /// rewritten, so each rewording is asserted present in Kimi first — which
    /// is how fabro's own version of this test once stopped testing anything.
    #[test]
    fn the_kimi_harnesss_rewordings_reach_no_other_model() {
        /// The one phrase this harness exists to put in front of its model.
        const KIMI_EDIT_MARKER: &str = "DO NOT call Edit from memory";

        let kimi: Arc<dyn AgentProfile> = Arc::new(KimiProfile::new(&ProfileDeps::default()));
        let others: [Arc<dyn AgentProfile>; 6] = [
            Arc::new(AnthropicProfile::new(&ProfileDeps::default())),
            Arc::new(Claude5Profile::new(&ProfileDeps::default())),
            Arc::new(GeminiProfile::new(&ProfileDeps::default())),
            Arc::new(Gpt56Profile::new(&ProfileDeps::default())),
            Arc::new(Gpt6Profile::new(&ProfileDeps::default())),
            Arc::new(OpenAiProfile::new(&ProfileDeps::default())),
        ];

        for name in SHARED_TOOLS {
            let kimi_text = describe(kimi.as_ref(), name)
                .unwrap_or_else(|| panic!("the Kimi harness offers {name}"));

            for profile in &others {
                let Some(other) = describe(profile.as_ref(), name) else {
                    continue;
                };
                assert_ne!(
                    other,
                    kimi_text,
                    "{name} should be reworded for Kimi only, not for {:?}",
                    profile.profile_kind()
                );
            }
        }

        // A reworded shell is still a Bash shell; a harness that lost that
        // would have its model writing for some other interpreter.
        assert!(
            describe(kimi.as_ref(), "shell")
                .expect("the Kimi harness offers a shell")
                .to_ascii_lowercase()
                .contains("bash")
        );

        let kimi_edit = describe(kimi.as_ref(), "edit_file").expect("the harness offers an editor");
        assert!(kimi_edit.contains(KIMI_EDIT_MARKER), "{kimi_edit}");
        for profile in &others {
            let Some(other) = describe(profile.as_ref(), "edit_file") else {
                continue;
            };
            assert!(
                !other.contains(KIMI_EDIT_MARKER),
                "Kimi's read-before-edit drilling leaked into {:?}",
                profile.profile_kind()
            );
        }
    }

    #[test]
    fn the_core_tools_are_the_ones_no_coding_agent_works_without() {
        let names: Vec<String> = core_tools(&NativeToolOptions::default(), None, None)
            .iter()
            .map(|tool| tool.definition.name.clone())
            .collect();

        assert_eq!(names, [
            "read_file",
            "write_file",
            "shell",
            "grep",
            "glob",
            "web_fetch"
        ]);
    }

    #[test]
    fn an_environment_block_names_the_three_things_the_environment_knows() {
        let block = build_env_context_block(&snapshot_context());

        assert_eq!(
            block,
            "<environment>\nWorking directory: /home/test\nIs git repository: false\nPlatform: \
             linux\nOS version: Linux 6.1.0\n</environment>"
        );
    }

    #[test]
    fn an_environment_block_carries_what_the_session_gathered() {
        let context = EnvContext {
            is_git_repo: true,
            git_branch: Some("main".to_owned()),
            current_date: "2026-02-20".to_owned(),
            model: "claude-opus-4-6".to_owned(),
            knowledge_cutoff: "May 2025".to_owned(),
            git_status_short: Some(" M src/lib.rs".to_owned()),
            git_recent_commits: Some("abc123 first".to_owned()),
            ..snapshot_context()
        };

        let block = build_env_context_block(&context);

        assert!(block.contains("Is git repository: true"));
        assert!(block.contains("Git branch: main"));
        assert!(block.contains("Today's date: 2026-02-20"));
        assert!(block.contains("Model: claude-opus-4-6"));
        assert!(block.contains("Knowledge cutoff: May 2025"));
        assert!(block.contains("Git status:\n M src/lib.rs"));
        assert!(block.contains("Recent commits:\nabc123 first"));
    }

    #[test]
    fn a_prompt_ends_with_the_sections_the_session_loaded() {
        let template = EmbeddedPrompt::new("test.md.j2", "Prompt.\n\n{{ inputs.env_block }}\n");
        let skills = [Skill {
            name:        "review".to_owned(),
            description: "Review a change".to_owned(),
            template:    "Read the diff.".to_owned(),
        }];

        let prompt = assemble_system_prompt(
            template,
            ToolVocabulary::Canonical,
            &snapshot_context(),
            &["# README".to_owned(), "# CONTRIBUTING".to_owned()],
            Some("Always write tests first"),
            &skills,
        );

        assert!(prompt.starts_with("Prompt.\n\n<environment>"));
        assert!(
            prompt.contains("</environment>\n\n# README\n\n# CONTRIBUTING\n\n# Available Skills")
        );
        assert!(prompt.contains("- `review`: Review a change"));
        assert!(prompt.ends_with("\n\n# User Instructions\nAlways write tests first"));
    }

    #[test]
    fn a_prompt_with_nothing_loaded_is_the_template_alone() {
        let template = EmbeddedPrompt::new("test.md.j2", "Prompt.\n\n{{ inputs.env_block }}\n");

        let prompt = assemble_system_prompt(
            template,
            ToolVocabulary::Canonical,
            &snapshot_context(),
            &[],
            None,
            &[],
        );

        assert!(prompt.ends_with("</environment>"));
    }

    #[test]
    fn the_skills_section_names_the_skill_tool_in_the_profile_vocabulary() {
        let skills = [Skill {
            name:        "review".to_owned(),
            description: String::new(),
            template:    "Read the diff.".to_owned(),
        }];

        let prompt = assemble_system_prompt(
            EmbeddedPrompt::new("test.md.j2", "Prompt.\n"),
            ToolVocabulary::Claude5,
            &snapshot_context(),
            &[],
            None,
            &skills,
        );

        assert!(prompt.contains("call the `Skill` tool"), "{prompt}");
    }

    #[test]
    #[should_panic(expected = "the embedded prompt template failed to render")]
    fn a_template_naming_an_input_it_was_not_given_is_a_bug() {
        let _ = EmbeddedPrompt::new("test.md.j2", "{{ inputs.absent }}").render(String::new());
    }
}
