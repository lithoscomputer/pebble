//! The tools pebble implements itself, and the names they can be expressed
//! under.
//!
//! Tool names reach a session from two very different places. The tools pebble
//! implements are a fixed set known at compile time; MCP, skill, and
//! application-registered tools are open-ended and named by whatever
//! registered them. This module covers the first group, so anything reasoning
//! about a built-in tool is checked by the compiler instead of matched on
//! string literals.
//!
//! A [`NativeTool`] is an identity, not a name. The same tool is expressed
//! under different names depending on the [`ToolVocabulary`] a profile speaks:
//! pebble's own names by default, Anthropic's names for Claude 5, Kimi Code's
//! names for the Kimi profile, and Codex's names for the GPT-5.6 profile.
//! Permissions, categories, and telemetry resolve any name back to the
//! identity, so behavior never depends on which vocabulary is in play.
//!
//! `ToolDefinition.name` and [`ToolRegistry`](super::registry::ToolRegistry)
//! keys stay `String`, because they carry both groups.

use std::fmt;

use crate::truncation::{ToolOutputLimits, TruncationMode};
use crate::types::ToolCategory;

/// A naming scheme for built-in tools.
///
/// Not serialized: a vocabulary describes how a session talks to one model
/// family, and the names it produces are what reach the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ToolVocabulary {
    /// Pebble's own names, which are also the canonical identities the rest of
    /// the crate reasons about.
    #[default]
    Canonical,
    /// The names Anthropic's Claude 5 coding harness exposes.
    Claude5,
    /// The names Kimi Code exposes, for models trained against that harness.
    KimiCode,
    /// The names Codex exposes, for the GPT-5.6 models trained against it.
    Codex,
}

impl ToolVocabulary {
    /// Every vocabulary pebble speaks.
    ///
    /// A slice rather than an array, so a vocabulary added later does not
    /// change this constant's type.
    pub const ALL: &'static [Self] = &[Self::Canonical, Self::Claude5, Self::KimiCode, Self::Codex];
}

/// A tool pebble implements itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NativeTool {
    /// Reads one file.
    ReadFile,
    /// Reads several files in one call.
    ReadManyFiles,
    /// Writes a whole file.
    WriteFile,
    /// Replaces a span inside a file.
    EditFile,
    /// Applies a patch in the codex `apply_patch` format.
    ApplyPatch,
    /// Lists a directory.
    ListDir,
    /// Searches file contents.
    Grep,
    /// Lists files matching a glob.
    Glob,
    /// Runs a command.
    Shell,
    /// Searches the web.
    WebSearch,
    /// Fetches one URL.
    WebFetch,
    /// Starts a child session and returns a handle to it.
    SpawnAgent,
    /// Sends input to a running child session.
    SendInput,
    /// Waits for a child session to produce output.
    Wait,
    /// Closes a child session.
    CloseAgent,
    /// Runs a child session in the background, Claude 5 style.
    BackgroundAgent,
    /// Polls or waits for a background child session's output.
    AgentOutput,
    /// Stops a background child session.
    StopAgent,
    /// Messages a background child session.
    MessageAgent,
    /// Activates a skill.
    UseSkill,
    /// Replaces the current plan.
    UpdatePlan,
    /// Creates one task.
    TaskCreate,
    /// Updates one task.
    TaskUpdate,
    /// Reads one task.
    TaskGet,
    /// Lists tasks.
    TaskList,
    /// Replaces the whole todo list.
    TodoList,
    /// Asks the human a question, Anthropic style.
    AskUserQuestion,
    /// Asks the human a question, OpenAI style.
    RequestUserInput,
}

impl NativeTool {
    /// Every built-in tool.
    ///
    /// A slice rather than an array, so a tool added later does not change
    /// this constant's type.
    pub const ALL: &'static [Self] = &[
        Self::ReadFile,
        Self::ReadManyFiles,
        Self::WriteFile,
        Self::EditFile,
        Self::ApplyPatch,
        Self::ListDir,
        Self::Grep,
        Self::Glob,
        Self::Shell,
        Self::WebSearch,
        Self::WebFetch,
        Self::SpawnAgent,
        Self::SendInput,
        Self::Wait,
        Self::CloseAgent,
        Self::BackgroundAgent,
        Self::AgentOutput,
        Self::StopAgent,
        Self::MessageAgent,
        Self::UseSkill,
        Self::UpdatePlan,
        Self::TaskCreate,
        Self::TaskUpdate,
        Self::TaskGet,
        Self::TaskList,
        Self::TodoList,
        Self::AskUserQuestion,
        Self::RequestUserInput,
    ];

    /// The canonical name: how pebble refers to this tool internally.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ReadFile => "read_file",
            Self::ReadManyFiles => "read_many_files",
            Self::WriteFile => "write_file",
            Self::EditFile => "edit_file",
            Self::ApplyPatch => "apply_patch",
            Self::ListDir => "list_dir",
            Self::Grep => "grep",
            Self::Glob => "glob",
            Self::Shell => "shell",
            Self::WebSearch => "web_search",
            Self::WebFetch => "web_fetch",
            Self::SpawnAgent => "spawn_agent",
            Self::SendInput => "send_input",
            Self::Wait => "wait",
            Self::CloseAgent => "close_agent",
            Self::BackgroundAgent => "background_agent",
            Self::AgentOutput => "agent_output",
            Self::StopAgent => "stop_agent",
            Self::MessageAgent => "message_agent",
            Self::UseSkill => "use_skill",
            Self::UpdatePlan => "update_plan",
            // The task and question tools are already PascalCase on the wire;
            // they came from the Claude Code vocabulary rather than pebble's
            // own.
            Self::TaskCreate => "TaskCreate",
            Self::TaskUpdate => "TaskUpdate",
            Self::TaskGet => "TaskGet",
            Self::TaskList => "TaskList",
            Self::TodoList => "TodoList",
            Self::AskUserQuestion => "AskUserQuestion",
            Self::RequestUserInput => "request_user_input",
        }
    }

    /// The names other harnesses use for this tool, beyond its canonical one.
    ///
    /// [`from_any_name`](Self::from_any_name) resolves these; the vocabulary
    /// tables in [`name`](Self::name) decide which one a session exposes.
    #[must_use]
    pub const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::ReadFile => &["Read"],
            Self::WriteFile => &["Write"],
            Self::EditFile => &["Edit"],
            Self::Grep => &["Grep"],
            Self::Glob => &["Glob"],
            Self::Shell => &["Bash", "shell_command"],
            Self::WebSearch => &["WebSearch"],
            Self::WebFetch => &["FetchURL", "WebFetch"],
            Self::BackgroundAgent => &["Agent"],
            Self::AgentOutput => &["TaskOutput"],
            Self::StopAgent => &["TaskStop"],
            Self::MessageAgent => &["SendMessage"],
            Self::UseSkill => &["Skill"],
            _ => &[],
        }
    }

    /// Resolves a canonical pebble name to its built-in identity.
    ///
    /// Unlike [`from_any_name`](Self::from_any_name), this deliberately
    /// ignores harness aliases. The registry uses it while registering tools,
    /// so an unrelated extension named `Read` is not silently treated as
    /// pebble's file reader.
    #[must_use]
    pub fn from_canonical_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|tool| tool.canonical_name() == name)
    }

    /// The name this tool is exposed under in `vocabulary`.
    ///
    /// A tool with no counterpart in the vocabulary keeps its canonical name.
    #[must_use]
    pub const fn name(self, vocabulary: ToolVocabulary) -> &'static str {
        match vocabulary {
            ToolVocabulary::Canonical => self.canonical_name(),
            ToolVocabulary::Claude5 => match self {
                Self::ReadFile => "Read",
                Self::WriteFile => "Write",
                Self::EditFile => "Edit",
                Self::Shell => "Bash",
                // Named for completeness: this arm describes the vocabulary,
                // not the profile's registry, and the Claude 5 profile
                // deliberately registers neither.
                Self::Grep => "Grep",
                Self::Glob => "Glob",
                Self::WebSearch => "WebSearch",
                Self::WebFetch => "WebFetch",
                Self::UseSkill => "Skill",
                Self::BackgroundAgent => "Agent",
                Self::AgentOutput => "TaskOutput",
                Self::StopAgent => "TaskStop",
                Self::MessageAgent => "SendMessage",
                other => other.canonical_name(),
            },
            ToolVocabulary::KimiCode => match self {
                Self::ReadFile => "Read",
                Self::WriteFile => "Write",
                Self::EditFile => "Edit",
                Self::Shell => "Bash",
                Self::Grep => "Grep",
                Self::Glob => "Glob",
                Self::WebSearch => "WebSearch",
                Self::WebFetch => "FetchURL",
                Self::UseSkill => "Skill",
                // Deliberately unmapped. Kimi Code's `Agent` launches a
                // subagent and returns its result; pebble's `spawn_agent`
                // returns a handle that `send_input`, `wait`, and
                // `close_agent` then drive. Borrowing the name without the
                // semantics would promise a result the tool does not return --
                // the same mistake as exposing incremental task tools under a
                // whole-list name.
                other => other.canonical_name(),
            },
            // Codex names its shell `shell_command`. Its remaining tools that
            // pebble also implements -- apply_patch, update_plan,
            // request_user_input -- already agree with pebble's names, and the
            // tools pebble has that Codex does not keep pebble's names.
            //
            // Deliberately unmapped: Codex's sub-agent tools differ by
            // multi-agent protocol version rather than by name alone
            // (`resume_agent` has no pebble counterpart), and its `web.run` is
            // a namespaced tool, which the registry cannot express.
            ToolVocabulary::Codex => match self {
                Self::Shell => "shell_command",
                other => other.canonical_name(),
            },
        }
    }

    /// Resolves a name in any known vocabulary back to the tool it identifies.
    ///
    /// Returns `None` for MCP, skill, and application-registered tools, whose
    /// names are not drawn from this set.
    #[must_use]
    pub fn from_any_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|tool| tool.canonical_name() == name || tool.aliases().contains(&name))
    }

    /// The output limits history keeps for this tool's results by default.
    ///
    /// Matched exhaustively so a new built-in tool has to state its answer.
    /// `None` budgets keep everything, which is also what tools pebble does
    /// not know get.
    #[must_use]
    pub(crate) const fn default_output_limits(self) -> ToolOutputLimits {
        const fn limits(
            max_chars: Option<usize>,
            max_lines: Option<usize>,
            mode: TruncationMode,
        ) -> ToolOutputLimits {
            ToolOutputLimits {
                max_chars,
                max_lines,
                mode,
            }
        }

        match self {
            Self::ReadFile => limits(Some(50_000), None, TruncationMode::HeadTail),
            Self::Shell => limits(Some(30_000), Some(256), TruncationMode::HeadTail),
            Self::Grep => limits(Some(20_000), Some(200), TruncationMode::Tail),
            Self::Glob => limits(Some(20_000), Some(500), TruncationMode::Tail),
            Self::SpawnAgent => limits(Some(20_000), None, TruncationMode::HeadTail),
            Self::EditFile | Self::ApplyPatch => limits(Some(10_000), None, TruncationMode::Tail),
            Self::WriteFile => limits(Some(1_000), None, TruncationMode::Tail),
            Self::ReadManyFiles
            | Self::ListDir
            | Self::WebSearch
            | Self::WebFetch
            | Self::SendInput
            | Self::Wait
            | Self::CloseAgent
            | Self::BackgroundAgent
            | Self::AgentOutput
            | Self::StopAgent
            | Self::MessageAgent
            | Self::UseSkill
            | Self::UpdatePlan
            | Self::TaskCreate
            | Self::TaskUpdate
            | Self::TaskGet
            | Self::TaskList
            | Self::TodoList
            | Self::AskUserQuestion
            | Self::RequestUserInput => limits(None, None, TruncationMode::HeadTail),
        }
    }

    /// The coarse access category, or `None` when the tool is not part of the
    /// permission taxonomy.
    ///
    /// Matched exhaustively so a new built-in tool has to state its answer.
    /// `None` is a real answer, and callers disagree about what it means: a
    /// permission gate treats an uncategorized tool as
    /// [`ToolCategory::Shell`], requiring approval, while a tool summary
    /// reports [`ToolCategory::Other`].
    #[must_use]
    pub const fn category(self) -> Option<ToolCategory> {
        match self {
            Self::ReadFile | Self::ReadManyFiles | Self::Grep | Self::Glob | Self::ListDir => {
                Some(ToolCategory::Read)
            }
            Self::WriteFile | Self::EditFile | Self::ApplyPatch => Some(ToolCategory::Write),
            Self::Shell => Some(ToolCategory::Shell),
            Self::SpawnAgent
            | Self::SendInput
            | Self::Wait
            | Self::CloseAgent
            | Self::BackgroundAgent
            | Self::AgentOutput
            | Self::StopAgent
            | Self::MessageAgent => Some(ToolCategory::Subagent),
            // Uncategorized. Giving these a category would change what a
            // permission gate auto-approves, which is a behavior change rather
            // than a classification cleanup, so they keep their answer.
            Self::WebSearch
            | Self::WebFetch
            | Self::UseSkill
            | Self::UpdatePlan
            | Self::TaskCreate
            | Self::TaskUpdate
            | Self::TaskGet
            | Self::TaskList
            | Self::TodoList
            | Self::AskUserQuestion
            | Self::RequestUserInput => None,
        }
    }
}

impl fmt::Display for NativeTool {
    /// Writes the canonical name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.canonical_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_names_round_trip() {
        for tool in NativeTool::ALL.iter().copied() {
            assert_eq!(
                NativeTool::from_any_name(tool.canonical_name()),
                Some(tool),
                "{tool} should resolve from its canonical name"
            );
            assert_eq!(
                NativeTool::from_canonical_name(tool.canonical_name()),
                Some(tool)
            );
            assert_eq!(tool.to_string(), tool.canonical_name());
        }
    }

    #[test]
    fn every_name_in_every_vocabulary_resolves_back_to_its_tool() {
        for tool in NativeTool::ALL.iter().copied() {
            for vocabulary in ToolVocabulary::ALL.iter().copied() {
                let name = tool.name(vocabulary);
                assert_eq!(
                    NativeTool::from_any_name(name),
                    Some(tool),
                    "{name} ({vocabulary:?}) should resolve back to {tool}"
                );
            }
        }
    }

    #[test]
    fn every_alias_resolves_to_its_tool() {
        for tool in NativeTool::ALL.iter().copied() {
            for alias in tool.aliases() {
                assert_eq!(
                    NativeTool::from_any_name(alias),
                    Some(tool),
                    "alias {alias} should resolve to {tool}"
                );
            }
        }
    }

    /// Two tools resolving to the same name would make `from_any_name`
    /// ambiguous and silently mis-categorize one of them.
    #[test]
    fn vocabularies_do_not_collide() {
        let mut seen: Vec<(&str, NativeTool)> = Vec::new();
        for tool in NativeTool::ALL.iter().copied() {
            let names = ToolVocabulary::ALL
                .iter()
                .map(|vocabulary| tool.name(*vocabulary))
                .chain(tool.aliases().iter().copied());
            for name in names {
                if let Some((_, other)) = seen.iter().find(|(seen, _)| *seen == name) {
                    assert_eq!(*other, tool, "name '{name}' is claimed by two tools");
                } else {
                    seen.push((name, tool));
                }
            }
        }
    }

    #[test]
    fn kimi_vocabulary_renames_only_where_kimi_code_differs() {
        assert_eq!(NativeTool::ReadFile.name(ToolVocabulary::KimiCode), "Read");
        assert_eq!(NativeTool::Shell.name(ToolVocabulary::KimiCode), "Bash");
        assert_eq!(
            NativeTool::WebFetch.name(ToolVocabulary::KimiCode),
            "FetchURL"
        );
        // No Kimi Code counterpart: keeps pebble's name.
        assert_eq!(
            NativeTool::TaskCreate.name(ToolVocabulary::KimiCode),
            "TaskCreate"
        );
        assert_eq!(
            NativeTool::SpawnAgent.name(ToolVocabulary::KimiCode),
            "spawn_agent"
        );
    }

    #[test]
    fn claude5_vocabulary_uses_anthropic_harness_names() {
        assert_eq!(NativeTool::ReadFile.name(ToolVocabulary::Claude5), "Read");
        assert_eq!(NativeTool::Shell.name(ToolVocabulary::Claude5), "Bash");
        assert_eq!(
            NativeTool::WebFetch.name(ToolVocabulary::Claude5),
            "WebFetch"
        );
        assert_eq!(
            NativeTool::BackgroundAgent.name(ToolVocabulary::Claude5),
            "Agent"
        );
        assert_eq!(
            NativeTool::AgentOutput.name(ToolVocabulary::Claude5),
            "TaskOutput"
        );
        assert_eq!(
            NativeTool::StopAgent.name(ToolVocabulary::Claude5),
            "TaskStop"
        );
        assert_eq!(
            NativeTool::MessageAgent.name(ToolVocabulary::Claude5),
            "SendMessage"
        );
    }

    /// The harness name is how a tool is expressed, not what it is: the
    /// identity keeps a pebble name, and the harness name resolves back to it.
    #[test]
    fn claude5_subagent_tools_keep_canonical_names() {
        for (tool, canonical, claude5) in [
            (NativeTool::BackgroundAgent, "background_agent", "Agent"),
            (NativeTool::AgentOutput, "agent_output", "TaskOutput"),
            (NativeTool::StopAgent, "stop_agent", "TaskStop"),
            (NativeTool::MessageAgent, "message_agent", "SendMessage"),
        ] {
            assert_eq!(tool.canonical_name(), canonical);
            assert_eq!(tool.name(ToolVocabulary::Canonical), canonical);
            assert_eq!(tool.name(ToolVocabulary::Claude5), claude5);
            assert_eq!(NativeTool::from_any_name(canonical), Some(tool));
            assert_eq!(NativeTool::from_any_name(claude5), Some(tool));
        }
    }

    #[test]
    fn codex_vocabulary_renames_only_the_shell() {
        assert_eq!(
            NativeTool::Shell.name(ToolVocabulary::Codex),
            "shell_command"
        );
        // Already agree with Codex's names.
        assert_eq!(
            NativeTool::ApplyPatch.name(ToolVocabulary::Codex),
            "apply_patch"
        );
        assert_eq!(
            NativeTool::UpdatePlan.name(ToolVocabulary::Codex),
            "update_plan"
        );
        assert_eq!(
            NativeTool::RequestUserInput.name(ToolVocabulary::Codex),
            "request_user_input"
        );
        // No Codex counterpart: keeps pebble's name.
        assert_eq!(
            NativeTool::ReadFile.name(ToolVocabulary::Codex),
            "read_file"
        );
    }

    /// The canonical name is what permissions, categories, and telemetry key
    /// on, so carrying `shell_command` as an alias must not change it.
    #[test]
    fn shell_keeps_its_canonical_name_alongside_the_codex_alias() {
        assert_eq!(NativeTool::Shell.canonical_name(), "shell");
        assert_eq!(NativeTool::Shell.to_string(), "shell");
        assert_eq!(
            NativeTool::from_any_name("shell_command"),
            Some(NativeTool::Shell)
        );
        assert_eq!(NativeTool::from_canonical_name("shell_command"), None);
    }

    #[test]
    fn categories_are_vocabulary_independent() {
        for tool in NativeTool::ALL.iter().copied() {
            for vocabulary in ToolVocabulary::ALL.iter().copied() {
                let resolved = NativeTool::from_any_name(tool.name(vocabulary))
                    .expect("known name should resolve");
                assert_eq!(resolved.category(), tool.category());
            }
        }
    }

    #[test]
    fn unknown_names_resolve_to_nothing() {
        assert_eq!(NativeTool::from_any_name("mcp__files__read_file"), None);
        assert_eq!(
            NativeTool::from_canonical_name("mcp__files__read_file"),
            None
        );
    }

    #[test]
    fn the_default_vocabulary_is_pebbles_own() {
        assert_eq!(ToolVocabulary::default(), ToolVocabulary::Canonical);
    }
}
