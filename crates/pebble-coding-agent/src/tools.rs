//! The tools pebble ships.
//!
//! Each one is built by a `make_*` function that answers a
//! [`RegisteredTool`]: what the model is told, and what
//! runs when it calls. A profile decides which of them a session starts with,
//! and the registry decides what the model sees them called, so the same
//! `read_file` here is `Read` to a model that expects that name.
//!
//! Every one of them acts through
//! [`Environment`](crate::environment::Environment) — no tool touches a file or
//! starts a process itself — which is what lets a session work inside a
//! container, on a remote workspace, or against a test double without any of
//! them knowing.
//!
//! What a tool answers with is the model's to read, so the strings here are
//! part of pebble's contract: the schemas, the descriptions, the rendered
//! output, and the messages a failure carries.

// The modules stay reachable inside the crate — `crate::tools::shell` — so a
// profile that gives its model a shell tool of its own shape can call the
// pipeline underneath rather than the environment directly. What an
// application outside pebble names is the flat facade below.
pub(crate) mod apply_patch;
pub(crate) mod files;
pub(crate) mod question;
pub(crate) mod search;
pub(crate) mod shell;
pub(crate) mod skill;
#[cfg(test)]
pub(crate) mod testing;
pub(crate) mod todo;
pub(crate) mod web;
pub(crate) mod web_search;

pub use self::apply_patch::make_apply_patch_tool;
pub use self::files::{
    make_edit_file_tool, make_read_file_tool, make_read_many_files_tool, make_write_file_tool,
};
pub use self::question::{
    make_anthropic_question_tool, make_claude5_question_tool, make_openai_question_tool,
    make_question_tool,
};
pub use self::search::{grep_result_path, make_glob_tool, make_grep_tool, make_list_dir_tool};
pub use self::shell::{make_shell_tool, make_shell_tool_with_options};
pub use self::todo::{
    TodoRuntime, make_task_create_tool, make_task_get_tool, make_task_list_tool,
    make_task_update_tool, make_todo_list_tool, make_update_plan_tool,
};
pub use self::web::{WebFetchSummarizer, make_web_fetch_tool};
pub use self::web_search::make_web_search_tool;
pub use crate::config::{
    NativeToolOptions, ToolAccess, ToolAccessPolicy, ToolApprovalAdapter, ToolApprovalFn,
    ToolExposureMode, ToolHookCallback, ToolHookDecision,
};
pub use crate::event::OutputCaptureStats;
pub use crate::tool::{
    CodingEventEmitter, CodingToolSet, RegisteredTool, StaticEnvProvider, ToolContext,
    ToolEnvProvider, ToolError, ToolEventCallback, ToolExecutor, ToolRunner, canonical_tool_name,
};
pub use crate::types::{PermissionLevel, ToolCategory, ToolErrorKind, ToolSource, ToolSummary};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NativeToolOptions;
    use crate::tool::RegisteredTool;
    use crate::types::ToolSource;

    /// The tools every profile is expected to offer, whatever else it adds.
    fn core_tools() -> Vec<RegisteredTool> {
        vec![
            make_read_file_tool(),
            make_write_file_tool(),
            make_edit_file_tool(),
            make_shell_tool_with_options(&NativeToolOptions::default()),
            make_grep_tool(),
            make_glob_tool(),
            make_web_fetch_tool(None),
        ]
    }

    /// A model chooses a tool from its description, so each one says what it
    /// is for and how to steer it.
    #[test]
    fn every_core_description_says_what_the_tool_is_for() {
        let tools = core_tools();
        let description = |name: &str| {
            tools
                .iter()
                .find(|tool| tool.definition.name == name)
                .unwrap_or_else(|| panic!("the {name} tool is built"))
                .definition
                .description
                .as_str()
        };

        assert!(description("read_file").contains("Read files before editing"));
        assert!(description("read_file").contains("offset"));
        assert!(description("write_file").contains("new files"));
        assert!(description("write_file").contains("overwrites"));
        assert!(description("edit_file").contains("exact match"));
        assert!(description("edit_file").contains("unique"));
        assert!(description("shell").contains("tests and builds"));
        assert!(description("shell").contains("timeout_ms"));
        assert!(description("grep").contains("regex"));
        assert!(description("grep").contains("glob_filter"));
        assert!(description("glob").contains("search-root-relative"));
        assert!(description("glob").contains("`**` searches recursively"));
        assert!(description("web_fetch").contains("http:// or https://"));
        assert!(description("web_fetch").contains("prompt"));
    }

    /// Descriptions are copied between harnesses, and a description that
    /// promises a capability pebble does not have costs a round trip every
    /// time a model believes it.
    #[test]
    fn no_description_promises_something_pebble_cannot_do() {
        for tool in core_tools() {
            let text = &tool.definition.description;
            assert!(!text.contains("addComment"), "no comment API: {text}");
            assert!(
                !text.contains("background Bash"),
                "no background commands: {text}"
            );
            assert!(!text.contains("PDF"), "no PDF reads: {text}");
            assert!(!text.contains("image"), "no image reads: {text}");
        }
    }

    #[test]
    fn every_core_tool_is_a_native_one() {
        for tool in core_tools() {
            assert_eq!(
                tool.source,
                ToolSource::Native,
                "{} is one of pebble's own",
                tool.definition.name
            );
        }
    }
}
