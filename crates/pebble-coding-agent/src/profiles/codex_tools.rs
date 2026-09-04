//! The direct-call form of the tools shared by Codex-trained model families.
//!
//! Codex drives these models in code mode, with an execution tool that reaches
//! the rest through JavaScript. Pebble calls tools directly, so these
//! definitions preserve the names, parameters, and guidance without that
//! indirection.

use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde_json::{Value, json};

use super::{FileEditToolKind, ProfileDeps};
use crate::config::NativeToolOptions;
use crate::tool::{NativeTool, RegisteredTool, optional_integer_arg, required_str};
use crate::tools::shell::run_shell_command;
use crate::tools::{TodoRuntime, make_update_plan_tool, make_web_search_tool};
use crate::types::{AgentProfileKind, ToolSource};

/// Codex's shell, patch, and plan tools, plus web search when configured.
pub(super) fn codex_tools(
    profile_kind: AgentProfileKind,
    deps: &ProfileDeps,
) -> Vec<RegisteredTool> {
    let options = NativeToolOptions::for_profile(profile_kind);
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
    tools
}

/// Codex's `shell_command`: a shell script plus an explicit `workdir`.
///
/// Pebble's own `shell` tool has no `workdir`, and its description steers the
/// model toward dedicated read and search tools that these profiles do not
/// offer.
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

/// The shell these harnesses offer, describing the editor built beside it.
///
/// Both arguments must agree because the description points at the file editor
/// by name. Building them together keeps the prompt from naming a tool the
/// model was not given.
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
                let timeout_ms = optional_integer_arg(&arguments, "timeout_ms")
                    .unwrap_or(default_timeout_ms)
                    .min(max_timeout_ms);

                run_shell_command(&context, command, timeout_ms, workdir).await
            })
        })).with_source(ToolSource::Native)
}
