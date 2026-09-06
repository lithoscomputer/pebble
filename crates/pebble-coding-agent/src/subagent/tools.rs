//! Pebble's own four subagent tools.
//!
//! Spawn a child, send it more input, wait for it, close it. A profile whose
//! model expects a different family — the Claude 5 background-agent tools —
//! contributes those instead; both families drive the same
//! [`SubagentSupervisor`].
//!
//! Every string here is model-visible: the names, the descriptions, the
//! schemas, and the text a call answers with. They are the contract a model
//! was trained against, so they are ported byte for byte.

use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde_json::json;

use super::SubagentSupervisor;
use crate::tool::{NativeTool, RegisteredTool, ToolError, required_str};
use crate::types::ToolSource;

/// The four tools a session drives its own children with.
pub(crate) fn subagent_tools(supervisor: &SubagentSupervisor) -> Vec<RegisteredTool> {
    vec![
        spawn_agent_tool(supervisor.clone()),
        send_input_tool(supervisor.clone()),
        wait_tool(supervisor.clone()),
        close_agent_tool(supervisor.clone()),
    ]
}

/// Starts a child on a task and answers with its identifier.
fn spawn_agent_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::SpawnAgent.canonical_name(),
            "Spawn a subagent for independent work or context isolation. Use it for tasks that \
             can proceed separately, and avoid duplicating the same work in the parent session.",
            json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task description for the subagent"
                    }
                },
                "required": ["task"]
            }),
        ),
        Arc::new(move |arguments, context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let task = required_str(&arguments, "task")?;
                let parent = context.session();
                supervisor.spawn(parent, task.to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// Gives a child more to do.
fn send_input_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::SendInput.canonical_name(),
            "Send a follow-up message to a subagent. A running agent receives it at a safe turn \
             boundary. A completed agent starts another turn in the same session with its \
             existing history.",
            json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The ID of the agent to send input to"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send to the agent"
                    }
                },
                "required": ["agent_id", "message"]
            }),
        ),
        Arc::new(move |arguments, _context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let agent_id = required_str(&arguments, "agent_id")?;
                let message = required_str(&arguments, "message")?;
                supervisor.send_input(agent_id, message)?;
                Ok(format!("Message sent to agent {agent_id}"))
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// Waits for a child's current turn to finish, or for every child's.
fn wait_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::Wait.canonical_name(),
            "Wait for a subagent to complete, then use the result to synthesize the outcome for \
             the user. Omit agent_id to wait for every running subagent at once.",
            json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The ID of the agent to wait for. When omitted, waits for \
                                        every subagent and reports each one."
                    }
                }
            }),
        ),
        Arc::new(move |arguments, context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                // The context's token is the composite one: an interrupted
                // turn and an ended prompt both reach the wait, and both close
                // the child on their way out.
                let agent_id = arguments
                    .get("agent_id")
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        value.as_str().ok_or_else(|| {
                            ToolError::invalid_arguments("agent_id must be a string")
                        })
                    })
                    .transpose()?;
                if let Some(agent_id) = agent_id {
                    let result = supervisor
                        .wait_with_cancel(agent_id, &context.cancel)
                        .await?;
                    return Ok(format_wait_result(&result));
                }
                let results = supervisor.wait_all_with_cancel(&context.cancel).await?;
                if results.is_empty() {
                    return Ok("No subagents are running.".to_owned());
                }
                Ok(results
                    .iter()
                    .map(|(agent_id, result)| match result {
                        Ok(result) => {
                            format!("Agent {agent_id}: {}", format_wait_result(result))
                        }
                        Err(error) => format!("Agent {agent_id}: failed: {}", error.message()),
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n"))
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// How a finished child's turn reads to the parent.
fn format_wait_result(result: &super::SubagentResult) -> String {
    format!(
        "Agent completed (success: {}, turns: {})\n\n{}",
        result.success, result.turns_used, result.output
    )
}

/// Closes a child that is no longer needed.
fn close_agent_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::CloseAgent.canonical_name(),
            "Close a running or completed subagent that is no longer needed.",
            json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The ID of the agent to close"
                    }
                },
                "required": ["agent_id"]
            }),
        ),
        Arc::new(move |arguments, _context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let agent_id = required_str(&arguments, "agent_id")?;
                supervisor.close_agent(agent_id).await?;
                Ok(format!("Agent {agent_id} closed"))
            })
        }),
    )
    .with_source(ToolSource::Native)
}
