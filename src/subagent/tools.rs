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
use crate::tool::{RegisteredTool, ToolError, required_str};
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
    RegisteredTool {
        definition: ToolDefinition::function(
            "spawn_agent",
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
        executor:   Arc::new(move |arguments, context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let task = required_str(&arguments, "task")?;
                let Some(session_id) = context.session_id.as_deref() else {
                    return Err(ToolError::execution(
                        "A subagent can only be spawned from inside a session",
                    ));
                };
                // The child inherits the root of this tree, so root-scoped
                // tools — one shared todo list — cover the whole tree.
                let root_session_id = context.root_session_id.as_deref().unwrap_or(session_id);
                supervisor.spawn(session_id, root_session_id, task.to_owned())
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Gives a child more to do.
fn send_input_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "send_input",
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
        executor:   Arc::new(move |arguments, _context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let agent_id = required_str(&arguments, "agent_id")?;
                let message = required_str(&arguments, "message")?;
                supervisor.send_input(agent_id, message)?;
                Ok(format!("Message sent to agent {agent_id}"))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Waits for a child's current turn to finish.
fn wait_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "wait",
            "Wait for a subagent to complete, then use the result to synthesize the outcome for \
             the user.",
            json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The ID of the agent to wait for"
                    }
                },
                "required": ["agent_id"]
            }),
        ),
        executor:   Arc::new(move |arguments, context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let agent_id = required_str(&arguments, "agent_id")?;
                // The context's token is the composite one: an interrupted
                // round and an ended run both reach the wait, and both close
                // the child on their way out.
                let result = supervisor
                    .wait_with_cancel(agent_id, &context.cancel)
                    .await?;
                Ok(format!(
                    "Agent completed (success: {}, turns: {})\n\n{}",
                    result.success, result.turns_used, result.output
                ))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Closes a child that is no longer needed.
fn close_agent_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "close_agent",
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
        executor:   Arc::new(move |arguments, _context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let agent_id = required_str(&arguments, "agent_id")?;
                supervisor.close_agent(agent_id).await?;
                Ok(format!("Agent {agent_id} closed"))
            })
        }),
        source:     ToolSource::Native,
    }
}
