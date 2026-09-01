//! The Claude 5 harness's own tool adapters.
//!
//! What a tool *does* stays shared with the rest of pebble wherever the two
//! agree; this module narrows the schemas the model is shown and supplies the
//! four background-agent tools, whose lifecycle differs from pebble's own
//! spawn-and-wait family.
//!
//! Two things run through everything here. First, every schema is a **strict
//! object**: `additionalProperties: false`, so a field the model invented is
//! refused rather than ignored. Second, every definition is built under the
//! tool's *canonical* pebble name and renamed on the way into the registry, so
//! `Read`, `Bash` and `Agent` come from the vocabulary rather than from a
//! spelling repeated here.
//!
//! Every string in this file is model-visible contract.

use std::sync::Arc;
use std::time::Duration;

use lithos_llm::types::ToolDefinitionKind;
use serde_json::{Value, json};
use tokio::time;

use super::definition;
use crate::config::NativeToolOptions;
use crate::search::SearchProvider;
use crate::subagent::{SubagentResult, SubagentStatus, SubagentSupervisor, tree_position};
use crate::tool::{NativeTool, RegisteredTool, ToolError, required_str};
use crate::tools::shell::run_shell_command;
use crate::tools::{
    WebFetchSummarizer, make_edit_file_tool, make_read_file_tool, make_web_fetch_tool,
    make_web_search_tool, make_write_file_tool,
};
use crate::types::ToolSource;

/// The schema keeps `block` and `timeout` required, which is the Claude 5
/// contract, so these defaults only cover a model that omits them anyway.
const TASK_OUTPUT_DEFAULT_BLOCK: bool = true;
/// How long `TaskOutput` waits when the model names no timeout.
const TASK_OUTPUT_DEFAULT_TIMEOUT_MS: u64 = 30_000;
/// The longest `TaskOutput` waits, whatever the model asks for.
const TASK_OUTPUT_MAX_TIMEOUT_MS: u64 = 600_000;

/// The same tool, refusing a top-level field its schema does not name.
///
/// # Panics
///
/// Panics when `tool` is not described by an object-shaped JSON schema. Every
/// caller passes one of pebble's own function tools, so this is a programmer
/// error rather than a runtime condition.
#[must_use]
pub(crate) fn strict_object_tool(mut tool: RegisteredTool) -> RegisteredTool {
    let ToolDefinitionKind::Function { input_schema } = &mut tool.definition.kind else {
        panic!("the Claude 5 harness can only tighten a JSON-schema tool");
    };
    input_schema
        .as_object_mut()
        .expect("a built-in tool's schema is an object")
        .insert("additionalProperties".to_owned(), Value::Bool(false));
    tool
}

/// `Read`: pebble's file reader, strictly.
#[must_use]
pub(crate) fn make_read_tool() -> RegisteredTool {
    strict_object_tool(make_read_file_tool())
}

/// `Write`: pebble's file writer, strictly.
#[must_use]
pub(crate) fn make_write_tool() -> RegisteredTool {
    strict_object_tool(make_write_file_tool())
}

/// `Edit`: pebble's exact-string editor, strictly.
#[must_use]
pub(crate) fn make_edit_tool() -> RegisteredTool {
    strict_object_tool(make_edit_file_tool())
}

/// `Bash`, whose `timeout` is milliseconds and is bounded by the schema.
#[must_use]
pub(crate) fn make_bash_tool(options: &NativeToolOptions) -> RegisteredTool {
    let default_timeout_ms = options.default_command_timeout_ms;
    let max_timeout_ms = options.max_command_timeout_ms;
    RegisteredTool {
        definition: definition(
            NativeTool::Shell,
            format!(
                "Execute a Bash command in a fresh foreground non-login shell. Use this for \
                 searches, git inspection, builds, tests, package managers, and terminal \
                 operations. Prefer `rg` for content search and `rg --files` for file discovery. \
                 Working-directory and environment changes do not persist between calls. \
                 `timeout` is in milliseconds, defaults to {default_timeout_ms}, and is capped at \
                 {max_timeout_ms}."
            ),
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash source to evaluate."
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": max_timeout_ms,
                        "description": format!(
                            "Maximum runtime in milliseconds (default {default_timeout_ms})."
                        )
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of what the command does."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        ),
        executor:   Arc::new(move |arguments, context| {
            Box::pin(async move {
                let command = required_str(&arguments, "command")?;
                let timeout_ms = arguments
                    .get("timeout")
                    .and_then(Value::as_u64)
                    .unwrap_or(default_timeout_ms)
                    .min(max_timeout_ms);
                run_shell_command(&context, command, timeout_ms, None).await
            })
        }),
        source:     ToolSource::Native,
    }
}

/// `WebSearch`, which takes a query and nothing else.
///
/// The engine, the bounding and the rendered results are pebble's canonical
/// ones; only the schema narrows, because this harness's model was trained on
/// a search tool with no result count to name.
#[must_use]
pub(crate) fn make_claude5_web_search_tool(provider: Arc<dyn SearchProvider>) -> RegisteredTool {
    let mut tool = make_web_search_tool(provider);
    tool.definition = definition(
        NativeTool::WebSearch,
        "Search the web when current external information is needed. Returns result titles, URLs, \
         and descriptions; use WebFetch to inspect a specific URL.",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
    );
    tool
}

/// `WebFetch`, which requires the prompt pebble's own tool leaves optional.
#[must_use]
pub(crate) fn make_claude5_web_fetch_tool(
    summarizer: Option<Arc<WebFetchSummarizer>>,
) -> RegisteredTool {
    let mut tool = make_web_fetch_tool(summarizer);
    tool.definition = definition(
        NativeTool::WebFetch,
        "Fetch an HTTP or HTTPS URL and answer the supplied prompt from its contents.",
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The HTTP or HTTPS URL to fetch."
                },
                "prompt": {
                    "type": "string",
                    "description": "The question or extraction instruction to apply to the page."
                }
            },
            "required": ["url", "prompt"],
            "additionalProperties": false
        }),
    );
    tool
}

/// The four tools this harness drives its children with.
///
/// Pebble's own family — spawn, send input, wait, close — is a supervisor
/// handle the model polls; this one is Claude 5's background-agent contract,
/// where a finished child announces itself and the model only asks when it
/// deliberately wants to block. Both drive the same supervisor.
pub(crate) fn background_agent_tools(supervisor: &SubagentSupervisor) -> Vec<RegisteredTool> {
    vec![
        make_agent_tool(supervisor.clone()),
        make_task_output_tool(supervisor.clone()),
        make_task_stop_tool(supervisor.clone()),
        make_send_message_tool(supervisor.clone()),
    ]
}

/// How a finished child's turn reads to the parent.
fn format_agent_result(result: &SubagentResult) -> String {
    format!(
        "Agent completed (success: {}, turns: {})\n\n{}",
        result.success, result.turns_used, result.output
    )
}

/// `Agent`: starts a child, in the background unless told otherwise.
#[must_use]
fn make_agent_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool {
        definition: definition(
            NativeTool::BackgroundAgent,
            "Launch a child agent for an independent task. Agents run in the background by \
             default and notify the parent when they finish. Set run_in_background to false to \
             wait for the result synchronously.",
            json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short 3-5 word description of the task."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the agent to perform."
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": "Whether to return immediately (default true)."
                    }
                },
                "required": ["description", "prompt"],
                "additionalProperties": false
            }),
        ),
        executor:   Arc::new(move |arguments, context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let description = required_str(&arguments, "description")?;
                let prompt = required_str(&arguments, "prompt")?;
                let run_in_background = arguments
                    .get("run_in_background")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let (session_id, root_session_id) = tree_position(&context)?;

                if run_in_background {
                    let task_id = supervisor.spawn_with_parent_notification(
                        session_id,
                        root_session_id,
                        prompt.to_owned(),
                        description.to_owned(),
                    )?;
                    Ok(format!(
                        "Agent started in the background.\n\nTask ID: {task_id}"
                    ))
                } else {
                    let task_id =
                        supervisor.spawn(session_id, root_session_id, prompt.to_owned())?;
                    let result = supervisor
                        .wait_with_cancel(&task_id, &context.cancel)
                        .await?;
                    Ok(format_agent_result(&result))
                }
            })
        }),
        source:     ToolSource::Native,
    }
}

/// The value of an optional boolean argument.
fn optional_bool(arguments: &Value, key: &str, default: bool) -> Result<bool, ToolError> {
    match arguments.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| ToolError::invalid_arguments(format!("{key} must be a boolean"))),
    }
}

/// The value of an optional whole-number argument.
fn optional_u64(arguments: &Value, key: &str, default: u64) -> Result<u64, ToolError> {
    match arguments.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value.as_u64().ok_or_else(|| {
            ToolError::invalid_arguments(format!("{key} must be a non-negative integer"))
        }),
    }
}

/// `TaskOutput`: a background agent's status, or its final output.
#[must_use]
fn make_task_output_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool {
        definition: definition(
            NativeTool::AgentOutput,
            "Get a background agent's current status or wait for its final output. Automatic \
             completion notifications make ordinary polling unnecessary.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The background agent task ID."
                    },
                    "block": {
                        "type": "boolean",
                        "default": TASK_OUTPUT_DEFAULT_BLOCK,
                        "description": "Whether to wait for completion."
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": TASK_OUTPUT_MAX_TIMEOUT_MS,
                        "default": TASK_OUTPUT_DEFAULT_TIMEOUT_MS,
                        "description": "Maximum wait time in milliseconds."
                    }
                },
                "required": ["task_id", "block", "timeout"],
                "additionalProperties": false
            }),
        ),
        executor:   Arc::new(move |arguments, context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let task_id = required_str(&arguments, "task_id")?;
                let block = optional_bool(&arguments, "block", TASK_OUTPUT_DEFAULT_BLOCK)?;
                let timeout_ms =
                    optional_u64(&arguments, "timeout", TASK_OUTPUT_DEFAULT_TIMEOUT_MS)?;
                if timeout_ms > TASK_OUTPUT_MAX_TIMEOUT_MS {
                    return Err(ToolError::invalid_arguments(format!(
                        "timeout must be between 0 and {TASK_OUTPUT_MAX_TIMEOUT_MS} milliseconds"
                    )));
                }

                match supervisor.status(task_id) {
                    Some(SubagentStatus::Running) if !block => {
                        return Ok(format!("Agent {task_id} is still running."));
                    }
                    Some(SubagentStatus::Closing | SubagentStatus::Closed) => {
                        return Ok(format!("Agent {task_id} has been stopped."));
                    }
                    None => {
                        return Err(ToolError::invalid_arguments(format!(
                            "No agent found with id: {task_id} (it was never spawned)"
                        )));
                    }
                    // A finished agent answers from the cache and a running one
                    // is what the wait exists for, so both fall through to it.
                    Some(SubagentStatus::Finished { .. } | SubagentStatus::Running) => {}
                }

                // The wait answers a finished turn from the cache immediately,
                // so the branch above only has to decide whether to enter it.
                // Every ending but the timeout is the result this call asked
                // for, so the automatic notification for it is withdrawn.
                match time::timeout(
                    Duration::from_millis(timeout_ms),
                    supervisor.wait_with_cancel(task_id, &context.cancel),
                )
                .await
                {
                    Ok(result) => {
                        supervisor.suppress_parent_notification(task_id);
                        Ok(format_agent_result(&result?))
                    }
                    Err(_) => Ok(format!(
                        "Agent {task_id} is still running after waiting {timeout_ms} ms."
                    )),
                }
            })
        }),
        source:     ToolSource::Native,
    }
}

/// `TaskStop`: closes a background agent.
#[must_use]
fn make_task_stop_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool {
        definition: definition(
            NativeTool::StopAgent,
            "Stop a running or completed background agent by task ID.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The background agent task ID to stop."
                    }
                },
                "required": ["task_id"],
                "additionalProperties": false
            }),
        ),
        executor:   Arc::new(move |arguments, _context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let task_id = required_str(&arguments, "task_id")?;
                supervisor.close_agent(task_id).await?;
                Ok(format!("Agent {task_id} stopped."))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// `SendMessage`: gives a background agent more to do.
#[must_use]
fn make_send_message_tool(supervisor: SubagentSupervisor) -> RegisteredTool {
    RegisteredTool {
        definition: definition(
            NativeTool::MessageAgent,
            "Send additional instructions to a background agent by its task ID. A running agent \
             receives them at a safe turn boundary. A completed agent starts another turn in the \
             same session with its existing history.",
            json!({
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "The background agent task ID."
                    },
                    "message": {
                        "type": "string",
                        "description": "The follow-up message."
                    },
                    "summary": {
                        "type": "string",
                        "maxLength": 200,
                        "description": "Optional short preview of the message."
                    }
                },
                "required": ["to", "message"],
                "additionalProperties": false
            }),
        ),
        executor:   Arc::new(move |arguments, _context| {
            let supervisor = supervisor.clone();
            Box::pin(async move {
                let recipient = required_str(&arguments, "to")?;
                let message = required_str(&arguments, "message")?;
                supervisor.send_input(recipient, message)?;
                Ok(format!("Message sent to agent {recipient}."))
            })
        }),
        source:     ToolSource::Native,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::advanced::Session;
    use crate::profiles::tests::UnusedSearch;
    use crate::session::testing::TestSession;
    use crate::test_support::{MockEnvironment, ScriptedCall, text_response};
    use crate::tool::ToolContext;
    use crate::tools::testing::{context, schema_of};
    use crate::tools::{
        TodoRuntime, make_task_create_tool, make_task_get_tool, make_task_list_tool,
        make_task_update_tool,
    };
    use crate::types::{AgentProfileKind, ToolErrorKind};

    /// The properties `tool`'s schema names.
    fn property_names(tool: &RegisteredTool) -> BTreeSet<&str> {
        schema_of(tool)["properties"]
            .as_object()
            .expect("the schema names properties")
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// The properties `tool`'s schema requires.
    fn required_names(tool: &RegisteredTool) -> BTreeSet<&str> {
        schema_of(tool)["required"]
            .as_array()
            .map(|required| {
                required
                    .iter()
                    .map(|value| value.as_str().expect("a required name is a string"))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Asserts `tool` is a strict object naming exactly these properties.
    fn assert_schema(tool: &RegisteredTool, properties: &[&str], required: &[&str]) {
        let schema = schema_of(tool);
        assert_eq!(schema["type"], "object");
        assert_eq!(
            schema["additionalProperties"],
            Value::Bool(false),
            "{} accepts fields it does not name",
            tool.definition.name
        );
        assert_eq!(property_names(tool), properties.iter().copied().collect());
        assert_eq!(required_names(tool), required.iter().copied().collect());
    }

    #[test]
    fn core_adapter_schemas_match_the_claude5_contract() {
        let options = NativeToolOptions::for_profile(AgentProfileKind::Claude5);

        assert_schema(&make_read_tool(), &["file_path", "limit", "offset"], &[
            "file_path",
        ]);
        assert_schema(&make_write_tool(), &["content", "file_path"], &[
            "content",
            "file_path",
        ]);
        assert_schema(
            &make_edit_tool(),
            &["file_path", "new_string", "old_string", "replace_all"],
            &["file_path", "new_string", "old_string"],
        );
        let bash = make_bash_tool(&options);
        assert_schema(&bash, &["command", "description", "timeout"], &["command"]);
        assert_eq!(
            schema_of(&bash)["properties"]["timeout"]["maximum"],
            600_000
        );
        assert_schema(&make_claude5_web_fetch_tool(None), &["prompt", "url"], &[
            "prompt", "url",
        ]);
        assert_schema(
            &make_claude5_web_search_tool(Arc::new(UnusedSearch)),
            &["query"],
            &["query"],
        );

        let runtime = Arc::new(TodoRuntime::new());
        assert_schema(
            &strict_object_tool(make_task_create_tool(Arc::clone(&runtime))),
            &["activeForm", "description", "metadata", "subject"],
            &["description", "subject"],
        );
        assert_schema(
            &strict_object_tool(make_task_update_tool(Arc::clone(&runtime))),
            &[
                "activeForm",
                "addBlockedBy",
                "addBlocks",
                "description",
                "metadata",
                "owner",
                "status",
                "subject",
                "taskId",
            ],
            &["taskId"],
        );
        assert_schema(
            &strict_object_tool(make_task_get_tool(Arc::clone(&runtime))),
            &["taskId"],
            &["taskId"],
        );
        assert_schema(&strict_object_tool(make_task_list_tool(runtime)), &[], &[]);
    }

    /// The adapters narrow the schema and nothing else: a model that reads the
    /// canonical description would be told about a tool it does not have.
    #[test]
    fn the_narrowed_search_and_fetch_tools_keep_their_own_wording() {
        let search = make_claude5_web_search_tool(Arc::new(UnusedSearch));
        let fetch = make_claude5_web_fetch_tool(None);

        assert!(search.definition.description.contains("use WebFetch"));
        assert!(!search.definition.description.contains("web_fetch"));
        assert!(
            fetch
                .definition
                .description
                .contains("answer the supplied prompt")
        );
        // Both are built under the canonical identity, so the registry's
        // rename decides what the model calls them.
        assert_eq!(search.definition.name, "web_search");
        assert_eq!(fetch.definition.name, "web_fetch");
    }

    #[test]
    fn a_tightened_tool_keeps_everything_but_its_open_door() {
        let canonical = make_read_file_tool();
        let strict = make_read_tool();

        assert_eq!(strict.definition.name, canonical.definition.name);
        assert_eq!(
            strict.definition.description,
            canonical.definition.description
        );
        assert_eq!(
            schema_of(&canonical).get("additionalProperties"),
            None,
            "pebble's own tools accept what they do not name"
        );
        assert_eq!(
            schema_of(&strict)["additionalProperties"],
            Value::Bool(false)
        );
    }

    // --- The background-agent family ---

    /// A parent that can spawn, and answers `report` from every child turn.
    ///
    /// The parent never runs itself, so the whole script belongs to the
    /// children it spawns.
    fn parent_answering(report: &str) -> Session {
        let (session, _provider) = TestSession::new(vec![
            ScriptedCall::response(text_response(report)),
            ScriptedCall::response(text_response(report)),
        ])
        .with_subagents()
        .build();
        session
    }

    /// The supervisor `parent` drives its children with.
    fn supervisor_of(parent: &Session) -> SubagentSupervisor {
        parent
            .subagent_supervisor()
            .expect("the test session was given a factory")
            .clone()
    }

    /// A call made from inside `parent`.
    fn call_in(parent: &Session) -> ToolContext {
        context(MockEnvironment::linux()).with_session(parent.id(), parent.root_session_id())
    }

    #[tokio::test]
    async fn lifecycle_adapter_schemas_match_the_claude5_contract() {
        let parent = parent_answering("unused");
        let tools = background_agent_tools(&supervisor_of(&parent));
        let named = |name: &str| {
            tools
                .iter()
                .find(|tool| tool.definition.name == name)
                .unwrap_or_else(|| panic!("the harness offers {name}"))
        };

        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.definition.name.as_str())
                .collect::<Vec<_>>(),
            [
                "background_agent",
                "agent_output",
                "stop_agent",
                "message_agent"
            ],
            "the definitions carry canonical names for the registry to rename"
        );
        assert_schema(
            named("background_agent"),
            &["description", "prompt", "run_in_background"],
            &["description", "prompt"],
        );
        assert_schema(named("agent_output"), &["block", "task_id", "timeout"], &[
            "block", "task_id", "timeout",
        ]);
        assert_schema(named("stop_agent"), &["task_id"], &["task_id"]);
        let send_message = named("message_agent");
        assert_schema(send_message, &["message", "summary", "to"], &[
            "message", "to",
        ]);
        assert!(
            send_message
                .definition
                .description
                .contains("A completed agent")
        );
        assert!(send_message.definition.description.contains("same session"));
    }

    #[tokio::test]
    async fn agent_defaults_to_background_and_produces_a_parent_notification() {
        let parent = parent_answering("child report");
        let supervisor = supervisor_of(&parent);
        let tool = make_agent_tool(supervisor.clone());

        let output = (tool.executor)(
            json!({
                "description": "Inspect child",
                "prompt": "Inspect the child task"
            }),
            call_in(&parent),
        )
        .await
        .expect("the spawn succeeds");

        let task_id = output
            .strip_prefix("Agent started in the background.\n\nTask ID: ")
            .expect("a background agent answers with its task id");
        let notifications = supervisor
            .next_parent_notification_batch(&CancellationToken::new())
            .await
            .expect("the wait is not interrupted")
            .expect("the child finished");
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].agent_id, task_id);
        assert_eq!(notifications[0].description, "Inspect child");
        assert_eq!(
            notifications[0]
                .result
                .as_ref()
                .expect("the child succeeded")
                .output,
            "child report"
        );

        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn a_synchronous_agent_answers_with_its_childs_report() {
        let parent = parent_answering("child report");
        let supervisor = supervisor_of(&parent);
        let tool = make_agent_tool(supervisor.clone());

        let output = (tool.executor)(
            json!({
                "description": "Inspect child",
                "prompt": "Inspect the child task",
                "run_in_background": false
            }),
            call_in(&parent),
        )
        .await
        .expect("the child answers");

        assert_eq!(
            output,
            "Agent completed (success: true, turns: 2)\n\nchild report"
        );

        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn task_output_suppresses_a_racing_automatic_notification() {
        let parent = parent_answering("explicit report");
        let supervisor = supervisor_of(&parent);
        let task_id = supervisor
            .spawn_with_parent_notification(
                parent.id(),
                parent.root_session_id(),
                "Inspect".to_owned(),
                "Inspect explicitly".to_owned(),
            )
            .expect("the spawn succeeds");
        supervisor
            .wait_with_cancel(&task_id, &CancellationToken::new())
            .await
            .expect("the child answers");

        let tool = make_task_output_tool(supervisor.clone());
        let output = (tool.executor)(
            json!({"task_id": task_id, "block": false, "timeout": 0}),
            call_in(&parent),
        )
        .await
        .expect("a finished agent answers from its cache");

        assert!(output.contains("explicit report"), "{output}");
        assert!(
            supervisor
                .next_parent_notification_batch(&CancellationToken::new())
                .await
                .expect("the wait is not interrupted")
                .is_none(),
            "a result the model asked for is not also announced"
        );

        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn task_output_applies_the_schema_defaults_when_the_model_omits_them() {
        let parent = parent_answering("defaulted report");
        let supervisor = supervisor_of(&parent);
        let task_id = supervisor
            .spawn(parent.id(), parent.root_session_id(), "Inspect".to_owned())
            .expect("the spawn succeeds");
        supervisor
            .wait_with_cancel(&task_id, &CancellationToken::new())
            .await
            .expect("the child answers");

        let tool = make_task_output_tool(supervisor.clone());
        let output = (tool.executor)(json!({"task_id": task_id}), call_in(&parent))
            .await
            .expect("the defaults cover the omitted arguments");

        assert!(output.contains("defaulted report"), "{output}");

        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn task_output_rejects_a_wrongly_typed_optional_parameter() {
        let parent = parent_answering("unused");
        let supervisor = supervisor_of(&parent);
        let tool = make_task_output_tool(supervisor.clone());

        let error = (tool.executor)(
            json!({"task_id": "agent-1", "block": "yes"}),
            call_in(&parent),
        )
        .await
        .expect_err("a string is not a boolean");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "block must be a boolean");

        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn task_output_names_an_agent_that_was_never_spawned() {
        let parent = parent_answering("unused");
        let supervisor = supervisor_of(&parent);
        let tool = make_task_output_tool(supervisor.clone());

        let error = (tool.executor)(json!({"task_id": "agent-1"}), call_in(&parent))
            .await
            .expect_err("there is no such agent");

        assert_eq!(
            error.message(),
            "No agent found with id: agent-1 (it was never spawned)"
        );

        supervisor.shutdown_all().await;
    }

    #[tokio::test]
    async fn stopping_and_messaging_an_agent_answer_the_way_claude_5_reads_them() {
        let parent = parent_answering("child report");
        let supervisor = supervisor_of(&parent);
        let task_id = supervisor
            .spawn(parent.id(), parent.root_session_id(), "Inspect".to_owned())
            .expect("the spawn succeeds");
        supervisor
            .wait_with_cancel(&task_id, &CancellationToken::new())
            .await
            .expect("the child answers");

        let message = make_send_message_tool(supervisor.clone());
        let sent = (message.executor)(
            json!({"to": task_id, "message": "Keep going"}),
            call_in(&parent),
        )
        .await
        .expect("a finished agent takes another turn");
        assert_eq!(sent, format!("Message sent to agent {task_id}."));

        let stop = make_task_stop_tool(supervisor.clone());
        let stopped = (stop.executor)(json!({"task_id": task_id}), call_in(&parent))
            .await
            .expect("the agent stops");
        assert_eq!(stopped, format!("Agent {task_id} stopped."));

        supervisor.shutdown_all().await;
    }
}
