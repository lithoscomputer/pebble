//! Keeping a plan the model wrote, in the three shapes models expect.
//!
//! Three harnesses ask for the same thing in different words. Codex sends the
//! whole plan every time ([`make_update_plan_tool`]); Kimi Code does the same
//! with its own field names and spells the finished status `done`
//! ([`make_todo_list_tool`]); Claude works one task at a time
//! ([`make_task_create_tool`], [`make_task_update_tool`],
//! [`make_task_get_tool`], [`make_task_list_tool`]). One [`TodoRuntime`] holds
//! the lists all of them write, so the events an application sees are the same
//! whichever model is running.
//!
//! The two scopes differ on purpose. A plan belongs to the session that wrote
//! it, so a child planning its own work does not overwrite its parent's; a task
//! list belongs to the whole session tree, because Claude's task tools are how
//! a parent and its children divide work they can both see.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::tool::{NativeTool, RegisteredTool, ToolContext, ToolError, required_str};
use crate::types::{TodoListKind, TodoProjection, TodoStatus, TodoUpdatedProps, ToolSource};

mod runtime;
#[cfg(test)]
mod testing;

pub use self::runtime::TodoRuntime;

/// The list identifier for a session-scoped tool.
///
/// # Errors
///
/// Returns a [`ToolError`] of kind
/// [`Unavailable`](crate::tools::ToolErrorKind::Unavailable) outside a session,
/// which is the only place these tools have no list to write.
fn session_todo_scope(
    ctx: &ToolContext,
    kind: TodoListKind,
    tool_name: &str,
) -> Result<String, ToolError> {
    ctx.session_id
        .as_ref()
        .map(|session_id| kind.list_id(session_id))
        .ok_or_else(|| ToolError::unavailable(format!("{tool_name} requires an active session")))
}

/// The list identifier the task tools share across a session tree.
///
/// The root of the tree names the list, so a child writes to its parent's. A
/// session with no root recorded falls back to its own identity, which is what
/// a session outside a tree is.
fn anthropic_task_scope(ctx: &ToolContext) -> Result<String, ToolError> {
    ctx.root_session_id
        .as_ref()
        .or(ctx.session_id.as_ref())
        .map(|session_id| TodoListKind::AnthropicTasks.list_id(session_id))
        .ok_or_else(|| ToolError::unavailable("task tools require an active session"))
}

/// One status as the plan and task tools spell it.
///
/// `allow_deleted` is false for `update_plan`, which reconciles a whole list
/// and has no way to say "delete this one".
fn parse_status(value: &str, allow_deleted: bool) -> Result<TodoStatus, ToolError> {
    match value {
        "pending" => Ok(TodoStatus::Pending),
        "in_progress" => Ok(TodoStatus::InProgress),
        "completed" => Ok(TodoStatus::Completed),
        "deleted" if allow_deleted => Ok(TodoStatus::Deleted),
        _ if allow_deleted => Err(ToolError::invalid_arguments(format!(
            "Invalid status `{value}` (expected pending|in_progress|completed|deleted)"
        ))),
        _ => Err(ToolError::invalid_arguments(format!(
            "Invalid status `{value}` (expected pending|in_progress|completed)"
        ))),
    }
}

const TASK_CREATE_DESCRIPTION: &str = "Create pending tasks in the current session. \
Use concise subjects, descriptions, optional activeForm text, and metadata. Check \
TaskList first to avoid duplicate tasks.";

const TASK_UPDATE_DESCRIPTION: &str = "Update an existing task's status, text, owner, \
metadata, or dependencies. Valid statuses are pending, in_progress, completed, and \
deleted. After completing a task, call TaskList to find newly unblocked work.";

const TASK_LIST_DESCRIPTION: &str = "List tasks for the current session, including \
status, owner, and blocking dependencies. Use TaskGet with a taskId for full \
description and dependency details.";

const TASK_GET_DESCRIPTION: &str = "Get one task by taskId, including subject, status, \
description, owner, blockedBy, and blocks.";

/// The identity of a todo in a list that is replaced whole.
///
/// A tool that submits the entire plan every time has no identifiers to send,
/// so the text is the identity: an entry whose words did not change keeps its
/// identifier, and an application projecting the events sees one update rather
/// than a delete and a create.
fn todo_text_id(list_id: &str, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(list_id.as_bytes());
    hasher.update(b"\x00");
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(16);
    for byte in &digest[..8] {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// One entry of a whole-list submission.
struct ReplacementTodo {
    id:      String,
    subject: String,
    status:  TodoStatus,
}

/// Makes the stored list match `incoming`, announcing only what changed.
///
/// An entry that is gone is deleted, one whose status, order, and text are all
/// unchanged is left alone, and everything else is updated or created. Doing
/// this by comparison rather than by clearing and rewriting is what keeps a
/// resubmitted plan from looking like a whole new plan.
fn reconcile_replacement_list(
    runtime: &TodoRuntime,
    ctx: &ToolContext,
    kind: TodoListKind,
    list_id: &str,
    incoming: &[ReplacementTodo],
) {
    let previous = runtime
        .snapshot(list_id)
        .map(|list| list.items)
        .unwrap_or_default();
    let previous_by_id: HashMap<&str, &TodoProjection> = previous
        .iter()
        .map(|todo| (todo.id.as_str(), todo))
        .collect();
    let incoming_ids: HashSet<&str> = incoming.iter().map(|todo| todo.id.as_str()).collect();

    for todo in &previous {
        if !incoming_ids.contains(todo.id.as_str()) {
            runtime.delete(ctx, kind, list_id.to_owned(), todo.id.clone());
        }
    }

    for (index, todo) in incoming.iter().enumerate() {
        let order = u32::try_from(index).unwrap_or(u32::MAX);
        match previous_by_id.get(todo.id.as_str()) {
            Some(previous)
                if previous.status == todo.status
                    && previous.order == order
                    && previous.subject == todo.subject => {}
            Some(_) => {
                runtime.update(ctx, TodoUpdatedProps {
                    status: Some(todo.status),
                    order: Some(order),
                    subject: Some(todo.subject.clone()),
                    ..TodoUpdatedProps::new(list_id, kind, &todo.id)
                });
            }
            None => {
                let mut projection =
                    TodoProjection::new(todo.id.clone(), order, todo.subject.clone());
                projection.status = todo.status;
                runtime.create(ctx, kind, list_id.to_owned(), projection);
            }
        }
    }
}

/// Replaces the whole plan, the way Codex asks for it.
///
/// The list is scoped to the calling session, so a child plans its own work.
#[must_use]
pub fn make_update_plan_tool(runtime: Arc<TodoRuntime>) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::UpdatePlan.canonical_name(),
            "Update the multi-step plan for the current task. Submit the entire plan; existing \
             steps are reconciled by exact step text.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Optional natural-language note about why the plan changed"
                    },
                    "plan": {
                        "type": "array",
                        "description": "Ordered list of plan steps, each with a status",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["step", "status"]
                        }
                    }
                },
                "required": ["plan"]
            }),
        ),
        Arc::new(move |args, ctx| {
            let runtime = Arc::clone(&runtime);
            Box::pin(async move {
                let list_id = session_todo_scope(&ctx, TodoListKind::OpenAiPlan, "update_plan")?;
                let plan = args.get("plan").and_then(Value::as_array).ok_or_else(|| {
                    ToolError::invalid_arguments("Missing required parameter: plan")
                })?;

                // Identifiers are derived from the step text, so two steps
                // that read the same would be one entry. Saying so is better
                // than silently keeping one of them.
                let mut incoming = Vec::with_capacity(plan.len());
                let mut seen_steps: HashSet<&str> = HashSet::with_capacity(plan.len());
                for (index, entry) in plan.iter().enumerate() {
                    let step = entry.get("step").and_then(Value::as_str).ok_or_else(|| {
                        ToolError::invalid_arguments(format!("plan[{index}] is missing `step`"))
                    })?;
                    let status = entry.get("status").and_then(Value::as_str).ok_or_else(|| {
                        ToolError::invalid_arguments(format!("plan[{index}] is missing `status`"))
                    })?;
                    let status = parse_status(status, false)?;
                    if !seen_steps.insert(step) {
                        return Err(ToolError::invalid_arguments(format!(
                            "Duplicate plan step `{step}` — step text must be unique"
                        )));
                    }
                    incoming.push(ReplacementTodo {
                        id: todo_text_id(&list_id, step),
                        subject: step.to_owned(),
                        status,
                    });
                }

                reconcile_replacement_list(
                    &runtime,
                    &ctx,
                    TodoListKind::OpenAiPlan,
                    &list_id,
                    &incoming,
                );

                Ok("Plan updated".to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// One status as Kimi Code spells it.
fn parse_kimi_status(value: &str) -> Result<TodoStatus, ToolError> {
    match value {
        "pending" => Ok(TodoStatus::Pending),
        "in_progress" => Ok(TodoStatus::InProgress),
        "done" => Ok(TodoStatus::Completed),
        _ => Err(ToolError::invalid_arguments(format!(
            "Invalid status `{value}` (expected pending|in_progress|done)"
        ))),
    }
}

/// How Kimi Code writes a status back.
///
/// It has three words for four statuses, and the one it lacks — `deleted` — is
/// never in a list it reads, because a deleted todo is removed.
const fn kimi_status_name(status: TodoStatus) -> &'static str {
    match status {
        TodoStatus::Pending => "pending",
        TodoStatus::InProgress => "in_progress",
        TodoStatus::Completed | TodoStatus::Deleted => "done",
    }
}

/// The list as Kimi Code reads it.
fn render_kimi_todos<'a>(items: impl IntoIterator<Item = (TodoStatus, &'a str)>) -> String {
    let mut items = items.into_iter().peekable();
    if items.peek().is_none() {
        return "The todo list is empty.".to_owned();
    }
    let mut out = String::new();
    for (status, subject) in items {
        let _ = writeln!(out, "[{}] {subject}", kimi_status_name(status));
    }
    out.truncate(out.trim_end().len());
    out
}

/// Replaces the whole todo list, the way Kimi Code asks for it.
///
/// One tool reads and writes, which is the surface Kimi models are trained
/// against: omit `todos` to read, send `[]` to clear, send a list to replace.
/// Entries carry a title and a status and nothing else, and the finished status
/// is spelled `done`.
///
/// The same [`TodoRuntime`] backs it, and entries are identified by their text
/// exactly as `update_plan`'s are, so a resubmitted list keeps the identity of
/// everything that did not change.
#[must_use]
pub fn make_todo_list_tool(runtime: Arc<TodoRuntime>) -> RegisteredTool {
    RegisteredTool::new(ToolDefinition::function(
            NativeTool::TodoList.canonical_name(),
            "Maintain a structured TODO list for the current task. Use it proactively for \
             multi-step work. Pass `todos` to replace the entire list, omit `todos` to read the \
             current list without changing it, and pass an empty array to clear it. Keep exactly \
             one item `in_progress` while work is underway, and mark an item `done` as soon as it \
             is finished rather than batching completions at the end.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The updated todo list. Omit to read the current list without making changes. Pass an empty array to clear the list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Short, actionable title for the todo."
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done"],
                                    "description": "Current status of the todo."
                                }
                            },
                            "required": ["title", "status"]
                        }
                    }
                }
            }),
        ), Arc::new(move |args, ctx| {
            let runtime = Arc::clone(&runtime);
            Box::pin(async move {
                let list_id = session_todo_scope(&ctx, TodoListKind::KimiTodos, "TodoList")?;

                // Reading is `todos` left out entirely, which is different
                // from sending an empty list to clear it.
                let Some(todos) = args.get("todos") else {
                    let items = runtime
                        .snapshot(&list_id)
                        .map(|list| list.items)
                        .unwrap_or_default();
                    return Ok(render_kimi_todos(
                        items
                            .iter()
                            .map(|todo| (todo.status, todo.subject.as_str())),
                    ));
                };
                let todos = todos
                    .as_array()
                    .ok_or_else(|| ToolError::invalid_arguments("`todos` must be an array"))?;

                let mut incoming = Vec::with_capacity(todos.len());
                let mut seen: HashSet<&str> = HashSet::with_capacity(todos.len());
                for (index, entry) in todos.iter().enumerate() {
                    let title = entry.get("title").and_then(Value::as_str).ok_or_else(|| {
                        ToolError::invalid_arguments(format!("todos[{index}] is missing `title`"))
                    })?;
                    let status = entry.get("status").and_then(Value::as_str).ok_or_else(|| {
                        ToolError::invalid_arguments(format!("todos[{index}] is missing `status`"))
                    })?;
                    let status = parse_kimi_status(status)?;
                    if !seen.insert(title) {
                        return Err(ToolError::invalid_arguments(format!(
                            "Duplicate todo `{title}` — titles must be unique"
                        )));
                    }
                    incoming.push(ReplacementTodo {
                        id: todo_text_id(&list_id, title),
                        subject: title.to_owned(),
                        status,
                    });
                }

                reconcile_replacement_list(
                    &runtime,
                    &ctx,
                    TodoListKind::KimiTodos,
                    &list_id,
                    &incoming,
                );
                Ok(render_kimi_todos(
                    incoming
                        .iter()
                        .map(|todo| (todo.status, todo.subject.as_str())),
                ))
            })
        })).with_source(ToolSource::Native)
}

/// One optional string argument, or `None` when it is absent or not a string.
fn optional_string(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(ToOwned::to_owned)
}

/// One optional list-of-strings argument, ignoring entries that are not
/// strings.
fn optional_string_vec(args: &Value, key: &str) -> Option<Vec<String>> {
    args.get(key).and_then(Value::as_array).map(|values| {
        values
            .iter()
            .filter_map(|value| value.as_str().map(ToOwned::to_owned))
            .collect()
    })
}

/// The `metadata` argument as a map, or an empty one.
fn metadata_map(args: &Value) -> BTreeMap<String, Value> {
    args.get("metadata")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default()
}

/// Appends `"\n{label}: #a, #b"`, or nothing when there is nothing to name.
fn append_task_refs(out: &mut String, label: &str, task_ids: &[String]) {
    if task_ids.is_empty() {
        return;
    }
    let _ = write!(out, "\n{label}: ");
    for (index, task_id) in task_ids.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "#{task_id}");
    }
}

/// One task, written out in full.
fn format_task_details(todo: &TodoProjection) -> String {
    let mut out = format!(
        "Task #{}: {}\nStatus: {}\nDescription: {}",
        todo.id, todo.subject, todo.status, todo.description
    );
    if let Some(owner) = todo.owner.as_ref() {
        let _ = write!(out, "\nOwner: {owner}");
    }
    append_task_refs(&mut out, "Blocked by", &todo.blocked_by);
    append_task_refs(&mut out, "Blocks", &todo.blocks);
    out
}

/// Creates one task in the session tree's shared list.
#[must_use]
pub fn make_task_create_tool(runtime: Arc<TodoRuntime>) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::TaskCreate.canonical_name(),
            TASK_CREATE_DESCRIPTION,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "subject":     {"type": "string"},
                    "description": {"type": "string"},
                    "activeForm":  {"type": "string"},
                    "metadata":    {"type": "object", "additionalProperties": true}
                },
                "required": ["subject", "description"]
            }),
        ),
        Arc::new(move |args, ctx| {
            let runtime = Arc::clone(&runtime);
            Box::pin(async move {
                let list_id = anthropic_task_scope(&ctx)?;
                let subject = required_str(&args, "subject")?.to_owned();
                let description = required_str(&args, "description")?.to_owned();
                let task_id = runtime.next_task_id(&list_id);
                let order = u32::try_from(task_id.saturating_sub(1)).unwrap_or(u32::MAX);

                let mut projection =
                    TodoProjection::new(task_id.to_string(), order, subject.clone());
                projection.description = description;
                projection.active_form = optional_string(&args, "activeForm");
                projection.metadata = metadata_map(&args);

                runtime.create(&ctx, TodoListKind::AnthropicTasks, list_id, projection);

                Ok(format!("Task #{task_id} created successfully: {subject}"))
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// Changes one task.
///
/// An argument that is absent leaves its field alone, and a JSON `null` clears
/// it — the distinction Claude's task tools rely on to let a caller remove an
/// owner without knowing what it was.
#[must_use]
pub fn make_task_update_tool(runtime: Arc<TodoRuntime>) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::TaskUpdate.canonical_name(),
            TASK_UPDATE_DESCRIPTION,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "taskId":       {"type": "string"},
                    "subject":      {"type": "string"},
                    "description":  {"type": "string"},
                    "activeForm":   {"type": "string"},
                    "status":       {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "deleted"]
                    },
                    "owner":        {"type": "string"},
                    "addBlocks":    {"type": "array", "items": {"type": "string"}},
                    "addBlockedBy": {"type": "array", "items": {"type": "string"}},
                    "metadata":     {"type": "object", "additionalProperties": true}
                },
                "required": ["taskId"]
            }),
        ),
        Arc::new(move |args, ctx| {
            let runtime = Arc::clone(&runtime);
            Box::pin(async move {
                let list_id = anthropic_task_scope(&ctx)?;
                let task_id = required_str(&args, "taskId")?.to_owned();

                let status = args
                    .get("status")
                    .and_then(Value::as_str)
                    .map(|status| parse_status(status, true))
                    .transpose()?;

                let props = TodoUpdatedProps {
                    status,
                    subject: optional_string(&args, "subject"),
                    description: optional_string(&args, "description"),
                    active_form: args
                        .get("activeForm")
                        .map(|value| value.as_str().map(ToOwned::to_owned)),
                    owner: args
                        .get("owner")
                        .map(|value| value.as_str().map(ToOwned::to_owned)),
                    add_blocks: optional_string_vec(&args, "addBlocks"),
                    add_blocked_by: optional_string_vec(&args, "addBlockedBy"),
                    metadata_patch: metadata_map(&args),
                    ..TodoUpdatedProps::new(&list_id, TodoListKind::AnthropicTasks, &task_id)
                };

                if runtime.update(&ctx, props) {
                    Ok(format!("Task #{task_id} updated"))
                } else {
                    // Not an error: Claude's task tools answer a missing task
                    // by saying so, and a model that asked about a task
                    // someone else finished has nothing to repair.
                    Ok("Task not found".to_owned())
                }
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// Reads one task in full.
#[must_use]
pub fn make_task_get_tool(runtime: Arc<TodoRuntime>) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::TaskGet.canonical_name(),
            TASK_GET_DESCRIPTION,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "taskId": {"type": "string"}
                },
                "required": ["taskId"]
            }),
        ),
        Arc::new(move |args, ctx| {
            let runtime = Arc::clone(&runtime);
            Box::pin(async move {
                let list_id = anthropic_task_scope(&ctx)?;
                let task_id = required_str(&args, "taskId")?.to_owned();

                let Some(snapshot) = runtime.snapshot(&list_id) else {
                    return Ok("Task not found".to_owned());
                };
                let Some(todo) = snapshot.get(&task_id) else {
                    return Ok("Task not found".to_owned());
                };

                Ok(format_task_details(todo))
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// Lists the session tree's tasks, one to a line.
#[must_use]
pub fn make_task_list_tool(runtime: Arc<TodoRuntime>) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            NativeTool::TaskList.canonical_name(),
            TASK_LIST_DESCRIPTION,
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ),
        Arc::new(move |_args, ctx| {
            let runtime = Arc::clone(&runtime);
            Box::pin(async move {
                let list_id = anthropic_task_scope(&ctx)?;
                let snapshot = runtime.snapshot(&list_id);
                let items: &[TodoProjection] = snapshot.as_ref().map_or(&[], |list| &list.items);
                if items.is_empty() {
                    return Ok("No tasks found".to_owned());
                }
                // Built once, so the per-row blocker filter costs one lookup
                // rather than a scan of the whole list.
                let status_by_id: HashMap<&str, TodoStatus> = items
                    .iter()
                    .map(|todo| (todo.id.as_str(), todo.status))
                    .collect();

                let mut out = String::new();
                for todo in items {
                    let _ = write!(out, "#{} [{}] {}", todo.id, todo.status, todo.subject);
                    if let Some(owner) = todo.owner.as_ref() {
                        let _ = write!(out, " (owner: {owner})");
                    }
                    // Only blockers that are still open: a finished one no
                    // longer holds anything up, and naming it would read as a
                    // reason not to start.
                    let mut blockers = todo.blocked_by.iter().filter(|id| {
                        status_by_id
                            .get(id.as_str())
                            .copied()
                            .is_none_or(|status| status != TodoStatus::Completed)
                    });
                    if let Some(first) = blockers.next() {
                        let _ = write!(out, " (blocked by: {first}");
                        for blocker in blockers {
                            let _ = write!(out, ", {blocker}");
                        }
                        out.push(')');
                    }
                    out.push('\n');
                }
                Ok(out.trim_end().to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native)
}

#[cfg(test)]
mod kimi_tests {
    use serde_json::json;

    use super::testing::context_for;
    use super::*;
    use crate::types::ToolErrorKind;

    async fn call(tool: &RegisteredTool, args: Value) -> Result<String, ToolError> {
        (tool.executor)(args, context_for("ses_kimi", "ses_kimi")).await
    }

    #[tokio::test]
    async fn replaces_the_whole_list_and_reads_it_back() {
        let tool = make_todo_list_tool(Arc::new(TodoRuntime::new()));

        call(
            &tool,
            json!({"todos": [
                {"title": "read the config", "status": "done"},
                {"title": "patch the parser", "status": "in_progress"},
                {"title": "add a test", "status": "pending"}
            ]}),
        )
        .await
        .expect("the list is replaced");

        // Reading is `todos` left out entirely.
        let listed = call(&tool, json!({})).await.expect("the list reads back");
        assert!(listed.contains("[done] read the config"), "{listed}");
        assert!(
            listed.contains("[in_progress] patch the parser"),
            "{listed}"
        );

        // A shorter list drops the entries it left out.
        call(
            &tool,
            json!({"todos": [{"title": "add a test", "status": "done"}]}),
        )
        .await
        .expect("the list is replaced");
        let listed = call(&tool, json!({})).await.expect("the list reads back");
        assert!(listed.contains("[done] add a test"), "{listed}");
        assert!(!listed.contains("patch the parser"), "{listed}");
    }

    #[tokio::test]
    async fn empty_array_clears_the_list() {
        let tool = make_todo_list_tool(Arc::new(TodoRuntime::new()));

        call(
            &tool,
            json!({"todos": [{"title": "x", "status": "pending"}]}),
        )
        .await
        .expect("the list is replaced");
        call(&tool, json!({"todos": []}))
            .await
            .expect("the list is cleared");

        assert_eq!(
            call(&tool, json!({})).await.expect("the list reads back"),
            "The todo list is empty."
        );
    }

    /// Kimi Code spells the finished status `done`; `completed` is the
    /// Anthropic and Codex spelling, and accepting it here would teach the
    /// model a word its own harness does not use.
    #[tokio::test]
    async fn status_vocabulary_is_kimi_codes() {
        let tool = make_todo_list_tool(Arc::new(TodoRuntime::new()));

        let error = call(
            &tool,
            json!({"todos": [{"title": "x", "status": "completed"}]}),
        )
        .await
        .expect_err("`completed` is not a Kimi status");

        assert!(
            error
                .message()
                .contains("expected pending|in_progress|done"),
            "{}",
            error.message()
        );
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    #[tokio::test]
    async fn duplicate_titles_are_rejected() {
        let tool = make_todo_list_tool(Arc::new(TodoRuntime::new()));

        let error = call(
            &tool,
            json!({"todos": [
                {"title": "same", "status": "pending"},
                {"title": "same", "status": "done"}
            ]}),
        )
        .await
        .expect_err("two entries cannot share a title");

        assert!(
            error.message().contains("must be unique"),
            "{}",
            error.message()
        );
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::testing::context_for;
    use super::*;
    use crate::test_support::MockEnvironment;
    use crate::tools::testing::context;
    use crate::types::ToolErrorKind;

    fn openai_list(session: &str) -> String {
        TodoListKind::OpenAiPlan.list_id(session)
    }

    fn anthropic_list(session: &str) -> String {
        TodoListKind::AnthropicTasks.list_id(session)
    }

    async fn call(tool: &RegisteredTool, args: Value) -> Result<String, ToolError> {
        (tool.executor)(args, context_for("ses_a", "ses_a")).await
    }

    // --- update_plan ---

    #[tokio::test]
    async fn update_plan_creates_initial_steps() {
        let runtime = Arc::new(TodoRuntime::new());
        let tool = make_update_plan_tool(Arc::clone(&runtime));

        let out = call(
            &tool,
            json!({"plan": [
                {"step": "a", "status": "pending"},
                {"step": "b", "status": "in_progress"}
            ]}),
        )
        .await
        .expect("the plan is accepted");

        assert_eq!(out, "Plan updated");
        let list = runtime
            .snapshot(&openai_list("ses_a"))
            .expect("the plan exists");
        assert_eq!(list.items.len(), 2);
        assert_eq!(list.items[0].subject, "a");
        assert_eq!(list.items[1].subject, "b");
        assert_eq!(list.items[1].status, TodoStatus::InProgress);
    }

    #[tokio::test]
    async fn update_plan_updates_status_and_order() {
        let runtime = Arc::new(TodoRuntime::new());
        let tool = make_update_plan_tool(Arc::clone(&runtime));

        call(
            &tool,
            json!({"plan": [
                {"step": "a", "status": "pending"},
                {"step": "b", "status": "pending"}
            ]}),
        )
        .await
        .expect("the plan is accepted");
        call(
            &tool,
            json!({"plan": [
                {"step": "b", "status": "in_progress"},
                {"step": "a", "status": "completed"}
            ]}),
        )
        .await
        .expect("the plan is accepted");

        let list = runtime
            .snapshot(&openai_list("ses_a"))
            .expect("the plan exists");
        assert_eq!(list.items.len(), 2);
        assert_eq!(list.items[0].subject, "b");
        assert_eq!(list.items[0].status, TodoStatus::InProgress);
        assert_eq!(list.items[1].subject, "a");
        assert_eq!(list.items[1].status, TodoStatus::Completed);
    }

    #[tokio::test]
    async fn update_plan_deletes_omitted_steps() {
        let runtime = Arc::new(TodoRuntime::new());
        let tool = make_update_plan_tool(Arc::clone(&runtime));

        call(
            &tool,
            json!({"plan": [
                {"step": "a", "status": "pending"},
                {"step": "b", "status": "pending"},
                {"step": "c", "status": "pending"}
            ]}),
        )
        .await
        .expect("the plan is accepted");
        call(
            &tool,
            json!({"plan": [{"step": "b", "status": "completed"}]}),
        )
        .await
        .expect("the plan is accepted");

        let list = runtime
            .snapshot(&openai_list("ses_a"))
            .expect("the plan exists");
        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].subject, "b");
    }

    #[tokio::test]
    async fn update_plan_rejects_duplicate_steps() {
        let tool = make_update_plan_tool(Arc::new(TodoRuntime::new()));

        let error = call(
            &tool,
            json!({"plan": [
                {"step": "same", "status": "pending"},
                {"step": "same", "status": "completed"}
            ]}),
        )
        .await
        .expect_err("two steps cannot read the same");

        assert!(
            error.message().contains("Duplicate plan step"),
            "{}",
            error.message()
        );
    }

    #[tokio::test]
    async fn update_plan_rejects_a_status_it_cannot_reconcile() {
        let tool = make_update_plan_tool(Arc::new(TodoRuntime::new()));

        let error = call(&tool, json!({"plan": [{"step": "a", "status": "deleted"}]}))
            .await
            .expect_err("`deleted` is not a plan status");

        assert_eq!(
            error.message(),
            "Invalid status `deleted` (expected pending|in_progress|completed)"
        );
    }

    /// A plan belongs to the session that wrote it, so a child planning its own
    /// work leaves its parent's plan alone.
    #[tokio::test]
    async fn update_plan_subagent_writes_a_different_list_than_its_parent() {
        let runtime = Arc::new(TodoRuntime::new());
        let tool = make_update_plan_tool(Arc::clone(&runtime));

        (tool.executor)(
            json!({"plan": [{"step": "parent_step", "status": "pending"}]}),
            context_for("ses_parent", "ses_parent"),
        )
        .await
        .expect("the plan is accepted");
        (tool.executor)(
            json!({"plan": [{"step": "child_step", "status": "pending"}]}),
            context_for("ses_child", "ses_parent"),
        )
        .await
        .expect("the plan is accepted");

        let parent = runtime
            .snapshot(&openai_list("ses_parent"))
            .expect("the parent's plan exists");
        let child = runtime
            .snapshot(&openai_list("ses_child"))
            .expect("the child's plan exists");
        assert_eq!(parent.items.len(), 1);
        assert_eq!(parent.items[0].subject, "parent_step");
        assert_eq!(child.items.len(), 1);
        assert_eq!(child.items[0].subject, "child_step");
    }

    // --- the task tools ---

    #[tokio::test]
    async fn task_create_returns_a_numeric_id_and_message() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));

        let out = call(
            &create,
            json!({"subject": "Do thing", "description": "details"}),
        )
        .await
        .expect("the task is created");

        assert_eq!(out, "Task #1 created successfully: Do thing");
        let list = runtime
            .snapshot(&anthropic_list("ses_a"))
            .expect("the list exists");
        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].id, "1");
        assert_eq!(list.items[0].subject, "Do thing");
        assert_eq!(list.items[0].description, "details");
    }

    /// Four tool descriptions share one context window with everything else a
    /// session carries, so they say what the tool is for and stop.
    #[test]
    fn the_task_tool_descriptions_are_concise() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));
        let list = make_task_list_tool(runtime);

        assert!(
            create
                .definition
                .description
                .contains("Create pending tasks")
        );
        assert!(create.definition.description.contains("activeForm"));
        assert!(update.definition.description.contains("pending"));
        assert!(update.definition.description.contains("deleted"));
        assert!(
            list.definition
                .description
                .contains("blocking dependencies")
        );

        let total = create.definition.description.len()
            + update.definition.description.len()
            + list.definition.description.len();
        assert!(total < 600, "the three descriptions total {total} bytes");
        assert!(!create.definition.description.contains("##"));
        assert!(!update.definition.description.contains("```"));
    }

    #[tokio::test]
    async fn task_create_list_update_delete_cycle() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));
        let list_tool = make_task_list_tool(Arc::clone(&runtime));

        call(&create, json!({"subject": "First", "description": "desc"}))
            .await
            .expect("the task is created");
        call(&create, json!({"subject": "Second", "description": "desc"}))
            .await
            .expect("the task is created");

        let listing = call(&list_tool, json!({})).await.expect("the list reads");
        assert!(listing.contains("#1 [pending] First"));
        assert!(listing.contains("#2 [pending] Second"));

        call(&update, json!({"taskId": "1", "status": "completed"}))
            .await
            .expect("the task is updated");
        call(&update, json!({"taskId": "2", "status": "deleted"}))
            .await
            .expect("the task is deleted");

        let listing = call(&list_tool, json!({})).await.expect("the list reads");
        assert!(listing.contains("#1 [completed] First"));
        assert!(!listing.contains("#2"));
    }

    #[tokio::test]
    async fn task_update_metadata_merges_and_null_deletes() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));

        call(
            &create,
            json!({"subject": "t", "description": "d", "metadata": {"k1": "v1"}}),
        )
        .await
        .expect("the task is created");
        call(&update, json!({"taskId": "1", "metadata": {"k2": "v2"}}))
            .await
            .expect("the task is updated");
        call(&update, json!({"taskId": "1", "metadata": {"k1": null}}))
            .await
            .expect("the task is updated");

        let list = runtime
            .snapshot(&anthropic_list("ses_a"))
            .expect("the list exists");
        let metadata = &list.items[0].metadata;
        assert!(!metadata.contains_key("k1"));
        assert_eq!(metadata.get("k2"), Some(&json!("v2")));
    }

    #[tokio::test]
    async fn task_update_omitted_optional_strings_do_not_clear_existing_values() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));

        call(
            &create,
            json!({"subject": "t", "description": "d", "activeForm": "doing t"}),
        )
        .await
        .expect("the task is created");
        call(&update, json!({"taskId": "1", "owner": "alice"}))
            .await
            .expect("the task is updated");
        call(&update, json!({"taskId": "1", "metadata": {"k": "v"}}))
            .await
            .expect("the task is updated");

        let list = runtime
            .snapshot(&anthropic_list("ses_a"))
            .expect("the list exists");
        assert_eq!(list.items[0].active_form.as_deref(), Some("doing t"));
        assert_eq!(list.items[0].owner.as_deref(), Some("alice"));
    }

    #[tokio::test]
    async fn task_update_clears_a_field_a_null_names() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));

        call(
            &create,
            json!({"subject": "t", "description": "d", "activeForm": "doing t"}),
        )
        .await
        .expect("the task is created");
        call(&update, json!({"taskId": "1", "activeForm": null}))
            .await
            .expect("the task is updated");

        let list = runtime
            .snapshot(&anthropic_list("ses_a"))
            .expect("the list exists");
        assert_eq!(list.items[0].active_form, None);
    }

    #[tokio::test]
    async fn task_update_add_blocks_and_add_blocked_by_dedupe() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));

        call(&create, json!({"subject": "t", "description": "d"}))
            .await
            .expect("the task is created");
        call(
            &update,
            json!({"taskId": "1", "addBlocks": ["b1", "b2"], "addBlockedBy": ["c1"]}),
        )
        .await
        .expect("the task is updated");
        call(&update, json!({"taskId": "1", "addBlocks": ["b1", "b3"]}))
            .await
            .expect("the task is updated");

        let list = runtime
            .snapshot(&anthropic_list("ses_a"))
            .expect("the list exists");
        assert_eq!(list.items[0].blocks, vec!["b1", "b2", "b3"]);
        assert_eq!(list.items[0].blocked_by, vec!["c1"]);
    }

    #[tokio::test]
    async fn task_list_empty_returns_no_tasks_found() {
        let tool = make_task_list_tool(Arc::new(TodoRuntime::new()));

        let out = call(&tool, json!({})).await.expect("the list reads");

        assert_eq!(out, "No tasks found");
    }

    /// A finished blocker no longer holds anything up, so it is not named.
    #[tokio::test]
    async fn task_list_names_only_the_blockers_still_open() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));
        let list_tool = make_task_list_tool(Arc::clone(&runtime));

        call(&create, json!({"subject": "First", "description": "d"}))
            .await
            .expect("the task is created");
        call(&create, json!({"subject": "Second", "description": "d"}))
            .await
            .expect("the task is created");
        call(&create, json!({"subject": "Third", "description": "d"}))
            .await
            .expect("the task is created");
        call(
            &update,
            json!({"taskId": "3", "addBlockedBy": ["1", "2"], "owner": "agent-1"}),
        )
        .await
        .expect("the task is updated");
        call(&update, json!({"taskId": "1", "status": "completed"}))
            .await
            .expect("the task is updated");

        let listing = call(&list_tool, json!({})).await.expect("the list reads");

        assert!(
            listing.contains("#3 [pending] Third (owner: agent-1) (blocked by: 2)"),
            "{listing}"
        );
    }

    #[tokio::test]
    async fn task_update_missing_task_returns_not_found() {
        let tool = make_task_update_tool(Arc::new(TodoRuntime::new()));

        let out = call(&tool, json!({"taskId": "999", "status": "completed"}))
            .await
            .expect("a missing task is not an error");

        assert_eq!(out, "Task not found");
    }

    #[tokio::test]
    async fn task_get_returns_full_task_details() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));
        let update = make_task_update_tool(Arc::clone(&runtime));
        let get = make_task_get_tool(runtime);

        call(
            &create,
            json!({
                "subject": "Investigate failing tests",
                "description": "Find the failing assertions and identify the smallest fix."
            }),
        )
        .await
        .expect("the task is created");
        call(
            &update,
            json!({
                "taskId": "1",
                "status": "in_progress",
                "owner": "agent-1",
                "addBlockedBy": ["2", "3"],
                "addBlocks": ["4"]
            }),
        )
        .await
        .expect("the task is updated");

        let out = call(&get, json!({"taskId": "1"}))
            .await
            .expect("the task reads");

        assert_eq!(
            out,
            "\
Task #1: Investigate failing tests
Status: in_progress
Description: Find the failing assertions and identify the smallest fix.
Owner: agent-1
Blocked by: #2, #3
Blocks: #4"
        );
    }

    #[tokio::test]
    async fn task_get_missing_task_returns_not_found() {
        let tool = make_task_get_tool(Arc::new(TodoRuntime::new()));

        let out = call(&tool, json!({"taskId": "999"}))
            .await
            .expect("a missing task is not an error");

        assert_eq!(out, "Task not found");
    }

    /// The task tools are how a parent and its children divide work, so they
    /// all write the root's list.
    #[tokio::test]
    async fn parent_and_subagent_share_one_task_list() {
        let runtime = Arc::new(TodoRuntime::new());
        let create = make_task_create_tool(Arc::clone(&runtime));

        (create.executor)(
            json!({"subject": "p", "description": "d"}),
            context_for("ses_parent", "ses_parent"),
        )
        .await
        .expect("the task is created");
        (create.executor)(
            json!({"subject": "c", "description": "d"}),
            context_for("ses_child", "ses_parent"),
        )
        .await
        .expect("the task is created");

        assert!(runtime.snapshot(&anthropic_list("ses_child")).is_none());
        let list = runtime
            .snapshot(&anthropic_list("ses_parent"))
            .expect("the root's list exists");
        assert_eq!(list.items.len(), 2);
    }

    // --- scoping ---

    /// Every todo tool writes a list keyed by a session, so one called outside
    /// a session says so rather than writing somewhere nobody reads.
    #[tokio::test]
    async fn a_todo_tool_called_outside_a_session_is_unavailable() {
        let runtime = Arc::new(TodoRuntime::new());
        let plan = make_update_plan_tool(Arc::clone(&runtime));
        let tasks = make_task_list_tool(runtime);
        let bare = || context(MockEnvironment::default());

        let error = (plan.executor)(json!({"plan": []}), bare())
            .await
            .expect_err("there is no session to scope the plan to");
        assert_eq!(error.message(), "update_plan requires an active session");
        assert_eq!(error.kind(), ToolErrorKind::Unavailable);

        let error = (tasks.executor)(json!({}), bare())
            .await
            .expect_err("there is no session to scope the list to");
        assert_eq!(error.message(), "task tools require an active session");
    }

    /// An entry's identity is its text within its list, so the same words in
    /// two lists are two different todos.
    #[test]
    fn a_todo_identity_is_its_text_within_its_own_list() {
        let first = todo_text_id("openai_plan:ses_a", "write the test");
        let again = todo_text_id("openai_plan:ses_a", "write the test");
        let elsewhere = todo_text_id("openai_plan:ses_b", "write the test");

        assert_eq!(first, again);
        assert_ne!(first, elsewhere);
        assert_eq!(first.len(), 16);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
