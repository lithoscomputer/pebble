//! The list every todo tool acts on.

use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};

use crate::tool::ToolContext;
use crate::types::{
    CodingEvent, TodoCreatedProps, TodoDeletedProps, TodoListKind, TodoListProjection,
    TodoProjection, TodoStatus, TodoUpdatedProps,
};

/// The lists and their identifier counters, behind one lock so a list and its
/// counter can never be read out of step.
#[derive(Debug, Default)]
struct TodoRuntimeState {
    lists:         BTreeMap<String, TodoListProjection>,
    task_counters: BTreeMap<String, u64>,
}

/// The todo lists a session's tools share.
///
/// One runtime backs every todo vocabulary — `update_plan`, `TodoList`, and the
/// task tools — so a profile builds one, wraps it in an
/// [`Arc`](std::sync::Arc), and hands a clone to each tool it registers. Lists
/// are kept apart by identifier rather than by runtime: `update_plan` writes
/// one list per session, while the task tools write one list per session tree.
///
/// Every change is announced. The runtime is the live state, and the
/// [`TodoCreated`](crate::events::CodingEvent::TodoCreated),
/// [`TodoUpdated`](crate::events::CodingEvent::TodoUpdated) and
/// [`TodoDeleted`](crate::events::CodingEvent::TodoDeleted) events are how an
/// application projects the same state for itself.
#[derive(Debug, Default)]
pub struct TodoRuntime {
    state: Mutex<TodoRuntimeState>,
}

impl TodoRuntime {
    /// A runtime holding no lists.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(TodoRuntimeState::default()),
        }
    }

    /// The next task identifier for `list_id`.
    ///
    /// The counter lives beside the list, so a parent and its children creating
    /// tasks in the same shared list still get distinct numbers.
    pub(crate) fn next_task_id(&self, list_id: &str) -> u64 {
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let counter = guard.task_counters.entry(list_id.to_owned()).or_default();
        *counter = counter.saturating_add(1);
        *counter
    }

    /// The list `list_id` holds, or `None` when nothing has been written to it.
    #[must_use]
    pub fn snapshot(&self, list_id: &str) -> Option<TodoListProjection> {
        let guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.lists.get(list_id).cloned()
    }

    /// Adds `todo` to a list, replacing an entry with the same identifier, and
    /// announces it.
    pub fn create(
        &self,
        ctx: &ToolContext,
        kind: TodoListKind,
        list_id: String,
        todo: TodoProjection,
    ) {
        let props = TodoCreatedProps {
            list_id:     list_id.clone(),
            list_kind:   kind,
            todo_id:     todo.id.clone(),
            status:      todo.status,
            order:       todo.order,
            subject:     todo.subject.clone(),
            description: todo.description.clone(),
            active_form: todo.active_form.clone(),
            owner:       todo.owner.clone(),
            blocks:      todo.blocks.clone(),
            blocked_by:  todo.blocked_by.clone(),
            metadata:    todo.metadata.clone(),
        };
        {
            let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            guard
                .lists
                .entry(list_id)
                .or_insert_with(|| TodoListProjection::new(kind, props.list_id.clone()))
                .upsert(todo);
        }
        ctx.emit_coding_event(CodingEvent::TodoCreated(props));
    }

    /// Applies a patch and announces it, answering whether the todo was there.
    ///
    /// A patch that sets [`TodoStatus::Deleted`] is a deletion: it removes the
    /// todo and announces only that, because a consumer that received an update
    /// and then a delete would have to reconcile a status no list ever holds.
    pub fn update(&self, ctx: &ToolContext, props: TodoUpdatedProps) -> bool {
        if matches!(props.status, Some(TodoStatus::Deleted)) {
            return self.delete(ctx, props.list_kind, props.list_id, props.todo_id);
        }

        let applied = {
            let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(list) = guard.lists.get_mut(&props.list_id) else {
                return false;
            };
            list.apply_patch(&props.todo_id, &props)
        };
        if applied {
            ctx.emit_coding_event(CodingEvent::TodoUpdated(props));
        }
        applied
    }

    /// Removes a todo and announces it, answering whether it was there.
    pub fn delete(
        &self,
        ctx: &ToolContext,
        kind: TodoListKind,
        list_id: String,
        todo_id: String,
    ) -> bool {
        let removed = {
            let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(list) = guard.lists.get_mut(&list_id) else {
                return false;
            };
            list.remove(&todo_id)
        };
        if removed {
            ctx.emit_coding_event(CodingEvent::TodoDeleted(TodoDeletedProps {
                list_id,
                list_kind: kind,
                todo_id,
            }));
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::todo::testing::{CollectingEmitter, context_emitting};

    #[test]
    fn create_then_update_then_delete_announces_three_changes() {
        let runtime = TodoRuntime::new();
        let collector = Arc::new(CollectingEmitter::default());
        let ctx = context_emitting(Arc::clone(&collector));
        let list_id = TodoListKind::OpenAiPlan.list_id("ses_a");

        runtime.create(
            &ctx,
            TodoListKind::OpenAiPlan,
            list_id.clone(),
            TodoProjection::new("a", 0, "first"),
        );
        runtime.update(&ctx, TodoUpdatedProps {
            status: Some(TodoStatus::InProgress),
            ..TodoUpdatedProps::new(&list_id, TodoListKind::OpenAiPlan, "a")
        });
        runtime.delete(&ctx, TodoListKind::OpenAiPlan, list_id, "a".to_owned());

        let events = collector.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], CodingEvent::TodoCreated(_)));
        assert!(matches!(events[1], CodingEvent::TodoUpdated(_)));
        assert!(matches!(events[2], CodingEvent::TodoDeleted(_)));
    }

    #[test]
    fn a_patch_that_deletes_announces_only_the_deletion() {
        let runtime = TodoRuntime::new();
        let collector = Arc::new(CollectingEmitter::default());
        let ctx = context_emitting(Arc::clone(&collector));
        let list_id = TodoListKind::AnthropicTasks.list_id("r");

        runtime.create(
            &ctx,
            TodoListKind::AnthropicTasks,
            list_id.clone(),
            TodoProjection::new("1", 0, "task"),
        );
        runtime.update(&ctx, TodoUpdatedProps {
            status: Some(TodoStatus::Deleted),
            ..TodoUpdatedProps::new(&list_id, TodoListKind::AnthropicTasks, "1")
        });

        let events = collector.events();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1], CodingEvent::TodoDeleted(_)));
        assert!(
            runtime
                .snapshot(&list_id)
                .expect("the list exists")
                .items
                .is_empty()
        );
    }

    #[test]
    fn updating_a_todo_that_is_not_there_changes_nothing() {
        let runtime = TodoRuntime::new();
        let collector = Arc::new(CollectingEmitter::default());
        let ctx = context_emitting(Arc::clone(&collector));
        let list_id = TodoListKind::AnthropicTasks.list_id("r");

        let found = runtime.update(
            &ctx,
            TodoUpdatedProps::new(&list_id, TodoListKind::AnthropicTasks, "missing"),
        );

        assert!(!found);
        assert!(collector.events().is_empty());
    }

    /// Task numbering is per list, and a list is shared by a whole session
    /// tree, so two sessions creating tasks never collide.
    #[test]
    fn task_identifiers_count_up_within_one_list() {
        let runtime = TodoRuntime::new();
        let list_id = TodoListKind::AnthropicTasks.list_id("root");

        assert_eq!(runtime.next_task_id(&list_id), 1);
        assert_eq!(runtime.next_task_id(&list_id), 2);
        assert_eq!(runtime.next_task_id("another"), 1);
    }
}
