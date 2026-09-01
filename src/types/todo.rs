//! The shared todo / task vocabulary.
//!
//! Three tool families — OpenAI's `update_plan`, Anthropic's task tools, and
//! Kimi's `TodoList` — share one event-sourced projection. The only difference
//! is the scoping convention captured by [`TodoListKind`]. All mutations are
//! projected from the individual created, updated, and deleted events.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// The lifecycle status of a todo.
///
/// `Deleted` is reachable for Anthropic-style tasks, where the model can ask
/// for it directly. The projection treats it as a hard delete: an update
/// carrying `Deleted` is followed by a delete event and the todo disappears
/// from the projected list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TodoStatus {
    /// Not started.
    Pending,
    /// Being worked on now.
    InProgress,
    /// Finished.
    Completed,
    /// Removed from the list.
    Deleted,
}

impl TodoStatus {
    /// The wire spelling of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Deleted => "deleted",
        }
    }
}

impl fmt::Display for TodoStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The scoping convention of a [`TodoListProjection`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TodoListKind {
    /// `update_plan`. Scoped to the emitting session.
    #[default]
    #[serde(rename = "openai_plan")]
    OpenAiPlan,
    /// The Anthropic task tools. Scoped to the root session and shared by its
    /// subagent sessions.
    #[serde(rename = "anthropic_tasks")]
    AnthropicTasks,
    /// Kimi's `TodoList`. Like [`Self::OpenAiPlan`] it replaces the whole list
    /// in one call, but with Kimi's field names. CodingRuntime-scoped.
    #[serde(rename = "kimi_todos")]
    KimiTodos,
}

impl TodoListKind {
    /// The wire spelling of this kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiPlan => "openai_plan",
            Self::AnthropicTasks => "anthropic_tasks",
            Self::KimiTodos => "kimi_todos",
        }
    }

    /// Builds the list identifier (`"<kind>:<session>"`) used as the
    /// projection key.
    #[must_use]
    pub fn list_id(self, session: &str) -> String {
        format!("{}:{session}", self.as_str())
    }
}

impl fmt::Display for TodoListKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One projected todo item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoProjection {
    /// Identity within the list.
    pub id:          String,
    /// The lifecycle status. `Deleted` never appears in a current projection,
    /// because such todos are removed entirely.
    pub status:      TodoStatus,
    /// Ordering within the list. Lower comes first.
    pub order:       u32,
    /// A free-form summary.
    pub subject:     String,
    /// A longer description; empty when not provided.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The phrasing used while the task is in progress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_form: Option<String>,
    /// Who owns the task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner:       Option<String>,
    /// Identifiers of tasks this one blocks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks:      Vec<String>,
    /// Identifiers of tasks this one is blocked by.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_by:  Vec<String>,
    /// A per-todo metadata bag.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata:    BTreeMap<String, serde_json::Value>,
}

impl TodoProjection {
    /// Builds a minimal projection for a freshly created todo.
    #[must_use]
    pub fn new(id: impl Into<String>, order: u32, subject: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            status: TodoStatus::Pending,
            order,
            subject: subject.into(),
            description: String::new(),
            active_form: None,
            owner: None,
            blocks: Vec::new(),
            blocked_by: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Applies a patch in place.
    ///
    /// `add_blocks` and `add_blocked_by` dedupe against existing entries.
    /// `metadata_patch` keys with a `null` value delete that key; non-null
    /// values overwrite. Returns whether `order` changed, which is what
    /// [`TodoListProjection`] uses to decide whether to re-sort.
    pub fn apply_patch(&mut self, patch: &TodoUpdatedProps) -> bool {
        let order_changed = patch.order.is_some_and(|order| order != self.order);
        if let Some(status) = patch.status {
            self.status = status;
        }
        if let Some(order) = patch.order {
            self.order = order;
        }
        if let Some(subject) = patch.subject.as_deref() {
            self.subject.clear();
            self.subject.push_str(subject);
        }
        if let Some(description) = patch.description.as_deref() {
            self.description.clear();
            self.description.push_str(description);
        }
        if let Some(active_form) = patch.active_form.as_ref() {
            self.active_form.clone_from(active_form);
        }
        if let Some(owner) = patch.owner.as_ref() {
            self.owner.clone_from(owner);
        }
        if let Some(extra) = patch.add_blocks.as_deref() {
            for id in extra {
                if !self.blocks.contains(id) {
                    self.blocks.push(id.clone());
                }
            }
        }
        if let Some(extra) = patch.add_blocked_by.as_deref() {
            for id in extra {
                if !self.blocked_by.contains(id) {
                    self.blocked_by.push(id.clone());
                }
            }
        }
        for (key, value) in &patch.metadata_patch {
            if value.is_null() {
                self.metadata.remove(key);
            } else {
                self.metadata.insert(key.clone(), value.clone());
            }
        }
        order_changed
    }
}

/// Every currently projected todo for one list.
///
/// Items are kept sorted by `(order, id)` so callers do not have to re-sort
/// the projection on every read.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoListProjection {
    /// The scoping convention of the list.
    pub kind:    TodoListKind,
    /// The list identifier.
    pub list_id: String,
    /// The items currently in the list, in display order.
    #[serde(default)]
    pub items:   Vec<TodoProjection>,
}

impl TodoListProjection {
    /// Builds an empty projection for one list.
    #[must_use]
    pub fn new(kind: TodoListKind, list_id: impl Into<String>) -> Self {
        Self {
            kind,
            list_id: list_id.into(),
            items: Vec::new(),
        }
    }

    /// Looks up a todo by identifier.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&TodoProjection> {
        self.items.iter().find(|todo| todo.id == id)
    }

    /// Inserts or replaces a todo and re-sorts by `(order, id)`.
    pub fn upsert(&mut self, todo: TodoProjection) {
        match self
            .items
            .iter()
            .position(|existing| existing.id == todo.id)
        {
            Some(index) => self.items[index] = todo,
            None => self.items.push(todo),
        }
        self.sort();
    }

    /// Applies `patch` to the todo with identifier `todo_id`, reporting
    /// whether the todo was found. Re-sorts only when `order` changed.
    pub fn apply_patch(&mut self, todo_id: &str, patch: &TodoUpdatedProps) -> bool {
        let Some(index) = self.items.iter().position(|todo| todo.id == todo_id) else {
            return false;
        };
        if self.items[index].apply_patch(patch) {
            self.sort();
        }
        true
    }

    /// Removes a todo by identifier, reporting whether anything was removed.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.items.len();
        self.items.retain(|todo| todo.id != id);
        before != self.items.len()
    }

    fn sort(&mut self) {
        self.items.sort_by(|left, right| {
            left.order
                .cmp(&right.order)
                .then_with(|| left.id.cmp(&right.id))
        });
    }
}

/// The body of a todo-created event.
///
/// It carries the whole row, so a projection can be reconstructed from the
/// created event alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoCreatedProps {
    /// The list the todo belongs to.
    pub list_id:     String,
    /// The scoping convention of the list.
    pub list_kind:   TodoListKind,
    /// Identity within the list.
    pub todo_id:     String,
    /// The lifecycle status at creation.
    pub status:      TodoStatus,
    /// Ordering within the list.
    pub order:       u32,
    /// A free-form summary.
    pub subject:     String,
    /// A longer description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The phrasing used while the task is in progress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_form: Option<String>,
    /// Who owns the task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner:       Option<String>,
    /// Identifiers of tasks this one blocks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks:      Vec<String>,
    /// Identifiers of tasks this one is blocked by.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_by:  Vec<String>,
    /// A per-todo metadata bag.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata:    BTreeMap<String, serde_json::Value>,
}

/// The body of a todo-updated event.
///
/// Every patch field is optional and absent means "leave alone".
/// `active_form` and `owner` are double-`Option`: `Some(None)` clears the
/// field and serializes as JSON `null`, while absence leaves it unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoUpdatedProps {
    /// The list the todo belongs to.
    pub list_id:        String,
    /// The scoping convention of the list.
    pub list_kind:      TodoListKind,
    /// Identity within the list.
    pub todo_id:        String,
    /// The new status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status:         Option<TodoStatus>,
    /// The new order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order:          Option<u32>,
    /// The new subject.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject:        Option<String>,
    /// The new description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description:    Option<String>,
    /// The new in-progress phrasing; `Some(None)` clears it.
    #[serde(
        default,
        with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub active_form:    Option<Option<String>>,
    /// The new owner; `Some(None)` clears it.
    #[serde(
        default,
        with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub owner:          Option<Option<String>>,
    /// Identifiers to add to `blocks`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_blocks:     Option<Vec<String>>,
    /// Identifiers to add to `blocked_by`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_blocked_by: Option<Vec<String>>,
    /// Metadata keys to set, or to delete with a `null` value.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata_patch: BTreeMap<String, serde_json::Value>,
}

impl TodoUpdatedProps {
    /// Builds an empty patch targeting `todo_id` in `list_id`.
    ///
    /// Every optional field defaults to "leave alone". Use the result with
    /// struct-update syntax to fill in what the caller wants to change.
    #[must_use]
    pub fn new(
        list_id: impl Into<String>,
        list_kind: TodoListKind,
        todo_id: impl Into<String>,
    ) -> Self {
        Self {
            list_id: list_id.into(),
            list_kind,
            todo_id: todo_id.into(),
            status: None,
            order: None,
            subject: None,
            description: None,
            active_form: None,
            owner: None,
            add_blocks: None,
            add_blocked_by: None,
            metadata_patch: BTreeMap::new(),
        }
    }
}

/// The body of a todo-deleted event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoDeletedProps {
    /// The list the todo belonged to.
    pub list_id:   String,
    /// The scoping convention of the list.
    pub list_kind: TodoListKind,
    /// Identity within the list.
    pub todo_id:   String,
}

/// Serde support for a patch field that tells "unchanged" from "cleared".
///
/// Serde reads both an absent member and an explicit `null` into a plain
/// `Option` as `None`. On an `Option<Option<T>>` field carrying `default` and
/// `skip_serializing_if = "Option::is_none"`, this module keeps the two apart:
/// an absent member is `None` and stays off the wire, and a `null` is
/// `Some(None)` and is written back as `null`.
mod double_option {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[expect(
        clippy::option_option,
        reason = "the outer option is the patch field's \"unchanged\" state"
    )]
    #[expect(
        clippy::ref_option,
        reason = "serde's `with` contract hands the field over by reference"
    )]
    pub(super) fn serialize<T, S>(
        value: &Option<Option<T>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        match value {
            Some(Some(inner)) => serializer.serialize_some(inner),
            // `skip_serializing_if` keeps the outer `None` off the wire, so
            // this renders the cleared field.
            Some(None) | None => serializer.serialize_none(),
        }
    }

    #[expect(
        clippy::option_option,
        reason = "the outer option is the patch field's \"unchanged\" state"
    )]
    pub(super) fn deserialize<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Option::deserialize(deserializer).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn list_id_is_kind_colon_session() {
        assert_eq!(
            TodoListKind::OpenAiPlan.list_id("ses_abc"),
            "openai_plan:ses_abc"
        );
        assert_eq!(
            TodoListKind::AnthropicTasks.list_id("ses_root"),
            "anthropic_tasks:ses_root"
        );
        assert_eq!(
            TodoListKind::KimiTodos.list_id("ses_abc"),
            "kimi_todos:ses_abc"
        );
    }

    #[test]
    fn list_kind_default_is_the_openai_plan() {
        assert_eq!(TodoListKind::default(), TodoListKind::OpenAiPlan);
    }

    #[test]
    fn upsert_orders_by_order_then_id() {
        let mut list = TodoListProjection::new(TodoListKind::OpenAiPlan, "openai_plan:s");
        list.upsert(TodoProjection::new("a", 2, "second"));
        list.upsert(TodoProjection::new("b", 0, "first"));
        list.upsert(TodoProjection::new("c", 2, "second-tie"));

        let ids: Vec<&str> = list.items.iter().map(|todo| todo.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a", "c"]);
    }

    #[test]
    fn upsert_replaces_an_existing_id() {
        let mut list = TodoListProjection::new(TodoListKind::OpenAiPlan, "openai_plan:s");
        list.upsert(TodoProjection::new("a", 0, "first"));
        let mut updated = TodoProjection::new("a", 0, "first");
        updated.status = TodoStatus::Completed;
        list.upsert(updated);

        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].status, TodoStatus::Completed);
        assert!(list.get("a").is_some());
    }

    #[test]
    fn remove_reports_whether_anything_went() {
        let mut list = TodoListProjection::new(TodoListKind::OpenAiPlan, "openai_plan:s");
        list.upsert(TodoProjection::new("a", 0, "first"));
        assert!(list.remove("a"));
        assert!(!list.remove("a"));
        assert!(list.items.is_empty());
    }

    #[test]
    fn a_patch_clears_a_field_and_deletes_a_metadata_key() {
        let mut todo = TodoProjection::new("a", 0, "first");
        todo.active_form = Some("doing first".into());
        todo.metadata.insert("k".into(), json!("v"));

        let props = TodoUpdatedProps {
            active_form: Some(None),
            metadata_patch: BTreeMap::from([("k".to_owned(), serde_json::Value::Null)]),
            ..TodoUpdatedProps::new("openai_plan:s", TodoListKind::OpenAiPlan, "a")
        };
        let order_changed = todo.apply_patch(&props);

        assert!(!order_changed);
        assert_eq!(todo.active_form, None);
        assert!(todo.metadata.is_empty());
    }

    #[test]
    fn a_patch_dedupes_added_blockers_and_re_sorts_on_order() {
        let mut list = TodoListProjection::new(TodoListKind::OpenAiPlan, "openai_plan:s");
        list.upsert(TodoProjection::new("a", 0, "first"));
        list.upsert(TodoProjection::new("b", 1, "second"));

        let props = TodoUpdatedProps {
            order: Some(9),
            add_blocks: Some(vec!["b".into(), "b".into()]),
            ..TodoUpdatedProps::new("openai_plan:s", TodoListKind::OpenAiPlan, "a")
        };
        assert!(list.apply_patch("a", &props));

        let ids: Vec<&str> = list.items.iter().map(|todo| todo.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a"]);
        assert_eq!(list.get("a").expect("present").blocks, vec!["b".to_owned()]);
    }

    #[test]
    fn a_patch_for_an_unknown_todo_reports_a_miss() {
        let mut list = TodoListProjection::new(TodoListKind::OpenAiPlan, "openai_plan:s");
        let props = TodoUpdatedProps::new("openai_plan:s", TodoListKind::OpenAiPlan, "missing");
        assert!(!list.apply_patch("missing", &props));
    }

    #[test]
    fn created_props_omit_every_empty_member() {
        let props = TodoCreatedProps {
            list_id:     "openai_plan:s".into(),
            list_kind:   TodoListKind::OpenAiPlan,
            todo_id:     "a".into(),
            status:      TodoStatus::Pending,
            order:       0,
            subject:     "first".into(),
            description: String::new(),
            active_form: None,
            owner:       None,
            blocks:      Vec::new(),
            blocked_by:  Vec::new(),
            metadata:    BTreeMap::new(),
        };
        assert_eq!(
            serde_json::to_value(&props).expect("serializes"),
            json!({
                "list_id": "openai_plan:s",
                "list_kind": "openai_plan",
                "todo_id": "a",
                "status": "pending",
                "order": 0,
                "subject": "first",
            })
        );
    }

    #[test]
    fn a_cleared_field_serializes_as_null_and_an_unchanged_one_is_absent() {
        let props = TodoUpdatedProps {
            active_form: Some(None),
            owner: Some(Some("ada".into())),
            ..TodoUpdatedProps::new("openai_plan:s", TodoListKind::OpenAiPlan, "a")
        };
        assert_eq!(
            serde_json::to_value(&props).expect("serializes"),
            json!({
                "list_id": "openai_plan:s",
                "list_kind": "openai_plan",
                "todo_id": "a",
                "active_form": null,
                "owner": "ada",
            })
        );
    }

    #[test]
    fn a_null_patch_field_reads_as_cleared_and_an_absent_one_as_unchanged() {
        let cleared: TodoUpdatedProps = serde_json::from_value(json!({
            "list_id": "openai_plan:s",
            "list_kind": "openai_plan",
            "todo_id": "a",
            "active_form": null,
            "owner": "ada",
        }))
        .expect("parses");

        assert_eq!(cleared.active_form, Some(None), "an explicit null clears");
        assert_eq!(cleared.owner, Some(Some("ada".to_owned())));

        let unchanged: TodoUpdatedProps = serde_json::from_value(json!({
            "list_id": "openai_plan:s",
            "list_kind": "openai_plan",
            "todo_id": "a",
        }))
        .expect("parses");

        assert_eq!(
            unchanged.active_form, None,
            "an absent member leaves the field alone"
        );
        assert_eq!(unchanged.owner, None);
    }

    #[test]
    fn null_and_absent_patch_fields_round_trip_distinctly() {
        let cleared = TodoUpdatedProps {
            active_form: Some(None),
            ..TodoUpdatedProps::new("openai_plan:s", TodoListKind::OpenAiPlan, "a")
        };
        let unchanged = TodoUpdatedProps::new("openai_plan:s", TodoListKind::OpenAiPlan, "a");

        for props in [cleared, unchanged] {
            let json = serde_json::to_string(&props).expect("serializes");
            let restored: TodoUpdatedProps = serde_json::from_str(&json).expect("parses");
            assert_eq!(
                restored, props,
                "round trip lost null-versus-absent: {json}"
            );
        }
    }

    #[test]
    fn deleted_props_round_trip() {
        let props = TodoDeletedProps {
            list_id:   "kimi_todos:s".into(),
            list_kind: TodoListKind::KimiTodos,
            todo_id:   "a".into(),
        };
        let value = serde_json::to_value(&props).expect("serializes");
        assert_eq!(value["list_kind"], json!("kimi_todos"));
        assert_eq!(
            serde_json::from_value::<TodoDeletedProps>(value).expect("parses"),
            props
        );
    }
}
