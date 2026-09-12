//! A value that folds a session's events into what a view or an accountant
//! needs, the same way live or replayed.
//!
//! Both embedders kept their own tallies over the durable event stream — token
//! counts and provider-reported cost, the context window, which tools ran,
//! skills, todo lists, the children and how they ended, compactions, and the
//! files a prompt touched — each written once per application in a different
//! shape. [`SessionProjection`] is that fold, once: feed it every
//! [`CodingAgentEvent`] of one session tree, in order, and read the answers.
//! It is serializable, so a view resumes from a stored value and applies the
//! events after it; and it keeps a [`PromptDelta`] for the prompt in
//! progress, because a retained session spans stages and a stage wants what
//! its own prompts did, not the session's lifetime total.
//!
//! The projection reports counts and provider-reported cost only; pricing a
//! count from a catalog is the application's.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::compaction::CompactionReason;
use crate::error::ErrorData;
use crate::file_tracker;
use crate::types::{
    CodingAgentEvent, CodingEvent, ContextWindowSnapshot, InputSource, McpToolSummary,
    SkillActivationSource, SkillSummary, TodoListProjection, TodoProjection, TokenUsage,
};

/// Where a session stands, as its events tell it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionActivity {
    /// No prompt is running.
    #[default]
    Idle,
    /// A prompt is running.
    Running,
    /// A round was interrupted and the prompt waits for a steer.
    WaitingForSteer,
    /// The session ended.
    Ended,
}

/// The route a session runs on, as it reported it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteProjection {
    pub provider: Option<String>,
    pub model:    Option<String>,
}

/// What one descendant session spent, as its own events reported it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescendantAccount {
    /// The session that spawned it.
    pub parent:          String,
    pub usage:           TokenUsage,
    pub cost_usd_micros: Option<u64>,
    /// Committed assistant messages.
    pub messages:        u64,
    /// Compactions it completed.
    pub compactions:     u64,
}

/// How one tool has been used across the tree.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolActivity {
    /// Calls started.
    pub calls:  u64,
    /// Calls that completed as errors.
    pub errors: u64,
    /// Calls started and not yet completed.
    pub open:   u64,
}

/// How many child lifecycle events the tree recorded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCounts {
    pub spawned:       u64,
    pub turns_started: u64,
    pub completed:     u64,
    pub failed:        u64,
    pub closed:        u64,
}

/// One MCP server the session configured, and whether it has been called.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerProjection {
    pub tools:        Vec<McpToolSummary>,
    /// Why it did not start, when it did not.
    pub error:        Option<String>,
    /// Whether any of its tools has been called.
    pub invoked:      bool,
    /// What closed its connection during the session, when it closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disconnected: Option<String>,
}

/// A skill the session activated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivatedSkill {
    pub name:   String,
    pub source: SkillActivationSource,
}

/// The skills the root session found and the ones activated anywhere in the
/// tree.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillsProjection {
    pub available: Vec<SkillSummary>,
    pub activated: Vec<ActivatedSkill>,
}

/// Where a child stands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SubagentStatus {
    Running,
    Completed { success: bool, turns_used: usize },
    Failed { error: ErrorData },
    Closed,
}

/// One child the root spawned. A reused child stays one row: every event
/// after the spawn moves its status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentProjection {
    pub agent_id: String,
    pub depth:    usize,
    pub task:     String,
    pub status:   SubagentStatus,
}

/// One compaction the root session completed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionProjection {
    pub reason:                 CompactionReason,
    pub original_turn_count:    usize,
    pub preserved_turn_count:   usize,
    pub summary_token_estimate: usize,
    pub tracked_file_count:     usize,
}

/// What the prompt in progress, or the last one, did: reset when a prompt
/// starts, complete once [`completed`](Self::completed) is set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptDelta {
    /// Whether the prompt reached its end.
    pub completed:         bool,
    /// The root session's usage over the prompt.
    pub usage:             TokenUsage,
    pub cost_usd_micros:   Option<u64>,
    /// Committed assistant messages.
    pub messages:          u64,
    /// The latest context window the prompt reported.
    pub context_window:    Option<ContextWindowSnapshot>,
    /// Tool calls started, across the tree.
    pub tool_calls:        u64,
    /// What each descendant spent during the prompt, by session id.
    pub descendants:       BTreeMap<String, DescendantAccount>,
    /// Child lifecycle events during the prompt.
    pub subagents:         SubagentCounts,
    /// Compactions the root completed during the prompt.
    pub compactions:       Vec<CompactionProjection>,
    /// Files written or edited during the prompt, across the tree, sorted.
    pub files_touched:     Vec<String>,
    pub last_file_touched: Option<String>,
}

impl PromptDelta {
    /// What every descendant spent during the prompt, summed.
    #[must_use]
    pub fn descendant_usage(&self) -> (TokenUsage, Option<u64>) {
        sum_accounts(self.descendants.values())
    }

    fn touch(&mut self, paths: &[String]) {
        for path in paths {
            if !self.files_touched.contains(path) {
                self.files_touched.push(path.clone());
                self.files_touched.sort();
            }
            self.last_file_touched = Some(path.clone());
        }
    }
}

/// The fold over one session tree's events.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionProjection {
    /// The root session, once an event named it.
    pub root_session_id:   Option<String>,
    pub route:             RouteProjection,
    pub activity:          SessionActivity,
    /// The root session's lifetime usage.
    pub usage:             TokenUsage,
    pub cost_usd_micros:   Option<u64>,
    pub messages:          u64,
    /// Every descendant session, by id.
    pub descendants:       BTreeMap<String, DescendantAccount>,
    /// The root session's latest context window.
    pub context_window:    Option<ContextWindowSnapshot>,
    /// Every tool called anywhere in the tree, by the name the model used.
    pub tools:             BTreeMap<String, ToolActivity>,
    /// Every MCP server the root configured, by name.
    pub mcp_servers:       BTreeMap<String, McpServerProjection>,
    pub skills:            SkillsProjection,
    /// Child lifecycle events over the session's life, across the tree: a
    /// child's own children count.
    pub subagent_counts:   SubagentCounts,
    /// Every todo list in the tree, by list id.
    pub todos:             BTreeMap<String, TodoListProjection>,
    pub subagents:         Vec<SubagentProjection>,
    /// The root session's compactions, in order.
    pub compactions:       Vec<CompactionProjection>,
    /// Files written or edited over the session's life, across the tree,
    /// sorted.
    pub files_touched:     Vec<String>,
    pub last_file_touched: Option<String>,
    /// How many prompts have started.
    pub prompts:           u64,
    /// What the prompt in progress, or the last one, did.
    pub prompt:            PromptDelta,
    /// Writes and edits started and not yet completed, by tool call id.
    pending_writes:        BTreeMap<String, Vec<String>>,
}

impl SessionProjection {
    /// A projection that has seen nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one event in.
    ///
    /// Events are applied in stream order; an event of a session whose root
    /// this projection has not seen is taken to belong to the same tree.
    pub fn apply(&mut self, event: &CodingAgentEvent) {
        let is_root = event.parent_session_id.is_none();
        if is_root && self.root_session_id.is_none() {
            self.root_session_id = Some(event.session_id.clone());
        }
        match &event.event {
            CodingEvent::SessionStarted { provider, model } if is_root => {
                self.route = RouteProjection {
                    provider: provider.clone(),
                    model:    model.clone(),
                };
            }
            CodingEvent::RouteFailover { to, .. } if is_root => {
                if let Some((provider, model)) = to.split_once('/') {
                    self.route = RouteProjection {
                        provider: Some(provider.to_owned()),
                        model:    Some(model.to_owned()),
                    };
                }
            }
            CodingEvent::SessionEnded if is_root => self.activity = SessionActivity::Ended,
            CodingEvent::UserInput { source, .. } if is_root => {
                if *source == InputSource::Prompt {
                    self.prompts += 1;
                    self.prompt = PromptDelta::default();
                }
                self.activity = SessionActivity::Running;
            }
            CodingEvent::ProcessingEnd if is_root => {
                self.prompt.completed = true;
                if self.activity != SessionActivity::Ended {
                    self.activity = SessionActivity::Idle;
                }
            }
            CodingEvent::RoundInterrupted { .. } if is_root => {
                self.activity = SessionActivity::WaitingForSteer;
            }
            CodingEvent::SteeringInjected { .. } if is_root => {
                self.activity = SessionActivity::Running;
            }
            CodingEvent::AssistantMessage {
                usage,
                cost_usd_micros,
                context_window,
                ..
            } => {
                if is_root {
                    self.usage = self.usage.saturating_add(*usage);
                    add_cost(&mut self.cost_usd_micros, *cost_usd_micros);
                    self.messages += 1;
                    self.prompt.usage = self.prompt.usage.saturating_add(*usage);
                    add_cost(&mut self.prompt.cost_usd_micros, *cost_usd_micros);
                    self.prompt.messages += 1;
                    if let Some(window) = context_window {
                        self.context_window = Some(window.clone());
                        self.prompt.context_window = Some(window.clone());
                    }
                } else if let Some(parent) = &event.parent_session_id {
                    for account in [
                        descendant(&mut self.descendants, &event.session_id, parent),
                        descendant(&mut self.prompt.descendants, &event.session_id, parent),
                    ] {
                        account.usage = account.usage.saturating_add(*usage);
                        add_cost(&mut account.cost_usd_micros, *cost_usd_micros);
                        account.messages += 1;
                    }
                }
            }
            CodingEvent::ToolCallStarted {
                tool_name,
                tool_call_id,
                arguments,
            } => {
                let activity = self.tools.entry(tool_name.clone()).or_default();
                activity.calls += 1;
                activity.open += 1;
                self.prompt.tool_calls += 1;
                if let Some(server) = mcp_server_of(tool_name)
                    && let Some(projection) = self
                        .mcp_servers
                        .iter_mut()
                        .find(|(name, _)| sanitized(name) == server)
                        .map(|(_, projection)| projection)
                {
                    projection.invoked = true;
                }
                let written = file_tracker::written_paths_from_arguments(tool_name, arguments);
                if !written.is_empty() {
                    self.pending_writes.insert(tool_call_id.clone(), written);
                }
            }
            CodingEvent::ToolCallCompleted {
                tool_name,
                tool_call_id,
                is_error,
                ..
            } => {
                if let Some(activity) = self.tools.get_mut(tool_name) {
                    activity.open = activity.open.saturating_sub(1);
                    if *is_error {
                        activity.errors += 1;
                    }
                }
                if let Some(paths) = self.pending_writes.remove(tool_call_id)
                    && !*is_error
                {
                    self.prompt.touch(&paths);
                    for path in paths {
                        if !self.files_touched.contains(&path) {
                            self.files_touched.push(path.clone());
                            self.files_touched.sort();
                        }
                        self.last_file_touched = Some(path);
                    }
                }
            }
            CodingEvent::McpServerReady { server, tools } if is_root => {
                let projection = self.mcp_servers.entry(server.clone()).or_default();
                projection.tools.clone_from(tools);
                projection.error = None;
            }
            CodingEvent::McpServerFailed { server, error } if is_root => {
                let projection = self.mcp_servers.entry(server.clone()).or_default();
                projection.tools.clear();
                projection.error = Some(error.clone());
            }
            // Reported by whichever session's call first observed the close,
            // so this arm is not limited to the root.
            CodingEvent::McpServerDisconnected { server, error } => {
                let projection = self.mcp_servers.entry(server.clone()).or_default();
                projection.disconnected = Some(error.clone());
            }
            CodingEvent::SkillsDiscovered { skills, .. } if is_root => {
                self.skills.available.clone_from(skills);
            }
            CodingEvent::SkillActivated { skill_name, source } => {
                self.skills.activated.push(ActivatedSkill {
                    name:   skill_name.clone(),
                    source: *source,
                });
            }
            CodingEvent::TodoCreated(props) => {
                let list = self.todos.entry(props.list_id.clone()).or_insert_with(|| {
                    TodoListProjection::new(props.list_kind, props.list_id.clone())
                });
                list.upsert(TodoProjection {
                    id:          props.todo_id.clone(),
                    status:      props.status,
                    order:       props.order,
                    subject:     props.subject.clone(),
                    description: props.description.clone(),
                    active_form: props.active_form.clone(),
                    owner:       props.owner.clone(),
                    blocks:      props.blocks.clone(),
                    blocked_by:  props.blocked_by.clone(),
                    metadata:    props.metadata.clone(),
                });
            }
            CodingEvent::TodoUpdated(props) => {
                if let Some(list) = self.todos.get_mut(&props.list_id) {
                    list.apply_patch(&props.todo_id, props);
                }
            }
            CodingEvent::TodoDeleted(props) => {
                if let Some(list) = self.todos.get_mut(&props.list_id) {
                    list.remove(&props.todo_id);
                    if list.items.is_empty() {
                        self.todos.remove(&props.list_id);
                    }
                }
            }
            // Children spawn children: the rows and the counts are the
            // tree's, whichever session recorded the event.
            CodingEvent::SubAgentSpawned {
                agent_id,
                depth,
                task,
                ..
            } => {
                self.subagent_counts.spawned += 1;
                self.prompt.subagents.spawned += 1;
                if let Some(existing) = self.subagent_mut(agent_id) {
                    existing.status = SubagentStatus::Running;
                } else {
                    self.subagents.push(SubagentProjection {
                        agent_id: agent_id.clone(),
                        depth:    *depth,
                        task:     task.clone(),
                        status:   SubagentStatus::Running,
                    });
                }
            }
            CodingEvent::SubAgentTurnStarted { agent_id, .. } => {
                self.subagent_counts.turns_started += 1;
                self.prompt.subagents.turns_started += 1;
                self.set_subagent_status(agent_id, SubagentStatus::Running);
            }
            CodingEvent::SubAgentCompleted {
                agent_id,
                success,
                turns_used,
                ..
            } => {
                self.subagent_counts.completed += 1;
                self.prompt.subagents.completed += 1;
                self.set_subagent_status(agent_id, SubagentStatus::Completed {
                    success:    *success,
                    turns_used: *turns_used,
                });
            }
            CodingEvent::SubAgentFailed {
                agent_id, error, ..
            } => {
                self.subagent_counts.failed += 1;
                self.prompt.subagents.failed += 1;
                self.set_subagent_status(agent_id, SubagentStatus::Failed {
                    error: error.clone(),
                });
            }
            CodingEvent::SubAgentClosed { agent_id, .. } => {
                self.subagent_counts.closed += 1;
                self.prompt.subagents.closed += 1;
                self.set_subagent_status(agent_id, SubagentStatus::Closed);
            }
            CodingEvent::CompactionCompleted {
                original_turn_count,
                preserved_turn_count,
                summary_token_estimate,
                tracked_file_count,
                reason,
            } => {
                if is_root {
                    let compaction = CompactionProjection {
                        reason:                 *reason,
                        original_turn_count:    *original_turn_count,
                        preserved_turn_count:   *preserved_turn_count,
                        summary_token_estimate: *summary_token_estimate,
                        tracked_file_count:     *tracked_file_count,
                    };
                    self.prompt.compactions.push(compaction.clone());
                    self.compactions.push(compaction);
                } else if let Some(parent) = &event.parent_session_id {
                    descendant(&mut self.descendants, &event.session_id, parent).compactions += 1;
                    descendant(&mut self.prompt.descendants, &event.session_id, parent)
                        .compactions += 1;
                }
            }
            _ => {}
        }
    }

    /// Folds every event in, in order.
    pub fn apply_all<'a>(&mut self, events: impl IntoIterator<Item = &'a CodingAgentEvent>) {
        for event in events {
            self.apply(event);
        }
    }

    /// What every descendant spent over the session's life, summed.
    #[must_use]
    pub fn descendant_usage(&self) -> (TokenUsage, Option<u64>) {
        sum_accounts(self.descendants.values())
    }

    fn subagent_mut(&mut self, agent_id: &str) -> Option<&mut SubagentProjection> {
        self.subagents
            .iter_mut()
            .find(|subagent| subagent.agent_id == agent_id)
    }

    fn set_subagent_status(&mut self, agent_id: &str, status: SubagentStatus) {
        if let Some(subagent) = self.subagent_mut(agent_id) {
            subagent.status = status;
        }
    }
}

fn descendant<'a>(
    accounts: &'a mut BTreeMap<String, DescendantAccount>,
    session_id: &str,
    parent: &str,
) -> &'a mut DescendantAccount {
    accounts
        .entry(session_id.to_owned())
        .or_insert_with(|| DescendantAccount {
            parent: parent.to_owned(),
            ..DescendantAccount::default()
        })
}

fn sum_accounts<'a>(
    accounts: impl Iterator<Item = &'a DescendantAccount>,
) -> (TokenUsage, Option<u64>) {
    let mut usage = TokenUsage::default();
    let mut cost = None;
    for account in accounts {
        usage = usage.saturating_add(account.usage);
        add_cost(&mut cost, account.cost_usd_micros);
    }
    (usage, cost)
}

fn add_cost(total: &mut Option<u64>, cost: Option<u64>) {
    if let Some(cost) = cost {
        *total = Some(total.unwrap_or(0).saturating_add(cost));
    }
}

/// The server segment of an `mcp__<server>__<tool>` name.
fn mcp_server_of(tool_name: &str) -> Option<&str> {
    let rest = tool_name.strip_prefix("mcp__")?;
    let (server, _) = rest.split_once("__")?;
    (!server.is_empty()).then_some(server)
}

/// A server name as the registry spells it in a qualified tool name.
fn sanitized(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use serde_json::json;

    use super::*;
    use crate::types::{TodoCreatedProps, TodoDeletedProps, TodoListKind, TodoStatus};

    fn root(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_root".to_owned(), event, SystemTime::UNIX_EPOCH)
    }

    fn child(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_child".to_owned(), event, SystemTime::UNIX_EPOCH)
            .with_parent_session_id("ses_root".to_owned())
    }

    fn message(input: u64, cost: Option<u64>) -> CodingEvent {
        CodingEvent::AssistantMessage {
            text:            "ok".into(),
            model:           "model".into(),
            usage:           TokenUsage {
                input,
                ..TokenUsage::default()
            },
            cost_usd_micros: cost,
            cost_source:     None,
            tool_call_count: 0,
            context_window:  None,
            reasoning:       None,
        }
    }

    #[test]
    fn usage_is_the_roots_and_descendants_are_kept_apart() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::SessionStarted {
            provider: Some("test".into()),
            model:    Some("model".into()),
        }));
        projection.apply(&root(CodingEvent::UserInput {
            text:    "go".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&root(message(10, Some(5))));
        projection.apply(&child(message(7, None)));
        projection.apply(&root(message(20, Some(1))));
        projection.apply(&root(CodingEvent::ProcessingEnd));

        assert_eq!(projection.route.model.as_deref(), Some("model"));
        assert_eq!(projection.usage.input, 30);
        assert_eq!(projection.cost_usd_micros, Some(6));
        assert_eq!(projection.messages, 2);
        assert_eq!(projection.prompt.usage.input, 30);
        assert!(projection.prompt.completed);
        assert_eq!(projection.activity, SessionActivity::Idle);
        let (descendant_usage, descendant_cost) = projection.descendant_usage();
        assert_eq!(descendant_usage.input, 7);
        assert_eq!(descendant_cost, None);
        assert_eq!(projection.descendants["ses_child"].parent, "ses_root");
        assert_eq!(
            projection.prompt.descendant_usage().0.input,
            7,
            "the prompt's delta keeps the descendants' spend apart from the root's"
        );
        assert_eq!(projection.prompts, 1);

        projection.apply(&root(CodingEvent::UserInput {
            text:    "again".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        assert!(projection.prompt.descendants.is_empty());
        assert_eq!(projection.descendants.len(), 1, "the lifetime map keeps it");
    }

    #[test]
    fn a_new_prompt_starts_a_new_delta_and_a_follow_up_does_not() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::UserInput {
            text:    "one".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&root(message(10, None)));
        projection.apply(&root(CodingEvent::UserInput {
            text:    "more".into(),
            content: None,
            source:  InputSource::FollowUp,
        }));
        projection.apply(&root(message(5, None)));
        assert_eq!(
            projection.prompt.usage.input, 15,
            "a follow-up is the same prompt"
        );
        projection.apply(&root(CodingEvent::ProcessingEnd));
        projection.apply(&root(CodingEvent::UserInput {
            text:    "two".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        assert_eq!(
            projection.prompt.usage.input, 0,
            "a new prompt starts from nothing"
        );
        assert!(!projection.prompt.completed);
        assert_eq!(
            projection.usage.input, 15,
            "the lifetime total keeps counting"
        );
        assert_eq!(projection.prompts, 2);
    }

    #[test]
    fn files_are_touched_by_successful_writes_across_the_tree() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::ToolCallStarted {
            tool_name:    "write_file".into(),
            tool_call_id: "w1".into(),
            arguments:    json!({"file_path": "/w/b.txt", "content": "x"}),
        }));
        projection.apply(&child(CodingEvent::ToolCallStarted {
            tool_name:    "edit_file".into(),
            tool_call_id: "e1".into(),
            arguments:    json!({"path": "/w/a.txt", "old_string": "x", "new_string": "y"}),
        }));
        projection.apply(&root(CodingEvent::ToolCallStarted {
            tool_name:    "write_file".into(),
            tool_call_id: "w2".into(),
            arguments:    json!({"file_path": "/w/denied.txt", "content": "x"}),
        }));
        for (id, is_error, name) in [
            ("w1", false, "write_file"),
            ("e1", false, "edit_file"),
            ("w2", true, "write_file"),
        ] {
            let event = CodingEvent::ToolCallCompleted {
                tool_name: name.into(),
                tool_call_id: id.into(),
                output: json!("done"),
                metadata: pebble_agent::ToolOutputMetadata::default(),
                is_error,
                error_kind: None,
                output_bytes_observed: 0,
                output_bytes_retained: 0,
                output_bytes_omitted: 0,
            };
            if id == "e1" {
                projection.apply(&child(event));
            } else {
                projection.apply(&root(event));
            }
        }
        assert_eq!(projection.files_touched, ["/w/a.txt", "/w/b.txt"]);
        assert_eq!(projection.last_file_touched.as_deref(), Some("/w/a.txt"));
        assert_eq!(projection.prompt.files_touched, ["/w/a.txt", "/w/b.txt"]);
        assert_eq!(projection.tools["write_file"].calls, 2);
        assert_eq!(projection.tools["write_file"].errors, 1);
        assert_eq!(projection.tools["write_file"].open, 0);
        assert_eq!(projection.tools["edit_file"].calls, 1);
    }

    #[test]
    fn mcp_servers_skills_todos_and_subagents_fold() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::McpServerReady {
            server: "my-server".into(),
            tools:  vec![McpToolSummary {
                name:          "mcp__my_server__echo".into(),
                original_name: "echo".into(),
            }],
        }));
        projection.apply(&root(CodingEvent::McpServerFailed {
            server: "broken".into(),
            error:  "could not launch".into(),
        }));
        projection.apply(&root(CodingEvent::ToolCallStarted {
            tool_name:    "mcp__my_server__echo".into(),
            tool_call_id: "m1".into(),
            arguments:    json!({}),
        }));
        assert!(projection.mcp_servers["my-server"].invoked);
        assert!(!projection.mcp_servers["broken"].invoked);
        assert_eq!(
            projection.mcp_servers["broken"].error.as_deref(),
            Some("could not launch")
        );
        assert_eq!(projection.mcp_servers["my-server"].disconnected, None);
        // A child's call may be the one that sees the connection close.
        projection.apply(&child(CodingEvent::McpServerDisconnected {
            server: "my-server".into(),
            error:  "transport closed".into(),
        }));
        assert_eq!(
            projection.mcp_servers["my-server"].disconnected.as_deref(),
            Some("transport closed")
        );
        assert_eq!(projection.mcp_servers["my-server"].error, None);

        projection.apply(&root(CodingEvent::SkillsDiscovered {
            profile:     "anthropic".into(),
            source_dirs: vec![],
            skills:      vec![SkillSummary {
                name:        "commit".into(),
                description: "Commit".into(),
            }],
            skipped:     vec![],
        }));
        projection.apply(&child(CodingEvent::SkillActivated {
            skill_name: "commit".into(),
            source:     SkillActivationSource::Tool,
        }));
        assert_eq!(projection.skills.available.len(), 1);
        assert_eq!(projection.skills.activated[0].name, "commit");

        let list_id = TodoListKind::AnthropicTasks.list_id("ses_root");
        projection.apply(&root(CodingEvent::TodoCreated(TodoCreatedProps {
            list_id:     list_id.clone(),
            list_kind:   TodoListKind::AnthropicTasks,
            todo_id:     "t1".into(),
            status:      TodoStatus::Pending,
            order:       0,
            subject:     "write tests".into(),
            description: String::new(),
            active_form: None,
            owner:       None,
            blocks:      vec![],
            blocked_by:  vec![],
            metadata:    BTreeMap::new(),
        })));
        assert_eq!(projection.todos[&list_id].items.len(), 1);
        projection.apply(&root(CodingEvent::TodoDeleted(TodoDeletedProps {
            list_id:   list_id.clone(),
            list_kind: TodoListKind::AnthropicTasks,
            todo_id:   "t1".into(),
        })));
        assert!(
            !projection.todos.contains_key(&list_id),
            "an emptied list is dropped"
        );

        projection.apply(&root(CodingEvent::SubAgentSpawned {
            agent_id:   "a1".into(),
            depth:      1,
            task:       "review".into(),
            generation: 1,
        }));
        projection.apply(&root(CodingEvent::SubAgentCompleted {
            agent_id:   "a1".into(),
            depth:      1,
            generation: 1,
            success:    true,
            turns_used: 3,
        }));
        assert_eq!(projection.subagents.len(), 1);
        assert_eq!(projection.subagents[0].status, SubagentStatus::Completed {
            success:    true,
            turns_used: 3,
        });
        projection.apply(&root(CodingEvent::SubAgentTurnStarted {
            agent_id:   "a1".into(),
            depth:      1,
            task:       "again".into(),
            generation: 2,
        }));
        assert_eq!(
            projection.subagents.len(),
            1,
            "a reused child stays one row"
        );
        assert_eq!(projection.subagents[0].status, SubagentStatus::Running);
        assert_eq!(projection.subagent_counts, SubagentCounts {
            spawned:       1,
            turns_started: 1,
            completed:     1,
            failed:        0,
            closed:        0,
        });
        assert_eq!(projection.prompt.subagents, projection.subagent_counts);

        // A grandchild is spawned by the child and counts for the tree.
        projection.apply(&child(CodingEvent::SubAgentSpawned {
            agent_id:   "a2".into(),
            depth:      2,
            task:       "deeper".into(),
            generation: 1,
        }));
        assert_eq!(projection.subagent_counts.spawned, 2);
        assert_eq!(projection.subagents.len(), 2);
        assert_eq!(projection.subagents[1].depth, 2);
    }

    #[test]
    fn the_projection_survives_a_round_trip_and_resumes() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::UserInput {
            text:    "go".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&root(message(10, None)));
        let stored = serde_json::to_string(&projection).expect("serializes");
        let mut resumed: SessionProjection = serde_json::from_str(&stored).expect("parses");
        assert_eq!(resumed, projection);
        resumed.apply(&root(message(5, None)));
        assert_eq!(resumed.usage.input, 15);
    }
}
