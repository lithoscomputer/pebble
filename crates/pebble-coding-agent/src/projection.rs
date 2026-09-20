//! A value that folds a session's events into what a view or an accountant
//! needs, the same way live or replayed.
//!
//! Both embedders kept their own tallies over the durable event stream — what
//! each answer used and cost, the context window, which tools ran,
//! skills, todo lists, the children and how they ended, compactions, and the
//! files a prompt touched — each written once per application in a different
//! shape. [`SessionProjection`] is that fold, once: feed it every
//! [`CodingAgentEvent`] of one session tree, in order, and read the answers.
//! It is serializable, so a view resumes from a stored value and applies the
//! events after it; and it keeps a [`PromptDelta`] for the prompt in
//! progress, because a retained session spans stages and a stage wants what
//! its own prompts did, not the session's lifetime total.
//!
//! Every spend the projection reports is a [`Usage`]: the tokens the events
//! carry and the cost the catalog or the provider put on them, summed with
//! [`Usage::saturating_add`], so a total has a cost only when every answer in
//! it was priced. Pricing an unpriced answer is the application's.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::compaction::CompactionReason;
use crate::error::ErrorData;
use crate::file_tracker;
use crate::types::{
    CodingAgentEvent, CodingEvent, ContextWindowSnapshot, FailoverContinuation, FailoverStop,
    InputSource, McpToolSummary, SkillActivationSource, SkillSummary, TodoListProjection,
    TodoProjection, Usage, parse_qualified_name, sanitize_mcp_name,
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
    pub parent:      String,
    /// The route it runs on, as its `SessionStarted` reported it, so an
    /// application prices its tokens at its own model. When the start was
    /// not seen, the model of its first answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider:    Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model:       Option<String>,
    /// What it spent: its answers and the summary call of each compaction it
    /// completed.
    pub usage:       Usage,
    /// Committed assistant messages.
    pub messages:    u64,
    /// Compactions it completed.
    pub compactions: u64,
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
    /// How long it took from launch to its outcome, in milliseconds: to its
    /// tools being listed, or to the failure. `None` until either event has
    /// been seen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_ms:   Option<u64>,
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
    /// The summary call's usage: a breakdown of the session's and the
    /// prompt's, which already include it.
    #[serde(default)]
    pub usage:                  Usage,
}

/// One move the root session made to a fallback route, as the stream reported
/// it from the route it moved to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteFailoverProjection {
    /// The `provider/model` that failed.
    pub from:         String,
    /// The `provider/model` the prompt continued on.
    pub to:           String,
    /// How many routes the prompt had moved through, this one included.
    pub attempt:      u32,
    /// The failure that ended the previous route.
    pub error:        ErrorData,
    /// What the prompt spent on the failed route. Already in the session's
    /// and the prompt's totals through that route's committed answers, so a
    /// breakdown of them and not an addition.
    pub usage:        Usage,
    /// Time the prompt spent waiting on the failed route's model, in
    /// milliseconds.
    pub inference_ms: u64,
    /// Time the prompt spent running tools on the failed route, in
    /// milliseconds.
    pub tool_ms:      u64,
    /// How the new route carried the prompt on.
    pub continuation: FailoverContinuation,
}

/// Why a prompt stayed on its route and ended there although fallback routes
/// were named.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailoverStopProjection {
    /// The `provider/model` the prompt ended on.
    pub route:   String,
    /// How many fallback routes the prompt had moved through: `0` on the
    /// route it started on.
    pub attempt: u32,
    pub reason:  FailoverStop,
    /// The failure that ended the prompt.
    pub error:   ErrorData,
}

/// What a span of the session did and spent: the tallies kept once for the
/// session's life and once for the prompt in progress, and moved the same way
/// by every event that touches them.
///
/// Embedded flat in [`SessionProjection`] and [`PromptDelta`], so a stored
/// value has these members at the top level of each.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Totals {
    /// The root session's usage.
    pub usage:             Usage,
    /// Committed assistant messages.
    pub messages:          u64,
    /// The root session's latest context window.
    pub context_window:    Option<ContextWindowSnapshot>,
    /// Model calls retried after a failed attempt, across the tree: every
    /// `LlmRetry`, whichever session's call was replayed.
    #[serde(default)]
    pub retries:           u64,
    /// What each descendant session spent, by id.
    pub descendants:       BTreeMap<String, DescendantAccount>,
    /// Child lifecycle events, across the tree: a child's own children count.
    #[serde(alias = "subagents")]
    pub subagent_counts:   SubagentCounts,
    /// The root session's compactions, in order.
    pub compactions:       Vec<CompactionProjection>,
    /// Files written or edited, across the tree, sorted.
    pub files_touched:     Vec<String>,
    pub last_file_touched: Option<String>,
}

impl Totals {
    /// What every descendant spent, summed.
    #[must_use]
    pub fn descendant_usage(&self) -> Usage {
        self.descendants
            .values()
            .fold(Usage::default(), |sum, account| {
                sum.saturating_add(account.usage)
            })
    }

    /// One more answer from the root, and the context window it reported.
    fn record_answer(&mut self, usage: Usage, context_window: Option<&ContextWindowSnapshot>) {
        self.usage = self.usage.saturating_add(usage);
        self.messages += 1;
        if let Some(window) = context_window {
            self.context_window = Some(window.clone());
        }
    }

    /// A compaction the root completed. Its summary call is billed to the
    /// root, as the prompt report bills it.
    fn record_compaction(&mut self, compaction: CompactionProjection) {
        self.usage = self.usage.saturating_add(compaction.usage);
        self.compactions.push(compaction);
    }

    /// Paths a successful write or edit touched, in touch order.
    fn touch(&mut self, paths: &[String]) {
        for path in paths {
            if !self.files_touched.contains(path) {
                self.files_touched.push(path.clone());
                self.files_touched.sort();
            }
            self.last_file_touched = Some(path.clone());
        }
    }

    /// The account of one descendant, opened under `parent` on first sight.
    fn descendant(&mut self, session_id: &str, parent: &str) -> &mut DescendantAccount {
        self.descendants
            .entry(session_id.to_owned())
            .or_insert_with(|| DescendantAccount {
                parent: parent.to_owned(),
                ..DescendantAccount::default()
            })
    }
}

/// What the prompt in progress, or the last one, did: reset when a prompt
/// starts, complete once [`completed`](Self::completed) is set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptDelta {
    /// Whether the prompt reached its end.
    pub completed:  bool,
    /// What the prompt did and spent.
    #[serde(flatten)]
    pub totals:     Totals,
    /// Tool calls started, across the tree.
    pub tool_calls: u64,
    /// Moves the root made to a fallback route during the prompt.
    #[serde(default)]
    pub failovers:  u32,
}

/// The fold over one session tree's events.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionProjection {
    /// The root session, once an event named it.
    pub root_session_id:  Option<String>,
    pub route:            RouteProjection,
    pub activity:         SessionActivity,
    /// What the session did and spent over its life.
    #[serde(flatten)]
    pub totals:           Totals,
    /// Every tool called anywhere in the tree, by the name the model used.
    pub tools:            BTreeMap<String, ToolActivity>,
    /// Every MCP server the root configured, by name.
    pub mcp_servers:      BTreeMap<String, McpServerProjection>,
    pub skills:           SkillsProjection,
    /// Every todo list in the tree, by list id.
    pub todos:            BTreeMap<String, TodoListProjection>,
    pub subagents:        Vec<SubagentProjection>,
    /// Every move the root made to a fallback route over the session's life,
    /// in order.
    #[serde(default)]
    pub failovers:        Vec<RouteFailoverProjection>,
    /// Why the prompt in progress, or the last one, stayed on its route and
    /// ended there although routes were named, when it did. Cleared when a
    /// prompt starts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failover_stopped: Option<FailoverStopProjection>,
    /// How many prompts have started.
    pub prompts:          u64,
    /// What the prompt in progress, or the last one, did.
    pub prompt:           PromptDelta,
    /// Writes and edits started and not yet completed, by tool call id.
    ///
    /// In-flight bookkeeping, not a fact about the session: it is filled
    /// between a write's `ToolCallStarted` and its `ToolCallCompleted` and
    /// empty otherwise, so it is serialized only when a value is taken
    /// mid-write and a value stored between prompts has no such member.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pending_writes:       BTreeMap<String, Vec<String>>,
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
        // A descendant's event names its parent; the root's names none.
        let descendant = event
            .parent_session_id
            .as_deref()
            .map(|parent| (event.session_id.as_str(), parent));
        let is_root = descendant.is_none();
        if is_root && self.root_session_id.is_none() {
            self.root_session_id = Some(event.session_id.clone());
        }
        match &event.event {
            CodingEvent::SessionStarted { provider, model } => match descendant {
                None => {
                    self.route = RouteProjection {
                        provider: provider.clone(),
                        model:    model.clone(),
                    };
                }
                Some((session_id, parent)) => {
                    for totals in self.both() {
                        let account = totals.descendant(session_id, parent);
                        account.provider.clone_from(provider);
                        account.model.clone_from(model);
                    }
                }
            },
            // The failed route's usage on the event is what that route's
            // `AssistantMessage`s already folded in, so the report and this
            // projection agree without counting it again.
            CodingEvent::RouteFailover {
                from,
                to,
                attempt,
                error,
                usage,
                inference_ms,
                tool_ms,
                continuation,
            } if is_root => {
                if let Some((provider, model)) = to.split_once('/') {
                    self.route = RouteProjection {
                        provider: Some(provider.to_owned()),
                        model:    Some(model.to_owned()),
                    };
                }
                self.prompt.failovers += 1;
                self.failovers.push(RouteFailoverProjection {
                    from:         from.clone(),
                    to:           to.clone(),
                    attempt:      *attempt,
                    error:        error.clone(),
                    usage:        *usage,
                    inference_ms: *inference_ms,
                    tool_ms:      *tool_ms,
                    continuation: *continuation,
                });
            }
            CodingEvent::RouteFailoverStopped {
                route,
                attempt,
                reason,
                error,
            } if is_root => {
                self.failover_stopped = Some(FailoverStopProjection {
                    route:   route.clone(),
                    attempt: *attempt,
                    reason:  *reason,
                    error:   error.clone(),
                });
            }
            CodingEvent::SessionEnded if is_root => self.activity = SessionActivity::Ended,
            CodingEvent::UserInput { source, .. } if is_root => {
                if *source == InputSource::Prompt {
                    self.prompts += 1;
                    self.prompt = PromptDelta::default();
                    self.failover_stopped = None;
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
                model,
                usage,
                context_window,
                ..
            } => match descendant {
                None => {
                    for totals in self.both() {
                        totals.record_answer(*usage, context_window.as_ref());
                    }
                }
                Some((session_id, parent)) => {
                    for totals in self.both() {
                        let account = totals.descendant(session_id, parent);
                        account.usage = account.usage.saturating_add(*usage);
                        account.messages += 1;
                        // The start names the route; an answer seen without
                        // one still says which model to price it at.
                        account.model.get_or_insert_with(|| model.clone());
                    }
                }
            },
            // A child's retries count for the tree, as its tool calls do.
            CodingEvent::LlmRetry { .. } => {
                for totals in self.both() {
                    totals.retries += 1;
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
                if let Some((server, _)) = parse_qualified_name(tool_name)
                    && let Some(projection) = self
                        .mcp_servers
                        .iter_mut()
                        .find(|(name, _)| sanitize_mcp_name(name) == server)
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
                    for totals in self.both() {
                        totals.touch(&paths);
                    }
                }
            }
            CodingEvent::McpServerReady {
                server,
                tools,
                startup_ms,
            } if is_root => {
                let projection = self.mcp_servers.entry(server.clone()).or_default();
                projection.tools.clone_from(tools);
                projection.error = None;
                projection.startup_ms = Some(*startup_ms);
            }
            CodingEvent::McpServerFailed {
                server,
                error,
                startup_ms,
            } if is_root => {
                let projection = self.mcp_servers.entry(server.clone()).or_default();
                projection.tools.clear();
                projection.error = Some(error.clone());
                projection.startup_ms = Some(*startup_ms);
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
                for totals in self.both() {
                    totals.subagent_counts.spawned += 1;
                }
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
                for totals in self.both() {
                    totals.subagent_counts.turns_started += 1;
                }
                self.set_subagent_status(agent_id, SubagentStatus::Running);
            }
            CodingEvent::SubAgentCompleted {
                agent_id,
                success,
                turns_used,
                ..
            } => {
                for totals in self.both() {
                    totals.subagent_counts.completed += 1;
                }
                self.set_subagent_status(agent_id, SubagentStatus::Completed {
                    success:    *success,
                    turns_used: *turns_used,
                });
            }
            CodingEvent::SubAgentFailed {
                agent_id, error, ..
            } => {
                for totals in self.both() {
                    totals.subagent_counts.failed += 1;
                }
                self.set_subagent_status(agent_id, SubagentStatus::Failed {
                    error: error.clone(),
                });
            }
            CodingEvent::SubAgentClosed { agent_id, .. } => {
                for totals in self.both() {
                    totals.subagent_counts.closed += 1;
                }
                self.set_subagent_status(agent_id, SubagentStatus::Closed);
            }
            // The summary call is billed to the session that compacted, as
            // its report bills it. A failed compaction is not: the report
            // leaves it out, so the fold does too, and `CompactionFailed`
            // falls through below.
            CodingEvent::CompactionCompleted {
                original_turn_count,
                preserved_turn_count,
                summary_token_estimate,
                tracked_file_count,
                reason,
                usage,
            } => match descendant {
                None => {
                    let compaction = CompactionProjection {
                        reason:                 *reason,
                        original_turn_count:    *original_turn_count,
                        preserved_turn_count:   *preserved_turn_count,
                        summary_token_estimate: *summary_token_estimate,
                        tracked_file_count:     *tracked_file_count,
                        usage:                  *usage,
                    };
                    for totals in self.both() {
                        totals.record_compaction(compaction.clone());
                    }
                }
                Some((session_id, parent)) => {
                    for totals in self.both() {
                        let account = totals.descendant(session_id, parent);
                        account.usage = account.usage.saturating_add(*usage);
                        account.compactions += 1;
                    }
                }
            },
            _ => {}
        }
    }

    /// Folds every event in, in order.
    pub fn apply_all<'a>(&mut self, events: impl IntoIterator<Item = &'a CodingAgentEvent>) {
        for event in events {
            self.apply(event);
        }
    }

    /// The session's lifetime tallies and the prompt's, for a fact both keep.
    fn both(&mut self) -> [&mut Totals; 2] {
        [&mut self.totals, &mut self.prompt.totals]
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

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use serde_json::json;

    use super::*;
    use crate::error::ErrorKind;
    use crate::types::{
        Cost, CostSource, LlmRetryPhase, TodoCreatedProps, TodoDeletedProps, TodoListKind,
        TodoStatus, TokenCounts,
    };

    fn root(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_root".to_owned(), event, SystemTime::UNIX_EPOCH)
    }

    fn child(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_child".to_owned(), event, SystemTime::UNIX_EPOCH)
            .with_parent_session_id("ses_root".to_owned())
    }

    /// `input` tokens, priced from the catalog when `cost` is given.
    fn priced(input: u64, cost: Option<u64>) -> Usage {
        Usage {
            tokens: TokenCounts {
                input,
                ..TokenCounts::default()
            },
            cost:   cost.map(|usd_micros| Cost {
                usd_micros,
                source: CostSource::Catalog,
            }),
        }
    }

    fn message(input: u64, cost: Option<u64>) -> CodingEvent {
        CodingEvent::AssistantMessage {
            text:            "ok".into(),
            model:           "model".into(),
            usage:           priced(input, cost),
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
        assert_eq!(projection.totals.usage, priced(30, Some(6)));
        assert_eq!(projection.totals.messages, 2);
        assert_eq!(projection.prompt.totals.usage, priced(30, Some(6)));
        assert!(projection.prompt.completed);
        assert_eq!(projection.activity, SessionActivity::Idle);
        assert_eq!(
            projection.totals.descendant_usage(),
            priced(7, None),
            "an unpriced answer leaves the sum unpriced"
        );
        assert_eq!(
            projection.totals.descendants["ses_child"].parent,
            "ses_root"
        );
        assert_eq!(
            projection.prompt.totals.descendant_usage(),
            priced(7, None),
            "the prompt's delta keeps the descendants' spend apart from the root's"
        );
        assert_eq!(projection.prompts, 1);

        projection.apply(&root(CodingEvent::UserInput {
            text:    "again".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        assert!(projection.prompt.totals.descendants.is_empty());
        assert_eq!(
            projection.totals.descendants.len(),
            1,
            "the lifetime map keeps it"
        );
    }

    fn compaction(input: u64, cost: Option<u64>) -> CodingEvent {
        CodingEvent::CompactionCompleted {
            original_turn_count:    6,
            preserved_turn_count:   2,
            summary_token_estimate: 40,
            tracked_file_count:     1,
            reason:                 CompactionReason::Threshold,
            usage:                  priced(input, cost),
        }
    }

    #[test]
    fn a_compactions_summary_call_is_billed_where_the_report_bills_it() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::UserInput {
            text:    "go".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&root(message(10, Some(5))));
        projection.apply(&root(compaction(30, Some(2))));
        projection.apply(&child(compaction(4, None)));
        projection.apply(&root(CodingEvent::CompactionFailed {
            reason: CompactionReason::Manual,
            error:  ErrorData::new(ErrorKind::Compaction, "empty summary"),
            usage:  Some(priced(100, Some(50))),
        }));

        assert_eq!(
            projection.totals.usage,
            priced(40, Some(7)),
            "the summary call is in the total, its cost included"
        );
        assert_eq!(projection.prompt.totals.usage, priced(40, Some(7)));
        assert_eq!(projection.totals.messages, 1, "a compaction is not a turn");
        assert_eq!(projection.totals.compactions[0].usage, priced(30, Some(2)));
        assert_eq!(
            projection.prompt.totals.compactions,
            projection.totals.compactions
        );
        assert_eq!(
            projection.totals.descendants["ses_child"].usage,
            priced(4, None),
            "a child's compaction is the child's spend"
        );
        assert_eq!(projection.totals.descendants["ses_child"].compactions, 1);
        assert_eq!(
            projection.totals.usage.tokens.input, 40,
            "a failed compaction is not billed, as the report does not bill it"
        );
    }

    fn retry() -> CodingEvent {
        CodingEvent::LlmRetry {
            provider:   "test".into(),
            model:      "model".into(),
            attempt:    0,
            delay_secs: 0.1,
            error:      ErrorData::new(ErrorKind::Llm, "slow down"),
            phase:      LlmRetryPhase::Open,
        }
    }

    #[test]
    fn a_descendant_is_priced_at_its_own_model() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::SessionStarted {
            provider: Some("test".into()),
            model:    Some("big".into()),
        }));
        projection.apply(&root(CodingEvent::UserInput {
            text:    "go".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&child(CodingEvent::SessionStarted {
            provider: Some("test".into()),
            model:    Some("small".into()),
        }));
        projection.apply(&child(message(7, None)));
        // A grandchild whose start this fold never saw: its answer's model
        // stands in.
        let grandchild = CodingAgentEvent::new(
            "ses_grandchild".to_owned(),
            CodingEvent::AssistantMessage {
                text:            "ok".into(),
                model:           "tiny".into(),
                usage:           Usage::default(),
                tool_call_count: 0,
                context_window:  None,
                reasoning:       None,
            },
            SystemTime::UNIX_EPOCH,
        )
        .with_parent_session_id("ses_child".to_owned());
        projection.apply(&grandchild);

        assert_eq!(projection.route.model.as_deref(), Some("big"));
        let child_account = &projection.totals.descendants["ses_child"];
        assert_eq!(child_account.provider.as_deref(), Some("test"));
        assert_eq!(
            child_account.model.as_deref(),
            Some("small"),
            "the start's route wins over the answer's model"
        );
        assert_eq!(child_account.usage, priced(7, None));
        assert_eq!(
            projection.prompt.totals.descendants["ses_child"]
                .model
                .as_deref(),
            Some("small"),
            "the prompt's account names the model too"
        );
        let grandchild_account = &projection.totals.descendants["ses_grandchild"];
        assert_eq!(grandchild_account.parent, "ses_child");
        assert_eq!(grandchild_account.provider, None);
        assert_eq!(grandchild_account.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn retries_count_across_the_tree_and_restart_with_the_prompt() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::UserInput {
            text:    "go".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&root(retry()));
        projection.apply(&child(retry()));
        assert_eq!(
            projection.totals.retries, 2,
            "a child's retry counts for the tree"
        );
        assert_eq!(projection.prompt.totals.retries, 2);

        projection.apply(&root(CodingEvent::ProcessingEnd));
        projection.apply(&root(CodingEvent::UserInput {
            text:    "again".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        assert_eq!(
            projection.prompt.totals.retries, 0,
            "a new prompt starts from nothing"
        );
        assert_eq!(
            projection.totals.retries, 2,
            "the lifetime count keeps counting"
        );
    }

    fn failover(from: &str, to: &str, attempt: u32) -> CodingEvent {
        CodingEvent::RouteFailover {
            from: from.into(),
            to: to.into(),
            attempt,
            error: ErrorData::new(ErrorKind::Llm, "key revoked"),
            usage: priced(10, Some(7)),
            inference_ms: 120,
            tool_ms: 30,
            continuation: FailoverContinuation::ContinueTurn,
        }
    }

    #[test]
    fn failovers_are_kept_in_order_and_a_stop_lasts_one_prompt() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::SessionStarted {
            provider: Some("a".into()),
            model:    Some("one".into()),
        }));
        projection.apply(&root(CodingEvent::UserInput {
            text:    "go".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&root(message(10, Some(7))));
        projection.apply(&root(failover("a/one", "b/two", 1)));
        projection.apply(&root(failover("b/two", "c/three", 2)));
        projection.apply(&root(CodingEvent::RouteFailoverStopped {
            route:   "c/three".into(),
            attempt: 2,
            reason:  FailoverStop::Exhausted,
            error:   ErrorData::new(ErrorKind::Llm, "key revoked"),
        }));
        projection.apply(&child(failover("x/y", "z/w", 9)));

        assert_eq!(projection.failovers.len(), 2, "a child's move is its own");
        assert_eq!(projection.failovers[0].from, "a/one");
        assert_eq!(projection.failovers[0].to, "b/two");
        assert_eq!(projection.failovers[0].attempt, 1);
        assert_eq!(projection.failovers[0].usage, priced(10, Some(7)));
        assert_eq!(projection.failovers[0].inference_ms, 120);
        assert_eq!(projection.failovers[0].tool_ms, 30);
        assert_eq!(
            projection.failovers[0].continuation,
            FailoverContinuation::ContinueTurn
        );
        assert_eq!(projection.failovers[1].attempt, 2);
        assert_eq!(projection.prompt.failovers, 2);
        assert_eq!(projection.route.provider.as_deref(), Some("c"));
        assert_eq!(projection.route.model.as_deref(), Some("three"));
        assert_eq!(
            projection.totals.usage,
            priced(10, Some(7)),
            "the failed route's spend on the event is not counted again"
        );
        let stopped = projection
            .failover_stopped
            .as_ref()
            .expect("the stop is kept");
        assert_eq!(stopped.route, "c/three");
        assert_eq!(stopped.attempt, 2);
        assert_eq!(stopped.reason, FailoverStop::Exhausted);
        assert_eq!(stopped.error.message, "key revoked");

        projection.apply(&root(CodingEvent::UserInput {
            text:    "again".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        assert_eq!(
            projection.prompt.failovers, 0,
            "the prompt's count restarts"
        );
        assert!(
            projection.failover_stopped.is_none(),
            "a stop is the prompt's, not the session's"
        );
        assert_eq!(projection.failovers.len(), 2, "the history keeps its moves");
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
            projection.prompt.totals.usage.tokens.input, 15,
            "a follow-up is the same prompt"
        );
        projection.apply(&root(CodingEvent::ProcessingEnd));
        projection.apply(&root(CodingEvent::UserInput {
            text:    "two".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        assert_eq!(
            projection.prompt.totals.usage.tokens.input, 0,
            "a new prompt starts from nothing"
        );
        assert!(!projection.prompt.completed);
        assert_eq!(
            projection.totals.usage.tokens.input, 15,
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
        assert_eq!(projection.totals.files_touched, ["/w/a.txt", "/w/b.txt"]);
        assert_eq!(
            projection.totals.last_file_touched.as_deref(),
            Some("/w/a.txt")
        );
        assert_eq!(projection.prompt.totals.files_touched, [
            "/w/a.txt", "/w/b.txt"
        ]);
        assert_eq!(projection.tools["write_file"].calls, 2);
        assert_eq!(projection.tools["write_file"].errors, 1);
        assert_eq!(projection.tools["write_file"].open, 0);
        assert_eq!(projection.tools["edit_file"].calls, 1);
    }

    #[test]
    fn mcp_servers_skills_todos_and_subagents_fold() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::McpServerReady {
            server:     "my-server".into(),
            tools:      vec![McpToolSummary {
                name:          "mcp__my_server__echo".into(),
                original_name: "echo".into(),
            }],
            startup_ms: 120,
        }));
        projection.apply(&root(CodingEvent::McpServerFailed {
            server:     "broken".into(),
            error:      "could not launch".into(),
            startup_ms: 3,
        }));
        projection.apply(&root(CodingEvent::ToolCallStarted {
            tool_name:    "mcp__my_server__echo".into(),
            tool_call_id: "m1".into(),
            arguments:    json!({}),
        }));
        assert!(projection.mcp_servers["my-server"].invoked);
        assert!(!projection.mcp_servers["broken"].invoked);
        assert_eq!(projection.mcp_servers["my-server"].startup_ms, Some(120));
        assert_eq!(projection.mcp_servers["broken"].startup_ms, Some(3));
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
        assert_eq!(projection.totals.subagent_counts, SubagentCounts {
            spawned:       1,
            turns_started: 1,
            completed:     1,
            failed:        0,
            closed:        0,
        });
        assert_eq!(
            projection.prompt.totals.subagent_counts,
            projection.totals.subagent_counts
        );

        // A grandchild is spawned by the child and counts for the tree.
        projection.apply(&child(CodingEvent::SubAgentSpawned {
            agent_id:   "a2".into(),
            depth:      2,
            task:       "deeper".into(),
            generation: 1,
        }));
        assert_eq!(projection.totals.subagent_counts.spawned, 2);
        assert_eq!(projection.subagents.len(), 2);
        assert_eq!(projection.subagents[1].depth, 2);
    }

    #[test]
    fn in_flight_writes_are_serialized_only_mid_write() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::ToolCallStarted {
            tool_name:    "write_file".into(),
            tool_call_id: "w1".into(),
            arguments:    json!({"file_path": "/w/b.txt", "content": "x"}),
        }));
        let mid_write = serde_json::to_value(&projection).expect("serializes");
        assert_eq!(
            mid_write["pending_writes"]["w1"],
            json!(["/w/b.txt"]),
            "a value taken mid-write carries the open write"
        );
        let resumed: SessionProjection = serde_json::from_value(mid_write).expect("parses");
        assert_eq!(resumed, projection);

        projection.apply(&root(CodingEvent::ToolCallCompleted {
            tool_name:             "write_file".into(),
            tool_call_id:          "w1".into(),
            output:                json!("done"),
            metadata:              pebble_agent::ToolOutputMetadata::default(),
            is_error:              false,
            error_kind:            None,
            output_bytes_observed: 0,
            output_bytes_retained: 0,
            output_bytes_omitted:  0,
        }));
        let settled = serde_json::to_value(&projection).expect("serializes");
        assert!(
            settled.get("pending_writes").is_none(),
            "nothing in flight, nothing on the wire: {settled}"
        );
        assert_eq!(projection.totals.files_touched, ["/w/b.txt"]);
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
        assert_eq!(resumed.totals.usage.tokens.input, 15);
    }

    /// The tallies are one type held twice, but a stored value shows them
    /// flat: at the top level of the session and of its prompt, as before
    /// there was one type.
    #[test]
    fn the_tallies_are_stored_flat_in_the_session_and_the_prompt() {
        let mut projection = SessionProjection::new();
        projection.apply(&root(CodingEvent::UserInput {
            text:    "go".into(),
            content: None,
            source:  InputSource::Prompt,
        }));
        projection.apply(&root(message(10, None)));
        let stored = serde_json::to_value(&projection).expect("serializes");

        for object in [&stored, &stored["prompt"]] {
            let keys = object.as_object().expect("an object");
            assert!(keys.contains_key("usage"), "{object}");
            assert!(keys.contains_key("messages"), "{object}");
            assert!(keys.contains_key("subagent_counts"), "{object}");
            assert!(keys.contains_key("files_touched"), "{object}");
            assert!(!keys.contains_key("totals"), "{object}");
        }
        assert_eq!(stored["messages"], 1);
        assert_eq!(stored["prompt"]["messages"], 1);
        assert_eq!(stored["prompt"]["completed"], false);
    }
}
