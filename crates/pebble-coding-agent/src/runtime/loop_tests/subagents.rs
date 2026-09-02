//! What a session does with the children it spawns.
//!
//! Fabro's session-level subagent tests, ported: background results arrive in
//! one extra parent turn and are never read as skill references, an interrupt
//! or a cancellation while the parent waits closes the child, and a session
//! closes its children before it publishes its own end.
//!
//! Where fabro gave a blocked child a session of its own, these tests supervise
//! a blocked *task* instead: pebble's children share their parent's scripted
//! client, so a second session would race the parent for the script. What is
//! under test either way is the parent — the wait tool it is blocked in, the
//! close its interrupt causes, and the token the child sees.
//!
//! Beside them are the rules pebble added and the plan asks for: what a child
//! may be given (never a person to ask, never a tool or a permission its parent
//! lacked), what a tree may hold open, where a grandchild's news arrives, and
//! what a closed tree leaves running.

use std::collections::HashMap;
use std::iter;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use serde_json::json;
use tokio::runtime::Handle;
use tokio::time::timeout;

use super::super::testing::{builder, wait_for_event};
use super::*;
use crate::event::{EventSink, EventSinkError};
use crate::human_input::{Answer, HumanInputError, HumanInputProvider, Question};
use crate::subagent::{
    ChildObserver, SubagentLimits, SubagentResult, SubagentStatus, SubagentSupervisor,
};
use crate::test_support::{MockEnvironment, ScriptedProvider, scripted_client};
use crate::types::ToolErrorKind;

/// The identifier the blocked child is supervised under, so a script can name
/// it before it exists.
const BLOCKED_AGENT: &str = "blocked-agent";

/// One child a recording factory built, so a test can drive the tree below it.
#[derive(Clone)]
struct ChildHandle {
    id:         String,
    supervisor: SubagentSupervisor,
}

/// An observer that keeps hold of each child the tree builds.
///
/// A child session is moved into its runner task as soon as it is supervised,
/// so the observer is the only place a test can take its identity and the
/// supervisor that drives its own children.
fn recording_observer() -> (ChildObserver, Arc<Mutex<Vec<ChildHandle>>>) {
    let recorded: Arc<Mutex<Vec<ChildHandle>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&recorded);
    let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
        if let Some(supervisor) = child.subagent_supervisor() {
            recorder
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(ChildHandle {
                    id:         child.id().to_owned(),
                    supervisor: supervisor.clone(),
                });
        }
    });
    (observer, recorded)
}

/// The first child the recording factory built.
fn first_child(recorded: &Arc<Mutex<Vec<ChildHandle>>>) -> ChildHandle {
    nth_child(recorded, 0)
}

/// The `index`th session the recording factory built, in the order it was
/// asked for them.
///
/// A child inherits the factory, so the whole tree below the root is recorded
/// here: the root's child first, then that child's, and so on down.
fn nth_child(recorded: &Arc<Mutex<Vec<ChildHandle>>>, index: usize) -> ChildHandle {
    recorded
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(index)
        .cloned()
        .unwrap_or_else(|| panic!("the factory built at least {} sessions", index + 1))
}

/// A tool that reports whether the call it ran in could have asked a person.
fn person_probe(seen: Arc<Mutex<Vec<bool>>>) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            "probe_for_a_person",
            "Reports whether this call could ask a person",
            json!({ "type": "object" }),
        ),
        Arc::new(move |_arguments, context| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                seen.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(context.human_input.is_some());
                Ok("recorded".to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// A parent whose one round calls `wait` on a child that never finishes.
///
/// Answers with the session, its supervisor, and the token the child watches,
/// which is what a test asserts the close reached.
fn parent_waiting_on_a_blocked_child() -> (
    CodingRuntime,
    SubagentSupervisor,
    CancellationToken,
    JoinHandle<()>,
) {
    let (session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "wait",
            "parent_wait_call",
            json!({ "agent_id": BLOCKED_AGENT }),
        )),
        ScriptedCall::response(text_response("resumed")),
    ])
    .with_subagents()
    .build();
    let supervisor = session
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();

    let child_cancel = CancellationToken::new();
    let watched = child_cancel.clone();
    let child = tokio::spawn(async move {
        watched.cancelled().await;
        Ok(SubagentResult {
            output:     String::new(),
            success:    false,
            turns_used: 0,
        })
    });
    supervisor.supervise_test_task(BLOCKED_AGENT.to_owned(), child, child_cancel.clone());

    let events = session.subscribe();
    let watcher = tokio::spawn(async move {
        let mut events = events;
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::ToolCallStarted { tool_name, .. } if tool_name == "wait")
        })
        .await;
    });

    (session, supervisor, child_cancel, watcher)
}

#[tokio::test]
async fn background_agent_notifications_are_batched_into_one_parent_turn() {
    let (mut parent, _provider) = TestSession::new(vec![
        ScriptedCall::response(text_response("first result")),
        ScriptedCall::response(text_response("second result")),
        ScriptedCall::response(text_response("Parent is waiting")),
        ScriptedCall::response(text_response("Synthesized both results")),
    ])
    .with_subagents()
    .build();
    let supervisor = parent
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();

    // Both results are ready before the parent reaches a boundary, one child at
    // a time so each takes the script entry meant for it.
    let first = supervisor
        .spawn_with_parent_notification(
            parent.id(),
            parent.root_session_id(),
            "first task".to_owned(),
            "Inspect first".to_owned(),
        )
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&first, &CancellationToken::new())
        .await
        .expect("the first child answers");
    let second = supervisor
        .spawn_with_parent_notification(
            parent.id(),
            parent.root_session_id(),
            "second task".to_owned(),
            "Inspect second".to_owned(),
        )
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&second, &CancellationToken::new())
        .await
        .expect("the second child answers");

    let output = parent
        .prompt("Delegate both tasks")
        .await
        .expect("the prompt succeeds");

    assert_eq!(output.as_deref(), Some("Synthesized both results"));
    let turns = parent.history().turns().to_vec();
    assert_eq!(turns.len(), 4, "one extra turn carries both results");
    let Some(Message::User {
        content: notification,
        ..
    }) = turns.get(2)
    else {
        panic!("the third turn delivers the background results: {turns:?}");
    };
    assert_eq!(notification.matches("<task-notification>").count(), 2);
    assert!(notification.contains(&first));
    assert!(notification.contains(&second));
    assert!(notification.contains("first result"));
    assert!(notification.contains("second result"));

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

#[tokio::test]
async fn background_agent_output_is_not_parsed_for_skill_references() {
    let environment = Arc::new(MockEnvironment {
        files: HashMap::from([(
            "/skills/commit/SKILL.md".to_owned(),
            "---\nname: commit\ndescription: Make a commit\n---\nReview changes and commit."
                .to_owned(),
        )]),
        glob_results: vec!["/skills/commit/SKILL.md".to_owned()],
        ..MockEnvironment::linux()
    });
    let (mut parent, _provider) = TestSession::new(vec![
        ScriptedCall::response(text_response("Cleaned up /tmp and exited")),
        ScriptedCall::response(text_response("Delegated")),
        ScriptedCall::response(text_response("Acknowledged")),
    ])
    .environment(environment)
    .options(CodingAgentOptions {
        skill_dirs: vec!["/skills".to_owned()],
        ..CodingAgentOptions::default()
    })
    .with_subagents()
    .build();
    parent.initialize().await.expect("initialization succeeds");
    let supervisor = parent
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();
    let child = supervisor
        .spawn_with_parent_notification(
            parent.id(),
            parent.root_session_id(),
            "clean up".to_owned(),
            "Clean scratch files".to_owned(),
        )
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&child, &CancellationToken::new())
        .await
        .expect("the child answers");

    // A child that mentions a bare path must not fail the parent turn on
    // `Unknown skill: /tmp`, nor have its report replaced by a skill body.
    let output = parent
        .prompt("Delegate the cleanup")
        .await
        .expect("the prompt succeeds");

    assert_eq!(output.as_deref(), Some("Acknowledged"));
    let turns = parent.history().turns().to_vec();
    let Some(Message::User {
        content: notification,
        ..
    }) = turns.get(2)
    else {
        panic!("the third turn delivers the background result: {turns:?}");
    };
    assert!(notification.contains("Cleaned up /tmp and exited"));
    assert!(!notification.contains("Review changes and commit."));

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

#[tokio::test]
async fn control_interrupt_during_subagent_wait_closes_child_and_resumes_after_steer() {
    let (mut session, supervisor, child_cancel, waiting) = parent_waiting_on_a_blocked_child();
    let control = session.control_handle();
    let mut recorded = session.subscribe();
    let mut watched = session.subscribe();
    let controller = control.clone();
    let interrupter = tokio::spawn(async move {
        waiting.await.expect("the wait tool started");
        controller.interrupt();
        wait_for_event(&mut watched, |event| {
            matches!(event, CodingEvent::SubAgentClosed { .. })
        })
        .await;
        wait_for_event(&mut watched, |event| {
            matches!(event, CodingEvent::RoundInterrupted { generation: 1 })
        })
        .await;
        assert!(controller.is_paused());
        controller.enqueue_steering("resume after interrupt");
    });

    let output = timeout(Duration::from_secs(5), session.prompt("wait for the child"))
        .await
        .expect("an interrupt unblocks the subagent wait")
        .expect("the prompt succeeds");
    interrupter.await.expect("the controller finishes");

    assert_eq!(output.as_deref(), Some("resumed"));
    assert_eq!(session.state(), CodingAgentState::Idle);
    assert!(child_cancel.is_cancelled(), "the child was closed with it");
    assert!(matches!(
        supervisor.status(BLOCKED_AGENT),
        Some(SubagentStatus::Closed)
    ));
    assert!(!control.is_paused());

    let published = drained(&mut recorded).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::RoundInterrupted { .. }
        )),
        1,
        "one gesture is announced once"
    );
    let closed = position(&published, |event| {
        matches!(event, CodingEvent::SubAgentClosed { .. })
    })
    .expect("the child was closed");
    let settled = position(&published, |event| {
        matches!(event, CodingEvent::RoundInterrupted { .. })
    })
    .expect("the interrupt was announced");
    assert!(closed < settled, "the child closes as the round unwinds");

    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

#[tokio::test]
async fn terminal_cancel_during_subagent_wait_closes_child_and_session() {
    let (mut session, supervisor, child_cancel, waiting) = parent_waiting_on_a_blocked_child();
    let cancel = session.cancel_token();
    let canceller = tokio::spawn(async move {
        waiting.await.expect("the wait tool started");
        cancel.cancel();
    });

    let result = timeout(Duration::from_secs(5), session.prompt("wait for the child"))
        .await
        .expect("a terminal cancellation unblocks the subagent wait");
    canceller.await.expect("the controller finishes");

    assert!(
        matches!(result, Err(Error::Interrupted(InterruptReason::Cancelled))),
        "{result:?}"
    );
    assert_eq!(session.state(), CodingAgentState::Closed);
    assert!(child_cancel.is_cancelled(), "the child was closed with it");
    assert!(matches!(
        supervisor.status(BLOCKED_AGENT),
        Some(SubagentStatus::Closed)
    ));
}

#[tokio::test]
async fn shutdown_cleans_up_subagents_before_emitting_session_ended() {
    // Every model call takes a minute, so the child is still working when its
    // parent is asked to close. The parent itself never calls the model.
    let (mut session, _provider) = TestSession::new(answers("child done"))
        .delayed(Duration::from_secs(60))
        .with_subagents()
        .build();
    let supervisor = session
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();
    let agent_id = supervisor
        .spawn(session.id(), session.root_session_id(), "task".to_owned())
        .expect("the spawn succeeds");
    let mut events = session.subscribe();

    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");

    assert!(matches!(
        supervisor.status(&agent_id),
        Some(SubagentStatus::Closed)
    ));
    let published: Vec<_> = iter::from_fn(|| events.try_recv().ok()).collect();
    let child_ended = published
        .iter()
        .position(|event| {
            event.session_id != session.id() && matches!(event.event, CodingEvent::SessionEnded)
        })
        .expect("the child published its end");
    let closed = published
        .iter()
        .position(|event| matches!(event.event, CodingEvent::SubAgentClosed { .. }))
        .expect("the child was closed");
    let ended = published
        .iter()
        .position(|event| {
            event.session_id == session.id() && matches!(event.event, CodingEvent::SessionEnded)
        })
        .expect("the session published its end");
    assert!(
        child_ended < closed && closed < ended,
        "a tree unwinds from the leaves"
    );
}

// --- What a child may and may not be given ---

/// A person who is never actually asked, for a session that has one.
struct AlwaysSilent;

#[async_trait::async_trait]
impl HumanInputProvider for AlwaysSilent {
    async fn ask_questions(
        &self,
        _tool_call_id: &str,
        _questions: Vec<Question>,
        _cancel_token: CancellationToken,
    ) -> StdResult<Vec<Answer>, HumanInputError> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn a_child_cannot_ask_a_person_a_question() {
    let seen: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let (mut parent, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "probe_for_a_person",
            "parent_probe",
            json!({}),
        )),
        ScriptedCall::response(text_response("the parent could ask")),
        ScriptedCall::response(tool_call_response(
            "probe_for_a_person",
            "child_probe",
            json!({}),
        )),
        ScriptedCall::response(text_response("the child could not")),
    ])
    .tools(vec![person_probe(Arc::clone(&seen))])
    .human_input(Arc::new(AlwaysSilent))
    .with_subagents()
    .build();
    let supervisor = parent
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();

    parent
        .prompt("probe")
        .await
        .expect("the parent's prompt succeeds");
    let agent_id = supervisor
        .spawn(
            parent.id(),
            parent.root_session_id(),
            "probe too".to_owned(),
        )
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&agent_id, &CancellationToken::new())
        .await
        .expect("the child answers");

    assert_eq!(
        *seen.lock().unwrap_or_else(PoisonError::into_inner),
        vec![true, false],
        "the root reaches a person; the child it spawned never can"
    );
    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

#[tokio::test]
async fn a_spawn_the_tree_has_no_room_for_is_answered_and_the_parent_carries_on() {
    // One open session, counting the root: the tools are registered and
    // answering, and every spawn is refused.
    let (mut parent, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "spawn_agent",
            "spawn_call",
            json!({ "task": "review the diff" }),
        )),
        ScriptedCall::response(text_response("I reviewed it myself")),
    ])
    .with_subagents()
    .subagent_limits(SubagentLimits::new(1))
    .build();
    let mut events = parent.subscribe();

    let output = parent
        .prompt("delegate the review")
        .await
        .expect("a refused spawn is the tool's answer, not the prompt's failure");

    assert_eq!(output.as_deref(), Some("I reviewed it myself"));
    let results = tool_results(&parent, 2);
    let refusal = result_text(results.first().expect("the spawn was answered"));
    assert!(
        refusal.contains("Cannot spawn another agent"),
        "the model is told why: {refusal}"
    );
    assert!(refusal.contains("(1, counting the root)"), "{refusal}");
    let published = settled(&mut parent, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::ToolCallCompleted {
                tool_name,
                is_error: true,
                error_kind: Some(ToolErrorKind::Denied),
                ..
            } if tool_name == "spawn_agent"
        )),
        "{published:?}"
    );
    assert!(
        !published
            .iter()
            .any(|event| matches!(event, CodingEvent::SubAgentSpawned { .. })),
        "nothing was spawned"
    );
}

// --- Events from deeper in the tree ---

#[tokio::test]
async fn a_grandchilds_news_reaches_the_root_stream() {
    let (observer, children) = recording_observer();
    let (mut parent, _provider) = TestSession::new(answers("done"))
        .observe_children(observer)
        .build();
    let supervisor = parent
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();
    let mut events = parent.subscribe();

    let child_id = supervisor
        .spawn(parent.id(), parent.root_session_id(), "delegate".to_owned())
        .expect("the spawn succeeds");
    let child = first_child(&children);
    let grandchild = child
        .supervisor
        .spawn(
            &child.id,
            parent.root_session_id(),
            "the leaf task".to_owned(),
        )
        .expect("a child may spawn a child of its own");
    // A third level proves that every descendant writes to the same stream and
    // still names its immediate parent.
    let grandchild_session = nth_child(&children, 1);
    let great_grandchild = grandchild_session
        .supervisor
        .spawn(
            &grandchild_session.id,
            parent.root_session_id(),
            "the deepest task".to_owned(),
        )
        .expect("a grandchild may spawn a child of its own");

    let (from_depth_two, from_depth_three) = timeout(Duration::from_secs(5), async {
        let (mut two, mut three) = (None, None);
        while two.is_none() || three.is_none() {
            let event = events.recv().await.expect("the parent's stream stays open");
            match event.event {
                CodingEvent::SubAgentSpawned { depth: 2, .. } => two = Some(event),
                CodingEvent::SubAgentSpawned { depth: 3, .. } => three = Some(event),
                _ => {}
            }
        }
        (two.expect("seen"), three.expect("seen"))
    })
    .await
    .expect("news from either level reaches the root");

    assert_eq!(
        from_depth_two.session_id, child.id,
        "the event keeps the session it happened in"
    );
    assert_eq!(
        from_depth_two.parent_session_id.as_deref(),
        Some(parent.id()),
        "and names its immediate parent"
    );
    assert_eq!(from_depth_two.stream_id, parent.id());
    assert!(
        from_depth_two.seq > 0,
        "a child event takes a shared-stream sequence number"
    );
    let CodingEvent::SubAgentSpawned { agent_id, .. } = &from_depth_two.event else {
        panic!("the event is a spawn: {from_depth_two:?}");
    };
    assert_eq!(agent_id, &grandchild);

    assert_eq!(
        from_depth_three.session_id, grandchild_session.id,
        "two hops later the event still names the session it happened in"
    );
    assert_eq!(
        from_depth_three.parent_session_id.as_deref(),
        Some(child.id.as_str()),
        "the parent it names is its immediate parent, not the root"
    );
    assert_eq!(from_depth_three.stream_id, parent.id());
    assert!(from_depth_three.seq > 0);
    let CodingEvent::SubAgentSpawned { agent_id, .. } = &from_depth_three.event else {
        panic!("the event is a spawn: {from_depth_three:?}");
    };
    assert_eq!(agent_id, &great_grandchild);

    assert_eq!(
        supervisor.open_sessions(),
        4,
        "four levels share the one budget the root created"
    );
    assert!(
        supervisor.status(&child_id).is_some(),
        "the child that carried it is still supervised"
    );

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

// --- What a tree owns ---

#[tokio::test]
async fn a_shutdown_joins_every_task_in_a_tree() {
    let (observer, children) = recording_observer();
    let (mut parent, _provider) = TestSession::new(answers("done"))
        .observe_children(observer)
        .build();
    let supervisor = parent
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();
    let child_id = supervisor
        .spawn(parent.id(), parent.root_session_id(), "delegate".to_owned())
        .expect("the spawn succeeds");
    let child = first_child(&children);
    let grandchild = child
        .supervisor
        .spawn(
            &child.id,
            parent.root_session_id(),
            "the leaf task".to_owned(),
        )
        .expect("a child may spawn a child of its own");
    assert!(
        Handle::current().metrics().num_alive_tasks() > 0,
        "three sessions are running"
    );

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");

    assert!(matches!(
        supervisor.status(&child_id),
        Some(SubagentStatus::Closed)
    ));
    assert!(matches!(
        child.supervisor.status(&grandchild),
        Some(SubagentStatus::Closed)
    ));
    assert_eq!(
        Handle::current().metrics().num_alive_tasks(),
        0,
        "a closed tree leaves no task running: no runner, monitor, cleanup or pump"
    );
}

#[tokio::test]
async fn a_session_with_no_factory_answers_a_spawn_as_a_tool_it_does_not_have() {
    let (mut parent, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "spawn_agent",
            "spawn_call",
            json!({ "task": "review the diff" }),
        )),
        ScriptedCall::response(text_response("I reviewed it myself")),
    ])
    .build();
    let mut events = parent.subscribe();

    let output = parent
        .prompt("delegate the review")
        .await
        .expect("a tool the session does not have is the model's mistake, not a failure");

    assert_eq!(output.as_deref(), Some("I reviewed it myself"));
    let published = settled(&mut parent, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::ToolCallCompleted {
                tool_name,
                is_error: true,
                error_kind: Some(ToolErrorKind::Unavailable),
                ..
            } if tool_name == "spawn_agent"
        )),
        "{published:?}"
    );
}

/// A sink that remembers where each event it recorded came from.
#[derive(Debug, Default)]
struct TreeSink {
    recorded: Mutex<Vec<CodingAgentEvent>>,
}

impl TreeSink {
    fn recorded(&self) -> Vec<CodingAgentEvent> {
        self.recorded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait::async_trait]
impl EventSink for TreeSink {
    async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
        self.recorded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event.clone());
        Ok(())
    }
}

#[tokio::test]
async fn a_childs_events_reach_the_parents_durable_stream() {
    // A child has no pump or sink of its own. The whole tree writes directly
    // through the root pipeline the application configured.
    let sink = Arc::new(TreeSink::default());
    let (client, _provider) = scripted_client(answers("child result"));
    let mut parent = builder(client)
        .event_sink(Arc::clone(&sink) as Arc<dyn EventSink>)
        .subagents(SubagentOptions::enabled())
        .build()
        .expect("the session builds");
    let supervisor = parent
        .subagent_supervisor()
        .expect("the session was given a factory")
        .clone();

    let agent_id = supervisor
        .spawn(parent.id(), parent.root_session_id(), "task".to_owned())
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&agent_id, &CancellationToken::new())
        .await
        .expect("the child answers");
    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");

    let recorded = sink.recorded();
    assert!(
        recorded.iter().any(|event| {
            event.session_id != parent.id()
                && event.parent_session_id.as_deref() == Some(parent.id())
                && matches!(event.event, CodingEvent::SessionEnded)
        }),
        "the durable stream contains the child's final event: {recorded:?}"
    );
    assert!(
        recorded.iter().all(|event| event.stream_id == parent.id()),
        "every event names the root stream: {recorded:?}"
    );
    let numbering: Vec<u64> = recorded.iter().map(|event| event.seq).collect();
    assert_eq!(
        numbering,
        (1..=u64::try_from(recorded.len()).expect("a small stream")).collect::<Vec<_>>(),
        "one numbering covers the tree, in the order the pump recorded it"
    );
}

// --- What a child is given ---

/// An application tool is root-only unless marked, a marked tool that needs a
/// person is withheld all the same, and the parent's middleware binds the
/// child: the child can never be shown more than its parent was.
#[tokio::test]
async fn a_child_inherits_only_marked_tools_and_its_parents_middleware() {
    use pebble_agent::ToolDescriptor;

    use crate::tool::{PermissionMiddleware, ToolPermission, ToolPermissionPolicy};

    /// Denies the shell to the whole tree.
    struct NoShell;

    impl ToolPermissionPolicy for NoShell {
        fn permission(&self, tool: &ToolDescriptor) -> ToolPermission {
            if tool.id().as_str() == "shell" {
                ToolPermission::Deny {
                    reason: "shell is disabled".to_owned(),
                }
            } else {
                ToolPermission::Allow
            }
        }
    }

    fn application_tool(name: &str) -> RegisteredTool {
        RegisteredTool::function(
            name,
            "an application tool",
            json!({"type": "object"}),
            |_context, _arguments| async { Ok("ok".to_owned()) },
        )
    }

    let child_tools: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&child_tools);
    let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
        let mut names: Vec<String> = child
            .effective_tools()
            .into_iter()
            .map(|tool| tool.definition.name)
            .collect();
        names.sort();
        recorder
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(names);
    });

    // Registered on the builder as an application would register them, not
    // through the test profile, which would make them the harness's own.
    let (client, provider) = scripted_client(answers("done"));
    let mut parent = CodingRuntime::builder(client)
        .model("test/model")
        .environment(Arc::new(MockEnvironment::linux()))
        .tools([
            application_tool("audit"),
            application_tool("lint").allow_in_subagents(),
            application_tool("ask_ops")
                .allow_in_subagents()
                .requires_human_input(),
        ])
        .tool_middleware(Arc::new(PermissionMiddleware::new(Arc::new(NoShell))))
        .observe_children(observer)
        .build()
        .expect("the parent builds");
    parent.initialize().await.expect("initialization succeeds");
    let supervisor = parent
        .subagent_supervisor()
        .expect("the parent can spawn")
        .clone();

    let agent_id = supervisor
        .spawn(parent.id(), parent.root_session_id(), "work".to_owned())
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&agent_id, &CancellationToken::new())
        .await
        .expect("the child answers");

    let parent_tools: Vec<String> = parent
        .effective_tools()
        .into_iter()
        .map(|tool| tool.definition.name)
        .collect();
    for name in ["audit", "lint", "ask_ops"] {
        assert!(parent_tools.contains(&name.to_owned()), "{parent_tools:?}");
    }

    let child = child_tools
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .first()
        .cloned()
        .expect("one child was built");
    assert!(child.contains(&"lint".to_owned()), "{child:?}");
    assert!(
        !child.contains(&"audit".to_owned()),
        "an unmarked application tool is root-only: {child:?}"
    );
    assert!(
        !child.contains(&"ask_ops".to_owned()),
        "a tool that needs a person never reaches a child: {child:?}"
    );
    assert!(child.contains(&"shell".to_owned()), "{child:?}");
    assert!(
        child.contains(&"read_file".to_owned()),
        "built-in tools are inherited: {child:?}"
    );
    let requests = provider.requests();
    let child_request = requests.first().expect("the child asked its model");
    assert!(
        child_request
            .tools()
            .iter()
            .all(|tool| tool.name != "shell"),
        "the inherited middleware filters the child request: {:?}",
        child_request.tools()
    );

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

// --- The reason a wait on a child ends with ---

/// A parent whose answer arrives while a background child is still running, so
/// its prompt parks at the boundary waiting for the child's notification.
///
/// The child takes the first script entry, whose open never completes, and is
/// answered only when something cancels it; the parent takes the second. Both
/// are built with the default options, because a child inherits its parent's
/// options and a budget on the parent would be a budget on the child too.
/// Answers with the parent and the provider both read from.
async fn parent_parked_on_a_background_child() -> (CodingRuntime, Arc<ScriptedProvider>) {
    let (parent, provider) = TestSession::new(vec![
        ScriptedCall::PendingOpen,
        ScriptedCall::response(text_response("delegated")),
    ])
    .with_subagents()
    .build();
    let supervisor = parent
        .subagent_supervisor()
        .expect("the test session was given a factory")
        .clone();
    supervisor
        .spawn_with_parent_notification(
            parent.id(),
            parent.root_session_id(),
            "take your time".to_owned(),
            "Slow task".to_owned(),
        )
        .expect("the spawn succeeds");
    // The child's model call has begun, so the parent's prompt is next in line
    // for the script and the child is still running when it gets there.
    provider.wait_for_call().await;
    (parent, provider)
}

#[tokio::test]
async fn a_budget_that_runs_out_while_the_parent_waits_on_a_child_is_the_reason_reported() {
    // What the wall-clock timer does when it fires, done by hand once the
    // prompt is parked, so the timing is the test's rather than the clock's.
    // The timer's own path to the reason is covered with the interrupts.
    let (mut parent, provider) = parent_parked_on_a_background_child().await;
    let reason = parent.interrupt_reason_handle();
    let budget = CancellationToken::new();
    let watchdog = budget.clone();
    let mut events = parent.subscribe();
    let timer = tokio::spawn(async move {
        // The answer is committed by the time it is published, so the prompt is
        // at, or on its way to, the wait on the child.
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::AssistantMessage { .. })
        })
        .await;
        reason.record(InterruptReason::WallClockTimeout);
        watchdog.cancel();
    });

    let error = timeout(
        Duration::from_secs(1),
        parent.prompt_with_cancellation("Delegate this", Some(&budget)),
    )
    .await
    .expect("the budget ends the prompt")
    .expect_err("the prompt ran out of time");
    timer.await.expect("the timer finishes");

    assert!(
        matches!(error, Error::Interrupted(InterruptReason::WallClockTimeout)),
        "the budget, not a plain cancellation, is the reason: {error:?}"
    );
    assert_eq!(
        provider.call_count(),
        2,
        "the child's call and the parent's"
    );
    assert!(
        matches!(
            parent.history().turns().last(),
            Some(Message::Assistant { content, .. }) if content == "delegated"
        ),
        "the prompt was parked on the child after its answer: {:?}",
        parent.history().turns()
    );
    // Running out of time is the prompt's failure, not the session's.
    assert_eq!(parent.state(), CodingAgentState::Idle);

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

#[tokio::test]
async fn a_session_cancelled_while_the_parent_waits_on_a_child_closes() {
    let (mut parent, _provider) = parent_parked_on_a_background_child().await;
    let reason = parent.interrupt_reason_handle();
    let terminal = parent.cancel_token();
    let mut events = parent.subscribe();
    let canceller = tokio::spawn(async move {
        // The answer is committed by the time it is published, so the prompt is
        // at, or on its way to, the wait on the child.
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::AssistantMessage { .. })
        })
        .await;
        reason.record(InterruptReason::Cancelled);
        terminal.cancel();
    });

    let error = timeout(Duration::from_secs(1), parent.prompt("Delegate this"))
        .await
        .expect("the cancellation ends the prompt")
        .expect_err("the prompt was cancelled");
    canceller.await.expect("the canceller finishes");

    assert!(
        matches!(error, Error::Interrupted(InterruptReason::Cancelled)),
        "{error:?}"
    );
    assert_eq!(
        parent.state(),
        CodingAgentState::Closed,
        "the session's own cancellation closes it, wherever the prompt was"
    );
}
