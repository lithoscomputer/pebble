//! Invariant 3: an interrupt is announced exactly once.
//!
//! Every interrupt gesture raises a generation, and the loop settles the count
//! as it unwinds — one
//! [`RoundInterrupted`](crate::events::CodingEvent::RoundInterrupted) per
//! gesture, before the steer that replaces the abandoned round is delivered.
//! These tests interrupt the loop everywhere it can be interrupted: before it
//! starts, while it waits on the model, and while a tool is running.
//!
//! What the session owns is here too, because a prompt that ends — however it
//! ends — has to leave nothing behind: the wall-clock timer stops, the event
//! pump is joined, and a session restored from its record carries on where the
//! last one stopped.

use std::future::pending;
use std::time::Duration;

use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::{Message as LlmMessage, Role, ToolCall, ToolDefinition};
use serde_json::json;
use tokio::runtime::Handle;
use tokio::sync::{Notify, broadcast};
use tokio::task::yield_now;
use tokio::time::{sleep, timeout};

use super::super::testing::{builder, event_name, wait_for_event};
use super::*;
use crate::event::{EventSink, EventSinkError};
use crate::task_reminder::TASK_REMINDER_TEXT;
use crate::test_support::{
    ScriptedCompletion, ScriptedProvider, message_text, multi_tool_call_response, scripted_client,
    text_delta_events, tool_call_events,
};
use crate::types::LlmOutputKind;

/// How long a test waits for a prompt that another task has to unblock.
const PATIENCE: Duration = Duration::from_secs(1);

// --- Steering ---

#[tokio::test]
async fn a_steer_lands_as_its_own_turn() {
    let (mut session, _provider) = TestSession::answering(answers("OK"));
    session.steer("Focus on the task");

    session
        .prompt("Do something")
        .await
        .expect("the prompt succeeds");

    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 3, "input, steer, answer");
    assert!(matches!(&turns[0], Message::User { .. }));
    assert!(
        matches!(&turns[1], Message::Steering { content, .. } if content == "Focus on the task")
    );
    assert!(matches!(&turns[2], Message::Assistant { .. }));
}

#[tokio::test]
async fn a_steer_is_announced_with_its_text() {
    let (mut session, _provider) = TestSession::answering(answers("OK"));
    let mut events = session.subscribe();
    session.steer("hi there");

    session
        .prompt("Do something")
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let steered = published.iter().find_map(|event| match event {
        CodingEvent::SteeringInjected { text, .. } => Some(text.clone()),
        _ => None,
    });
    assert_eq!(steered.as_deref(), Some("hi there"));
}

#[tokio::test]
async fn an_interrupt_with_nothing_running_does_nothing() {
    // There is no round to abandon, so nothing is counted and nothing parks:
    // the next prompt runs as if the gesture had never been made.
    let (mut session, _provider) = TestSession::answering(answers("OK"));
    let mut events = session.subscribe();
    let handle = session.control_handle();

    assert!(!handle.interrupt(), "there is no round to interrupt");
    session.prompt("start").await.expect("the prompt succeeds");

    assert_eq!(session.history().turns().len(), 2, "input and answer");
    assert!(!handle.is_parked());
    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::RoundInterrupted { .. }
        )),
        0,
        "no gesture, no announcement"
    );
}

#[tokio::test]
async fn an_interrupt_that_lands_while_the_session_is_parked_is_announced_too() {
    // The hardest of the exactly-once cases: the second gesture arrives after
    // the first has settled and the session is already waiting for a steer.
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::PendingOpen,
        ScriptedCall::response(text_response("resumed")),
    ]);
    let control = session.control_handle();
    let mut controller_events = session.subscribe();
    let mut recorded = session.subscribe();
    let waiting = Arc::clone(&provider);

    let controller = tokio::spawn(async move {
        waiting.wait_for_call().await;
        assert!(control.interrupt());
        wait_for_event(&mut controller_events, |event| {
            matches!(event, CodingEvent::RoundInterrupted { generation: 1 })
        })
        .await;
        assert!(control.is_parked());
        assert!(control.interrupt(), "a parked prompt is still running");
        control.steer("carry on", None);
    });

    timeout(PATIENCE, session.prompt("start"))
        .await
        .expect("the steer wakes the parked session")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    let published = settled(&mut session, &mut recorded).await;
    let generations: Vec<u64> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::RoundInterrupted { generation } => Some(*generation),
            _ => None,
        })
        .collect();
    assert_eq!(
        generations,
        [1, 2],
        "one announcement per gesture, in order"
    );
    assert_eq!(
        provider.call_count(),
        2,
        "the abandoned call and the one that resumed"
    );
    assert!(matches!(
        &session.history().turns()[1],
        Message::Steering { content, .. } if content == "carry on"
    ));
}

#[tokio::test]
async fn a_gesture_whose_round_cancel_was_lost_is_still_announced() {
    // What a race between two gestures can leave behind: the generation is
    // raised, but the cancel landed on the token the loop was already
    // replacing, so the round that follows ends normally. The announcement is
    // owed all the same, and waiting for the next interrupt to pay it would
    // break the exactly-once promise.
    let (mut session, _provider) = TestSession::answering(answers("OK"));
    let mut events = session.subscribe();
    session.control_handle().record_interrupt_without_a_round();

    session.prompt("start").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let generations: Vec<u64> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::RoundInterrupted { generation } => Some(*generation),
            _ => None,
        })
        .collect();
    assert_eq!(generations, [1], "the raised generation is announced once");
}

#[tokio::test]
async fn an_interrupt_settles_before_the_steer_that_replaces_it() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::PendingOpen,
        ScriptedCall::response(text_response("OK")),
    ]);
    let mut events = session.subscribe();
    let handle = session.control_handle();
    let steerer = handle.clone();
    let controller = tokio::spawn(async move {
        provider.wait_for_call().await;
        steerer.interrupt_then_steer("stop now", None);
    });

    timeout(PATIENCE, session.prompt("start"))
        .await
        .expect("the steer unblocks the hanging call")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    assert!(!handle.is_parked());
    let published = settled(&mut session, &mut events).await;
    let settled_at = position(&published, |event| {
        matches!(event, CodingEvent::RoundInterrupted { generation: 1 })
    })
    .expect("the interrupt settled");
    let steered_at = position(
        &published,
        |event| matches!(event, CodingEvent::SteeringInjected { text, .. } if text == "stop now"),
    )
    .expect("the steer was delivered");
    assert!(settled_at < steered_at);
}

#[tokio::test]
async fn an_interrupt_while_the_model_is_thinking_settles_once() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::PendingOpen,
        ScriptedCall::response(text_response("resumed")),
    ]);
    let control = session.control_handle();
    let mut controller_events = session.subscribe();
    let mut recorded = session.subscribe();
    let waiting = Arc::clone(&provider);
    let controller = tokio::spawn(async move {
        waiting.wait_for_call().await;
        control.interrupt();
        wait_for_event(&mut controller_events, |event| {
            matches!(event, CodingEvent::RoundInterrupted { generation: 1 })
        })
        .await;
        assert!(control.is_parked());
        control.steer("resume inference", None);
        control
    });

    timeout(PATIENCE, session.prompt("start"))
        .await
        .expect("the interrupt unblocks the hanging call")
        .expect("the prompt succeeds");
    let control = controller.await.expect("the controller finishes");

    assert_eq!(provider.call_count(), 2, "the round was asked again");
    assert!(!control.is_parked());
    let published = settled(&mut session, &mut recorded).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::RoundInterrupted { .. }
        )),
        1
    );
    let settled_at = position(&published, |event| {
        matches!(event, CodingEvent::RoundInterrupted { .. })
    })
    .expect("the interrupt settled");
    let steered_at = position(&published, |event| {
        matches!(event, CodingEvent::SteeringInjected { .. })
    })
    .expect("the steer was delivered");
    assert!(settled_at < steered_at);
}

#[tokio::test]
async fn an_interrupted_round_leaves_no_task_reminder_behind() {
    // Ten answered inputs with no task tool used, then the round that is
    // interrupted, then the round the steer resumes.
    let mut calls: Vec<ScriptedCall> = (0..10)
        .map(|_| ScriptedCall::response(text_response("done")))
        .collect();
    calls.push(ScriptedCall::EventsThenPending(tool_call_events(
        &ToolCall::function(
            "call_1",
            "TaskUpdate",
            json!({"taskId": "1", "status": "completed"}),
        ),
    )));
    calls.push(ScriptedCall::response(text_response("resumed")));
    let (mut session, provider) = TestSession::new(calls)
        .tools([noop_tool("TaskCreate"), noop_tool("TaskUpdate")])
        .build();
    for index in 0..10 {
        session
            .prompt(&format!("turn {index}"))
            .await
            .expect("the prompt succeeds");
    }

    let control = session.control_handle();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::LlmFirstOutput {
                kind: LlmOutputKind::ToolCall,
            })
        })
        .await;
        control.interrupt();
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::RoundInterrupted { generation: 1 })
        })
        .await;
        control.steer("wrap up now", None);
    });

    timeout(PATIENCE, session.prompt("continue"))
        .await
        .expect("the interrupted session resumes after steering")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    let requests = provider.requests();
    let interrupted = requests
        .get(10)
        .expect("the interrupted round was requested");
    let staged = interrupted
        .messages()
        .last()
        .expect("the request is not empty");
    assert_eq!(staged.role(), Role::System);
    assert_eq!(message_text(staged), TASK_REMINDER_TEXT);

    let resumed = requests.get(11).expect("the steer asked again");
    let [.., steering, reminder] = resumed.messages() else {
        panic!("the resumed request should end with the steer and a restaged reminder");
    };
    assert_eq!(steering.role(), Role::User);
    assert_eq!(message_text(steering), "wrap up now");
    assert_eq!(reminder.role(), Role::System);
    assert_eq!(message_text(reminder), TASK_REMINDER_TEXT);

    let history = session.history();
    let [
        ..,
        Message::System {
            content: committed, ..
        },
        Message::Assistant { content, .. },
    ] = history.turns()
    else {
        panic!("the reminder commits with the assistant turn that read it");
    };
    assert_eq!(committed, TASK_REMINDER_TEXT);
    assert_eq!(content, "resumed");
}

#[tokio::test]
async fn an_interrupt_mid_stream_withdraws_what_the_turn_showed_and_commits_nothing() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::EventsThenPending(text_delta_events("half an answer")),
        ScriptedCall::response(text_response("the answer after the steer")),
    ]);
    let control = session.control_handle();
    let mut controller_events = session.subscribe();
    let mut recorded = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(
            &mut controller_events,
            |event| matches!(event, CodingEvent::TextDelta { delta } if delta == "half an answer"),
        )
        .await;
        control.interrupt_then_steer("say it differently", None);
    });

    let answer = timeout(PATIENCE, session.prompt("say something"))
        .await
        .expect("the interrupt unblocks the hanging stream")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    assert_eq!(answer.as_deref(), Some("the answer after the steer"));
    assert_eq!(provider.call_count(), 2);
    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 3, "input, steer, answer");
    assert!(matches!(&turns[1], Message::Steering { .. }));
    assert!(
        matches!(&turns[2], Message::Assistant { content, .. } if content == "the answer after the steer"),
        "the abandoned turn is not committed"
    );

    let published = settled(&mut session, &mut recorded).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::AssistantOutputReplace { .. }
        )),
        1,
        "the abandoned turn's output is withdrawn once"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::RoundInterrupted { .. }
        )),
        1
    );
    let withdrawn = position(&published, |event| {
        matches!(event, CodingEvent::AssistantOutputReplace { .. })
    })
    .expect("the withdrawal was published");
    let second_delta = published
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event, CodingEvent::TextDelta { .. }))
        .nth(1)
        .map(|(index, _)| index)
        .expect("the replayed turn published its own text");
    assert!(
        withdrawn < second_delta,
        "nothing is shown twice: the withdrawal precedes the new output"
    );
}

#[tokio::test]
async fn ending_the_prompt_while_a_replay_waits_ends_it_as_a_cancellation() {
    // The wait between replays is one of the places a prompt can be ended, and
    // a prompt ended anywhere ends the same way: interrupted, and closed.
    let (mut session, _provider) = TestSession::new(vec![ScriptedCall::fails_after(
        "partial",
        ScriptedFailure::retryable(LlmErrorKind::Network, "connection reset"),
    )])
    .options(CodingAgentOptions {
        turn_replay: RetryPolicy::exponential()
            .max_attempts(4)
            .initial_delay(Duration::from_secs(30)),
        ..CodingAgentOptions::default()
    })
    .build();
    let cancel = session.cancel_token();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::LlmRetry { .. })
        })
        .await;
        cancel.cancel();
    });

    let error = timeout(PATIENCE, session.prompt("Hello"))
        .await
        .expect("ending the prompt does not wait out the backoff")
        .expect_err("the prompt was ended");
    controller.await.expect("the controller finishes");

    assert!(
        matches!(error, Error::Interrupted(InterruptReason::Cancelled)),
        "a cancelled prompt reports a cancellation, not a stream failure: {error:?}"
    );
    assert_eq!(session.state(), CodingAgentState::Closed);
}

#[tokio::test]
async fn an_interrupted_parallel_round_answers_every_call_it_made() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(multi_tool_call_response(vec![
            ("block", "call_1", json!({})),
            ("block", "call_2", json!({})),
            ("block", "call_3", json!({})),
        ])),
        ScriptedCall::response(text_response("done")),
    ])
    .tools([blocking_tool("block")])
    .build();
    let control = session.control_handle();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::ToolCallStarted { .. })
        })
        .await;
        control.interrupt_then_steer("stop all of that", None);
    });

    timeout(PATIENCE, session.prompt("run three tools"))
        .await
        .expect("the interrupt unblocks every call")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    let results = tool_results(&session, 2);
    assert_eq!(
        results
            .iter()
            .map(|result| result.tool_call_id.as_str())
            .collect::<Vec<_>>(),
        ["call_1", "call_2", "call_3"],
        "every call the model made is answered, in call order"
    );
    assert!(
        results.iter().all(|result| result.is_error),
        "an interrupted call answers with its cancellation"
    );
    assert!(
        results
            .iter()
            .all(|result| result_text(result) == "Cancelled"),
        "a call the round cancelled says so in the words the model reads: {:?}",
        results.iter().map(result_text).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_tool_that_ignores_its_cancellation_holds_the_round_open() {
    // Tool cancellation is cooperative, and the session waits for the call it
    // made rather than dropping it, which is what keeps every call paired with
    // a result. The cost is pinned here: a tool that never watches its token
    // holds the round, and the prompt ending it, open until it returns.
    let stubborn = RegisteredTool::new(
        ToolDefinition::function("stubborn", "Never answers", json!({"type": "object"})),
        Arc::new(|_arguments, _context| Box::pin(pending())),
    )
    .with_source(ToolSource::Native);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("stubborn", "call_1", json!({}))),
        ScriptedCall::response(text_response("done")),
    ])
    .tools([stubborn])
    .build();
    let cancel = session.cancel_token();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        timeout(
            PATIENCE,
            wait_for_event(&mut events, |event| {
                matches!(event, CodingEvent::ToolCallStarted { .. })
            }),
        )
        .await
        .expect("the tool call starts");
        cancel.cancel();
    });

    let outcome = timeout(Duration::from_millis(200), session.prompt("use the tool")).await;
    controller.await.expect("the controller finishes");

    assert!(
        outcome.is_err(),
        "the cancelled round is still waiting for the call it made"
    );
}

/// A steering lease parks natural completion, so a steer that arrives after
/// the first answer still reaches the session and drives another round.
///
/// This is the close-door race the removed completion coordinator used to
/// settle: an external steering source holds a lease across the prompt, so a
/// plain answer parks instead of finishing. The source then steers and drops
/// its lease, and the queued steer sends the loop around once more.
#[tokio::test]
async fn a_steering_lease_lets_a_late_steer_drive_another_round() {
    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(text_response("First reply")),
        ScriptedCall::response(text_response("Second reply, after steer")),
    ]);
    let handle = session.control_handle();
    let mut events = session.subscribe();
    // Held across the whole prompt: the first answer parks rather than
    // completing while this is alive.
    let lease = session.steering_lease();

    let steering = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::AssistantMessage { text, .. } if text == "First reply")
        })
        .await;
        // Queue the steer, then drop the lease. The queued item makes the
        // parked prompt run another round rather than complete.
        handle.steer("after-completion steer", None);
        drop(lease);
    });

    session.prompt("hi").await.expect("the prompt succeeds");
    steering.await.expect("the steering task finishes");

    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 4, "input, answer, steer, answer");
    assert!(matches!(&turns[1], Message::Assistant { content, .. } if content == "First reply"));
    assert!(
        matches!(&turns[2], Message::Steering { content, .. } if content == "after-completion steer")
    );
    assert!(
        matches!(&turns[3], Message::Assistant { content, .. } if content == "Second reply, after steer")
    );
}

/// Dropping the final steering lease with nothing queued wakes a parked prompt
/// and lets it complete.
#[tokio::test]
async fn dropping_the_last_lease_wakes_a_parked_prompt() {
    let (mut session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(text_response("only reply"))]);
    let mut events = session.subscribe();
    let lease = session.steering_lease();

    let releaser = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::AssistantMessage { text, .. } if text == "only reply")
        })
        .await;
        // Nothing queued: the prompt is parked purely on the lease, and
        // dropping it must let the prompt finish.
        drop(lease);
    });

    let answer = timeout(PATIENCE, session.prompt("hi"))
        .await
        .expect("dropping the lease lets the parked prompt complete")
        .expect("the prompt succeeds");
    releaser.await.expect("the releaser finishes");

    assert_eq!(answer.as_deref(), Some("only reply"));
    assert_eq!(session.history().turns().len(), 2, "input and answer");
}

// --- The wall clock ---

#[tokio::test]
async fn a_prompt_that_outlasts_its_budget_ends_with_the_budget_as_its_reason() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("slow_tool", "call_1", json!({}))),
        ScriptedCall::response(text_response("Should not reach this")),
    ])
    .tools([blocking_tool("slow_tool")])
    .options(CodingAgentOptions {
        wall_clock_timeout: Some(Duration::from_millis(10)),
        enable_loop_detection: false,
        ..CodingAgentOptions::default()
    })
    .build();

    let error = timeout(PATIENCE, session.prompt("Do something slow"))
        .await
        .expect("the budget ends the prompt")
        .expect_err("the prompt ran out of time");

    assert!(
        matches!(error, Error::Interrupted(InterruptReason::WallClockTimeout)),
        "{error:?}"
    );
    // Running out of time is the prompt's failure, not the session's: the
    // session is idle again, and the next prompt gets a fresh budget.
    assert_eq!(session.state(), CodingAgentState::Idle);
    assert!(
        matches!(
            session.history().turns().last(),
            Some(Message::ToolResults { results, .. }) if results.len() == 1
        ),
        "the interrupted call still has its result: {:?}",
        session.history().turns()
    );
    let answer = timeout(PATIENCE, session.prompt("Try again"))
        .await
        .expect("the next prompt runs")
        .expect("the next prompt succeeds");
    assert_eq!(answer.as_deref(), Some("Should not reach this"));
}

/// A session whose first turn calls `count` and then compacts, with the
/// summarizing call held behind the returned gate.
///
/// The tool-calling turn reports no usage of its own, so the checkpoint after
/// it measures the conversation the session holds, and compacts it.
fn compacting_after_a_tool_call(
    options: CodingAgentOptions,
) -> (
    CodingRuntime,
    Arc<ScriptedProvider>,
    Arc<Counter>,
    Arc<Notify>,
) {
    let runs = Arc::new(Counter::default());
    let counter = Arc::clone(&runs);
    let counted = RegisteredTool::new(
        ToolDefinition::function("count", "Records that it ran", json!({"type": "object"})),
        Arc::new(move |_arguments, _context| {
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                counter.bump();
                Ok("ran".to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native);
    let (summary, gate) = ScriptedCompletion::gated(text_response("The summary so far."));
    let (session, provider) = TestSession::new(vec![
        ScriptedCall::response(with_usage(
            tool_call_response("count", "call_1", json!({})),
            TokenCounts::default(),
        )),
        ScriptedCall::response(text_response("done")),
    ])
    .model("test/small")
    .tools([counted])
    .completing(vec![summary])
    .options(CodingAgentOptions {
        enable_context_compaction: true,
        compaction_preserve_turns: 1,
        enable_loop_detection: false,
        ..options
    })
    .build();
    (session, provider, runs, gate)
}

/// What a prompt ended during compaction leaves behind: the tool never ran,
/// its call is answered `Cancelled`, and the conversation is paired.
fn assert_the_call_was_answered_as_cancelled(session: &CodingRuntime, runs: &Counter) {
    assert_eq!(session.state(), CodingAgentState::Idle);
    assert_eq!(
        runs.count(),
        0,
        "a tool the caller stopped before never runs"
    );
    let turns = session.history().turns().to_vec();
    let [
        ..,
        Message::Assistant { tool_calls, .. },
        Message::ToolResults { results, .. },
    ] = turns.as_slice()
    else {
        panic!("history ends with the tool call and its result: {turns:?}");
    };
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool_call_id, "call_1");
    assert!(results[0].is_error);
    assert_eq!(result_text(&results[0]), "Cancelled");
    // What the next request would carry: the call and its result, adjacent.
    let roles = session
        .history()
        .to_llm_messages()
        .iter()
        .map(LlmMessage::role)
        .collect::<Vec<_>>();
    assert!(
        roles.ends_with(&[Role::Assistant, Role::Tool]),
        "the conversation the prompt left is paired: {roles:?}"
    );
}

/// The small window compacts again before the next turn, so what pins the
/// repair is that the next prompt runs at all.
async fn assert_the_next_prompt_runs(session: &mut CodingRuntime, provider: &ScriptedProvider) {
    let answer = timeout(PATIENCE, session.prompt("again"))
        .await
        .expect("the next prompt runs")
        .expect("the next prompt succeeds");

    assert_eq!(answer.as_deref(), Some("done"));
    assert_eq!(provider.call_count(), 2, "one call per prompt");
}

/// A cancellation that lands while the assistant turn is being compacted still
/// answers the tool calls that turn made, so the record the interrupted prompt
/// leaves behind is one the next prompt can send.
#[tokio::test]
async fn a_cancellation_during_compaction_answers_the_tool_call_before_ending_the_prompt() {
    let (mut session, provider, runs, gate) =
        compacting_after_a_tool_call(CodingAgentOptions::default());
    let cancel = CancellationToken::new();
    let mut events = session.subscribe();
    let controller = {
        let cancel = cancel.clone();
        tokio::spawn(async move {
            wait_for_event(&mut events, |event| {
                matches!(event, CodingEvent::CompactionStarted { .. })
            })
            .await;
            // The summarizing call is in flight: end the prompt, then let the
            // summary arrive.
            cancel.cancel();
            // The caller's token reaches the prompt through a linking task,
            // which has to run before the summary wakes the prompt.
            yield_now().await;
            gate.notify_one();
        })
    };

    let error = timeout(
        PATIENCE,
        session.prompt_with_cancellation(&"x".repeat(400), Some(&cancel)),
    )
    .await
    .expect("the prompt ends")
    .expect_err("the prompt was cancelled");
    controller.await.expect("the controller finishes");

    assert!(
        matches!(error, Error::Interrupted(InterruptReason::Cancelled)),
        "{error:?}"
    );
    assert_the_call_was_answered_as_cancelled(&session, &runs);
    assert_the_next_prompt_runs(&mut session, &provider).await;
}

/// The same window, reached by the wall clock: the budget runs out while the
/// summarizing call is in flight.
///
/// The clock is paused, so it advances only when nothing can run: that is
/// while the prompt waits on the gate, never during the turn that reaches
/// it. The budget therefore runs out inside the window, however slow the
/// machine, and the second prompt's fresh budget is never eaten by a wait.
#[tokio::test(start_paused = true)]
async fn a_budget_that_runs_out_during_compaction_answers_the_tool_call_before_ending_the_prompt() {
    let (mut session, provider, runs, gate) = compacting_after_a_tool_call(CodingAgentOptions {
        wall_clock_timeout: Some(Duration::from_millis(20)),
        ..CodingAgentOptions::default()
    });
    let reason = session.interrupt_reason_handle();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::CompactionStarted { .. })
        })
        .await;
        // The summary is held until the timer has fired, so the budget runs
        // out while the summarizing call is in flight.
        while reason.reason().is_none() {
            sleep(Duration::from_millis(1)).await;
        }
        // The timer records its reason, then cancels the prompt; let its task
        // get there before the summary wakes the prompt.
        yield_now().await;
        gate.notify_one();
    });

    let error = timeout(PATIENCE, session.prompt(&"x".repeat(400)))
        .await
        .expect("the budget ends the prompt")
        .expect_err("the prompt ran out of time");
    // A controller still waiting on an event it missed fails the test rather
    // than hanging it.
    timeout(PATIENCE, controller)
        .await
        .expect("the controller saw the summarizing call start")
        .expect("the controller finishes");

    assert!(
        matches!(error, Error::Interrupted(InterruptReason::WallClockTimeout)),
        "{error:?}"
    );
    assert_the_call_was_answered_as_cancelled(&session, &runs);
    assert_the_next_prompt_runs(&mut session, &provider).await;
}

#[tokio::test]
async fn the_reason_an_outside_task_recorded_first_is_the_one_reported() {
    // What the handle is for: a watchdog names why it is stopping the prompt, and
    // the cancellation that follows does not overwrite it.
    let (mut session, _provider) = TestSession::answering(answers("never reached"));
    let reason = session.interrupt_reason_handle();

    assert!(reason.record(InterruptReason::WallClockTimeout));
    assert!(
        !reason.record(InterruptReason::Cancelled),
        "a second writer does not replace the reason already recorded"
    );
    session.interrupt();

    let error = session
        .prompt("Do something")
        .await
        .expect_err("the prompt was cancelled");

    assert!(
        matches!(error, Error::Interrupted(InterruptReason::WallClockTimeout)),
        "{error:?}"
    );
    assert_eq!(reason.reason(), Some(InterruptReason::WallClockTimeout));
}

#[tokio::test]
async fn a_prompt_inside_its_budget_is_untouched() {
    let (mut session, _provider) = TestSession::new(answers("Fast response"))
        .options(CodingAgentOptions {
            wall_clock_timeout: Some(Duration::from_secs(10)),
            ..CodingAgentOptions::default()
        })
        .build();

    session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(session.state(), CodingAgentState::Idle);
    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 2);
    assert!(matches!(&turns[1], Message::Assistant { content, .. } if content == "Fast response"));
}

#[tokio::test]
async fn a_finished_prompt_leaves_no_timer_behind() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(text_response("first")),
        ScriptedCall::response(text_response("second")),
    ])
    .options(CodingAgentOptions {
        wall_clock_timeout: Some(Duration::from_millis(20)),
        ..CodingAgentOptions::default()
    })
    .build();

    session
        .prompt("one")
        .await
        .expect("the first prompt succeeds");
    // Well past the budget the finished prompt was given: a timer left running
    // would cancel the session here.
    sleep(Duration::from_millis(60)).await;

    assert!(!session.cancel_token().is_cancelled());
    session
        .prompt("two")
        .await
        .expect("the next prompt is unaffected by the last prompt's budget");
}

// --- What the session owns ---

#[tokio::test]
async fn a_shutdown_joins_everything_the_session_owns() {
    let (mut session, _provider) = TestSession::new(answers("done"))
        .options(CodingAgentOptions {
            wall_clock_timeout: Some(Duration::from_secs(10)),
            ..CodingAgentOptions::default()
        })
        .build();

    session
        .prompt("do a thing")
        .await
        .expect("the prompt succeeds");
    assert!(
        Handle::current().metrics().num_alive_tasks() > 0,
        "the event pump is still running"
    );

    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");

    assert!(session.pump.is_none(), "the pump was joined, not abandoned");
    assert!(
        session.emitter.is_closed(),
        "the pipeline is closed to anything emitted afterwards"
    );
    assert_eq!(
        Handle::current().metrics().num_alive_tasks(),
        0,
        "a closed session leaves no task running"
    );
}

/// A sink that remembers what it was given, in the order it was given it.
#[derive(Debug, Default)]
struct RecordingSink {
    recorded: Mutex<Vec<(u64, String)>>,
}

impl RecordingSink {
    fn recorded(&self) -> Vec<(u64, String)> {
        self.recorded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait::async_trait]
impl EventSink for RecordingSink {
    async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
        self.recorded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((event.seq, event_name(&event.event).to_owned()));
        Ok(())
    }
}

#[tokio::test]
async fn the_sink_sees_every_event_in_order_and_so_does_a_subscriber() {
    let sink = Arc::new(RecordingSink::default());
    let (client, _provider) = scripted_client(answers("done"));
    let mut session = builder(client)
        .event_sink(Arc::clone(&sink) as Arc<dyn EventSink>)
        .build()
        .expect("the session builds");
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");
    session
        .prompt("do a thing")
        .await
        .expect("the prompt succeeds");
    let published = settled(&mut session, &mut events).await;

    let observed: Vec<&'static str> = published.iter().map(event_name).collect();
    let recorded = sink.recorded();
    assert_eq!(
        recorded
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>(),
        observed,
        "the sink and the live stream carry the same events in the same order"
    );
    assert_eq!(
        recorded.iter().map(|(seq, _)| *seq).collect::<Vec<_>>(),
        (1..=recorded.len() as u64).collect::<Vec<_>>(),
        "sequence numbers are assigned once, in order, with no gaps"
    );
    assert_eq!(observed.first().copied(), Some("started"));
    assert_eq!(observed.last().copied(), Some("ended"));
}

#[tokio::test]
async fn a_resumed_session_carries_on_the_conversation_and_the_numbering() {
    let (mut session, _provider) = TestSession::answering(answers("first answer"));
    session
        .prompt("first input")
        .await
        .expect("the prompt succeeds");
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
    let record = session.to_record();
    let last_seq = record.last_event_seq;
    assert!(last_seq > 0);

    let (client, provider) = scripted_client(answers("second answer"));
    let mut resumed =
        CodingRuntime::from_record(record.clone(), &ResumeMode::RecordedModel, builder(client))
            .expect("the record restores");
    let mut events = resumed.subscribe();

    let answer = resumed
        .prompt("second input")
        .await
        .expect("the resumed prompt succeeds");

    assert_eq!(answer.as_deref(), Some("second answer"));
    assert_eq!(resumed.id(), session.id());
    assert_eq!(
        resumed.history().turns().len(),
        4,
        "the restored conversation and the new exchange"
    );
    assert_eq!(provider.call_count(), 1);

    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
    let seqs = sequence_numbers(&mut events);
    assert_eq!(
        seqs.first().copied(),
        Some(last_seq + 1),
        "the resumed session numbers its events after the record's last"
    );
    assert!(
        seqs.windows(2).all(|pair| pair[1] == pair[0] + 1),
        "numbering stays contiguous: {seqs:?}"
    );
}

/// The sequence number of every event the receiver holds.
fn sequence_numbers(receiver: &mut broadcast::Receiver<CodingAgentEvent>) -> Vec<u64> {
    let mut seqs = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        seqs.push(event.seq);
    }
    seqs
}

#[tokio::test]
async fn a_tool_that_ends_the_round_is_still_answered_before_the_next_one() {
    // The round token fires while the tool is running, so the loop commits the
    // result and goes round again rather than dropping the call.
    let started = Arc::new(Counter::default());
    let counter = Arc::clone(&started);
    let watcher = RegisteredTool::new(
        ToolDefinition::function("watch", "Waits", json!({"type": "object"})),
        Arc::new(move |_arguments, context| {
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                counter.bump();
                context.cancel.cancelled().await;
                Err(ToolError::cancelled("Cancelled"))
            })
        }),
    )
    .with_source(ToolSource::Native);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("watch", "call_1", json!({}))),
        ScriptedCall::response(text_response("after the interrupt")),
    ])
    .tools([watcher])
    .build();
    let control = session.control_handle();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::ToolCallStarted { .. })
        })
        .await;
        control.interrupt_then_steer("stop that", None);
    });

    let answer = timeout(PATIENCE, session.prompt("watch something"))
        .await
        .expect("the interrupt unblocks the tool")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    assert_eq!(answer.as_deref(), Some("after the interrupt"));
    assert_eq!(started.count(), 1);
    let results = tool_results(&session, 2);
    assert_eq!(results.len(), 1, "the interrupted call still has a result");
    assert!(matches!(
        session.history().turns().get(3),
        Some(Message::Steering { .. })
    ));
}
