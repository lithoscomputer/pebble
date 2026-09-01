//! Invariant 3: an interrupt is announced exactly once.
//!
//! Every interrupt gesture raises a generation, and the loop settles the count
//! as it unwinds — one
//! [`RoundInterrupted`](crate::AgentEvent::RoundInterrupted) per gesture,
//! before the steer that replaces the abandoned round is delivered. These
//! tests interrupt the loop everywhere it can be interrupted: before it starts,
//! while it waits on the model, and while a tool is running.
//!
//! What the session owns is here too, because a prompt that ends — however it
//! ends — has to leave nothing behind: the wall-clock timer stops, the event
//! pump is joined, and a session restored from its record carries on where the
//! last one stopped.

use std::future::pending;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::{Role, ToolCall, ToolDefinition};
use serde_json::json;
use tokio::runtime::Handle;
use tokio::sync::broadcast;
use tokio::time::{sleep, timeout};

use super::super::testing::{builder, event_name, wait_for_event};
use super::*;
use crate::event::{EventSink, EventSinkError};
use crate::task_reminder::TASK_REMINDER_TEXT;
use crate::test_support::{
    message_text, multi_tool_call_response, scripted_client, text_delta_events, tool_call_events,
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

    let turns = session.history().turns();
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
        AgentEvent::SteeringInjected { text, .. } => Some(text.clone()),
        _ => None,
    });
    assert_eq!(steered.as_deref(), Some("hi there"));
}

#[tokio::test]
async fn a_bare_interrupt_parks_the_session_until_a_steer_arrives() {
    let (mut session, _provider) = TestSession::answering(answers("OK"));
    let mut events = session.subscribe();
    let handle = session.control_handle();
    handle.interrupt();

    let waker = handle.clone();
    let steering = tokio::spawn(async move {
        sleep(Duration::from_millis(10)).await;
        waker.steer("resume now", None);
    });
    timeout(PATIENCE, session.prompt("start"))
        .await
        .expect("the parked session wakes when steering arrives")
        .expect("the prompt succeeds");
    steering.await.expect("the steering task finishes");

    assert!(matches!(
        &session.history().turns()[1],
        Message::Steering { content, .. } if content == "resume now"
    ));
    assert!(!handle.is_waiting_for_steer());
    let published = settled(&mut session, &mut events).await;
    let generations: Vec<u64> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RoundInterrupted { generation } => Some(*generation),
            _ => None,
        })
        .collect();
    assert_eq!(generations, [1], "one gesture, one announcement");
}

#[tokio::test]
async fn an_interrupt_that_lands_while_the_session_is_parked_is_announced_too() {
    // The hardest of the exactly-once cases: the second gesture arrives after
    // the first has settled and the session is already waiting for a steer.
    let (mut session, provider) = TestSession::answering(answers("resumed"));
    let control = session.control_handle();
    let mut controller_events = session.subscribe();
    let mut recorded = session.subscribe();
    control.interrupt();

    let controller = tokio::spawn(async move {
        wait_for_event(&mut controller_events, |event| {
            matches!(event, AgentEvent::RoundInterrupted { generation: 1 })
        })
        .await;
        control.interrupt();
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
            AgentEvent::RoundInterrupted { generation } => Some(*generation),
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
        1,
        "a round abandoned before it opened costs no model call"
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
    {
        let mut control = session
            .control_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        control.interrupt_generation = control.interrupt_generation.saturating_add(1);
    }

    session.prompt("start").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let generations: Vec<u64> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RoundInterrupted { generation } => Some(*generation),
            _ => None,
        })
        .collect();
    assert_eq!(generations, [1], "the raised generation is announced once");
}

#[tokio::test]
async fn an_interrupt_settles_before_the_steer_that_replaces_it() {
    let (mut session, _provider) = TestSession::answering(answers("OK"));
    let mut events = session.subscribe();
    let handle = session.control_handle();
    handle.interrupt_then_steer("stop now", None);

    session.prompt("start").await.expect("the prompt succeeds");

    assert!(!handle.is_waiting_for_steer());
    let published = settled(&mut session, &mut events).await;
    let settled_at = position(&published, |event| {
        matches!(event, AgentEvent::RoundInterrupted { generation: 1 })
    })
    .expect("the interrupt settled");
    let steered_at = position(
        &published,
        |event| matches!(event, AgentEvent::SteeringInjected { text, .. } if text == "stop now"),
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
            matches!(event, AgentEvent::RoundInterrupted { generation: 1 })
        })
        .await;
        assert!(control.is_waiting_for_steer());
        control.steer("resume inference", None);
        control
    });

    timeout(PATIENCE, session.prompt("start"))
        .await
        .expect("the interrupt unblocks the hanging call")
        .expect("the prompt succeeds");
    let control = controller.await.expect("the controller finishes");

    assert_eq!(provider.call_count(), 2, "the round was asked again");
    assert!(!control.is_waiting_for_steer());
    let published = settled(&mut session, &mut recorded).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::RoundInterrupted { .. }
        )),
        1
    );
    let settled_at = position(&published, |event| {
        matches!(event, AgentEvent::RoundInterrupted { .. })
    })
    .expect("the interrupt settled");
    let steered_at = position(&published, |event| {
        matches!(event, AgentEvent::SteeringInjected { .. })
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
            matches!(event, AgentEvent::LlmFirstOutput {
                kind: LlmOutputKind::ToolCall,
            })
        })
        .await;
        control.interrupt();
        wait_for_event(&mut events, |event| {
            matches!(event, AgentEvent::RoundInterrupted { generation: 1 })
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

    let [
        ..,
        Message::System {
            content: committed, ..
        },
        Message::Assistant { content, .. },
    ] = session.history().turns()
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
            |event| matches!(event, AgentEvent::TextDelta { delta } if delta == "half an answer"),
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
    let turns = session.history().turns();
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
            AgentEvent::AssistantOutputReplace { .. }
        )),
        1,
        "the abandoned turn's output is withdrawn once"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::RoundInterrupted { .. }
        )),
        1
    );
    let withdrawn = position(&published, |event| {
        matches!(event, AgentEvent::AssistantOutputReplace { .. })
    })
    .expect("the withdrawal was published");
    let second_delta = published
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event, AgentEvent::TextDelta { .. }))
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
    .options(SessionOptions {
        turn_replay: RetryPolicy::exponential()
            .max_attempts(4)
            .initial_delay(Duration::from_secs(30)),
        ..SessionOptions::default()
    })
    .build();
    let cancel = session.cancel_token();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, AgentEvent::LlmRetry { .. })
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
    assert_eq!(session.state(), SessionState::Closed);
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
            matches!(event, AgentEvent::ToolCallStarted { .. })
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
    let stubborn = RegisteredTool {
        definition: ToolDefinition::function(
            "stubborn",
            "Never answers",
            json!({"type": "object"}),
        ),
        executor:   Arc::new(|_arguments, _context| Box::pin(pending())),
        source:     ToolSource::Native,
    };
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
                matches!(event, AgentEvent::ToolCallStarted { .. })
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

#[tokio::test]
async fn a_coordinator_can_send_the_loop_round_again() {
    struct OnceCoordinator {
        calls:  AtomicUsize,
        handle: SessionControlHandle,
    }

    impl CompletionCoordinator for OnceCoordinator {
        fn on_natural_completion(&self) -> bool {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                // A steer that arrived as the first answer completed: queue it
                // and ask for another round.
                self.handle.steer("after-completion steer", None);
                true
            } else {
                false
            }
        }
    }

    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(text_response("First reply")),
        ScriptedCall::response(text_response("Second reply, after steer")),
    ]);
    let handle = session.control_handle();
    session.set_completion_coordinator(Arc::new(OnceCoordinator {
        calls: AtomicUsize::new(0),
        handle,
    }));

    session.prompt("hi").await.expect("the prompt succeeds");

    let turns = session.history().turns();
    assert_eq!(turns.len(), 4, "input, answer, steer, answer");
    assert!(matches!(&turns[1], Message::Assistant { content, .. } if content == "First reply"));
    assert!(
        matches!(&turns[2], Message::Steering { content, .. } if content == "after-completion steer")
    );
    assert!(
        matches!(&turns[3], Message::Assistant { content, .. } if content == "Second reply, after steer")
    );
}

// --- The wall clock ---

#[tokio::test]
async fn a_prompt_that_outlasts_its_budget_ends_with_the_budget_as_its_reason() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("slow_tool", "call_1", json!({}))),
        ScriptedCall::response(text_response("Should not reach this")),
    ])
    .tools([blocking_tool("slow_tool")])
    .options(SessionOptions {
        wall_clock_timeout: Some(Duration::from_millis(10)),
        enable_loop_detection: false,
        ..SessionOptions::default()
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
    assert_eq!(session.state(), SessionState::Closed);
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
        .options(SessionOptions {
            wall_clock_timeout: Some(Duration::from_secs(10)),
            ..SessionOptions::default()
        })
        .build();

    session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(session.state(), SessionState::Idle);
    let turns = session.history().turns();
    assert_eq!(turns.len(), 2);
    assert!(matches!(&turns[1], Message::Assistant { content, .. } if content == "Fast response"));
}

#[tokio::test]
async fn a_finished_prompt_leaves_no_timer_behind() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(text_response("first")),
        ScriptedCall::response(text_response("second")),
    ])
    .options(SessionOptions {
        wall_clock_timeout: Some(Duration::from_millis(20)),
        ..SessionOptions::default()
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
        .options(SessionOptions {
            wall_clock_timeout: Some(Duration::from_secs(10)),
            ..SessionOptions::default()
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
    async fn record(&self, event: &SessionEvent) -> StdResult<(), EventSinkError> {
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
    let mut resumed = Session::from_record(&record, builder(client)).expect("the record restores");
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
fn sequence_numbers(receiver: &mut broadcast::Receiver<SessionEvent>) -> Vec<u64> {
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
    let watcher = RegisteredTool {
        definition: ToolDefinition::function("watch", "Waits", json!({"type": "object"})),
        executor:   Arc::new(move |_arguments, context| {
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                counter.bump();
                context.cancel.cancelled().await;
                Err(ToolError::cancelled("Cancelled"))
            })
        }),
        source:     ToolSource::Native,
    };
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
            matches!(event, AgentEvent::ToolCallStarted { .. })
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
