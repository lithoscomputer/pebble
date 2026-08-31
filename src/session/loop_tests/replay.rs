//! Invariant 2: a replayed turn withdraws what it showed.
//!
//! A stream that breaks after the model has already produced visible output
//! cannot be reconnected underneath a reader, so the session withdraws the
//! turn — one empty
//! [`AssistantOutputReplace`](crate::AgentEvent::AssistantOutputReplace) — and
//! plays it again. These tests pin what a reader sees while that happens, how
//! many times it may happen, and which failures are not worth repeating at all.

use std::time::Duration;

use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::{ContentPart, FinishReason, RetryClassification, StreamEvent, ToolCall};
use tokio::time::timeout;

use super::*;
use crate::reasoning::ReasoningOutput;
use crate::test_support::{
    reasoning_delta_events, reasoning_response, text_delta_events, tool_call_events,
    with_finish_reason,
};
use crate::types::LlmRetryPhase;

/// The failure a stream that dropped mid-turn reports.
///
/// Fabro's `LlmError::Stream` has no single lithos counterpart; a dropped
/// connection arrives as a network failure the client may repeat.
fn dropped_stream() -> ScriptedFailure {
    ScriptedFailure::retryable(LlmErrorKind::Network, "connection reset")
}

/// The reasoning every `AssistantMessage` carried, in order.
fn reasoning_of(events: &[AgentEvent]) -> Vec<Option<ReasoningOutput>> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::AssistantMessage { reasoning, .. } => Some(reasoning.clone()),
            _ => None,
        })
        .collect()
}

/// The replay-relevant events, rendered the way fabro's tests read them.
fn output_trace(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TextDelta { delta } => Some(format!("delta:{delta}")),
            AgentEvent::AssistantOutputReplace { text, reasoning } => {
                Some(format!("replace:{text}:{reasoning:?}"))
            }
            AgentEvent::AssistantMessage { text, .. } => Some(format!("message:{text}")),
            AgentEvent::Error { .. } => Some("error".to_owned()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_streamed_turn_publishes_its_text() {
    let (mut session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(text_response("Hello there!"))]);
    let mut events = session.subscribe();

    session.run("Hi").await.expect("the run succeeds");

    let published = settled(&mut session, &mut events).await;
    let deltas: Vec<&str> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TextDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, ["Hello there!"]);
}

#[tokio::test(start_paused = true)]
async fn a_broken_stream_is_replayed_and_only_the_recovered_turn_is_committed() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::fails_after("partial", dropped_stream()),
        ScriptedCall::response(text_response("Recovered")),
    ]);
    let mut events = session.subscribe();

    session.run("Hello").await.expect("the run succeeds");

    assert_eq!(provider.call_count(), 2);
    assert_eq!(session.history().turns().len(), 2);
    assert!(matches!(
        session.history().turns().last(),
        Some(Message::Assistant { content, .. }) if content == "Recovered"
    ));

    let published = settled(&mut session, &mut events).await;
    let retries: Vec<&AgentEvent> = published
        .iter()
        .filter(|event| matches!(event, AgentEvent::LlmRetry { .. }))
        .collect();
    assert_eq!(retries.len(), 1);
    assert!(matches!(
        retries[0],
        AgentEvent::LlmRetry { error, .. }
            if error.retry == Some(RetryClassification::Safe)
    ));
    assert_eq!(output_trace(&published), [
        "delta:partial",
        "replace::None",
        "delta:Recovered",
        "message:Recovered",
    ]);
}

#[tokio::test(start_paused = true)]
async fn a_replay_re_arms_the_first_output_latch() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::Events(text_delta_events("Hel")),
        ScriptedCall::response(text_response("Hello")),
    ]);
    let mut events = session.subscribe();

    session.run("Hello").await.expect("the run succeeds");

    assert_eq!(provider.call_count(), 2);
    assert!(matches!(
        session.history().turns().last(),
        Some(Message::Assistant { content, .. }) if content == "Hello"
    ));

    let published = settled(&mut session, &mut events).await;
    let observed: Vec<String> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRequestStarted { .. } => Some("start".to_owned()),
            AgentEvent::LlmFirstOutput { kind } => Some(format!("first:{}", kind.as_str())),
            AgentEvent::TextDelta { delta } => Some(format!("delta:{delta}")),
            AgentEvent::AssistantOutputReplace { text, reasoning } => {
                Some(format!("replace:{text}:{reasoning:?}"))
            }
            AgentEvent::LlmRetry { phase, .. } => Some(format!("retry:{}", phase.as_str())),
            AgentEvent::AssistantMessage { text, .. } => Some(format!("message:{text}")),
            _ => None,
        })
        .collect();
    assert_eq!(observed, [
        "start",
        "first:text",
        "delta:Hel",
        "replace::None",
        "retry:consume",
        "first:text",
        "delta:Hello",
        "message:Hello",
    ]);
}

#[tokio::test(start_paused = true)]
async fn a_stream_that_ends_without_finishing_is_replayed_with_nothing_to_withdraw() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::Events(Vec::new()),
        ScriptedCall::response(text_response("Recovered")),
    ]);
    let mut events = session.subscribe();

    session.run("Hello").await.expect("the run succeeds");

    assert_eq!(provider.call_count(), 2);
    assert!(matches!(
        session.history().turns().last(),
        Some(Message::Assistant { content, .. }) if content == "Recovered"
    ));

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::LlmRequestStarted { .. }
        )),
        1,
        "the replay happens inside the round, not as a new one"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing is withdrawn"
    );
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(
        retries,
        [(0, LlmRetryPhase::Consume)],
        "the one restart with no error behind it is still announced"
    );
}

#[tokio::test(start_paused = true)]
async fn a_stream_that_never_finishes_fails_once_its_replays_are_spent() {
    // The last scripted call repeats, so every attempt ends the same way: the
    // stream stops without a response, four times over.
    let (mut session, provider) = TestSession::answering(vec![ScriptedCall::Events(Vec::new())]);
    let mut events = session.subscribe();

    let error = session
        .run("Hello")
        .await
        .expect_err("no attempt produced a response");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::StreamDecode),
        "{error:?}"
    );
    assert!(error.to_string().contains("after every replay"));
    assert_eq!(provider.call_count(), 4, "one attempt and three replays");
    assert_eq!(
        session.history().turns().len(),
        1,
        "an uncommitted turn leaves only the input"
    );

    let published = drained(&mut events).await;
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(retries, [
        (0, LlmRetryPhase::Consume),
        (1, LlmRetryPhase::Consume),
        (2, LlmRetryPhase::Consume),
    ]);
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing is withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::Error { .. }
        )),
        1
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantMessage { .. }
        )),
        0
    );
}

#[tokio::test(start_paused = true)]
async fn the_last_unfinished_attempt_withdraws_what_it_showed() {
    let (mut session, provider) =
        TestSession::answering(vec![ScriptedCall::Events(text_delta_events("partial"))]);
    let mut events = session.subscribe();

    let error = session
        .run("Hello")
        .await
        .expect_err("no attempt produced a response");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::StreamDecode),
        "{error:?}"
    );
    assert_eq!(provider.call_count(), 4);

    let published = drained(&mut events).await;
    assert_eq!(output_trace(&published), [
        "delta:partial",
        "replace::None",
        "delta:partial",
        "replace::None",
        "delta:partial",
        "replace::None",
        "delta:partial",
        "replace::None",
        "error",
    ]);
}

#[tokio::test(start_paused = true)]
async fn a_truncated_response_is_replayed_rather_than_committed() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::response(with_finish_reason(
            text_response("half an ans"),
            FinishReason::Incomplete,
        )),
        ScriptedCall::response(text_response("the whole answer")),
    ]);
    let mut events = session.subscribe();

    let answer = session.run("Hello").await.expect("the run succeeds");

    assert_eq!(answer.as_deref(), Some("the whole answer"));
    assert_eq!(provider.call_count(), 2);
    assert_eq!(
        session.history().turns().len(),
        2,
        "the truncated turn is not committed"
    );

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantOutputReplace { .. }
        )),
        1,
        "what the truncated turn showed is withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantMessage { .. }
        )),
        1
    );
}

#[tokio::test(start_paused = true)]
async fn a_failure_worth_no_repeat_ends_the_run_on_the_first_attempt() {
    for kind in [
        LlmErrorKind::Authentication,
        LlmErrorKind::ContextLength,
        LlmErrorKind::QuotaExceeded,
    ] {
        let (mut session, provider) = TestSession::answering(vec![
            ScriptedCall::fails_after(
                "partial",
                ScriptedFailure::terminal(kind, format!("deterministic provider error: {kind:?}")),
            ),
            ScriptedCall::response(text_response("should not replay")),
        ]);
        let mut events = session.subscribe();

        let error = session.run("Hello").await.expect_err("the call failed");

        assert!(
            matches!(&error, Error::Llm(inner) if inner.kind() == kind),
            "the run reports the provider's own failure: {error:?}"
        );
        assert_eq!(provider.call_count(), 1);
        assert_eq!(session.history().turns().len(), 1);

        let published = drained(&mut events).await;
        assert_eq!(
            count(&published, |event| matches!(
                event,
                AgentEvent::LlmRetry { .. }
            )),
            0
        );
        assert_eq!(output_trace(&published), [
            "delta:partial",
            "replace::None",
            "error",
        ]);
        let reported = published
            .iter()
            .find_map(|event| match event {
                AgentEvent::Error { error } => Some(error.clone()),
                _ => None,
            })
            .expect("the failure is published");
        assert_eq!(reported.llm_kind, Some(kind));
    }
}

#[tokio::test(start_paused = true)]
async fn spent_quota_is_never_replayed() {
    let (mut session, provider) = TestSession::answering(vec![ScriptedCall::fails_after(
        "partial",
        ScriptedFailure::terminal(
            LlmErrorKind::QuotaExceeded,
            "You exceeded your current quota",
        )
        .with_provider_code("insufficient_quota"),
    )]);

    let error = session.run("Hello").await.expect_err("the call failed");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::QuotaExceeded),
        "{error:?}"
    );
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_turn_that_never_arrives_fails_once_and_commits_nothing() {
    let mut events = text_delta_events("partial");
    events.extend(tool_call_events(&ToolCall::function(
        "call_1",
        "echo",
        json!({"text": "should not run"}),
    )));
    events.push(Err(dropped_stream()));
    let (mut session, provider) = TestSession::new(vec![ScriptedCall::Events(events)])
        .tools([echo_tool()])
        .build();
    let mut subscriber = session.subscribe();

    let error = session
        .run("Hello")
        .await
        .expect_err("every attempt failed");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::Network),
        "{error:?}"
    );
    assert_eq!(provider.call_count(), 4, "one attempt and three replays");
    assert_eq!(
        session.history().turns().len(),
        1,
        "an uncommitted turn leaves only the input"
    );

    let published = drained(&mut subscriber).await;
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(
        retries,
        [
            (0, LlmRetryPhase::Consume),
            (1, LlmRetryPhase::Consume),
            (2, LlmRetryPhase::Consume),
        ],
        "three replays, numbered from zero"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantOutputReplace { text, reasoning }
                if text.is_empty() && reasoning.is_none()
        )),
        4,
        "one withdrawal per attempt that showed something, the last one included"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::Error { .. }
        )),
        1
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantMessage { .. }
                | AgentEvent::ToolCallStarted { .. }
                | AgentEvent::ToolCallCompleted { .. }
        )),
        0,
        "a turn that never committed runs no tools"
    );
}

#[tokio::test]
async fn a_replay_never_waits_on_the_stream_it_replaced() {
    // The client's concurrency limiter holds its permit for exactly as long as
    // the stream it handed out lives, and the reopen asks the same limiter for
    // a permit. A replay that kept its failed stream alive would wait on
    // itself, so the session drops it first.
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::fails_after("partial", dropped_stream()),
        ScriptedCall::response(text_response("Recovered")),
    ])
    .limited(1)
    .options(SessionOptions {
        retry_policy: RetryPolicy::exponential()
            .max_attempts(4)
            .initial_delay(Duration::from_millis(1)),
        ..SessionOptions::default()
    })
    .build();

    let answer = timeout(Duration::from_secs(5), session.run("Hello"))
        .await
        .expect("the replay reopens rather than waiting on a permit it holds")
        .expect("the run succeeds");

    assert_eq!(answer.as_deref(), Some("Recovered"));
    assert_eq!(provider.call_count(), 2);
}

#[tokio::test(start_paused = true)]
async fn a_reopen_that_fails_on_the_credential_closes_the_session() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::Events(text_delta_events("Hel")),
        ScriptedCall::Failure(
            ScriptedFailure::terminal(LlmErrorKind::Authentication, "bad key").with_status(401),
        ),
    ]);
    let mut events = session.subscribe();

    let error = session
        .run("Hello")
        .await
        .expect_err("the credential failed");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::Authentication),
        "{error:?}"
    );
    assert_eq!(provider.call_count(), 2);
    assert_eq!(session.state(), SessionState::Closed);

    let published = drained(&mut events).await;
    assert_eq!(output_trace(&published), [
        "delta:Hel",
        "replace::None",
        "error",
    ]);
    let reported = published
        .iter()
        .find_map(|event| match event {
            AgentEvent::Error { error } => Some(error.clone()),
            _ => None,
        })
        .expect("the failure is published");
    assert_eq!(reported.llm_kind, Some(LlmErrorKind::Authentication));
    assert_eq!(reported.status, Some(401));
}

#[tokio::test(start_paused = true)]
async fn the_clients_own_reconnect_is_published_as_an_open_phase_retry() {
    // The half of the retry story pebble does not own: a failure before any
    // visible output is the middleware's to repeat, and the session hears about
    // it only through the observer an application installs.
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::Failure(dropped_stream()),
        ScriptedCall::response(text_response("Recovered")),
    ])
    .retrying(RetryPolicy::exponential().max_attempts(4))
    .build();
    let mut events = session.subscribe();

    let answer = session.run("Hello").await.expect("the client reconnects");

    assert_eq!(answer.as_deref(), Some("Recovered"));
    assert_eq!(provider.call_count(), 2, "the client opened the call again");
    let published = settled(&mut session, &mut events).await;
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(
        retries,
        [(0, LlmRetryPhase::Open)],
        "one reconnect, numbered from zero, named as the client's own"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing had to be withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::LlmRequestStarted { .. }
        )),
        1,
        "the reconnect happens inside the round the session opened"
    );
}

#[tokio::test(start_paused = true)]
async fn a_drop_before_any_output_is_the_clients_to_repeat() {
    // The client reconnects a stream that failed before a reader saw anything,
    // so the session neither replays it nor withdraws anything — it only
    // reports what the client did, as a consume-phase retry.
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::Events(vec![Err(dropped_stream())]),
        ScriptedCall::response(text_response("Recovered")),
    ])
    .retrying(RetryPolicy::exponential().max_attempts(4))
    .build();
    let mut events = session.subscribe();

    let answer = session.run("Hello").await.expect("the client reconnects");

    assert_eq!(answer.as_deref(), Some("Recovered"));
    assert_eq!(provider.call_count(), 2);

    let published = settled(&mut session, &mut events).await;
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(
        retries,
        [(0, LlmRetryPhase::Consume)],
        "one reconnect, numbered from zero, named for the stream it lost"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing had to be withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::LlmRequestStarted { .. }
        )),
        1
    );
}

#[tokio::test(start_paused = true)]
async fn one_failure_is_replayed_by_one_layer() {
    // The whole retry story in one round, with the middleware installed: the
    // open fails, the next stream drops before anything is visible, the one
    // after drops with output on the screen, and the fourth answers. The first
    // two are the client's to repeat and the session never learns of them
    // except through the observer; only the third is the session's, and only
    // it withdraws anything.
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::Failure(dropped_stream()),
        ScriptedCall::Events(vec![Err(dropped_stream())]),
        ScriptedCall::fails_after("partial", dropped_stream()),
        ScriptedCall::response(text_response("Recovered")),
    ])
    .retrying(RetryPolicy::exponential().max_attempts(4))
    .build();
    let mut events = session.subscribe();

    let answer = session.run("Hello").await.expect("the fourth call answers");

    assert_eq!(answer.as_deref(), Some("Recovered"));
    assert_eq!(provider.call_count(), 4);
    assert_eq!(session.history().turns().len(), 2, "one committed turn");

    let published = settled(&mut session, &mut events).await;
    let phases: Vec<LlmRetryPhase> = published
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRetry { phase, .. } => Some(*phase),
            _ => None,
        })
        .collect();
    assert_eq!(
        phases,
        [
            LlmRetryPhase::Open,
            LlmRetryPhase::Consume,
            LlmRetryPhase::Consume,
        ],
        "three failures, three retries: none of them counted twice"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantOutputReplace { .. }
        )),
        1,
        "only the failure a reader saw is withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::LlmRequestStarted { .. }
        )),
        1,
        "every replay happens inside the one round"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            AgentEvent::AssistantMessage { .. }
        )),
        1
    );
}

// --- Reasoning ---

#[tokio::test]
async fn a_completed_response_reports_its_reasoning_once() {
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::response(
        reasoning_response("4.", "the user wants 2+2", "2+2 is 4"),
    )]);
    let mut events = session.subscribe();

    session.run("What is 2+2?").await.expect("the run succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(reasoning_of(&published), [Some(ReasoningOutput::new(
        "the user wants 2+2",
        "2+2 is 4",
    ))]);
}

#[tokio::test]
async fn a_turn_with_no_visible_text_still_reports_its_reasoning() {
    let mut response = tool_call_response("nonexistent_tool", "call_1", json!({}));
    response.content = vec![
        ContentPart::opaque(
            "openai_compatible.reasoning_details",
            json!([{"type": "reasoning.summary", "summary": "call the tool"}]),
        ),
        ContentPart::ToolCall(ToolCall::function("call_1", "nonexistent_tool", json!({}))),
    ];
    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(response),
        ScriptedCall::response(text_response("OK")),
    ]);
    let mut events = session.subscribe();

    session.run("Do something").await.expect("the run succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(reasoning_of(&published), [
        Some(ReasoningOutput::from_summary("call the tool")),
        None,
    ]);
}

#[tokio::test(start_paused = true)]
async fn only_the_replayed_turn_contributes_reasoning() {
    let mut broken = reasoning_delta_events("discarded thinking");
    broken.push(Err(dropped_stream()));
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::Events(broken),
        ScriptedCall::response(reasoning_response(
            "Recovered",
            "final summary",
            "final trace",
        )),
    ]);
    let mut events = session.subscribe();

    session.run("Hello").await.expect("the run succeeds");

    assert_eq!(provider.call_count(), 2);
    let published = settled(&mut session, &mut events).await;
    assert_eq!(reasoning_of(&published), [Some(ReasoningOutput::new(
        "final summary",
        "final trace",
    ))]);
}

// --- The inference bracket ---

/// The bracket events, as `(label, detail)` pairs.
fn bracket(events: &[AgentEvent]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::LlmRequestStarted { requested_model } => {
                Some(("started".to_owned(), requested_model.clone()))
            }
            AgentEvent::LlmFirstOutput { kind } => {
                Some(("first_output".to_owned(), kind.as_str().to_owned()))
            }
            AgentEvent::AssistantMessage { text, .. } => Some(("message".to_owned(), text.clone())),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn one_bracket_wraps_a_text_first_turn() {
    let (mut session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(text_response("Hello"))]);
    let mut events = session.subscribe();

    session.run("Hi").await.expect("the run succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(bracket(&published), [
        ("started".to_owned(), "model".to_owned()),
        ("first_output".to_owned(), "text".to_owned()),
        ("message".to_owned(), "Hello".to_owned()),
    ]);
}

#[tokio::test]
async fn reasoning_that_arrives_first_is_what_the_bracket_reports() {
    let mut events = reasoning_delta_events("weighing options");
    events.extend(text_delta_events("Hello"));
    events.push(Ok(StreamEvent::Completed {
        response: text_response("Hello"),
    }));
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::Events(events)]);
    let mut subscriber = session.subscribe();

    session.run("Hi").await.expect("the run succeeds");

    let published = settled(&mut session, &mut subscriber).await;
    assert_eq!(bracket(&published), [
        ("started".to_owned(), "model".to_owned()),
        ("first_output".to_owned(), "reasoning".to_owned()),
        ("message".to_owned(), "Hello".to_owned()),
    ]);
}

#[tokio::test]
async fn a_turn_that_only_calls_a_tool_reports_a_tool_call_first() {
    let mut response = tool_call_response("nonexistent_tool", "call_1", json!({}));
    response.content = vec![ContentPart::ToolCall(ToolCall::function(
        "call_1",
        "nonexistent_tool",
        json!({}),
    ))];
    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(response),
        ScriptedCall::response(text_response("Done")),
    ]);
    let mut events = session.subscribe();

    session.run("Use the tool").await.expect("the run succeeds");

    let published = settled(&mut session, &mut events).await;
    let observed = bracket(&published);
    let kinds: Vec<&str> = observed
        .iter()
        .filter(|(label, _)| label == "first_output")
        .map(|(_, kind)| kind.as_str())
        .collect();
    assert_eq!(kinds, ["tool_call", "text"]);
    assert_eq!(
        observed
            .iter()
            .filter(|(label, _)| label == "started")
            .count(),
        2,
        "one bracket per round"
    );
}
