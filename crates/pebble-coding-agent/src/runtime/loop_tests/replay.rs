//! Invariant 2: a replayed turn withdraws what it showed.
//!
//! A stream that breaks after the model has already produced visible output
//! cannot be reconnected underneath a reader, so the session withdraws the
//! turn — one empty
//! [`AssistantOutputReplace`](crate::events::CodingEvent::AssistantOutputReplace) — and
//! plays it again. These tests pin what a reader sees while that happens, how
//! many times it may happen, and which failures are not worth repeating at all.

use std::time::Duration;

use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::{
    ContentPart, FinishReason, RetryClassification, StreamEvent, ToolCall, Warning,
};
use tokio::time::timeout;

use super::*;
use crate::reasoning::ReasoningOutput;
use crate::test_support::{
    reasoning_delta_events, reasoning_response, responses_reasoning_response, text_delta_events,
    tool_call_events, with_finish_reason,
};
use crate::types::{InputSource, LlmRetryPhase};

/// The failure a stream that dropped mid-turn reports.
///
/// Fabro's `LlmError::Stream` has no single lithos counterpart; a dropped
/// connection arrives as a network failure the client may repeat.
fn dropped_stream() -> ScriptedFailure {
    ScriptedFailure::retryable(LlmErrorKind::Network, "connection reset")
}

/// The reasoning every `AssistantMessage` carried, in order.
fn reasoning_of(events: &[CodingEvent]) -> Vec<Option<ReasoningOutput>> {
    events
        .iter()
        .filter_map(|event| match event {
            CodingEvent::AssistantMessage { reasoning, .. } => Some(reasoning.clone()),
            _ => None,
        })
        .collect()
}

/// The replay-relevant events, rendered the way fabro's tests read them.
fn output_trace(events: &[CodingEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            CodingEvent::TextDelta { delta } => Some(format!("delta:{delta}")),
            CodingEvent::AssistantOutputReplace { text, reasoning } => {
                Some(format!("replace:{text}:{reasoning:?}"))
            }
            CodingEvent::AssistantMessage { text, .. } => Some(format!("message:{text}")),
            CodingEvent::Error { .. } => Some("error".to_owned()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_streamed_turn_publishes_its_text() {
    let (mut session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(text_response("Hello there!"))]);
    let mut events = session.subscribe();

    session.prompt("Hi").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let deltas: Vec<&str> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::TextDelta { delta } => Some(delta.as_str()),
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

    session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(provider.call_count(), 2);
    assert_eq!(session.history().turns().len(), 2);
    assert!(matches!(
        session.history().turns().last(),
        Some(Message::Assistant { content, .. }) if content == "Recovered"
    ));

    let published = settled(&mut session, &mut events).await;
    let retries: Vec<&CodingEvent> = published
        .iter()
        .filter(|event| matches!(event, CodingEvent::LlmRetry { .. }))
        .collect();
    assert_eq!(retries.len(), 1);
    assert!(matches!(
        retries[0],
        CodingEvent::LlmRetry { error, .. }
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

    session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(provider.call_count(), 2);
    assert!(matches!(
        session.history().turns().last(),
        Some(Message::Assistant { content, .. }) if content == "Hello"
    ));

    let published = settled(&mut session, &mut events).await;
    let observed: Vec<String> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::LlmRequestStarted { .. } => Some("start".to_owned()),
            CodingEvent::LlmFirstOutput { kind } => Some(format!("first:{}", kind.as_str())),
            CodingEvent::TextDelta { delta } => Some(format!("delta:{delta}")),
            CodingEvent::AssistantOutputReplace { text, reasoning } => {
                Some(format!("replace:{text}:{reasoning:?}"))
            }
            CodingEvent::LlmRetry { phase, .. } => Some(format!("retry:{}", phase.as_str())),
            CodingEvent::AssistantMessage { text, .. } => Some(format!("message:{text}")),
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

    session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(provider.call_count(), 2);
    assert!(matches!(
        session.history().turns().last(),
        Some(Message::Assistant { content, .. }) if content == "Recovered"
    ));

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::LlmRequestStarted { .. }
        )),
        1,
        "the replay happens inside the round, not as a new one"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing is withdrawn"
    );
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(
        retries,
        [(0, LlmRetryPhase::Consume)],
        "the restart reports the incomplete stream error"
    );
}

#[tokio::test(start_paused = true)]
async fn a_stream_that_never_finishes_fails_once_its_replays_are_spent() {
    // The last scripted call repeats, so every attempt ends the same way: the
    // stream stops without a response, four times over.
    let (mut session, provider) = TestSession::answering(vec![ScriptedCall::Events(Vec::new())]);
    let mut events = session.subscribe();

    let error = session
        .prompt("Hello")
        .await
        .expect_err("no attempt produced a response");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::StreamDecode),
        "{error:?}"
    );
    assert!(
        ErrorData::from(&error)
            .message
            .contains("ended without completion")
    );
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
            CodingEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
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
            CodingEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing is withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::Error { .. }
        )),
        1
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::AssistantMessage { .. }
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
        .prompt("Hello")
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

    let answer = session.prompt("Hello").await.expect("the prompt succeeds");

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
            CodingEvent::AssistantOutputReplace { .. }
        )),
        1,
        "what the truncated turn showed is withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::AssistantMessage { .. }
        )),
        1
    );
}

#[tokio::test(start_paused = true)]
async fn a_failure_worth_no_repeat_ends_the_prompt_on_the_first_attempt() {
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

        let error = session.prompt("Hello").await.expect_err("the call failed");

        assert!(
            matches!(&error, Error::Llm(inner) if inner.kind() == kind),
            "the prompt reports the provider's own failure: {error:?}"
        );
        assert_eq!(provider.call_count(), 1);
        assert_eq!(session.history().turns().len(), 1);

        let published = drained(&mut events).await;
        assert_eq!(
            count(&published, |event| matches!(
                event,
                CodingEvent::LlmRetry { .. }
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
                CodingEvent::Error { error } => Some(error.clone()),
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

    let error = session.prompt("Hello").await.expect_err("the call failed");

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
        .prompt("Hello")
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
            CodingEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
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
            CodingEvent::AssistantOutputReplace { text, reasoning }
                if text.is_empty() && reasoning.is_none()
        )),
        4,
        "one withdrawal per attempt that showed something, the last one included"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::Error { .. }
        )),
        1
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::AssistantMessage { .. }
                | CodingEvent::ToolCallStarted { .. }
                | CodingEvent::ToolCallCompleted { .. }
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
    .options(CodingAgentOptions {
        turn_replay: RetryPolicy::exponential()
            .max_attempts(4)
            .initial_delay(Duration::from_millis(1)),
        ..CodingAgentOptions::default()
    })
    .build();

    let answer = timeout(Duration::from_secs(5), session.prompt("Hello"))
        .await
        .expect("the replay reopens rather than waiting on a permit it holds")
        .expect("the prompt succeeds");

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
        .prompt("Hello")
        .await
        .expect_err("the credential failed");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::Authentication),
        "{error:?}"
    );
    assert_eq!(provider.call_count(), 2);
    assert_eq!(session.state(), CodingAgentState::Closed);

    let published = drained(&mut events).await;
    assert_eq!(output_trace(&published), [
        "delta:Hel",
        "replace::None",
        "error",
    ]);
    let reported = published
        .iter()
        .find_map(|event| match event {
            CodingEvent::Error { error } => Some(error.clone()),
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

    let answer = session
        .prompt("Hello")
        .await
        .expect("the client reconnects");

    assert_eq!(answer.as_deref(), Some("Recovered"));
    assert_eq!(provider.call_count(), 2, "the client opened the call again");
    let published = settled(&mut session, &mut events).await;
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
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
            CodingEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing had to be withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::LlmRequestStarted { .. }
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

    let answer = session
        .prompt("Hello")
        .await
        .expect("the client reconnects");

    assert_eq!(answer.as_deref(), Some("Recovered"));
    assert_eq!(provider.call_count(), 2);

    let published = settled(&mut session, &mut events).await;
    let retries: Vec<(usize, LlmRetryPhase)> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::LlmRetry { attempt, phase, .. } => Some((*attempt, *phase)),
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
            CodingEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing was shown, so nothing had to be withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::LlmRequestStarted { .. }
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

    let answer = session
        .prompt("Hello")
        .await
        .expect("the fourth call answers");

    assert_eq!(answer.as_deref(), Some("Recovered"));
    assert_eq!(provider.call_count(), 4);
    assert_eq!(session.history().turns().len(), 2, "one committed turn");

    let published = settled(&mut session, &mut events).await;
    let phases: Vec<LlmRetryPhase> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::LlmRetry { phase, .. } => Some(*phase),
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
            CodingEvent::AssistantOutputReplace { .. }
        )),
        1,
        "only the failure a reader saw is withdrawn"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::LlmRequestStarted { .. }
        )),
        1,
        "every replay happens inside the one round"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::AssistantMessage { .. }
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

    session
        .prompt("What is 2+2?")
        .await
        .expect("the prompt succeeds");

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

    session
        .prompt("Do something")
        .await
        .expect("the prompt succeeds");

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

    session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(provider.call_count(), 2);
    let published = settled(&mut session, &mut events).await;
    assert_eq!(reasoning_of(&published), [Some(ReasoningOutput::new(
        "final summary",
        "final trace",
    ))]);
}

// The lithos OpenAI Responses codec decodes a `reasoning` item into a readable
// `Reasoning` part plus the whole item as an opaque `openai.reasoning` part.
// The readable text is the item's `reasoning_text` entries or, when it has
// none, its `summary_text` entries, either way joined by a blank line. Pebble's
// normalizer keeps that fallback out of `trace` only because the codec's
// separator matches its own, so these tests pin both the shape and the
// dependency.

/// The `reasoning` item a summary-only response carries.
fn two_summary_item() -> serde_json::Value {
    json!({
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "gAAAAA",
        "summary": [
            {"type": "summary_text", "text": "A"},
            {"type": "summary_text", "text": "B"},
        ],
    })
}

#[tokio::test]
async fn a_responses_item_with_only_summary_blocks_reports_a_summary_and_no_trace() {
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::response(
        responses_reasoning_response("4.", Some("A\n\nB"), two_summary_item()),
    )]);
    let mut events = session.subscribe();

    session
        .prompt("What is 2+2?")
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let reasoning = reasoning_of(&published);
    assert_eq!(reasoning, [Some(ReasoningOutput::from_summary("A\n\nB"))]);
    let value = serde_json::to_value(&reasoning[0]).expect("serializes");
    assert_eq!(value, json!({"summary": "A\n\nB"}));
}

#[tokio::test]
async fn a_fallback_trace_joined_differently_from_the_summary_survives_as_a_trace() {
    // The normalizer drops the codec's summary fallback by equality alone. A
    // codec that joined the summaries with "" instead of a blank line would
    // hand a reader this bogus trace, which is the regression lithos-llm PR #2
    // fixed; this pins that pebble relies on the codec's separator.
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::response(
        responses_reasoning_response("4.", Some("AB"), two_summary_item()),
    )]);
    let mut events = session.subscribe();

    session
        .prompt("What is 2+2?")
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(reasoning_of(&published), [Some(ReasoningOutput::new(
        "A\n\nB", "AB",
    ))]);
}

#[tokio::test]
async fn a_responses_item_with_reasoning_text_reports_it_once_as_the_trace() {
    let (mut session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(responses_reasoning_response(
            "4.",
            Some("step one"),
            json!({
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "A"}],
                "content": [{"type": "reasoning_text", "text": "step one"}],
            }),
        ))]);
    let mut events = session.subscribe();

    session
        .prompt("What is 2+2?")
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let reasoning = reasoning_of(&published);
    assert_eq!(reasoning, [Some(ReasoningOutput::new("A", "step one"))]);
    let value = serde_json::to_value(&reasoning[0]).expect("serializes");
    assert_eq!(value, json!({"summary": "A", "trace": "step one"}));
}

// --- The inference bracket ---

/// The bracket events, as `(label, detail)` pairs.
fn bracket(events: &[CodingEvent]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|event| match event {
            CodingEvent::LlmRequestStarted { requested_model } => {
                Some(("started".to_owned(), requested_model.clone()))
            }
            CodingEvent::LlmFirstOutput { kind } => {
                Some(("first_output".to_owned(), kind.as_str().to_owned()))
            }
            CodingEvent::AssistantMessage { text, .. } => {
                Some(("message".to_owned(), text.clone()))
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn one_bracket_wraps_a_text_first_turn() {
    let (mut session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(text_response("Hello"))]);
    let mut events = session.subscribe();

    session.prompt("Hi").await.expect("the prompt succeeds");

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

    session.prompt("Hi").await.expect("the prompt succeeds");

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

    session
        .prompt("Use the tool")
        .await
        .expect("the prompt succeeds");

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

// --- An answer the output limit cut short ---
//
// A `Length` finish is not a broken stream: the model used its whole output
// budget, and the same request would stop at the same place. The answer as far
// as it got is kept, and the model is asked once to continue it.

#[tokio::test(start_paused = true)]
async fn an_answer_cut_at_the_output_limit_is_continued_once() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::response(with_finish_reason(
            text_response("The answer begins and"),
            FinishReason::Length,
        )),
        ScriptedCall::response(text_response(" ends here.")),
    ]);
    let mut events = session.subscribe();

    let answer = session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(answer.as_deref(), Some(" ends here."));
    assert_eq!(provider.call_count(), 2);
    let turns = session.history().turns().to_vec();
    assert!(
        matches!(&turns[1], Message::Assistant { content, .. } if content == "The answer begins and"),
        "the cut answer is committed as it stood: {turns:?}"
    );
    assert!(
        matches!(
            &turns[2],
            Message::User { content, .. } if content.text_content().contains("stopped at the output limit")
        ),
        "the continuation request follows it: {turns:?}"
    );

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::Warning { kind, .. } if kind == "output_limit"
        )),
        1
    );
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::UserInput {
                source: InputSource::Agent,
                ..
            })),
        "the continuation is agent input, not the user's"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::AssistantOutputReplace { .. }
        )),
        0,
        "nothing the model showed is withdrawn"
    );
}

#[tokio::test(start_paused = true)]
async fn an_answer_cut_twice_is_left_as_it_stands() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::response(with_finish_reason(
            text_response("Part one,"),
            FinishReason::Length,
        )),
        ScriptedCall::response(with_finish_reason(
            text_response(" part two,"),
            FinishReason::Length,
        )),
        ScriptedCall::response(text_response("never asked for")),
    ]);
    let mut events = session.subscribe();

    let answer = session.prompt("Hello").await.expect("the prompt succeeds");

    assert_eq!(answer.as_deref(), Some(" part two,"));
    assert_eq!(provider.call_count(), 2, "one continuation per prompt");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::Warning { kind, .. } if kind == "output_limit"
        )),
        2,
        "every cut is reported, whether or not it is continued"
    );
}

#[tokio::test(start_paused = true)]
async fn the_continuation_budget_is_per_prompt() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::response(with_finish_reason(
            text_response("first,"),
            FinishReason::Length,
        )),
        ScriptedCall::response(text_response(" done.")),
        ScriptedCall::response(with_finish_reason(
            text_response("second,"),
            FinishReason::Length,
        )),
        ScriptedCall::response(text_response(" also done.")),
    ]);

    session
        .prompt("One")
        .await
        .expect("the first prompt succeeds");
    let answer = session
        .prompt("Two")
        .await
        .expect("the second prompt succeeds");

    assert_eq!(answer.as_deref(), Some(" also done."));
    assert_eq!(provider.call_count(), 4);
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
}

#[tokio::test(start_paused = true)]
async fn a_model_layer_warning_is_reported_with_the_turn() {
    let mut response = text_response("Answer");
    response.warnings.push(Warning {
        code:    "truncated_tool_call".to_owned(),
        message: "the output limit cut off a call to write_note".to_owned(),
    });
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::response(response)]);
    let mut events = session.subscribe();

    session.prompt("Hello").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::Warning { kind, message, .. }
                if kind == "truncated_tool_call" && message.contains("write_note")
        )),
        "the response's warning reaches the event stream: {published:?}"
    );
}
