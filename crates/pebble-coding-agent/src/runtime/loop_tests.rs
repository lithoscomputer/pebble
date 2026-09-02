//! The turn loop, driven end to end against a scripted provider.
//!
//! These are fabro's own session tests, ported. They are the specification for
//! the three rules the loop exists to hold — every tool call gets a result, a
//! replayed turn withdraws what it showed, and an interrupt is announced
//! exactly once — plus everything else a round does on the way.
//!
//! The submodules split the suite by subject: what the session sends
//! ([`requests`]), what it does with a broken stream ([`replay`]), what happens
//! when someone interrupts it ([`interrupts`]), what it does as the window
//! fills ([`compaction`]), which harness a model resolves to ([`profiles`]),
//! what it does with the children it spawns ([`subagents`]), and what it hands
//! the built-in tools ([`tools`]).
//!
//! The subagent tests ([`subagents`]) arrived with the supervisor, and the
//! `use_skill`, `web_search` and `apply_patch` ones ([`tools`]) with the tools
//! they are about: a test written for a part of the loop lands beside that
//! part rather than in the order fabro happened to hold it.

mod compaction;
mod interrupts;
mod profiles;
mod replay;
mod requests;
mod subagents;
mod tools;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lithos_llm::types::{TokenCounts, ToolDefinition, ToolResult};
use serde_json::json;

use super::testing::{
    TestSession, blocking_tool, count, drain, drained, echo_tool, failing_tool, noop_tool,
    position, settled, wait_for_event,
};
use super::*;
use crate::test_support::{
    ScriptedCall, ScriptedFailure, multi_tool_call_response, text_response, tool_call_response,
    with_cost, with_usage,
};
use crate::tool::{ToolError, result_text};
use crate::types::{ContextWindowCountMethod, Message, ToolSource};

/// The one-call script most tests want: answer with this text, forever.
fn answers(text: &str) -> Vec<ScriptedCall> {
    vec![ScriptedCall::response(text_response(text))]
}

/// The tool results of the turn at `index`.
fn tool_results(session: &CodingRuntime, index: usize) -> Vec<ToolResult> {
    match session.history().turns().get(index) {
        Some(Message::ToolResults { results, .. }) => results.clone(),
        other => panic!("turn {index} should carry tool results, found {other:?}"),
    }
}

// --- Basics ---

#[tokio::test]
async fn a_new_session_starts_idle() {
    let (session, _provider) = TestSession::answering(answers("done"));

    assert_eq!(session.state(), CodingAgentState::Idle);
}

#[tokio::test]
async fn a_text_only_response_completes_the_prompt() {
    let (mut session, _provider) = TestSession::answering(answers("Hello there!"));

    let output = session.prompt("Hi").await.expect("the prompt succeeds");

    assert_eq!(output.as_deref(), Some("Hello there!"));
    assert_eq!(session.state(), CodingAgentState::Idle);
    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 2, "the input and the answer");
    assert!(matches!(&turns[0], Message::User { content, .. } if content == "Hi"));
    assert!(matches!(&turns[1], Message::Assistant { content, .. } if content == "Hello there!"));
}

#[tokio::test]
async fn a_blank_answer_reports_no_output() {
    let (mut session, _provider) = TestSession::answering(answers("  "));

    let output = session.prompt("Hi").await.expect("the prompt succeeds");

    assert_eq!(output, None);
}

#[tokio::test]
async fn inputs_are_processed_one_after_another() {
    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(text_response("First")),
        ScriptedCall::response(text_response("Second")),
    ]);

    session
        .prompt("one")
        .await
        .expect("the first prompt succeeds");
    assert_eq!(session.state(), CodingAgentState::Idle);
    session
        .prompt("two")
        .await
        .expect("the second prompt succeeds");
    assert_eq!(session.state(), CodingAgentState::Idle);

    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 4);
    assert!(matches!(&turns[0], Message::User { content, .. } if content == "one"));
    assert!(matches!(&turns[1], Message::Assistant { content, .. } if content == "First"));
    assert!(matches!(&turns[2], Message::User { content, .. } if content == "two"));
    assert!(matches!(&turns[3], Message::Assistant { content, .. } if content == "Second"));
}

#[tokio::test]
async fn a_follow_up_starts_another_cycle() {
    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(text_response("First response")),
        ScriptedCall::response(text_response("Followup response")),
    ]);
    session.follow_up("followup message");

    session
        .prompt("initial message")
        .await
        .expect("the prompt succeeds");

    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 4);
    assert!(matches!(&turns[0], Message::User { content, .. } if content == "initial message"));
    assert!(matches!(&turns[1], Message::Assistant { content, .. } if content == "First response"));
    assert!(matches!(&turns[2], Message::User { content, .. } if content == "followup message"));
    assert!(
        matches!(&turns[3], Message::Assistant { content, .. } if content == "Followup response")
    );
}

// --- Invariant 1: every tool call gets a result ---

#[tokio::test]
async fn a_tool_round_pairs_the_call_with_its_result() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_1",
            json!({"text": "hello"}),
        )),
        ScriptedCall::response(text_response("Done!")),
    ])
    .tools([echo_tool()])
    .build();

    session
        .prompt("Use echo tool")
        .await
        .expect("the prompt succeeds");

    assert_eq!(session.state(), CodingAgentState::Idle);
    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 4, "input, call, result, answer");
    assert!(matches!(&turns[0], Message::User { .. }));
    assert!(matches!(&turns[1], Message::Assistant { tool_calls, .. } if tool_calls.len() == 1));
    assert!(matches!(&turns[3], Message::Assistant { content, .. } if content == "Done!"));

    let results = tool_results(&session, 2);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool_call_id, "call_1");
    assert!(!results[0].is_error);
}

#[tokio::test]
async fn every_call_of_a_parallel_round_comes_back() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(multi_tool_call_response(vec![
            ("echo", "call_1", json!({"text": "first"})),
            ("echo", "call_2", json!({"text": "second"})),
            ("echo", "call_3", json!({"text": "third"})),
        ])),
        ScriptedCall::response(text_response("All done!")),
    ])
    .tools([echo_tool()])
    .build();
    let mut events = session.subscribe();

    session
        .prompt("Use echo three times")
        .await
        .expect("the prompt succeeds");

    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 4);
    let results = tool_results(&session, 2);
    assert_eq!(
        results
            .iter()
            .map(|result| result.tool_call_id.as_str())
            .collect::<Vec<_>>(),
        ["call_1", "call_2", "call_3"],
        "results come back in call order"
    );
    assert!(results.iter().all(|result| !result.is_error));

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::ToolCallStarted { .. }
        )),
        3
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::ToolCallCompleted { .. }
        )),
        3
    );
}

#[tokio::test]
async fn an_unknown_tool_still_answers_its_call() {
    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(tool_call_response("nonexistent_tool", "call_1", json!({}))),
        ScriptedCall::response(text_response("OK")),
    ]);

    session
        .prompt("Do something")
        .await
        .expect("the prompt succeeds");

    assert_eq!(session.history().turns().len(), 4);
    let results = tool_results(&session, 2);
    assert!(results[0].is_error);
    assert_eq!(result_text(&results[0]), "Unknown tool: nonexistent_tool");
}

#[tokio::test]
async fn a_tool_that_fails_still_answers_its_call() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("fail_tool", "call_1", json!({}))),
        ScriptedCall::response(text_response("OK")),
    ])
    .tools([failing_tool()])
    .build();

    session
        .prompt("Use fail tool")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert!(results[0].is_error);
    assert_eq!(result_text(&results[0]), "tool execution failed");
}

#[tokio::test]
async fn arguments_that_miss_the_schema_answer_with_a_validation_error() {
    let strict = RegisteredTool::new(
        ToolDefinition::function(
            "strict_tool",
            "Tool with required params",
            json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }),
        ),
        Arc::new(|_arguments, _context| Box::pin(async { Ok("should not reach".to_owned()) })),
    )
    .with_source(ToolSource::Native);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("strict_tool", "call_1", json!({}))),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([strict])
    .build();

    session
        .prompt("Use strict tool")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert!(results[0].is_error);
    let text = result_text(&results[0]);
    assert!(
        text.contains("text") && text.contains("required"),
        "the validation error names the missing property: {text}"
    );
}

#[tokio::test]
async fn arguments_that_match_the_schema_reach_the_tool() {
    let strict = RegisteredTool::new(
        ToolDefinition::function(
            "strict_tool",
            "Tool with required params",
            json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }),
        ),
        Arc::new(|_arguments, _context| Box::pin(async { Ok("tool executed".to_owned()) })),
    )
    .with_source(ToolSource::Native);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "strict_tool",
            "call_1",
            json!({"text": "hello"}),
        )),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([strict])
    .build();

    session
        .prompt("Use strict tool")
        .await
        .expect("the prompt succeeds");

    assert!(!tool_results(&session, 2)[0].is_error);
}

#[tokio::test]
async fn a_completed_call_reports_the_output_the_model_read() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_1",
            json!({"text": "hello world"}),
        )),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([echo_tool()])
    .build();
    let mut events = session.subscribe();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let completions: Vec<&CodingEvent> = published
        .iter()
        .filter(|event| matches!(event, CodingEvent::ToolCallCompleted { .. }))
        .collect();
    assert_eq!(completions.len(), 1);
    assert!(matches!(
        completions[0],
        CodingEvent::ToolCallCompleted { output, .. } if *output == json!("echo: hello world")
    ));
}

#[tokio::test]
async fn a_tool_that_ends_the_prompt_still_has_its_result_committed() {
    // The tool ends the whole prompt — the terminal gesture, not the round
    // interrupt — and learns the session's token once the session exists.
    let token: Arc<OnceLock<CancellationToken>> = Arc::new(OnceLock::new());
    let held = Arc::clone(&token);
    let stopping = RegisteredTool::new(
        ToolDefinition::function("set_abort", "Ends the prompt", json!({"type": "object"})),
        Arc::new(move |_arguments, _context| {
            let held = Arc::clone(&held);
            Box::pin(async move {
                held.get().expect("the session was built").cancel();
                Ok("done".to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("set_abort", "call_1", json!({}))),
        ScriptedCall::response(text_response("Should not reach this")),
    ])
    .options(CodingAgentOptions {
        enable_loop_detection: false,
        ..CodingAgentOptions::default()
    })
    .tools([stopping])
    .build();
    token
        .set(session.cancel_token())
        .expect("the token is set once");

    let error = session
        .prompt("Do something")
        .await
        .expect_err("the prompt ended");

    assert!(matches!(error, Error::Interrupted(_)));
    assert_eq!(session.state(), CodingAgentState::Closed);
    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 3, "input, call, and the call's result");
    assert!(matches!(&turns[1], Message::Assistant { tool_calls, .. } if tool_calls.len() == 1));
    assert!(matches!(&turns[2], Message::ToolResults { .. }));
}

// --- Loop detection ---

#[tokio::test]
async fn repeating_the_same_call_warns_the_model() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_1",
            json!({"text": "same"}),
        )),
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_2",
            json!({"text": "same"}),
        )),
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_3",
            json!({"text": "same"}),
        )),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([echo_tool()])
    .options(CodingAgentOptions {
        enable_loop_detection: true,
        loop_detection_window: 3,
        ..CodingAgentOptions::default()
    })
    .build();
    let mut events = session.subscribe();

    session
        .prompt("Keep echoing")
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::LoopDetected))
    );
    assert!(
        session.history().turns().iter().any(|turn| {
            matches!(turn, Message::Steering { content, .. } if content.contains("Loop detected"))
        }),
        "the warning is written into the conversation"
    );
}

// --- Accounting ---

#[tokio::test]
async fn a_prompt_sums_the_cost_of_every_response() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(with_cost(
            tool_call_response("echo", "call_1", json!({"text": "hello"})),
            40_000,
        )),
        ScriptedCall::response(with_cost(text_response("Done!"), 60_000)),
    ])
    .tools([echo_tool()])
    .build();

    session
        .prompt("Use echo tool")
        .await
        .expect("the prompt succeeds");

    assert_eq!(session.last_prompt_cost_usd_micros(), Some(100_000));
}

#[tokio::test]
async fn a_prompt_reports_where_it_spent_its_time() {
    let slow_tool = RegisteredTool::new(
        ToolDefinition::function(
            "slow_tool",
            "Sleeps before returning",
            json!({"type": "object"}),
        ),
        Arc::new(|_arguments, _context| {
            Box::pin(async {
                sleep(Duration::from_millis(30)).await;
                Ok("slept".to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("slow_tool", "call_1", json!({}))),
        ScriptedCall::response(text_response("Done!")),
        ScriptedCall::response(text_response("Second response")),
    ])
    .tools([slow_tool])
    .delayed(Duration::from_millis(20))
    .build();

    session
        .prompt("use the slow tool")
        .await
        .expect("the prompt succeeds");

    let first = session.last_prompt_timing();
    assert!(
        first.inference >= Duration::from_millis(35),
        "both of the prompt's two model calls are counted: {first:?}"
    );
    assert!(
        first.tool >= Duration::from_millis(30),
        "the tool's own time is counted apart from them: {first:?}"
    );

    session
        .prompt("no tools this time")
        .await
        .expect("the prompt succeeds");

    let second = session.last_prompt_timing();
    assert!(
        second.inference >= Duration::from_millis(15),
        "each prompt is timed on its own: {second:?}"
    );
    assert_eq!(
        second.tool,
        Duration::ZERO,
        "a prompt with no tools spends no tool time"
    );
}

#[tokio::test]
async fn every_tool_round_resolves_the_environment_again() {
    struct SequenceEnvProvider {
        values: Mutex<VecDeque<HashMap<String, String>>>,
    }

    #[async_trait::async_trait]
    impl ToolEnvProvider for SequenceEnvProvider {
        async fn resolve(&self) -> StdResult<HashMap<String, String>, ToolError> {
            self.values
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop_front()
                .ok_or_else(|| ToolError::unavailable("the env script ran out"))
        }
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    let record_env = RegisteredTool::new(
        ToolDefinition::function(
            "record_env",
            "Records resolved env",
            json!({"type": "object"}),
        ),
        Arc::new(move |_arguments, context| {
            let seen = Arc::clone(&recorder);
            Box::pin(async move {
                let env = context.resolve_tool_env().await?.unwrap_or_default();
                seen.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(env.get("TOKEN").cloned().unwrap_or_default());
                Ok("recorded".to_owned())
            })
        }),
    )
    .with_source(ToolSource::Native);

    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("record_env", "call_1", json!({}))),
        ScriptedCall::response(tool_call_response("record_env", "call_2", json!({}))),
        ScriptedCall::response(text_response("Done!")),
    ])
    .tools([record_env])
    .build();
    session.set_tool_env_provider(Arc::new(SequenceEnvProvider {
        values: Mutex::new(VecDeque::from([
            HashMap::from([("TOKEN".to_owned(), "t1".to_owned())]),
            HashMap::from([("TOKEN".to_owned(), "t2".to_owned())]),
        ])),
    }));

    session
        .prompt("Use tools")
        .await
        .expect("the prompt succeeds");

    assert_eq!(
        seen.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_slice(),
        ["t1".to_owned(), "t2".to_owned()],
        "each round asked for its own environment"
    );
}

// --- Ending a session ---

#[tokio::test]
async fn a_cancelled_session_never_calls_the_model() {
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("echo", "call_1", json!({"text": "a"}))),
        ScriptedCall::response(tool_call_response("echo", "call_2", json!({"text": "b"}))),
    ])
    .tools([echo_tool()])
    .options(CodingAgentOptions {
        enable_loop_detection: false,
        ..CodingAgentOptions::default()
    })
    .build();
    session.interrupt();

    let error = session
        .prompt("Do something")
        .await
        .expect_err("the prompt ended");

    assert!(matches!(error, Error::Interrupted(_)));
    assert_eq!(session.state(), CodingAgentState::Closed);
    assert_eq!(provider.call_count(), 0);
    let turns = session.history().turns().to_vec();
    assert_eq!(turns.len(), 1, "only the input was recorded");
    assert!(matches!(&turns[0], Message::User { .. }));
}

#[tokio::test]
async fn a_credential_failure_closes_the_session() {
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::Failure(
        ScriptedFailure::terminal(LlmErrorKind::Authentication, "invalid api key"),
    )]);

    let error = session.prompt("Hello").await.expect_err("the call failed");

    assert!(matches!(error, Error::Llm(_)));
    assert_eq!(session.state(), CodingAgentState::Closed);
}

#[tokio::test]
async fn a_closed_session_announces_no_new_start() {
    let (mut session, _provider) = TestSession::answering(answers("done"));
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
    let mut events = session.subscribe();

    let error = session
        .prompt("Hello")
        .await
        .expect_err("the session ended");

    assert!(matches!(error, Error::SessionClosed));
    assert!(
        drained(&mut events).await.is_empty(),
        "a closed session publishes nothing"
    );
}

#[tokio::test]
async fn returning_to_idle_ends_the_processing_cycle() {
    let (mut session, _provider) = TestSession::answering(answers("Hello"));
    session.initialize().await.expect("initialization succeeds");
    let mut events = session.subscribe();

    session.prompt("Hi").await.expect("the prompt succeeds");

    assert_eq!(session.state(), CodingAgentState::Idle);
    let published = drained(&mut events).await;
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::ProcessingEnd))
    );
}

#[tokio::test]
async fn a_session_starts_and_ends_once_however_many_inputs_it_answers() {
    let (mut session, _provider) = TestSession::answering(vec![
        ScriptedCall::response(text_response("First")),
        ScriptedCall::response(text_response("Second")),
    ]);
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");
    session
        .prompt("one")
        .await
        .expect("the first prompt succeeds");
    session
        .prompt("two")
        .await
        .expect("the second prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::SessionStarted { .. }
        )),
        1
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::SessionEnded
        )),
        1
    );
}

// --- What a prompt publishes ---

#[tokio::test]
async fn a_prompt_publishes_its_input_its_answer_and_the_window_it_used() {
    let (mut session, _provider) = TestSession::answering(answers("Hello"));
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");
    session.prompt("Hi").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::SessionStarted { .. }))
    );
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::UserInput { .. }))
    );
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::SessionEnded))
    );

    let snapshot = published
        .iter()
        .find_map(|event| match event {
            CodingEvent::AssistantMessage { context_window, .. } => context_window.as_ref(),
            _ => None,
        })
        .expect("the assistant turn carries a context window");
    assert_eq!(
        snapshot.count_method,
        ContextWindowCountMethod::ResponseUsageScaledBreakdown,
        "a response that reported usage is measured from it"
    );
}

#[tokio::test]
async fn a_response_that_reports_no_usage_is_measured_locally() {
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::response(
        with_usage(text_response("Hello"), TokenCounts::default()),
    )]);
    let mut events = session.subscribe();

    session.prompt("Hi").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let snapshot = published
        .iter()
        .find_map(|event| match event {
            CodingEvent::AssistantMessage { context_window, .. } => context_window.as_ref(),
            _ => None,
        })
        .expect("the assistant turn carries a context window");
    assert_eq!(
        snapshot.count_method,
        ContextWindowCountMethod::LocalEstimate
    );
    assert!(snapshot.input_tokens > 0);
}

/// A counter a test can read from inside a tool.
#[derive(Debug, Default)]
struct Counter(AtomicUsize);

impl Counter {
    fn bump(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

#[tokio::test]
async fn a_blocking_tool_is_cancelled_rather_than_dropped() {
    let runs = Arc::new(Counter::default());
    let counter = Arc::clone(&runs);
    let watcher = RegisteredTool::new(
        ToolDefinition::function(
            "watch",
            "Records that it ran, then waits",
            json!({"type": "object"}),
        ),
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
    let (mut session, _provider) = TestSession::new(vec![ScriptedCall::response(
        tool_call_response("watch", "call_1", json!({})),
    )])
    .tools([watcher])
    .build();
    let cancel = session.cancel_token();
    let mut events = session.subscribe();
    let stopper = tokio::spawn(async move {
        // Ended once the call is running, so the tool is cancelled rather than
        // never started.
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::ToolCallStarted { .. })
        })
        .await;
        cancel.cancel();
    });

    let error = session
        .prompt("watch something")
        .await
        .expect_err("the prompt was ended");
    stopper.await.expect("the stopper finishes");

    assert!(matches!(error, Error::Interrupted(_)));
    assert_eq!(runs.count(), 1, "the tool ran");
    let results = tool_results(&session, 2);
    assert_eq!(results.len(), 1, "the cancelled call still has a result");
    assert!(results[0].is_error);
}

#[tokio::test]
async fn a_blocking_tool_answers_the_round_that_was_interrupted() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("block", "call_block", json!({}))),
        ScriptedCall::response(text_response("resumed")),
    ])
    .tools([blocking_tool("block")])
    .build();
    let control = session.control_handle();
    let mut events = session.subscribe();
    let mut recorded = session.subscribe();
    let controller = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::ToolCallStarted { tool_name, .. } if tool_name == "block")
        })
        .await;
        control.interrupt();
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::RoundInterrupted { generation: 1 })
        })
        .await;
        control.steer("resume after tool", None);
    });

    session
        .prompt("use the tool")
        .await
        .expect("the prompt resumes after the steer");
    controller.await.expect("the controller finishes");

    let published = settled(&mut session, &mut recorded).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::RoundInterrupted { .. }
        )),
        1,
        "one gesture, one announcement"
    );
    let completed = position(&published, |event| {
        matches!(event, CodingEvent::ToolCallCompleted { .. })
    })
    .expect("the tool call completed");
    let settled_at = position(&published, |event| {
        matches!(event, CodingEvent::RoundInterrupted { .. })
    })
    .expect("the interrupt settled");
    assert!(
        completed < settled_at,
        "the call is answered before the interrupt settles"
    );
    assert!(matches!(
        session.history().turns().get(2),
        Some(Message::ToolResults { .. })
    ));
}
