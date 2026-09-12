//! The tool-round budget, as an application configures and observes it.
//!
//! A round is one model turn that asks for tools plus the execution of those
//! calls. The budget bounds rounds, not calls. When the model asks for tools
//! once the budget is spent, the prompt ends with a failure the application
//! can tell from every other one, the refused calls are recorded as
//! `Cancelled` without running, and the agent stays open. A workflow engine
//! that runs an agent as a hook reads that failure and falls back to its
//! default decision, the way its reference implementation does.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pebble_coding_agent::events::{CodingAgentEvent, CodingAgentState, CodingEvent};
use pebble_coding_agent::state::Message;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedProvider, client_from, multi_tool_call_response,
    text_response, tool_call_response,
};
use pebble_coding_agent::tools::RegisteredTool;
use pebble_coding_agent::{
    CodingAgent, CodingAgentOptions, Error, InterruptReason, ShutdownReason,
};
use serde_json::{Value, json};
use tokio::sync::broadcast;

/// A tool that records how many times it ran.
fn counting_tool(executions: Arc<AtomicUsize>) -> RegisteredTool {
    RegisteredTool::function(
        "count",
        "Counts its executions",
        json!({"type": "object"}),
        move |_context, _arguments| {
            let executions = Arc::clone(&executions);
            async move {
                executions.fetch_add(1, Ordering::SeqCst);
                Ok("counted".to_owned())
            }
        },
    )
}

/// A turn that asks for the counting tool once.
fn asks_for_a_tool() -> ScriptedCall {
    ScriptedCall::response(tool_call_response("count", "call", json!({})))
}

/// A turn that asks for the counting tool `calls` times at once.
fn asks_for_tools(calls: usize) -> ScriptedCall {
    let ids: Vec<String> = (0..calls).map(|index| format!("call_{index}")).collect();
    let calls: Vec<(&str, &str, Value)> = ids
        .iter()
        .map(|id| ("count", id.as_str(), json!({})))
        .collect();
    ScriptedCall::response(multi_tool_call_response(calls))
}

/// A turn that answers `text`.
fn answers(text: &str) -> ScriptedCall {
    ScriptedCall::response(text_response(text))
}

/// An agent on the counting tool whose model follows `calls`, with the
/// provider handle and the tool's counter.
///
/// Loop detection is off: a model that repeats one tool call is exactly what
/// these tests script, and the warning it would inject is another test's
/// subject.
async fn agent_with(
    calls: Vec<ScriptedCall>,
    options: CodingAgentOptions,
) -> (CodingAgent, Arc<ScriptedProvider>, Arc<AtomicUsize>) {
    let (client, provider) = client_from(ScriptedProvider::new(calls));
    let executions = Arc::new(AtomicUsize::new(0));
    let agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .tools([counting_tool(Arc::clone(&executions))])
        .options(options.with_loop_detection(false))
        .build()
        .await
        .expect("the coding agent builds");
    (agent, provider, executions)
}

/// Everything the receiver holds.
fn drained(events: &mut broadcast::Receiver<CodingAgentEvent>) -> Vec<CodingEvent> {
    let mut published = Vec::new();
    while let Ok(event) = events.try_recv() {
        published.push(event.event);
    }
    published
}

/// How many of `published` match `wanted`.
fn count(published: &[CodingEvent], wanted: impl Fn(&CodingEvent) -> bool) -> usize {
    published.iter().filter(|event| wanted(event)).count()
}

/// Runs a prompt whose model never stops asking for tools under `limit`, and
/// checks the shape the budget leaves behind: `limit` rounds ran, the refused
/// turn is committed with its call answered as `Cancelled`, the event is
/// published once, and the agent is idle.
async fn assert_exhausted_after(limit: usize) {
    let (mut agent, provider, executions) = agent_with(
        vec![asks_for_a_tool()],
        CodingAgentOptions::default().with_max_tool_rounds(limit),
    )
    .await;
    let mut events = agent.subscribe();

    let report = agent.prompt("work").await;

    assert!(
        matches!(report.result, Err(Error::ToolRoundsExhausted { limit: seen }) if seen == limit),
        "unexpected report under a limit of {limit}: {report:?}"
    );
    assert_eq!(
        provider.call_count(),
        limit + 1,
        "one turn for each round that ran, and the refused one"
    );
    assert_eq!(executions.load(Ordering::SeqCst), limit);
    assert!(report.usage.input > 0, "accounting survives: {report:?}");

    let turns = agent.history().turns().to_vec();
    assert_eq!(
        turns.len(),
        1 + 2 * (limit + 1),
        "the input, then a turn and its results for every round and the refused one: {turns:?}"
    );
    match turns.last() {
        Some(Message::ToolResults { results, .. }) => {
            assert_eq!(results.len(), 1);
            assert!(
                results[0].is_error,
                "the refused call is answered as cancelled: {results:?}"
            );
        }
        other => panic!("the refused turn should end with its results, found {other:?}"),
    }
    assert_eq!(agent.state(), CodingAgentState::Idle);

    let published = drained(&mut events);
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::ToolRoundsExhausted { limit: seen } if *seen == limit
        )),
        1,
        "{published:?}"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::ToolCallStarted { .. }
        )),
        limit,
        "the refused call never starts: {published:?}"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::ProcessingEnd
        )),
        1,
        "the prompt still ends at its barrier: {published:?}"
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_zero_round_budget_refuses_the_first_tool_turn() {
    assert_exhausted_after(0).await;
}

#[tokio::test]
async fn one_tool_round_runs_before_the_budget_ends_the_prompt() {
    assert_exhausted_after(1).await;
}

#[tokio::test]
async fn several_tool_rounds_run_before_the_budget_ends_the_prompt() {
    assert_exhausted_after(3).await;
}

#[tokio::test]
async fn a_round_of_several_calls_spends_one_round() {
    let (mut agent, provider, executions) = agent_with(
        vec![asks_for_tools(3)],
        CodingAgentOptions::default().with_max_tool_rounds(1),
    )
    .await;

    let report = agent.prompt("work").await;

    assert!(
        matches!(report.result, Err(Error::ToolRoundsExhausted { limit: 1 })),
        "{report:?}"
    );
    assert_eq!(
        executions.load(Ordering::SeqCst),
        3,
        "every call of the one round ran"
    );
    assert_eq!(provider.call_count(), 2);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn an_answer_within_the_budget_completes_the_prompt() {
    let (mut agent, provider, executions) = agent_with(
        vec![asks_for_a_tool(), asks_for_a_tool(), answers("done")],
        CodingAgentOptions::default().with_max_tool_rounds(2),
    )
    .await;
    let mut events = agent.subscribe();

    let report = agent.prompt("work").await;

    let output = report.result.expect("the prompt fits its budget");
    assert_eq!(output.text.as_deref(), Some("done"));
    assert_eq!(provider.call_count(), 3);
    assert_eq!(executions.load(Ordering::SeqCst), 2);
    assert_eq!(
        count(&drained(&mut events), |event| matches!(
            event,
            CodingEvent::ToolRoundsExhausted { .. }
        )),
        0
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn an_exhausted_prompt_leaves_the_agent_open_for_the_next() {
    let (mut agent, _provider, executions) = agent_with(
        vec![asks_for_a_tool(), asks_for_a_tool(), answers("again")],
        CodingAgentOptions::default().with_max_tool_rounds(0),
    )
    .await;

    let first = agent.prompt("work").await;
    let second = agent.prompt("work").await;

    assert!(
        matches!(first.result, Err(Error::ToolRoundsExhausted { limit: 0 })),
        "{first:?}"
    );
    // The budget started over: the second prompt's tool turn is refused too,
    // which leaves the third scripted turn for a prompt that fits.
    assert!(
        matches!(second.result, Err(Error::ToolRoundsExhausted { limit: 0 })),
        "{second:?}"
    );
    let third = agent.prompt("work").await;
    assert_eq!(
        third
            .result
            .expect("a text answer fits any budget")
            .text
            .as_deref(),
        Some("again")
    );
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn without_a_budget_tool_rounds_are_unlimited() {
    let mut calls: Vec<ScriptedCall> = (0..8).map(|_| asks_for_a_tool()).collect();
    calls.push(answers("done"));
    let (mut agent, provider, executions) = agent_with(calls, CodingAgentOptions::default()).await;
    let mut events = agent.subscribe();

    let report = agent.prompt("work").await;

    assert_eq!(
        report.result.expect("the prompt succeeds").text.as_deref(),
        Some("done")
    );
    assert_eq!(provider.call_count(), 9);
    assert_eq!(executions.load(Ordering::SeqCst), 8);
    assert_eq!(
        count(&drained(&mut events), |event| matches!(
            event,
            CodingEvent::ToolRoundsExhausted { .. }
        )),
        0
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

/// The two budgets count different things and are enforced independently:
/// `max_tool_rounds` refuses a turn that asks for tools past its limit, while
/// `max_turns` ends the prompt before the model is asked again past its own.
/// Whichever is reached first ends the prompt, with its own error.
#[tokio::test]
async fn both_budgets_apply_and_the_tighter_one_ends_the_prompt() {
    // Two rounds fit the round budget; the turn budget stops the third ask.
    let (mut agent, provider, executions) = agent_with(
        vec![asks_for_a_tool(), asks_for_a_tool(), asks_for_a_tool()],
        CodingAgentOptions::default()
            .with_max_tool_rounds(5)
            .with_max_turns(2),
    )
    .await;
    let report = agent.prompt("work").await;
    assert!(
        matches!(
            report.result,
            Err(Error::Interrupted(InterruptReason::TurnLimit))
        ),
        "the turn budget ends the prompt first: {report:?}"
    );
    assert_eq!(
        provider.call_count(),
        2,
        "two turns ran, the third was never asked"
    );
    assert_eq!(executions.load(Ordering::SeqCst), 2);
    assert_eq!(agent.state(), CodingAgentState::Idle);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");

    // One round fits the round budget; the second ask is refused before the
    // turn budget is anywhere near spent.
    let (mut agent, provider, executions) = agent_with(
        vec![asks_for_a_tool(), asks_for_a_tool()],
        CodingAgentOptions::default()
            .with_max_tool_rounds(1)
            .with_max_turns(5),
    )
    .await;
    let report = agent.prompt("work").await;
    assert!(
        matches!(report.result, Err(Error::ToolRoundsExhausted { limit: 1 })),
        "the round budget ends the prompt first: {report:?}"
    );
    assert_eq!(
        provider.call_count(),
        2,
        "the refused turn was asked, then refused"
    );
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(agent.state(), CodingAgentState::Idle);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}
