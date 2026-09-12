//! Fallback routes, as an application configures and observes them.
//!
//! An application names the routes a prompt may continue on; pebble executes
//! that list and nothing more. When the model fails for a reason another route
//! might not share, the conversation moves as it stands: the failed session's
//! record resumes on the next route, the input it still held moves with it,
//! and the prompt continues without repeating a tool effect. The stream says
//! so from the new route, and the report names the route the prompt ended on.

use std::sync::Arc;

use lithos_llm::types::{ErrorKind as LlmErrorKind, Message as LlmMessage, ReasoningEffort, Role};
use pebble_coding_agent::environment::Environment;
use pebble_coding_agent::events::{
    CodingAgentEvent, CodingAgentState, CodingEvent, FailoverContinuation, FailoverStop,
    PermissionLevel, TokenUsage,
};
use pebble_coding_agent::state::Message;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedFailure, ScriptedProvider, client_from, text_response,
    tool_call_response, with_cost,
};
use pebble_coding_agent::{
    CodingAgent, CodingAgentOptions, Error, FallbackRoute, ResumeMode, ShutdownReason,
};
use serde_json::json;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

/// The route every test starts on.
const PRIMARY: &str = "test/model";

/// Another model on the same provider, which the test catalog gives a profile,
/// so it is a route a session can be built on.
const FALLBACK: &str = "test/vision";

/// A model failure another route might not share.
fn credentials_rejected() -> ScriptedCall {
    ScriptedCall::Failure(ScriptedFailure::terminal(
        LlmErrorKind::Authentication,
        "primary key revoked",
    ))
}

/// A failure that is the request's own fault, so no route would do better.
fn bad_request() -> ScriptedCall {
    ScriptedCall::Failure(ScriptedFailure::terminal(
        LlmErrorKind::InvalidRequest,
        "malformed tool schema",
    ))
}

/// Everything the receiver holds.
fn drained(events: &mut broadcast::Receiver<CodingAgentEvent>) -> Vec<CodingAgentEvent> {
    let mut published = Vec::new();
    while let Ok(event) = events.try_recv() {
        published.push(event);
    }
    published
}

/// How many of `published` are `wanted`.
fn count(published: &[CodingAgentEvent], wanted: impl Fn(&CodingEvent) -> bool) -> usize {
    published
        .iter()
        .filter(|event| wanted(&event.event))
        .count()
}

/// The failovers in `published`, in order.
fn failovers(published: &[CodingAgentEvent]) -> Vec<&CodingAgentEvent> {
    published
        .iter()
        .filter(|event| matches!(event.event, CodingEvent::RouteFailover { .. }))
        .collect()
}

/// The failover stops in `published`, in order.
fn stops(published: &[CodingAgentEvent]) -> Vec<&CodingAgentEvent> {
    published
        .iter()
        .filter(|event| matches!(event.event, CodingEvent::RouteFailoverStopped { .. }))
        .collect()
}

/// An agent on [`PRIMARY`] whose one script serves every route in the test
/// catalog, with `routes` to fall over to, working in a mock environment the
/// test keeps a handle on.
async fn agent_with(
    calls: Vec<ScriptedCall>,
    routes: Vec<FallbackRoute>,
) -> (CodingAgent, Arc<ScriptedProvider>, Arc<MockEnvironment>) {
    let (client, provider) = client_from(ScriptedProvider::new(calls));
    let environment = Arc::new(MockEnvironment::linux());
    let agent = CodingAgent::builder(client, environment.clone() as Arc<dyn Environment>)
        .model(PRIMARY)
        .permission_level(PermissionLevel::Full)
        .options(CodingAgentOptions::default().with_loop_detection(false))
        .fallback_routes(routes)
        .build()
        .await
        .expect("the coding agent builds");
    (agent, provider, environment)
}

#[tokio::test]
async fn a_failover_eligible_error_moves_the_conversation_to_the_next_route() {
    let (mut agent, provider, environment) = agent_with(
        vec![
            ScriptedCall::response(with_cost(
                tool_call_response(
                    "write_file",
                    "call_1",
                    json!({"file_path": "/home/test/once.txt", "content": "written once"}),
                ),
                7,
            )),
            credentials_rejected(),
            ScriptedCall::response(text_response("recovered on the fallback")),
        ],
        vec![FallbackRoute::new(FALLBACK)],
    )
    .await;
    let handle = agent.control_handle();
    let mut events = agent.subscribe();

    let report = agent.prompt("write the file").await;

    let output = report.result.as_ref().expect("the fallback answers");
    assert_eq!(output.text.as_deref(), Some("recovered on the fallback"));
    assert_eq!(report.route, FALLBACK, "the report names the final route");
    assert_eq!(agent.provider(), "test");
    assert_eq!(agent.model(), "vision");
    assert_eq!(
        report.files_touched,
        ["/home/test/once.txt"],
        "the work the failed route did is on the report"
    );
    assert_eq!(
        environment
            .written_files
            .lock()
            .expect("written_files lock")
            .len(),
        1,
        "the tool effect is not repeated"
    );
    assert!(
        report.usage.input > 0,
        "accounting spans both routes: {report:?}"
    );

    // The fallback was asked on the history as it stood, with no new input.
    let requests = provider.requests();
    assert_eq!(requests.len(), 3, "one call per scripted turn");
    let roles: Vec<Role> = requests[2]
        .messages()
        .iter()
        .map(LlmMessage::role)
        .collect();
    assert_eq!(roles, [
        Role::System,
        Role::User,
        Role::Assistant,
        Role::Tool
    ]);
    assert_eq!(requests[2].messages()[3].tool_call_id(), Some("call_1"));

    // The history is one conversation, not two.
    let turns = agent.history().turns().to_vec();
    assert!(
        matches!(turns.last(), Some(Message::Assistant { .. })),
        "{turns:?}"
    );
    assert_eq!(
        turns
            .iter()
            .filter(|turn| matches!(turn, Message::ToolResults { .. }))
            .count(),
        1
    );

    // The stream: the failed session ends, the fallback starts and reports
    // the move, all on the session's one stream with no number reused.
    let published = drained(&mut events);
    let failovers = failovers(&published);
    assert_eq!(failovers.len(), 1, "{published:?}");
    let CodingEvent::RouteFailover {
        from,
        to,
        attempt,
        error,
        usage,
        cost_usd_micros,
        continuation,
        ..
    } = &failovers[0].event
    else {
        unreachable!()
    };
    assert_eq!(from, PRIMARY);
    assert_eq!(to, FALLBACK);
    assert_eq!(*attempt, 1);
    assert!(error.message.contains("primary key revoked"), "{error:?}");
    // The failed route answered once before it failed: that answer is what
    // it spent, and what the next route continues from.
    assert_eq!(
        usage.input, 10,
        "the tool-call response's tokens: {usage:?}"
    );
    assert_eq!(usage.output, 5);
    assert_eq!(*cost_usd_micros, Some(7));
    assert_eq!(*continuation, FailoverContinuation::ContinueTurn);
    assert!(
        stops(&published).is_empty(),
        "a prompt that moved has nothing to say about stopping: {published:?}"
    );
    // The subscription was taken after the first route had started, so the
    // one start it sees is the fallback's, on its own route.
    let started: Vec<&CodingAgentEvent> = published
        .iter()
        .filter(|event| matches!(event.event, CodingEvent::SessionStarted { .. }))
        .collect();
    assert_eq!(started.len(), 1, "the fallback started: {published:?}");
    assert!(
        matches!(
            &started[0].event,
            CodingEvent::SessionStarted { provider, model }
                if provider.as_deref() == Some("test") && model.as_deref() == Some("vision")
        ),
        "{:?}",
        started[0].event
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::SessionEnded
        )),
        1,
        "the failed session ended; the fallback is still open"
    );
    assert!(
        started[0].seq < failovers[0].seq,
        "the failover is reported from the route it moved to"
    );
    let mut seqs: Vec<u64> = published.iter().map(|event| event.seq).collect();
    seqs.sort_unstable();
    let before = seqs.len();
    seqs.dedup();
    assert_eq!(seqs.len(), before, "one stream, no sequence number reused");
    assert!(
        published
            .iter()
            .all(|event| event.stream_id == published[0].stream_id),
        "both routes publish on the session's one stream"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::ToolCallStarted { .. }
        )),
        1
    );

    // A handle taken before the prompt reaches the replacement.
    assert!(
        !handle.is_closed(),
        "the handle follows the agent to its new route"
    );
    assert_eq!(agent.state(), CodingAgentState::Idle);
    assert!(agent.remaining_fallback_routes().is_empty());
    let record = agent.to_record();
    assert_eq!(record.provider.as_deref(), Some("test"));
    assert_eq!(record.model.as_deref(), Some("vision"));

    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn queued_follow_ups_move_with_the_conversation() {
    let (mut agent, provider, _environment) = agent_with(
        vec![
            credentials_rejected(),
            ScriptedCall::response(text_response("recovered")),
            ScriptedCall::response(text_response("and the follow-up")),
        ],
        vec![FallbackRoute::new(FALLBACK)],
    )
    .await;
    agent.queue_follow_up("then this");
    let mut events = agent.subscribe();

    let report = agent.prompt("first").await;

    assert_eq!(
        report.result.expect("the prompt succeeds").text.as_deref(),
        Some("and the follow-up"),
        "the follow-up the failed session held ran on the new route"
    );
    assert_eq!(provider.call_count(), 3);
    // The first call failed before any token was spent, so the new route is
    // asked the prompt again and the failed route accounts for nothing.
    let published = drained(&mut events);
    let failovers = failovers(&published);
    assert_eq!(failovers.len(), 1, "{published:?}");
    let CodingEvent::RouteFailover {
        usage,
        cost_usd_micros,
        tool_ms,
        continuation,
        ..
    } = &failovers[0].event
    else {
        unreachable!()
    };
    assert_eq!(*usage, TokenUsage::default(), "{usage:?}");
    assert_eq!(*cost_usd_micros, None);
    assert_eq!(*tool_ms, 0, "no tool ran on the failed route");
    assert_eq!(*continuation, FailoverContinuation::ReplayPrompt);
    let follow_ups = agent
        .history()
        .turns()
        .iter()
        .filter(|turn| {
            matches!(turn, Message::User { content, .. } if content.text_content() == "then this")
        })
        .count();
    assert_eq!(follow_ups, 1, "{:?}", agent.history().turns());
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn an_ineligible_error_ends_the_prompt_on_its_route() {
    let (mut agent, provider, _environment) =
        agent_with(vec![bad_request()], vec![FallbackRoute::new(FALLBACK)]).await;
    let mut events = agent.subscribe();

    let report = agent.prompt("work").await;

    assert!(matches!(report.result, Err(Error::Llm(_))), "{report:?}");
    assert_eq!(report.route, PRIMARY);
    assert_eq!(provider.call_count(), 1, "no other route was asked");
    assert_eq!(
        agent.remaining_fallback_routes().len(),
        1,
        "the route is still there for a failure that deserves it"
    );
    // The stream says why the routes were not used.
    let published = drained(&mut events);
    assert!(failovers(&published).is_empty(), "{published:?}");
    let stops = stops(&published);
    assert_eq!(stops.len(), 1, "{published:?}");
    let CodingEvent::RouteFailoverStopped {
        route,
        attempt,
        reason,
        error,
    } = &stops[0].event
    else {
        unreachable!()
    };
    assert_eq!(route, PRIMARY);
    assert_eq!(*attempt, 0, "the prompt never left its first route");
    assert_eq!(*reason, FailoverStop::Ineligible);
    assert!(error.message.contains("malformed tool schema"), "{error:?}");
    let reported = published
        .iter()
        .position(|event| matches!(event.event, CodingEvent::Error { .. }))
        .expect("the failure is reported");
    let ended = published
        .iter()
        .position(|event| matches!(event.event, CodingEvent::ProcessingEnd))
        .expect("the prompt ends");
    let stopped = published
        .iter()
        .position(|event| matches!(event.event, CodingEvent::RouteFailoverStopped { .. }))
        .expect("the stop is reported");
    assert!(
        reported < stopped && stopped < ended,
        "the stop follows the failure and precedes the prompt's end: {published:?}"
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_spent_chain_reports_the_last_routes_error() {
    let (mut agent, provider, _environment) =
        agent_with(vec![credentials_rejected(), credentials_rejected()], vec![
            FallbackRoute::new(FALLBACK),
        ])
        .await;
    let mut events = agent.subscribe();

    let report = agent.prompt("work").await;

    assert!(matches!(report.result, Err(Error::Llm(_))), "{report:?}");
    assert_eq!(report.route, FALLBACK, "the prompt ended on the last route");
    assert_eq!(provider.call_count(), 2);
    let published = drained(&mut events);
    assert_eq!(failovers(&published).len(), 1, "{published:?}");
    // The last route's failure is reported as the end of the plan, on that
    // route, before the session it closed says it ended.
    let stops = stops(&published);
    assert_eq!(stops.len(), 1, "{published:?}");
    let CodingEvent::RouteFailoverStopped {
        route,
        attempt,
        reason,
        error,
    } = &stops[0].event
    else {
        unreachable!()
    };
    assert_eq!(route, FALLBACK);
    assert_eq!(*attempt, 1);
    assert_eq!(*reason, FailoverStop::Exhausted);
    assert!(error.message.contains("primary key revoked"), "{error:?}");
    let last_end = published
        .iter()
        .rev()
        .find(|event| matches!(event.event, CodingEvent::SessionEnded))
        .expect("the credential failure closed the last session");
    assert!(
        stops[0].seq < last_end.seq,
        "the stop is on the stream before the session ends: {published:?}"
    );
    assert!(agent.remaining_fallback_routes().is_empty());
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_route_that_cannot_be_built_ends_the_prompt() {
    let (mut agent, provider, _environment) =
        agent_with(vec![credentials_rejected()], vec![FallbackRoute::new(
            "nowhere/model",
        )])
        .await;

    let report = agent.prompt("work").await;

    match report.result {
        Err(Error::FallbackRoute { route, .. }) => assert_eq!(route, "nowhere/model"),
        other => panic!("expected the route to be reported unavailable, got {other:?}"),
    }
    assert_eq!(provider.call_count(), 1);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_route_carries_its_own_request_controls() {
    let (mut agent, provider, _environment) = agent_with(
        vec![
            credentials_rejected(),
            ScriptedCall::response(text_response("recovered")),
        ],
        vec![
            FallbackRoute::new("test/thinking")
                .with_reasoning_effort(Some(ReasoningEffort::High))
                .with_max_tokens(Some(4_096)),
        ],
    )
    .await;

    let report = agent.prompt("work").await;

    assert!(report.result.is_ok(), "{report:?}");
    assert_eq!(report.route, "test/thinking");
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].reasoning_effort(), None);
    assert_eq!(
        requests[1].reasoning_effort(),
        Some(ReasoningEffort::High),
        "the fallback route's controls apply to its requests"
    );
    assert_eq!(requests[1].max_output_tokens(), Some(4_096));
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn without_routes_a_model_error_ends_the_prompt() {
    let (mut agent, provider, _environment) =
        agent_with(vec![credentials_rejected()], vec![]).await;
    let mut events = agent.subscribe();

    let report = agent.prompt("work").await;

    assert!(matches!(report.result, Err(Error::Llm(_))), "{report:?}");
    assert_eq!(provider.call_count(), 1);
    assert!(agent.remaining_fallback_routes().is_empty());
    let published = drained(&mut events);
    assert!(
        failovers(&published).is_empty() && stops(&published).is_empty(),
        "with no plan there is nothing to say about routes: {published:?}"
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_cancelled_prompt_neither_moves_nor_reports_a_stop() {
    let (mut agent, provider, _environment) =
        agent_with(vec![ScriptedCall::PendingOpen], vec![FallbackRoute::new(
            FALLBACK,
        )])
        .await;
    let mut events = agent.subscribe();
    let cancel = CancellationToken::new();
    let canceller = {
        let cancel = cancel.clone();
        let provider = Arc::clone(&provider);
        tokio::spawn(async move {
            provider.wait_for_call().await;
            cancel.cancel();
        })
    };

    let report = agent.prompt_with_cancellation("work", &cancel).await;

    canceller.await.expect("the canceller finishes");
    assert!(
        matches!(report.result, Err(Error::Interrupted(_))),
        "{report:?}"
    );
    assert_eq!(report.route, PRIMARY);
    assert_eq!(
        agent.remaining_fallback_routes().len(),
        1,
        "a cancelled prompt never moves"
    );
    let published = drained(&mut events);
    assert!(
        failovers(&published).is_empty() && stops(&published).is_empty(),
        "a cancellation says nothing about routes: {published:?}"
    );
    agent
        .shutdown(ShutdownReason::Cancelled)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_prompt_resumed_mid_turn_continues_the_turn_on_the_new_route() {
    // A session whose model failed right after a tool round, stored as it
    // stood: the record ends with the tool's result.
    let (mut stored, _provider, _environment) = agent_with(
        vec![
            ScriptedCall::response(tool_call_response(
                "write_file",
                "call_1",
                json!({"file_path": "/home/test/once.txt", "content": "written once"}),
            )),
            credentials_rejected(),
        ],
        vec![],
    )
    .await;
    let failed = stored.prompt("write the file").await;
    assert!(matches!(failed.result, Err(Error::Llm(_))), "{failed:?}");
    let record = stored.to_record();
    assert!(
        matches!(
            stored.history().turns().last(),
            Some(Message::ToolResults { .. })
        ),
        "{:?}",
        stored.history().turns()
    );
    stored
        .shutdown(ShutdownReason::Error)
        .await
        .expect("the stored agent shuts down");

    // Resumed on its recorded route, which fails again before it answers.
    let (client, provider) = client_from(ScriptedProvider::new(vec![
        credentials_rejected(),
        ScriptedCall::response(text_response("finished on the fallback")),
    ]));
    let environment = Arc::new(MockEnvironment::linux());
    let mut agent = CodingAgent::resume(
        client,
        environment as Arc<dyn Environment>,
        record,
        ResumeMode::RecordedModel,
    )
    .permission_level(PermissionLevel::Full)
    .options(CodingAgentOptions::default().with_loop_detection(false))
    .fallback_routes(vec![FallbackRoute::new(FALLBACK)])
    .build()
    .await
    .expect("the record resumes");
    let mut events = agent.subscribe();

    let report = agent.continue_prompt().await;

    let output = report.result.as_ref().expect("the fallback answers");
    assert_eq!(output.text.as_deref(), Some("finished on the fallback"));
    assert_eq!(report.route, FALLBACK);
    assert_eq!(provider.call_count(), 2);
    let published = drained(&mut events);
    let failovers = failovers(&published);
    assert_eq!(failovers.len(), 1, "{published:?}");
    let CodingEvent::RouteFailover {
        usage,
        continuation,
        ..
    } = &failovers[0].event
    else {
        unreachable!()
    };
    assert_eq!(
        *usage,
        TokenUsage::default(),
        "the resumed route spent nothing before it failed: {usage:?}"
    );
    assert_eq!(
        *continuation,
        FailoverContinuation::ContinueTurn,
        "the record ends mid-turn, so the new route continues the turn even \
         though the failed route committed nothing"
    );
    let requests = provider.requests();
    assert_eq!(
        requests[1].messages().last().map(LlmMessage::role),
        Some(Role::Tool),
        "the fallback was asked on the tool result the record held"
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}
