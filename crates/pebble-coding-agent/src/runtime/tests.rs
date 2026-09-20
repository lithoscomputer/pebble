//! The runtime's own tests: building, initializing, prompting, shutting
//! down, and storing a session.

use std::result::Result as StdResult;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Notify;
use tokio::sync::broadcast::error::RecvError;
use tokio::task::yield_now;
use tokio::time::timeout;

use super::testing::{TestProfile, TestSession, builder, event_names, settled};
use super::*;
use crate::error::ErrorKind;
use crate::event::{EventSink, EventSinkError};
use crate::record::SESSION_RECORD_FORMAT_VERSION;
use crate::test_support::{
    MockEnvironment, ScriptedCall, ScriptedFailure, scripted_client, text_delta_events,
    text_response,
};
use crate::types::Message;

/// A client that answers every round with the same text.
fn client() -> Client {
    let (client, _provider) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
    client
}

/// A session that answers every round with the same text.
fn session() -> CodingRuntime {
    let (session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(text_response("done"))]);
    session
}

// --- Building ---

#[tokio::test]
async fn a_session_needs_a_model() {
    let error = CodingRuntime::builder(client())
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .expect_err("no model was named");

    assert!(matches!(error, CodingAgentBuildError::MissingModel));
}

#[tokio::test]
async fn a_session_needs_an_environment() {
    let error = CodingRuntime::builder(client())
        .model("test/model")
        .build()
        .expect_err("no environment was given");

    assert!(matches!(error, CodingAgentBuildError::MissingEnvironment));
}

#[tokio::test]
async fn a_selector_that_names_nothing_is_refused() {
    let error = CodingRuntime::builder(client())
        .model("no-such-model")
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .expect_err("the selector names nothing");

    assert!(matches!(
        error,
        CodingAgentBuildError::ModelSelection { ref selector, .. } if selector == "no-such-model"
    ));
}

#[tokio::test]
async fn a_blank_selector_is_refused() {
    let error = CodingRuntime::builder(client())
        .model("   ")
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .expect_err("a blank selector names nothing");

    assert!(matches!(error, CodingAgentBuildError::Selector { .. }));
}

#[tokio::test]
async fn a_model_that_names_no_profile_is_refused() {
    let error = CodingRuntime::builder(client())
        .model("bare/plain")
        .environment(Arc::new(MockEnvironment::linux()))
        .with_profile(TestProfile::shared())
        .build()
        .expect_err("nothing names a harness");

    assert!(matches!(
        error,
        CodingAgentBuildError::MissingProfileMetadata { ref model } if model == "bare/plain"
    ));
}

#[tokio::test]
async fn a_profile_pebble_does_not_know_is_refused() {
    let error = CodingRuntime::builder(client())
        .model("test/strange")
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .expect_err("`nonesuch` is not a pebble profile");

    assert!(matches!(
        error,
        CodingAgentBuildError::UnknownProfile { ref profile, .. } if profile == "nonesuch"
    ));
}

#[tokio::test]
async fn a_models_profile_beats_its_providers() {
    let resolved = |selector: &str| {
        CodingRuntime::builder(client())
            .model(selector)
            .environment(Arc::new(MockEnvironment::linux()))
            .build()
            .expect("the session builds")
            .profile_kind()
    };

    // The model row names `anthropic`; its provider row names `openai`.
    assert_eq!(resolved("test/model"), AgentProfileKind::Anthropic);
    // This row names nothing, so the provider's answer stands.
    assert_eq!(resolved("test/inherited"), AgentProfileKind::OpenAi);
}

#[tokio::test]
async fn a_built_session_pins_what_it_resolved() {
    let session = session();

    assert_eq!(session.provider(), "test");
    assert_eq!(session.model(), "model");
    assert_eq!(session.model_context.model_selector, "test/model");
    assert_eq!(session.profile_kind(), AgentProfileKind::Anthropic);
    assert_eq!(session.model_facts().context_window_tokens, 200_000);
    assert_eq!(session.state(), CodingAgentState::Idle);
    assert!(session.id().starts_with("ses_"));
    assert_eq!(session.session().root_session_id().as_str(), session.id());
}

#[tokio::test]
async fn the_catalog_says_which_models_reason_without_being_asked() {
    let facts = |selector: &str| {
        CodingRuntime::builder(client())
            .model(selector)
            .environment(Arc::new(MockEnvironment::linux()))
            .with_profile(TestProfile::shared())
            .build()
            .expect("the session builds")
            .model_facts()
            .reasons_by_default
    };

    assert!(!facts("test/model"), "a model that cannot reason");
    assert!(facts("test/thinking"), "a model that takes an effort level");
    assert!(
        facts("test/always-thinking"),
        "a row that says so itself, where the capabilities cannot"
    );
}

// --- Initializing ---

#[tokio::test]
async fn initializing_reports_what_it_loaded_and_where_it_is_working() {
    let mut session = session();
    let mut events = session.subscribe();
    let env_context = session
        .build_env_context(&CancellationToken::new())
        .await
        .expect("the environment is described");

    session.initialize().await.expect("initialization succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(event_names(&published), [
        "started", "memory", "skills", "ended"
    ]);
    assert!(matches!(
        &published[1],
        CodingEvent::MemoryLoaded {
            files,
            total_loaded_bytes: 0,
            budget_bytes: ProjectMemory::BUDGET_BYTES,
            ..
        } if files.is_empty()
    ));
    assert!(
        session
            .resources
            .system_prompt
            .contains("test assistant working in /home/test")
    );
    assert_eq!(env_context.knowledge_cutoff, "May 2026");
    assert_eq!(env_context.model, "model");
    assert_eq!(env_context.platform, "linux");
    assert_eq!(
        env_context.current_date.len(),
        10,
        "an environment that cannot date itself still dates the prompt"
    );
}

#[tokio::test]
async fn initializing_a_cancelled_session_stops() {
    let mut session = session();
    session.interrupt();

    let error = session
        .initialize()
        .await
        .expect_err("a cancelled session initializes nothing");

    assert!(matches!(
        error,
        Error::Interrupted(InterruptReason::Cancelled)
    ));
}

// --- Running ---

#[tokio::test]
async fn a_text_answer_ends_the_prompt() {
    let (mut session, _provider) =
        TestSession::answering(vec![ScriptedCall::response(text_response("all done"))]);
    let mut events = session.subscribe();
    session.initialize().await.expect("initialization succeeds");

    let answer = session
        .prompt("fix the test")
        .await
        .expect("the prompt succeeds");

    assert_eq!(answer.as_deref(), Some("all done"));
    assert_eq!(session.history().len(), 2, "the input and the answer");
    assert_eq!(session.state(), CodingAgentState::Idle);
    assert_eq!(event_names(&settled(&mut session, &mut events).await), [
        "started",
        "memory",
        "skills",
        "input",
        "request",
        "first_output",
        "delta",
        "message",
        "processing_end",
        "ended",
    ]);
}

#[tokio::test]
async fn an_interrupt_is_announced_once_and_its_steer_follows_it() {
    let (mut session, provider) = TestSession::answering(vec![
        ScriptedCall::PendingOpen,
        ScriptedCall::response(text_response("done")),
    ]);
    let mut events = session.subscribe();
    // Two gestures against one hanging round: one round to settle, two
    // generations to announce.
    let handle = session.control_handle();
    let controller = tokio::spawn(async move {
        provider.wait_for_call().await;
        assert!(handle.interrupt());
        handle.steer("do this instead");
    });

    timeout(Duration::from_secs(5), session.prompt("do a thing"))
        .await
        .expect("the steer unblocks the hanging call")
        .expect("the prompt succeeds");
    controller.await.expect("the controller finishes");

    let published = settled(&mut session, &mut events).await;
    let interrupts: Vec<u64> = published
        .iter()
        .filter_map(|event| match event {
            CodingEvent::RoundInterrupted { generation } => Some(*generation),
            _ => None,
        })
        .collect();
    assert_eq!(interrupts, [1, 2], "one announcement per gesture");
    let position = |matcher: fn(&CodingEvent) -> bool| {
        published
            .iter()
            .position(&matcher)
            .expect("the event was published")
    };
    assert!(
        position(|event| matches!(event, CodingEvent::RoundInterrupted { generation: 2 }))
            < position(|event| matches!(event, CodingEvent::SteeringInjected { .. })),
        "the interrupt settles before its steer is delivered"
    );
    assert!(matches!(
        session.history().turns()[1],
        Message::Steering { .. }
    ));
}

#[tokio::test]
async fn a_closed_session_refuses_input() {
    let mut session = session();
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");

    let error = session
        .prompt("anything")
        .await
        .expect_err("the session ended");

    assert!(matches!(error, Error::SessionClosed));
}

// --- Shutting down ---

#[tokio::test]
async fn only_the_first_shutdown_does_anything() {
    let mut session = session();
    let mut events = session.subscribe();

    assert!(
        session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the shutdown succeeds")
    );
    assert!(
        !session
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("a second shutdown does nothing")
    );

    let mut published = Vec::new();
    while let Ok(event) = events.try_recv() {
        published.push(event.event);
    }
    assert_eq!(
        published
            .iter()
            .filter(|event| matches!(event, CodingEvent::SessionEnded))
            .count(),
        1
    );
    assert_eq!(session.state(), CodingAgentState::Closed);
}

#[tokio::test]
async fn shutting_down_ends_the_streams_the_session_handed_out() {
    let mut session = session();
    // The renderer an application writes: read the stream until it ends,
    // then report. Nothing tells it to stop except the stream itself.
    let mut events = session.subscribe();
    let renderer = tokio::spawn(async move {
        let mut seen = 0_usize;
        loop {
            match events.recv().await {
                Ok(_) => seen += 1,
                Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => break,
            }
        }
        seen
    });
    session
        .prompt("do a thing")
        .await
        .expect("the prompt succeeds");

    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");

    let seen = timeout(Duration::from_secs(5), renderer)
        .await
        .expect("the stream ends when the session is shut down, not when it is dropped")
        .expect("the renderer finishes");
    assert!(seen > 0, "the renderer read the prompt it was watching");
    // Read after the join on purpose: the session is still alive here,
    // which is the order an application works in — wait for the renderer,
    // then let the session go.
    assert_eq!(session.state(), CodingAgentState::Closed);
    assert!(
        matches!(session.subscribe().recv().await, Err(RecvError::Closed)),
        "subscribing to a closed session answers with a stream that has ended"
    );
}

/// A sink that records nothing, so the pump stops on the first event.
struct RefusingSink;

#[async_trait]
impl EventSink for RefusingSink {
    async fn record(&self, _event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
        Err(EventSinkError::new("the disk is full"))
    }
}

/// Holds the user event until a test lets the next model request begin.
#[derive(Debug, Default)]
struct UserInputGate {
    reached: Notify,
    release: Notify,
}

#[async_trait]
impl EventSink for UserInputGate {
    async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
        if matches!(event.event, CodingEvent::UserInput { .. }) {
            self.reached.notify_one();
            self.release.notified().await;
        }
        Ok(())
    }
}

/// Accepts setup events, then breaks while a model stream is still open.
struct RefuseTextDeltaSink;

#[async_trait]
impl EventSink for RefuseTextDeltaSink {
    async fn record(&self, event: &CodingAgentEvent) -> StdResult<(), EventSinkError> {
        if matches!(event.event, CodingEvent::TextDelta { .. }) {
            return Err(EventSinkError::new("the event store disconnected"));
        }
        Ok(())
    }
}

#[tokio::test]
async fn a_model_request_waits_until_its_input_is_durable() {
    let sink = Arc::new(UserInputGate::default());
    let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
    let mut session = builder(client)
        .event_sink(Arc::clone(&sink) as Arc<dyn EventSink>)
        .build()
        .expect("the session builds");
    let prompt = session.prompt("do a thing");
    tokio::pin!(prompt);

    tokio::select! {
        () = sink.reached.notified() => {}
        result = &mut prompt => panic!("the prompt passed its durability boundary: {result:?}"),
    }
    assert_eq!(
        provider.call_count(),
        0,
        "the model is not called before its input reaches the sink"
    );

    sink.release.notify_one();
    prompt.await.expect("the prompt continues after the commit");
}

#[tokio::test]
async fn a_mid_stream_sink_failure_cancels_the_model_and_keeps_its_error() {
    let (client, _provider) = scripted_client(vec![ScriptedCall::EventsThenPending(
        text_delta_events("partial"),
    )]);
    let mut session = builder(client)
        .event_sink(Arc::new(RefuseTextDeltaSink))
        .build()
        .expect("the session builds");

    let failure = timeout(Duration::from_secs(5), session.prompt("do a thing"))
        .await
        .expect("the failed stream cancels a model response that never ends")
        .expect_err("the prompt reports the sink failure");

    assert_eq!(failure.kind(), ErrorKind::EventStream);
    assert!(
        ErrorData::from(&failure)
            .message
            .contains("the event store disconnected")
    );
    assert_eq!(session.state(), CodingAgentState::Closed);
    assert!(session.ended, "the failing prompt completed its shutdown");
    assert!(session.pump.is_none(), "the failing prompt joined its pump");
}

#[tokio::test]
async fn a_refusing_sink_stops_the_prompt() {
    let (client, _provider) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
    let mut session = builder(client)
        .event_sink(Arc::new(RefusingSink))
        .build()
        .expect("the session builds");

    let failure = session
        .prompt("do a thing")
        .await
        .expect_err("the prompt waits for its sink failure");

    assert_eq!(failure.kind(), ErrorKind::EventStream);
    assert!(
        ErrorData::from(&failure)
            .message
            .contains("the disk is full")
    );
}

#[tokio::test]
async fn a_session_whose_sink_refused_takes_no_further_input() {
    let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
    let mut session = builder(client)
        .event_sink(Arc::new(RefusingSink))
        .build()
        .expect("the session builds");
    let mut events = session.subscribe();

    let failure = session
        .prompt("do a thing")
        .await
        .expect_err("the current prompt reports the sink failure");
    let calls_before = provider.call_count();

    assert_eq!(failure.kind(), ErrorKind::EventStream);
    assert_eq!(
        session.state(),
        CodingAgentState::Closed,
        "a session whose events go nowhere stops"
    );
    assert!(
        matches!(
            session.prompt("and another").await,
            Err(Error::SessionClosed)
        ),
        "the next prompt is refused rather than run blind"
    );
    assert_eq!(
        provider.call_count(),
        calls_before,
        "the refused prompt asks the model nothing"
    );
    assert!(
        events.try_recv().is_err(),
        "nothing reached a subscriber after the sink refused"
    );
}

// --- The state machine ---

#[tokio::test]
async fn a_tool_round_moves_the_state_through_executing_and_back() {
    // The moves the bridge makes around a tool round, in the order the loop
    // makes them. A move the table does not allow panics in a debug build,
    // so reaching the end is the assertion that the table allows them all.
    let (mut session, _provider) = TestSession::answering(vec![]);
    let mut events = session.subscribe();
    let machine = session.state_machine();

    machine.transition(CodingAgentState::Thinking);
    machine.transition(CodingAgentState::Executing);
    assert_eq!(
        session.state(),
        CodingAgentState::Executing,
        "the session reads the state the bridge moved"
    );
    machine.transition(CodingAgentState::Thinking);
    assert_eq!(session.state(), CodingAgentState::Thinking);
    machine.transition(CodingAgentState::Idle);
    assert_eq!(session.state(), CodingAgentState::Idle);

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        published
            .iter()
            .filter(|event| matches!(event, CodingEvent::ProcessingEnd))
            .count(),
        1,
        "returning to idle ends one processing cycle, and executing ends none"
    );
}

// --- Storing and resuming ---

#[tokio::test]
async fn a_session_round_trips_through_its_record() {
    let mut session = session();
    session
        .prompt("do a thing")
        .await
        .expect("the prompt succeeds");
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
    let record = session.to_record();

    let resumed = CodingRuntime::from_record(
        record.clone(),
        &ResumeMode::RecordedModel,
        builder(client()),
    )
    .expect("the record restores");

    assert_eq!(resumed.id(), session.id());
    assert_eq!(resumed.session().root_session_id().as_str(), session.id());
    assert_eq!(resumed.history().turns(), session.history().turns());
    assert_eq!(record.provider.as_deref(), Some("test"));
    assert_eq!(record.model.as_deref(), Some("model"));
    assert!(record.last_event_seq > 0);
    assert_eq!(
        resumed.state(),
        CodingAgentState::Idle,
        "a resumed session is idle whatever ended the last one"
    );
}

#[tokio::test]
async fn a_resumed_session_keeps_numbering_where_it_left_off() {
    let mut session = session();
    session
        .prompt("do a thing")
        .await
        .expect("the prompt succeeds");
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
    let record = session.to_record();
    let last_seq = record.last_event_seq;

    let resumed = CodingRuntime::from_record(
        record.clone(),
        &ResumeMode::RecordedModel,
        builder(client()),
    )
    .expect("the record restores");
    let mut events = resumed.subscribe();
    resumed.emit(CodingEvent::LoopDetected);

    assert_eq!(
        events.recv().await.expect("the event is published").seq,
        last_seq + 1
    );
}

#[tokio::test]
async fn a_record_taken_mid_life_covers_the_events_still_queued() {
    // The checkpoint case: a record is stored while the session runs on,
    // and the numbers a resumed session would issue must start above every
    // event this one has already emitted, published or not.
    let mut session = session();
    let mut events = session.subscribe();
    session
        .prompt("do a thing")
        .await
        .expect("the prompt succeeds");

    let record = session.to_record();

    let mut published = Vec::new();
    for _ in 0..8 {
        while let Ok(event) = events.try_recv() {
            published.push(event.seq);
        }
        yield_now().await;
    }
    assert!(
        !published.is_empty(),
        "the prompt published events for the record to cover"
    );
    assert!(
        published.iter().all(|seq| *seq <= record.last_event_seq),
        "a record taken while the pipeline is behind still covers what the \
         prompt emitted: {published:?} against {}",
        record.last_event_seq
    );
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
}

#[tokio::test]
async fn a_record_from_a_newer_pebble_is_refused() {
    let mut record = SessionRecord::new(SessionScope::root(SessionId::new("ses_1")));
    record.format_version = SESSION_RECORD_FORMAT_VERSION + 1;

    let error = CodingRuntime::from_record(
        record.clone(),
        &ResumeMode::RecordedModel,
        builder(client()),
    )
    .expect_err("this build is too old for the record");

    assert!(matches!(
        error,
        CodingAgentBuildError::UnsupportedRecord { version, supported }
            if version == SESSION_RECORD_FORMAT_VERSION + 1
                && supported == SESSION_RECORD_FORMAT_VERSION
    ));
}

// --- Small parts ---

#[test]
fn a_date_is_recognized_by_its_shape() {
    assert!(is_iso_date("2026-08-31"));
    assert!(!is_iso_date("mock output"));
    assert!(!is_iso_date("2026-08-3"));
    assert!(!is_iso_date("2026/08/31"));
    assert!(!is_iso_date(""));
}

#[test]
fn only_a_credential_failure_closes_a_session() {
    for kind in [LlmErrorKind::Authentication, LlmErrorKind::AccessDenied] {
        assert!(is_auth_error(&LlmError::new(kind, "no")));
    }
    for kind in [
        LlmErrorKind::RateLimit,
        LlmErrorKind::Server,
        LlmErrorKind::Network,
    ] {
        assert!(!is_auth_error(&LlmError::new(kind, "no")));
    }
}

#[tokio::test]
async fn a_scripted_failure_reaches_the_session_as_the_error_it_names() {
    let (mut session, _provider) = TestSession::answering(vec![ScriptedCall::Failure(
        ScriptedFailure::terminal(LlmErrorKind::Server, "the provider is down"),
    )]);

    let error = session
        .prompt("anything")
        .await
        .expect_err("the call failed");

    assert!(
        matches!(&error, Error::Llm(inner) if inner.kind() == LlmErrorKind::Server),
        "{error:?}"
    );
}
