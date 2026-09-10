//! Resuming a session, through the public API.
//!
//! A stored record names the exact route its session ran on. Resuming on the
//! recorded model restores that route or refuses; resuming on a named model is
//! the failover path, visible at the call site. A live agent can also hand its
//! warm state to a successor in the same process, which carries on without
//! initializing again. These tests drive every one of those paths the way an
//! application does.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::client::ClientBuild;
use lithos_llm::types::{ErrorKind as LlmErrorKind, Message as LlmMessage, Request, Role};
use pebble_agent::{AgentError, SessionId, SessionScope};
use pebble_coding_agent::environment::Environment;
use pebble_coding_agent::events::{CodingAgentEvent, CodingAgentState, EventSink, EventSinkError};
use pebble_coding_agent::extensions::{Answer, HumanInputError, HumanInputProvider, Question};
use pebble_coding_agent::state::{
    Message, SESSION_RECORD_FORMAT_VERSION, SessionRecord, StoredMessage,
};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedFailure, ScriptedProvider, TEST_CATALOG, message_text,
    scripted_client, scripted_client_builder, text_response, tool_call_response,
};
use pebble_coding_agent::tools::RegisteredTool;
use pebble_coding_agent::{
    CodingAgent, CodingAgentBuildError, CodingAgentOptions, Error, ResumeMode, ShutdownReason,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The current format fixture, which names a route the test catalog lacks.
const SAMPLE_RECORD_V1: &str = include_str!("fixtures/session_record_v4.json");

/// A second provider offering a model called `model`, like the `test` provider
/// does, so a resume that matched on the model name alone could pick the
/// wrong one.
const SAME_MODEL_NAME_ELSEWHERE: &str = r#"
schema_version = 1

[providers.bare.models.model]
display_name = "The other provider's model"
api_model = "model"
capabilities = { text = true, tools = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

[providers.bare.models.model.metadata.agent]
profile = "anthropic"
"#;

/// A client whose catalog has `model` on two providers.
fn ambiguous_client(calls: Vec<ScriptedCall>) -> (Client, Arc<ScriptedProvider>) {
    let catalog = Catalog::builder()
        .overlay_toml(TEST_CATALOG)
        .expect("the test catalog parses")
        .overlay_toml(SAME_MODEL_NAME_ELSEWHERE)
        .expect("the overlay parses")
        .build()
        .expect("the catalog validates");
    let (builder, provider) = scripted_client_builder(ScriptedProvider::new(calls));
    let ClientBuild { client, .. } = builder
        .catalog(catalog)
        .build()
        .expect("the scripted client builds");
    (client, provider)
}

fn environment() -> Arc<dyn Environment> {
    Arc::new(MockEnvironment::linux())
}

/// An environment with one skill for initialization to discover.
fn environment_with_a_skill() -> Arc<dyn Environment> {
    Arc::new(MockEnvironment {
        files: [(
            "/skills/commit/SKILL.md".to_owned(),
            "---\nname: commit\ndescription: Make a commit\n---\nRun commit.".to_owned(),
        )]
        .into_iter()
        .collect(),
        glob_results: vec!["/skills/commit/SKILL.md".to_owned()],
        ..MockEnvironment::linux()
    })
}

/// A sink that keeps every stream position it is given.
#[derive(Default)]
struct SequenceLog {
    positions: Mutex<Vec<(String, u64)>>,
}

impl SequenceLog {
    fn seqs(&self) -> Vec<u64> {
        self.positions
            .lock()
            .expect("the log lock is held")
            .iter()
            .map(|(_, seq)| *seq)
            .collect()
    }

    fn stream_ids(&self) -> Vec<String> {
        self.positions
            .lock()
            .expect("the log lock is held")
            .iter()
            .map(|(stream_id, _)| stream_id.clone())
            .collect()
    }
}

#[async_trait]
impl EventSink for SequenceLog {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        self.positions
            .lock()
            .expect("the log lock is held")
            .push((event.stream_id().to_owned(), event.seq));
        Ok(())
    }
}

/// Someone to ask, who is never asked anything here.
struct Unused;

#[async_trait]
impl HumanInputProvider for Unused {
    async fn ask_questions(
        &self,
        _tool_call_id: &str,
        _questions: Vec<Question>,
        _cancel_token: CancellationToken,
    ) -> Result<Vec<Answer>, HumanInputError> {
        Ok(Vec::new())
    }
}

/// A session on `selector` that has answered one prompt and been stored.
async fn stored_session(client: Client, selector: &str) -> (SessionRecord, Vec<Message>, String) {
    let mut agent = CodingAgent::builder(client, environment())
        .model(selector)
        .build()
        .await
        .expect("the agent builds");
    agent
        .prompt("first")
        .await
        .result
        .expect("the first prompt succeeds");
    let record = agent.to_record();
    let history = agent.history().turns().to_vec();
    let id = agent.id().to_owned();
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
    (record, history, id)
}

fn tool_names(request: &Request) -> Vec<String> {
    let mut names: Vec<String> = request
        .tools()
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    names.sort();
    names
}

fn system_prompt(request: &Request) -> String {
    let first = request
        .messages()
        .first()
        .expect("the request is not empty");
    assert_eq!(first.role(), Role::System);
    message_text(first)
}

// --- The recorded route ---

#[tokio::test]
async fn a_session_resumes_on_its_recorded_route_exactly() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("one"))]);
    let (record, history, id) = stored_session(client, "test/model").await;
    assert_eq!(record.provider.as_deref(), Some("test"));
    assert_eq!(record.model.as_deref(), Some("model"));

    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("two"))]);
    let mut resumed = CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
        .build()
        .await
        .expect("the record resumes on its recorded route");

    assert_eq!(resumed.id(), id);
    assert_eq!(resumed.provider(), "test");
    assert_eq!(resumed.model(), "model");
    assert_eq!(resumed.history().turns(), history.as_slice());
    let outcome = resumed
        .prompt("second")
        .await
        .result
        .expect("the resumed prompt succeeds");
    assert_eq!(outcome.text.as_deref(), Some("two"));
    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the resumed agent shuts down");
}

#[tokio::test]
async fn a_recorded_route_is_never_swapped_for_another_provider_with_the_same_model_name() {
    // Two providers offer a model called `model`. A resume that matched on the
    // name alone could land on either; the recorded route says which.
    for provider in ["test", "bare"] {
        let (client, _) = ambiguous_client(vec![ScriptedCall::response(text_response("one"))]);
        let (record, _, _) = stored_session(client, &format!("{provider}/model")).await;
        assert_eq!(record.provider.as_deref(), Some(provider));

        let (client, _) = ambiguous_client(vec![ScriptedCall::response(text_response("two"))]);
        let mut resumed =
            CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
                .build()
                .await
                .expect("the record resumes on its recorded route");

        assert_eq!(
            resumed.provider(),
            provider,
            "the recorded provider is restored"
        );
        assert_eq!(resumed.model(), "model");
        resumed
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("the resumed agent shuts down");
    }
}

#[tokio::test]
async fn a_record_naming_an_unavailable_route_is_refused_rather_than_rerouted() {
    // The frozen fixture ran on `anthropic/claude-sonnet-5`, which the scripted
    // catalog does not have.
    let record: SessionRecord = serde_json::from_str(SAMPLE_RECORD_V1).expect("the fixture parses");
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("never"))]);

    let error = CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
        .build()
        .await
        .err()
        .expect("an unavailable route is refused");

    assert!(
        matches!(
            &error,
            CodingAgentBuildError::RecordedRouteUnavailable { provider, model, .. }
                if provider == "anthropic" && model == "claude-sonnet-5"
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_record_naming_no_route_is_refused_on_the_recorded_model() {
    let record = SessionRecord::new(SessionScope::root(SessionId::new("ses_unrouted")));
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("never"))]);

    let error = CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
        .build()
        .await
        .err()
        .expect("a record with no route cannot resume on it");

    assert!(
        matches!(
            &error,
            CodingAgentBuildError::RecordedRouteMissing { session_id } if session_id == "ses_unrouted"
        ),
        "{error:?}"
    );
}

// --- Failover ---

#[tokio::test]
async fn failover_keeps_the_conversation_and_changes_the_route_visibly() {
    let (client, _) = ambiguous_client(vec![ScriptedCall::response(text_response("one"))]);
    let (record, history, id) = stored_session(client, "test/model").await;

    let (client, _) = ambiguous_client(vec![ScriptedCall::response(text_response("two"))]);
    let mut resumed = CodingAgent::resume(
        client,
        environment(),
        record,
        ResumeMode::UseModel("bare/model".into()),
    )
    .build()
    .await
    .expect("the record resumes on the fallback route");

    assert_eq!(resumed.id(), id);
    assert_eq!(resumed.provider(), "bare");
    assert_eq!(resumed.model(), "model");
    assert_eq!(
        resumed.history().turns(),
        history.as_slice(),
        "failover keeps the conversation"
    );
    let outcome = resumed
        .prompt("second")
        .await
        .result
        .expect("the resumed prompt succeeds");
    assert_eq!(outcome.text.as_deref(), Some("two"));
    // The new route is what the next record stores.
    let stored_again = resumed.to_record();
    assert_eq!(stored_again.provider.as_deref(), Some("bare"));
    assert_eq!(stored_again.model.as_deref(), Some("model"));
    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the resumed agent shuts down");
}

#[tokio::test]
async fn a_resume_builder_refuses_a_model_of_its_own() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("one"))]);
    let (record, _, _) = stored_session(client, "test/model").await;
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("never"))]);

    let error = CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
        .model("test/thinking")
        .build()
        .await
        .err()
        .expect("the resume mode decides the model");

    assert!(
        matches!(error, CodingAgentBuildError::ModelConflictsWithResume),
        "{error:?}"
    );
}

/// A tool whose every run is an effect the world keeps, counted in `runs`.
fn appending_tool(runs: Arc<AtomicUsize>) -> RegisteredTool {
    RegisteredTool::function(
        "append",
        "Appends one line to the log",
        json!({"type": "object"}),
        move |_context, _arguments| {
            runs.fetch_add(1, Ordering::SeqCst);
            async { Ok("appended".to_owned()) }
        },
    )
}

/// The prompt that a failed model call left unfinished continues on the
/// fallback route from the tool results it had committed: the model is asked
/// once, on the history as it stands, and the effect is not repeated.
#[tokio::test]
async fn an_unfinished_prompt_continues_on_another_model_without_repeating_its_effect() {
    let runs = Arc::new(AtomicUsize::new(0));
    // The primary asks for the effect, then fails for good on its next call.
    let (client, _) = ambiguous_client(vec![
        ScriptedCall::response(tool_call_response("append", "call_1", json!({}))),
        ScriptedCall::Failure(ScriptedFailure::terminal(
            LlmErrorKind::Server,
            "overloaded",
        )),
    ]);
    let mut agent = CodingAgent::builder(client, environment())
        .model("test/model")
        .tools([appending_tool(Arc::clone(&runs))])
        .build()
        .await
        .expect("the agent builds");
    let failed = agent.prompt("append once").await;
    assert!(matches!(failed.result, Err(Error::Llm(_))), "{failed:?}");
    assert_eq!(runs.load(Ordering::SeqCst), 1, "the effect happened");
    let record = agent.to_record();
    assert!(
        matches!(
            record.messages.last(),
            Some(StoredMessage::ToolResults { .. })
        ),
        "the record stops at the tool results: {:?}",
        record.messages
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");

    let (client, fallback) = ambiguous_client(vec![ScriptedCall::response(text_response(
        "appended, done",
    ))]);
    let mut resumed = CodingAgent::resume(
        client,
        environment(),
        record,
        ResumeMode::UseModel("bare/model".into()),
    )
    .tools([appending_tool(Arc::clone(&runs))])
    .build()
    .await
    .expect("the record resumes on the fallback route");

    let report = resumed.continue_prompt().await;

    let output = report.result.expect("the continuation succeeds");
    assert_eq!(output.text.as_deref(), Some("appended, done"));
    assert_eq!(runs.load(Ordering::SeqCst), 1, "the effect is not repeated");
    let requests = fallback.requests();
    assert_eq!(requests.len(), 1, "the fallback was asked once");
    let roles: Vec<Role> = requests[0]
        .messages()
        .iter()
        .map(LlmMessage::role)
        .collect();
    assert_eq!(
        roles,
        [Role::System, Role::User, Role::Assistant, Role::Tool],
        "the history as it stood, and no new input"
    );
    assert_eq!(requests[0].messages()[3].tool_call_id(), Some("call_1"));
    let turns = resumed.history().turns().to_vec();
    assert_eq!(turns.len(), 4, "only the answer was committed: {turns:?}");
    assert!(matches!(turns[3], Message::Assistant { .. }));
    assert_eq!(resumed.provider(), "bare");
    assert_eq!(resumed.state(), CodingAgentState::Idle);
    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the resumed agent shuts down");
}

#[tokio::test]
async fn a_finished_conversation_has_nothing_to_continue() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("one"))]);
    let (record, history, _) = stored_session(client, "test/model").await;
    let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("two"))]);
    let mut resumed = CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
        .build()
        .await
        .expect("the record resumes");

    let report = resumed.continue_prompt().await;

    assert!(
        matches!(
            report.result,
            Err(Error::Agent(AgentError::NothingToContinue))
        ),
        "{report:?}"
    );
    assert_eq!(provider.call_count(), 0, "the model was not asked");
    assert_eq!(resumed.history().turns(), history.as_slice());
    assert_eq!(resumed.state(), CodingAgentState::Idle);
    let next = resumed.prompt("second").await;
    assert_eq!(
        next.result
            .expect("the agent is still usable")
            .text
            .as_deref(),
        Some("two")
    );
    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the resumed agent shuts down");
}

// --- What the record carries ---

#[tokio::test]
async fn the_frozen_version_one_record_resumes_through_the_public_api() {
    let record: SessionRecord = serde_json::from_str(SAMPLE_RECORD_V1).expect("the fixture parses");
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("never"))]);

    let mut resumed = CodingAgent::resume(
        client,
        environment(),
        record,
        ResumeMode::UseModel("test/model".into()),
    )
    .build()
    .await
    .expect("the fixture resumes");

    assert_eq!(resumed.id(), "ses_root");
    assert_eq!(resumed.history().len(), 5);
    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the resumed agent shuts down");
}

#[tokio::test]
async fn a_restored_conversation_replays_its_tool_results_paired() {
    // The fixture's assistant turn carries provider-native reasoning parts
    // that the scripted test model refuses, so this replays the same exchange
    // without them. What is under test is the pairing, not the parts.
    let mut record: SessionRecord =
        serde_json::from_str(SAMPLE_RECORD_V1).expect("the fixture parses");
    for message in &mut record.messages {
        if let StoredMessage::Assistant { provider_parts, .. } = message {
            provider_parts.clear();
        }
    }
    let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("next"))]);

    let mut resumed = CodingAgent::resume(
        client,
        environment(),
        record,
        ResumeMode::UseModel("test/model".into()),
    )
    .build()
    .await
    .expect("the record resumes");
    resumed
        .prompt("carry on")
        .await
        .result
        .expect("the resumed prompt succeeds");

    // The restored conversation reaches the model as a valid exchange: the
    // assistant's tool call is still answered by its result.
    let requests = provider.requests();
    let request = requests.first().expect("one round was requested");
    let roles: Vec<Role> = request.messages().iter().map(LlmMessage::role).collect();
    assert_eq!(roles, [
        Role::System,
        Role::User,
        Role::Assistant,
        Role::Tool,
        Role::System,
        Role::User,
        Role::User,
    ]);
    assert_eq!(request.messages()[3].tool_call_id(), Some("call_1"));
    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the resumed agent shuts down");
}

#[tokio::test]
async fn a_record_from_a_format_this_build_does_not_read_is_refused() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("one"))]);
    let (mut record, _, _) = stored_session(client, "test/model").await;
    record.format_version = SESSION_RECORD_FORMAT_VERSION + 1;
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("never"))]);

    let error = CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
        .build()
        .await
        .err()
        .expect("a future format is refused");

    assert!(
        matches!(
            error,
            CodingAgentBuildError::UnsupportedRecord { version, supported }
                if version == SESSION_RECORD_FORMAT_VERSION + 1
                    && supported == SESSION_RECORD_FORMAT_VERSION
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn event_numbering_continues_where_the_record_left_off() {
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("one"))]);
    let (mut record, _, _) = stored_session(client, "test/model").await;
    let recorded_seq = record.last_event_seq;
    let stream_id = record.scope.session_id().to_string();
    assert!(recorded_seq > 0, "the first session published events");
    let durable_seq = recorded_seq + 3;
    record.advance_event_cursor(durable_seq);

    let log = Arc::new(SequenceLog::default());
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("two"))]);
    let mut resumed = CodingAgent::resume(client, environment(), record, ResumeMode::RecordedModel)
        .event_sink(Arc::clone(&log) as Arc<dyn EventSink>)
        .build()
        .await
        .expect("the record resumes");
    resumed
        .prompt("second")
        .await
        .result
        .expect("the resumed prompt succeeds");
    resumed
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the resumed agent shuts down");

    let seqs = log.seqs();
    assert_eq!(
        seqs.first().copied(),
        Some(durable_seq + 1),
        "the resumed session starts above both the record and durable log"
    );
    assert!(
        seqs.windows(2).all(|pair| pair[0] < pair[1]),
        "sequence numbers stay strictly increasing: {seqs:?}"
    );
    assert!(
        log.stream_ids().iter().all(|id| id == &stream_id),
        "resuming keeps the stable stream identity"
    );
}

// --- The warm handoff ---

#[tokio::test]
async fn an_export_continues_in_memory_without_initializing_again() {
    // The first agent discovers a skill, so its prompt advertises the tool
    // that loads one. Its successor is built over an environment with no
    // skills at all: if it initialized again it would lose the tool, and the
    // only way it can still advertise it is by carrying the export's state.
    let (client, first_provider) =
        scripted_client(vec![ScriptedCall::response(text_response("one"))]);
    let mut first = CodingAgent::builder(client, environment_with_a_skill())
        .model("test/model")
        .options(CodingAgentOptions::default().with_skill_dirs(["/skills".to_owned()]))
        .build()
        .await
        .expect("the first agent builds");
    first
        .prompt("first")
        .await
        .result
        .expect("the first prompt succeeds");
    let mut export = first.export();
    let history = first.history().turns().to_vec();
    let id = first.id().to_owned();
    first
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the first agent shuts down");
    let closed_seq = first.committed_event_seq();
    assert!(closed_seq > export.record().last_event_seq);
    export.advance_event_cursor(closed_seq);
    export.advance_event_cursor(closed_seq - 1);
    assert_eq!(export.record().last_event_seq, closed_seq);

    let log = Arc::new(SequenceLog::default());
    let (client, second_provider) =
        scripted_client(vec![ScriptedCall::response(text_response("two"))]);
    let mut second = CodingAgent::resume_from_export(client, environment(), export.clone())
        // A service bound for the successor's life: the first agent had nobody
        // to ask, and this one does.
        .human_input(Arc::new(Unused))
        .event_sink(Arc::clone(&log) as Arc<dyn EventSink>)
        .build()
        .await
        .expect("the export resumes");

    assert_eq!(second.id(), id);
    assert_eq!(second.provider(), "test");
    assert_eq!(second.history().turns(), history.as_slice());
    let outcome = second
        .prompt("second")
        .await
        .result
        .expect("the successor's prompt succeeds");
    assert_eq!(outcome.text.as_deref(), Some("two"));

    let first_requests = first_provider.requests();
    let second_requests = second_provider.requests();
    let first_request = first_requests.first().expect("the first agent ran a round");
    let second_request = second_requests.first().expect("the successor ran a round");
    assert_eq!(
        system_prompt(second_request),
        system_prompt(first_request),
        "the successor runs on the system prompt it was handed"
    );
    let first_tools = tool_names(first_request);
    let second_tools = tool_names(second_request);
    assert!(
        first_tools.contains(&"use_skill".to_owned()),
        "the first agent discovered a skill: {first_tools:?}"
    );
    assert!(
        second_tools.contains(&"use_skill".to_owned()),
        "the successor kept the skill without rediscovering it: {second_tools:?}"
    );
    assert!(
        second_tools.contains(&"AskUserQuestion".to_owned())
            && !first_tools.contains(&"AskUserQuestion".to_owned()),
        "the successor binds its own services: {second_tools:?}"
    );

    second
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the successor shuts down");
    let seqs = log.seqs();
    assert_eq!(
        seqs.first().copied(),
        Some(export.record().last_event_seq + 1),
        "numbering continues from the export on the successor's new pump"
    );
    assert!(
        log.stream_ids().iter().all(|stream_id| stream_id == &id),
        "a warm successor keeps the stable stream identity"
    );
}
