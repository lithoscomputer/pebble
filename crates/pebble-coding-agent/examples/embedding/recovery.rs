//! Recovery uses actual application storage. Invalid logs remain untouched.
//! Process interruption is gated at a storage boundary; this is not a
//! power-loss test.

use std::fs as sync_fs;
use std::future::pending;
use std::io::Write as _;
use std::process::Stdio;

use pebble_coding_agent::events::CodingEvent;
use pebble_coding_agent::test_support::message_text;
use tokio::io::{AsyncBufReadExt as _, AsyncRead, BufReader};
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use super::*;

const OLD_INPUT: &str = "saved conversation before interruption";

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Self {
        Self(env::temp_dir().join(format!("pebble-recovery-{}", Uuid::new_v4())))
    }

    fn state(&self) -> PathBuf {
        self.0.join("state")
    }

    async fn seed(&self) -> AppResult<SessionRecord> {
        let application = Application::open(&self.0).await?;
        let (client, _) =
            scripted_client(vec![ScriptedCall::response(text_response("old answer"))]);
        let mut agent = application.start(client).await?;
        let saved = async {
            agent.prompt(OLD_INPUT).await?;
            application.checkpoint(&mut agent).await?;
            Ok::<_, Box<dyn StdError + Send + Sync>>(())
        }
        .await;
        let shutdown = agent.shutdown(ShutdownReason::Completed).await;
        saved?;
        shutdown?;
        Ok(serde_json::from_slice(
            &fs::read(self.state().join("session.json")).await?,
        )?)
    }

    async fn log_bytes(&self) -> AppResult<Vec<u8>> {
        Ok(fs::read(self.state().join("events.jsonl")).await?)
    }

    async fn events(&self) -> AppResult<Vec<CodingAgentEvent>> {
        let log = String::from_utf8(self.log_bytes().await?)?;
        Ok(log
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?)
    }

    async fn replace_log(&self, bytes: &[u8]) -> AppResult {
        fs::write(self.state().join("events.jsonl"), bytes).await?;
        Ok(())
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = sync_fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn a_partial_final_log_entry_is_rejected_without_repair() -> AppResult {
    reject_unterminated_event(false).await
}

#[tokio::test]
async fn complete_json_without_its_line_terminator_is_rejected_without_repair() -> AppResult {
    reject_unterminated_event(true).await
}

async fn reject_unterminated_event(complete: bool) -> AppResult {
    let workspace = Workspace::new();
    workspace.seed().await?;
    let mut next = workspace
        .events()
        .await?
        .pop()
        .expect("seed produced events");
    next.seq += 1;
    next.event = CodingEvent::TextDelta {
        delta: "uncommitted tail".to_owned(),
    };
    let mut tail = serde_json::to_vec(&next)?;
    if !complete {
        tail.truncate(tail.len() / 2);
    }
    let mut damaged = workspace.log_bytes().await?;
    damaged.extend_from_slice(&tail);
    workspace.replace_log(&damaged).await?;
    let checkpoint = fs::read(workspace.state().join("session.json")).await?;

    let failure = match Application::open(&workspace.0).await {
        Ok(_) => panic!("an incomplete log must prevent application startup"),
        Err(error) => error,
    };
    assert!(
        failure.to_string().contains("incomplete final entry"),
        "{failure}"
    );
    assert_eq!(
        workspace.log_bytes().await?,
        damaged,
        "opening must not repair or append"
    );
    assert_eq!(
        fs::read(workspace.state().join("session.json")).await?,
        checkpoint
    );
    Ok(())
}

#[tokio::test]
async fn a_log_truncated_behind_its_checkpoint_cannot_resume() -> AppResult {
    let workspace = Workspace::new();
    let record = workspace.seed().await?;
    assert!(record.last_event_seq > 1);
    let mut truncated = Vec::new();
    for event in workspace
        .events()
        .await?
        .iter()
        .filter(|event| event.seq < record.last_event_seq)
    {
        truncated.extend_from_slice(&serde_json::to_vec(event)?);
        truncated.push(b'\n');
    }
    workspace.replace_log(&truncated).await?;
    let checkpoint = fs::read(workspace.state().join("session.json")).await?;
    let application = Application::open(&workspace.0).await?;
    let (client, provider) = scripted_client(vec![]);
    let failure = match application.resume(client).await {
        Ok(_) => panic!("resume must reject missing durable events"),
        Err(error) => error,
    };
    assert!(
        failure.to_string().contains("same durable stream"),
        "{failure}"
    );
    assert_eq!(provider.call_count(), 0, "recovery does not call the model");
    assert_eq!(workspace.log_bytes().await?, truncated);
    assert_eq!(
        fs::read(workspace.state().join("session.json")).await?,
        checkpoint
    );
    Ok(())
}

const NEW_INPUT: &str = "conversation newer than the saved checkpoint";
const PATIENCE: Duration = Duration::from_secs(15);
const WORKSPACE_ENV: &str = "PEBBLE_RECOVERY_WORKSPACE";
const SCENARIO_ENV: &str = "PEBBLE_RECOVERY_SCENARIO";
const READY: &str = "PEBBLE_RECOVERY_READY";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CheckpointStage {
    PartialWrite,
    Synced,
    Replaced,
}

pub(super) async fn park_at_checkpoint(
    configured: Option<CheckpointStage>,
    reached: CheckpointStage,
) -> AppResult {
    if configured == Some(reached) {
        signal_and_park().await?;
    }
    Ok(())
}

async fn signal_and_park() -> AppResult {
    {
        let mut pipe = io::stdout().lock();
        writeln!(pipe, "{READY}")?;
        pipe.flush()?;
    }
    pending().await
}

struct LostAcknowledgment(Arc<EventLog>);

#[async_trait]
impl EventSink for LostAcknowledgment {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        self.0.record(event).await?;
        if matches!(&event.event, CodingEvent::ToolCallCompleted { tool_call_id, .. } if tool_call_id == "unacknowledged")
        {
            signal_and_park().await.map_err(|source| {
                EventSinkError::new("signalling committed event").with_source(source)
            })?;
        }
        Ok(())
    }
}

#[tokio::test]
#[ignore = "subprocess entry point, selected explicitly by the recovery tests"]
async fn crash_worker() -> AppResult {
    let workspace = PathBuf::from(
        env::var_os(WORKSPACE_ENV).ok_or_else(|| io::Error::other("recovery workspace missing"))?,
    );
    let scenario = env::var(SCENARIO_ENV)?;
    let mut application = Application::open(&workspace).await?;
    if scenario == "acknowledgment" {
        let (client, _) = scripted_client(vec![ScriptedCall::response(tool_call_response(
            "save_note",
            "unacknowledged",
            json!({"text":"side effect survived"}),
        ))]);
        let record = application.store.load().await?;
        let mut agent = application
            .configure(CodingAgent::resume(
                client,
                application.environment.clone(),
                record,
                ResumeMode::RecordedModel,
            ))
            .event_sink(Arc::new(LostAcknowledgment(
                application.store.events.clone(),
            )))
            .build()
            .await?;
        agent.prompt(NEW_INPUT).await?;
        return Err(io::Error::other("the event acknowledgment was not withheld").into());
    }

    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("new answer"))]);
    let mut agent = if scenario == "first-save" {
        application.start(client).await?
    } else {
        application.resume(client).await?
    };
    agent.prompt(NEW_INPUT).await?;
    fs::write(
        application.store.directory.join("expected.json"),
        serde_json::to_vec(&agent.to_record())?,
    )
    .await?;
    application.store.checkpoint_stage = Some(match scenario.as_str() {
        "partial-write" => CheckpointStage::PartialWrite,
        "synced" | "first-save" => CheckpointStage::Synced,
        "replaced" => CheckpointStage::Replaced,
        _ => return Err(io::Error::other(format!("unknown recovery scenario: {scenario}")).into()),
    });
    application.checkpoint(&mut agent).await?;
    Err(io::Error::other("the checkpoint boundary was not reached").into())
}

impl Workspace {
    async fn interrupt(&self, scenario: &str) -> AppResult {
        let mut child = Command::new(env::current_exe()?)
            .args([
                "--exact",
                "recovery::crash_worker",
                "--ignored",
                "--nocapture",
            ])
            .env(WORKSPACE_ENV, &self.0)
            .env(SCENARIO_ENV, scenario)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let (ready, received) = oneshot::channel();
        let stdout = tokio::spawn(read_child_output(
            child.stdout.take().expect("stdout is piped"),
            Some(ready),
        ));
        let stderr = tokio::spawn(read_child_output(
            child.stderr.take().expect("stderr is piped"),
            None,
        ));
        let reached = timeout(PATIENCE, received).await;
        // Always stop, reap, and join readers before reporting a setup failure.
        let killed = child.start_kill();
        let exited = timeout(PATIENCE, child.wait()).await;
        let stdout = finish_reader(stdout).await;
        let stderr = finish_reader(stderr).await;
        let status = exited??;
        let stdout = stdout?;
        let stderr = stderr?;
        if !matches!(reached, Ok(Ok(()))) || killed.is_err() || status.success() {
            return Err(io::Error::other(format!(
                "scenario {scenario}: readiness {reached:?}, kill {killed:?}, exit {status}; stdout: {stdout}; stderr: {stderr}"
            )).into());
        }
        Ok(())
    }

    async fn expected_record(&self) -> AppResult<SessionRecord> {
        Ok(serde_json::from_slice(
            &fs::read(self.state().join("expected.json")).await?,
        )?)
    }
}

async fn read_child_output(
    reader: impl AsyncRead + Unpin,
    mut ready: Option<oneshot::Sender<()>>,
) -> io::Result<String> {
    let mut lines = BufReader::new(reader).lines();
    let mut output = String::new();
    while let Some(line) = lines.next_line().await? {
        if line == READY {
            if let Some(ready) = ready.take() {
                let _ = ready.send(());
            }
        }
        // Retain diagnostics without allowing a failing helper to exhaust RAM.
        if output.len() < 64 * 1024 {
            output.push_str(&line);
            output.push('\n');
        }
    }
    Ok(output)
}

async fn finish_reader(mut reader: JoinHandle<io::Result<String>>) -> AppResult<String> {
    match timeout(PATIENCE, &mut reader).await {
        Ok(result) => Ok(result??),
        Err(error) => {
            reader.abort();
            let _ = reader.await;
            Err(error.into())
        }
    }
}

#[tokio::test]
async fn a_committed_event_with_lost_acknowledgment_advances_resume_without_replaying_history()
-> AppResult {
    let workspace = Workspace::new();
    let old = workspace.seed().await?;
    workspace.interrupt("acknowledgment").await?;
    let before = workspace.log_bytes().await?;
    let events = workspace.events().await?;
    let committed = events.last().expect("the event was synced");
    assert!(
        matches!(&committed.event, CodingEvent::ToolCallCompleted { tool_call_id, .. } if tool_call_id == "unacknowledged")
    );
    assert_eq!(events.iter().filter(|event| matches!(&event.event, CodingEvent::ToolCallCompleted { tool_call_id, .. } if tool_call_id == "unacknowledged")).count(), 1);
    assert!(committed.seq > old.last_event_seq);
    assert_eq!(
        fs::read_to_string(workspace.0.join("workspace/note.txt")).await?,
        "side effect survived"
    );
    let saved: SessionRecord =
        serde_json::from_slice(&fs::read(workspace.state().join("session.json")).await?)?;
    assert_eq!(saved, old, "the process never saved its new conversation");

    let application = Application::open(&workspace.0).await?;
    let recovered = application.store.load().await?;
    assert_eq!(recovered.messages, old.messages);
    assert_eq!(recovered.last_event_seq, committed.seq);
    let (client, provider) =
        scripted_client(vec![ScriptedCall::response(text_response("resumed"))]);
    let mut agent = application.resume(client).await?;
    let result = agent.prompt("continue from the saved checkpoint").await;
    let shutdown = agent.shutdown(ShutdownReason::Completed).await;
    result?;
    shutdown?;
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    let history: Vec<_> = requests[0].messages().iter().map(message_text).collect();
    assert!(history.iter().any(|text| text == OLD_INPUT));
    assert!(!history.iter().any(|text| text == NEW_INPUT));
    let after = workspace.log_bytes().await?;
    assert!(
        after.starts_with(&before),
        "resume leaves prior log entries unchanged"
    );
    let events = workspace.events().await?;
    let first_new = events
        .iter()
        .find(|event| event.seq > committed.seq)
        .expect("resume publishes events");
    assert_eq!(first_new.seq, committed.seq + 1);
    assert!(
        events
            .iter()
            .all(|event| event.stream_id() == old.session_id)
    );
    assert!(events.windows(2).all(|pair| pair[0].seq + 1 == pair[1].seq));
    Ok(())
}

#[tokio::test]
async fn a_partial_checkpoint_write_keeps_the_previous_record() -> AppResult {
    recover_checkpoint("partial-write", CheckpointStage::PartialWrite).await
}

#[tokio::test]
async fn a_synced_temporary_checkpoint_does_not_replace_the_previous_record() -> AppResult {
    recover_checkpoint("synced", CheckpointStage::Synced).await
}

#[tokio::test]
async fn a_replaced_checkpoint_survives_a_lost_save_acknowledgment() -> AppResult {
    recover_checkpoint("replaced", CheckpointStage::Replaced).await
}

async fn recover_checkpoint(scenario: &str, stage: CheckpointStage) -> AppResult {
    let workspace = Workspace::new();
    let old = workspace.seed().await?;
    workspace.interrupt(scenario).await?;
    let expected = workspace.expected_record().await?;
    assert_ne!(
        expected.messages, old.messages,
        "the replacement has distinct conversation content"
    );
    let store = Store::open(&workspace.state()).await?;
    let recovered = store.load().await?;
    let authoritative: SessionRecord =
        serde_json::from_slice(&fs::read(workspace.state().join("session.json")).await?)?;
    let temporary = workspace.state().join("session.json.tmp");
    if stage == CheckpointStage::Replaced {
        assert_eq!(authoritative.messages, expected.messages);
        assert_eq!(authoritative.last_event_seq, expected.last_event_seq);
        assert!(!fs::try_exists(&temporary).await?);
    } else {
        assert_eq!(authoritative, old);
        assert_eq!(recovered.messages, old.messages);
        let bytes = fs::read(&temporary).await?;
        assert!(!bytes.is_empty());
        let partial = serde_json::from_slice::<SessionRecord>(&bytes);
        if stage == CheckpointStage::PartialWrite {
            assert!(partial.is_err(), "only a prefix was written");
        } else {
            assert_eq!(partial?.messages, expected.messages);
        }
    }
    assert_eq!(
        recovered.last_event_seq, expected.last_event_seq,
        "cursor catches up without replacing saved messages"
    );
    store.save(&expected).await?;
    let reopened = Store::open(&workspace.state()).await?;
    assert_eq!(
        reopened.load().await?,
        expected,
        "a later normal save succeeds"
    );
    assert!(!fs::try_exists(temporary).await?);
    Ok(())
}

#[tokio::test]
async fn an_interrupted_first_save_does_not_promote_its_temporary_file() -> AppResult {
    let workspace = Workspace::new();
    workspace.interrupt("first-save").await?;
    assert!(!fs::try_exists(workspace.state().join("session.json")).await?);
    let temporary = fs::read(workspace.state().join("session.json.tmp")).await?;
    let expected = workspace.expected_record().await?;
    assert_eq!(
        serde_json::from_slice::<SessionRecord>(&temporary)?.messages,
        expected.messages
    );
    let log = workspace.log_bytes().await?;
    let store = Store::open(&workspace.state()).await?;
    let error = store
        .load()
        .await
        .expect_err("there is no authoritative checkpoint");
    assert_eq!(
        error.downcast_ref::<io::Error>().map(io::Error::kind),
        Some(io::ErrorKind::NotFound)
    );
    assert_eq!(workspace.log_bytes().await?, log);
    assert_eq!(
        fs::read(workspace.state().join("session.json.tmp")).await?,
        temporary
    );
    Ok(())
}
