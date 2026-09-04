//! Recovery uses actual application storage. Invalid logs remain untouched.

use std::fs as sync_fs;

use pebble_coding_agent::events::CodingEvent;

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
            Ok::<_, Box<dyn StdError + Send + Sync>>(agent.to_record())
        }
        .await;
        let shutdown = agent.shutdown(ShutdownReason::Completed).await;
        let record = saved?;
        shutdown?;
        Ok(record)
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
