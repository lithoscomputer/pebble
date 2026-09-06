//! An application that owns tools, permissions, storage, and session lifetime.
//!
//! Run without credentials or a network connection:
//!
//! ```sh
//! cargo run --locked -p pebble-coding-agent --features test-util --example embedding
//! ```
//!
//! The model is scripted so every run exercises the same path: write a note,
//! refuse a deletion, cancel an active tool, save history, close, and resume
//! from disk. Replace `scripted_client` with an application-built lithos client
//! to use a real model. The environment and storage already use real files.
//!
//! `Application` supplies the services again on resume; executable tools and
//! permission policies are not stored in a `SessionRecord`. `EventLog` accepts
//! an event only after syncing it. `Store` replaces checkpoints atomically and
//! reconciles their event cursor with the log before resume.
//!
//! This example has one writer and one root session per store. An incomplete
//! log is an error. Cursor reconciliation prevents duplicate sequence numbers;
//! it does not replay side effects or recover history newer than a checkpoint.
//! Applications that need those guarantees must coordinate their own storage.
//! Log repair and deduplication of application retries also belong there.

#![expect(
    clippy::print_stderr,
    reason = "the executable example reports its result"
)]

use std::error::Error as StdError;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::{env, io};

use async_trait::async_trait;
use lithos_llm::Client;
use pebble_agent::{SessionScope, ToolDescriptor, ToolScheduling};
use pebble_coding_agent::environment::{Environment, LocalEnvironment};
use pebble_coding_agent::events::{CodingAgentEvent, EventSink, EventSinkError};
use pebble_coding_agent::state::SessionRecord;
use pebble_coding_agent::test_support::{
    ScriptedCall, scripted_client, text_response, tool_call_response,
};
use pebble_coding_agent::tools::{
    PermissionMiddleware, RegisteredTool, ToolError, ToolPermission, ToolPermissionPolicy,
};
use pebble_coding_agent::{CodingAgent, CodingAgentBuilder, Error, ResumeMode, ShutdownReason};
use serde_json::json;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::AsyncWriteExt as _;
use tokio::sync::{Mutex, Notify};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

type AppResult<T = ()> = Result<T, Box<dyn StdError + Send + Sync>>;

#[tokio::main]
async fn main() -> AppResult {
    let root = env::temp_dir().join(format!("pebble-embedding-{}", Uuid::new_v4()));
    run(&root).await?;
    eprintln!("Saved, cancelled, and resumed. Files: {}", root.display());
    Ok(())
}

struct NotesPermission;

impl ToolPermissionPolicy for NotesPermission {
    fn permission(&self, _session: &SessionScope, tool: &ToolDescriptor) -> ToolPermission {
        match tool.id().as_str() {
            "save_note" | "read_note" | "wait_for_cancel" => ToolPermission::Allow,
            _ => ToolPermission::Deny {
                reason: "This application permits only saving and reading notes.".to_owned(),
            },
        }
    }
}

struct Application {
    environment:  Arc<dyn Environment>,
    store:        Store,
    tool_started: Arc<Notify>,
}

impl Application {
    async fn open(root: &Path) -> AppResult<Self> {
        let environment = LocalEnvironment::new(root.join("workspace"));
        environment.prepare().await?;
        Ok(Self {
            environment:  Arc::new(environment),
            store:        Store::open(&root.join("state")).await?,
            tool_started: Arc::new(Notify::new()),
        })
    }

    fn configure(&self, builder: CodingAgentBuilder) -> CodingAgentBuilder {
        builder
            .tools(self.tools())
            .event_sink(self.store.events.clone())
            .tool_middleware(Arc::new(PermissionMiddleware::new(Arc::new(
                NotesPermission,
            ))))
    }

    async fn start(&self, client: Client) -> AppResult<CodingAgent> {
        Ok(self
            .configure(CodingAgent::builder(client, self.environment.clone()).model("test/model"))
            .build()
            .await?)
    }

    async fn resume(&self, client: Client) -> AppResult<CodingAgent> {
        let record = self.store.load().await?;
        Ok(self
            .configure(CodingAgent::resume(
                client,
                self.environment.clone(),
                record,
                ResumeMode::RecordedModel,
            ))
            .build()
            .await?)
    }

    fn tools(&self) -> Vec<RegisteredTool> {
        let save = RegisteredTool::function(
            "save_note",
            "Save a note",
            json!({
                "type":"object", "properties":{"text":{"type":"string"}}, "required":["text"]
            }),
            |context, arguments| async move {
                let text = arguments["text"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_arguments("text must be a string"))?;
                context.env().write_file("note.txt", text).await?;
                Ok("saved".to_owned())
            },
        )
        .with_scheduling(ToolScheduling::Sequential);
        let read = RegisteredTool::function(
            "read_note",
            "Read the saved note",
            json!({"type":"object"}),
            |context, _| async move { Ok(context.env().read_file_text("note.txt").await?) },
        );
        // Registered so even a model that tries a hidden tool exercises the
        // application's permission boundary. Its executor must never run.
        let erase = RegisteredTool::function(
            "erase_note",
            "Delete the note",
            json!({"type":"object"}),
            |context, _| async move {
                context.env().delete_file("note.txt").await?;
                Ok("deleted".to_owned())
            },
        )
        .with_scheduling(ToolScheduling::Sequential);
        let started = self.tool_started.clone();
        let wait = RegisteredTool::function(
            "wait_for_cancel",
            "Wait for application cancellation",
            json!({"type":"object"}),
            move |context, _| {
                let started = started.clone();
                async move {
                    started.notify_one();
                    context.cancel().cancelled().await;
                    Err(ToolError::cancelled(
                        "the application cancelled this prompt",
                    ))
                }
            },
        );
        vec![save, read, erase, wait]
    }

    async fn cancel_active_tool(&self, agent: &mut CodingAgent) -> AppResult {
        let cancel = CancellationToken::new();
        let mut prompt = Box::pin(agent.prompt_with_cancellation("wait until cancelled", &cancel));
        tokio::select! {
            result = &mut prompt => return Err(io::Error::other(format!("tool did not wait: {result:?}")).into()),
            ready = timeout(Duration::from_secs(5), self.tool_started.notified()) => { ready?; }
        }
        cancel.cancel();
        match timeout(Duration::from_secs(5), prompt).await?.result {
            Err(Error::Interrupted(_)) => Ok(()),
            Err(error) => Err(error.into()),
            Ok(_) => Err(io::Error::other("cancelled prompt unexpectedly completed").into()),
        }
    }

    async fn checkpoint(&self, agent: &mut CodingAgent) -> AppResult {
        agent.flush_events().await?;
        self.store.save(&agent.to_record()).await
    }
}

struct Store {
    directory:        PathBuf,
    events:           Arc<EventLog>,
    #[cfg(test)]
    checkpoint_stage: Option<recovery::CheckpointStage>,
}

impl Store {
    async fn open(directory: &Path) -> AppResult<Self> {
        fs::create_dir_all(directory).await?;
        let events = Arc::new(EventLog::open(&directory.join("events.jsonl")).await?);
        sync_directory(directory).await?;
        Ok(Self {
            directory: directory.to_owned(),
            events,
            #[cfg(test)]
            checkpoint_stage: None,
        })
    }

    /// The previous checkpoint stays authoritative until replacement. Loading
    /// never promotes a leftover temporary file, including after a first save.
    async fn save(&self, record: &SessionRecord) -> AppResult {
        let temporary = self.directory.join("session.json.tmp");
        let mut file = File::create(&temporary).await?;
        let bytes = serde_json::to_vec_pretty(record)?;
        #[cfg(test)]
        if self.checkpoint_stage == Some(recovery::CheckpointStage::PartialWrite) {
            file.write_all(&bytes[..bytes.len() / 2]).await?;
            file.flush().await?;
            recovery::park_at_checkpoint(
                self.checkpoint_stage,
                recovery::CheckpointStage::PartialWrite,
            )
            .await?;
        }
        file.write_all(&bytes).await?;
        file.flush().await?;
        file.sync_all().await?;
        #[cfg(test)]
        recovery::park_at_checkpoint(self.checkpoint_stage, recovery::CheckpointStage::Synced)
            .await?;
        drop(file);
        fs::rename(temporary, self.directory.join("session.json")).await?;
        sync_directory(&self.directory).await?;
        #[cfg(test)]
        recovery::park_at_checkpoint(self.checkpoint_stage, recovery::CheckpointStage::Replaced)
            .await?;
        Ok(())
    }

    /// Acknowledgment may have been lost after an event was synced. Reconcile
    /// the cursor from disk without inventing newer conversation messages.
    async fn load(&self) -> AppResult<SessionRecord> {
        let mut record: SessionRecord =
            serde_json::from_slice(&fs::read(self.directory.join("session.json")).await?)?;
        let log = self.events.state.lock().await;
        if log.stream.as_deref() != Some(record.scope.session_id().as_str())
            || record.last_event_seq > log.last_seq
        {
            return Err(io::Error::other(
                "checkpoint and event log do not describe the same durable stream",
            )
            .into());
        }
        record.advance_event_cursor(log.last_seq);
        Ok(record)
    }
}

struct EventLog {
    state: Mutex<LogState>,
}

struct LogState {
    file:     File,
    stream:   Option<String>,
    last_seq: u64,
}

impl EventLog {
    async fn open(path: &Path) -> AppResult<Self> {
        let content = match fs::read_to_string(path).await {
            Ok(content) => content,
            Err(error) if error.kind() == io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(error.into()),
        };
        if !content.is_empty() && !content.ends_with('\n') {
            return Err(io::Error::other("event log has an incomplete final entry").into());
        }
        let mut stream = None;
        let mut last_seq = 0;
        for line in content.lines() {
            let event: CodingAgentEvent = serde_json::from_str(line)?;
            validate_position(stream.as_deref(), last_seq, &event)?;
            stream = Some(event.stream_id().to_owned());
            last_seq = event.seq;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            state: Mutex::new(LogState {
                file,
                stream,
                last_seq,
            }),
        })
    }
}

#[async_trait]
impl EventSink for EventLog {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        let mut state = self.state.lock().await;
        let write = async {
            validate_position(state.stream.as_deref(), state.last_seq, event)?;
            let mut bytes = serde_json::to_vec(event)?;
            bytes.push(b'\n');
            state.file.write_all(&bytes).await?;
            state.file.flush().await?;
            state.file.sync_all().await?;
            state.stream = Some(event.stream_id().to_owned());
            state.last_seq = event.seq;
            Ok::<_, Box<dyn StdError + Send + Sync>>(())
        }
        .await;
        write.map_err(|source| EventSinkError::new("persisting an event").with_source(source))
    }
}

fn validate_position(
    stream: Option<&str>,
    last_seq: u64,
    event: &CodingAgentEvent,
) -> io::Result<()> {
    if event.seq <= last_seq || stream.is_some_and(|stream| stream != event.stream_id()) {
        Err(io::Error::other(
            "event sequence is repeated, out of order, or belongs to another stream",
        ))
    } else {
        Ok(())
    }
}

async fn sync_directory(path: &Path) -> AppResult {
    // Unix permits syncing the directory entry after creating or renaming a
    // file. File contents are synced on every supported platform.
    #[cfg(unix)]
    File::open(path).await?.sync_all().await?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

async fn run(root: &Path) -> AppResult {
    let application = Application::open(root).await?;
    let (client, _) = scripted_client(vec![
        ScriptedCall::response(tool_call_response(
            "save_note",
            "save",
            json!({"text":"Remember this after resume."}),
        )),
        ScriptedCall::response(tool_call_response("erase_note", "denied", json!({}))),
        ScriptedCall::response(text_response("Saved the note. Deletion was refused.")),
        ScriptedCall::response(tool_call_response("wait_for_cancel", "cancel", json!({}))),
    ]);
    let mut agent = application.start(client).await?;
    let work = async {
        agent
            .prompt("save a note, then try to delete it")
            .await
            .result?;
        application.cancel_active_tool(&mut agent).await?;
        application.checkpoint(&mut agent).await
    }
    .await;
    // Cleanup runs even when a prompt or the application's storage fails.
    let closed = agent.shutdown(ShutdownReason::Completed).await;
    work?;
    closed?;
    drop(agent);
    drop(application);

    // Reopen both storage and services. Shutdown appended events after the
    // checkpoint, so loading must advance its cursor before building an agent.
    let application = Application::open(root).await?;
    let (client, _) = scripted_client(vec![
        ScriptedCall::response(tool_call_response("read_note", "read", json!({}))),
        ScriptedCall::response(text_response("The saved note is still here.")),
    ]);
    let mut agent = application.resume(client).await?;
    let work = async {
        agent.prompt("read the saved note").await.result?;
        application.checkpoint(&mut agent).await
    }
    .await;
    let closed = agent.shutdown(ShutdownReason::Completed).await;
    work?;
    closed?;
    Ok(())
}

#[cfg(test)]
#[path = "embedding/recovery.rs"]
mod recovery;

#[cfg(test)]
mod tests {
    use pebble_coding_agent::events::{CodingEvent, ToolErrorKind};

    use super::*;

    #[tokio::test]
    async fn embedding_persists_permissions_cancellation_and_resume() {
        let root = env::temp_dir().join(format!("pebble-embedding-test-{}", Uuid::new_v4()));
        let result = verify(&root).await;
        let cleanup = fs::remove_dir_all(&root).await;
        result.expect("embedding completes");
        cleanup.expect("remove fixtures");
    }

    async fn verify(root: &Path) -> AppResult {
        timeout(Duration::from_secs(15), run(root)).await??;
        assert_eq!(
            fs::read_to_string(root.join("workspace/note.txt")).await?,
            "Remember this after resume."
        );
        let log = fs::read_to_string(root.join("state/events.jsonl")).await?;
        let events: Vec<CodingAgentEvent> = log
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert!(events.windows(2).all(
            |pair| pair[0].seq + 1 == pair[1].seq && pair[0].stream_id() == pair[1].stream_id()
        ));
        for (id, failed) in [
            ("save", false),
            ("denied", true),
            ("cancel", true),
            ("read", false),
        ] {
            let completions: Vec<_> = events.iter().filter(|event| matches!(&event.event, CodingEvent::ToolCallCompleted { tool_call_id, is_error, .. } if tool_call_id == id && *is_error == failed)).collect();
            assert_eq!(completions.len(), 1, "one completion for {id}");
        }
        assert!(events.iter().any(|event| matches!(&event.event, CodingEvent::ToolCallCompleted { tool_call_id, error_kind: Some(ToolErrorKind::Cancelled), .. } if tool_call_id == "cancel")));
        assert!(events.iter().any(|event| matches!(&event.event, CodingEvent::ToolCallCompleted { tool_call_id, output, .. } if tool_call_id == "read" && output.to_string().contains("Remember this after resume."))));
        let stored: SessionRecord =
            serde_json::from_slice(&fs::read(root.join("state/session.json")).await?)?;
        let store = Store::open(&root.join("state")).await?;
        let resumed = store.load().await?;
        assert!(
            resumed.last_event_seq > stored.last_event_seq,
            "shutdown advances the durable log past the checkpoint"
        );
        assert_eq!(resumed.messages, stored.messages);
        Ok(())
    }
}
