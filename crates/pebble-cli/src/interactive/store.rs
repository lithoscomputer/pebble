//! Local event journals and atomic idle checkpoints.

use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs as sync_fs, io};

use anyhow::{Context as _, Result, bail};
use async_trait::async_trait;
use lithos_llm::types::ReasoningEffort;
use pebble_coding_agent::events::{CodingAgentEvent, EventSink, EventSinkError};
use pebble_coding_agent::state::SessionRecord;
#[cfg(unix)]
use rustix::fs::{FlockOperation, flock};
use serde::{Deserialize, Serialize};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::sync::Mutex;
use tokio::task::spawn_blocking;

use crate::application::PermissionArg;
use crate::storage::atomic_write;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Metadata {
    pub id:           String,
    pub name:         String,
    pub cwd:          PathBuf,
    pub model:        String,
    pub permission:   PermissionArg,
    pub reasoning:    Option<ReasoningEffort>,
    pub subagents:    bool,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default = "approvals_default")]
    pub approvals:    bool,
    pub updated_at:   u64,
    #[serde(default)]
    pub forked_from:  Option<String>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Checkpoint {
    pub metadata: Metadata,
    pub record:   SessionRecord,
}

pub(super) struct Store {
    directory:         PathBuf,
    file:              Mutex<File>,
    // An advisory OS lock is released even if the process crashes.
    _lock:             sync_fs::File,
    pub repaired_tail: bool,
}

impl Store {
    pub(super) async fn open(root: &Path, id: Option<&str>) -> Result<Arc<Self>> {
        let id = id.map_or_else(|| uuid::Uuid::new_v4().to_string(), str::to_owned);
        if id.is_empty()
            || !id
                .chars()
                .all(|value| value.is_ascii_alphanumeric() || value == '-')
        {
            bail!("invalid session id: {id}");
        }
        let directory = root.join(&id);
        let prepare = directory.clone();
        let (lock, repaired_tail) = spawn_blocking(move || -> Result<_> {
            #[cfg(unix)]
            use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};

            use io::{Read as _, Seek as _, SeekFrom};

            let mut builder = sync_fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            builder.mode(0o700);
            builder.create(&prepare)?;
            let mut options = sync_fs::OpenOptions::new();
            options.create(true).read(true).write(true).truncate(false);
            #[cfg(unix)]
            options.mode(0o600);
            let lock = options.open(prepare.join("session.lock"))?;
            #[cfg(unix)]
            flock(&lock, FlockOperation::NonBlockingLockExclusive)
                .context("this session is already open in another Pebble process")?;
            let mut log = options.open(prepare.join("events.jsonl"))?;
            let length = log.metadata()?.len();
            let mut repaired = false;
            if length > 0 {
                log.seek(SeekFrom::End(-1))?;
                let mut last = [0];
                log.read_exact(&mut last)?;
                if last != *b"\n" {
                    let mut end = length;
                    let boundary = loop {
                        let start = end.saturating_sub(8192);
                        log.seek(SeekFrom::Start(start))?;
                        let mut buffer = vec![0; usize::try_from(end - start)?];
                        log.read_exact(&mut buffer)?;
                        if let Some(offset) = buffer.iter().rposition(|byte| *byte == b'\n') {
                            break start + u64::try_from(offset)? + 1;
                        }
                        if start == 0 {
                            break 0;
                        }
                        end = start;
                    };
                    log.seek(SeekFrom::Start(boundary))?;
                    let mut partial =
                        options.open(prepare.join(format!("events.partial-{}", timestamp())))?;
                    io::copy(&mut log, &mut partial)?;
                    partial.sync_all()?;
                    log.set_len(boundary)?;
                    log.sync_all()?;
                    repaired = true;
                }
            }
            Ok((lock, repaired))
        })
        .await
        .context("joining session storage setup")??;
        let file = OpenOptions::new()
            .append(true)
            .open(directory.join("events.jsonl"))
            .await?;
        Ok(Arc::new(Self {
            directory,
            file: Mutex::new(file),
            _lock: lock,
            repaired_tail,
        }))
    }

    pub(super) fn id(&self) -> String {
        self.directory
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }

    pub(super) fn directory(&self) -> &Path {
        &self.directory
    }

    pub(super) async fn checkpoint(
        &self,
        metadata: &Metadata,
        record: SessionRecord,
    ) -> Result<()> {
        let mut metadata = metadata.clone();
        metadata.updated_at = timestamp();
        let content = serde_json::to_vec_pretty(&Checkpoint { metadata, record })?;
        atomic_write(self.directory.join("checkpoint.json"), content).await
    }

    pub(super) async fn load(&self) -> Result<Checkpoint> {
        let bytes = fs::read(self.directory.join("checkpoint.json"))
            .await
            .context("reading the session checkpoint")?;
        let checkpoint: Checkpoint =
            serde_json::from_slice(&bytes).context("parsing the session checkpoint")?;
        if checkpoint.record.last_event_seq > self.last_sequence().await? {
            bail!(
                "the session journal ends before its checkpoint; restore the missing journal before resuming"
            );
        }
        Ok(checkpoint)
    }

    pub(super) async fn reader(&self) -> Result<EventReader> {
        Ok(EventReader {
            reader: BufReader::new(File::open(self.directory.join("events.jsonl")).await?),
            line:   String::new(),
        })
    }

    pub(super) async fn last_sequence(&self) -> Result<u64> {
        let mut reader = self.reader().await?;
        let mut seq = 0;
        while let Some(event) = reader.next().await? {
            seq = seq.max(event.seq);
        }
        Ok(seq)
    }

    pub(super) async fn list(root: &Path) -> Result<Vec<Metadata>> {
        let mut directory = match fs::read_dir(root).await {
            Ok(directory) => directory,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut sessions = Vec::new();
        while let Some(entry) = directory.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let path = entry.path().join("checkpoint.json");
            if let Ok(bytes) = fs::read(path).await
                && let Ok(checkpoint) = serde_json::from_slice::<Checkpoint>(&bytes)
            {
                sessions.push(checkpoint.metadata);
            }
        }
        sessions.sort_by_key(|metadata| Reverse(metadata.updated_at));
        Ok(sessions)
    }
}

#[async_trait]
impl EventSink for Store {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        let mut bytes = serde_json::to_vec(event).map_err(|error| {
            EventSinkError::new("serializing a session event").with_source(error)
        })?;
        bytes.push(b'\n');
        let mut file = self.file.lock().await;
        file.write_all(&bytes)
            .await
            .map_err(|error| EventSinkError::new("writing a session event").with_source(error))?;
        file.sync_data()
            .await
            .map_err(|error| EventSinkError::new("saving a session event").with_source(error))
    }
}

pub(super) struct EventReader {
    reader: BufReader<File>,
    line:   String,
}

impl EventReader {
    pub(super) async fn next(&mut self) -> Result<Option<CodingAgentEvent>> {
        self.line.clear();
        if self.reader.read_line(&mut self.line).await? == 0 {
            return Ok(None);
        }
        // A live writer may not have finished this line. A later replay starts
        // from the last applied sequence and reads the completed event.
        if !self.line.ends_with('\n') {
            return Ok(None);
        }
        Ok(Some(
            serde_json::from_str(&self.line).context("parsing a saved session event")?,
        ))
    }
}

fn approvals_default() -> bool {
    true
}

pub(super) fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pebble_coding_agent::events::CodingEvent;

    use super::*;

    #[tokio::test]
    async fn journal_repairs_only_the_incomplete_tail_and_retains_sequence() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = Store::open(root.path(), Some("session-test")).await?;
        let event = CodingAgentEvent::new(
            "session",
            CodingEvent::SessionEnded,
            UNIX_EPOCH + Duration::from_secs(10),
        )
        .with_seq(7);
        store.record(&event).await?;
        let log = store.directory().join("events.jsonl");
        drop(store);
        let mut file = OpenOptions::new().append(true).open(log).await?;
        file.write_all(b"{broken").await?;
        file.flush().await?;
        drop(file);
        let recovered = Store::open(root.path(), Some("session-test")).await?;
        assert!(recovered.repaired_tail);
        assert_eq!(recovered.last_sequence().await?, 7);
        assert_eq!(recovered.reader().await?.next().await?, Some(event));
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_session_has_one_writer_and_can_reopen_after_the_owner_exits() -> Result<()> {
        let root = tempfile::tempdir()?;
        let first = Store::open(root.path(), Some("same")).await?;
        assert!(Store::open(root.path(), Some("same")).await.is_err());
        drop(first);
        assert!(Store::open(root.path(), Some("same")).await.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn a_truncated_journal_cannot_resume_history_from_a_later_checkpoint() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = Store::open(root.path(), None).await?;
        let event =
            CodingAgentEvent::new("root", CodingEvent::SessionEnded, UNIX_EPOCH).with_seq(1);
        store.record(&event).await?;
        let metadata = Metadata {
            id:           store.id(),
            name:         "Saved".into(),
            cwd:          root.path().into(),
            model:        "model".into(),
            permission:   PermissionArg::ReadWrite,
            reasoning:    None,
            subagents:    false,
            instructions: None,
            approvals:    true,
            updated_at:   0,
            forked_from:  None,
        };
        let mut record = SessionRecord::new("root");
        record.advance_event_cursor(1);
        store.checkpoint(&metadata, record).await?;
        assert!(store.load().await.is_ok());
        fs::write(store.directory().join("events.jsonl"), b"").await?;
        let error = store
            .load()
            .await
            .err()
            .context("truncated journal must fail")?;
        assert!(
            error
                .to_string()
                .contains("journal ends before its checkpoint")
        );
        Ok(())
    }
}
