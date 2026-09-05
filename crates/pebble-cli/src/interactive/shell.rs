//! User shell commands, bounded live output, and durable next-prompt context.

use std::io::ErrorKind;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use lithos_llm::types::ContentPart;
use pebble_coding_agent::events::CodingEvent;
#[cfg(unix)]
use rustix::process::{Pid, Signal, kill_process_group};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::AsyncReadExt as _;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use super::services::Services;
use super::store::{Metadata, Store};
use crate::application::PermissionArg;
use crate::storage::atomic_write;

const OUTPUT_LIMIT: usize = 64 * 1024;

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct Record {
    pub id:              String,
    pub command:         String,
    pub output:          String,
    pub status:          String,
    pub after_seq:       u64,
    pub include_context: bool,
    #[serde(default)]
    pub consumed:        bool,
}

impl Record {
    pub(super) fn marker(&self) -> String {
        format!("[Shell result {}]", self.id)
    }

    pub(super) fn display(&self) -> String {
        format!(
            "$ {}\n{}\nShell: {}{}",
            self.command,
            self.output,
            self.status,
            if self.include_context {
                " · included in next prompt"
            } else {
                " · excluded from model context"
            }
        )
    }

    pub(super) async fn save(&self, store: &Store) -> Result<()> {
        atomic_write(
            store
                .directory()
                .join("shell")
                .join(format!("{}.json", self.id)),
            serde_json::to_vec(self)?,
        )
        .await
    }
}

pub(super) struct Job {
    pub output: mpsc::Receiver<String>,
    pub task:   JoinHandle<Result<Record>>,
    pub cancel: CancellationToken,
}

impl Job {
    pub(super) async fn start(
        command: &str,
        include_context: bool,
        metadata: Metadata,
        store: Arc<Store>,
        services: Arc<Services>,
    ) -> Result<Self> {
        if metadata.permission != PermissionArg::Full && !metadata.approvals {
            bail!("Shell commands need full permission or per-command approval.");
        }
        let record = Record {
            id: uuid::Uuid::new_v4().to_string(),
            command: command.into(),
            output: String::new(),
            status: "interrupted before completion".into(),
            after_seq: store.last_sequence().await?,
            include_context,
            consumed: false,
        };
        record.save(&store).await?;
        let (sender, output) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let task = tokio::spawn(async move {
            let mut record = record;
            if metadata.permission != PermissionArg::Full
                && !services.approve_shell(&record.command, &token).await
            {
                record.status = "denied".into();
                record.include_context = false;
            } else {
                match execute(&record.command, &metadata.cwd, &token, sender).await {
                    Ok((output, status)) => {
                        record.output = output;
                        record.status = status;
                    }
                    Err(error) => record.status = format!("failed: {error:#}"),
                }
            }
            record.save(&store).await?;
            Ok(record)
        });
        Ok(Self {
            output,
            task,
            cancel,
        })
    }

    pub(super) async fn shutdown(self) -> Result<()> {
        self.cancel.cancel();
        drop(self.output);
        self.task.await.context("joining the shell command")??;
        Ok(())
    }
}

// The process group is owned by this invocation. Background descendants must
// not outlive it, including when a pipe read fails or this future is dropped.
pub(super) struct ProcessGroup(pub(super) Option<u32>);
impl ProcessGroup {
    pub(super) fn terminate(&self) {
        #[cfg(unix)]
        if let Some(pid) = self
            .0
            .and_then(|id| i32::try_from(id).ok())
            .and_then(Pid::from_raw)
        {
            let _ = kill_process_group(pid, Signal::TERM);
        }
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self
            .0
            .and_then(|id| i32::try_from(id).ok())
            .and_then(Pid::from_raw)
        {
            let _ = kill_process_group(pid, Signal::KILL);
        }
    }
}

async fn execute(
    command: &str,
    cwd: &Path,
    cancel: &CancellationToken,
    sender: mpsc::Sender<String>,
) -> Result<(String, String)> {
    if cancel.is_cancelled() {
        return Ok((String::new(), "cancelled".into()));
    }
    let mut process = Command::new("/bin/sh");
    process
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        process.as_std_mut().process_group(0);
    }
    let mut child = process.spawn().context("starting the shell command")?;
    let mut group = Some(ProcessGroup(child.id()));
    let mut stdout = child.stdout.take().context("opening shell stdout")?;
    let mut stderr = child.stderr.take().context("opening shell stderr")?;
    let mut out = [0; 4096];
    let mut err = [0; 4096];
    let (mut out_open, mut err_open) = (true, true);
    let mut status = None;
    let mut retained = Vec::new();
    let mut line = Vec::new();
    let mut omitted = 0_usize;
    let mut skipped = false;
    while status.is_none() || out_open || err_open {
        let bytes = tokio::select! {
            biased;
            () = cancel.cancelled(), if status.is_none() => {
                if let Some(group) = &group { group.terminate(); }
                // A child may be mid-fork when TERM is delivered. Keep the group
                // owned through a grace period, then kill surviving descendants.
                sleep(Duration::from_millis(100)).await;
                drop(group.take()); child.kill().await.ok(); child.wait().await?;
                status = Some("cancelled".into()); continue;
            }
            result = child.wait(), if status.is_none() => {
                let exit = result?;
                status = Some(exit.code().map_or_else(|| "terminated by signal".into(), |code| format!("exit {code}")));
                if let Some(group) = &group { group.terminate(); }
                sleep(Duration::from_millis(100)).await;
                drop(group.take()); continue;
            }
            result = stdout.read(&mut out), if out_open => { let count = result?; out_open = count != 0; &out[..count] }
            result = stderr.read(&mut err), if err_open => { let count = result?; err_open = count != 0; &err[..count] }
            () = sleep(Duration::from_millis(100)), if !line.is_empty() => {
                skipped |= sender.try_send(String::from_utf8_lossy(&line).into()).is_err();
                line.clear(); continue;
            }
            () = sleep(Duration::from_secs(1)), if status.is_some() => break,
        };
        let keep = bytes.len().min(OUTPUT_LIMIT.saturating_sub(retained.len()));
        retained.extend_from_slice(&bytes[..keep]);
        omitted = omitted.saturating_add(bytes.len() - keep);
        // Bound rendering as well as storage. Drain the pipes even after the cap.
        for byte in &bytes[..keep] {
            line.push(*byte);
            if *byte == b'\n' || line.len() >= 8192 {
                skipped |= sender
                    .try_send(String::from_utf8_lossy(&line).trim_end_matches('\n').into())
                    .is_err();
                line.clear();
            }
        }
    }
    if !line.is_empty() {
        skipped |= sender
            .try_send(String::from_utf8_lossy(&line).into())
            .is_err();
    }
    let mut output = String::from_utf8_lossy(&retained).into_owned();
    if omitted > 0 {
        use std::fmt::Write as _;
        write!(output, "\n[{omitted} output bytes omitted]")?;
    }
    let mut status = status.unwrap_or_else(|| "interrupted".into());
    if skipped {
        status.push_str(" · some live output skipped; /shells shows retained output");
    }
    Ok((output, status))
}

pub(super) async fn records(store: &Store) -> Result<Vec<Record>> {
    let mut directory = match fs::read_dir(store.directory().join("shell")).await {
        Ok(directory) => directory,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut records = Vec::new();
    while let Some(entry) = directory.next_entry().await? {
        if entry.path().extension().is_some_and(|ext| ext == "json") {
            records.push(serde_json::from_slice::<Record>(
                &fs::read(entry.path()).await?,
            )?);
        }
    }
    records.sort_by(|a, b| a.after_seq.cmp(&b.after_seq).then(a.id.cmp(&b.id)));
    Ok(records)
}

// Reconcile before advancing a recovered journal cursor. Only input covered by
// a saved checkpoint consumes shell context; an interrupted turn can retry it.
pub(super) async fn reconcile(store: &Store) -> Result<()> {
    let checkpoint = store.load().await?;
    let mut records = records(store).await?;
    let mut reader = store.reader().await?;
    while let Some(event) = reader.next().await? {
        if event.seq > checkpoint.record.last_event_seq {
            break;
        }
        if event.session_id == checkpoint.record.session_id
            && let CodingEvent::UserInput { text, .. } = event.event
        {
            for record in &mut records {
                if !record.consumed
                    && record.include_context
                    && event.seq > record.after_seq
                    && text.contains(&record.marker())
                {
                    record.consumed = true;
                    record.save(store).await?;
                }
            }
        }
    }
    Ok(())
}

pub(super) async fn context(store: &Store) -> Result<Vec<ContentPart>> {
    let records = records(store).await?;
    let pending: Vec<_> = records
        .iter()
        .filter(|record| record.include_context && !record.consumed)
        .collect();
    if pending
        .iter()
        .map(|r| r.command.len() + r.output.len())
        .sum::<usize>()
        > 512 * 1024
    {
        bail!("Pending shell context exceeds 512 KiB. Use /shells clear before sending a prompt.");
    }
    Ok(pending
        .iter()
        .map(|record| ContentPart::Text {
            text: format!(
                "{}\n{}\n[End shell result]\n\n",
                record.marker(),
                record.display()
            ),
        })
        .collect())
}

pub(super) async fn clear(store: &Store) -> Result<()> {
    for mut record in records(store).await? {
        record.consumed = true;
        record.save(store).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tokio::time::timeout;

    use super::*;

    #[tokio::test]
    async fn cancellation_stops_descendants_and_keeps_partial_output() -> Result<()> {
        let cwd = tempfile::tempdir()?;
        let (sender, mut output) = mpsc::channel(10);
        let cancel = CancellationToken::new();
        let operation = execute(
            "printf 'started\\n'; (sleep 2; touch escaped) & wait",
            cwd.path(),
            &cancel,
            sender,
        );
        tokio::pin!(operation);
        tokio::select! {
            result = &mut operation => bail!("command ended before cancellation: {result:?}"),
            line = output.recv() => assert_eq!(line.as_deref(), Some("started")),
        }
        cancel.cancel();
        let (output, status) = timeout(Duration::from_secs(2), operation).await??;
        assert_eq!(status, "cancelled");
        assert_eq!(output, "started\n");
        sleep(Duration::from_millis(2100)).await;
        assert!(!cwd.path().join("escaped").exists());
        Ok(())
    }

    #[tokio::test]
    async fn flood_output_is_drained_and_bounded_even_without_a_live_reader() -> Result<()> {
        let cwd = tempfile::tempdir()?;
        let (sender, _output) = mpsc::channel(1);
        let (output, status) = timeout(
            Duration::from_secs(5),
            execute(
                "head -c 200000 /dev/zero | tr '\\000' x; printf error >&2; exit 7",
                cwd.path(),
                &CancellationToken::new(),
                sender,
            ),
        )
        .await??;
        assert!(status.starts_with("exit 7"));
        assert!(output.len() < OUTPUT_LIMIT + 100);
        assert!(output.contains("output bytes omitted"));
        Ok(())
    }
}
