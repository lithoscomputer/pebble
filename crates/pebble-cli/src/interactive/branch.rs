//! Durable conversation boundaries and non-destructive branch navigation.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use lithos_llm::types::ContentPart;
use pebble_coding_agent::InputContent;
use pebble_coding_agent::events::{CodingEvent, EventSink as _};
use tokio::fs;

use super::menu::{Menu, Purpose};
use super::store::{Checkpoint, Store};
use super::{App, shell, text};
use crate::storage::atomic_write;

pub(super) async fn checkpoints(store: &Store) -> Result<Vec<u64>> {
    let mut points = Vec::new();
    match fs::read_dir(store.directory().join("checkpoints")).await {
        Ok(mut directory) => {
            while let Some(entry) = directory.next_entry().await? {
                if let Some(seq) = entry
                    .path()
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.parse().ok())
                {
                    points.push(seq);
                }
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    points.sort_unstable();
    points.dedup();
    Ok(points)
}

async fn load_point(store: &Store, seq: u64) -> Result<Checkpoint> {
    let checkpoint: Checkpoint = serde_json::from_slice(
        &fs::read(
            store
                .directory()
                .join("checkpoints")
                .join(format!("{seq}.json")),
        )
        .await
        .context("this boundary predates saved checkpoints; choose a newer point")?,
    )?;
    if checkpoint.record.last_event_seq != seq || seq > store.last_sequence().await? {
        bail!("invalid branch checkpoint");
    }
    Ok(checkpoint)
}

async fn bookmarks(store: &Store) -> Result<BTreeMap<String, u64>> {
    match fs::read(store.directory().join("bookmarks.json")).await {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(error.into()),
    }
}

// Copy bounded checkpoints and the journal prefix. Never reconstruct model
// history from display events or repeat tools to arrive at an earlier state.
async fn create(root: &Path, source: &Store, mut checkpoint: Checkpoint) -> Result<String> {
    let target = Store::open(root, None).await?;
    let id = target.id();
    let old_root = checkpoint.record.session_id.clone();
    let seq = checkpoint.record.last_event_seq;
    let name = format!("{} · branch", checkpoint.metadata.name);
    let source_id = source.id();
    let remap = |checkpoint: &mut Checkpoint| {
        checkpoint.metadata.id.clone_from(&id);
        checkpoint.metadata.name.clone_from(&name);
        checkpoint.metadata.forked_from = Some(source_id.clone());
        checkpoint.metadata.forked_at = Some(seq);
        checkpoint.record.session_id.clone_from(&id);
        checkpoint.record.parent_session_id = None;
    };
    let mut shell_records = shell::records(source).await?;
    let mut reader = source.reader().await?;
    while let Some(mut event) = reader.next().await? {
        if event.seq > seq {
            if let CodingEvent::UserInput { text, .. } = event.event {
                for record in &mut shell_records {
                    if text.contains(&record.marker()) {
                        record.consumed = false;
                    }
                }
            }
            continue;
        }
        if event.session_id == old_root {
            event.session_id.clone_from(&id);
        }
        if event.parent_session_id.as_deref() == Some(&old_root) {
            event.parent_session_id = Some(id.clone());
        }
        event.stream_id.clone_from(&id);
        target.record(&event).await?;
    }
    copy_attachments(source, &target).await?;
    for record in shell_records {
        if record.after_seq <= seq {
            record.save(&target).await?;
        }
    }
    for point in checkpoints(source)
        .await?
        .into_iter()
        .filter(|point| *point <= seq)
    {
        let mut saved = load_point(source, point).await?;
        remap(&mut saved);
        atomic_write(
            target
                .directory()
                .join("checkpoints")
                .join(format!("{point}.json")),
            serde_json::to_vec_pretty(&saved)?,
        )
        .await?;
    }
    let marks: BTreeMap<_, _> = bookmarks(source)
        .await?
        .into_iter()
        .filter(|(_, point)| *point <= seq)
        .collect();
    atomic_write(
        target.directory().join("bookmarks.json"),
        serde_json::to_vec_pretty(&marks)?,
    )
    .await?;
    remap(&mut checkpoint);
    // Publish the complete branch last. Session lists only expose this file.
    target
        .checkpoint(&checkpoint.metadata, checkpoint.record)
        .await?;
    shell::reconcile(&target).await?;
    Ok(id)
}

async fn copy_attachments(source: &Store, target: &Store) -> Result<()> {
    let mut directory = match fs::read_dir(source.directory().join("attachments")).await {
        Ok(directory) => directory,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    while let Some(entry) = directory.next_entry().await? {
        if entry.file_type().await?.is_file() {
            atomic_write(
                target
                    .directory()
                    .join("attachments")
                    .join(entry.file_name()),
                fs::read(entry.path()).await?,
            )
            .await?;
        }
    }
    Ok(())
}

impl App {
    pub(super) async fn clone_session(&mut self) -> Result<()> {
        let checkpoint = Checkpoint {
            metadata: self.metadata.clone(),
            record:   self.worker()?.record().await?,
        };
        let id = create(&self.root, &self.store, checkpoint).await?;
        self.change_session(Some(&id), None).await
    }

    pub(super) async fn fork_session(&mut self, argument: &str) -> Result<()> {
        if let Some(label) = argument.strip_prefix('@') {
            let seq = *bookmarks(&self.store)
                .await?
                .get(label)
                .context("unknown bookmark")?;
            return self.branch_at(seq).await;
        }
        if let Some(seq) = argument.strip_prefix("checkpoint:") {
            return self
                .branch_at(seq.parse().context("invalid checkpoint")?)
                .await;
        }
        let points = checkpoints(&self.store).await?;
        let selected = if argument.is_empty() {
            None
        } else {
            Some(
                argument
                    .parse::<u64>()
                    .context("usage: /fork [input-event | @bookmark]")?,
            )
        };
        let mut reader = self.store.reader().await?;
        let mut items = Vec::new();
        while let Some(event) = reader.next().await? {
            if event.session_id != self.transcript.root {
                continue;
            }
            if let CodingEvent::UserInput {
                text: input,
                content,
                ..
            } = event.event
            {
                let Some(point) = points
                    .iter()
                    .copied()
                    .filter(|point| *point < event.seq)
                    .max()
                else {
                    continue;
                };
                if selected == Some(event.seq) {
                    let checkpoint = load_point(&self.store, point).await?;
                    let id = create(&self.root, &self.store, checkpoint).await?;
                    self.change_session(Some(&id), None).await?;
                    if self.metadata.id != id {
                        bail!("The branch was saved but could not be opened.");
                    }
                    // Shell context is independently restored from the branch's
                    // records. Keep it out of the editable prompt.
                    let parts = content.map_or_else(
                        || vec![ContentPart::Text { text: input }],
                        InputContent::into_parts,
                    );
                    let parts: Vec<_> = parts
                        .into_iter()
                        .filter_map(|part| match part {
                            ContentPart::Text { text } => {
                                let mut rest = text.as_str();
                                while rest.starts_with("[Shell result ") {
                                    let Some((_, after)) =
                                        rest.split_once("[End shell result]\n\n")
                                    else {
                                        break;
                                    };
                                    rest = after;
                                }
                                (!rest.is_empty()).then(|| ContentPart::Text { text: rest.into() })
                            }
                            other => Some(other),
                        })
                        .collect();
                    let draft = self.attachments.render_parts(&parts).await?;
                    self.editor.restore(&draft);
                    self.terminal.message(&format!(
                        "Forked before input #{}; edit the restored prompt and send it when ready.",
                        event.seq
                    ))?;
                    return Ok(());
                }
                items.push((
                    format!(
                        "#{} · {}",
                        event.seq,
                        text::plain(&input)
                            .replace('\n', " ")
                            .chars()
                            .take(160)
                            .collect::<String>()
                    ),
                    format!("/fork {}", event.seq),
                ));
            }
        }
        if selected.is_some() {
            bail!("that input has no saved earlier checkpoint");
        }
        if items.is_empty() {
            bail!("No earlier prompts with saved checkpoints. New turns will appear here.");
        }
        self.menu = Some(Menu::new(Purpose::Navigate, items));
        Ok(())
    }

    async fn branch_at(&mut self, seq: u64) -> Result<()> {
        let checkpoint = load_point(&self.store, seq).await?;
        let id = create(&self.root, &self.store, checkpoint).await?;
        self.change_session(Some(&id), None).await
    }

    pub(super) async fn bookmark(&mut self, label: &str) -> Result<()> {
        let mut marks = bookmarks(&self.store).await?;
        if label.is_empty() {
            self.menu = Some(Menu::new(
                Purpose::Navigate,
                marks
                    .into_iter()
                    .map(|(name, seq)| (format!("{name} · #{seq}"), format!("/fork @{name}")))
                    .collect(),
            ));
            return Ok(());
        }
        if label.len() > 120 || label.chars().any(char::is_control) {
            bail!("Bookmark names must be printable and at most 120 bytes.");
        }
        let record = self.worker()?.record().await?;
        let seq = record.last_event_seq;
        self.store.checkpoint(&self.metadata, record).await?;
        marks.insert(label.into(), seq);
        atomic_write(
            self.store.directory().join("bookmarks.json"),
            serde_json::to_vec_pretty(&marks)?,
        )
        .await?;
        self.terminal.message(&format!(
            "Bookmarked #{seq} as {label}. Use /fork @{label} to branch here."
        ))?;
        Ok(())
    }

    pub(super) async fn tree(&mut self) -> Result<()> {
        let sessions = Store::list(&self.root).await?;
        let mut items = Vec::new();
        let mut ordered = Vec::new();
        for session in &sessions {
            let mut ancestors = Vec::new();
            let mut parent = session.forked_from.as_ref();
            while let Some(id) = parent {
                if ancestors.contains(&id) || ancestors.len() >= 32 {
                    break;
                }
                ancestors.push(id);
                parent = sessions
                    .iter()
                    .find(|candidate| &candidate.id == id)
                    .and_then(|candidate| candidate.forked_from.as_ref());
            }
            let marker = if session.id == self.metadata.id {
                "current · "
            } else {
                ""
            };
            let ancestry = session.forked_from.as_ref().map_or_else(String::new, |id| {
                format!(" · from {id} #{}", session.forked_at.unwrap_or(0))
            });
            let mut order: Vec<_> = ancestors.iter().rev().map(|id| (*id).clone()).collect();
            order.push(session.id.clone());
            ordered.push((
                order,
                (
                    format!(
                        "{}{}{} · {}{ancestry}",
                        "  ↳ ".repeat(ancestors.len()),
                        marker,
                        session.name,
                        session.id
                    ),
                    format!("/resume {}", session.id),
                ),
            ));
        }
        ordered.sort_by(|a, b| a.0.cmp(&b.0));
        items.extend(ordered.into_iter().map(|(_, item)| item));
        let marks = bookmarks(&self.store).await?;
        for seq in checkpoints(&self.store).await? {
            let labels = marks
                .iter()
                .filter(|(_, point)| **point == seq)
                .map(|(label, _)| label.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            items.push((
                format!(
                    "  Current history · #{seq}{}",
                    if labels.is_empty() {
                        String::new()
                    } else {
                        format!(" · ★ {labels}")
                    }
                ),
                format!("/fork checkpoint:{seq}"),
            ));
        }
        self.menu = Some(Menu::new(Purpose::Navigate, items));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::slice;
    use std::time::UNIX_EPOCH;

    use lithos_llm::types::{ImageContent, MediaSource};
    use pebble_coding_agent::events::CodingAgentEvent;
    use pebble_coding_agent::state::SessionRecord;
    use serde_json::json;

    use super::super::attachments::Attachments;
    use super::*;

    #[tokio::test]
    async fn earlier_checkpoint_survives_replaced_history_and_copies_media_and_bookmarks()
    -> Result<()> {
        let root = tempfile::tempdir()?;
        let source = Store::open(root.path(), None).await?;
        let metadata = serde_json::from_value(
            json!({"id":source.id(),"name":"Original","cwd":root.path(),"model":"gpt-5.6","permission":"ReadWrite","reasoning":null,"subagents":false,"updated_at":0}),
        )?;
        let mut record = SessionRecord::new("old-root");
        source
            .record(
                &CodingAgentEvent::new(
                    "old-root",
                    CodingEvent::SessionStarted {
                        provider: None,
                        model:    None,
                    },
                    UNIX_EPOCH,
                )
                .with_seq(1),
            )
            .await?;
        record.advance_event_cursor(1);
        source.checkpoint(&metadata, record.clone()).await?;
        let mut attachments = Attachments::open(source.directory()).await?;
        let media = ContentPart::Image(ImageContent::new(MediaSource::url(
            "https://example.test/image.png",
        )));
        let draft = attachments.render_parts(slice::from_ref(&media)).await?;
        atomic_write(
            source.directory().join("bookmarks.json"),
            serde_json::to_vec(&json!({"before":1,"after":2}))?,
        )
        .await?;
        source
            .record(
                &CodingAgentEvent::new("old-root", CodingEvent::ProcessingEnd, UNIX_EPOCH)
                    .with_seq(2),
            )
            .await?;
        record.advance_event_cursor(2);
        // A later checkpoint can replace all context, as compaction does.
        record.messages = serde_json::from_value(
            json!([{"kind":"compaction","summary":"summary","timestamp":"1970-01-01T00:00:00.000Z"}]),
        )?;
        source.checkpoint(&metadata, record).await?;
        let cleared = shell::Record {
            id:              "cleared".into(),
            command:         "printf cleared".into(),
            output:          "cleared output".into(),
            status:          "exit 0".into(),
            after_seq:       1,
            include_context: true,
            consumed:        true,
        };
        cleared.save(&source).await?;
        let original = fs::read(source.directory().join("checkpoint.json")).await?;
        let branch_id = create(root.path(), &source, load_point(&source, 1).await?).await?;
        let branch = Store::open(root.path(), Some(&branch_id)).await?;
        let checkpoint = branch.load().await?;
        assert!(
            shell::context(&branch).await?.is_empty(),
            "a branch must respect explicitly cleared shell context"
        );
        assert_eq!(checkpoint.record.last_event_seq, 1);
        assert!(checkpoint.record.messages.is_empty());
        assert_eq!(checkpoint.record.session_id, branch_id);
        assert_eq!(
            checkpoint.metadata.forked_from.as_deref(),
            Some(source.id().as_str())
        );
        assert_eq!(checkpoint.metadata.forked_at, Some(1));
        assert_eq!(
            branch
                .reader()
                .await?
                .next()
                .await?
                .context("copied event")?
                .stream_id,
            branch_id
        );
        assert_eq!(
            Attachments::open(branch.directory())
                .await?
                .expand(&draft)
                .await?,
            vec![media]
        );
        assert_eq!(
            bookmarks(&branch).await?,
            BTreeMap::from([("before".into(), 1)])
        );
        assert_eq!(
            fs::read(source.directory().join("checkpoint.json")).await?,
            original
        );
        Ok(())
    }
}
