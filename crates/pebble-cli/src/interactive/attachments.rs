//! Attachment placeholders backed by local files, including recalled drafts.

use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hasher as _};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use lithos_llm::types::{ContentPart, ImageContent, MediaSource};
use tokio::fs;

use crate::storage;

pub(super) struct Attachments {
    directory: PathBuf,
    entries:   BTreeMap<String, String>,
}

impl Attachments {
    pub(super) async fn open(session: &Path) -> Result<Self> {
        let directory = session.join("attachments");
        let entries = match fs::read(directory.join("index.json")).await {
            Ok(bytes) => serde_json::from_slice(&bytes).context("reading the attachment index")?,
            Err(error) if error.kind() == ErrorKind::NotFound => BTreeMap::new(),
            Err(error) => return Err(error.into()),
        };
        let entries: BTreeMap<String, String> = entries;
        if entries.iter().any(|(marker, file)| {
            marker.is_empty()
                || file.is_empty()
                || !file
                    .chars()
                    .all(|value| value.is_ascii_alphanumeric() || value == '-' || value == '.')
        }) {
            bail!("invalid attachment index");
        }
        Ok(Self { directory, entries })
    }

    pub(super) async fn attach_image(&mut self, path: &Path) -> Result<String> {
        let media_type = match path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            _ => bail!(
                "Attach a PNG, JPEG, GIF, or WebP image. Reference text files with @ or a path."
            ),
        };
        if fs::metadata(path).await?.len() > 5 * 1024 * 1024 {
            bail!("Images must be at most 5 MiB.");
        }
        let bytes = fs::read(path)
            .await
            .with_context(|| format!("reading {}", path.display()))?;
        if bytes.len() > 5 * 1024 * 1024 {
            bail!("The image grew past 5 MiB while being read.");
        }
        let data = STANDARD.encode(bytes);
        self.register(&ContentPart::Image(ImageContent::new(MediaSource::base64(
            data, media_type,
        ))))
        .await
    }

    /// Restores the exact order of text and media using editable placeholders.
    pub(super) async fn render_parts(&mut self, parts: &[ContentPart]) -> Result<String> {
        let mut draft = String::new();
        for part in parts {
            if let ContentPart::Text { text } = part {
                draft.push_str(text);
            } else {
                draft.push_str(&self.register(part).await?);
            }
        }
        Ok(draft)
    }

    pub(super) async fn expand(&self, draft: &str) -> Result<Vec<ContentPart>> {
        let mut parts = Vec::new();
        let mut rest = draft;
        loop {
            let found = self
                .entries
                .iter()
                .filter_map(|(marker, file)| rest.find(marker).map(|offset| (offset, marker, file)))
                .min_by_key(|(offset, _, _)| *offset);
            let Some((offset, marker, file)) = found else {
                break;
            };
            if offset > 0 {
                parts.push(ContentPart::Text {
                    text: rest[..offset].into(),
                });
            }
            if !file
                .chars()
                .all(|value| value.is_ascii_alphanumeric() || value == '-' || value == '.')
            {
                bail!("invalid attachment cache entry");
            }
            let bytes = fs::read(self.directory.join(file))
                .await
                .context("loading the attached content")?;
            parts.push(serde_json::from_slice(&bytes).context("parsing the attached content")?);
            rest = &rest[offset + marker.len()..];
        }
        if !rest.is_empty() {
            parts.push(ContentPart::Text { text: rest.into() });
        }
        Ok(parts)
    }

    async fn register(&mut self, part: &ContentPart) -> Result<String> {
        let bytes = serde_json::to_vec(part)?;
        let mut hasher = DefaultHasher::new();
        hasher.write(&bytes);
        let mut name = format!("{:016x}.json", hasher.finish());
        if let Ok(existing) = fs::read(self.directory.join(&name)).await {
            // Hashing only indexes this cache. A collision must never select different
            // content.
            if existing != bytes {
                name = format!("{}.json", uuid::Uuid::new_v4());
            } else if let Some((marker, _)) = self.entries.iter().find(|(_, file)| **file == name) {
                return Ok(marker.clone());
            }
        }
        storage::atomic_write(self.directory.join(&name), bytes).await?;
        let kind = match part {
            ContentPart::Image(_) => "image",
            ContentPart::Audio(_) => "audio",
            ContentPart::Document(_) => "document",
            _ => "attachment",
        };
        let marker = format!("[{kind} {}]", self.entries.len() + 1);
        self.entries.insert(marker.clone(), name);
        storage::atomic_write(
            self.directory.join("index.json"),
            serde_json::to_vec_pretty(&self.entries)?,
        )
        .await?;
        Ok(marker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queued_media_keeps_its_order_and_survives_reopening_the_editor() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let parts = vec![
            ContentPart::Text {
                text: "Compare ".into(),
            },
            ContentPart::Image(ImageContent::new(MediaSource::url(
                "https://example.test/one.png",
            ))),
            ContentPart::Text {
                text: " with ".into(),
            },
            ContentPart::Image(ImageContent::new(MediaSource::url(
                "https://example.test/two.png",
            ))),
        ];
        let mut attachments = Attachments::open(directory.path()).await?;
        let draft = attachments.render_parts(&parts).await?;
        drop(attachments);
        let reopened = Attachments::open(directory.path()).await?;
        assert_eq!(reopened.expand(&draft).await?, parts);
        assert_eq!(reopened.expand("ordinary prompt").await?, vec![
            ContentPart::Text {
                text: "ordinary prompt".into(),
            }
        ]);
        Ok(())
    }
}
