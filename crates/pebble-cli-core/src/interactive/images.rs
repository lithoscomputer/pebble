//! Bounded image input and terminal graphics encoded only from validated bytes.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use std::{env, str};

#[cfg(not(target_os = "macos"))]
use anyhow::bail;
use anyhow::{Context as _, Result, ensure};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use lithos_llm::types::ImageContent;
#[cfg(unix)]
use rustix::fs::OFlags;
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncReadExt as _;
use tokio::process::Command;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use super::shell::ProcessGroup;

pub(super) const LIMIT: usize = 5 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Protocol {
    Kitty,
    Iterm,
}

impl Protocol {
    pub(super) fn detect() -> Option<Self> {
        detect(&|key| env::var(key).ok())
    }
}

fn detect(get: &impl Fn(&str) -> Option<String>) -> Option<Protocol> {
    match get("PEBBLE_IMAGE_PROTOCOL").as_deref() {
        Some("kitty") => return Some(Protocol::Kitty),
        Some("iterm2") => return Some(Protocol::Iterm),
        Some("none" | "0") => return None,
        _ => {}
    }
    let term = get("TERM").unwrap_or_default().to_lowercase();
    if get("TMUX").is_some()
        || get("STY").is_some()
        || term.starts_with("screen")
        || term.starts_with("tmux")
    {
        return None;
    }
    let program = get("TERM_PROGRAM").unwrap_or_default().to_lowercase();
    if get("KITTY_WINDOW_ID").is_some()
        || matches!(program.as_str(), "kitty" | "ghostty")
        || term.contains("kitty")
        || term.contains("ghostty")
    {
        return Some(Protocol::Kitty);
    }
    if get("ITERM_SESSION_ID").is_some() || matches!(program.as_str(), "iterm.app" | "wezterm") {
        return Some(Protocol::Iterm);
    }
    None
}

pub(super) struct Image {
    pub bytes:  Vec<u8>,
    pub mime:   &'static str,
    pub width:  u32,
    pub height: u32,
}

impl Image {
    pub(super) fn parse(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() <= LIMIT, "Images must be at most 5 MiB.");
        let (mime, width, height) =
            dimensions(&bytes).context("Invalid image header. Use PNG, JPEG, GIF, or WebP.")?;
        ensure!(
            width > 0 && height > 0 && u64::from(width) * u64::from(height) <= 64_000_000,
            "Images must contain 1 to 64 million pixels."
        );
        Ok(Self {
            bytes,
            mime,
            width,
            height,
        })
    }

    pub(super) fn cells(&self, columns: u16, rows: u16) -> (u16, u16) {
        // Use the same conservative 9x18 fallback as Pi. Both protocols fit the
        // original image inside this cell rectangle without changing its bytes.
        let max_columns = u64::from(columns.clamp(1, 60));
        let max_rows = u64::from(rows.clamp(1, 12));
        let width = u64::from(self.width);
        let height = u64::from(self.height);
        let mut columns = width.div_ceil(9).min(max_columns).max(1);
        let mut rows = (height * columns).div_ceil(width * 2).max(1);
        if rows > max_rows {
            rows = max_rows;
            columns = (width * rows * 2).div_ceil(height).clamp(1, max_columns);
        }
        (
            u16::try_from(columns).unwrap_or(1),
            u16::try_from(rows).unwrap_or(1),
        )
    }

    pub(super) fn encode(&self, protocol: Protocol, columns: u16, rows: u16) -> Option<String> {
        let data = STANDARD.encode(&self.bytes);
        match protocol {
            Protocol::Kitty if self.mime == "image/png" => {
                let id = u32::from_le_bytes(uuid::Uuid::new_v4().as_bytes()[..4].try_into().ok()?)
                    .max(1);
                let mut encoded = String::new();
                let chunks = data.as_bytes().chunks(4096);
                let count = chunks.len();
                for (index, chunk) in chunks.enumerate() {
                    let more = u8::from(index + 1 < count);
                    if index == 0 {
                        write!(
                            encoded,
                            "\x1b_Ga=T,f=100,q=2,C=1,i={id},c={columns},r={rows},m={more};"
                        )
                        .ok()?;
                    } else {
                        write!(encoded, "\x1b_Gm={more};").ok()?;
                    }
                    encoded.push_str(str::from_utf8(chunk).ok()?);
                    encoded.push_str("\x1b\\");
                }
                Some(encoded)
            }
            Protocol::Iterm => Some(format!(
                "\x1b]1337;File=inline=1;size={};width={columns};height={rows};preserveAspectRatio=1:{data}\x07",
                self.bytes.len()
            )),
            Protocol::Kitty => None,
        }
    }
}

fn dimensions(bytes: &[u8]) -> Option<(&'static str, u32, u32)> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        if bytes.get(8..16)? != b"\0\0\0\rIHDR"
            || bytes.len() < 45
            || !bytes.ends_with(b"IEND\xaeB`\x82")
        {
            return None;
        }
        return Some((
            "image/png",
            u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?),
            u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?),
        ));
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        if bytes.last() != Some(&0x3b) {
            return None;
        }
        return Some((
            "image/gif",
            u32::from(u16::from_le_bytes(bytes.get(6..8)?.try_into().ok()?)),
            u32::from(u16::from_le_bytes(bytes.get(8..10)?.try_into().ok()?)),
        ));
    }
    if bytes.starts_with(b"\xff\xd8") {
        let mut at = 2;
        while at < bytes.len() {
            if *bytes.get(at)? != 0xff {
                return None;
            }
            while *bytes.get(at)? == 0xff {
                at += 1;
            }
            let marker = *bytes.get(at)?;
            at += 1;
            if marker == 0xd9 || marker == 0xda {
                return None;
            }
            if marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
                continue;
            }
            let length = usize::from(u16::from_be_bytes(bytes.get(at..at + 2)?.try_into().ok()?));
            if length < 2 || at + length > bytes.len() {
                return None;
            }
            if matches!(marker, 0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf) {
                if length < 8 {
                    return None;
                }
                return Some((
                    "image/jpeg",
                    u32::from(u16::from_be_bytes(
                        bytes.get(at + 5..at + 7)?.try_into().ok()?,
                    )),
                    u32::from(u16::from_be_bytes(
                        bytes.get(at + 3..at + 5)?.try_into().ok()?,
                    )),
                ));
            }
            at += length;
        }
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12)? == b"WEBP" {
        let size = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?);
        if u64::from(size) + 8 != u64::try_from(bytes.len()).ok()? {
            return None;
        }
        let le24 = |at| {
            let v: &[u8] = bytes.get(at..at + 3)?;
            Some(u32::from(v[0]) | (u32::from(v[1]) << 8) | (u32::from(v[2]) << 16))
        };
        return match bytes.get(12..16)? {
            b"VP8X" => Some(("image/webp", le24(24)? + 1, le24(27)? + 1)),
            b"VP8L" if *bytes.get(20)? == 0x2f => {
                let bits = u32::from_le_bytes(bytes.get(21..25)?.try_into().ok()?);
                Some((
                    "image/webp",
                    (bits & 0x3fff) + 1,
                    ((bits >> 14) & 0x3fff) + 1,
                ))
            }
            b"VP8 " if bytes.get(23..26)? == b"\x9d\x01\x2a" => Some((
                "image/webp",
                u32::from(u16::from_le_bytes(bytes.get(26..28)?.try_into().ok()?) & 0x3fff),
                u32::from(u16::from_le_bytes(bytes.get(28..30)?.try_into().ok()?) & 0x3fff),
            )),
            _ => None,
        };
    }
    None
}

pub(super) enum Source {
    File(PathBuf),
    Clipboard,
}
pub(super) struct Prepared {
    pub original: Image,
    pub preview:  Option<Image>,
}
pub(super) struct Job {
    pub task:   JoinHandle<Result<Prepared>>,
    pub cancel: CancellationToken,
}

impl Job {
    pub(super) fn start(source: Source) -> Self {
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let task = tokio::spawn(async move {
            let bytes = match source {
                Source::File(path) => read_file(&path).await?,
                Source::Clipboard => clipboard(&token).await?,
            };
            let original = Image::parse(bytes)?;
            // Save a PNG preview for Kitty and future resume in another terminal.
            // Original bytes always remain the model attachment.
            let preview = if original.mime == "image/png" {
                None
            } else {
                png_preview(&original, &token).await.ok()
            };
            ensure!(!token.is_cancelled(), "Image read cancelled.");
            Ok(Prepared { original, preview })
        });
        Self { task, cancel }
    }

    pub(super) async fn shutdown(self) -> Result<()> {
        self.cancel.cancel();
        let _ = self.task.await.context("joining image input")?;
        Ok(())
    }
}

pub(super) async fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(i32::try_from(OFlags::NONBLOCK.bits())?);
    let file = options
        .open(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        file.metadata().await?.is_file(),
        "Attach a regular image file."
    );
    let mut bytes = Vec::new();
    file.take(u64::try_from(LIMIT)? + 1)
        .read_to_end(&mut bytes)
        .await?;
    ensure!(bytes.len() <= LIMIT, "Images must be at most 5 MiB.");
    Ok(bytes)
}

async fn capture(mut command: Command, cancel: &CancellationToken) -> Result<Vec<u8>> {
    ensure!(!cancel.is_cancelled(), "Image read cancelled.");
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.as_std_mut().process_group(0);
    }
    let mut child = command.spawn().context("starting the image helper")?;
    let group = ProcessGroup(child.id());
    let output = child.stdout.take().context("reading image helper output")?;
    let result = tokio::select! {
        () = cancel.cancelled() => Err(anyhow::anyhow!("Image read cancelled.")),
        result = timeout(Duration::from_secs(3), async {
            let mut bytes = Vec::new();
            output.take(u64::try_from(LIMIT)? + 1).read_to_end(&mut bytes).await?;
            ensure!(bytes.len() <= LIMIT, "Image helper output exceeds 5 MiB.");
            ensure!(child.wait().await?.success(), "Image helper found no supported image.");
            Ok(bytes)
        }) => result.unwrap_or_else(|error| Err(anyhow::anyhow!("image helper timed out: {error}"))),
    };
    group.terminate();
    sleep(Duration::from_millis(100)).await;
    drop(group);
    let _ = child.kill().await;
    let _ = child.wait().await;
    result
}

async fn clipboard(cancel: &CancellationToken) -> Result<Vec<u8>> {
    if let Ok(helper) = env::var("PEBBLE_CLIPBOARD_COMMAND") {
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(helper);
        return capture(command, cancel).await;
    }
    #[cfg(target_os = "macos")]
    {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("clipboard.png");
        let script = r"on run argv
set imageData to the clipboard as «class PNGf»
set targetFile to open for access POSIX file (item 1 of argv) with write permission
try
set eof targetFile to 0
write imageData to targetFile
close access targetFile
on error messageText
close access targetFile
error messageText
end try
end run";
        let mut command = Command::new("osascript");
        command.arg("-e").arg(script).arg(&path);
        capture(command, cancel)
            .await
            .context("No clipboard image is available. Copy an image or use /attach <path>.")?;
        return read_file(&path).await;
    }
    #[cfg(not(target_os = "macos"))]
    {
        for mime in ["image/png", "image/jpeg", "image/webp", "image/gif"] {
            for (program, args) in [
                ("wl-paste", vec!["--type", mime, "--no-newline"]),
                ("xclip", vec!["-selection", "clipboard", "-t", mime, "-o"]),
            ] {
                ensure!(!cancel.is_cancelled(), "Image read cancelled.");
                let mut command = Command::new(program);
                command.args(args);
                if let Ok(bytes) = capture(command, cancel).await
                    && !bytes.is_empty()
                {
                    return Ok(bytes);
                }
            }
        }
        bail!(
            "No clipboard image is available. Use wl-paste, xclip, PEBBLE_CLIPBOARD_COMMAND, or /attach <path>."
        )
    }
}

async fn png_preview(image: &Image, cancel: &CancellationToken) -> Result<Image> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("image");
    fs::write(&path, &image.bytes).await?;
    #[cfg(target_os = "macos")]
    {
        let output = directory.path().join("preview.png");
        let mut command = Command::new("sips");
        command.args(["-s", "format", "png"]);
        if image.width.max(image.height) > 1024 {
            command.args(["--resampleHeightWidthMax", "1024"]);
        }
        command.arg(&path).arg("--out").arg(&output);
        capture(command, cancel).await?;
        Image::parse(read_file(&output).await?)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut command = Command::new("magick");
        command
            .arg(format!("{}[0]", path.display()))
            .args(["-resize", "1024x1024>", "png:-"]);
        Image::parse(capture(command, cancel).await?)
    }
}

pub(super) fn from_content(content: &ImageContent) -> Result<Option<Image>> {
    use lithos_llm::types::MediaSource;
    let MediaSource::Base64 { data, media_type } = &content.source else {
        return Ok(None);
    };
    ensure!(
        data.len() <= LIMIT.div_ceil(3) * 4,
        "Saved image exceeds 5 MiB."
    );
    let image = Image::parse(STANDARD.decode(data).context("decoding the saved image")?)?;
    ensure!(
        image.mime == media_type,
        "Saved image media type does not match its header."
    );
    Ok(Some(image))
}

pub(super) fn markdown(content: &ImageContent) -> Result<Option<String>> {
    Ok(from_content(content)?.map(|image| {
        format!(
            "![Attached image ({} × {})](data:{};base64,{})\n\n",
            image.width,
            image.height,
            image.mime,
            STANDARD.encode(image.bytes)
        )
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use anyhow::bail;

    use super::*;

    const PNG: &[u8] = include_bytes!("../../tests/fixtures/tiny.png");

    #[test]
    fn formats_dimensions_and_limits_are_checked_before_rendering() -> Result<()> {
        let image = Image::parse(PNG.to_vec())?;
        assert_eq!((image.width, image.height, image.mime), (3, 2, "image/png"));
        for end in 0..PNG.len() {
            assert!(Image::parse(PNG[..end].to_vec()).is_err());
        }
        let mut huge = PNG.to_vec();
        huge[16..20].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(Image::parse(huge).is_err());
        assert!(Image::parse(vec![0; LIMIT + 1]).is_err());
        assert!(Image::parse(b"not an image\x1b_Ginjected".to_vec()).is_err());
        let gif = STANDARD.decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")?;
        let gif = Image::parse(gif)?;
        assert_eq!((gif.width, gif.height, gif.mime), (1, 1, "image/gif"));
        assert!(gif.encode(Protocol::Kitty, 1, 1).is_none());
        assert!(gif.encode(Protocol::Iterm, 1, 1).is_some());
        Ok(())
    }

    #[test]
    fn graphic_payloads_are_bounded_chunks_and_cells_fit_the_viewport() -> Result<()> {
        let image = Image {
            bytes:  vec![0xab; 16000],
            mime:   "image/png",
            width:  4000,
            height: 3000,
        };
        assert_eq!(image.cells(1, 1), (1, 1));
        let (columns, rows) = image.cells(60, 10);
        assert!(columns <= 60 && rows <= 10);
        let encoded = image
            .encode(Protocol::Kitty, columns, rows)
            .context("kitty graphics")?;
        assert!(encoded.starts_with("\x1b_Ga=T,f=100,q=2,C=1,"));
        let mut payload = String::new();
        for chunk in encoded.split("\x1b\\").filter(|chunk| !chunk.is_empty()) {
            let (_, data) = chunk.split_once(';').context("graphics separator")?;
            assert!(data.len() <= 4096);
            payload.push_str(data);
        }
        assert!(encoded.contains("\x1b_Gm=0;"));
        assert_eq!(STANDARD.decode(payload)?, image.bytes);
        let iterm = image
            .encode(Protocol::Iterm, columns, rows)
            .context("iterm graphics")?;
        assert!(iterm.starts_with("\x1b]1337;File=inline=1;size=16000;"));
        assert!(iterm.ends_with('\x07'));
        Ok(())
    }

    #[test]
    fn capability_detection_is_conservative_and_explicitly_overridable() {
        let detect_values = |values: &[(&str, &str)]| {
            let values: BTreeMap<_, _> = values.iter().copied().collect();
            detect(&|key| values.get(key).map(|value| (*value).into()))
        };
        assert_eq!(detect_values(&[]), None);
        assert_eq!(
            detect_values(&[("TERM_PROGRAM", "ghostty")]),
            Some(Protocol::Kitty)
        );
        assert_eq!(
            detect_values(&[("TERM_PROGRAM", "iTerm.app")]),
            Some(Protocol::Iterm)
        );
        assert_eq!(
            detect_values(&[("TERM_PROGRAM", "ghostty"), ("TMUX", "session")]),
            None
        );
        assert_eq!(
            detect_values(&[
                ("TERM_PROGRAM", "ghostty"),
                ("PEBBLE_IMAGE_PROTOCOL", "none")
            ]),
            None
        );
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn jpeg_preview_conversion_preserves_the_original_image() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let png = directory.path().join("image.png");
        let jpeg = directory.path().join("image.jpg");
        fs::write(&png, PNG).await?;
        let mut command = Command::new("sips");
        command
            .args(["-s", "format", "jpeg"])
            .arg(&png)
            .arg("--out")
            .arg(&jpeg);
        capture(command, &CancellationToken::new()).await?;
        let original = read_file(&jpeg).await?;
        let prepared = Job::start(Source::File(jpeg)).task.await??;
        assert_eq!(prepared.original.mime, "image/jpeg");
        assert_eq!((prepared.original.width, prepared.original.height), (3, 2));
        assert_eq!(prepared.original.bytes, original);
        let preview = prepared.preview.context("PNG conversion")?;
        assert_eq!(preview.mime, "image/png");
        assert_eq!((preview.width, preview.height), (3, 2));
        Ok(())
    }

    #[tokio::test]
    async fn image_helper_cancellation_reaps_children_and_caps_output() -> Result<()> {
        let cancel = CancellationToken::new();
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "sleep 30"]);
        let operation = capture(command, &cancel);
        tokio::pin!(operation);
        tokio::select! { result = &mut operation => bail!("helper ended early: {result:?}"), () = sleep(Duration::from_millis(50)) => {} }
        cancel.cancel();
        assert!(timeout(Duration::from_secs(2), operation).await?.is_err());
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "head -c 6000000 /dev/zero"]);
        let error = capture(command, &CancellationToken::new())
            .await
            .err()
            .context("oversized output must fail")?;
        assert!(error.to_string().contains("exceeds 5 MiB"));
        Ok(())
    }
}
