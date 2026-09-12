//! Private, atomic application file writes.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use tokio::task::spawn_blocking;

pub(crate) fn private_directory(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .with_context(|| format!("creating {}", path.display()))
}

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("the saved file needs a parent directory")?;
    private_directory(parent)?;
    // NamedTempFile creates mode 0600 on Unix, including when replacing an
    // existing file with broader permissions.
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .context("replacing the saved file")?;
    #[cfg(unix)]
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

pub(crate) async fn atomic_write(path: PathBuf, bytes: Vec<u8>) -> Result<()> {
    spawn_blocking(move || write_atomic(&path, &bytes))
        .await
        .context("joining the file writer")?
}
