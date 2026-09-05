//! Refresh resources while retaining the session and editor state.

use anyhow::{Context as _, Result};

use super::{App, Worker};
use crate::settings::{Settings, project};

impl App {
    pub(super) async fn reload(&mut self) -> Result<()> {
        self.require_idle()?;
        // Validate both preference files before replacing any live state.
        // Model and reasoning are session choices; these files supply defaults.
        let settings = Settings::load(&self.settings_path).await?;
        project::load(&self.metadata.cwd, &settings).await?;
        let mut previous = self.worker()?.export().await?;
        let environment = self.worker()?.environment.clone();
        self.terminal
            .message("Reloading settings, skills, and instructions…")?;
        if let Some(worker) = self.worker.take() {
            worker.shutdown().await?;
        }
        let mut record = previous.record().clone();
        record.advance_event_cursor(self.store.last_sequence().await?);
        let replacement = Worker::start(
            self.client.clone(),
            self.metadata.clone(),
            self.store.clone(),
            Some(record),
            self.services.clone(),
        )
        .await;
        match replacement {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => {
                // Keep the old prompt and skill bodies even if the files that
                // supplied them have changed. Failed initialization may have
                // appended events, so the restored pump starts above them.
                previous.advance_event_cursor(self.store.last_sequence().await?);
                self.worker = Some(
                    Worker::restore(
                        self.client.clone(),
                        self.metadata.clone(),
                        self.store.clone(),
                        previous,
                        environment,
                        self.services.clone(),
                    )
                    .await
                    .context("restoring the agent after reload failed")?,
                );
                self.replay(false).await?;
                return Err(error)
                    .context("reload failed; previous settings and resources restored");
            }
        }
        self.settings = settings;
        self.transcript.show_reasoning = self.settings.show_reasoning;
        self.transcript.expand_tools = self.settings.expand_tools;
        self.completion_files = None;
        self.replay(false).await?;
        self.terminal.message(&format!(
            "Reloaded settings, keybindings, {} skills, and {} instruction files. Current model and reasoning kept.",
            self.worker()?.snapshot.skills().len(),
            self.worker()?.snapshot.memory().len(),
        ))?;
        Ok(())
    }
}
