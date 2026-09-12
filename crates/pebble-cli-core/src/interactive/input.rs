//! An owned input reader that can stop before another program uses the
//! terminal.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context as _, Result};
use crossterm::event::{self, Event};
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, spawn_blocking};

pub(super) struct Input {
    receiver: mpsc::Receiver<io::Result<Event>>,
    stop:     Arc<AtomicBool>,
    task:     Option<JoinHandle<()>>,
}

impl Input {
    pub(super) fn start() -> Self {
        let (sender, receiver) = mpsc::channel(128);
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let task = spawn_blocking(move || {
            while !stopped.load(Ordering::Acquire) {
                let next = match event::poll(Duration::from_millis(25)) {
                    Ok(true) => event::read(),
                    Ok(false) => continue,
                    Err(error) => Err(error),
                };
                let failed = next.is_err();
                if sender.blocking_send(next).is_err() || failed {
                    break;
                }
            }
        });
        Self {
            receiver,
            stop,
            task: Some(task),
        }
    }

    pub(super) async fn recv(&mut self) -> Option<io::Result<Event>> {
        self.receiver.recv().await
    }

    pub(super) async fn shutdown(mut self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        self.receiver.close();
        if let Some(task) = self.task.take() {
            task.await.context("joining terminal input")?;
        }
        Ok(())
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.receiver.close();
    }
}
