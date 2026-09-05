//! Bounded credential input with no history, undo buffer, or debug output.

use std::io::{self, IsTerminal as _, Write as _};
use std::mem;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::{execute, terminal};
use lithos_llm::credentials::SecretValue;
use tokio::signal::ctrl_c;
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};
use tokio::task::spawn_blocking;
use tokio_util::sync::CancellationToken;

use crate::credentials::{MAX_KEY_BYTES, valid_secret};

#[derive(Default)]
pub(crate) struct SecretInput {
    value: String,
}

pub(crate) enum Action {
    Editing,
    Submit(SecretValue),
    Cancel,
}

impl SecretInput {
    pub(crate) fn masked(&self) -> String {
        "*".repeat(self.value.len().min(60))
    }

    pub(crate) fn paste(&mut self, text: &str) -> bool {
        let text = text.trim();
        if !valid_secret(text) || self.value.len().saturating_add(text.len()) > MAX_KEY_BYTES {
            return false;
        }
        self.value.push_str(text);
        true
    }

    pub(crate) fn key(&mut self, key: KeyEvent) -> Action {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => return Action::Cancel,
            KeyCode::Char('c' | 'd' | 'q') if control => return Action::Cancel,
            KeyCode::Char('u') if control => self.value.clear(),
            KeyCode::Backspace => {
                self.value.pop();
            }
            KeyCode::Enter if !self.value.trim().is_empty() => {
                return Action::Submit(SecretValue::new(
                    mem::take(&mut self.value).trim().to_owned(),
                ));
            }
            KeyCode::Char(value)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && value.is_ascii()
                    && !value.is_control()
                    && self.value.len() < MAX_KEY_BYTES =>
            {
                self.value.push(value);
            }
            _ => {}
        }
        Action::Editing
    }
}

pub(crate) async fn prompt() -> Result<SecretValue> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        bail!("login needs a terminal; use --stdin to read an API key from a pipe");
    }
    let cancel = CancellationToken::new();
    let reader_cancel = cancel.clone();
    // Install shutdown handlers before raw mode is enabled in the reader.
    #[cfg(unix)]
    let mut terminate = signal(SignalKind::terminate())?;
    #[cfg(unix)]
    let mut hangup = signal(SignalKind::hangup())?;
    let shutdown = async {
        #[cfg(unix)]
        tokio::select! { _ = terminate.recv() => {}, _ = hangup.recv() => {}, _ = ctrl_c() => {} }
        #[cfg(not(unix))]
        {
            let _ = ctrl_c().await;
        }
    };
    let mut reader = spawn_blocking(move || read_terminal(&reader_cancel));
    let result = tokio::select! {
        result = &mut reader => result,
        () = shutdown => { cancel.cancel(); reader.await }
    };
    result
        .context("joining credential input")??
        .context("login cancelled")
}

fn read_terminal(cancel: &CancellationToken) -> Result<Option<SecretValue>> {
    struct RawInput;
    impl Drop for RawInput {
        fn drop(&mut self) {
            let _ = execute!(io::stderr(), event::DisableBracketedPaste);
            let _ = terminal::disable_raw_mode();
            let _ = writeln!(io::stderr());
        }
    }
    terminal::enable_raw_mode()?;
    let _raw = RawInput;
    execute!(io::stderr(), event::EnableBracketedPaste)?;
    let mut input = SecretInput::default();
    loop {
        write!(io::stderr(), "\rAPI key: {}\x1b[K", input.masked())?;
        io::stderr().flush()?;
        if cancel.is_cancelled() {
            return Ok(None);
        }
        if !event::poll(Duration::from_millis(50))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => match input.key(key) {
                Action::Submit(value) => return Ok(Some(value)),
                Action::Cancel => return Ok(None),
                Action::Editing => {}
            },
            Event::Paste(text) => {
                input.paste(&text);
            }
            _ => {}
        }
    }
}

pub(crate) async fn from_stdin() -> Result<SecretValue> {
    if io::stdin().is_terminal() {
        bail!("--stdin expects a pipe; omit it for masked terminal input");
    }
    spawn_blocking(|| -> Result<SecretValue> {
        use std::io::Read as _;
        let mut value = String::new();
        io::stdin()
            .lock()
            // Permit a full-size key followed by CRLF, plus one byte to
            // detect oversized input without buffering the whole pipe.
            .take((MAX_KEY_BYTES + 3) as u64)
            .read_to_string(&mut value)
            .context("reading the API key from standard input")?;
        if value.len() > MAX_KEY_BYTES + 2 || !valid_secret(value.trim()) {
            bail!("API key must be nonempty printable ASCII, at most 8192 bytes");
        }
        Ok(SecretValue::new(value.trim().to_owned()))
    })
    .await
    .context("joining credential input")?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_is_masked_and_editing_cannot_restore_submitted_credentials() {
        let mut input = SecretInput::default();
        assert!(input.paste("fixture-secret\n"));
        assert!(!input.masked().contains("fixture"));
        let Action::Submit(key) = input.key(KeyCode::Enter.into()) else {
            panic!("submit key");
        };
        assert_eq!(key.expose_secret(), "fixture-secret");
        assert!(input.masked().is_empty());
        input.key(KeyCode::Up.into());
        input.key(KeyEvent::new(KeyCode::Char('_'), KeyModifiers::CONTROL));
        assert!(input.masked().is_empty());
        assert!(!input.paste("secret\nsecond-line"));
        assert!(!input.paste(&"a".repeat(MAX_KEY_BYTES + 1)));
    }
}
