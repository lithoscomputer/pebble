//! Persistent application preferences and configurable keyboard actions.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lithos_llm::types::ReasoningEffort;
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::storage;

pub(crate) mod project;

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Settings {
    pub model:           Option<String>,
    pub favorite_models: Vec<String>,
    pub reasoning:       Option<ReasoningEffort>,
    pub show_reasoning:  bool,
    pub expand_tools:    bool,
    pub external_editor: Option<String>,
    pub keybindings:     BTreeMap<String, String>,
    #[serde(flatten)]
    extra:               BTreeMap<String, serde_json::Value>,
}

impl Settings {
    pub(crate) async fn load(path: &Path) -> Result<Self> {
        let bytes = match fs::read(path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
        };
        let settings: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if settings.favorite_models.len() > 256
            || settings.favorite_models.iter().any(|model| {
                model.trim().is_empty() || model.len() > 512 || model.chars().any(char::is_control)
            })
        {
            bail!(
                "favorite_models must contain at most 256 nonempty model selectors, each at most 512 bytes"
            );
        }
        for (key, action) in &settings.keybindings {
            if action_key(action).is_none() {
                bail!("unknown keybinding action {action:?} for {key:?}");
            }
        }
        Ok(settings)
    }

    pub(crate) async fn save(&self, path: &Path) -> Result<()> {
        storage::atomic_write(path.to_path_buf(), serde_json::to_vec_pretty(self)?).await
    }

    pub(crate) fn remap(&self, key: KeyEvent) -> KeyEvent {
        self.keybindings
            .get(&key_name(key))
            .and_then(|action| action_key(action))
            .unwrap_or(key)
    }
}

fn key_name(key: KeyEvent) -> String {
    let mut name = String::new();
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        name.push_str("ctrl+");
    }
    if key.modifiers.contains(KeyModifiers::ALT) {
        name.push_str("alt+");
    }
    if key.modifiers.contains(KeyModifiers::SHIFT) {
        name.push_str("shift+");
    }
    name.push_str(&match key.code {
        KeyCode::Char(value) => value.to_lowercase().to_string(),
        KeyCode::Enter => "enter".into(),
        KeyCode::Esc => "esc".into(),
        KeyCode::Tab | KeyCode::BackTab => "tab".into(),
        KeyCode::Up => "up".into(),
        KeyCode::Down => "down".into(),
        KeyCode::Left => "left".into(),
        KeyCode::Right => "right".into(),
        KeyCode::Backspace => "backspace".into(),
        KeyCode::Delete => "delete".into(),
        KeyCode::Home => "home".into(),
        KeyCode::End => "end".into(),
        other => format!("{other:?}").to_lowercase(),
    });
    name
}

fn action_key(action: &str) -> Option<KeyEvent> {
    let (code, modifiers) = match action {
        "submit" => (KeyCode::Enter, KeyModifiers::NONE),
        "follow-up" => (KeyCode::Enter, KeyModifiers::ALT),
        "newline" => (KeyCode::Enter, KeyModifiers::SHIFT),
        "cancel" => (KeyCode::Esc, KeyModifiers::NONE),
        "quit" => (KeyCode::Char('q'), KeyModifiers::CONTROL),
        "paste-image" => (KeyCode::Char('v'), KeyModifiers::CONTROL),
        "external-editor" => (KeyCode::Char('g'), KeyModifiers::CONTROL),
        "toggle-tools" => (KeyCode::Char('o'), KeyModifiers::CONTROL),
        "toggle-reasoning" => (KeyCode::Char('t'), KeyModifiers::CONTROL),
        "recover-input" => (KeyCode::Up, KeyModifiers::ALT),
        "model-picker" => (KeyCode::Char('l'), KeyModifiers::CONTROL),
        "reload" => (KeyCode::Char('r'), KeyModifiers::CONTROL),
        "next-model" => (KeyCode::Char('p'), KeyModifiers::CONTROL),
        "previous-model" => (
            KeyCode::Char('p'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        ),
        "cycle-thinking" => (KeyCode::BackTab, KeyModifiers::SHIFT),
        "complete" => (KeyCode::Tab, KeyModifiers::NONE),
        "history-up" => (KeyCode::Up, KeyModifiers::NONE),
        "history-down" => (KeyCode::Down, KeyModifiers::NONE),
        "undo" => (KeyCode::Char('_'), KeyModifiers::CONTROL),
        "suspend" => (KeyCode::Char('z'), KeyModifiers::CONTROL),
        _ => return None,
    };
    Some(KeyEvent::new(code, modifiers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_bindings_override_the_action_without_changing_other_keys() {
        let settings = Settings {
            keybindings: BTreeMap::from([("ctrl+e".into(), "external-editor".into())]),
            ..Settings::default()
        };
        assert_eq!(
            settings.remap(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL)),
            KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL)
        );
        assert_eq!(settings.remap(KeyCode::Left.into()), KeyCode::Left.into());
    }
}
