//! Repository model defaults, kept separate from global application
//! preferences.

use std::io::ErrorKind;
use std::path::{Path, PathBuf, absolute};

use anyhow::{Context as _, Result};
use lithos_llm::types::ReasoningEffort;
use serde_json::{Map, Value};
use tokio::fs;

use super::Settings;
use crate::storage;

pub(crate) struct Defaults {
    pub model:     Option<String>,
    pub reasoning: Option<ReasoningEffort>,
}

/// Find the nearest project file within this Git repository. Worktrees use a
/// `.git` file and have the same boundary. Outside Git, use only the working
/// directory.
pub(crate) async fn path(cwd: &Path) -> Result<PathBuf> {
    let cwd = match fs::canonicalize(cwd).await {
        Ok(path) => path,
        Err(error) if error.kind() == ErrorKind::NotFound => absolute(cwd)?,
        Err(error) => return Err(error).context("locating the project directory"),
    };
    let mut root = None;
    for ancestor in cwd.ancestors() {
        if fs::try_exists(ancestor.join(".git")).await? {
            root = Some(ancestor);
            break;
        }
    }
    let Some(root) = root else {
        return Ok(cwd.join(".pebble/settings.json"));
    };
    for ancestor in cwd.ancestors() {
        let path = ancestor.join(".pebble/settings.json");
        if fs::try_exists(&path).await? {
            return Ok(path);
        }
        if ancestor == root {
            break;
        }
    }
    Ok(root.join(".pebble/settings.json"))
}

async fn read(path: &Path) -> Result<Map<String, Value>> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Map::new()),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub(crate) async fn load(cwd: &Path, global: &Settings) -> Result<Defaults> {
    let path = path(cwd).await?;
    let values = read(&path).await?;
    let model: Option<String> = values
        .get("model")
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()
        .with_context(|| format!("reading model in {}", path.display()))?
        .flatten();
    let reasoning = if let Some(value) = values.get("reasoning") {
        serde_json::from_value(value.clone())
            .with_context(|| format!("reading reasoning in {}", path.display()))?
    } else {
        global.reasoning
    };
    Ok(Defaults {
        model: model.or_else(|| global.model.clone()),
        reasoning,
    })
}

pub(crate) async fn save(
    cwd: &Path,
    model: &str,
    reasoning: Option<ReasoningEffort>,
) -> Result<PathBuf> {
    let path = path(cwd).await?;
    let mut values = read(&path).await?;
    values.insert("model".into(), Value::String(model.into()));
    // Null deliberately overrides a global reasoning preference with the model
    // default.
    values.insert("reasoning".into(), serde_json::to_value(reasoning)?);
    storage::atomic_write(path.clone(), serde_json::to_vec_pretty(&values)?).await?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn discovery_respects_repository_boundaries_and_worktrees() {
        let temp = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(temp.path()).await.unwrap();
        fs::create_dir(root.join(".git")).await.unwrap();
        let nested = root.join("crates/one");
        fs::create_dir_all(&nested).await.unwrap();
        assert_eq!(
            path(&nested).await.unwrap(),
            root.join(".pebble/settings.json")
        );
        save(&root.join("crates"), "root-model", None)
            .await
            .unwrap();
        fs::create_dir_all(root.join("crates/.pebble"))
            .await
            .unwrap();
        fs::write(root.join("crates/.pebble/settings.json"), "{}")
            .await
            .unwrap();
        assert_eq!(
            path(&nested).await.unwrap(),
            root.join("crates/.pebble/settings.json")
        );
        // A nested repository must not inherit its parent's project settings.
        fs::write(nested.join(".git"), "gitdir: /unused/worktree")
            .await
            .unwrap();
        assert_eq!(
            path(&nested).await.unwrap(),
            nested.join(".pebble/settings.json")
        );
        let outside = tempfile::tempdir().unwrap();
        let child = outside.path().join("child");
        fs::create_dir(&child).await.unwrap();
        save(outside.path(), "outside", None).await.unwrap();
        assert_eq!(
            path(&child).await.unwrap(),
            fs::canonicalize(&child)
                .await
                .unwrap()
                .join(".pebble/settings.json")
        );
    }

    #[tokio::test]
    async fn missing_fields_inherit_and_null_reasoning_clears_the_global_default() {
        let root = tempfile::tempdir().unwrap();
        let global = Settings {
            model: Some("global".into()),
            reasoning: Some(ReasoningEffort::High),
            ..Settings::default()
        };
        let defaults = load(root.path(), &global).await.unwrap();
        assert_eq!(defaults.model.as_deref(), Some("global"));
        assert_eq!(defaults.reasoning, Some(ReasoningEffort::High));
        fs::create_dir(root.path().join(".pebble")).await.unwrap();
        fs::write(root.path().join(".pebble/settings.json"), r#"{"model":"project","keep":{"x":1},"keybindings":{"ctrl+x":"unknown-project-action"}}"#).await.unwrap();
        let defaults = load(root.path(), &global).await.unwrap();
        assert_eq!(defaults.model.as_deref(), Some("project"));
        assert_eq!(defaults.reasoning, Some(ReasoningEffort::High));
        let path = save(root.path(), "saved", None).await.unwrap();
        let defaults = load(root.path(), &global).await.unwrap();
        assert_eq!(defaults.reasoning, None);
        let values = read(&path).await.unwrap();
        assert_eq!(values["keep"]["x"], 1);
        assert!(values["keybindings"].is_object());
        fs::write(&path, r#"{"reasoning":"unknown"}"#)
            .await
            .unwrap();
        assert!(load(root.path(), &global).await.is_err());
        fs::write(&path, "[]").await.unwrap();
        assert!(save(root.path(), "saved", None).await.is_err());
        assert_eq!(fs::read_to_string(&path).await.unwrap(), "[]");
    }
}
