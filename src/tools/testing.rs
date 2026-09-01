//! What the built-in tools' own tests build a call out of.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{env, fs, process};

use lithos_llm::types::ToolDefinitionKind;
use serde_json::Value;

use crate::environment::Environment;
use crate::tool::{RegisteredTool, ToolContext};

/// A call context acting on `environment` and nothing else.
pub(crate) fn context(environment: impl Environment + 'static) -> ToolContext {
    context_for(Arc::new(environment))
}

/// A call context acting on a shared environment, so a test can keep a handle
/// on what the tool did to it.
pub(crate) fn context_for<E: Environment + 'static>(environment: Arc<E>) -> ToolContext {
    ToolContext::new(environment)
}

/// One tool's parameter schema.
pub(crate) fn schema_of(tool: &RegisteredTool) -> &Value {
    match &tool.definition.kind {
        ToolDefinitionKind::Function { input_schema } => input_schema,
        other => panic!("a built-in tool is a function tool, not {other:?}"),
    }
}

/// A directory that deletes itself when the test ends.
///
/// A tool that has to be tested against real files — `apply_patch` reads a file
/// back to match a hunk against it — needs somewhere to put them.
pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// A fresh directory named after `label`, the current process, and a
    /// counter, so parallel tests never share one.
    pub(crate) fn new(label: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = env::temp_dir().join(format!(
            "pebble-{label}-{}-{}",
            process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("temporary directory is creatable");
        Self { path }
    }

    /// Where the directory is.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Writes a fixture file, creating the directories above it.
    pub(crate) fn write(&self, relative: &str, content: &str) {
        let path = self.path.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent directory is creatable");
        }
        fs::write(path, content).expect("fixture is writable");
    }

    /// Reads a file back.
    pub(crate) fn read(&self, relative: &str) -> String {
        fs::read_to_string(self.path.join(relative)).expect("the file is readable")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
