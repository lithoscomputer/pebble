//! The application's conventional locations for instructions and skills.

use std::env;
use std::path::{Path, PathBuf, absolute};

use anyhow::Result;
use pebble_coding_agent::CodingAgentOptions;
use tokio::fs;

use crate::application::Paths;

pub(crate) async fn options(cwd: &Path, options: CodingAgentOptions) -> Result<CodingAgentOptions> {
    let paths = Paths::from_env()?;
    let home = absolute(paths.home)?;
    let cwd = fs::canonicalize(cwd).await?;
    let mut ancestors = vec![cwd.as_path()];
    let mut found_root = false;
    for ancestor in cwd.ancestors() {
        if ancestor != cwd {
            ancestors.push(ancestor);
        }
        if fs::try_exists(ancestor.join(".git")).await? {
            found_root = true;
            break;
        }
    }
    if !found_root {
        ancestors.truncate(1);
    }
    ancestors.reverse();
    let mut memory = vec![home.join("AGENTS.md")];
    let mut skills = Vec::new();
    if let Some(home) = env::var_os("HOME").filter(|home| !home.is_empty()) {
        skills.push(PathBuf::from(home).join(".agents/skills"));
    }
    skills.push(home.join("skills"));
    for ancestor in ancestors {
        memory.push(ancestor.join("AGENTS.md"));
        skills.push(ancestor.join(".agents/skills"));
        skills.push(ancestor.join(".pebble/skills"));
    }
    Ok(options
        .with_memory_files(
            memory
                .iter()
                .map(|path| path.to_string_lossy().into_owned()),
        )
        .with_skill_dirs(
            skills
                .iter()
                .map(|path| path.to_string_lossy().into_owned()),
        ))
}
