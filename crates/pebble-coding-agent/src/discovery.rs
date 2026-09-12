//! Where a session looks for its project memory and its skills, when the
//! application names a convention rather than a list.
//!
//! [`CodingAgentOptions::with_memory_files`](crate::CodingAgentOptions::with_memory_files)
//! and [`with_skill_dirs`](crate::CodingAgentOptions::with_skill_dirs) take
//! explicit paths. Both embedders built those lists the same way: the
//! instruction files the session's harness reads, in every directory from
//! the repository root down to the working directory, root first; and a few
//! skill directories, some under the repository root, some named by a
//! workflow and expected to exist. [`MemoryDiscovery`] and [`SkillDiscovery`]
//! name those conventions, and pebble does the walking, the git-root probe,
//! and the existence checks through the session's [`Environment`].
//!
//! Which files a harness reads is the profile's knowledge:
//! [`AgentProfileKind::memory_filenames`].

use tokio_util::sync::CancellationToken;

use crate::environment::{Environment, ExecRequest};
use crate::error::{Error, InterruptReason, Result};
use crate::types::{AgentProfileKind, SkippedSkill, SkippedSkillReason};

/// How long the git-root probe may take.
const PROBE_TIMEOUT_MS: u64 = 5_000;

/// The topmost directory memory files are looked for in.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryRoot {
    /// The working directory alone.
    WorkingDirectory,
    /// The root of the git repository the working directory is in, found by
    /// asking the environment; the working directory alone outside one.
    GitRoot,
    /// A directory the application names. The working directory must be
    /// inside it; otherwise only the working directory is read.
    Path(String),
}

/// Where the instruction files a session's profile reads are looked for.
///
/// Every directory from the root down to the working directory is a
/// candidate, root first, each holding the profile's filenames in the
/// profile's order, so a nested project's instructions come after the
/// repository's and the loader charges them against the budget in that order.
/// Missing files are skipped by the loader; nothing is probed first.
///
/// ```
/// use pebble_coding_agent::events::AgentProfileKind;
/// use pebble_coding_agent::{CodingAgentOptions, MemoryDiscovery};
///
/// let options =
///     CodingAgentOptions::default().with_memory_discovery(MemoryDiscovery::from_git_root());
/// # let _ = options;
/// assert_eq!(
///     MemoryDiscovery::candidates(
///         AgentProfileKind::Anthropic,
///         Some("/repo"),
///         "/repo/crates/app"
///     ),
///     [
///         "/repo/AGENTS.md",
///         "/repo/CLAUDE.md",
///         "/repo/crates/AGENTS.md",
///         "/repo/crates/CLAUDE.md",
///         "/repo/crates/app/AGENTS.md",
///         "/repo/crates/app/CLAUDE.md",
///     ]
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDiscovery {
    root: MemoryRoot,
}

impl MemoryDiscovery {
    /// The profile's files in the working directory alone.
    #[must_use]
    pub const fn working_directory() -> Self {
        Self {
            root: MemoryRoot::WorkingDirectory,
        }
    }

    /// The profile's files from the git root down to the working directory.
    #[must_use]
    pub const fn from_git_root() -> Self {
        Self {
            root: MemoryRoot::GitRoot,
        }
    }

    /// The profile's files from `root` down to the working directory.
    pub fn from_root(root: impl Into<String>) -> Self {
        Self {
            root: MemoryRoot::Path(root.into()),
        }
    }

    /// Where the walk starts.
    #[must_use]
    pub const fn root(&self) -> &MemoryRoot {
        &self.root
    }

    /// The candidate paths for `profile` in every directory from `root` down
    /// to `working_dir`, root first. With no root, or a working directory
    /// outside the root, the working directory alone.
    #[must_use]
    pub fn candidates(
        profile: AgentProfileKind,
        root: Option<&str>,
        working_dir: &str,
    ) -> Vec<String> {
        let working_dir = trim_slash(working_dir);
        let dirs: Vec<String> = match root.map(trim_slash) {
            Some(root) if working_dir == root => vec![root.to_owned()],
            Some(root) if working_dir.starts_with(&format!("{root}/")) => {
                let mut dirs = vec![root.to_owned()];
                let mut current = root.to_owned();
                for component in working_dir[root.len() + 1..].split('/') {
                    current = format!("{current}/{component}");
                    dirs.push(current.clone());
                }
                dirs
            }
            _ => vec![working_dir.to_owned()],
        };
        dirs.iter()
            .flat_map(|dir| {
                profile
                    .memory_filenames()
                    .iter()
                    .map(move |name| format!("{dir}/{name}"))
            })
            .collect()
    }

    /// The candidate paths for `profile` in `env`, probing the git root when
    /// the discovery asks for it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Interrupted`] when `cancel` fires around the probe.
    pub async fn resolve(
        &self,
        env: &dyn Environment,
        profile: AgentProfileKind,
        cancel: &CancellationToken,
    ) -> Result<Vec<String>> {
        let root = match &self.root {
            MemoryRoot::WorkingDirectory => None,
            MemoryRoot::Path(path) => Some(path.clone()),
            MemoryRoot::GitRoot => git_root(env, cancel).await?,
        };
        Ok(Self::candidates(
            profile,
            root.as_deref(),
            env.working_directory(),
        ))
    }
}

/// Where a skill directory's path is anchored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SkillSearchBase {
    /// As given; a relative path is under the working directory.
    WorkingDirectory,
    /// Under the git root, or the working directory outside a repository.
    GitRoot,
}

/// One directory a skill discovery searches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillSearch {
    path:     String,
    base:     SkillSearchBase,
    required: bool,
}

impl SkillSearch {
    /// The path as the application gave it.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Where a relative path is anchored.
    #[must_use]
    pub const fn base(&self) -> SkillSearchBase {
        self.base
    }

    /// Whether an absent directory is reported rather than passed over.
    #[must_use]
    pub const fn is_required(&self) -> bool {
        self.required
    }
}

/// The directories a session's skills are discovered in, in search order,
/// later ones overriding earlier names.
///
/// A conventional directory that is absent is ordinary and silent, as it is
/// for an explicit
/// [`with_skill_dirs`](crate::CodingAgentOptions::with_skill_dirs); a
/// directory the application [requires](Self::require) is reported as
/// [`SkippedSkillReason::MissingDirectory`] when it is not there, so a
/// workflow that named it learns of the mistake.
///
/// ```
/// use pebble_coding_agent::{CodingAgentOptions, SkillDiscovery};
///
/// let options = CodingAgentOptions::default().with_skill_discovery(
///     SkillDiscovery::new()
///         .search("/home/me/.fabro/skills")
///         .search_under_git_root(".fabro/skills")
///         .search_under_git_root("skills")
///         .require("tools/skills"),
/// );
/// # let _ = options;
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SkillDiscovery {
    searches: Vec<SkillSearch>,
}

impl SkillDiscovery {
    /// A discovery that searches nowhere yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            searches: Vec::new(),
        }
    }

    /// Searches `dir`, skipping it silently when absent. A relative path is
    /// under the working directory.
    #[must_use]
    pub fn search(mut self, dir: impl Into<String>) -> Self {
        self.searches.push(SkillSearch {
            path:     dir.into(),
            base:     SkillSearchBase::WorkingDirectory,
            required: false,
        });
        self
    }

    /// Searches `dir` under the git root (the working directory outside a
    /// repository), skipping it silently when absent.
    #[must_use]
    pub fn search_under_git_root(mut self, dir: impl Into<String>) -> Self {
        self.searches.push(SkillSearch {
            path:     dir.into(),
            base:     SkillSearchBase::GitRoot,
            required: false,
        });
        self
    }

    /// Searches `dir`, reporting it as missing when absent. A relative path
    /// is under the working directory.
    #[must_use]
    pub fn require(mut self, dir: impl Into<String>) -> Self {
        self.searches.push(SkillSearch {
            path:     dir.into(),
            base:     SkillSearchBase::WorkingDirectory,
            required: true,
        });
        self
    }

    /// The searches, in order.
    #[must_use]
    pub fn searches(&self) -> &[SkillSearch] {
        &self.searches
    }

    /// Whether any search is anchored at the git root.
    #[must_use]
    pub fn needs_git_root(&self) -> bool {
        self.searches
            .iter()
            .any(|search| search.base == SkillSearchBase::GitRoot)
    }

    /// The directories to search in `env`, in order, and the required ones
    /// that are not there. `git_root` is the probe's answer when a search
    /// asked for it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Interrupted`] when `cancel` fires around a check.
    pub async fn resolve(
        &self,
        env: &dyn Environment,
        git_root: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<ResolvedSkillDirs> {
        let working_dir = trim_slash(env.working_directory());
        let mut dirs = Vec::with_capacity(self.searches.len());
        let mut skipped = Vec::new();
        for search in &self.searches {
            let anchor = match search.base {
                SkillSearchBase::WorkingDirectory => working_dir,
                SkillSearchBase::GitRoot => git_root.map_or(working_dir, trim_slash),
            };
            let path = if search.path.starts_with('/') {
                search.path.clone()
            } else {
                format!("{anchor}/{}", trim_slash(&search.path))
            };
            if search.required {
                if cancel.is_cancelled() {
                    return Err(Error::Interrupted(InterruptReason::Cancelled));
                }
                let exists = env.file_exists(&path).await;
                if cancel.is_cancelled() {
                    return Err(Error::Interrupted(InterruptReason::Cancelled));
                }
                if !matches!(exists, Ok(true)) {
                    skipped.push(SkippedSkill {
                        path,
                        reason: SkippedSkillReason::MissingDirectory,
                        message: "the required skills directory does not exist".to_owned(),
                    });
                    continue;
                }
            }
            if !dirs.contains(&path) {
                dirs.push(path);
            }
        }
        Ok(ResolvedSkillDirs { dirs, skipped })
    }
}

/// What a [`SkillDiscovery`] resolved to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedSkillDirs {
    /// The directories to search, in order, each once.
    pub dirs:    Vec<String>,
    /// The required directories that are not there.
    pub skipped: Vec<SkippedSkill>,
}

/// The root of the git repository `env`'s working directory is in, or `None`
/// outside one or where git is unavailable.
///
/// # Errors
///
/// Returns [`Error::Interrupted`] when `cancel` fires around the probe.
pub(crate) async fn git_root(
    env: &dyn Environment,
    cancel: &CancellationToken,
) -> Result<Option<String>> {
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }
    let outcome = env
        .exec(ExecRequest {
            timeout_ms: Some(PROBE_TIMEOUT_MS),
            cancel_token: Some(cancel.child_token()),
            ..ExecRequest::new("git rev-parse --show-toplevel")
        })
        .await;
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }
    let Ok(outcome) = outcome else {
        return Ok(None);
    };
    if !outcome.result.is_success() {
        return Ok(None);
    }
    let root = outcome.result.stdout.trim();
    Ok((!root.is_empty() && root.starts_with('/')).then(|| root.to_owned()))
}

fn trim_slash(path: &str) -> &str {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() { "/" } else { trimmed }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::environment::ExecResult;
    use crate::test_support::MockEnvironment;
    use crate::types::CommandTermination;

    #[test]
    fn candidates_walk_root_first_then_the_working_dir() {
        assert_eq!(
            MemoryDiscovery::candidates(
                AgentProfileKind::Anthropic,
                Some("/repo"),
                "/repo/crates/app"
            ),
            [
                "/repo/AGENTS.md",
                "/repo/CLAUDE.md",
                "/repo/crates/AGENTS.md",
                "/repo/crates/CLAUDE.md",
                "/repo/crates/app/AGENTS.md",
                "/repo/crates/app/CLAUDE.md",
            ]
        );
        assert_eq!(
            MemoryDiscovery::candidates(AgentProfileKind::Gemini, Some("/repo"), "/repo"),
            ["/repo/AGENTS.md", "/repo/GEMINI.md"]
        );
        assert_eq!(
            MemoryDiscovery::candidates(AgentProfileKind::Gemini, None, "/w/"),
            ["/w/AGENTS.md", "/w/GEMINI.md"]
        );
        assert_eq!(
            MemoryDiscovery::candidates(AgentProfileKind::Gpt56, Some("/elsewhere"), "/w"),
            ["/w/AGENTS.md", "/w/.codex/instructions.md"],
            "a working directory outside the root reads itself alone"
        );
        assert_eq!(
            MemoryDiscovery::candidates(AgentProfileKind::Kimi, Some("/repo"), "/repository"),
            ["/repository/AGENTS.md"],
            "a sibling that shares a prefix is not inside the root"
        );
    }

    fn git_env(stdout: &str) -> MockEnvironment {
        MockEnvironment {
            exec_result: ExecResult {
                stdout:      stdout.to_owned(),
                stderr:      String::new(),
                exit_code:   Some(0),
                termination: CommandTermination::Exited,
                duration_ms: 1,
            },
            ..MockEnvironment::linux()
        }
    }

    #[tokio::test]
    async fn the_git_root_discovery_probes_the_environment() {
        let env = git_env("/home\n");
        let paths = MemoryDiscovery::from_git_root()
            .resolve(&env, AgentProfileKind::Kimi, &CancellationToken::new())
            .await
            .expect("resolves");
        assert_eq!(paths, ["/home/AGENTS.md", "/home/test/AGENTS.md"]);

        let outside = git_env("fatal: not a git repository");
        let paths = MemoryDiscovery::from_git_root()
            .resolve(&outside, AgentProfileKind::Kimi, &CancellationToken::new())
            .await
            .expect("resolves");
        assert_eq!(
            paths,
            ["/home/test/AGENTS.md"],
            "an answer that is not a path is no root"
        );
    }

    #[tokio::test]
    async fn skill_searches_resolve_against_their_anchor_and_report_missing_required_dirs() {
        let env = MockEnvironment {
            files: HashMap::from([("/home/test/tools/skills/x".to_owned(), String::new())]),
            ..MockEnvironment::linux()
        };
        let discovery = SkillDiscovery::new()
            .search("/opt/skills")
            .search_under_git_root(".fabro/skills")
            .search_under_git_root("skills")
            .search("local")
            .require("named/skills");
        assert!(discovery.needs_git_root());
        let resolved = discovery
            .resolve(&env, Some("/home"), &CancellationToken::new())
            .await
            .expect("resolves");
        assert_eq!(resolved.dirs, [
            "/opt/skills",
            "/home/.fabro/skills",
            "/home/skills",
            "/home/test/local",
        ]);
        assert_eq!(resolved.skipped.len(), 1);
        assert_eq!(resolved.skipped[0].path, "/home/test/named/skills");
        assert_eq!(
            resolved.skipped[0].reason,
            SkippedSkillReason::MissingDirectory
        );

        let no_repo = discovery
            .resolve(&env, None, &CancellationToken::new())
            .await
            .expect("resolves");
        assert_eq!(
            &no_repo.dirs[1..3],
            ["/home/test/.fabro/skills", "/home/test/skills"],
            "outside a repository the working directory stands in for the root"
        );
    }
}
