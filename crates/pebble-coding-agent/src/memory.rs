//! Project instructions loaded into the system prompt.
//!
//! Memory is the standing instruction a repository keeps for whoever works in
//! it — `AGENTS.md` and its relatives. Pebble reads the files an application
//! names and nothing else: no conventional filename, no walk up to a
//! repository root, no home directory. An application that wants fabro's
//! conventions builds the path list itself and hands it over.
//!
//! What this module owns is the budget. Memory competes with the conversation
//! for the context window, so the loader spends at most
//! [`MEMORY_BUDGET_BYTES`] across every file, skips a file whose text it has
//! already loaded, and cuts the file that crosses the line rather than
//! dropping it.

use std::collections::HashSet;

use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::char_boundary::floor_char_boundary;
use crate::environment::Environment;
use crate::error::{Error, InterruptReason, Result};
use crate::types::MemoryFileSummary;

/// The total bytes of memory one session loads.
pub(crate) const MEMORY_BUDGET_BYTES: usize = 32_768;

/// What replaces the text the budget could not fit.
const TRUNCATION_MARKER: &str = "[Project instructions truncated at 32KB]";

/// One loaded memory file.
///
/// [`content`](Self::content) is what goes into the system prompt. The rest
/// describes the file for the
/// [`MemoryLoaded`](crate::events::CodingEvent::MemoryLoaded) event, which
/// deliberately carries the description and never the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryDocument {
    /// The path the file was read from.
    pub(crate) path:         String,
    /// The text loaded into the system prompt, already cut to the budget.
    pub(crate) content:      String,
    /// The file's full size, in bytes.
    pub(crate) byte_count:   usize,
    /// How many bytes of it were loaded.
    pub(crate) loaded_bytes: usize,
    /// Whether the budget cut the file short.
    pub(crate) truncated:    bool,
}

impl MemoryDocument {
    /// The description of this file that the event stream carries.
    #[must_use]
    pub(crate) fn to_summary(&self) -> MemoryFileSummary {
        MemoryFileSummary {
            path:         self.path.clone(),
            byte_count:   self.byte_count,
            loaded_bytes: self.loaded_bytes,
            truncated:    self.truncated,
        }
    }
}

/// Loads the configured memory files, in order, within the budget.
///
/// `paths` are resolved through `env`, so a session working in a container
/// reads the container's files. A path that cannot be read, and a file that is
/// empty, is skipped rather than failing the load: an application normally
/// names more candidate paths than any one repository has.
///
/// A file whose text has already been loaded is skipped, which is what makes a
/// symlinked `CLAUDE.md` beside an `AGENTS.md` cost one budget's worth instead
/// of two. Files are charged against the budget in the order given; the first
/// one that does not fit is cut to what remains and the rest are skipped.
///
/// Returns [`Error::Interrupted`] when `cancel` fires, which is checked around
/// every read.
pub(crate) async fn load_memory(
    env: &dyn Environment,
    paths: &[String],
    cancel: &CancellationToken,
) -> Result<Vec<MemoryDocument>> {
    let mut documents: Vec<MemoryDocument> = Vec::new();
    let mut budget_remaining = MEMORY_BUDGET_BYTES;
    let mut seen_content: HashSet<String> = HashSet::new();

    for path in paths {
        let Some(content) = read_memory_file(env, path, cancel).await? else {
            continue;
        };

        if !seen_content.insert(content.clone()) {
            debug!(path, "Memory file duplicates one already loaded, skipping");
            continue;
        }

        let byte_count = content.len();
        if byte_count <= budget_remaining {
            debug!(path, size_bytes = byte_count, "Memory file loaded");
            budget_remaining -= byte_count;
            documents.push(MemoryDocument {
                path: path.clone(),
                content,
                byte_count,
                loaded_bytes: byte_count,
                truncated: false,
            });
        } else if budget_remaining > 0 {
            warn!(
                path,
                size_bytes = byte_count,
                budget_remaining,
                "Memory file truncated to fit the budget"
            );
            let content = truncate_to_budget(&content, budget_remaining);
            let loaded_bytes = content.len();
            budget_remaining = 0;
            documents.push(MemoryDocument {
                path: path.clone(),
                content,
                byte_count,
                loaded_bytes,
                truncated: true,
            });
        } else {
            warn!(
                path,
                size_bytes = byte_count,
                "Memory file skipped, budget exhausted"
            );
        }
    }

    Ok(documents)
}

/// Reads one memory file, answering `None` for anything the loader skips.
///
/// The cancellation token is checked on both sides of the read, so a cancel
/// that arrives while a slow environment is reading is noticed before the next
/// file is opened.
async fn read_memory_file(
    env: &dyn Environment,
    path: &str,
    cancel: &CancellationToken,
) -> Result<Option<String>> {
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    let read = env.read_file_text(path).await;

    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    let Ok(content) = read else {
        debug!(path, "Memory file could not be read, skipping");
        return Ok(None);
    };

    if content.is_empty() {
        warn!(path, "Memory file is empty, skipping");
        return Ok(None);
    }

    Ok(Some(content))
}

/// Cuts `content` to `budget` bytes and marks where it was cut.
///
/// The marker itself is charged against the budget, so the result never
/// exceeds it. A budget too small for the marker returns as much of the marker
/// as fits, which tells a reader that something was dropped even when nothing
/// of the file survived.
fn truncate_to_budget(content: &str, budget: usize) -> String {
    if budget <= TRUNCATION_MARKER.len() {
        // The marker is ASCII, so any byte index inside it is a boundary.
        return TRUNCATION_MARKER[..budget].to_owned();
    }

    let usable = budget - TRUNCATION_MARKER.len();
    let end = floor_char_boundary(content, usable);
    format!("{}{TRUNCATION_MARKER}", &content[..end])
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::test_support::MockEnvironment;

    fn environment(files: &[(&str, &str)]) -> MockEnvironment {
        MockEnvironment {
            files: files
                .iter()
                .map(|(path, content)| ((*path).to_owned(), (*content).to_owned()))
                .collect::<HashMap<_, _>>(),
            ..MockEnvironment::default()
        }
    }

    fn paths(paths: &[&str]) -> Vec<String> {
        paths.iter().map(|path| (*path).to_owned()).collect()
    }

    #[tokio::test]
    async fn a_configured_file_is_loaded_with_its_measurements() {
        let env = environment(&[("/repo/AGENTS.md", "Agent instructions")]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/AGENTS.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].path, "/repo/AGENTS.md");
        assert_eq!(documents[0].content, "Agent instructions");
        assert_eq!(documents[0].byte_count, "Agent instructions".len());
        assert_eq!(documents[0].loaded_bytes, documents[0].byte_count);
        assert!(!documents[0].truncated);
    }

    #[tokio::test]
    async fn exactly_the_configured_paths_are_read_in_order() {
        let env = environment(&[
            ("/repo/AGENTS.md", "agents"),
            ("/repo/CLAUDE.md", "claude"),
            ("/repo/GEMINI.md", "gemini"),
        ]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/GEMINI.md", "/repo/AGENTS.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(
            documents
                .iter()
                .map(|document| document.content.as_str())
                .collect::<Vec<_>>(),
            ["gemini", "agents"]
        );
    }

    #[tokio::test]
    async fn no_configured_paths_load_nothing() {
        let env = environment(&[("/repo/AGENTS.md", "agents")]);

        let documents = load_memory(&env, &[], &CancellationToken::new())
            .await
            .expect("the load succeeds");

        assert!(documents.is_empty());
    }

    #[tokio::test]
    async fn a_missing_or_empty_file_is_skipped() {
        let env = environment(&[("/repo/EMPTY.md", ""), ("/repo/AGENTS.md", "agents")]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/MISSING.md", "/repo/EMPTY.md", "/repo/AGENTS.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].path, "/repo/AGENTS.md");
    }

    #[tokio::test]
    async fn the_file_that_crosses_the_budget_is_cut_rather_than_dropped() {
        let first = "x".repeat(30_000);
        let env = environment(&[
            ("/repo/AGENTS.md", first.as_str()),
            ("/repo/CLAUDE.md", &"y".repeat(5_000)),
        ]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/AGENTS.md", "/repo/CLAUDE.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].content, first);
        assert!(!documents[0].truncated);
        assert!(documents[1].truncated);
        assert!(documents[1].content.ends_with(TRUNCATION_MARKER));
        assert!(documents[1].byte_count > documents[1].content.len());
        assert!(documents[0].content.len() + documents[1].content.len() <= MEMORY_BUDGET_BYTES);
    }

    #[tokio::test]
    async fn a_file_past_an_exhausted_budget_is_skipped() {
        let env = environment(&[
            ("/repo/A.md", &"x".repeat(MEMORY_BUDGET_BYTES)),
            ("/repo/B.md", "still worth reading"),
        ]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/A.md", "/repo/B.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].path, "/repo/A.md");
    }

    #[tokio::test]
    async fn a_file_repeating_text_already_loaded_is_skipped() {
        let env = environment(&[
            ("/repo/AGENTS.md", "shared instructions"),
            ("/repo/CLAUDE.md", "shared instructions"),
        ]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/AGENTS.md", "/repo/CLAUDE.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].content, "shared instructions");
    }

    #[tokio::test]
    async fn a_single_oversized_file_reports_both_sizes() {
        let whole = "x".repeat(MEMORY_BUDGET_BYTES + 1_024);
        let env = environment(&[("/repo/AGENTS.md", whole.as_str())]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/AGENTS.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(documents.len(), 1);
        assert!(documents[0].truncated);
        assert_eq!(documents[0].byte_count, whole.len());
        assert_eq!(documents[0].loaded_bytes, documents[0].content.len());
        assert!(documents[0].content.len() < documents[0].byte_count);
        assert!(documents[0].content.len() <= MEMORY_BUDGET_BYTES);
    }

    #[tokio::test]
    async fn a_cancelled_load_is_interrupted() {
        let env = environment(&[("/repo/AGENTS.md", "agents")]);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let error = load_memory(&env, &paths(&["/repo/AGENTS.md"]), &cancel)
            .await
            .expect_err("a cancelled load fails");

        assert!(matches!(
            error,
            Error::Interrupted(InterruptReason::Cancelled)
        ));
    }

    #[tokio::test]
    async fn a_loaded_document_describes_itself_for_the_event_stream() {
        let env = environment(&[("/repo/AGENTS.md", "agents")]);

        let documents = load_memory(
            &env,
            &paths(&["/repo/AGENTS.md"]),
            &CancellationToken::new(),
        )
        .await
        .expect("the load succeeds");

        assert_eq!(documents[0].to_summary(), MemoryFileSummary {
            path:         "/repo/AGENTS.md".to_owned(),
            byte_count:   6,
            loaded_bytes: 6,
            truncated:    false,
        });
    }

    #[test]
    fn truncation_keeps_whole_characters_and_fits_the_budget() {
        let content = "€".repeat(100);

        let truncated = truncate_to_budget(&content, TRUNCATION_MARKER.len() + 10);

        assert!(truncated.ends_with(TRUNCATION_MARKER));
        assert!(truncated.len() <= TRUNCATION_MARKER.len() + 10);
        // Nine of the ten usable bytes hold three whole three-byte characters.
        assert_eq!(truncated.len(), TRUNCATION_MARKER.len() + 9);
    }

    #[test]
    fn a_budget_smaller_than_the_marker_keeps_what_fits_of_it() {
        assert_eq!(truncate_to_budget("anything", 7), "[Projec");
        assert_eq!(truncate_to_budget("anything", 0), "");
    }
}
