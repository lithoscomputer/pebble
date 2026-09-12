//! Pieces an [`Environment`] adapter needs that are pebble's to define.
//!
//! An application that implements [`Environment`] over a container, a VM, or a
//! remote workspace maps its own driver onto pebble's contract, and part of
//! that contract is behavior the model has learned from pebble's own
//! [`LocalEnvironment`]: which glob patterns are refused and in what words, how
//! a directory listing is ordered, how bounded process output is counted, and
//! which [`EnvironmentErrorKind`] a failed command reports. These helpers are
//! that behavior as one implementation, shared between `LocalEnvironment` and
//! the adapters, so an adapter that uses them satisfies the
//! `EnvironmentContract` checks (behind the `test-util` feature) for what they
//! cover without writing the rules a second time.
//!
//! Nothing here is a new trait or a new obligation. An adapter that does the
//! same work another way is still correct; these exist so it does not have to.
//!
//! [`Environment`]: super::Environment
//! [`LocalEnvironment`]: super::LocalEnvironment

use std::cmp::Ordering;

pub use super::capture::OutputCaptureBuffer;
use super::glob::WorkspaceGlob;
use super::{DirEntry, EnvResult, EnvironmentError, EnvironmentErrorKind};
use crate::event::OutputCaptureStats;

/// Checks `pattern` against pebble's glob grammar.
///
/// The grammar is the one [`Environment::glob`](super::Environment::glob)
/// documents: a relative, `/`-separated pattern in which `*` and `?` stay
/// inside one segment, `**` is a whole segment, and `[...]` holds neither `/`
/// nor a wildcard; a trailing `/` is refused because a glob names files. The
/// error is exactly what [`LocalEnvironment`](super::LocalEnvironment) reports
/// for the pattern: [`EnvironmentErrorKind::InvalidInput`], with a message that
/// quotes the pattern as the model sent it and names the mistake, so the model
/// corrects itself the same way wherever pebble runs.
///
/// An adapter whose driver has a glob of its own calls this first, so a
/// pattern pebble refuses is refused before the driver sees it, and a pattern
/// the driver would read differently — a trailing `/` as a directory, say —
/// never reaches it.
///
/// # Errors
///
/// [`EnvironmentErrorKind::InvalidInput`] when the pattern is outside the
/// grammar.
pub fn validate_glob(pattern: &str) -> EnvResult<()> {
    compile_glob(pattern).map(|_glob| ())
}

/// Compiles `pattern`, reporting a refusal as the [`EnvironmentError`] the
/// model reads.
pub(crate) fn compile_glob(pattern: &str) -> EnvResult<WorkspaceGlob> {
    WorkspaceGlob::try_new(pattern).map_err(|error| {
        // The reason goes in the message because that is all the model reads;
        // it has to see what was wrong to try again. The pattern is quoted as
        // the model sent it. Nothing branches on the glob error, so it is not
        // kept as a source, which would repeat the reason.
        EnvironmentError::new(
            EnvironmentErrorKind::InvalidInput,
            format!("Invalid glob pattern {pattern:?}: {error}"),
        )
    })
}

/// Sorts a directory listing into tree order, the order
/// [`Environment::list_directory`](super::Environment::list_directory)
/// promises.
///
/// Tree order is by file name within each directory, with an entry's children
/// directly after it. It is what a recursive walk sorted at every level
/// produces, and it differs from sorting the joined names as strings, where
/// `foo-bar` would fall between `foo` and `foo/x.txt`. Names are compared as
/// [`DirEntry::name`] spells them — `/`-joined and relative to the listed
/// directory — segment by segment, each segment by its bytes.
///
/// ```
/// use pebble_coding_agent::environment::DirEntry;
/// use pebble_coding_agent::environment::support::tree_order;
///
/// let mut entries: Vec<DirEntry> = ["foo.txt", "foo-bar", "foo/x.txt", "foo"]
///     .into_iter()
///     .map(|name| DirEntry {
///         name:   name.to_owned(),
///         is_dir: !name.contains('.'),
///         size:   None,
///     })
///     .collect();
///
/// tree_order(&mut entries);
///
/// let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_str()).collect();
/// assert_eq!(names, ["foo", "foo/x.txt", "foo-bar", "foo.txt"]);
/// ```
///
/// The sort is stable, so two entries with the same name keep the order they
/// arrived in. An adapter whose driver lists in some other order — flat
/// lexicographic, or as the filesystem returned them — calls this on the mapped
/// entries and nothing else changes.
pub fn tree_order(entries: &mut [DirEntry]) {
    entries.sort_by(|left, right| compare_tree_names(&left.name, &right.name));
}

/// Compares two `/`-joined names segment by segment, so a directory sorts
/// before what is inside it and its children before the sibling that follows.
fn compare_tree_names(left: &str, right: &str) -> Ordering {
    left.split('/').cmp(right.split('/'))
}

/// The byte accounting for one stream of process output bounded to
/// `output_bytes_cap`.
///
/// This is the arithmetic behind [`ExecOutcome::stdout_capture`] and
/// [`ExecOutcome::stderr_capture`]: `observed_bytes` is everything the process
/// wrote to the stream, kept or not — an adapter drains past the cap, so the
/// count is what ran through the pipe — and what survives is the whole of it
/// under or at the cap and exactly the cap over it, the rest omitted. `None` is
/// no cap, so nothing is omitted. Pass the cap that bounded the stream: when
/// the request named none and the driver applied a default of its own, that
/// default is the cap.
///
/// [`OutputCaptureBuffer`] reports these same numbers for the bytes it holds.
/// This function is for an adapter whose driver does the bounding and reports
/// only what it saw.
///
/// [`ExecOutcome::stdout_capture`]: super::ExecOutcome::stdout_capture
/// [`ExecOutcome::stderr_capture`]: super::ExecOutcome::stderr_capture
#[must_use]
pub fn capture_stats(observed_bytes: usize, output_bytes_cap: Option<usize>) -> OutputCaptureStats {
    let retained_bytes = output_bytes_cap.map_or(observed_bytes, |cap| observed_bytes.min(cap));
    OutputCaptureStats {
        observed_bytes,
        retained_bytes,
        omitted_bytes: observed_bytes.saturating_sub(retained_bytes),
    }
}

/// How a command failed, when it failed rather than ran.
///
/// [`Environment::exec`](super::Environment::exec) returns an error only when
/// the command never ran or what it produced could not be collected; a command
/// that ran and exited badly, timed out, or was cancelled is a result. An
/// adapter names which of these it hit and [`classify_exec_error`] says what
/// kind that is, so every environment reports the same kind for the same
/// failure. New failures may appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExecFailure {
    /// The process could not be started: no usable interpreter, an unusable
    /// working directory, a launcher that refused.
    Start,
    /// The process started, but its exit status or its output could not be
    /// collected: a pipe, a drain task, or the transport that carried them
    /// failed.
    Collect,
    /// The environment does not run commands, or not the way the request asks.
    Unsupported,
}

/// The [`EnvironmentErrorKind`] a failed command reports, as
/// [`LocalEnvironment`](super::LocalEnvironment) classifies its own failures.
///
/// A command that never started is [`Spawn`](EnvironmentErrorKind::Spawn); one
/// that started but whose status or output was lost is
/// [`Io`](EnvironmentErrorKind::Io); an operation the environment does not
/// offer is [`Unsupported`](EnvironmentErrorKind::Unsupported). The line
/// between the first two is whether the process ran: a tool reading the error
/// can tell a machine that cannot run commands from one that ran a command and
/// lost its output.
#[must_use]
pub const fn classify_exec_error(failure: ExecFailure) -> EnvironmentErrorKind {
    match failure {
        ExecFailure::Start => EnvironmentErrorKind::Spawn,
        ExecFailure::Collect => EnvironmentErrorKind::Io,
        ExecFailure::Unsupported => EnvironmentErrorKind::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str) -> DirEntry {
        let is_dir = !name.rsplit('/').next().unwrap_or(name).contains('.');
        DirEntry {
            name: name.to_owned(),
            is_dir,
            size: (!is_dir).then_some(1),
        }
    }

    fn names(entries: &[DirEntry]) -> Vec<&str> {
        entries.iter().map(|entry| entry.name.as_str()).collect()
    }

    #[test]
    fn a_pattern_in_the_grammar_is_accepted() {
        for pattern in [
            "**/*.txt",
            "?.txt",
            "[ab].txt",
            "./src/**",
            "src/[!m]ib.rs",
            "src/[a-z]ib.rs",
            "docs/**/*.md",
            "src//*.rs",
        ] {
            assert!(validate_glob(pattern).is_ok(), "pattern {pattern:?}");
        }
    }

    /// Fabro's hand-written check refused a `?` inside a class; pebble's
    /// grammar reads it as a literal member, and pebble's grammar is the one
    /// the model is taught.
    #[test]
    fn a_question_mark_is_a_valid_class_member() {
        assert!(validate_glob("src/[?]ib.rs").is_ok());
    }

    #[test]
    fn a_pattern_outside_the_grammar_is_refused_with_the_mistake_named() {
        let cases = [
            ("", "pattern cannot be empty"),
            ("./", "pattern cannot be empty"),
            ("/tmp/*.md", "pattern must be relative"),
            ("C:/tmp/*.md", "pattern must be relative"),
            (r"dir\*.rs", "pattern must use \"/\" as its path separator"),
            ("../*.md", "pattern cannot traverse to a parent directory"),
            (
                "nested/",
                "pattern ends with \"/\"; glob matches files, drop the trailing slash or add a \
                 filename pattern",
            ),
            ("src/[abc", "pattern has an unclosed character class"),
            ("[a/]", "a \"/\" cannot appear inside a character class"),
            ("[a*]", "wildcards are not valid inside a character class"),
            ("src**/*.rs", "\"**\" must be a whole path segment"),
        ];

        for (pattern, reason) in cases {
            let error = validate_glob(pattern).expect_err("the pattern is refused");

            assert_eq!(
                error.kind(),
                EnvironmentErrorKind::InvalidInput,
                "pattern {pattern:?}"
            );
            assert_eq!(
                error.message(),
                format!("Invalid glob pattern {pattern:?}: {reason}"),
                "pattern {pattern:?}"
            );
            // The reason is in the message and nowhere else, so the rendering
            // the model reads does not say it twice.
            assert_eq!(error.detail(), error.message(), "pattern {pattern:?}");
        }
    }

    /// The patterns the `EnvironmentContract` suite sends every environment.
    #[test]
    fn the_contract_suites_invalid_patterns_are_refused() {
        for pattern in ["/absolute", "../escape", "a\\b", "nested/", "[a/]"] {
            let error = validate_glob(pattern).expect_err("the pattern is refused");

            assert_eq!(
                error.kind(),
                EnvironmentErrorKind::InvalidInput,
                "pattern {pattern:?}"
            );
        }
    }

    #[test]
    fn children_follow_their_parent_before_the_next_sibling() {
        // Flat lexicographic order, which is what a driver that sorts the
        // joined paths produces.
        let mut entries: Vec<DirEntry> =
            ["foo", "foo-bar", "foo-bar/y.txt", "foo.txt", "foo/x.txt"]
                .into_iter()
                .map(entry)
                .collect();

        tree_order(&mut entries);

        assert_eq!(names(&entries), [
            "foo",
            "foo/x.txt",
            "foo-bar",
            "foo-bar/y.txt",
            "foo.txt"
        ]);
    }

    #[test]
    fn nested_directories_are_listed_depth_first() {
        let mut entries: Vec<DirEntry> = [
            "g.txt",
            "a/f.txt",
            "a/b/e.txt",
            "a/b/c/d.txt",
            "a/b/c",
            "a/b",
            "a",
            "a/b/c/.hidden",
        ]
        .into_iter()
        .map(entry)
        .collect();

        tree_order(&mut entries);

        assert_eq!(names(&entries), [
            "a",
            "a/b",
            "a/b/c",
            "a/b/c/.hidden",
            "a/b/c/d.txt",
            "a/b/e.txt",
            "a/f.txt",
            "g.txt",
        ]);
    }

    #[test]
    fn names_within_one_directory_sort_by_their_bytes() {
        let mut entries: Vec<DirEntry> = ["b.txt", "B.txt", "a.txt", "_.txt", "日本.txt"]
            .into_iter()
            .map(entry)
            .collect();

        tree_order(&mut entries);

        assert_eq!(names(&entries), [
            "B.txt",
            "_.txt",
            "a.txt",
            "b.txt",
            "日本.txt"
        ]);
    }

    #[test]
    fn entries_with_the_same_name_keep_their_arrival_order() {
        let mut entries = vec![
            entry("same/second.txt"),
            DirEntry {
                name:   "same".to_owned(),
                is_dir: false,
                size:   Some(3),
            },
            DirEntry {
                name:   "same".to_owned(),
                is_dir: true,
                size:   None,
            },
            entry("same/first.txt"),
        ];

        tree_order(&mut entries);

        assert_eq!(names(&entries), [
            "same",
            "same",
            "same/first.txt",
            "same/second.txt"
        ]);
        assert!(!entries[0].is_dir, "the file arrived first");
        assert!(entries[1].is_dir);
    }

    #[test]
    fn an_empty_listing_stays_empty() {
        let mut entries: Vec<DirEntry> = Vec::new();

        tree_order(&mut entries);

        assert!(entries.is_empty());
    }

    #[test]
    fn output_under_the_cap_is_complete() {
        assert_eq!(capture_stats(5, Some(8)), OutputCaptureStats::complete(5));
    }

    #[test]
    fn output_at_the_cap_is_complete() {
        assert_eq!(capture_stats(8, Some(8)), OutputCaptureStats::complete(8));
    }

    #[test]
    fn output_over_the_cap_retains_the_cap_and_omits_the_rest() {
        assert_eq!(capture_stats(20, Some(8)), OutputCaptureStats {
            observed_bytes: 20,
            retained_bytes: 8,
            omitted_bytes:  12,
        });
    }

    #[test]
    fn output_without_a_cap_omits_nothing() {
        assert_eq!(capture_stats(20, None), OutputCaptureStats::complete(20));
        assert_eq!(capture_stats(0, None), OutputCaptureStats::complete(0));
    }

    /// A caller that wants no output still needs the byte counts: the pipe
    /// was drained, and how much ran through it is what a consumer is told.
    #[test]
    fn a_zero_byte_cap_retains_nothing_and_still_counts() {
        assert_eq!(capture_stats(6, Some(0)), OutputCaptureStats {
            observed_bytes: 6,
            retained_bytes: 0,
            omitted_bytes:  6,
        });
    }

    /// The arithmetic and the buffer describe the same capture, so an adapter
    /// that drains through the buffer and one whose driver bounds for it
    /// report the same numbers for the same output.
    #[test]
    fn the_buffer_reports_what_the_arithmetic_says() {
        for (observed, cap) in [
            (5, Some(8)),
            (8, Some(8)),
            (9, Some(8)),
            (20, Some(8)),
            (20, Some(7)),
            (6, Some(0)),
            (20, None),
            (0, Some(8)),
        ] {
            let mut buffer = OutputCaptureBuffer::new(cap);
            let output = vec![b'x'; observed];
            for chunk in output.chunks(3) {
                buffer.push(chunk);
            }

            assert_eq!(
                buffer.stats(),
                capture_stats(observed, cap),
                "observed {observed}, cap {cap:?}"
            );
        }
    }

    #[test]
    fn every_failure_has_the_kind_local_environment_reports() {
        let cases = [
            (ExecFailure::Start, EnvironmentErrorKind::Spawn),
            (ExecFailure::Collect, EnvironmentErrorKind::Io),
            (ExecFailure::Unsupported, EnvironmentErrorKind::Unsupported),
        ];

        for (failure, kind) in cases {
            assert_eq!(classify_exec_error(failure), kind, "{failure:?}");
        }
    }
}
