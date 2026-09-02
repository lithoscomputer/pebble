//! Workspace-relative glob patterns.
//!
//! Every [`Environment::glob`](super::Environment::glob) implementation shares
//! these semantics so a pattern means the same thing wherever the tools run:
//! patterns are relative to a base directory, use `/` as their separator, and
//! never escape the base. `*` and `?` stay inside one path segment, `**`
//! crosses segments, and `[abc]` matches a character class. A `**` at the end
//! of a pattern matches at least one segment, so `docs/**` names what is inside
//! `docs` rather than `docs` itself.
//!
//! A glob names files, so a pattern that ends with `/` is rejected rather than
//! read as if the slash were not there, and a `/` or a wildcard inside `[...]`
//! is rejected too. Each rejection names the mistake: the patterns come from a
//! model, and a precise error is what lets it correct itself. Fabro's glob
//! crate accepted all of these; this is a deliberate divergence.
//!
//! The matcher is written here rather than taken from a glob crate so the
//! semantics stay pinned to the tests below.

use std::borrow::Cow;
use std::iter::Peekable;
use std::path::Path;
use std::str::Chars;

/// A validated glob matched against `/`-separated paths relative to a base
/// directory.
#[derive(Clone, Debug)]
pub(crate) struct WorkspaceGlob {
    segments:       Vec<Segment>,
    traversal_root: String,
}

impl WorkspaceGlob {
    /// Compiles `source`, rejecting patterns that reach outside the base
    /// directory or that the matcher cannot read.
    pub(crate) fn try_new(source: &str) -> Result<Self, WorkspaceGlobError> {
        let source = strip_current_dir_prefix(source);
        if source.is_empty() {
            return Err(WorkspaceGlobError::Empty);
        }
        if source.contains('\\') {
            return Err(WorkspaceGlobError::BackslashSeparator {
                pattern: source.to_owned(),
            });
        }
        if is_absolute(source) {
            return Err(WorkspaceGlobError::Absolute {
                pattern: source.to_owned(),
            });
        }
        if source.split('/').any(|segment| segment == "..") {
            return Err(WorkspaceGlobError::ParentTraversal {
                pattern: source.to_owned(),
            });
        }
        if source.ends_with('/') {
            return Err(WorkspaceGlobError::TrailingSeparator {
                pattern: source.to_owned(),
            });
        }

        Ok(Self {
            segments:       parse_segments(source)?,
            traversal_root: literal_traversal_root(source),
        })
    }

    /// Whether `relative_path` matches.
    pub(crate) fn is_match(&self, relative_path: &str) -> bool {
        let normalized = normalize_candidate(relative_path);
        let candidate = strip_current_dir_prefix(normalized.as_ref());
        if is_absolute(candidate) || candidate.split('/').any(|segment| segment == "..") {
            return false;
        }

        let path: Vec<&str> = candidate
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect();
        match_segments(&self.segments, &path)
    }

    /// A literal directory prefix a caller may start traversal from.
    ///
    /// This is only an optimization: every candidate it yields still has to
    /// pass [`is_match`](Self::is_match).
    pub(crate) fn traversal_root(&self) -> &str {
        &self.traversal_root
    }
}

/// Why a glob pattern was rejected.
///
/// The message is the reason alone, written so a model can act on it. The
/// caller that reports the error names the operation and the pattern.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WorkspaceGlobError {
    #[error("pattern cannot be empty")]
    Empty,

    #[error("pattern must use \"/\" as its path separator")]
    BackslashSeparator { pattern: String },

    #[error("pattern must be relative")]
    Absolute { pattern: String },

    #[error("pattern cannot traverse to a parent directory")]
    ParentTraversal { pattern: String },

    #[error(
        "pattern ends with \"/\"; glob matches files, drop the trailing slash or add a filename \
         pattern"
    )]
    TrailingSeparator { pattern: String },

    #[error("pattern has an unclosed character class")]
    UnclosedCharacterClass { pattern: String },

    #[error("a \"/\" cannot appear inside a character class")]
    SeparatorInCharacterClass { pattern: String },

    #[error("wildcards are not valid inside a character class")]
    WildcardInCharacterClass { pattern: String },

    #[error("\"**\" must be a whole path segment")]
    RecursiveWildcardInSegment { pattern: String },
}

impl WorkspaceGlobError {
    /// The pattern that was rejected, as it was read.
    pub(crate) fn pattern(&self) -> &str {
        match self {
            Self::Empty => "",
            Self::BackslashSeparator { pattern }
            | Self::Absolute { pattern }
            | Self::ParentTraversal { pattern }
            | Self::TrailingSeparator { pattern }
            | Self::UnclosedCharacterClass { pattern }
            | Self::SeparatorInCharacterClass { pattern }
            | Self::WildcardInCharacterClass { pattern }
            | Self::RecursiveWildcardInSegment { pattern } => pattern,
        }
    }
}

#[derive(Clone, Debug)]
enum Segment {
    /// `**`: zero or more path segments.
    Recursive,
    /// One path segment, matched token by token.
    Tokens(Vec<Token>),
}

#[derive(Clone, Debug)]
enum Token {
    Literal(char),
    AnyCharacter,
    AnyRun,
    Class {
        negated: bool,
        members: Vec<ClassMember>,
    },
}

#[derive(Clone, Debug)]
enum ClassMember {
    Character(char),
    Range(char, char),
}

impl ClassMember {
    fn contains(&self, candidate: char) -> bool {
        match self {
            Self::Character(character) => *character == candidate,
            Self::Range(start, end) => (*start..=*end).contains(&candidate),
        }
    }
}

/// Reads `pattern` into one segment per `/`-separated part.
///
/// Character classes are read before the pattern is split, so a `/` inside
/// `[...]` is reported as that rather than as an unclosed class. A repeated
/// separator (`a//b`) reads as one.
fn parse_segments(pattern: &str) -> Result<Vec<Segment>, WorkspaceGlobError> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut characters = pattern.chars().peekable();
    while characters.peek().is_some() {
        let tokens = parse_segment_tokens(&mut characters, pattern)?;
        if tokens.is_empty() {
            continue;
        }
        let parsed = segment_from_tokens(tokens, pattern)?;
        // `**/**` matches exactly what `**` does, and collapsing the pair
        // keeps matching from branching once per repetition.
        let repeats_recursion = matches!(parsed, Segment::Recursive)
            && matches!(segments.last(), Some(Segment::Recursive));
        if !repeats_recursion {
            segments.push(parsed);
        }
    }
    Ok(segments)
}

/// Reads tokens up to the next `/` outside a character class, consuming that
/// `/`, or to the end of the pattern.
fn parse_segment_tokens(
    characters: &mut Peekable<Chars<'_>>,
    pattern: &str,
) -> Result<Vec<Token>, WorkspaceGlobError> {
    let mut tokens = Vec::new();
    while let Some(character) = characters.next() {
        match character {
            '/' => break,
            '*' => tokens.push(Token::AnyRun),
            '?' => tokens.push(Token::AnyCharacter),
            '[' => tokens.push(parse_class(characters, pattern)?),
            literal => tokens.push(Token::Literal(literal)),
        }
    }
    Ok(tokens)
}

/// Reads the body of a character class; the opening `[` is already consumed.
fn parse_class(
    characters: &mut Peekable<Chars<'_>>,
    pattern: &str,
) -> Result<Token, WorkspaceGlobError> {
    let negated = characters.peek() == Some(&'!');
    if negated {
        characters.next();
    }
    let mut members = Vec::new();
    let mut closed = false;
    // A `]` immediately after the opening bracket is a member.
    while let Some(member) = characters.next() {
        if member == ']' && !members.is_empty() {
            closed = true;
            break;
        }
        let member = class_member(member, pattern)?;
        if characters.peek() == Some(&'-') {
            let mut lookahead = characters.clone();
            lookahead.next();
            match lookahead.peek().copied() {
                Some(end) if end != ']' => {
                    characters.next();
                    characters.next();
                    members.push(ClassMember::Range(member, class_member(end, pattern)?));
                    continue;
                }
                _ => {}
            }
        }
        members.push(ClassMember::Character(member));
    }
    if !closed {
        return Err(WorkspaceGlobError::UnclosedCharacterClass {
            pattern: pattern.to_owned(),
        });
    }
    Ok(Token::Class { negated, members })
}

/// A character that may stand in a class: not the separator and not `*`,
/// which are the mistakes a class is most often written with.
fn class_member(character: char, pattern: &str) -> Result<char, WorkspaceGlobError> {
    match character {
        '/' => Err(WorkspaceGlobError::SeparatorInCharacterClass {
            pattern: pattern.to_owned(),
        }),
        '*' => Err(WorkspaceGlobError::WildcardInCharacterClass {
            pattern: pattern.to_owned(),
        }),
        member => Ok(member),
    }
}

/// `**` on its own is the recursive segment; `**` next to anything else has
/// no meaning and is rejected.
fn segment_from_tokens(tokens: Vec<Token>, pattern: &str) -> Result<Segment, WorkspaceGlobError> {
    let has_recursive_run = tokens
        .windows(2)
        .any(|pair| matches!(pair, [Token::AnyRun, Token::AnyRun]));
    if !has_recursive_run {
        return Ok(Segment::Tokens(tokens));
    }
    if tokens.len() == 2 {
        return Ok(Segment::Recursive);
    }
    Err(WorkspaceGlobError::RecursiveWildcardInSegment {
        pattern: pattern.to_owned(),
    })
}

fn match_segments(pattern: &[Segment], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        // A trailing `**` names what is inside a directory, so it matches one
        // or more segments rather than zero: `docs/**` matches `docs/README.md`
        // and not a file named `docs`. In the middle of a pattern it still
        // matches zero segments, so `**/*.md` matches `README.md`.
        Some((Segment::Recursive, [])) => !path.is_empty(),
        Some((Segment::Recursive, rest)) => {
            (0..=path.len()).any(|skipped| match_segments(rest, &path[skipped..]))
        }
        Some((Segment::Tokens(tokens), rest)) => match path.split_first() {
            Some((head, tail)) => match_tokens(tokens, head) && match_segments(rest, tail),
            None => false,
        },
    }
}

fn match_tokens(tokens: &[Token], text: &str) -> bool {
    let Some((token, rest)) = tokens.split_first() else {
        return text.is_empty();
    };

    match token {
        Token::AnyRun => {
            // Try every split point, shortest run first.
            if match_tokens(rest, text) {
                return true;
            }
            let mut remaining = text;
            while let Some(next) = remaining.chars().next() {
                remaining = &remaining[next.len_utf8()..];
                if match_tokens(rest, remaining) {
                    return true;
                }
            }
            false
        }
        Token::AnyCharacter => match text.chars().next() {
            Some(character) => match_tokens(rest, &text[character.len_utf8()..]),
            None => false,
        },
        Token::Literal(expected) => match text.chars().next() {
            Some(character) if character == *expected => {
                match_tokens(rest, &text[character.len_utf8()..])
            }
            _ => false,
        },
        Token::Class { negated, members } => match text.chars().next() {
            Some(character) => {
                let matched = members.iter().any(|member| member.contains(character));
                matched != *negated && match_tokens(rest, &text[character.len_utf8()..])
            }
            None => false,
        },
    }
}

/// Windows paths arrive with backslashes; everywhere else the candidate is
/// already `/`-separated.
fn normalize_candidate(path: &str) -> Cow<'_, str> {
    #[cfg(windows)]
    {
        Cow::Owned(path.replace('\\', "/"))
    }
    #[cfg(not(windows))]
    {
        Cow::Borrowed(path)
    }
}

fn strip_current_dir_prefix(mut path: &str) -> &str {
    while let Some(stripped) = path.strip_prefix("./") {
        path = stripped;
    }
    path
}

fn is_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();
    path.starts_with('/')
        || Path::new(path).is_absolute()
        || matches!(bytes, [drive, b':', ..] if drive.is_ascii_alphabetic())
}

fn literal_traversal_root(pattern: &str) -> String {
    let mut literal_segments = Vec::new();
    let mut saw_meta = false;

    for segment in pattern.split('/').filter(|segment| !segment.is_empty()) {
        if has_glob_meta(segment) {
            saw_meta = true;
            break;
        }
        literal_segments.push(segment);
    }

    if !saw_meta {
        // A pattern of literal segments names one file; traversal starts at
        // its parent.
        literal_segments.pop();
    }
    literal_segments.join("/")
}

fn has_glob_meta(segment: &str) -> bool {
    segment
        .chars()
        .any(|character| matches!(character, '*' | '?' | '['))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_have_root_relative_segment_semantics() {
        let cases = [
            ("*.md", "README.md", true),
            ("*.md", "docs/README.md", false),
            ("**/*.md", "README.md", true),
            ("**/*.md", "docs/README.md", true),
            (".ai/reports/*.md", ".ai/reports/result.md", true),
            (".ai/reports/*.md", ".ai/reports/nested/result.md", false),
            (".ai/reports/**/*.md", ".ai/reports/nested/result.md", true),
            (
                ".ai/plans/????-??-??-*.md",
                ".ai/plans/2026-07-25-globbing.md",
                true,
            ),
            (".ai/plans/????-??-??-*.md", ".ai/plans/DRAFTING.md", false),
            ("*/SKILL.md", "rust/SKILL.md", true),
            ("*/SKILL.md", "rust/review/SKILL.md", false),
            ("src/[lm]ib.rs", "src/lib.rs", true),
            ("src/[!m]ib.rs", "src/lib.rs", true),
            ("src/[!l]ib.rs", "src/lib.rs", false),
            ("src/[a-z]ib.rs", "src/lib.rs", true),
            ("**/.env", ".env", true),
            ("**/.env", "nested/.env", true),
            ("**", "a/b/c.txt", true),
            ("**", "c.txt", true),
            ("docs/**", "docs/a/b.md", true),
            ("docs/**", "docs/README.md", true),
            // A trailing `**` matches what is inside `docs`, never `docs`
            // itself, which is what the glob crate fabro used answers.
            ("docs/**", "docs", false),
            ("docs/**", "src/a/b.md", false),
            ("*.md", "notes.txt", false),
            ("a*c.rs", "abbbc.rs", true),
            ("a*c.rs", "abbb.rs", false),
        ];

        for (pattern, candidate, expected) in cases {
            let glob = WorkspaceGlob::try_new(pattern).expect("pattern compiles");
            assert_eq!(
                glob.is_match(candidate),
                expected,
                "pattern {pattern:?}, candidate {candidate:?}"
            );
        }
    }

    #[test]
    fn a_leading_current_directory_is_normalized() {
        let glob = WorkspaceGlob::try_new("./src/*.rs").expect("pattern compiles");

        assert!(glob.is_match("./src/lib.rs"));
        assert!(glob.is_match("src/lib.rs"));
        assert_eq!(glob.traversal_root(), "src");
    }

    #[test]
    fn candidates_outside_the_base_never_match() {
        let glob = WorkspaceGlob::try_new("**/*.md").expect("pattern compiles");

        assert!(!glob.is_match("/etc/passwd.md"));
        assert!(!glob.is_match("../outside.md"));
    }

    #[test]
    fn a_wildcard_matches_multibyte_names() {
        let glob = WorkspaceGlob::try_new("*.md").expect("pattern compiles");

        assert!(glob.is_match("日本語.md"));
        assert!(
            WorkspaceGlob::try_new("?.md")
                .expect("pattern compiles")
                .is_match("é.md")
        );
    }

    #[test]
    fn patterns_outside_the_root_are_rejected() {
        assert!(matches!(
            WorkspaceGlob::try_new(""),
            Err(WorkspaceGlobError::Empty)
        ));
        assert!(matches!(
            WorkspaceGlob::try_new("/tmp/*.md"),
            Err(WorkspaceGlobError::Absolute { .. })
        ));
        assert!(matches!(
            WorkspaceGlob::try_new("C:/tmp/*.md"),
            Err(WorkspaceGlobError::Absolute { .. })
        ));
        assert!(matches!(
            WorkspaceGlob::try_new(r"dir\*.rs"),
            Err(WorkspaceGlobError::BackslashSeparator { .. })
        ));
        assert!(matches!(
            WorkspaceGlob::try_new(r"\\server\share\*.md"),
            Err(WorkspaceGlobError::BackslashSeparator { .. })
        ));
        assert!(matches!(
            WorkspaceGlob::try_new("../*.md"),
            Err(WorkspaceGlobError::ParentTraversal { .. })
        ));
    }

    #[test]
    fn unreadable_patterns_are_rejected() {
        assert!(matches!(
            WorkspaceGlob::try_new("src/[abc"),
            Err(WorkspaceGlobError::UnclosedCharacterClass { .. })
        ));
        assert!(matches!(
            WorkspaceGlob::try_new("src**/*.rs"),
            Err(WorkspaceGlobError::RecursiveWildcardInSegment { .. })
        ));
    }

    #[test]
    fn malformed_patterns_are_rejected_with_the_mistake_named() {
        let cases = [
            (
                "src/[a/b].rs",
                "a \"/\" cannot appear inside a character class",
            ),
            (
                "**/[!/]*.md",
                "a \"/\" cannot appear inside a character class",
            ),
            (
                "src/*/",
                "pattern ends with \"/\"; glob matches files, drop the trailing slash or add a \
                 filename pattern",
            ),
            (
                "*/",
                "pattern ends with \"/\"; glob matches files, drop the trailing slash or add a \
                 filename pattern",
            ),
            ("[**]", "wildcards are not valid inside a character class"),
            (
                "src/[a-*].rs",
                "wildcards are not valid inside a character class",
            ),
            ("src/[abc", "pattern has an unclosed character class"),
            ("src**/*.rs", "\"**\" must be a whole path segment"),
        ];

        for (pattern, expected) in cases {
            let error = WorkspaceGlob::try_new(pattern).expect_err("pattern is malformed");
            assert_eq!(error.to_string(), expected, "pattern {pattern:?}");
            assert_eq!(error.pattern(), pattern, "pattern {pattern:?}");
        }
    }

    #[test]
    fn unusual_class_members_stay_valid() {
        let cases = [
            ("src/[?]ib.rs", "src/?ib.rs", true),
            ("src/[?]ib.rs", "src/lib.rs", false),
            ("src/[]a]ib.rs", "src/]ib.rs", true),
            ("src/[]a]ib.rs", "src/aib.rs", true),
            ("src/[a-]ib.rs", "src/-ib.rs", true),
            ("src/[!]]ib.rs", "src/lib.rs", true),
        ];

        for (pattern, candidate, expected) in cases {
            let glob = WorkspaceGlob::try_new(pattern).expect("pattern compiles");
            assert_eq!(
                glob.is_match(candidate),
                expected,
                "pattern {pattern:?}, candidate {candidate:?}"
            );
        }
    }

    #[test]
    fn a_repeated_separator_reads_as_one() {
        let glob = WorkspaceGlob::try_new("src//*.rs").expect("pattern compiles");

        assert!(glob.is_match("src/lib.rs"));
    }

    #[test]
    fn repeated_recursive_wildcards_match_like_one() {
        let glob = WorkspaceGlob::try_new("**/**/**/*.rs").expect("pattern compiles");

        assert!(glob.is_match("lib.rs"));
        assert!(glob.is_match("a/b/c/lib.rs"));
        assert!(!glob.is_match("a/b/c/lib.md"));
    }

    #[test]
    fn traversal_roots_stop_at_the_first_wildcard() {
        let cases = [
            ("*.md", ""),
            ("**/*.md", ""),
            (".ai/reports/*.md", ".ai/reports"),
            (".ai/reports/nested/file.md", ".ai/reports/nested"),
            ("src/**/lib.rs", "src"),
        ];

        for (pattern, expected) in cases {
            let glob = WorkspaceGlob::try_new(pattern).expect("pattern compiles");
            assert_eq!(glob.traversal_root(), expected, "pattern {pattern:?}");
        }
    }
}
