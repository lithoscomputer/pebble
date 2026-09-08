//! Prompts a person wrote for the model to follow on demand.
//!
//! A skill is a `SKILL.md` file: a short frontmatter block naming it and saying
//! when to reach for it, then the prompt itself. Pebble discovers skills in the
//! directories an application names, summarizes them in the system prompt, and
//! expands one into the conversation when a person types `/name` or the model
//! calls the skill tool.
//!
//! Discovery is path-explicit, the way memory loading is: pebble searches the
//! directories it is given and looks in no conventional location of its own.

use std::collections::HashMap;
use std::result::Result as StdResult;

use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::environment::{EnvResult, Environment};
use crate::error::{Error, InterruptReason, Result};
use crate::tool::{NativeTool, ToolVocabulary};
use crate::types::{SkillSummary, SkippedSkill, SkippedSkillReason};

/// The placeholder a template uses to say where the rest of the input goes.
///
/// Read here for a typed `/name`, and by the skill tool for what a model sent
/// with its call, so both put the input in the same place.
pub(crate) const USER_INPUT_PLACEHOLDER: &str = "{{user_input}}";

/// The file every skill directory holds.
const SKILL_FILE_GLOB: &str = "*/SKILL.md";

/// A prompt a person wrote for the model to follow on demand.
///
/// Pebble carries the three parts every profile's prompt assembly needs: what
/// to call it, when to reach for it, and what it says.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Skill {
    /// What the skill is named, and what a person types to invoke it.
    pub(crate) name:        String,
    /// When to use the skill, written for the model.
    pub(crate) description: String,
    /// The prompt the skill expands into.
    pub(crate) template:    String,
}

impl Skill {
    /// The description of this skill that the event stream carries.
    #[must_use]
    pub(crate) fn to_summary(&self) -> SkillSummary {
        SkillSummary {
            name:        self.name.clone(),
            description: self.description.clone(),
        }
    }
}

/// Why a `SKILL.md` file could not be read as a skill.
///
/// Discovery skips a file that fails to parse and reports the failure's text
/// on the [`SkippedSkill`] it records for the file.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub(crate) enum SkillParseError {
    /// The file does not open with a `---` frontmatter delimiter.
    #[error("Missing YAML frontmatter delimiters")]
    MissingFrontmatter,
    /// The frontmatter block is never closed.
    #[error("Missing closing frontmatter delimiter")]
    UnterminatedFrontmatter,
    /// The frontmatter names no skill.
    #[error("Missing required 'name' field in frontmatter")]
    MissingName,
}

/// Why a `/name` reference could not be expanded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SkillExpansionError {
    /// The input named more than one skill. One input runs one skill.
    #[error("Only one skill reference per input is allowed")]
    MultipleReferences,
    /// The input named a skill this session did not discover.
    #[error("Unknown skill: /{name}")]
    UnknownSkill {
        /// The name that was typed, without its leading `/`.
        name: String,
    },
}

/// Whether input is checked for a `/name` skill reference.
///
/// Input a person typed is expanded. Input the harness synthesized is not:
/// a subagent's result envelope can hold `/tmp` or a slash-prefixed word of its
/// own, and expanding it would either fail on an unknown skill or splice a
/// template over the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum SkillExpansion {
    /// Expand a `/name` reference in the input.
    Apply,
    /// Send the input through unchanged.
    Skip,
}

/// Input after a `/name` reference was expanded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpandedInput {
    /// What the session sends as the user turn.
    pub(crate) text:       String,
    /// The skill that was expanded, when one was.
    pub(crate) skill_name: Option<String>,
}

/// Reads a `SKILL.md` file.
///
/// The frontmatter is scanned line by line for `name:` and `description:`
/// rather than parsed as YAML, so a skill file stays readable to a person and
/// pebble takes on no YAML dependency. Anything else in the block is ignored,
/// `name` is required, `description` defaults to empty, and the body becomes
/// the template with its surrounding blank lines trimmed.
///
/// ```ignore
/// # use pebble_coding_agent::resources::{SkillParseError, parse_skill};
/// let skill = parse_skill("---\nname: commit\n---\nWrite a commit.")?;
/// assert_eq!(skill.name, "commit");
/// assert_eq!(skill.template, "Write a commit.");
/// # Ok::<(), SkillParseError>(())
/// ```
pub(crate) fn parse_skill(content: &str) -> StdResult<Skill, SkillParseError> {
    let trimmed = content.trim();
    let Some(after_opening) = trimmed.strip_prefix("---") else {
        return Err(SkillParseError::MissingFrontmatter);
    };

    let end = after_opening
        .find("\n---")
        .ok_or(SkillParseError::UnterminatedFrontmatter)?;
    let frontmatter = &after_opening[..end];
    let body = &after_opening[end + "\n---".len()..];

    let mut name: Option<String> = None;
    let mut description = String::new();

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("name:") {
            name = Some(value.trim().to_owned());
        } else if let Some(value) = line.strip_prefix("description:") {
            value.trim().clone_into(&mut description);
        }
    }

    Ok(Skill {
        name: name.ok_or(SkillParseError::MissingName)?,
        description,
        template: body.trim().to_owned(),
    })
}

/// Expands a `/name` reference in what a person typed.
///
/// Input naming no skill passes through unchanged. Input naming one has the
/// token removed and the remainder substituted into the template's
/// `{{user_input}}` placeholder; a template with no placeholder replaces the
/// input entirely.
///
/// ```ignore
/// # use pebble_coding_agent::resources::{Skill, SkillExpansionError, expand_skill};
/// let skills = [Skill {
///     name:        "commit".to_owned(),
///     description: "Make a commit".to_owned(),
///     template:    "Review changes and commit.\n\n{{user_input}}".to_owned(),
/// }];
///
/// let expanded = expand_skill(&skills, "/commit only the staged files")?;
///
/// assert_eq!(expanded.skill_name.as_deref(), Some("commit"));
/// assert!(expanded.text.ends_with("only the staged files"));
/// # Ok::<(), SkillExpansionError>(())
/// ```
pub(crate) fn expand_skill(
    skills: &[Skill],
    input: &str,
) -> StdResult<ExpandedInput, SkillExpansionError> {
    let mut references = find_skill_references(input);

    if references.len() > 1 {
        return Err(SkillExpansionError::MultipleReferences);
    }

    let Some(reference) = references.pop() else {
        return Ok(ExpandedInput {
            text:       input.to_owned(),
            skill_name: None,
        });
    };

    let skill = skills
        .iter()
        .find(|skill| skill.name == reference.name)
        .ok_or_else(|| SkillExpansionError::UnknownSkill {
            name: reference.name.clone(),
        })?;

    let remainder = format!("{}{}", &input[..reference.start], &input[reference.end..])
        .trim()
        .to_owned();

    Ok(ExpandedInput {
        text:       substitute(&skill.template, &remainder),
        skill_name: Some(reference.name),
    })
}

/// Puts `input` where the template asks for it, or answers the template alone.
fn substitute(template: &str, input: &str) -> String {
    if template.contains(USER_INPUT_PLACEHOLDER) {
        template.replace(USER_INPUT_PLACEHOLDER, input)
    } else {
        template.to_owned()
    }
}

/// A `/name` token found in input: the name and the bytes it occupies.
#[derive(Debug)]
struct SkillReference {
    name:  String,
    /// Byte offset of the `/`.
    start: usize,
    /// Byte offset just past the name.
    end:   usize,
}

/// Whether a byte may appear after the first character of a skill name.
const fn is_name_byte(byte: u8) -> bool {
    byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
}

/// Finds every `/name` token in `input`.
///
/// A token starts at a `/` that begins the input or follows whitespace, its
/// first character is a lowercase letter, and it ends at whitespace or the end
/// of the input. That is what keeps `/usr/bin/bash` from reading as a skill.
fn find_skill_references(input: &str) -> Vec<SkillReference> {
    let bytes = input.as_bytes();
    let mut references = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] != b'/' {
            index += 1;
            continue;
        }

        let starts_a_token = index == 0 || bytes[index - 1].is_ascii_whitespace();
        let name_start = index + 1;
        if !starts_a_token || name_start >= bytes.len() || !bytes[name_start].is_ascii_lowercase() {
            index += 1;
            continue;
        }

        let mut end = name_start + 1;
        while end < bytes.len() && is_name_byte(bytes[end]) {
            end += 1;
        }

        if end >= bytes.len() || bytes[end].is_ascii_whitespace() {
            references.push(SkillReference {
                name: input[name_start..end].to_owned(),
                start: index,
                end,
            });
        }

        index = end;
    }

    references
}

/// Renders the skills section of a system prompt.
///
/// Empty skills render nothing at all, so a session with none carries no
/// mention of a tool it does not have. The instruction names the skill tool the
/// way `vocabulary` spells it, because the model has to be told the name it can
/// actually call.
#[must_use]
pub(crate) fn format_skills_prompt_section(skills: &[Skill], vocabulary: ToolVocabulary) -> String {
    if skills.is_empty() {
        return String::new();
    }

    let skill_tool = NativeTool::UseSkill.name(vocabulary);
    let mut lines = vec![
        "# Available Skills".to_owned(),
        format!(
            "When the user's request matches a skill below, call the `{skill_tool}` tool to load \
             its instructions, then follow them."
        ),
    ];
    for skill in skills {
        if skill.description.is_empty() {
            lines.push(format!("- `{}`", skill.name));
        } else {
            lines.push(format!("- `{}`: {}", skill.name, skill.description));
        }
    }
    lines.join("\n")
}

/// What one discovery found, and what it had to skip.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct SkillDiscovery {
    /// The skills that were found, sorted by name.
    pub(crate) skills:  Vec<Skill>,
    /// The files and directories that were skipped, in discovery order.
    pub(crate) skipped: Vec<SkippedSkill>,
}

/// Discovers the skills in the given directories.
///
/// Each directory is searched one level deep for `<name>/SKILL.md`. A directory
/// that cannot be searched, a file that cannot be read, and a file that does
/// not parse are all skipped: one broken skill must not cost a session the
/// rest. Each skip is recorded on the result, so an application can tell a
/// person which file to fix without reading every skill itself.
///
/// Directories are searched in order and a later one wins, so an application
/// lists shared directories before the ones that should override them. The
/// skills are sorted by name.
///
/// Returns [`Error::Interrupted`] when `cancel` fires, which is checked around
/// every search and every read.
pub(crate) async fn discover_skills(
    env: &dyn Environment,
    dirs: &[String],
    cancel: &CancellationToken,
) -> Result<SkillDiscovery> {
    let mut by_name: HashMap<String, Skill> = HashMap::new();
    let mut skipped = Vec::new();

    for dir in dirs {
        let paths = match skill_files(env, dir, cancel).await? {
            Ok(paths) => paths,
            Err(error) => {
                debug!(dir, %error, "Skill directory could not be searched, skipping");
                skipped.push(SkippedSkill {
                    path:    dir.clone(),
                    reason:  SkippedSkillReason::UnsearchableDirectory,
                    message: error.to_string(),
                });
                continue;
            }
        };
        for path in paths {
            let content = match read_skill_file(env, &path, cancel).await? {
                Ok(content) => content,
                Err(error) => {
                    debug!(path, %error, "Skill file could not be read, skipping");
                    skipped.push(SkippedSkill {
                        path,
                        reason: SkippedSkillReason::UnreadableFile,
                        message: error.to_string(),
                    });
                    continue;
                }
            };
            match parse_skill(&content) {
                Ok(skill) => {
                    by_name.insert(skill.name.clone(), skill);
                }
                Err(error) => {
                    debug!(path, %error, "Skill file did not parse, skipping");
                    skipped.push(SkippedSkill {
                        path,
                        reason: SkippedSkillReason::Malformed,
                        message: error.to_string(),
                    });
                }
            }
        }
    }

    let mut skills: Vec<Skill> = by_name.into_values().collect();
    skills.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(SkillDiscovery { skills, skipped })
}

/// The `SKILL.md` files one directory holds, or the reason it could not be
/// searched.
async fn skill_files(
    env: &dyn Environment,
    dir: &str,
    cancel: &CancellationToken,
) -> Result<EnvResult<Vec<String>>> {
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    let found = env.glob(SKILL_FILE_GLOB, Some(dir)).await;

    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    Ok(found)
}

/// Reads one skill file, or the reason it could not be read.
async fn read_skill_file(
    env: &dyn Environment,
    path: &str,
    cancel: &CancellationToken,
) -> Result<EnvResult<String>> {
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    let read = env.read_file_text(path).await;

    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    Ok(read)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::test_support::MockEnvironment;

    fn test_skills() -> Vec<Skill> {
        vec![
            Skill {
                name:        "commit".into(),
                description: "Create a commit".into(),
                template:    "Review changes and commit.\n\n{{user_input}}".into(),
            },
            Skill {
                name:        "test".into(),
                description: "Run tests".into(),
                template:    "Run the test suite.".into(),
            },
        ]
    }

    fn environment(files: &[(&str, &str)], glob_results: &[&str]) -> MockEnvironment {
        MockEnvironment {
            files: files
                .iter()
                .map(|(path, content)| ((*path).to_owned(), (*content).to_owned()))
                .collect::<HashMap<_, _>>(),
            glob_results: glob_results.iter().map(|path| (*path).to_owned()).collect(),
            ..MockEnvironment::default()
        }
    }

    fn dirs(dirs: &[&str]) -> Vec<String> {
        dirs.iter().map(|dir| (*dir).to_owned()).collect()
    }

    // --- parsing ---

    #[test]
    fn a_skill_file_parses_into_its_three_parts() {
        let content = "\
---
name: commit
description: Create a git commit following best practices
---

Review staged and unstaged changes, then create a well-crafted commit.

{{user_input}}";

        let skill = parse_skill(content).expect("the file parses");

        assert_eq!(skill.name, "commit");
        assert_eq!(
            skill.description,
            "Create a git commit following best practices"
        );
        assert!(skill.template.contains("Review staged"));
        assert!(skill.template.contains("{{user_input}}"));
    }

    #[test]
    fn a_file_without_frontmatter_is_not_a_skill() {
        assert_eq!(
            parse_skill("Just some markdown without frontmatter"),
            Err(SkillParseError::MissingFrontmatter)
        );
    }

    #[test]
    fn unterminated_frontmatter_is_not_a_skill() {
        assert_eq!(
            parse_skill("---\nname: commit\nstill going"),
            Err(SkillParseError::UnterminatedFrontmatter)
        );
    }

    #[test]
    fn a_skill_must_be_named() {
        assert_eq!(
            parse_skill("---\ndescription: A skill without a name\n---\n\nSome template"),
            Err(SkillParseError::MissingName)
        );
    }

    #[test]
    fn a_description_is_optional() {
        let skill = parse_skill("---\nname: simple\n---\n\nJust a template").expect("it parses");

        assert_eq!(skill.name, "simple");
        assert_eq!(skill.description, "");
        assert_eq!(skill.template, "Just a template");
    }

    #[test]
    fn a_template_is_trimmed() {
        let content = "---\nname: trimmed\n---\n\n\n  Body with surrounding whitespace\n\n\n";

        let skill = parse_skill(content).expect("it parses");

        assert_eq!(skill.template, "Body with surrounding whitespace");
    }

    #[test]
    fn frontmatter_keys_pebble_does_not_know_are_ignored() {
        let skill =
            parse_skill("---\nname: keyed\nallowed-tools: Bash\n---\nBody").expect("it parses");

        assert_eq!(skill.name, "keyed");
        assert_eq!(skill.template, "Body");
    }

    // --- expansion ---

    #[test]
    fn input_naming_no_skill_passes_through() {
        let expanded = expand_skill(&test_skills(), "just some plain text").expect("it expands");

        assert_eq!(expanded.text, "just some plain text");
        assert_eq!(expanded.skill_name, None);
    }

    #[test]
    fn a_reference_at_the_start_takes_the_rest_as_input() {
        let expanded = expand_skill(&test_skills(), "/commit do the thing").expect("it expands");

        assert_eq!(expanded.text, "Review changes and commit.\n\ndo the thing");
        assert_eq!(expanded.skill_name.as_deref(), Some("commit"));
    }

    #[test]
    fn a_reference_mid_line_is_removed_from_the_input() {
        let expanded =
            expand_skill(&test_skills(), "please /commit the auth changes").expect("it expands");

        assert_eq!(
            expanded.text,
            "Review changes and commit.\n\nplease  the auth changes"
        );
        assert_eq!(expanded.skill_name.as_deref(), Some("commit"));
    }

    #[test]
    fn a_reference_alone_leaves_the_placeholder_empty() {
        let expanded = expand_skill(&test_skills(), "/commit").expect("it expands");

        assert_eq!(expanded.text, "Review changes and commit.\n\n");
        assert_eq!(expanded.skill_name.as_deref(), Some("commit"));
    }

    #[test]
    fn an_unknown_skill_is_reported_by_name() {
        assert_eq!(
            expand_skill(&test_skills(), "/nonexistent"),
            Err(SkillExpansionError::UnknownSkill {
                name: "nonexistent".to_owned(),
            })
        );
    }

    #[test]
    fn a_path_is_not_a_skill_reference() {
        let expanded = expand_skill(&test_skills(), "/usr/bin/bash").expect("it expands");

        assert_eq!(expanded.text, "/usr/bin/bash");
        assert_eq!(expanded.skill_name, None);
    }

    #[test]
    fn a_slash_inside_a_word_is_not_a_skill_reference() {
        let expanded = expand_skill(&test_skills(), "read src/commit and stop").expect("expands");

        assert_eq!(expanded.text, "read src/commit and stop");
        assert_eq!(expanded.skill_name, None);
    }

    #[test]
    fn one_input_may_name_only_one_skill() {
        assert_eq!(
            expand_skill(&test_skills(), "/commit and /test"),
            Err(SkillExpansionError::MultipleReferences)
        );
    }

    #[test]
    fn a_template_without_a_placeholder_replaces_the_input() {
        let expanded = expand_skill(&test_skills(), "/test please run").expect("it expands");

        assert_eq!(expanded.text, "Run the test suite.");
        assert_eq!(expanded.skill_name.as_deref(), Some("test"));
    }

    // --- prompt section ---

    #[test]
    fn no_skills_render_no_section() {
        assert_eq!(
            format_skills_prompt_section(&[], ToolVocabulary::Canonical),
            ""
        );
    }

    #[test]
    fn the_section_lists_every_skill() {
        let section = format_skills_prompt_section(&test_skills(), ToolVocabulary::Canonical);

        assert!(section.contains("# Available Skills"));
        assert!(section.contains("call the `use_skill` tool"));
        assert!(section.contains("- `commit`: Create a commit"));
        assert!(section.contains("- `test`: Run tests"));
    }

    #[test]
    fn the_section_names_the_skill_tool_the_way_the_vocabulary_does() {
        let skills = test_skills();

        assert!(
            format_skills_prompt_section(&skills, ToolVocabulary::Canonical)
                .contains("`use_skill`")
        );
        assert!(
            format_skills_prompt_section(&skills, ToolVocabulary::KimiCode).contains("`Skill`")
        );
        assert!(format_skills_prompt_section(&skills, ToolVocabulary::Claude5).contains("`Skill`"));
    }

    #[test]
    fn a_skill_without_a_description_is_listed_by_name_alone() {
        let skills = [Skill {
            name: "bare".into(),
            ..Skill::default()
        }];

        let section = format_skills_prompt_section(&skills, ToolVocabulary::Canonical);

        assert!(section.ends_with("- `bare`"));
    }

    // --- discovery ---

    #[tokio::test]
    async fn discovery_loads_the_skill_files_it_finds() {
        let env = environment(
            &[(
                "/skills/commit/SKILL.md",
                "---\nname: commit\ndescription: Make a commit\n---\nDo commit",
            )],
            &["/skills/commit/SKILL.md"],
        );

        let skills = discover_skills(&env, &dirs(&["/skills"]), &CancellationToken::new())
            .await
            .expect("discovery succeeds")
            .skills;

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "commit");
        assert_eq!(skills[0].description, "Make a commit");
    }

    #[tokio::test]
    async fn a_file_that_does_not_parse_is_skipped_and_reported() {
        let env = environment(
            &[
                ("/skills/good/SKILL.md", "---\nname: good\n---\nGood"),
                ("/skills/bad/SKILL.md", "no frontmatter here"),
            ],
            &["/skills/good/SKILL.md", "/skills/bad/SKILL.md"],
        );

        let discovery = discover_skills(&env, &dirs(&["/skills"]), &CancellationToken::new())
            .await
            .expect("discovery succeeds");

        assert_eq!(discovery.skills.len(), 1);
        assert_eq!(discovery.skills[0].name, "good");
        assert_eq!(discovery.skipped, [SkippedSkill {
            path:    "/skills/bad/SKILL.md".to_owned(),
            reason:  SkippedSkillReason::Malformed,
            message: SkillParseError::MissingFrontmatter.to_string(),
        }]);
    }

    #[tokio::test]
    async fn a_file_that_cannot_be_read_is_skipped_and_reported() {
        let env = environment(&[], &["/skills/gone/SKILL.md"]);

        let discovery = discover_skills(&env, &dirs(&["/skills"]), &CancellationToken::new())
            .await
            .expect("discovery succeeds");

        assert!(discovery.skills.is_empty());
        assert_eq!(discovery.skipped.len(), 1);
        assert_eq!(discovery.skipped[0].path, "/skills/gone/SKILL.md");
        assert_eq!(
            discovery.skipped[0].reason,
            SkippedSkillReason::UnreadableFile
        );
        assert!(
            !discovery.skipped[0].message.is_empty(),
            "the environment's reason is carried"
        );
    }

    #[tokio::test]
    async fn a_clean_discovery_reports_nothing_skipped() {
        let env = environment(
            &[("/skills/good/SKILL.md", "---\nname: good\n---\nGood")],
            &["/skills/good/SKILL.md"],
        );

        let discovery = discover_skills(&env, &dirs(&["/skills"]), &CancellationToken::new())
            .await
            .expect("discovery succeeds");

        assert_eq!(discovery.skills.len(), 1);
        assert!(discovery.skipped.is_empty());
    }

    #[tokio::test]
    async fn no_directories_discover_nothing() {
        let skills = discover_skills(&MockEnvironment::default(), &[], &CancellationToken::new())
            .await
            .expect("discovery succeeds")
            .skills;

        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn a_later_directory_overrides_an_earlier_one() {
        // The mock answers every search with the same list, which is the shape
        // discovery has to survive: the same name seen twice, the later read
        // winning.
        let env = environment(
            &[
                (
                    "/global/commit/SKILL.md",
                    "---\nname: commit\ndescription: Global commit\n---\nGlobal",
                ),
                (
                    "/project/commit/SKILL.md",
                    "---\nname: commit\ndescription: Project commit\n---\nProject",
                ),
            ],
            &["/global/commit/SKILL.md", "/project/commit/SKILL.md"],
        );

        let skills = discover_skills(
            &env,
            &dirs(&["/global", "/project"]),
            &CancellationToken::new(),
        )
        .await
        .expect("discovery succeeds")
        .skills;

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "Project commit");
    }

    #[tokio::test]
    async fn discovered_skills_are_sorted_by_name() {
        let env = environment(
            &[
                ("/skills/z/SKILL.md", "---\nname: zebra\n---\nZ"),
                ("/skills/a/SKILL.md", "---\nname: apple\n---\nA"),
            ],
            &["/skills/z/SKILL.md", "/skills/a/SKILL.md"],
        );

        let skills = discover_skills(&env, &dirs(&["/skills"]), &CancellationToken::new())
            .await
            .expect("discovery succeeds")
            .skills;

        assert_eq!(
            skills
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            ["apple", "zebra"]
        );
    }

    #[tokio::test]
    async fn a_cancelled_discovery_is_interrupted() {
        let env = environment(&[], &[]);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let error = discover_skills(&env, &dirs(&["/skills"]), &cancel)
            .await
            .expect_err("a cancelled discovery fails");

        assert!(matches!(
            error,
            Error::Interrupted(InterruptReason::Cancelled)
        ));
    }

    #[tokio::test]
    async fn a_discovered_skill_describes_itself_for_the_event_stream() {
        let skill = &test_skills()[0];

        assert_eq!(skill.to_summary(), SkillSummary {
            name:        "commit".to_owned(),
            description: "Create a commit".to_owned(),
        });
    }
}
