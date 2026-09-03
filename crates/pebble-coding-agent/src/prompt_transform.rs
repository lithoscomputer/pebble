//! Adjusting the system prompt without replacing the harness.
//!
//! The profile a model resolves to writes the system prompt, and an application
//! cannot install a profile of its own: the tool vocabulary, the compaction
//! behavior, and the harness selection stay pebble's. What an application can
//! do is see the prompt the profile wrote — with the context it was written
//! from — and use it, add to it, or replace it. That is the whole of this seam.
//!
//! A transform runs once, when the session initializes, and applies to the root
//! session only. A child session runs its parent's profile and prompt: it is
//! given a task, not the application's persona.

use std::fmt;

use crate::profile::EnvContext;
use crate::types::{MemoryFileSummary, SkillSummary, ToolSummary};

/// What a [`SystemPromptTransform`] is shown.
///
/// Everything here is read-only and describes the session as the default
/// prompt describes it: the environment the prompt was written for, the tools
/// registered before per-turn middleware filters them, and what was loaded
/// into the prompt as memory and skills. The summaries carry descriptions,
/// never the loaded text.
#[derive(Clone, Copy)]
pub struct SystemPromptContext<'a> {
    default_prompt: &'a str,
    environment:    &'a EnvContext,
    tools:          &'a [ToolSummary],
    memory:         &'a [MemoryFileSummary],
    skills:         &'a [SkillSummary],
}

impl<'a> SystemPromptContext<'a> {
    pub(crate) const fn new(
        default_prompt: &'a str,
        environment: &'a EnvContext,
        tools: &'a [ToolSummary],
        memory: &'a [MemoryFileSummary],
        skills: &'a [SkillSummary],
    ) -> Self {
        Self {
            default_prompt,
            environment,
            tools,
            memory,
            skills,
        }
    }

    /// The prompt the selected profile wrote.
    #[must_use]
    pub const fn default_prompt(&self) -> &'a str {
        self.default_prompt
    }

    /// Where the session is working, as the default prompt describes it.
    #[must_use]
    pub const fn environment(&self) -> &'a EnvContext {
        self.environment
    }

    /// The registered tools, under their model-visible names.
    ///
    /// Per-turn middleware can hide some of these tools later.
    #[must_use]
    pub const fn tools(&self) -> &'a [ToolSummary] {
        self.tools
    }

    /// The memory files loaded into the default prompt.
    #[must_use]
    pub const fn memory(&self) -> &'a [MemoryFileSummary] {
        self.memory
    }

    /// The skills discovered for the session.
    #[must_use]
    pub const fn skills(&self) -> &'a [SkillSummary] {
        self.skills
    }
}

impl fmt::Debug for SystemPromptContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SystemPromptContext")
            .field("default_prompt_len", &self.default_prompt.len())
            .field("environment", self.environment)
            .field("tools", &self.tools.len())
            .field("memory", &self.memory.len())
            .field("skills", &self.skills.len())
            .finish()
    }
}

/// What a [`SystemPromptTransform`] decided.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SystemPromptDecision {
    /// Send the default prompt unchanged.
    UseDefault,
    /// Send the default prompt followed by this text, separated by a blank
    /// line.
    Append(String),
    /// Send this text in place of the default prompt.
    ///
    /// The tools, their names, and the runtime behavior around them are
    /// unchanged; only the words the model reads first are.
    Replace(String),
}

impl SystemPromptDecision {
    /// The prompt the session sends, given the default.
    pub(crate) fn apply(self, default_prompt: String) -> String {
        match self {
            Self::UseDefault => default_prompt,
            Self::Append(text) => {
                if default_prompt.is_empty() {
                    text
                } else {
                    format!("{default_prompt}\n\n{text}")
                }
            }
            Self::Replace(text) => text,
        }
    }
}

/// Adjusts the system prompt a session starts with.
///
/// Called once, as the root session initializes, with the prompt the selected
/// profile wrote and the context it was written from. Implementations must be
/// cheap and must not block: they run inside the session's initialization.
pub trait SystemPromptTransform: Send + Sync {
    /// Decides what the session sends as its system prompt.
    fn transform(&self, context: SystemPromptContext<'_>) -> SystemPromptDecision;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_used_as_is() {
        assert_eq!(
            SystemPromptDecision::UseDefault.apply("default".to_owned()),
            "default"
        );
    }

    #[test]
    fn an_append_follows_the_default_after_a_blank_line() {
        assert_eq!(
            SystemPromptDecision::Append("more".to_owned()).apply("default".to_owned()),
            "default\n\nmore"
        );
    }

    #[test]
    fn an_append_to_an_empty_default_is_just_the_addition() {
        assert_eq!(
            SystemPromptDecision::Append("more".to_owned()).apply(String::new()),
            "more"
        );
    }

    #[test]
    fn a_replacement_discards_the_default() {
        assert_eq!(
            SystemPromptDecision::Replace("instead".to_owned()).apply("default".to_owned()),
            "instead"
        );
    }
}
