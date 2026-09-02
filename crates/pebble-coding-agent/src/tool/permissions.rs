//! Resolving tool names to identities, and the permission ladder's
//! auto-approval table.
//!
//! The session loop consults neither. It asks the
//! [`ToolAccessPolicy`](crate::tools::ToolAccessPolicy) an application
//! installed, and this module is what an application can build that policy out
//! of when it wants pebble's own answer instead of its own.

use super::native::NativeTool;
use crate::types::{PermissionLevel, ToolCategory};

/// Resolves a tool name in any profile's vocabulary to the canonical name the
/// rest of the crate reasons about.
///
/// A profile may expose a built-in tool under the vocabulary its model was
/// trained against — the Kimi profile uses Kimi Code's `Read`/`Edit`/`Bash`
/// names — but permissions, categories, and output limits must not depend on
/// which profile is running. Names that are not built-in (MCP, skill,
/// application-registered) pass through unchanged.
#[must_use]
pub fn canonical_tool_name(name: &str) -> &str {
    match NativeTool::from_any_name(name) {
        Some(tool) => tool.canonical_name(),
        None => name,
    }
}

/// The coarse access category of an exposed tool.
///
/// `None` means the name is outside the permission taxonomy, so callers can
/// decide what that means: a permission gate defaults it to
/// [`ToolCategory::Shell`], a tool summary reports [`ToolCategory::Other`].
#[must_use]
pub(crate) fn known_tool_category(name: &str) -> Option<ToolCategory> {
    NativeTool::from_any_name(name).and_then(NativeTool::category)
}

/// The permission-gate category of a tool.
///
/// An unknown tool falls back to [`ToolCategory::Shell`], so it needs explicit
/// approval at any level below [`PermissionLevel::Full`].
#[must_use]
pub(crate) fn tool_category(name: &str) -> ToolCategory {
    known_tool_category(name).unwrap_or(ToolCategory::Shell)
}

impl PermissionLevel {
    /// Whether this level runs `category` without asking.
    ///
    /// | category | `ReadOnly` | `ReadWrite` | `Full` |
    /// | --- | --- | --- | --- |
    /// | `Read`, `Subagent` | yes | yes | yes |
    /// | `Write` | no | yes | yes |
    /// | `Shell`, `Other` | no | no | yes |
    ///
    /// [`Full`](PermissionLevel::Full) approves everything, a category pebble
    /// adds later included; the levels below it approve only what they name,
    /// so a category they have never heard of needs approval.
    ///
    /// The `Full`/`Other` answer is the one place this table differs from
    /// fabro's, which refused that pair. No tool name reaches it either way —
    /// `tool_category` answers [`Shell`](ToolCategory::Shell) for a name it
    /// does not know, and [`Other`](ToolCategory::Other) is what a tool summary
    /// reports rather than what a gate is asked about.
    #[must_use]
    pub const fn auto_approves(self, category: ToolCategory) -> bool {
        match self {
            Self::Full => true,
            Self::ReadWrite => matches!(
                category,
                ToolCategory::Read | ToolCategory::Subagent | ToolCategory::Write
            ),
            Self::ReadOnly => matches!(category, ToolCategory::Read | ToolCategory::Subagent),
        }
    }

    /// Whether this level runs the named tool without asking.
    ///
    /// The name is resolved through `tool_category`, so a tool pebble does
    /// not know needs approval below [`PermissionLevel::Full`].
    #[must_use]
    pub fn auto_approves_tool(self, tool_name: &str) -> bool {
        self.auto_approves(tool_category(tool_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_names_pass_through() {
        assert_eq!(canonical_tool_name("read_file"), "read_file");
        assert_eq!(canonical_tool_name("shell"), "shell");
    }

    #[test]
    fn harness_names_resolve_to_canonical_names() {
        assert_eq!(canonical_tool_name("Read"), "read_file");
        assert_eq!(canonical_tool_name("Bash"), "shell");
        assert_eq!(canonical_tool_name("shell_command"), "shell");
        assert_eq!(canonical_tool_name("FetchURL"), "web_fetch");
    }

    #[test]
    fn unknown_names_pass_through() {
        assert_eq!(
            canonical_tool_name("mcp__filesystem__read_file"),
            "mcp__filesystem__read_file"
        );
    }

    #[test]
    fn known_categories_come_from_the_built_in_table() {
        assert_eq!(known_tool_category("Read"), Some(ToolCategory::Read));
        assert_eq!(
            known_tool_category("apply_patch"),
            Some(ToolCategory::Write)
        );
        assert_eq!(known_tool_category("Bash"), Some(ToolCategory::Shell));
        assert_eq!(
            known_tool_category("spawn_agent"),
            Some(ToolCategory::Subagent)
        );
        // Inside the built-in set, but outside the permission taxonomy.
        assert_eq!(known_tool_category("web_search"), None);
        assert_eq!(known_tool_category("mcp__files__read"), None);
    }

    #[test]
    fn uncategorized_tools_gate_as_shell() {
        assert_eq!(tool_category("web_search"), ToolCategory::Shell);
        assert_eq!(tool_category("mcp__files__read"), ToolCategory::Shell);
        assert_eq!(tool_category("Read"), ToolCategory::Read);
    }

    #[test]
    fn auto_approval_matches_the_permission_ladder() {
        let cases = [
            (ToolCategory::Read, [true, true, true]),
            (ToolCategory::Subagent, [true, true, true]),
            (ToolCategory::Write, [false, true, true]),
            (ToolCategory::Shell, [false, false, true]),
            (ToolCategory::Other, [false, false, true]),
        ];
        let levels = [
            PermissionLevel::ReadOnly,
            PermissionLevel::ReadWrite,
            PermissionLevel::Full,
        ];
        for (category, expected) in cases {
            for (level, expected) in levels.into_iter().zip(expected) {
                assert_eq!(
                    level.auto_approves(category),
                    expected,
                    "{level} should{} auto-approve {category}",
                    if expected { "" } else { " not" }
                );
            }
        }
    }

    #[test]
    fn tool_auto_approval_resolves_the_name_first() {
        // `Read` is Kimi Code's name for `read_file`.
        assert!(PermissionLevel::ReadOnly.auto_approves_tool("Read"));
        assert!(!PermissionLevel::ReadOnly.auto_approves_tool("Write"));
        assert!(PermissionLevel::ReadWrite.auto_approves_tool("Write"));
        assert!(!PermissionLevel::ReadWrite.auto_approves_tool("Bash"));
        assert!(PermissionLevel::Full.auto_approves_tool("Bash"));
        // Unknown tools need approval below `Full`.
        assert!(!PermissionLevel::ReadWrite.auto_approves_tool("mcp__files__write"));
        assert!(PermissionLevel::Full.auto_approves_tool("mcp__files__write"));
    }
}
