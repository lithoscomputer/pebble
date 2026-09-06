//! Permission policies with one fixed answer.

use pebble_agent::ToolDescriptor;

use crate::SessionScope;
use crate::tool::{ToolPermission, ToolPermissionPolicy};

/// A policy that denies exactly one tool, by its stable identity, and allows
/// the rest.
#[derive(Debug, Clone, Copy)]
pub struct DenyTool(pub &'static str);

impl ToolPermissionPolicy for DenyTool {
    fn permission(&self, _session: &SessionScope, tool: &ToolDescriptor) -> ToolPermission {
        if tool.id().as_str() == self.0 {
            ToolPermission::Deny {
                reason: format!("{} denied by tool permission policy", tool.id()),
            }
        } else {
            ToolPermission::Allow
        }
    }
}

/// A policy that answers the same for every tool.
#[derive(Debug, Clone)]
pub struct FixedPermission(pub ToolPermission);

impl ToolPermissionPolicy for FixedPermission {
    fn permission(&self, _session: &SessionScope, _tool: &ToolDescriptor) -> ToolPermission {
        self.0.clone()
    }
}
