//! Asking the person at the terminal before a tool the permission level does
//! not allow outright.

use std::io::{self, Write as _};
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use pebble_agent::{ToolCallRequest, ToolSystemError};
use pebble_coding_agent::tools::{ApprovalDecision, PermissionLevel, ToolApprovalService};
use tokio::task::spawn_blocking;

/// Approval at the terminal: a tool the level allows runs; otherwise the
/// person is asked on standard error and answers on standard input. Without
/// a terminal, or when asking was turned off, such tools are refused.
///
/// Answering "always" raises the level to [`PermissionLevel::Full`] for the
/// rest of the session.
pub struct TerminalApproval {
    level:       Mutex<PermissionLevel>,
    interactive: bool,
}

impl TerminalApproval {
    /// Approval that asks when `interactive` and refuses otherwise.
    #[must_use]
    pub fn new(level: PermissionLevel, interactive: bool) -> Self {
        Self {
            level: Mutex::new(level),
            interactive,
        }
    }

    fn level(&self) -> PermissionLevel {
        *self.level.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl ToolApprovalService for TerminalApproval {
    async fn approve(
        &self,
        request: &ToolCallRequest,
    ) -> Result<ApprovalDecision, ToolSystemError> {
        let tool_name = request.call().name.clone();
        if self.level().auto_approves_tool(&tool_name) {
            return Ok(ApprovalDecision::Allow);
        }
        if !self.interactive {
            return Ok(ApprovalDecision::Deny {
                reason: format!("{tool_name} tool denied at current permission level"),
            });
        }
        let asked = tool_name.clone();
        let answer = spawn_blocking(move || ask(&asked))
            .await
            .map_err(|error| ToolSystemError::new(format!("approval prompt failed: {error}")))?;
        Ok(match answer {
            Ok(Answer::Allow) => ApprovalDecision::Allow,
            Ok(Answer::AllowAlways) => {
                *self.level.lock().unwrap_or_else(PoisonError::into_inner) = PermissionLevel::Full;
                ApprovalDecision::Allow
            }
            Ok(Answer::Deny) => ApprovalDecision::Deny {
                reason: format!("{tool_name} tool denied by user"),
            },
            Err(reason) => ApprovalDecision::Deny { reason },
        })
    }
}

enum Answer {
    Allow,
    AllowAlways,
    Deny,
}

/// Asks on standard error and reads one line from standard input. Runs on a
/// blocking task: the person may take their time.
#[expect(
    clippy::print_stderr,
    reason = "the approval prompt is the command's stderr boundary"
)]
fn ask(tool_name: &str) -> Result<Answer, String> {
    eprint!("Allow {tool_name}? [y]es / [n]o / [a]lways: ");
    io::stderr().flush().ok();
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|error| format!("failed to read the answer: {error}"))?;
    Ok(match input.trim().to_lowercase().as_str() {
        "y" | "yes" => Answer::Allow,
        "a" | "always" => Answer::AllowAlways,
        _ => Answer::Deny,
    })
}
