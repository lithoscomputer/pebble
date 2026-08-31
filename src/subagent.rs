//! The supervisor a session delegates work to.
//!
//! A session that was given subagent support hands spawned tasks to a
//! [`SubagentSupervisor`], asks it at every input boundary whether a background
//! child has finished, and closes its children before it ends itself.
//!
//! This module is the seam only. The supervisor pebble ships today accepts no
//! spawns — it answers every one with an
//! [`Unavailable`](ToolErrorKind::Unavailable) tool error — so the session's
//! call sites are written once and the real supervisor replaces the bodies
//! below without touching them.

use std::result::Result as StdResult;

use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::tool::ToolError;

/// What a spawn is told while pebble ships no supervisor.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the spawn path belongs to the subagent tools, which land later"
    )
)]
const UNAVAILABLE: &str = "Subagents are not available in this session.";

/// One session's children: spawning, background results, and teardown.
///
/// Cloning is cheap and every clone drives the same children, which is what
/// lets a session hold one and hand another to the tools that spawn.
#[derive(Clone, Debug, Default)]
pub(crate) struct SubagentSupervisor;

impl SubagentSupervisor {
    /// Starts a child on `task`, answering with its identifier.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] the calling tool gives straight to the model,
    /// because a refused spawn is the tool's answer rather than the session's
    /// failure. Today every spawn is refused.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the spawn path belongs to the subagent tools, which land later"
        )
    )]
    #[expect(
        clippy::unused_self,
        reason = "the real supervisor spawns from state it keeps"
    )]
    pub(crate) fn spawn(&self, task: &str) -> StdResult<String, ToolError> {
        let _ = task;
        Err(ToolError::unavailable(UNAVAILABLE))
    }

    /// The next parent turn carrying finished background results, when there
    /// is one ready.
    ///
    /// Answers `None` as soon as nothing is pending, so the session's boundary
    /// drain ends rather than waiting for children it does not have.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Interrupted`] when `cancel` fires while the
    /// supervisor is waiting on a child.
    #[expect(
        clippy::unused_async,
        reason = "the real supervisor waits on its children here"
    )]
    pub(crate) async fn next_parent_notification_turn(
        &self,
        cancel: &CancellationToken,
    ) -> Result<Option<String>> {
        let _ = cancel;
        Ok(None)
    }

    /// Closes every child and waits for each to finish.
    ///
    /// The session calls this before it publishes its own end, so a child's
    /// closing events always precede the parent's.
    #[expect(
        clippy::unused_async,
        reason = "the real supervisor waits for its children to close"
    )]
    pub(crate) async fn shutdown_all(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolErrorKind;

    #[test]
    fn a_spawn_is_refused_with_a_message_for_the_model() {
        let error = SubagentSupervisor
            .spawn("review the diff")
            .expect_err("pebble ships no supervisor yet");

        assert_eq!(error.kind(), ToolErrorKind::Unavailable);
        assert_eq!(error.message(), UNAVAILABLE);
    }

    #[tokio::test]
    async fn nothing_is_pending_at_a_boundary() {
        let supervisor = SubagentSupervisor;

        assert_eq!(
            supervisor
                .next_parent_notification_turn(&CancellationToken::new())
                .await
                .expect("no children, no failure"),
            None
        );
        supervisor.shutdown_all().await;
    }
}
