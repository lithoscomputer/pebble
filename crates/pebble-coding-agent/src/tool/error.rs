//! What a tool reports when a call does not produce output.

use std::error::Error as StdError;
use std::fmt::Write as _;

use crate::environment::{EnvironmentError, EnvironmentErrorKind};
use crate::types::ToolErrorKind;

/// A failed tool call.
///
/// A tool error crosses a boundary no other error in pebble crosses: the model
/// reads it. So it carries two things that stay apart. The
/// [`message`](Self::message) is written for the model — it names what failed
/// and what to do about it, and it never carries a secret, a stack, or an
/// internal identifier — while the underlying failure stays attached as the
/// error's source, for logs and for the application.
///
/// The execution layer owns rendering: it puts the message in the tool result,
/// marks the result as an error, and reports [`kind`](Self::kind) on
/// [`CodingEvent::ToolCallCompleted`](crate::events::CodingEvent::ToolCallCompleted) so
/// hooks and event consumers can branch without parsing text.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
#[non_exhaustive]
pub struct ToolError {
    kind:    ToolErrorKind,
    message: String,
    #[source]
    source:  Option<Box<dyn StdError + Send + Sync + 'static>>,
}

impl ToolError {
    /// Builds an error with no underlying cause.
    #[must_use]
    pub fn new(kind: ToolErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    /// Builds an error that keeps `source` as its cause.
    #[must_use]
    pub fn with_source(
        kind: ToolErrorKind,
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Builds an error that keeps an already boxed `source` as its cause.
    ///
    /// For a caller that takes another error apart rather than wrapping it: an
    /// error whose message is copied into the tool error would otherwise say
    /// the same thing twice in [`detail`](Self::detail), once as the message
    /// and once as its own first cause.
    #[must_use]
    pub(crate) fn with_boxed_source(
        kind: ToolErrorKind,
        message: impl Into<String>,
        source: Box<dyn StdError + Send + Sync + 'static>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(source),
        }
    }

    /// The arguments did not match the tool's schema or were unusable.
    #[must_use]
    pub fn invalid_arguments(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::InvalidArguments, message)
    }

    /// A policy or an approval callback refused the call.
    #[must_use]
    pub fn denied(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Denied, message)
    }

    /// The call was cancelled before it finished.
    #[must_use]
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Cancelled, message)
    }

    /// The tool is not available in this session.
    #[must_use]
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Unavailable, message)
    }

    /// The tool ran and failed.
    #[must_use]
    pub fn execution(message: impl Into<String>) -> Self {
        Self::new(ToolErrorKind::Execution, message)
    }

    /// The same failure, said differently to the model.
    ///
    /// The kind and the cause are kept, so this is how a caller adds what only
    /// it knows — which tool, which stage — to a message another layer wrote.
    #[must_use]
    pub(crate) fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// The category of this failure.
    #[must_use]
    pub const fn kind(&self) -> ToolErrorKind {
        self.kind
    }

    /// The model-facing message, without its causes.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The message followed by one `"\n  caused by: ..."` line per cause.
    ///
    /// For logs. What the model reads is [`message`](Self::message).
    #[must_use]
    pub fn detail(&self) -> String {
        let mut rendered = self.message.clone();
        let mut current = StdError::source(self);
        while let Some(cause) = current {
            let _ = write!(rendered, "\n  caused by: {cause}");
            current = cause.source();
        }
        rendered
    }
}

impl From<EnvironmentError> for ToolError {
    /// Carries an environment failure to the model.
    ///
    /// The environment's message is already written for the model, so it is
    /// kept as-is and the environment error becomes the cause. Only the kind
    /// is translated: an operation the environment does not offer is
    /// [`Unavailable`](ToolErrorKind::Unavailable), an argument it rejected is
    /// [`InvalidArguments`](ToolErrorKind::InvalidArguments), and everything
    /// else is a failure of the running tool.
    fn from(error: EnvironmentError) -> Self {
        let kind = match error.kind() {
            EnvironmentErrorKind::Unsupported => ToolErrorKind::Unavailable,
            EnvironmentErrorKind::InvalidInput => ToolErrorKind::InvalidArguments,
            _ => ToolErrorKind::Execution,
        };
        Self::with_source(kind, error.message().to_owned(), error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("disk is full")]
    struct Cause;

    #[test]
    fn constructors_set_their_kind() {
        assert_eq!(
            ToolError::invalid_arguments("bad").kind(),
            ToolErrorKind::InvalidArguments
        );
        assert_eq!(ToolError::denied("no").kind(), ToolErrorKind::Denied);
        assert_eq!(
            ToolError::cancelled("stop").kind(),
            ToolErrorKind::Cancelled
        );
        assert_eq!(
            ToolError::unavailable("gone").kind(),
            ToolErrorKind::Unavailable
        );
        assert_eq!(
            ToolError::execution("boom").kind(),
            ToolErrorKind::Execution
        );
    }

    #[test]
    fn the_message_is_what_display_writes() {
        let error = ToolError::execution("Exit code: 7");
        assert_eq!(error.message(), "Exit code: 7");
        assert_eq!(error.to_string(), "Exit code: 7");
        assert_eq!(error.detail(), "Exit code: 7");
    }

    #[test]
    fn a_source_stays_out_of_the_model_facing_message() {
        let error = ToolError::with_source(
            ToolErrorKind::Execution,
            "Failed to write file: /work/out.txt",
            Cause,
        );
        assert_eq!(error.message(), "Failed to write file: /work/out.txt");
        assert_eq!(error.to_string(), "Failed to write file: /work/out.txt");
        assert_eq!(
            error.detail(),
            "Failed to write file: /work/out.txt\n  caused by: disk is full"
        );
        assert!(StdError::source(&error).is_some());
    }

    #[test]
    fn environment_failures_keep_their_message_and_become_a_cause() {
        let environment = EnvironmentError::new(
            EnvironmentErrorKind::NotFound,
            "File not found: /work/a.txt",
        );
        let error = ToolError::from(environment);

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert_eq!(error.message(), "File not found: /work/a.txt");
        assert_eq!(
            error.detail(),
            "File not found: /work/a.txt\n  caused by: File not found: /work/a.txt"
        );
    }

    #[test]
    fn environment_kinds_a_caller_can_act_on_are_translated() {
        let unsupported = ToolError::from(EnvironmentError::new(
            EnvironmentErrorKind::Unsupported,
            "This environment cannot run commands",
        ));
        assert_eq!(unsupported.kind(), ToolErrorKind::Unavailable);

        let invalid = ToolError::from(EnvironmentError::new(
            EnvironmentErrorKind::InvalidInput,
            "Invalid glob pattern: ../*",
        ));
        assert_eq!(invalid.kind(), ToolErrorKind::InvalidArguments);
    }
}
