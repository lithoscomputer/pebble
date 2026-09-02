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
/// error's source, for logs and for the application. An OS error string is
/// not a secret: an environment failure puts its causes into the message, so
/// the model can tell a missing file from a refused one.
///
/// The execution layer owns rendering: it puts the message in the tool result
/// after the session's redactor has seen it, marks the result as an error, and
/// reports [`kind`](Self::kind) on
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
    /// For logs. What the model reads is [`message`](Self::message); a
    /// producer that wants the model to see a cause writes it into the
    /// message, as [`From<EnvironmentError>`](Self::from) does.
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
    /// Carries an environment failure to the model, causes included.
    ///
    /// The model-facing message is the environment's
    /// [`detail`](EnvironmentError::detail): its message plus one `caused by`
    /// line per cause, so `Permission denied` and `No such file or directory`
    /// read differently. The environment error's own cause is carried on as
    /// the source, rather than the whole error, so [`detail`](Self::detail)
    /// does not repeat the message as its own first cause; it does repeat the
    /// cause line, because the message carries it too. Only the kind is
    /// translated: an operation the environment does not offer is
    /// [`Unavailable`](ToolErrorKind::Unavailable), an argument it rejected is
    /// [`InvalidArguments`](ToolErrorKind::InvalidArguments), and everything
    /// else is a failure of the running tool.
    fn from(error: EnvironmentError) -> Self {
        let kind = match error.kind() {
            EnvironmentErrorKind::Unsupported => ToolErrorKind::Unavailable,
            EnvironmentErrorKind::InvalidInput => ToolErrorKind::InvalidArguments,
            _ => ToolErrorKind::Execution,
        };
        let message = error.detail();
        match error.into_source() {
            Some(source) => Self::with_boxed_source(kind, message, source),
            None => Self::new(kind, message),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, ErrorKind};

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
    fn a_source_given_directly_stays_out_of_the_model_facing_message() {
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
    fn an_environment_io_cause_is_part_of_the_model_facing_message() {
        let refused = io::Error::new(
            ErrorKind::PermissionDenied,
            "Permission denied (os error 13)",
        );
        let error = ToolError::from(EnvironmentError::io(
            "Failed to read /work/config.yml",
            refused,
        ));

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert_eq!(
            error.message(),
            "Failed to read /work/config.yml\n  caused by: Permission denied (os error 13)"
        );
        assert_eq!(error.to_string(), error.message());
    }

    /// The source is the environment error's own cause, not the environment
    /// error: its message is already the tool error's message, and a log would
    /// otherwise read it twice.
    #[test]
    fn an_environment_failure_is_not_repeated_as_its_own_first_cause() {
        let error = ToolError::from(EnvironmentError::with_source(
            EnvironmentErrorKind::Io,
            "Failed to write /work/out.txt",
            Cause,
        ));

        assert_eq!(
            error.message(),
            "Failed to write /work/out.txt\n  caused by: disk is full"
        );
        assert_eq!(
            StdError::source(&error).map(ToString::to_string),
            Some("disk is full".to_owned())
        );
        assert_eq!(
            error.detail(),
            "Failed to write /work/out.txt\n  caused by: disk is full\n  caused by: disk is full"
        );
    }

    #[test]
    fn an_environment_failure_without_a_cause_has_no_source() {
        let error = ToolError::from(EnvironmentError::new(
            EnvironmentErrorKind::NotFound,
            "File not found: /work/a.txt",
        ));

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert_eq!(error.message(), "File not found: /work/a.txt");
        assert_eq!(error.detail(), "File not found: /work/a.txt");
        assert!(StdError::source(&error).is_none());
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
