//! A sandbox failure rendered for a log or an error response: the cause
//! chain, and the redacted tail of the output a failed command or git
//! operation left behind.

use std::error::Error as StdError;
use std::fmt::Write as _;

use sandbox_driver::Error as DriverError;

use super::exec::redacted_output_tail;
use crate::environment::DEFAULT_EXEC_OUTPUT_TAIL_BYTES;
use crate::redact::Redactor;
use crate::types::ExecOutputTail;

/// The output tail of the first sandbox-driver failure in `err`'s cause
/// chain that carries command output (a command that ran and failed, or a
/// git operation whose command output the driver kept as evidence), passed
/// through `redactor` and capped to [`DEFAULT_EXEC_OUTPUT_TAIL_BYTES`] per
/// stream.
#[must_use]
pub fn default_redacted_output_tail(
    err: &(dyn StdError + 'static),
    redactor: &dyn Redactor,
) -> Option<ExecOutputTail> {
    let mut current = Some(err);
    while let Some(err) = current {
        if let Some(tail) = err
            .downcast_ref::<DriverError>()
            .and_then(|driver| driver_output_tail(driver, redactor))
        {
            return Some(tail);
        }
        current = err.source();
    }
    None
}

fn driver_output_tail(error: &DriverError, redactor: &dyn Redactor) -> Option<ExecOutputTail> {
    let failure = match error {
        DriverError::Exec(failure) => failure,
        DriverError::Git(git) => git.output()?,
        _ => return None,
    };
    redacted_output_tail(
        &String::from_utf8_lossy(failure.stdout()),
        &String::from_utf8_lossy(failure.stderr()),
        DEFAULT_EXEC_OUTPUT_TAIL_BYTES,
        redactor,
    )
}

/// `err` with its causes, followed by the redacted output tail when a
/// driver failure in the chain carries one.
#[must_use]
pub fn display_for_log(err: &(dyn StdError + 'static), redactor: &dyn Redactor) -> String {
    let mut rendered = render_with_causes(err);
    if let Some(tail) = default_redacted_output_tail(err, redactor) {
        append_tail_for_log(
            &mut rendered,
            "stderr",
            tail.stderr.as_deref(),
            tail.stderr_truncated,
        );
        append_tail_for_log(
            &mut rendered,
            "stdout",
            tail.stdout.as_deref(),
            tail.stdout_truncated,
        );
    }
    rendered
}

/// `err`'s message, then each cause on its own `caused by:` line.
fn render_with_causes(err: &(dyn StdError + 'static)) -> String {
    let mut rendered = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        rendered.push_str("\n  caused by: ");
        rendered.push_str(&cause.to_string());
        source = cause.source();
    }
    rendered
}

fn append_tail_for_log(rendered: &mut String, stream: &str, tail: Option<&str>, truncated: bool) {
    let tail = tail.unwrap_or("");
    let _ = write!(
        rendered,
        "\n--- {stream} (truncated={truncated}, bytes={}) ---\n{tail}",
        tail.len()
    );
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::time::Duration;

    use sandbox_driver::{ExecFailure, Termination};

    use super::*;
    use crate::redact::NoRedaction;
    use crate::sandbox_driver::testing::MaskToken;

    const SECRET: &str = "ghs_xK9mZ2vL8nQ5rT1wY4bC7dF0gH3jE6pA";

    fn failed_push(stdout: &str, stderr: &str) -> DriverError {
        DriverError::from(
            ExecFailure::new(
                "git push origin refs/heads/run",
                Termination::Exited,
                Some(128),
                stdout.as_bytes().to_vec(),
                stderr.as_bytes().to_vec(),
            )
            .with_duration(Duration::from_millis(210)),
        )
    }

    #[derive(Debug, thiserror::Error)]
    #[error("{message}")]
    struct Wrapped {
        message: String,
        #[source]
        source:  DriverError,
    }

    #[test]
    fn display_for_log_walks_the_chain_and_emits_the_tail() {
        let error = Wrapped {
            message: "metadata push failed".to_string(),
            source:  failed_push("last stdout line", "last stderr line"),
        };

        let rendered = display_for_log(&error, &NoRedaction);

        assert!(rendered.contains("metadata push failed"));
        assert!(rendered.contains("caused by: "));
        assert!(rendered.contains("git push origin refs/heads/run"));
        assert!(rendered.contains("--- stderr (truncated=false, bytes=16) ---"));
        assert!(rendered.contains("last stderr line"));
        assert!(rendered.contains("--- stdout (truncated=false, bytes=16) ---"));
        assert!(rendered.contains("last stdout line"));
    }

    #[test]
    fn display_for_log_redacts_through_the_applications_redactor() {
        let error = failed_push(
            &format!("stdout secret {SECRET}"),
            &format!("stderr secret {SECRET}"),
        );

        let rendered = display_for_log(&error, &MaskToken(SECRET));

        assert!(
            !rendered.contains(SECRET),
            "log rendering leaked raw secret: {rendered}"
        );
        assert!(rendered.contains("REDACTED"));
    }

    #[test]
    fn display_for_log_for_a_plain_error_is_the_chain_alone() {
        let error = DriverError::io("reading the file", io::Error::other("leaf failure"));

        let rendered = display_for_log(&error, &NoRedaction);

        assert!(rendered.contains("leaf failure"), "{rendered}");
        assert!(!rendered.contains("--- stderr"));
        assert!(!rendered.contains("--- stdout"));
        assert!(default_redacted_output_tail(&error, &NoRedaction).is_none());
    }
}
