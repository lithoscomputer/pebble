//! The seam an application uses to strip secrets from text pebble records.
//!
//! Pebble ships no secret detector. It marks the places where text a process
//! or the operating system wrote leaves the session — the
//! [`ExecOutputTail`](crate::events::ExecOutputTail) a shell tool puts on the
//! event stream, and the model-facing message of every failed tool call,
//! which can carry an OS error naming a path — and calls the [`Redactor`] an
//! application installed. Without one, [`NoRedaction`] passes text through
//! unchanged.

use std::borrow::Cow;

/// Removes secrets from text on its way out of a session.
///
/// Implementations run on the session's critical path, so they must be
/// synchronous, cheap, and free of blocking work. They are called with
/// arbitrary process output and must never panic on it.
///
/// Returning [`Cow::Borrowed`] when nothing matched keeps the common case
/// allocation-free.
pub trait Redactor: Send + Sync {
    /// Returns `text` with every secret it recognizes replaced.
    fn redact<'a>(&self, text: &'a str) -> Cow<'a, str>;
}

/// A [`Redactor`] that changes nothing.
///
/// This is what a session uses when an application installs no redactor. It
/// is a deliberate choice, not a safe default: output reaches the event stream
/// exactly as the process produced it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoRedaction;

impl Redactor for NoRedaction {
    fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MaskDigits;

    impl Redactor for MaskDigits {
        fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
            if text.contains(|character: char| character.is_ascii_digit()) {
                Cow::Owned(
                    text.chars()
                        .map(|character| {
                            if character.is_ascii_digit() {
                                '#'
                            } else {
                                character
                            }
                        })
                        .collect(),
                )
            } else {
                Cow::Borrowed(text)
            }
        }
    }

    #[test]
    fn no_redaction_borrows_its_input() {
        let redactor = NoRedaction;
        let redacted = redactor.redact("token abc123");
        assert!(matches!(redacted, Cow::Borrowed("token abc123")));
    }

    #[test]
    fn a_redactor_is_usable_as_a_trait_object() {
        let redactor: &dyn Redactor = &MaskDigits;
        assert_eq!(redactor.redact("key 42"), "key ##");
        assert!(matches!(redactor.redact("no secrets"), Cow::Borrowed(_)));
    }
}
