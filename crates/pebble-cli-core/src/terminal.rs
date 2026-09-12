//! The standard error boundary every command writes through.

/// Writes one line to standard error.
#[expect(clippy::print_stderr, reason = "the command's stderr boundary")]
pub fn print_err(text: &str) {
    eprintln!("{text}");
}

/// Writes text to standard error with no line ending, for streamed output.
#[expect(clippy::print_stderr, reason = "the command's stderr boundary")]
pub fn print_err_fragment(text: &str) {
    eprint!("{text}");
}
