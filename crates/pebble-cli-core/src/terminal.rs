//! The terminal boundary every command writes through.

use std::io::{self, Write as _};

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

/// Writes one line to standard output and flushes it.
///
/// A write that fails is dropped: a closed pipe is the reader's choice, and
/// nothing can be said to them about it.
pub fn print_out(text: &str) {
    let mut stdout = io::stdout().lock();
    let _ = stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.write_all(b"\n"))
        .and_then(|()| stdout.flush());
}
