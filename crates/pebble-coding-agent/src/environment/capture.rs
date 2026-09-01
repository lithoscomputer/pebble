//! Bounded capture of a process's output.
//!
//! A command's output is always drained — a process that fills its pipe while
//! nobody reads blocks forever — but only a bounded slice is kept. The buffer
//! keeps a stable head and a rolling tail so both the start of the output and
//! its end survive, and counts what it dropped in between.

use std::collections::VecDeque;

use crate::event::OutputCaptureStats;

/// A byte buffer that keeps an equal-sized stable head and rolling tail.
///
/// Bytes past the cap are discarded from the middle: the head stops growing
/// once it fills half the cap, and the tail keeps the newest bytes.
#[derive(Debug)]
pub(crate) struct OutputCaptureBuffer {
    max_bytes:      Option<usize>,
    head:           Vec<u8>,
    tail:           VecDeque<u8>,
    observed_bytes: usize,
}

impl OutputCaptureBuffer {
    /// A buffer bounded to `max_bytes`, or unbounded when it is `None`.
    pub(crate) const fn new(max_bytes: Option<usize>) -> Self {
        Self {
            max_bytes,
            head: Vec::new(),
            tail: VecDeque::new(),
            observed_bytes: 0,
        }
    }

    /// Records `bytes`, keeping what fits.
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.observed_bytes = self.observed_bytes.saturating_add(bytes.len());
        let Some(max_bytes) = self.max_bytes else {
            self.head.extend_from_slice(bytes);
            return;
        };

        let head_budget = max_bytes / 2;
        let tail_budget = max_bytes.saturating_sub(head_budget);
        let head_remaining = head_budget.saturating_sub(self.head.len());
        let head_take = head_remaining.min(bytes.len());
        self.head.extend_from_slice(&bytes[..head_take]);

        let tail_bytes = &bytes[head_take..];
        let overflow = self
            .tail
            .len()
            .saturating_add(tail_bytes.len())
            .saturating_sub(tail_budget);
        if overflow >= self.tail.len() {
            let skip = overflow.saturating_sub(self.tail.len());
            self.tail.clear();
            self.tail.extend(&tail_bytes[skip..]);
        } else {
            self.tail.drain(..overflow);
            self.tail.extend(tail_bytes);
        }
    }

    /// What the buffer has seen and kept so far.
    pub(crate) fn stats(&self) -> OutputCaptureStats {
        let retained_bytes = self.head.len().saturating_add(self.tail.len());
        OutputCaptureStats {
            observed_bytes: self.observed_bytes,
            retained_bytes,
            omitted_bytes: self.observed_bytes.saturating_sub(retained_bytes),
        }
    }

    /// The retained bytes and their counts.
    pub(crate) fn into_parts(self) -> (Vec<u8>, OutputCaptureStats) {
        let stats = self.stats();
        let Self {
            head: mut bytes,
            tail,
            ..
        } = self;
        let (front, back) = tail.as_slices();
        bytes.extend_from_slice(front);
        bytes.extend_from_slice(back);
        (bytes, stats)
    }
}

/// Bounds one already-collected stream in place.
///
/// An environment that hands back whole output rather than draining a pipe
/// applies the caller's cap here, so its byte accounting matches an
/// environment that streamed. Inside pebble that is only the mock, so the
/// helper is compiled with it.
#[cfg(any(test, feature = "test-util"))]
pub(crate) fn capture_collected_stream(
    text: &mut String,
    cap: Option<usize>,
) -> OutputCaptureStats {
    match cap {
        Some(cap) if text.len() > cap => {
            let mut buffer = OutputCaptureBuffer::new(Some(cap));
            buffer.push(text.as_bytes());
            let (bytes, stats) = buffer.into_parts();
            *text = String::from_utf8_lossy(&bytes).into_owned();
            stats
        }
        _ => OutputCaptureStats::complete(text.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retained(buffer: OutputCaptureBuffer) -> (String, OutputCaptureStats) {
        let (bytes, stats) = buffer.into_parts();
        (
            String::from_utf8(bytes).expect("retained bytes are text"),
            stats,
        )
    }

    #[test]
    fn an_uncapped_buffer_keeps_everything() {
        let mut buffer = OutputCaptureBuffer::new(None);
        buffer.push(b"abcdefghij");

        let (output, stats) = retained(buffer);

        assert_eq!(output, "abcdefghij");
        assert_eq!(stats, OutputCaptureStats::complete(10));
    }

    #[test]
    fn a_capped_buffer_keeps_the_head_and_the_newest_tail() {
        let mut buffer = OutputCaptureBuffer::new(Some(8));
        buffer.push(b"abcdefghijklmnopqrst");

        let (output, stats) = retained(buffer);

        assert_eq!(output, "abcdqrst");
        assert_eq!(stats, OutputCaptureStats {
            observed_bytes: 20,
            retained_bytes: 8,
            omitted_bytes:  12,
        });
    }

    #[test]
    fn a_capped_buffer_rolls_its_tail_across_pushes() {
        let mut buffer = OutputCaptureBuffer::new(Some(8));
        for chunk in [&b"abcde"[..], b"fghij", b"klmno", b"pqrst"] {
            buffer.push(chunk);
        }

        let (output, stats) = retained(buffer);

        assert_eq!(output, "abcdqrst");
        assert_eq!(stats.observed_bytes, 20);
        assert_eq!(stats.omitted_bytes, 12);
    }

    #[test]
    fn output_within_the_cap_is_untouched() {
        let mut buffer = OutputCaptureBuffer::new(Some(64));
        buffer.push(b"short");

        let (output, stats) = retained(buffer);

        assert_eq!(output, "short");
        assert_eq!(stats, OutputCaptureStats::complete(5));
    }

    /// A caller that wants no output still needs the byte counts: the pipe was
    /// drained, and how much ran through it is what a consumer is told.
    #[test]
    fn a_zero_byte_cap_keeps_nothing_and_still_counts_what_was_drained() {
        let mut buffer = OutputCaptureBuffer::new(Some(0));
        buffer.push(b"abcdef");

        let (output, stats) = retained(buffer);

        assert!(output.is_empty());
        assert_eq!(stats, OutputCaptureStats {
            observed_bytes: 6,
            retained_bytes: 0,
            omitted_bytes:  6,
        });
    }

    #[test]
    fn a_collected_stream_under_a_zero_byte_cap_keeps_nothing() {
        let mut text = "abcdef".to_owned();

        let stats = capture_collected_stream(&mut text, Some(0));

        assert!(text.is_empty());
        assert_eq!(stats, OutputCaptureStats {
            observed_bytes: 6,
            retained_bytes: 0,
            omitted_bytes:  6,
        });
    }

    #[test]
    fn stats_are_readable_before_the_buffer_is_consumed() {
        let mut buffer = OutputCaptureBuffer::new(Some(4));
        buffer.push(b"abcdefgh");

        assert_eq!(buffer.stats(), OutputCaptureStats {
            observed_bytes: 8,
            retained_bytes: 4,
            omitted_bytes:  4,
        });
    }

    #[test]
    fn a_collected_stream_is_bounded_like_a_drained_one() {
        let mut text = "abcdefghijklmnopqrst".to_owned();

        let stats = capture_collected_stream(&mut text, Some(8));

        assert_eq!(text, "abcdqrst");
        assert_eq!(stats, OutputCaptureStats {
            observed_bytes: 20,
            retained_bytes: 8,
            omitted_bytes:  12,
        });
    }

    #[test]
    fn a_collected_stream_within_the_cap_is_left_alone() {
        let mut text = "abcd".to_owned();

        let stats = capture_collected_stream(&mut text, Some(8));

        assert_eq!(text, "abcd");
        assert_eq!(stats, OutputCaptureStats::complete(4));
    }

    #[test]
    fn a_collected_stream_without_a_cap_is_left_alone() {
        let mut text = "abcd".to_owned();

        let stats = capture_collected_stream(&mut text, None);

        assert_eq!(text, "abcd");
        assert_eq!(stats, OutputCaptureStats::complete(4));
    }
}
