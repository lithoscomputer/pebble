//! Char-boundary arithmetic for byte budgets.
//!
//! `str::floor_char_boundary` and `str::ceil_char_boundary` stabilized after
//! pebble's minimum supported Rust version, so the crate carries equivalents.
//! Both are used wherever a byte budget cuts text that may hold multi-byte
//! characters: output truncation, retained process output, and event tails.

/// The largest char boundary at or below `index`.
///
/// Returns `text.len()` when `index` is at or past the end, so a budget larger
/// than the text keeps all of it.
pub(crate) fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }

    let mut boundary = index;
    while !text.is_char_boundary(boundary) {
        // Byte 0 is always a boundary, so this terminates.
        boundary -= 1;
    }
    boundary
}

/// The smallest char boundary at or above `index`.
///
/// Returns `text.len()` when `index` is at or past the end. Unlike the
/// standard library's unstable equivalent, an out-of-range index saturates
/// instead of panicking.
pub(crate) fn ceil_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }

    let mut boundary = index;
    while !text.is_char_boundary(boundary) {
        // The end of the text is always a boundary, so this terminates.
        boundary += 1;
    }
    boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_indexes_are_already_boundaries() {
        let text = "hello";
        for index in 0..=text.len() {
            assert_eq!(floor_char_boundary(text, index), index);
            assert_eq!(ceil_char_boundary(text, index), index);
        }
    }

    #[test]
    fn interior_bytes_of_a_character_round_outward() {
        // "😀" is four bytes, so 1, 2 and 3 sit inside it.
        let text = "a😀z";
        assert_eq!(floor_char_boundary(text, 2), 1);
        assert_eq!(ceil_char_boundary(text, 2), 5);
        assert_eq!(floor_char_boundary(text, 4), 1);
        assert_eq!(ceil_char_boundary(text, 4), 5);
    }

    #[test]
    fn boundaries_are_returned_unchanged() {
        let text = "a😀z";
        for boundary in [0, 1, 5, 6] {
            assert_eq!(floor_char_boundary(text, boundary), boundary);
            assert_eq!(ceil_char_boundary(text, boundary), boundary);
        }
    }

    #[test]
    fn an_index_past_the_end_saturates_to_the_length() {
        let text = "a😀z";
        assert_eq!(floor_char_boundary(text, 99), text.len());
        assert_eq!(ceil_char_boundary(text, 99), text.len());
    }

    #[test]
    fn empty_text_has_only_the_zero_boundary() {
        assert_eq!(floor_char_boundary("", 0), 0);
        assert_eq!(ceil_char_boundary("", 3), 0);
    }

    #[test]
    fn every_result_is_a_boundary_of_the_text() {
        let text = "aé😀漢z";
        for index in 0..=text.len() {
            let floor = floor_char_boundary(text, index);
            let ceil = ceil_char_boundary(text, index);
            assert!(text.is_char_boundary(floor), "floor {floor} of {index}");
            assert!(text.is_char_boundary(ceil), "ceil {ceil} of {index}");
            assert!(floor <= index.min(text.len()));
            assert!(ceil >= index.min(text.len()));
        }
    }
}
