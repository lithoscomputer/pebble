//! Safe terminal text and display-column calculations.

use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::UnicodeWidthStr as _;

/// Removes terminal commands, including OSC clipboard and title sequences.
pub(super) fn plain(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '\u{1b}' => match chars.next() {
                Some('[') => {
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                Some(']' | 'P' | '_' | '^') => {
                    while let Some(next) = chars.next() {
                        if next == '\u{7}' {
                            break;
                        }
                        if next == '\u{1b}' && chars.peek() == Some(&'\\') {
                            chars.next();
                            break;
                        }
                    }
                }
                Some(_) | None => {}
            },
            '\r' if chars.peek() != Some(&'\n') => {
                output.push('\n');
            }
            '\n' => output.push('\n'),
            '\t' => output.push_str("    "),
            value if !value.is_control() => output.push(value),
            _ => {}
        }
    }
    output
}

pub(super) fn truncate(text: &str, width: usize) -> String {
    let text = plain(text).replace('\n', " ");
    if text.width() <= width {
        return text;
    }
    let mut result = String::new();
    let mut used = 0;
    for grapheme in text.graphemes(true) {
        let length = grapheme.width();
        if used + length + 1 > width {
            break;
        }
        result.push_str(grapheme);
        used += length;
    }
    if width > 0 {
        result.push('…');
    }
    result
}

/// Wraps plain text without splitting a grapheme or using the last terminal
/// column.
pub(super) fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows = vec![String::new()];
    let mut used = 0;
    for grapheme in text.graphemes(true) {
        if grapheme == "\n" || grapheme == "\r\n" {
            rows.push(String::new());
            used = 0;
            continue;
        }
        let size = grapheme.width();
        if used > 0 && used + size > width {
            rows.push(String::new());
            used = 0;
        }
        if size > width {
            continue;
        }
        if let Some(row) = rows.last_mut() {
            row.push_str(grapheme);
        }
        used += size;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_cannot_clear_the_screen_or_set_the_clipboard() {
        assert_eq!(
            plain("a\x1b[2Jb\x1b]52;c;secret\x07c\x1bPbad\x1b\\d"),
            "abcd"
        );
        assert_eq!(plain("a\r\nb\rc\t\0"), "a\nb\nc    ");
    }

    #[test]
    fn wraps_display_columns_and_preserves_combining_characters() {
        assert_eq!(wrap("a界e\u{301}x", 3), vec!["a界", "e\u{301}x"]);
        assert_eq!(truncate("界界界", 4), "界…");
    }
}
