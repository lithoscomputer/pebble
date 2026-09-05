//! Small lexical code styling. Unknown languages remain plain text.

use std::fmt::Write as _;

use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::UnicodeWidthStr as _;

use super::text;

const KEYWORDS: &str = "as async await break case catch class const continue def del do else elif enum except export extern false False finally fn for from function if impl import in interface let loop match mod move mut new None null package pass pub raise ref return self Self static struct super switch then this throw trait true True try type typeof unsafe use var void where while with yield";

pub(super) fn code(source: &str, language: &str, width: usize, color: bool) -> Vec<String> {
    let safe = text::plain(source);
    let language = language.to_ascii_lowercase();
    let known = matches!(
        language.as_str(),
        "rust"
            | "rs"
            | "javascript"
            | "js"
            | "typescript"
            | "ts"
            | "tsx"
            | "jsx"
            | "python"
            | "py"
            | "sh"
            | "bash"
            | "zsh"
            | "json"
            | "toml"
            | "yaml"
            | "yml"
            | "go"
            | "c"
            | "cpp"
            | "java"
            | "diff"
            | "patch"
    );
    let mut rows = Rows::new(width, color && known);
    let hash_comments = matches!(
        language.as_str(),
        "python" | "py" | "sh" | "bash" | "zsh" | "toml" | "yaml" | "yml"
    );
    let mut block_comment = false;
    for line in safe.split('\n') {
        if matches!(language.as_str(), "diff" | "patch") {
            let style = if line.starts_with('+') {
                32
            } else if line.starts_with('-') {
                31
            } else if line.starts_with("@@") || line.starts_with("***") {
                36
            } else {
                0
            };
            rows.push(line, style);
        } else {
            let mut rest = line;
            while !rest.is_empty() {
                if block_comment || rest.starts_with("/*") && !hash_comments {
                    let end = rest.find("*/").map(|offset| offset + 2);
                    block_comment = end.is_none();
                    let end = end.unwrap_or(rest.len());
                    rows.push(&rest[..end], 90);
                    rest = &rest[end..];
                } else if rest.starts_with("//") && !hash_comments
                    || rest.starts_with('#') && hash_comments
                {
                    rows.push(rest, 90);
                    break;
                } else if rest.starts_with(['"', '\'', '`']) {
                    let quote = rest.as_bytes()[0];
                    let mut escape = false;
                    let mut end = rest.len();
                    for (index, byte) in rest.bytes().enumerate().skip(1) {
                        if escape {
                            escape = false;
                        } else if byte == b'\\' {
                            escape = true;
                        } else if byte == quote {
                            end = index + 1;
                            break;
                        }
                    }
                    rows.push(&rest[..end], 32);
                    rest = &rest[end..];
                } else {
                    let first = rest.chars().next().expect("remaining code is not empty");
                    let end = if first.is_alphanumeric() || first == '_' {
                        rest.char_indices()
                            .find(|(_, c)| !c.is_alphanumeric() && *c != '_')
                            .map_or(rest.len(), |(index, _)| index)
                    } else {
                        first.len_utf8()
                    };
                    let token = &rest[..end];
                    let style = if first.is_ascii_digit() {
                        33
                    } else if KEYWORDS.split_whitespace().any(|keyword| token == keyword) {
                        35
                    } else {
                        0
                    };
                    rows.push(token, style);
                    rest = &rest[end..];
                }
            }
        }
        rows.newline();
    }
    rows.finish()
}

struct Rows {
    lines:  Vec<String>,
    column: usize,
    width:  usize,
    color:  bool,
    active: u8,
}

impl Rows {
    fn new(width: usize, color: bool) -> Self {
        Self {
            lines: vec![String::new()],
            column: 0,
            width: width.max(1),
            color,
            active: 0,
        }
    }
    fn newline(&mut self) {
        if self.active != 0 {
            self.lines
                .last_mut()
                .expect("there is always a row")
                .push_str("\x1b[0m");
            self.active = 0;
        }
        self.lines.push(String::new());
        self.column = 0;
    }
    fn push(&mut self, token: &str, style: u8) {
        for grapheme in token.graphemes(true) {
            let width = grapheme.width();
            if self.column + width > self.width {
                self.newline();
            }
            if width > self.width {
                continue;
            }
            let line = self
                .lines
                .last_mut()
                .expect("there is always a current row");
            if self.color && style != self.active {
                write!(line, "\x1b[{style}m").expect("writing a String cannot fail");
                self.active = style;
            }
            line.push_str(grapheme);
            self.column += width;
        }
    }
    fn finish(mut self) -> Vec<String> {
        if self.lines.last().is_some_and(String::is_empty) {
            self.lines.pop();
        }
        self.lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn highlighting_preserves_text_and_filters_terminal_controls() {
        let source = "fn main() { /* note */ let 界 = \"hi\"; }\x1b]52;c;secret\x07";
        let rendered = code(source, "rust", 24, true);
        assert!(rendered.iter().any(|line| line.contains("\x1b[35m")));
        assert!(rendered.iter().all(|line| text::plain(line).width() <= 24));
        assert!(!rendered.join("").contains("secret"));
        assert_eq!(text::plain(&rendered.join("")), text::plain(source));
        assert!(!code(source, "rust", 100, false).join("").contains('\x1b'));
        assert!(
            !code("unknown", "unknown", 100, true)
                .join("")
                .contains('\x1b')
        );
    }
}
