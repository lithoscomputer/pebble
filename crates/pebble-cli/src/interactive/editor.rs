//! Multiline prompt editing with grapheme-aware movement and undo.

use std::mem;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::UnicodeWidthStr as _;

use super::text;

const MAX_DRAFT_BYTES: usize = 1024 * 1024;
const MAX_UNDO: usize = 64;

#[derive(Default)]
pub(super) struct Editor {
    text:          String,
    cursor:        usize,
    undo:          Vec<(String, usize)>,
    history:       Vec<String>,
    history_index: Option<usize>,
    saved_draft:   String,
}

impl Editor {
    pub(super) fn text(&self) -> &str {
        &self.text
    }

    pub(super) fn set(&mut self, text: String) {
        self.remember();
        self.text = if text
            .chars()
            .any(|value| value.is_control() && value != '\n')
        {
            text::plain(&text)
        } else {
            text
        };
        self.cursor = self.text.len();
        self.history_index = None;
    }

    pub(super) fn restore(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.remember();
        if !self.text.is_empty() {
            self.text.push_str("\n\n");
        }
        self.text.push_str(text);
        self.cursor = self.text.len();
    }

    pub(super) fn insert(&mut self, text: &str) -> bool {
        let safe = text::plain(text);
        if self.text.len().saturating_add(safe.len()) > MAX_DRAFT_BYTES {
            return false;
        }
        self.remember();
        self.text.insert_str(self.cursor, &safe);
        self.cursor += safe.len();
        self.history_index = None;
        true
    }

    pub(super) fn replace_token(&mut self, start: usize, replacement: &str) {
        self.remember();
        self.text.replace_range(start..self.cursor, replacement);
        self.cursor = start + replacement.len();
    }

    pub(super) fn token(&self) -> (usize, &str) {
        let before = &self.text[..self.cursor];
        let start = before
            .char_indices()
            .rev()
            .find(|(_, value)| value.is_whitespace())
            .map_or(0, |(index, value)| index + value.len_utf8());
        (start, &before[start..])
    }

    pub(super) fn take(&mut self) -> String {
        let text = mem::take(&mut self.text);
        self.cursor = 0;
        self.undo.clear();
        self.history_index = None;
        self.add_history(text.clone());
        text
    }

    pub(super) fn add_history(&mut self, text: String) {
        if !text.trim().is_empty() && self.history.last() != Some(&text) {
            if self.history.len() == 200 {
                self.history.remove(0);
            }
            self.history.push(text);
        }
    }

    pub(super) fn clear(&mut self) {
        self.remember();
        self.text.clear();
        self.cursor = 0;
    }

    pub(super) fn handle(&mut self, key: KeyEvent) {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Char('z' | '_') if control => {
                if let Some((text, cursor)) = self.undo.pop() {
                    self.text = text;
                    self.cursor = cursor;
                }
            }
            KeyCode::Char('a') if control => self.cursor = self.line_start(),
            KeyCode::Char('e') if control => self.cursor = self.line_end(),
            KeyCode::Char('u') if control => self.delete(self.line_start(), self.cursor),
            KeyCode::Char('k') if control => self.delete(self.cursor, self.line_end()),
            KeyCode::Char('w') if control => {
                let start = self.word_left();
                self.delete(start, self.cursor);
            }
            KeyCode::Char(value) if !control && !alt => {
                self.insert(&value.to_string());
            }
            KeyCode::Left if control || alt => self.cursor = self.word_left(),
            KeyCode::Right if control || alt => {
                while self.cursor < self.text.len() {
                    let next = self.next();
                    let whitespace = self.text[self.cursor..next].trim().is_empty();
                    self.cursor = next;
                    if whitespace {
                        break;
                    }
                }
            }
            KeyCode::Left => self.cursor = self.previous(),
            KeyCode::Right => self.cursor = self.next(),
            KeyCode::Home => self.cursor = if control { 0 } else { self.line_start() },
            KeyCode::End => {
                self.cursor = if control {
                    self.text.len()
                } else {
                    self.line_end()
                }
            }
            KeyCode::Backspace => self.delete(self.previous(), self.cursor),
            KeyCode::Delete => self.delete(self.cursor, self.next()),
            KeyCode::Up => self.vertical(false),
            KeyCode::Down => self.vertical(true),
            _ => {}
        }
    }

    pub(super) fn layout(&self, width: usize, max_rows: usize) -> (Vec<String>, usize, usize) {
        let width = width.max(1);
        let mut rows = vec![String::new()];
        let mut used = 0;
        let mut cursor_row = 0;
        let mut cursor_column = 0;
        for (offset, grapheme) in self.text.grapheme_indices(true) {
            let size = grapheme.width();
            if grapheme != "\n" && used + size > width {
                rows.push(String::new());
                used = 0;
            }
            if offset == self.cursor {
                cursor_row = rows.len() - 1;
                cursor_column = used;
            }
            if grapheme == "\n" {
                rows.push(String::new());
                used = 0;
            } else if size <= width {
                if let Some(row) = rows.last_mut() {
                    row.push_str(grapheme);
                }
                used += size;
            }
        }
        if self.cursor == self.text.len() {
            if used >= width {
                rows.push(String::new());
                used = 0;
            }
            cursor_row = rows.len() - 1;
            cursor_column = used;
        }
        let first = cursor_row.saturating_sub(max_rows.saturating_sub(1));
        let visible = rows.into_iter().skip(first).take(max_rows.max(1)).collect();
        (visible, cursor_row - first, cursor_column)
    }

    fn remember(&mut self) {
        if self.undo.len() == MAX_UNDO {
            self.undo.remove(0);
        }
        self.undo.push((self.text.clone(), self.cursor));
    }

    fn previous(&self) -> usize {
        self.text[..self.cursor]
            .grapheme_indices(true)
            .next_back()
            .map_or(0, |(index, _)| index)
    }

    fn next(&self) -> usize {
        self.text[self.cursor..]
            .graphemes(true)
            .next()
            .map_or(self.cursor, |part| self.cursor + part.len())
    }

    fn line_start(&self) -> usize {
        self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |index| index + 1)
    }
    fn line_end(&self) -> usize {
        self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |index| self.cursor + index)
    }

    fn word_left(&self) -> usize {
        let mut start = self.cursor;
        let mut word = false;
        for (offset, grapheme) in self.text[..self.cursor].grapheme_indices(true).rev() {
            let space = grapheme.trim().is_empty();
            if word && space {
                break;
            }
            word |= !space;
            start = offset;
        }
        start
    }

    fn delete(&mut self, start: usize, end: usize) {
        if start == end {
            return;
        }
        self.remember();
        self.text.replace_range(start..end, "");
        self.cursor = start;
    }

    fn vertical(&mut self, down: bool) {
        let start = self.line_start();
        let end = self.line_end();
        let column = self.text[start..self.cursor].width();
        let target = if down && end < self.text.len() {
            Some(end + 1)
        } else if !down && start > 0 {
            Some(
                self.text[..start - 1]
                    .rfind('\n')
                    .map_or(0, |index| index + 1),
            )
        } else {
            None
        };
        if let Some(target) = target {
            self.cursor = target;
            let mut used = 0;
            for grapheme in self.text[target..].graphemes(true) {
                if grapheme == "\n" || used + grapheme.width() > column {
                    break;
                }
                used += grapheme.width();
                self.cursor += grapheme.len();
            }
        } else {
            self.recall(down);
        }
    }

    fn recall(&mut self, down: bool) {
        if self.history.is_empty() {
            return;
        }
        if self.history_index.is_none() {
            if down {
                return;
            }
            self.saved_draft.clone_from(&self.text);
        }
        let index = match self.history_index {
            Some(index) if down => index + 1,
            Some(index) => index.saturating_sub(1),
            None => self.history.len() - 1,
        };
        if index >= self.history.len() {
            self.text.clone_from(&self.saved_draft);
            self.history_index = None;
        } else {
            self.text.clone_from(&self.history[index]);
            self.history_index = Some(index);
        }
        self.cursor = self.text.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editing_and_undo_preserve_graphemes() {
        let mut editor = Editor::default();
        editor.insert("hi 👨‍👩‍👧‍👦e\u{301}");
        editor.handle(KeyCode::Backspace.into());
        assert_eq!(editor.text(), "hi 👨‍👩‍👧‍👦");
        editor.handle(KeyCode::Backspace.into());
        assert_eq!(editor.text(), "hi ");
        editor.handle(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL));
        assert_eq!(editor.text(), "hi 👨‍👩‍👧‍👦");
    }

    #[test]
    fn recalling_history_restores_the_unsent_draft() {
        let mut editor = Editor::default();
        editor.insert("first");
        editor.take();
        editor.insert("draft");
        editor.handle(KeyCode::Up.into());
        assert_eq!(editor.text(), "first");
        editor.handle(KeyCode::Down.into());
        assert_eq!(editor.text(), "draft");
        editor.restore("queued");
        assert_eq!(editor.text(), "draft\n\nqueued");
    }

    #[test]
    fn layout_keeps_the_cursor_visible_in_a_long_multiline_draft() {
        let mut editor = Editor::default();
        editor.insert("one\ntwo\n界界\nfour");
        assert_eq!(
            editor.layout(4, 2),
            (vec!["four".into(), String::new()], 1, 0)
        );
    }
}
