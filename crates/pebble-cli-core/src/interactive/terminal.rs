//! A small live area on the normal terminal screen. History is append-only.

use std::io::{self, Write as _};
use std::ops::Range;
use std::panic;

use crossterm::cursor::{self, Hide, MoveTo, Show};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{self, BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate};
use crossterm::{execute, queue};
use termimad::MadSkin;

use super::editor::Editor;
use super::images::{Image, Protocol};
use super::{highlight, text};

pub(super) struct Terminal {
    output:        io::Stdout,
    width:         u16,
    height:        u16,
    anchor:        u16,
    cursor_row:    u16,
    active:        bool,
    enhanced_keys: bool,
    skin:          MadSkin,
    color:         bool,
}

impl Terminal {
    pub(super) fn open(color: bool) -> io::Result<Self> {
        let (width, height) = terminal::size()?;
        let mut output = io::stdout();
        terminal::enable_raw_mode()?;
        let (column, mut row) = match cursor::position() {
            Ok(position) => position,
            Err(error) => {
                terminal::disable_raw_mode()?;
                return Err(error);
            }
        };
        if column > 0 {
            if let Err(error) = write!(output, "\r\n") {
                let _ = terminal::disable_raw_mode();
                return Err(error);
            }
            row = row.saturating_add(1).min(height.saturating_sub(1));
        }
        let enhanced_keys = terminal::supports_keyboard_enhancement().unwrap_or(false);
        let mut terminal = Self {
            output,
            width,
            height,
            anchor: row,
            cursor_row: row,
            active: true,
            enhanced_keys,
            color,
            skin: if color {
                MadSkin::default()
            } else {
                MadSkin::no_style()
            },
        };
        terminal.enable_input()?;
        Ok(terminal)
    }

    pub(super) fn install_panic_hook() {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let _ = terminal::disable_raw_mode();
            let _ = execute!(
                io::stdout(),
                EndSynchronizedUpdate,
                DisableBracketedPaste,
                PopKeyboardEnhancementFlags,
                ResetColor,
                SetAttribute(Attribute::Reset),
                Show
            );
            previous(info);
        }));
    }

    pub(super) fn width(&self) -> usize {
        usize::from(self.width.saturating_sub(2).max(1))
    }

    pub(super) fn resize(&mut self, width: u16, height: u16) {
        // Track the live area when the terminal removes rows from the top.
        let removed = self
            .cursor_row
            .saturating_add(1)
            .saturating_sub(height.max(1));
        self.anchor = self.anchor.saturating_sub(removed);
        self.width = width.max(1);
        self.height = height.max(1);
        self.anchor = self.anchor.min(self.height - 1);
        self.cursor_row = self.cursor_row.saturating_sub(removed).min(self.height - 1);
    }

    pub(super) fn markdown(&mut self, markdown: &str) -> io::Result<()> {
        let safe = text::plain(markdown);
        let mut prose = String::new();
        let mut fence: Option<(char, usize, String, String)> = None;
        for line in safe.split_inclusive('\n') {
            let trimmed = line.trim_start();
            let marker = trimmed.chars().next().filter(|c| matches!(c, '`' | '~'));
            let count = marker.map_or(0, |marker| {
                trimmed.chars().take_while(|c| *c == marker).count()
            });
            if let Some((opening, length, language, source)) = &mut fence {
                if marker == Some(*opening)
                    && count >= *length
                    && trimmed[count..].trim().is_empty()
                {
                    self.code(source.trim_end_matches('\n'), language)?;
                    fence = None;
                } else {
                    source.push_str(line);
                }
            } else if count >= 3 && line.len() - trimmed.len() <= 3 {
                self.prose(&prose)?;
                prose.clear();
                fence = Some((
                    marker.expect("a fence has a marker"),
                    count,
                    trimmed[count..]
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .into(),
                    String::new(),
                ));
            } else {
                prose.push_str(line);
            }
        }
        if let Some((_, _, language, source)) = fence {
            self.code(&source, &language)?;
        }
        self.prose(&prose)
    }

    fn prose(&mut self, safe: &str) -> io::Result<()> {
        if safe.is_empty() {
            return Ok(());
        }
        self.refresh_size()?;
        let width = self.width();
        let rendered = if width >= 3 {
            self.skin.text(safe, Some(width)).to_string()
        } else {
            safe.into()
        };
        self.append_rows(rendered.lines())
    }

    pub(super) fn code(&mut self, source: &str, language: &str) -> io::Result<()> {
        self.refresh_size()?;
        let rows = highlight::code(
            source,
            language,
            self.width().saturating_sub(2).max(1),
            self.color,
        )
        .into_iter()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>();
        self.append_rows(rows.iter().map(String::as_str))
    }

    pub(super) fn image(&mut self, image: &Image) -> io::Result<()> {
        self.refresh_size()?;
        let (columns, rows) =
            image.cells(self.width.saturating_sub(2), self.height.saturating_sub(4));
        let encoded = if self.width >= 4 && self.height >= 6 {
            Protocol::detect().and_then(|protocol| image.encode(protocol, columns, rows))
        } else {
            None
        };
        self.message(&format!(
            "Image · {} × {} · {}{}",
            image.width,
            image.height,
            image.mime,
            if encoded.is_none() {
                " · preview unavailable in this terminal"
            } else {
                ""
            }
        ))?;
        if let Some(encoded) = encoded {
            // Reserve real scrollback rows before placing graphics. Subsequent
            // prompt redraws start below these rows and do not erase the image.
            self.append_rows((0..rows).map(|_| ""))?;
            queue!(
                self.output,
                BeginSynchronizedUpdate,
                Hide,
                MoveTo(0, self.anchor.saturating_sub(rows))
            )?;
            self.output.write_all(encoded.as_bytes())?;
            queue!(
                self.output,
                MoveTo(0, self.anchor),
                Show,
                EndSynchronizedUpdate
            )?;
            self.output.flush()?;
        }
        Ok(())
    }

    pub(super) fn message(&mut self, message: &str) -> io::Result<()> {
        self.refresh_size()?;
        let rows = text::wrap(&text::plain(message), self.width());
        self.append_rows(rows.iter().map(String::as_str))
    }

    fn append_rows<'a>(&mut self, rows: impl IntoIterator<Item = &'a str>) -> io::Result<()> {
        queue!(
            self.output,
            BeginSynchronizedUpdate,
            Hide,
            MoveTo(0, self.anchor),
            Clear(ClearType::FromCursorDown)
        )?;
        for row in rows {
            write!(self.output, "{row}\r\n")?;
            self.anchor = self
                .anchor
                .saturating_add(1)
                .min(self.height.saturating_sub(1));
        }
        queue!(
            self.output,
            ResetColor,
            SetAttribute(Attribute::Reset),
            Show,
            EndSynchronizedUpdate
        )?;
        self.cursor_row = self.anchor;
        self.output.flush()
    }

    pub(super) fn draw(
        &mut self,
        editor: &Editor,
        status: &str,
        activity: &[String],
        menu: &[String],
    ) -> io::Result<()> {
        self.refresh_size()?;
        let available = usize::from(self.height);
        let max_editor = (available / 3).clamp(1, 8);
        let editor_width = self.width().saturating_sub(2).max(1);
        let (draft, cursor_row, cursor_column) = editor.layout(editor_width, max_editor);
        let mut lines = Vec::new();
        let extra_budget = available.saturating_sub(draft.len() + 1);
        let extras = if menu.is_empty() { activity } else { menu };
        lines.extend(
            extras
                .iter()
                .rev()
                .take(extra_budget.min(8))
                .rev()
                .map(|row| text::truncate(row, self.width())),
        );
        let first_editor = lines.len();
        for (index, row) in draft.iter().enumerate() {
            let prefix = if index == 0 { "› " } else { "  " };
            lines.push(format!("{prefix}{row}"));
        }
        if lines.len() < available {
            lines.push(text::truncate(status, self.width()));
        }
        lines.truncate(available);
        self.paint(
            &lines,
            first_editor..first_editor + draft.len(),
            first_editor + cursor_row,
            cursor_column + 2,
        )
    }

    pub(super) fn picker_capacity(&mut self) -> io::Result<usize> {
        self.refresh_size()?;
        Ok(usize::from(self.height).saturating_sub(5).clamp(1, 10))
    }

    pub(super) fn picker(
        &mut self,
        search: &Editor,
        title: &str,
        status: &str,
        choices: &[String],
    ) -> io::Result<()> {
        self.refresh_size()?;
        let (draft, _, column) = search.layout(self.width().saturating_sub(8).max(1), 1);
        let mut lines = vec![
            text::truncate(title, self.width()),
            format!("Search: {}", draft.first().map_or("", String::as_str)),
        ];
        lines.extend(
            choices
                .iter()
                .map(|line| text::truncate(line, self.width())),
        );
        lines.push(text::truncate(status, self.width()));
        lines.truncate(usize::from(self.height));
        self.paint(&lines, 1..2, 1, column + 8)
    }

    fn paint(
        &mut self,
        lines: &[String],
        focus: Range<usize>,
        cursor_row: usize,
        cursor_column: usize,
    ) -> io::Result<()> {
        let height = u16::try_from(lines.len()).unwrap_or(self.height);
        let scroll = self
            .anchor
            .saturating_add(height)
            .saturating_sub(self.height);
        queue!(self.output, BeginSynchronizedUpdate, Hide)?;
        if scroll > 0 {
            queue!(self.output, MoveTo(0, self.height - 1))?;
            for _ in 0..scroll {
                write!(self.output, "\r\n")?;
            }
            self.anchor = self.anchor.saturating_sub(scroll);
        }
        queue!(
            self.output,
            MoveTo(0, self.anchor),
            Clear(ClearType::FromCursorDown)
        )?;
        for (offset, line) in lines.iter().enumerate() {
            let row = self.anchor + u16::try_from(offset).unwrap_or(0);
            queue!(self.output, MoveTo(0, row))?;
            if self.color && (!focus.contains(&offset) && !line.starts_with('›')) {
                queue!(self.output, SetAttribute(Attribute::Dim))?;
            } else if self.color {
                queue!(self.output, SetForegroundColor(Color::Cyan))?;
            }
            write!(self.output, "{line}")?;
            queue!(self.output, ResetColor, SetAttribute(Attribute::Reset))?;
        }
        let row = self
            .anchor
            .saturating_add(u16::try_from(cursor_row).unwrap_or(0))
            .min(self.height - 1);
        let column = u16::try_from(cursor_column)
            .unwrap_or(0)
            .min(self.width.saturating_sub(1));
        self.cursor_row = row;
        queue!(
            self.output,
            MoveTo(column, row),
            Show,
            EndSynchronizedUpdate
        )?;
        self.output.flush()
    }

    pub(super) fn pause(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        execute!(
            self.output,
            MoveTo(0, self.anchor),
            Clear(ClearType::FromCursorDown),
            EndSynchronizedUpdate,
            DisableBracketedPaste,
            ResetColor,
            SetAttribute(Attribute::Reset),
            Show
        )?;
        if self.enhanced_keys {
            execute!(self.output, PopKeyboardEnhancementFlags)?;
        }
        terminal::disable_raw_mode()?;
        self.active = false;
        Ok(())
    }

    // A resize signal can arrive during setup or an external-editor handoff.
    // Read the current dimensions before painting, even if its event is late.
    fn refresh_size(&mut self) -> io::Result<()> {
        let (width, height) = terminal::size()?;
        if width != self.width || height != self.height {
            self.resize(width, height);
        }
        Ok(())
    }

    /// Call before restarting the input reader, which shares terminal input.
    pub(super) fn resume(&mut self) -> io::Result<()> {
        terminal::enable_raw_mode()?;
        self.active = true;
        let (width, height) = terminal::size()?;
        self.width = width.max(1);
        self.height = height.max(1);
        let (column, row) = cursor::position()?;
        self.anchor = row.min(self.height - 1);
        self.cursor_row = self.anchor;
        if column > 0 {
            write!(self.output, "\r\n")?;
            self.anchor = self.anchor.saturating_add(1).min(self.height - 1);
        }
        self.enable_input()
    }

    fn enable_input(&mut self) -> io::Result<()> {
        execute!(self.output, EnableBracketedPaste)?;
        if self.enhanced_keys {
            execute!(
                self.output,
                PushKeyboardEnhancementFlags(
                    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                        | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
                )
            )?;
        }
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        if self.pause().is_err() {
            let _ = terminal::disable_raw_mode();
            let _ = execute!(
                self.output,
                EndSynchronizedUpdate,
                DisableBracketedPaste,
                ResetColor,
                Show
            );
        }
    }
}
