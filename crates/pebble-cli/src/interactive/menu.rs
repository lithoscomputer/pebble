//! Searchable pickers and completion lists with separate input focus.

use std::io;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::{Editor, Terminal};

pub(super) enum Purpose {
    Command,
    Completion(usize),
    Sessions,
    Navigate,
    Models,
    Favorites,
    Login,
    Logout,
    Tools,
    Agents,
    Thinking,
    Settings,
}

pub(super) struct Menu {
    pub purpose: Purpose,
    items:       Vec<(String, String)>,
    search:      Editor,
    selected:    usize,
}

pub(super) enum MenuAction {
    Editing,
    Unhandled,
    Close,
    Select(String),
    SaveDefault(String),
    ToggleFavorite(String),
}

impl Menu {
    pub(super) fn new(purpose: Purpose, items: Vec<(String, String)>) -> Self {
        Self {
            purpose,
            items,
            search: Editor::default(),
            selected: 0,
        }
    }

    pub(super) fn is_completion(&self) -> bool {
        matches!(self.purpose, Purpose::Command | Purpose::Completion(_))
    }

    pub(super) fn filter(&mut self, query: &str) {
        self.search.set(query.into());
        self.selected = 0;
    }

    pub(super) fn replace_items(&mut self, items: Vec<(String, String)>) {
        self.items = items;
        self.selected = self.selected.min(self.matches().len().saturating_sub(1));
    }

    fn matches(&self) -> Vec<&(String, String)> {
        let mut matches: Vec<_> = self
            .items
            .iter()
            .filter_map(|item| score(self.search.text(), &item.0).map(|rank| (rank, item)))
            .collect();
        matches.sort_by_key(|(rank, _)| *rank);
        matches.into_iter().map(|(_, item)| item).collect()
    }

    pub(super) fn lines(&self, max_visible: usize) -> Vec<String> {
        let matches = self.matches();
        let selected = self.selected.min(matches.len().saturating_sub(1));
        let start = selected
            .saturating_sub(max_visible / 2)
            .min(matches.len().saturating_sub(max_visible));
        let mut lines: Vec<_> = matches
            .iter()
            .enumerate()
            .skip(start)
            .take(max_visible)
            .map(|(index, (label, _))| {
                format!("{} {label}", if index == selected { "›" } else { " " })
            })
            .collect();
        if matches.is_empty() {
            lines.push("No matches".into());
        }
        lines.push(format!(
            "{} of {} · ↑/↓ choose · {} · Esc closes",
            if matches.is_empty() { 0 } else { selected + 1 },
            matches.len(),
            if self.is_completion() {
                "Tab completes"
            } else if matches!(self.purpose, Purpose::Favorites) {
                "Enter toggles and saves"
            } else {
                "Enter selects"
            }
        ));
        if matches!(self.purpose, Purpose::Models) {
            lines.push("Ctrl+S selects and saves the default".into());
        }
        lines
    }

    pub(super) fn draw(&self, terminal: &mut Terminal, status: &str) -> io::Result<()> {
        let title = match self.purpose {
            Purpose::Models => "Models",
            Purpose::Favorites => "Favorite models",
            Purpose::Navigate => "Branches and history",
            Purpose::Sessions => "Sessions",
            Purpose::Login => "Log in",
            Purpose::Logout => "Log out",
            Purpose::Thinking => "Thinking",
            Purpose::Settings => "Settings",
            Purpose::Tools => "Tools",
            Purpose::Agents => "Agents",
            Purpose::Command | Purpose::Completion(_) => "Complete",
        };
        let capacity = terminal.picker_capacity()?;
        terminal.picker(&self.search, title, status, &self.lines(capacity))
    }

    pub(super) fn paste(&mut self, value: &str) {
        self.search.insert(&value.replace(['\r', '\n'], " "));
        self.selected = 0;
    }

    pub(super) fn key(&mut self, key: KeyEvent) -> MenuAction {
        let count = self.matches().len();
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        if self.is_completion() && key.modifiers.contains(KeyModifiers::ALT) {
            return MenuAction::Unhandled;
        }
        match key.code {
            KeyCode::Esc => MenuAction::Close,
            KeyCode::Char('c' | 'q' | 'd') if control && !self.is_completion() => MenuAction::Close,
            KeyCode::Enter if self.is_completion() => MenuAction::Unhandled,
            KeyCode::Up | KeyCode::BackTab => {
                self.selected = if self.selected == 0 {
                    count.saturating_sub(1)
                } else {
                    self.selected - 1
                };
                MenuAction::Editing
            }
            KeyCode::Down => {
                self.selected = if count == 0 {
                    0
                } else {
                    (self.selected + 1) % count
                };
                MenuAction::Editing
            }
            KeyCode::PageUp => {
                self.selected = self.selected.saturating_sub(10);
                MenuAction::Editing
            }
            KeyCode::PageDown => {
                self.selected = (self.selected + 10).min(count.saturating_sub(1));
                MenuAction::Editing
            }
            KeyCode::Enter | KeyCode::Tab => {
                self.matches()
                    .get(self.selected)
                    .map_or(MenuAction::Editing, |(_, value)| {
                        if matches!(self.purpose, Purpose::Favorites) {
                            MenuAction::ToggleFavorite(value.clone())
                        } else {
                            MenuAction::Select(value.clone())
                        }
                    })
            }
            KeyCode::Char('s') if control && matches!(self.purpose, Purpose::Models) => self
                .matches()
                .get(self.selected)
                .map_or(MenuAction::Editing, |(_, value)| {
                    MenuAction::SaveDefault(value.clone())
                }),
            _ if self.is_completion() => MenuAction::Unhandled,
            _ => {
                self.search.handle(key);
                self.selected = 0;
                MenuAction::Editing
            }
        }
    }
}

/// Exact and contiguous matches precede scattered letters. Each search word
/// may match anywhere, so both "openai sol" and "sol openai" work.
pub(super) fn score(query: &str, candidate: &str) -> Option<usize> {
    let query = query.trim().to_lowercase();
    let candidate = candidate.to_lowercase();
    if query.is_empty() || query == candidate {
        return Some(0);
    }
    if candidate.starts_with(&query) {
        return Some(1);
    }
    let mut rank = 0;
    for word in query.split_whitespace() {
        if let Some(offset) = candidate.find(word) {
            rank += 10 + offset;
        } else {
            let mut remaining = candidate.as_str();
            let mut gaps = 0;
            for needle in word.chars() {
                let offset = remaining.find(needle)?;
                gaps += offset;
                remaining = &remaining[offset + needle.len_utf8()..];
            }
            rank += 100 + gaps;
        }
    }
    Some(rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searches_rank_matches_and_accept_words_in_either_order() {
        let mut menu = Menu::new(Purpose::Models, vec![
            ("scattered output label".into(), "scattered".into()),
            ("Sol · openai/sol".into(), "sol".into()),
        ]);
        menu.filter("sol");
        assert!(
            matches!(menu.key(KeyCode::Enter.into()), MenuAction::Select(value) if value == "sol")
        );
        menu.filter("openai sol");
        assert_eq!(menu.matches().len(), 1);
        assert!(menu.lines(5).iter().any(|line| line.starts_with("1 of 1")));
        menu.filter("missing");
        assert!(menu.lines(5).iter().any(|line| line == "No matches"));
    }

    #[test]
    fn search_edits_graphemes_and_navigation_wraps() {
        let mut menu = Menu::new(Purpose::Sessions, vec![
            ("one".into(), "1".into()),
            ("two".into(), "2".into()),
        ]);
        menu.paste("e\u{301}");
        menu.key(KeyCode::Backspace.into());
        assert!(menu.search.text().is_empty());
        menu.key(KeyCode::Up.into());
        assert!(
            matches!(menu.key(KeyCode::Enter.into()), MenuAction::Select(value) if value == "2")
        );
    }
}
