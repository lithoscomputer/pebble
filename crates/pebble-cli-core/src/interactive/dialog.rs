//! Keyboard interaction for questions and explicit tool approvals.

use std::collections::BTreeSet;
use std::mem;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use pebble_coding_agent::extensions::{Answer, AnswerStatus, QuestionKind};

use super::editor::Editor;
use super::services::Request;

pub(super) struct Dialog {
    request:    Option<Request>,
    pub editor: Editor,
    index:      usize,
    selected:   usize,
    checked:    BTreeSet<usize>,
    answers:    Vec<Answer>,
}

impl Dialog {
    pub(super) fn new(request: Request) -> Self {
        Self {
            request:  Some(request),
            editor:   Editor::default(),
            index:    0,
            selected: 0,
            checked:  BTreeSet::new(),
            answers:  Vec::new(),
        }
    }

    pub(super) fn cancelled(&self) -> bool {
        match &self.request {
            Some(Request::Questions { cancel, .. } | Request::Approval { cancel, .. }) => {
                cancel.is_cancelled()
            }
            None => true,
        }
    }

    pub(super) fn description(&self) -> String {
        match &self.request {
            Some(Request::Approval { details, .. }) => {
                format!("\nApproval requested for this call:\n{details}")
            }
            Some(Request::Questions { questions, .. }) => questions
                .iter()
                .map(|question| {
                    let options = question
                        .options
                        .iter()
                        .map(|option| {
                            format!(
                                "  {} — {}{}",
                                option.key,
                                option.label,
                                option
                                    .description
                                    .as_ref()
                                    .map_or_else(String::new, |description| format!(
                                        ": {description}"
                                    ))
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("\n{}\n{options}", question.text)
                })
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        }
    }

    pub(super) fn lines(&self) -> Vec<String> {
        match &self.request {
            Some(Request::Approval { .. }) => vec![
                "Approve this tool call?".into(),
                format!(
                    "{} Deny    {} Allow once",
                    if self.selected == 0 { "›" } else { " " },
                    if self.selected == 1 { "›" } else { " " }
                ),
                "←/→ choose · Enter confirms · y allows · Esc denies".into(),
            ],
            Some(Request::Questions { questions, .. }) => {
                let Some(question) = questions.get(self.index) else {
                    return Vec::new();
                };
                let mut lines = vec![format!(
                    "Question {}/{}: {}",
                    self.index + 1,
                    questions.len(),
                    question.text
                )];
                let start = self.selected.saturating_sub(3);
                for (index, option) in question.options.iter().enumerate().skip(start).take(5) {
                    lines.push(format!(
                        "{} {}{}",
                        if index == self.selected { "›" } else { " " },
                        if self.checked.contains(&index) {
                            "[x] "
                        } else {
                            ""
                        },
                        option.label
                    ));
                }
                lines.push(
                    if question.allow_freeform {
                        "↑/↓ choose · type another answer · Enter submits · Esc skips"
                    } else {
                        "↑/↓ choose · Space toggles multiple choices · Enter submits · Esc skips"
                    }
                    .into(),
                );
                lines
            }
            None => Vec::new(),
        }
    }

    pub(super) fn key(&mut self, key: KeyEvent) -> bool {
        if self.cancelled() {
            self.request.take();
            return true;
        }
        if matches!(self.request, Some(Request::Approval { .. })) {
            match key.code {
                KeyCode::Left | KeyCode::Up => self.selected = 0,
                KeyCode::Right | KeyCode::Down => self.selected = 1,
                KeyCode::Enter | KeyCode::Esc | KeyCode::Char('y' | 'n') => {
                    let approved = key.code == KeyCode::Char('y')
                        || (key.code == KeyCode::Enter && self.selected == 1);
                    if let Some(Request::Approval { reply, .. }) = self.request.take() {
                        let _ = reply.send(approved);
                    }
                    return true;
                }
                _ => {}
            }
            return false;
        }
        let Some(Request::Questions { questions, .. }) = &self.request else {
            return true;
        };
        let Some(question) = questions.get(self.index) else {
            return true;
        };
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(question.options.len().saturating_sub(1));
            }
            KeyCode::Char(' ')
                if question.kind == QuestionKind::MultiSelect && self.editor.text().is_empty() =>
            {
                match self.checked.take(&self.selected) {
                    Some(_) => {}
                    None => {
                        self.checked.insert(self.selected);
                    }
                }
            }
            KeyCode::Enter => {
                let values = if question.allow_freeform && !self.editor.text().trim().is_empty() {
                    vec![self.editor.take()]
                } else if question.kind == QuestionKind::MultiSelect && !self.checked.is_empty() {
                    self.checked
                        .iter()
                        .filter_map(|index| {
                            question
                                .options
                                .get(*index)
                                .map(|option| option.key.clone())
                        })
                        .collect()
                } else if let Some(option) = question.options.get(self.selected) {
                    vec![option.key.clone()]
                } else {
                    return false;
                };
                self.answers.push(Answer::answered(question, values));
                self.index += 1;
                self.selected = 0;
                self.checked.clear();
                self.editor.clear();
            }
            KeyCode::Esc => {
                self.answers.extend(
                    questions[self.index..]
                        .iter()
                        .map(|question| Answer::unanswered(question, AnswerStatus::Skipped)),
                );
                self.index = questions.len();
            }
            _ if question.allow_freeform && !key.modifiers.contains(KeyModifiers::ALT) => {
                self.editor.handle(key);
            }
            _ => {}
        }
        if self.index >= questions.len() {
            if let Some(Request::Questions { reply, .. }) = self.request.take() {
                let _ = reply.send(mem::take(&mut self.answers));
            }
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::oneshot;
    use tokio_util::sync::CancellationToken;

    use super::*;

    #[test]
    fn approving_requires_an_explicit_choice_and_cancellation_never_approves() {
        let (reply, mut answer) = oneshot::channel();
        let mut dialog = Dialog::new(Request::Approval {
            details: "command".into(),
            reply,
            cancel: CancellationToken::new(),
        });
        assert!(dialog.key(KeyCode::Enter.into()));
        assert_eq!(answer.try_recv(), Ok(false));
        let (reply, mut answer) = oneshot::channel();
        let cancel = CancellationToken::new();
        let mut dialog = Dialog::new(Request::Approval {
            details: "command".into(),
            reply,
            cancel: cancel.clone(),
        });
        cancel.cancel();
        assert!(dialog.key(KeyCode::Char('y').into()));
        assert!(answer.try_recv().is_err());
    }
}
