//! Event projection shared by live rendering and journal replay.

use std::collections::{BTreeMap, VecDeque};
use std::mem;

use lithos_llm::types::ContentPart;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};

use super::{text, tool_render};

const OUTPUT_PREVIEW_BYTES: usize = 32 * 1024;
const MAX_TOOL_RECORDS: usize = 200;

pub(super) enum Output {
    Text(String),
    Markdown(String),
    Code { source: String, language: String },
}

#[derive(Clone)]
pub(super) struct ToolRecord {
    pub id:        String,
    pub session:   String,
    pub name:      String,
    pub arguments: String,
    pub output:    String,
    pub complete:  bool,
    pub failed:    bool,
}

#[derive(Default)]
pub(super) struct Transcript {
    pub root:            String,
    pub cursor:          u64,
    pub last_answer:     String,
    pub tools:           VecDeque<ToolRecord>,
    pub agents:          BTreeMap<String, String>,
    pub input_tokens:    u64,
    pub output_tokens:   u64,
    pub cost:            u64,
    pub unknown_cost:    bool,
    pub context_percent: Option<f64>,
    pub show_reasoning:  bool,
    pub expand_tools:    bool,
    pending:             String,
    committed:           String,
    reasoning:           String,
    phase:               String,
}

impl Transcript {
    pub(super) fn new(root: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            ..Self::default()
        }
    }

    pub(super) fn apply(&mut self, envelope: &CodingAgentEvent, replay: bool) -> Vec<Output> {
        if envelope.seq <= self.cursor {
            return Vec::new();
        }
        self.cursor = envelope.seq;
        let root = envelope.session_id == self.root;
        let mut output = Vec::new();
        match &envelope.event {
            CodingEvent::UserInput { text, content, .. }
            | CodingEvent::SteeringInjected { text, content, .. }
                if root =>
            {
                let prompt = content.as_ref().map_or_else(
                    || text.clone(),
                    |content| {
                        content
                            .parts()
                            .iter()
                            .map(|part| {
                                if let ContentPart::Text { text } = part {
                                    text.clone()
                                } else {
                                    " [attachment] ".into()
                                }
                            })
                            .collect::<String>()
                    },
                );
                output.push(Output::Text(format!("\n› {prompt}\n")));
            }
            CodingEvent::LlmRequestStarted { .. } if root => {
                self.phase = "Thinking".into();
                self.pending.clear();
                self.committed.clear();
                self.reasoning.clear();
            }
            CodingEvent::TextDelta { delta } if root => {
                self.phase = "Writing".into();
                self.pending.push_str(delta);
                if let Some(end) = stable_boundary(&self.pending) {
                    let stable: String = self.pending.drain(..end).collect();
                    self.committed.push_str(&stable);
                    output.push(Output::Markdown(stable));
                } else if self.pending.len() > OUTPUT_PREVIEW_BYTES {
                    // A very long unfinished block remains readable without retaining an unbounded
                    // tail.
                    let stable = mem::take(&mut self.pending);
                    self.committed.push_str(&stable);
                    output.push(Output::Text(stable));
                }
            }
            CodingEvent::ReasoningDelta { delta } if root => {
                append_tail(&mut self.reasoning, delta);
            }
            CodingEvent::AssistantOutputReplace { text, reasoning } if root => {
                if !self.committed.is_empty() {
                    output.push(Output::Text(
                        "[Previous response attempt replaced]\n".into(),
                    ));
                }
                self.committed.clear();
                self.pending.clone_from(text);
                self.reasoning = reasoning.clone().unwrap_or_default();
            }
            CodingEvent::AssistantMessage {
                text,
                reasoning,
                usage,
                cost_usd_micros,
                context_window,
                ..
            } => {
                self.input_tokens = self.input_tokens.saturating_add(usage.input);
                self.output_tokens = self.output_tokens.saturating_add(usage.output);
                if let Some(cost) = cost_usd_micros {
                    self.cost = self.cost.saturating_add(*cost);
                } else {
                    self.unknown_cost = true;
                }
                if root {
                    self.context_percent =
                        context_window.as_ref().map(|window| window.usage_percent);
                    if replay {
                        self.reasoning = reasoning.as_ref().map_or_else(String::new, |reasoning| {
                            [reasoning.summary(), reasoning.trace()]
                                .into_iter()
                                .flatten()
                                .collect::<Vec<_>>()
                                .join("\n")
                        });
                    }
                    if self.show_reasoning && !self.reasoning.is_empty() {
                        output.push(Output::Text(format!("Thinking:\n{}\n", self.reasoning)));
                    }
                    let remaining = text.strip_prefix(&self.committed).unwrap_or(text);
                    if !remaining.trim().is_empty() {
                        output.push(Output::Markdown(remaining.to_owned()));
                    }
                    if !text.is_empty() {
                        self.last_answer.clone_from(text);
                    }
                    self.pending.clear();
                    self.committed.clear();
                    self.reasoning.clear();
                    self.phase = "Working".into();
                }
            }
            CodingEvent::ToolCallStarted {
                tool_name,
                tool_call_id,
                arguments,
            } => {
                if self.tools.len() == MAX_TOOL_RECORDS {
                    self.tools.pop_front();
                }
                let arguments = arguments
                    .as_str()
                    .map_or_else(|| arguments.to_string(), str::to_owned);
                self.tools.push_back(ToolRecord {
                    id:        tool_call_id.clone(),
                    session:   envelope.session_id.clone(),
                    name:      tool_name.clone(),
                    arguments: arguments.clone(),
                    output:    String::new(),
                    complete:  false,
                    failed:    false,
                });
                if root {
                    self.phase = format!("Running {tool_name}");
                    output.push(Output::Text(format!(
                        "  {}",
                        tool_render::heading(tool_name, &arguments)
                    )));
                }
            }
            CodingEvent::ToolCallOutputDelta { delta } => {
                if let Some(tool) = self.find_tool(envelope, envelope.tool_call_id.as_deref()) {
                    append_tail(&mut tool.output, delta);
                }
            }
            CodingEvent::ToolCallCompleted {
                tool_call_id,
                output: result,
                is_error,
                output_bytes_omitted,
                ..
            } => {
                if let Some(tool) = self.find_tool(envelope, Some(tool_call_id)) {
                    tool.complete = true;
                    tool.failed = *is_error;
                    let result = result
                        .as_str()
                        .map_or_else(|| result.to_string(), str::to_owned);
                    tool.output.clear();
                    append_tail(&mut tool.output, &result);
                    if root {
                        output.extend(tool_render::result(
                            tool,
                            Some(if *is_error { 12 } else { 10 }),
                            *output_bytes_omitted,
                        ));
                    }
                }
            }
            CodingEvent::ToolProcessCompleted {
                exit_code,
                duration_ms,
                ..
            } if root => {
                output.push(Output::Text(format!(
                    "    exit {} · {:.1}s",
                    exit_code.map_or_else(|| "?".into(), |code| code.to_string()),
                    *duration_ms as f64 / 1000.0
                )));
            }
            CodingEvent::LlmRetry {
                attempt,
                delay_secs,
                error,
                ..
            } if root => {
                self.phase = format!("Retrying in {delay_secs:.1}s");
                output.push(Output::Text(format!(
                    "[Retry {} in {delay_secs:.1}s: {}]",
                    attempt + 1,
                    error.message
                )));
            }
            CodingEvent::RoundInterrupted { .. } if root => {
                output.extend(self.interrupted());
            }
            CodingEvent::CompactionStarted { .. } if root => {
                self.phase = "Compacting context".into();
            }
            CodingEvent::CompactionCompleted {
                original_turn_count,
                preserved_turn_count,
                ..
            } if root => {
                output.push(Output::Text(format!("[Context compacted: {original_turn_count} turns → {preserved_turn_count}; transcript preserved]")));
            }
            CodingEvent::CompactionFailed { error, .. } if root => output.push(Output::Text(
                format!("[Compaction failed: {}]", error.message),
            )),
            CodingEvent::CompactionCancelled { .. } if root => {
                output.push(Output::Text("[Compaction cancelled]".into()));
            }
            CodingEvent::Error { error } if root => {
                output.push(Output::Text(format!("[Error: {}]", error.message)));
            }
            CodingEvent::Warning { message, .. } if root => {
                output.push(Output::Text(format!("[Warning: {message}]")));
            }
            CodingEvent::SubAgentSpawned { agent_id, task, .. }
            | CodingEvent::SubAgentTurnStarted { agent_id, task, .. } => {
                self.agents.insert(
                    agent_id.clone(),
                    format!("working · {}", text::truncate(task, 100)),
                );
            }
            CodingEvent::SubAgentCompleted {
                agent_id, success, ..
            } => {
                self.agents.insert(
                    agent_id.clone(),
                    if *success {
                        "completed".into()
                    } else {
                        "failed".into()
                    },
                );
            }
            CodingEvent::SubAgentFailed {
                agent_id, error, ..
            } => {
                self.agents
                    .insert(agent_id.clone(), format!("failed · {}", error.message));
            }
            CodingEvent::ProcessingEnd if root => {
                output.extend(self.interrupted());
                self.phase.clear();
            }
            _ => {}
        }
        output
    }

    pub(super) fn interrupted(&mut self) -> Vec<Output> {
        let mut output = Vec::new();
        if !self.pending.is_empty() {
            output.push(Output::Text(mem::take(&mut self.pending)));
        }
        if !self.committed.is_empty() || !output.is_empty() {
            output.push(Output::Text("[Response interrupted]".into()));
        }
        self.committed.clear();
        self.reasoning.clear();
        output
    }

    pub(super) fn activity(&self, busy: bool, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        if busy {
            lines.push(format!(
                "{} · Esc to cancel",
                if self.phase.is_empty() {
                    "Working"
                } else {
                    &self.phase
                }
            ));
        }
        if self.show_reasoning && !self.reasoning.is_empty() {
            lines.extend(
                text::wrap(&text::plain(&self.reasoning), width)
                    .into_iter()
                    .rev()
                    .take(2)
                    .rev(),
            );
        }
        if !self.pending.is_empty() {
            lines.extend(
                text::wrap(&text::plain(&self.pending), width)
                    .into_iter()
                    .rev()
                    .take(5)
                    .rev(),
            );
        }
        for (id, state) in self
            .agents
            .iter()
            .filter(|(_, state)| state.starts_with("working"))
            .take(2)
        {
            lines.push(text::truncate(&format!("Agent {id}: {state}"), width));
        }
        if self.expand_tools {
            for tool in self.tools.iter().filter(|tool| !tool.complete).take(2) {
                let tail: Vec<_> = text::plain(&tool.output)
                    .lines()
                    .rev()
                    .take(3)
                    .map(str::to_owned)
                    .collect();
                lines.extend(tail.into_iter().rev());
            }
        }
        lines
    }

    fn find_tool(
        &mut self,
        envelope: &CodingAgentEvent,
        id: Option<&str>,
    ) -> Option<&mut ToolRecord> {
        self.tools
            .iter_mut()
            .rev()
            .find(|tool| tool.session == envelope.session_id && id == Some(tool.id.as_str()))
    }
}

/// Paragraph boundaries are stable only outside unfinished code fences and
/// tables.
fn stable_boundary(text: &str) -> Option<usize> {
    let mut fence: Option<(char, usize)> = None;
    let mut end = 0;
    let mut boundary = None;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if let Some(marker @ ('`' | '~')) = trimmed.chars().next() {
            let count = trimmed.chars().take_while(|value| *value == marker).count();
            if count >= 3 {
                match fence {
                    None => fence = Some((marker, count)),
                    Some((opening, length))
                        if marker == opening
                            && count >= length
                            && trimmed[count..].trim().is_empty() =>
                    {
                        fence = None;
                    }
                    _ => {}
                }
            }
        }
        end += line.len();
        if fence.is_none() && line.trim().is_empty() && line.ends_with('\n') {
            boundary = Some(end);
        }
    }
    boundary
}

fn append_tail(destination: &mut String, delta: &str) {
    destination.push_str(delta);
    if destination.len() > OUTPUT_PREVIEW_BYTES {
        let mut start = destination.len() - OUTPUT_PREVIEW_BYTES;
        while !destination.is_char_boundary(start) {
            start += 1;
        }
        destination.drain(..start);
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use pebble_coding_agent::events::TokenUsage;

    use super::*;

    fn event(seq: u64, event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("root", event, SystemTime::now()).with_seq(seq)
    }

    #[test]
    fn finalizing_streamed_paragraphs_does_not_print_them_twice() {
        let mut transcript = Transcript {
            root: "root".into(),
            ..Transcript::default()
        };
        let first = event(1, CodingEvent::TextDelta {
            delta: "Hello\n\nWorld".into(),
        });
        let output = transcript.apply(&first, false);
        assert!(matches!(&output[..], [Output::Markdown(text)] if text == "Hello\n\n"));
        assert!(transcript.apply(&first, false).is_empty());
        let output = transcript.apply(
            &event(2, CodingEvent::AssistantMessage {
                text:            "Hello\n\nWorld".into(),
                model:           "model".into(),
                usage:           TokenUsage::default(),
                cost_usd_micros: None,
                cost_source:     None,
                tool_call_count: 0,
                context_window:  None,
                reasoning:       None,
            }),
            false,
        );
        assert!(matches!(&output[..], [Output::Markdown(text)] if text == "World"));
    }

    #[test]
    fn interrupted_text_is_retained_during_replay_and_live_output() {
        for historical in [false, true] {
            let mut transcript = Transcript::new("root");
            let output = transcript.apply(
                &event(1, CodingEvent::TextDelta {
                    delta: "Unfinished answer".into(),
                }),
                historical,
            );
            assert!(output.is_empty());
            let output = transcript.apply(
                &event(2, CodingEvent::RoundInterrupted { generation: 1 }),
                historical,
            );
            assert!(
                matches!(&output[..], [Output::Text(text), Output::Text(notice)] if text == "Unfinished answer" && notice == "[Response interrupted]")
            );
            assert!(transcript.interrupted().is_empty());
        }
    }

    #[test]
    fn code_blocks_remain_mutable_until_their_closing_fence() {
        assert_eq!(stable_boundary("intro\n\n```rust\n\nlet x = 1;\n"), Some(7));
        assert_eq!(stable_boundary("```rust\n\nlet x = 1;\n```\n\n"), Some(25));
        assert_eq!(stable_boundary("````md\n```\n\n"), None);
        assert_eq!(stable_boundary("```sh\n~~~\n\n"), None);
    }
}
