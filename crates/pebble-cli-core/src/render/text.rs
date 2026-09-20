//! The readable lines of [`Style::Text`](super::Style::Text).

use std::collections::HashSet;

use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent, ReasoningOutput, Usage};
use serde_json::Value;

use super::BLOCK_LIMIT;
use crate::terminal::{print_err, print_err_fragment};

/// What the readable lines of [`Style::Text`](super::Style::Text) say beyond
/// the default.
///
/// Every option is off by default, so a renderer built without them prints
/// what it always has. The options apply to `Style::Text` only: JSON carries
/// every event whole, and quiet says nothing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderOptions {
    tool_results: bool,
    transcript:   bool,
}

impl RenderOptions {
    /// Everything the readable lines can say: tool results and the
    /// transcript.
    #[must_use]
    pub fn verbose() -> Self {
        Self {
            tool_results: true,
            transcript:   true,
        }
    }

    /// Prints each tool call's arguments in full under its `[tool]` line and
    /// a `[result]` block with what the call answered, success or failure.
    /// Both are cut to [`BLOCK_LIMIT`] bytes, with the byte count when
    /// longer. Off, a call is one line with its arguments abbreviated, and
    /// only a failed call says what it answered.
    #[must_use]
    pub fn tool_results(mut self, enabled: bool) -> Self {
        self.tool_results = enabled;
        self
    }

    /// Prints a `[reasoning]` block for each turn that carried reasoning:
    /// the model's summary of it, or its verbatim trace when there is no
    /// summary. Also prints the text of a turn that did not stream, so a
    /// route that answers in one piece still shows what the model said;
    /// text that streamed is not repeated. The final answer on standard
    /// output is unchanged.
    #[must_use]
    pub fn transcript(mut self, enabled: bool) -> Self {
        self.transcript = enabled;
        self
    }
}

/// Where the readable lines go.
enum Sink {
    /// Standard error, through the command's boundary.
    Stderr,
    /// A buffer, so a test can read what was said.
    #[cfg(test)]
    Memory(String),
}

impl Sink {
    /// Writes one line.
    fn line(&mut self, text: &str) {
        match self {
            Self::Stderr => print_err(text),
            #[cfg(test)]
            Self::Memory(buffer) => {
                buffer.push_str(text);
                buffer.push('\n');
            }
        }
    }

    /// Writes text with no line ending, for streamed output.
    fn fragment(&mut self, text: &str) {
        match self {
            Self::Stderr => print_err_fragment(text),
            #[cfg(test)]
            Self::Memory(buffer) => buffer.push_str(text),
        }
    }
}

/// Writes one readable line per event, with the model's text streamed.
pub(super) struct TextRenderer {
    options:   RenderOptions,
    sink:      Sink,
    /// Whether a streamed line is open on the terminal.
    streaming: bool,
    /// The sessions whose text has streamed since their last turn.
    streamed:  HashSet<String>,
}

impl TextRenderer {
    pub(super) fn new(options: RenderOptions) -> Self {
        Self {
            options,
            sink: Sink::Stderr,
            streaming: false,
            streamed: HashSet::new(),
        }
    }

    /// Ends a streamed line left open, once the stream is over.
    pub(super) fn close(&mut self) {
        if self.streaming {
            self.sink.line("");
            self.streaming = false;
        }
    }

    pub(super) fn render(&mut self, envelope: &CodingAgentEvent) {
        let event = &envelope.event;
        // The model's text arrives in pieces and is printed as it arrives; it
        // is the one thing that gets no line of its own.
        if let CodingEvent::TextDelta { delta } = event {
            self.sink.fragment(delta);
            self.streaming = true;
            if !self.streamed.contains(&envelope.session_id) {
                self.streamed.insert(envelope.session_id.clone());
            }
            return;
        }
        self.close();
        match event {
            CodingEvent::SessionStarted { provider, model } => self.sink.line(&format!(
                "[open] {}/{}",
                provider.as_deref().unwrap_or("?"),
                model.as_deref().unwrap_or("?")
            )),
            CodingEvent::LlmRequestStarted { requested_model } => {
                self.sink.line(&format!("[ask] {requested_model}"));
            }
            CodingEvent::AssistantMessage {
                text,
                usage,
                tool_call_count,
                reasoning,
                ..
            } => self.turn(
                &envelope.session_id,
                text,
                usage,
                *tool_call_count,
                reasoning.as_ref(),
            ),
            CodingEvent::ToolCallStarted {
                tool_name,
                arguments,
                ..
            } => self.tool_call(tool_name, arguments),
            CodingEvent::ToolCallCompleted {
                tool_name,
                is_error,
                output,
                output_bytes_omitted,
                ..
            } => self.tool_result(tool_name, *is_error, output, *output_bytes_omitted),
            CodingEvent::ToolProcessCompleted {
                exit_code,
                duration_ms,
                ..
            } => self.sink.line(&format!(
                "[exec] exit {} in {duration_ms} ms",
                exit_code.map_or_else(|| "?".to_owned(), |code| code.to_string())
            )),
            CodingEvent::LlmRetry {
                attempt,
                delay_secs,
                error,
                ..
            } => self.sink.line(&format!(
                "[retry] attempt {attempt} in {delay_secs:.1}s: {}",
                error.message
            )),
            CodingEvent::LoopDetected => self.sink.line("[loop] the agent is repeating itself"),
            CodingEvent::RouteFailover {
                from,
                to,
                attempt,
                error,
                ..
            } => self.sink.line(&format!(
                "[failover] {from} -> {to} (attempt {attempt}): {}",
                error.message
            )),
            CodingEvent::RouteFailoverStopped {
                route,
                reason,
                error,
                ..
            } => self.sink.line(&format!(
                "[failover] stopped on {route}, {reason}: {}",
                error.message
            )),
            CodingEvent::CompactionCompleted {
                original_turn_count,
                preserved_turn_count,
                ..
            } => self.sink.line(&format!(
                "[compact] {original_turn_count} turns down to {preserved_turn_count}"
            )),
            CodingEvent::Error { error } => self.sink.line(&format!("[error] {}", error.message)),
            CodingEvent::Warning { kind, message, .. } => {
                self.sink.line(&format!("[warning] {kind}: {message}"));
            }
            CodingEvent::SubAgentSpawned {
                agent_id, depth, ..
            } => {
                self.sink
                    .line(&format!("[spawn] {agent_id} at depth {depth}"));
            }
            CodingEvent::SubAgentCompleted {
                agent_id,
                success,
                turns_used,
                ..
            } => self.sink.line(&format!(
                "[child] {agent_id} {} after {turns_used} turn(s)",
                if *success { "completed" } else { "failed" }
            )),
            CodingEvent::SubAgentFailed {
                agent_id, error, ..
            } => self
                .sink
                .line(&format!("[child] {agent_id} failed: {}", error.message)),
            // Everything else is left off a terminal that is busy enough.
            // `CodingEvent` is `#[non_exhaustive]`, so this arm is required.
            _ => {}
        }
    }

    /// A committed turn: its `[reasoning]` block and its text under the
    /// transcript option, then its `[turn]` line.
    fn turn(
        &mut self,
        session_id: &str,
        text: &str,
        usage: &Usage,
        tool_call_count: usize,
        reasoning: Option<&ReasoningOutput>,
    ) {
        let streamed = self.streamed.remove(session_id);
        if self.options.transcript {
            let reasoning =
                reasoning.and_then(|reasoning| reasoning.summary().or_else(|| reasoning.trace()));
            if let Some(reasoning) = reasoning {
                self.block("[reasoning]", reasoning);
            }
            // Text that streamed is already on the terminal.
            if !streamed && !text.is_empty() {
                self.sink.line(text.trim_end_matches('\n'));
            }
        }
        self.sink.line(&format!(
            "[turn] {} tokens, {tool_call_count} tool call(s)",
            usage.total_tokens()
        ));
    }

    /// A tool call: its `[tool]` line, with the arguments under it in full
    /// under the tool-results option and cut to one line otherwise.
    fn tool_call(&mut self, tool_name: &str, arguments: &Value) {
        if self.options.tool_results {
            self.block(&format!("[tool] {tool_name}"), &value_text(arguments));
        } else {
            self.sink.line(&format!(
                "[tool] {tool_name} {}",
                abbreviate(&value_line(arguments))
            ));
        }
    }

    /// A tool's answer: a `[result]` block under the tool-results option,
    /// and otherwise one line for a call that failed.
    fn tool_result(&mut self, tool_name: &str, is_error: bool, output: &Value, omitted: usize) {
        if self.options.tool_results {
            let failed = if is_error { " failed" } else { "" };
            let omitted = if omitted > 0 {
                format!(" ({omitted} bytes not retained)")
            } else {
                String::new()
            };
            self.block(
                &format!("[result] {tool_name}{failed}{omitted}"),
                &value_text(output),
            );
        } else if is_error {
            self.sink.line(&format!(
                "[tool] {tool_name} failed: {}",
                abbreviate(output.as_str().unwrap_or_default())
            ));
        }
    }

    /// Prints `text` under `header`, each line indented, cut to
    /// [`BLOCK_LIMIT`] bytes with the byte count when it was longer.
    fn block(&mut self, header: &str, text: &str) {
        let text = text.trim_end_matches('\n');
        if text.is_empty() {
            self.sink.line(&format!("{header} (empty)"));
            return;
        }
        self.sink.line(header);
        let shown = floor_char_boundary(text, BLOCK_LIMIT);
        for line in text[..shown].lines() {
            if line.is_empty() {
                self.sink.line("");
            } else {
                self.sink.line(&format!("  {line}"));
            }
        }
        if shown < text.len() {
            self.sink
                .line(&format!("  … {shown} of {} bytes", text.len()));
        }
    }
}

/// A JSON value as text: a string as itself, anything else pretty-printed.
///
/// A custom tool's arguments are one free-form string, so a string is shown
/// as text rather than as a JSON literal. [`value_line`] makes the same
/// choice for one line.
fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

/// A JSON value on one line: a string as itself, anything else compact.
fn value_line(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// The largest char boundary at or below `index`, so a byte budget never
/// cuts a character. `text.len()` when `index` is at or past the end.
fn floor_char_boundary(text: &str, index: usize) -> usize {
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

/// One line of text, short enough to read.
fn abbreviate(text: &str) -> String {
    const LIMIT: usize = 120;
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= LIMIT {
        return flat;
    }
    flat.chars().take(LIMIT).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use pebble_coding_agent::events::ReasoningOutput;
    use serde_json::json;

    use super::super::fixtures::{
        ask, child, delta, root, scripted_tool_round, tool_answered, tool_started, turn,
    };
    use super::*;

    /// What the readable lines say of `events` under `options`.
    fn rendered(events: &[CodingAgentEvent], options: RenderOptions) -> String {
        let mut renderer = TextRenderer::new(options);
        renderer.sink = Sink::Memory(String::new());
        for event in events {
            renderer.render(event);
        }
        renderer.close();
        let Sink::Memory(said) = renderer.sink else {
            panic!("the test renderer writes to memory");
        };
        said
    }

    #[test]
    fn the_default_lines_are_unchanged() {
        let said = rendered(&scripted_tool_round(), RenderOptions::default());
        assert_eq!(
            said,
            "\
[ask] model
Reading.
[turn] 15 tokens, 0 tool call(s)
[tool] read_file {\"path\":\"/w/a.txt\"}
[tool] edit_file {\"path\":\"/w/a.txt\"}
[tool] edit_file failed: no such anchor
[ask] model
Done.
[turn] 15 tokens, 0 tool call(s)
"
        );
    }

    #[test]
    fn tool_results_show_each_calls_arguments_and_answer() {
        let options = RenderOptions::default().tool_results(true);
        let said = rendered(&scripted_tool_round(), options);
        assert_eq!(
            said,
            "\
[ask] model
Reading.
[turn] 15 tokens, 0 tool call(s)
[tool] read_file
  {
    \"path\": \"/w/a.txt\"
  }
[result] read_file
  one
  two
[tool] edit_file
  {
    \"path\": \"/w/a.txt\"
  }
[result] edit_file failed
  no such anchor
[ask] model
Done.
[turn] 15 tokens, 0 tool call(s)
"
        );
    }

    #[test]
    fn a_result_is_cut_at_a_character_with_its_byte_count() {
        // Three bytes a character, so the limit falls inside one: the cut
        // steps back to the boundary before it.
        let long = "€".repeat(BLOCK_LIMIT / 3 + 1);
        let shown = BLOCK_LIMIT - BLOCK_LIMIT % 3;
        let events = [
            root(tool_started("shell", "s1")),
            root(tool_answered("shell", "s1", json!(long), false, 7)),
            root(tool_started("shell", "s2")),
            root(tool_answered("shell", "s2", json!(""), false, 0)),
            root(CodingEvent::SessionEnded),
        ];
        let options = RenderOptions::default().tool_results(true);
        let said = rendered(&events, options);
        let kept = format!("  {}", &long[..shown]);
        let cut = format!("  … {shown} of {} bytes", long.len());
        let expected = [
            "[tool] shell",
            "  {",
            "    \"path\": \"/w/a.txt\"",
            "  }",
            "[result] shell (7 bytes not retained)",
            kept.as_str(),
            cut.as_str(),
            "[tool] shell",
            "  {",
            "    \"path\": \"/w/a.txt\"",
            "  }",
            "[result] shell (empty)",
            "",
        ]
        .join("\n");
        assert_eq!(said, expected);
    }

    #[test]
    fn a_structured_result_is_pretty_printed_and_a_blank_line_stays_blank() {
        let events = [
            root(tool_started("lookup", "l1")),
            root(tool_answered(
                "lookup",
                "l1",
                json!({"lines": "a\n\nb"}),
                false,
                0,
            )),
            root(CodingEvent::SessionEnded),
        ];
        let options = RenderOptions::default().tool_results(true);
        let said = rendered(&events, options);
        assert_eq!(
            said,
            "\
[tool] lookup
  {
    \"path\": \"/w/a.txt\"
  }
[result] lookup
  {
    \"lines\": \"a\\n\\nb\"
  }
"
        );
        let events = [
            root(tool_started("shell", "s1")),
            root(tool_answered("shell", "s1", json!("a\n\nb\n"), false, 0)),
            root(CodingEvent::SessionEnded),
        ];
        let said = rendered(&events, options);
        assert_eq!(
            said,
            "\
[tool] shell
  {
    \"path\": \"/w/a.txt\"
  }
[result] shell
  a

  b
"
        );
    }

    #[test]
    fn the_transcript_adds_reasoning_and_the_text_that_did_not_stream() {
        let events = [
            root(ask()),
            root(delta("Forty-")),
            root(delta("two.")),
            root(turn(
                "Forty-two.",
                Some(ReasoningOutput::from_trace("Six times seven.\n")),
            )),
            // A route that answers in one piece streams nothing: the text
            // is on the turn alone.
            root(ask()),
            root(turn(
                "Yes.\n",
                Some(ReasoningOutput::new("Confirming.", "It is 42.")),
            )),
            // A child's streamed text is the child's, not the root's.
            child(delta("hello")),
            root(ask()),
            root(turn("Bye.", None)),
            root(CodingEvent::ProcessingEnd),
            root(CodingEvent::SessionEnded),
        ];

        let plain = rendered(&events, RenderOptions::default());
        assert_eq!(
            plain,
            "\
[ask] model
Forty-two.
[turn] 15 tokens, 0 tool call(s)
[ask] model
[turn] 15 tokens, 0 tool call(s)
hello
[ask] model
[turn] 15 tokens, 0 tool call(s)
"
        );

        let transcript = rendered(&events, RenderOptions::default().transcript(true));
        assert_eq!(
            transcript,
            "\
[ask] model
Forty-two.
[reasoning]
  Six times seven.
[turn] 15 tokens, 0 tool call(s)
[ask] model
[reasoning]
  Confirming.
Yes.
[turn] 15 tokens, 0 tool call(s)
hello
[ask] model
Bye.
[turn] 15 tokens, 0 tool call(s)
"
        );
    }

    #[test]
    fn a_line_left_streaming_is_ended_when_the_stream_closes() {
        let events = [root(ask()), root(delta("Half"))];
        assert_eq!(
            rendered(&events, RenderOptions::default()),
            "[ask] model\nHalf\n"
        );
    }

    #[test]
    fn verbose_is_every_option() {
        assert_eq!(
            RenderOptions::verbose(),
            RenderOptions::default().tool_results(true).transcript(true)
        );
        assert_ne!(RenderOptions::verbose(), RenderOptions::default());
    }
}
