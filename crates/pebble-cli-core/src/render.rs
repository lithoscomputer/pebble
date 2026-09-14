//! Rendering the event stream to standard error.
//!
//! [`Style`] says how the stream reaches the terminal: readable lines, JSON
//! envelopes, or nothing but the summary. [`RenderOptions`] says what more
//! the readable lines carry. Every option is off by default, so a renderer
//! built without them prints what it always has. The `pebble` command turns
//! them on from its `--tool-results`, `--transcript`, and `--verbose` flags;
//! an embedder that runs its own command over the same session (fabro's
//! `fabro exec`) opts in through [`crate::session::run_prompt_with`].
//!
//! By default the readable lines say which model was asked, stream the
//! model's text as it arrives, and give each turn's size, each tool call's
//! name with its arguments cut to one line, each failed call, and each
//! process's exit. [`RenderOptions::tool_results`] prints the arguments in
//! full under the `[tool]` line and adds a `[result]` block with what each
//! call answered. [`RenderOptions::transcript`] adds a `[reasoning]` block
//! for each turn that carried one, and the text of a turn that did not
//! stream, so the transcript is whole on a route that answers in one piece.
//! Every block is cut to [`BLOCK_LIMIT`] bytes, with the byte count when it
//! was longer. The answer on standard output is the session's, not the
//! renderer's, and is not changed by any option.

use std::collections::HashSet;
use std::io::{self, Write as _};
#[cfg(test)]
use std::sync::{Arc, Mutex, PoisonError};

use pebble_coding_agent::CodingAgent;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use pebble_coding_agent::projection::SessionProjection;
use serde_json::Value;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;

use crate::terminal::{print_err, print_err_fragment};

/// How much of a block the readable lines show, in bytes.
///
/// A tool's arguments, a tool's result, and a turn's reasoning are each cut
/// here, at a character boundary, and the cut is reported with the byte
/// count.
pub const BLOCK_LIMIT: usize = 4096;

/// How events reach the terminal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Style {
    /// One readable line per event, with the model's text streamed.
    #[default]
    Text,
    /// One JSON envelope per line, unchanged from the event stream.
    Json,
    /// Nothing but the closing summary.
    Quiet,
}

/// What the readable lines of [`Style::Text`] say beyond the default.
///
/// Every option is off by default, so a renderer built without them prints
/// what it always has. The options apply to [`Style::Text`] only: JSON
/// carries every event whole, and quiet says nothing.
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

/// Where the JSON envelopes of [`Style::Json`] go.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JsonStream {
    /// Standard error, beside the other diagnostics, leaving standard output
    /// to the answer.
    #[default]
    Stderr,
    /// Standard output, as the command's product, for a caller that reads
    /// the event stream rather than the answer.
    Stdout,
}

/// Reads the live stream until the agent closes.
pub struct Renderer {
    style:     Style,
    json_to:   JsonStream,
    options:   RenderOptions,
    sink:      Sink,
    summary:   Summary,
    /// Whether a streamed line is open on the terminal.
    streaming: bool,
    /// The sessions whose text has streamed since their last turn.
    streamed:  HashSet<String>,
}

/// Where the readable lines go.
enum Sink {
    /// Standard error, through the command's boundary.
    Stderr,
    /// A buffer, so a test can read what was said.
    #[cfg(test)]
    Memory(Arc<Mutex<String>>),
}

impl Sink {
    /// Writes one line.
    fn line(&self, text: &str) {
        match self {
            Self::Stderr => print_err(text),
            #[cfg(test)]
            Self::Memory(buffer) => {
                let mut buffer = buffer.lock().unwrap_or_else(PoisonError::into_inner);
                buffer.push_str(text);
                buffer.push('\n');
            }
        }
    }

    /// Writes text with no line ending, for streamed output.
    fn fragment(&self, text: &str) {
        match self {
            Self::Stderr => print_err_fragment(text),
            #[cfg(test)]
            Self::Memory(buffer) => buffer
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push_str(text),
        }
    }
}

/// What the stream said, kept for the closing report.
///
/// The facts are the [`SessionProjection`]'s, the one fold every embedder
/// reads a session through, so this summary and an application's view of the
/// same stream agree. The one thing added is what this reader missed of the
/// stream, which is the reader's own and not the session's.
#[derive(Debug, Default)]
pub struct Summary {
    projection: SessionProjection,
    dropped:    u64,
}

impl Renderer {
    #[must_use]
    pub fn new(style: Style) -> Self {
        Self {
            style,
            json_to: JsonStream::default(),
            options: RenderOptions::default(),
            sink: Sink::Stderr,
            summary: Summary::default(),
            streaming: false,
            streamed: HashSet::new(),
        }
    }

    /// Sends [`Style::Json`] envelopes to `stream` instead of standard error.
    #[must_use]
    pub fn json_to(mut self, stream: JsonStream) -> Self {
        self.json_to = stream;
        self
    }

    /// Says what more the readable lines of [`Style::Text`] carry.
    #[must_use]
    pub fn options(mut self, options: RenderOptions) -> Self {
        self.options = options;
        self
    }

    /// Consumes the stream until it closes and answers with what it counted.
    ///
    /// The stream ends when the agent is shut down, which is what lets the
    /// caller join this task before the agent is dropped.
    pub async fn run(mut self, mut events: broadcast::Receiver<CodingAgentEvent>) -> Summary {
        loop {
            match events.recv().await {
                Ok(event) => {
                    let ended = matches!(event.event, CodingEvent::SessionEnded);
                    self.render(&event);
                    if ended {
                        break;
                    }
                }
                Err(RecvError::Lagged(dropped)) => {
                    self.summary.dropped += dropped;
                }
                Err(RecvError::Closed) => break,
            }
        }
        if self.streaming {
            self.sink.line("");
        }
        self.summary
    }

    fn render(&mut self, event: &CodingAgentEvent) {
        self.summary.projection.apply(event);
        match self.style {
            Style::Json => match serde_json::to_string(event) {
                Ok(line) => match self.json_to {
                    JsonStream::Stderr => self.sink.line(&line),
                    // A closed pipe is the reader's choice; nothing can be
                    // said to them about it.
                    JsonStream::Stdout => {
                        let mut stdout = io::stdout().lock();
                        let _ = stdout
                            .write_all(line.as_bytes())
                            .and_then(|()| stdout.write_all(b"\n"))
                            .and_then(|()| stdout.flush());
                    }
                },
                Err(error) => self
                    .sink
                    .line(&format!("error: rendering an event as JSON: {error}")),
            },
            Style::Text => self.render_text(event),
            Style::Quiet => {}
        }
    }

    fn render_text(&mut self, envelope: &CodingAgentEvent) {
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
        if self.streaming {
            self.sink.line("");
            self.streaming = false;
        }
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
            } => {
                let streamed = self.streamed.remove(&envelope.session_id);
                if self.options.transcript {
                    let reasoning = reasoning
                        .as_ref()
                        .and_then(|reasoning| reasoning.summary().or_else(|| reasoning.trace()));
                    if let Some(reasoning) = reasoning {
                        self.block("[reasoning]", reasoning);
                    }
                    if !streamed && !text.is_empty() {
                        self.sink.line(text.trim_end_matches('\n'));
                    }
                }
                self.sink.line(&format!(
                    "[turn] {} tokens, {tool_call_count} tool call(s)",
                    usage.total_tokens()
                ));
            }
            CodingEvent::ToolCallStarted {
                tool_name,
                arguments,
                ..
            } => {
                if self.options.tool_results {
                    self.block(&format!("[tool] {tool_name}"), &value_text(arguments));
                } else {
                    self.sink.line(&format!(
                        "[tool] {tool_name} {}",
                        // A custom tool's arguments are one free-form string,
                        // shown as text rather than as a JSON literal.
                        abbreviate(&match arguments {
                            Value::String(text) => text.clone(),
                            other => other.to_string(),
                        })
                    ));
                }
            }
            CodingEvent::ToolCallCompleted {
                tool_name,
                is_error,
                output,
                output_bytes_omitted,
                ..
            } => {
                if self.options.tool_results {
                    let failed = if *is_error { " failed" } else { "" };
                    let omitted = if *output_bytes_omitted > 0 {
                        format!(" ({output_bytes_omitted} bytes not retained)")
                    } else {
                        String::new()
                    };
                    let header = format!("[result] {tool_name}{failed}{omitted}");
                    self.block(&header, &value_text(output));
                } else if *is_error {
                    self.sink.line(&format!(
                        "[tool] {tool_name} failed: {}",
                        abbreviate(output.as_str().unwrap_or_default())
                    ));
                }
            }
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

    /// Prints `text` under `header`, each line indented, cut to
    /// [`BLOCK_LIMIT`] bytes with the byte count when it was longer.
    fn block(&self, header: &str, text: &str) {
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

impl Summary {
    /// The fold the report reads its facts from.
    #[must_use]
    pub fn projection(&self) -> &SessionProjection {
        &self.projection
    }

    /// Prints what the prompt used, after the answer.
    pub fn report(&self, style: Style) {
        if style != Style::Text {
            return;
        }
        print_err("");
        for line in self.lines() {
            print_err(&line);
        }
    }

    /// The report's lines, in order.
    ///
    /// Turns, tokens, and cost are the root session's, as the projection
    /// keeps them, and agree with the prompt's report: a compaction's summary
    /// call is on the stream with its usage, so the fold bills it as the
    /// report does. What the children did is on the `subagents` line. Tools
    /// and retries are the tree's.
    fn lines(&self) -> Vec<String> {
        let projection = &self.projection;
        let mut lines = vec![format!("turns:  {}", projection.messages)];

        let calls: u64 = projection.tools.values().map(|tool| tool.calls).sum();
        let failed: u64 = projection.tools.values().map(|tool| tool.errors).sum();
        let open: u64 = projection.tools.values().map(|tool| tool.open).sum();
        let unfinished = if open > 0 {
            format!(", {open} unfinished")
        } else {
            String::new()
        };
        lines.push(format!(
            "tools:  {calls} call(s), {failed} failed{unfinished}"
        ));
        let named = projection
            .tools
            .iter()
            .map(|(name, tool)| format!("{name} x{}", tool.calls))
            .collect::<Vec<_>>()
            .join(", ");
        if !named.is_empty() {
            lines.push(format!("        {named}"));
        }

        let tokens = projection.usage.tokens;
        lines.push(format!(
            "tokens: {} in, {} out, {} reasoning, {} cached ({} total)",
            tokens.input,
            tokens.output,
            tokens.reasoning,
            tokens.cache_read + tokens.cache_write,
            tokens.total()
        ));
        lines.push(match projection.usage.cost {
            Some(cost) => format!("cost:   {}", dollars(cost.usd_micros)),
            None => "cost:   not reported for this model".to_owned(),
        });

        let spawned = projection.subagent_counts.spawned;
        if spawned > 0 || !projection.descendants.is_empty() {
            let usage = projection.descendant_usage();
            let turns: u64 = projection
                .descendants
                .values()
                .map(|account| account.messages)
                .sum();
            let cost = usage.cost.map_or_else(String::new, |cost| {
                format!(", {}", dollars(cost.usd_micros))
            });
            lines.push(format!(
                "subagents: {spawned} spawned, {turns} turn(s), {} tokens{cost}",
                usage.total_tokens()
            ));
        }
        if projection.retries > 0 {
            lines.push(format!("retries: {}", projection.retries));
        }
        if self.dropped > 0 {
            lines.push(format!(
                "dropped: {} event(s) this reader missed",
                self.dropped
            ));
        }
        lines
    }
}

/// Millionths of a dollar, as dollars.
fn dollars(usd_micros: u64) -> String {
    format!("${:.4}", usd_micros as f64 / 1_000_000.0)
}

/// A JSON value as text: a string as itself, anything else pretty-printed.
fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
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

/// Reports what became of each MCP server the agent was built with: they
/// started while the agent was built, before any subscriber could hear the
/// events, so the outcomes are read from the agent.
pub fn report_mcp_servers(agent: &CodingAgent, style: Style) {
    if style != Style::Text {
        return;
    }
    for status in agent.snapshot().mcp_servers() {
        match &status.error {
            None => print_err(&format!(
                "[mcp] {}: {} tool(s)",
                status.server,
                status.tools.len()
            )),
            Some(error) => print_err(&format!("[mcp] {} failed: {error}", status.server)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use pebble_coding_agent::events::{
        CompactionReason, Cost, CostSource, ErrorData, ErrorKind, InputSource, LlmRetryPhase,
        ReasoningOutput, TokenCounts, Usage,
    };
    use serde_json::json;

    use super::*;

    fn root(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_root", event, SystemTime::UNIX_EPOCH)
    }

    fn child(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_child", event, SystemTime::UNIX_EPOCH)
            .with_parent_session_id("ses_root")
    }

    /// `input` and `output` tokens, priced from the catalog when `cost` is
    /// given.
    fn priced(input: u64, output: u64, cost: Option<u64>) -> Usage {
        Usage {
            tokens: TokenCounts {
                input,
                output,
                ..TokenCounts::default()
            },
            cost:   cost.map(|usd_micros| Cost {
                usd_micros,
                source: CostSource::Catalog,
            }),
        }
    }

    fn message(input: u64, output: u64, cost: Option<u64>) -> CodingEvent {
        CodingEvent::AssistantMessage {
            text:            "ok".into(),
            model:           "model".into(),
            usage:           priced(input, output, cost),
            tool_call_count: 0,
            context_window:  None,
            reasoning:       None,
        }
    }

    fn tool_started(tool_name: &str, tool_call_id: &str) -> CodingEvent {
        CodingEvent::ToolCallStarted {
            tool_name:    tool_name.into(),
            tool_call_id: tool_call_id.into(),
            arguments:    json!({"path": "/w/a.txt"}),
        }
    }

    fn tool_completed(tool_name: &str, tool_call_id: &str, is_error: bool) -> CodingEvent {
        tool_answered(tool_name, tool_call_id, json!("done"), is_error, 0)
    }

    fn tool_answered(
        tool_name: &str,
        tool_call_id: &str,
        output: Value,
        is_error: bool,
        output_bytes_omitted: usize,
    ) -> CodingEvent {
        CodingEvent::ToolCallCompleted {
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            output,
            metadata: pebble_agent::ToolOutputMetadata::default(),
            is_error,
            error_kind: None,
            output_bytes_observed: 0,
            output_bytes_retained: 0,
            output_bytes_omitted,
        }
    }

    fn ask() -> CodingEvent {
        CodingEvent::LlmRequestStarted {
            requested_model: "model".into(),
        }
    }

    fn delta(text: &str) -> CodingEvent {
        CodingEvent::TextDelta { delta: text.into() }
    }

    /// A committed turn with `text`, and its reasoning when the route
    /// carried one.
    fn turn(text: &str, reasoning: Option<ReasoningOutput>) -> CodingEvent {
        CodingEvent::AssistantMessage {
            text: text.into(),
            model: "model".into(),
            usage: priced(10, 5, None),
            tool_call_count: 0,
            context_window: None,
            reasoning,
        }
    }

    fn retry() -> CodingEvent {
        CodingEvent::LlmRetry {
            provider:   "test".into(),
            model:      "model".into(),
            attempt:    0,
            delay_secs: 0.1,
            error:      ErrorData::new(ErrorKind::Llm, "slow down"),
            phase:      LlmRetryPhase::Open,
        }
    }

    /// One prompt as the stream tells it: the root answers twice with a retry
    /// before the first, a child answers once and fails a tool call, and two
    /// of the root's tool calls succeed.
    fn scripted_prompt() -> Vec<CodingAgentEvent> {
        vec![
            root(CodingEvent::UserInput {
                text:    "go".into(),
                content: None,
                source:  InputSource::Prompt,
            }),
            root(tool_started("read_file", "r1")),
            root(tool_completed("read_file", "r1", false)),
            root(retry()),
            root(message(10, 5, Some(100))),
            root(CodingEvent::SubAgentSpawned {
                agent_id:   "a1".into(),
                depth:      1,
                task:       "look".into(),
                generation: 1,
            }),
            child(tool_started("read_file", "r2")),
            child(tool_completed("read_file", "r2", true)),
            child(message(7, 3, Some(400))),
            root(CodingEvent::SubAgentCompleted {
                agent_id:   "a1".into(),
                depth:      1,
                generation: 1,
                success:    true,
                turns_used: 1,
            }),
            root(tool_started("edit_file", "e1")),
            root(tool_completed("edit_file", "e1", false)),
            root(message(20, 5, Some(200))),
            root(CodingEvent::ProcessingEnd),
            root(CodingEvent::SessionEnded),
        ]
    }

    async fn summarize(events: &[CodingAgentEvent], capacity: usize) -> Summary {
        let (sender, receiver) = broadcast::channel(capacity);
        for event in events {
            sender
                .send(event.clone())
                .expect("the renderer holds the receiver");
        }
        Renderer::new(Style::Quiet).run(receiver).await
    }

    /// What the readable lines say of `events` under `options`.
    async fn rendered(events: &[CodingAgentEvent], options: RenderOptions) -> String {
        let (sender, receiver) = broadcast::channel(events.len().max(1));
        for event in events {
            sender
                .send(event.clone())
                .expect("the renderer holds the receiver");
        }
        let buffer = Arc::new(Mutex::new(String::new()));
        let mut renderer = Renderer::new(Style::Text).options(options);
        renderer.sink = Sink::Memory(Arc::clone(&buffer));
        renderer.run(receiver).await;
        let said = buffer.lock().unwrap_or_else(PoisonError::into_inner);
        said.clone()
    }

    /// A tool round as the stream tells it: the model asks for a tool while
    /// answering, the tool answers, and the model answers again.
    fn scripted_tool_round() -> Vec<CodingAgentEvent> {
        vec![
            root(ask()),
            root(delta("Reading.")),
            root(turn("Reading.", None)),
            root(tool_started("read_file", "r1")),
            root(tool_answered(
                "read_file",
                "r1",
                json!("one\ntwo\n"),
                false,
                0,
            )),
            root(tool_started("edit_file", "e1")),
            root(tool_answered(
                "edit_file",
                "e1",
                json!("no such anchor"),
                true,
                0,
            )),
            root(ask()),
            root(delta("Done.")),
            root(turn("Done.", None)),
            root(CodingEvent::ProcessingEnd),
            root(CodingEvent::SessionEnded),
        ]
    }

    #[tokio::test]
    async fn the_default_lines_are_unchanged() {
        let said = rendered(&scripted_tool_round(), RenderOptions::default()).await;
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

    #[tokio::test]
    async fn tool_results_show_each_calls_arguments_and_answer() {
        let options = RenderOptions::default().tool_results(true);
        let said = rendered(&scripted_tool_round(), options).await;
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

    #[tokio::test]
    async fn a_result_is_cut_at_a_character_with_its_byte_count() {
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
        let said = rendered(&events, options).await;
        let expected = format!(
            "[tool] shell\n  {{\n    \"path\": \"/w/a.txt\"\n  }}\n[result] shell (7 bytes not \
             retained)\n  {}\n  … {shown} of {} bytes\n[tool] shell\n  {{\n    \"path\": \
             \"/w/a.txt\"\n  }}\n[result] shell (empty)\n",
            &long[..shown],
            long.len()
        );
        assert_eq!(said, expected);
    }

    #[tokio::test]
    async fn a_structured_result_is_pretty_printed_and_a_blank_line_stays_blank() {
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
        let said = rendered(&events, options).await;
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
        let said = rendered(&events, options).await;
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

    #[tokio::test]
    async fn the_transcript_adds_reasoning_and_the_text_that_did_not_stream() {
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

        let plain = rendered(&events, RenderOptions::default()).await;
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

        let transcript = rendered(&events, RenderOptions::default().transcript(true)).await;
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
    fn verbose_is_every_option() {
        assert_eq!(
            RenderOptions::verbose(),
            RenderOptions::default().tool_results(true).transcript(true)
        );
        assert_ne!(RenderOptions::verbose(), RenderOptions::default());
    }

    #[tokio::test]
    async fn the_summary_is_the_projection_of_the_stream() {
        let events = scripted_prompt();
        let summary = summarize(&events, events.len()).await;

        let mut projection = SessionProjection::new();
        projection.apply_all(&events);
        assert_eq!(summary.projection(), &projection);
        assert_eq!(projection.messages, 2, "the root's turns");
        assert_eq!(projection.retries, 1);
        assert_eq!(projection.descendants["ses_child"].messages, 1);

        assert_eq!(summary.lines(), [
            "turns:  2",
            "tools:  3 call(s), 1 failed",
            "        edit_file x1, read_file x2",
            "tokens: 30 in, 10 out, 0 reasoning, 0 cached (40 total)",
            "cost:   $0.0003",
            "subagents: 1 spawned, 1 turn(s), 10 tokens, $0.0004",
            "retries: 1",
        ]);
    }

    #[tokio::test]
    async fn a_prompt_without_children_or_retries_says_nothing_of_them() {
        let events = [
            root(CodingEvent::UserInput {
                text:    "go".into(),
                content: None,
                source:  InputSource::Prompt,
            }),
            root(message(10, 5, None)),
            root(CodingEvent::ProcessingEnd),
            root(CodingEvent::SessionEnded),
        ];
        let summary = summarize(&events, events.len()).await;
        assert_eq!(summary.lines(), [
            "turns:  1",
            "tools:  0 call(s), 0 failed",
            "tokens: 10 in, 5 out, 0 reasoning, 0 cached (15 total)",
            "cost:   not reported for this model",
        ]);
    }

    #[tokio::test]
    async fn a_compactions_summary_call_is_on_the_tokens_line() {
        let events = [
            root(CodingEvent::UserInput {
                text:    "go".into(),
                content: None,
                source:  InputSource::Prompt,
            }),
            root(message(10, 5, Some(100))),
            root(CodingEvent::CompactionCompleted {
                original_turn_count:    4,
                preserved_turn_count:   1,
                summary_token_estimate: 20,
                tracked_file_count:     0,
                reason:                 CompactionReason::Threshold,
                usage:                  priced(30, 2, Some(200)),
            }),
            root(CodingEvent::ProcessingEnd),
            root(CodingEvent::SessionEnded),
        ];
        let summary = summarize(&events, events.len()).await;
        assert_eq!(summary.lines(), [
            "turns:  1",
            "tools:  0 call(s), 0 failed",
            "tokens: 40 in, 7 out, 0 reasoning, 0 cached (47 total)",
            "cost:   $0.0003",
        ]);
    }

    #[tokio::test]
    async fn a_lagging_reader_reports_what_it_missed() {
        let events = scripted_prompt();
        // A channel too small for the stream: the reader is told how many
        // events it lost, and the summary says so, on top of what it folded.
        let capacity = 2;
        let summary = summarize(&events, capacity).await;
        let dropped = u64::try_from(events.len() - capacity).expect("a small count");
        let lines = summary.lines();
        assert_eq!(
            lines.last().map(String::as_str),
            Some(format!("dropped: {dropped} event(s) this reader missed").as_str())
        );
    }
}
