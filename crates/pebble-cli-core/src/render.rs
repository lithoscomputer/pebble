//! Rendering the event stream to standard error.

use std::io::{self, Write as _};

use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use pebble_coding_agent::projection::SessionProjection;
use pebble_coding_agent::{CodingAgent, PromptReport};
use serde_json::Value;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;

use crate::terminal::{print_err, print_err_fragment};

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
    summary:   Summary,
    streaming: bool,
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
            summary: Summary::default(),
            streaming: false,
        }
    }

    /// Sends [`Style::Json`] envelopes to `stream` instead of standard error.
    #[must_use]
    pub fn json_to(mut self, stream: JsonStream) -> Self {
        self.json_to = stream;
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
            print_err("");
        }
        self.summary
    }

    fn render(&mut self, event: &CodingAgentEvent) {
        self.summary.projection.apply(event);
        match self.style {
            Style::Json => match serde_json::to_string(event) {
                Ok(line) => match self.json_to {
                    JsonStream::Stderr => print_err(&line),
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
                Err(error) => print_err(&format!("error: rendering an event as JSON: {error}")),
            },
            Style::Text => self.render_text(&event.event),
            Style::Quiet => {}
        }
    }

    fn render_text(&mut self, event: &CodingEvent) {
        // The model's text arrives in pieces and is printed as it arrives; it
        // is the one thing that gets no line of its own.
        if let CodingEvent::TextDelta { delta } = event {
            print_err_fragment(delta);
            self.streaming = true;
            return;
        }
        if self.streaming {
            print_err("");
            self.streaming = false;
        }
        match event {
            CodingEvent::SessionStarted { provider, model } => print_err(&format!(
                "[open] {}/{}",
                provider.as_deref().unwrap_or("?"),
                model.as_deref().unwrap_or("?")
            )),
            CodingEvent::LlmRequestStarted { requested_model } => {
                print_err(&format!("[ask] {requested_model}"));
            }
            CodingEvent::AssistantMessage {
                usage,
                tool_call_count,
                ..
            } => print_err(&format!(
                "[turn] {} tokens, {tool_call_count} tool call(s)",
                usage.total()
            )),
            CodingEvent::ToolCallStarted {
                tool_name,
                arguments,
                ..
            } => print_err(&format!(
                "[tool] {tool_name} {}",
                // A custom tool's arguments are one free-form string, shown as
                // text rather than as a JSON literal.
                abbreviate(&match arguments {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
            )),
            CodingEvent::ToolCallCompleted {
                tool_name,
                is_error: true,
                output,
                ..
            } => print_err(&format!(
                "[tool] {tool_name} failed: {}",
                abbreviate(output.as_str().unwrap_or_default())
            )),
            CodingEvent::ToolProcessCompleted {
                exit_code,
                duration_ms,
                ..
            } => print_err(&format!(
                "[exec] exit {} in {duration_ms} ms",
                exit_code.map_or_else(|| "?".to_owned(), |code| code.to_string())
            )),
            CodingEvent::LlmRetry {
                attempt,
                delay_secs,
                error,
                ..
            } => print_err(&format!(
                "[retry] attempt {attempt} in {delay_secs:.1}s: {}",
                error.message
            )),
            CodingEvent::LoopDetected => print_err("[loop] the agent is repeating itself"),
            CodingEvent::RouteFailover {
                from,
                to,
                attempt,
                error,
                ..
            } => print_err(&format!(
                "[failover] {from} -> {to} (attempt {attempt}): {}",
                error.message
            )),
            CodingEvent::RouteFailoverStopped {
                route,
                reason,
                error,
                ..
            } => print_err(&format!(
                "[failover] stopped on {route}, {reason}: {}",
                error.message
            )),
            CodingEvent::CompactionCompleted {
                original_turn_count,
                preserved_turn_count,
                ..
            } => print_err(&format!(
                "[compact] {original_turn_count} turns down to {preserved_turn_count}"
            )),
            CodingEvent::Error { error } => print_err(&format!("[error] {}", error.message)),
            CodingEvent::Warning { kind, message, .. } => {
                print_err(&format!("[warning] {kind}: {message}"));
            }
            CodingEvent::SubAgentSpawned {
                agent_id, depth, ..
            } => {
                print_err(&format!("[spawn] {agent_id} at depth {depth}"));
            }
            CodingEvent::SubAgentCompleted {
                agent_id,
                success,
                turns_used,
                ..
            } => print_err(&format!(
                "[child] {agent_id} {} after {turns_used} turn(s)",
                if *success { "completed" } else { "failed" }
            )),
            CodingEvent::SubAgentFailed {
                agent_id, error, ..
            } => print_err(&format!("[child] {agent_id} failed: {}", error.message)),
            // Everything else is left off a terminal that is busy enough.
            // `CodingEvent` is `#[non_exhaustive]`, so this arm is required.
            _ => {}
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
    pub fn report(&self, outcome: &PromptReport, style: Style) {
        if style != Style::Text {
            return;
        }
        print_err("");
        for line in self.lines(outcome) {
            print_err(&line);
        }
    }

    /// The report's lines, in order.
    ///
    /// Turns are the root session's, as the projection keeps them; what the
    /// children did is on the `subagents` line. Tools and retries are the
    /// tree's. Tokens and cost are the report's rather than the projection's
    /// because the report bills a compaction's summary call to the prompt and
    /// no event carries that call's usage.
    fn lines(&self, outcome: &PromptReport) -> Vec<String> {
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

        let usage = outcome.usage;
        lines.push(format!(
            "tokens: {} in, {} out, {} reasoning, {} cached ({} total)",
            usage.input,
            usage.output,
            usage.reasoning,
            usage.cache_read + usage.cache_write,
            usage.total()
        ));
        lines.push(match outcome.cost_usd_micros {
            Some(cost) => format!("cost:   {}", dollars(cost)),
            None => "cost:   not reported for this model".to_owned(),
        });

        let spawned = projection.subagent_counts.spawned;
        if spawned > 0 || !projection.descendants.is_empty() {
            let (usage, cost) = projection.descendant_usage();
            let turns: u64 = projection
                .descendants
                .values()
                .map(|account| account.messages)
                .sum();
            let cost = cost.map_or_else(String::new, |cost| format!(", {}", dollars(cost)));
            lines.push(format!(
                "subagents: {spawned} spawned, {turns} turn(s), {} tokens{cost}",
                usage.total()
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
        ErrorData, ErrorKind, InputSource, LlmRetryPhase, TokenUsage,
    };
    use pebble_coding_agent::{PromptOutput, PromptTiming};
    use serde_json::json;

    use super::*;

    fn root(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_root", event, SystemTime::UNIX_EPOCH)
    }

    fn child(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_child", event, SystemTime::UNIX_EPOCH)
            .with_parent_session_id("ses_root")
    }

    fn message(input: u64, output: u64, cost: Option<u64>) -> CodingEvent {
        CodingEvent::AssistantMessage {
            text:            "ok".into(),
            model:           "model".into(),
            usage:           TokenUsage {
                input,
                output,
                ..TokenUsage::default()
            },
            cost_usd_micros: cost,
            cost_source:     None,
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
        CodingEvent::ToolCallCompleted {
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            output: json!("done"),
            metadata: pebble_agent::ToolOutputMetadata::default(),
            is_error,
            error_kind: None,
            output_bytes_observed: 0,
            output_bytes_retained: 0,
            output_bytes_omitted: 0,
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

    fn report_of(projection: &SessionProjection) -> PromptReport {
        PromptReport {
            result:            Ok(PromptOutput {
                text:          Some("done".into()),
                final_message: None,
            }),
            usage:             projection.usage,
            cost_usd_micros:   projection.cost_usd_micros,
            timing:            PromptTiming::default(),
            files_touched:     Vec::new(),
            last_file_touched: None,
            route:             "test/model".into(),
            compactions:       Vec::new(),
        }
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

        assert_eq!(summary.lines(&report_of(&projection)), [
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
        assert_eq!(summary.lines(&report_of(summary.projection())), [
            "turns:  1",
            "tools:  0 call(s), 0 failed",
            "tokens: 10 in, 5 out, 0 reasoning, 0 cached (15 total)",
            "cost:   not reported for this model",
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
        let lines = summary.lines(&report_of(summary.projection()));
        assert_eq!(
            lines.last().map(String::as_str),
            Some(format!("dropped: {dropped} event(s) this reader missed").as_str())
        );
    }
}
