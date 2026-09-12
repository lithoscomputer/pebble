//! Rendering the event stream to standard error.

use std::collections::BTreeMap;
use std::io::{self, Write as _};

use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
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
#[derive(Debug, Default)]
pub struct Summary {
    tools:    BTreeMap<String, usize>,
    failures: usize,
    turns:    usize,
    retries:  usize,
    dropped:  u64,
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
        self.summary.observe(&event.event);
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
    fn observe(&mut self, event: &CodingEvent) {
        match event {
            CodingEvent::AssistantMessage { .. } => self.turns += 1,
            CodingEvent::ToolCallCompleted {
                tool_name,
                is_error,
                ..
            } => {
                *self.tools.entry(tool_name.clone()).or_default() += 1;
                if *is_error {
                    self.failures += 1;
                }
            }
            CodingEvent::LlmRetry { .. } => self.retries += 1,
            _ => {}
        }
    }

    /// Prints what the prompt used, after the answer.
    pub fn report(&self, outcome: &PromptReport, style: Style) {
        if style != Style::Text {
            return;
        }
        let calls: usize = self.tools.values().sum();
        let named = self
            .tools
            .iter()
            .map(|(name, count)| format!("{name} x{count}"))
            .collect::<Vec<_>>()
            .join(", ");
        let usage = outcome.usage;
        print_err("");
        print_err(&format!("turns:  {}", self.turns));
        print_err(&format!(
            "tools:  {calls} call(s), {} failed",
            self.failures
        ));
        if !named.is_empty() {
            print_err(&format!("        {named}"));
        }
        print_err(&format!(
            "tokens: {} in, {} out, {} reasoning, {} cached ({} total)",
            usage.input,
            usage.output,
            usage.reasoning,
            usage.cache_read + usage.cache_write,
            usage.total()
        ));
        match outcome.cost_usd_micros {
            Some(cost) => print_err(&format!("cost:   {}", dollars(cost))),
            None => print_err("cost:   not reported for this model"),
        }
        if self.retries > 0 {
            print_err(&format!("retries: {}", self.retries));
        }
        if self.dropped > 0 {
            print_err(&format!(
                "dropped: {} event(s) this reader missed",
                self.dropped
            ));
        }
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
