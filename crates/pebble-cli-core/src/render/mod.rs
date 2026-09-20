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
//!
//! The readable lines are the `text` module's, and the closing report is
//! the `summary` module's. [`Renderer`] reads the stream and hands each
//! event to both.

#[cfg(test)]
mod fixtures;
mod summary;
mod text;

use pebble_coding_agent::CodingAgent;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;

pub use self::summary::Summary;
pub use self::text::RenderOptions;
use self::text::TextRenderer;
use crate::terminal::{print_err, print_out};

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
    style:   Style,
    json_to: JsonStream,
    text:    TextRenderer,
    summary: Summary,
}

impl Renderer {
    #[must_use]
    pub fn new(style: Style) -> Self {
        Self {
            style,
            json_to: JsonStream::default(),
            text: TextRenderer::new(RenderOptions::default()),
            summary: Summary::default(),
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
        self.text = TextRenderer::new(options);
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
                Err(RecvError::Lagged(dropped)) => self.summary.missed(dropped),
                Err(RecvError::Closed) => break,
            }
        }
        self.text.close();
        self.summary
    }

    fn render(&mut self, event: &CodingAgentEvent) {
        self.summary.apply(event);
        match self.style {
            Style::Json => match serde_json::to_string(event) {
                Ok(line) => match self.json_to {
                    JsonStream::Stderr => print_err(&line),
                    JsonStream::Stdout => print_out(&line),
                },
                Err(error) => print_err(&format!("error: rendering an event as JSON: {error}")),
            },
            Style::Text => self.text.render(event),
            Style::Quiet => {}
        }
    }
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
    use pebble_coding_agent::projection::SessionProjection;

    use super::fixtures::scripted_prompt;
    use super::*;

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
    async fn the_stream_is_folded_until_the_session_ends() {
        let events = scripted_prompt();
        let summary = summarize(&events, events.len()).await;
        let mut projection = SessionProjection::new();
        projection.apply_all(&events);
        assert_eq!(summary.projection(), &projection);
    }

    #[tokio::test]
    async fn a_lagging_reader_reports_what_it_missed() {
        let events = scripted_prompt();
        // A channel too small for the stream: the reader is told how many
        // events it lost, and the summary says so, on top of what it folded.
        let capacity = 2;
        let summary = summarize(&events, capacity).await;
        let dropped = u64::try_from(events.len() - capacity).expect("a small count");
        assert_eq!(
            summary.lines().last().map(String::as_str),
            Some(format!("dropped: {dropped} event(s) this reader missed").as_str())
        );
    }
}
