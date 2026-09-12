//! One prompt on an agent the caller built, run to its end.
//!
//! The caller assembles the agent: its client, environment, tools, and
//! options are the application's. What is the same for every application is
//! here: the events rendered as they happen, the answer written to standard
//! output, the summary after it, and the agent shut down for the reason the
//! prompt ended with, so the renderer can be joined and its summary comes
//! last.

use std::io::{self, Write as _};

use anyhow::{Context as _, Result};
use pebble_coding_agent::{
    CodingAgent, CodingInput, Error as AgentError, PromptOutput, PromptReport, ShutdownReason,
};
use tokio_util::sync::CancellationToken;

use crate::render::{JsonStream, Renderer, Style};

/// How the prompt is shown and where its answer goes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionOptions {
    pub style:        Style,
    /// Where [`Style::Json`] envelopes go.
    pub json_to:      JsonStream,
    /// Whether the final answer is written to standard output. A caller that
    /// streams JSON on standard output turns this off so the two do not mix.
    pub write_answer: bool,
}

impl Default for SessionOptions {
    /// Readable text on standard error and the answer on standard output.
    fn default() -> Self {
        Self {
            style:        Style::Text,
            json_to:      JsonStream::Stderr,
            write_answer: true,
        }
    }
}

/// Runs `input` on `agent` and shuts the agent down.
///
/// Cancelling `cancel` ends the prompt as interrupted. The report's result
/// says how the prompt ended; an error here is the harness's own: the
/// renderer could not be joined, the answer could not be written, or the
/// agent did not shut down cleanly after answering.
pub async fn run_prompt(
    mut agent: CodingAgent,
    input: impl Into<CodingInput>,
    cancel: &CancellationToken,
    options: SessionOptions,
) -> Result<PromptReport> {
    let renderer = tokio::spawn(
        Renderer::new(options.style)
            .json_to(options.json_to)
            .run(agent.subscribe()),
    );

    let report = agent.prompt_with_cancellation(input, cancel).await;
    let reason = match &report.result {
        Ok(_) => ShutdownReason::Completed,
        Err(AgentError::Interrupted(_)) => ShutdownReason::Cancelled,
        Err(_) => ShutdownReason::Error,
    };
    // Closing publishes what is queued and ends the stream, which is what
    // lets the renderer be joined here and its summary come after the answer.
    let shutdown = agent.shutdown(reason).await;
    let summary = renderer.await.context("joining the event renderer")?;

    if let Ok(outcome) = &report.result
        && options.write_answer
    {
        write_answer(outcome)?;
    }
    summary.report(&report, options.style);
    if report.result.is_ok() {
        shutdown.context("shutting the agent down")?;
    }
    Ok(report)
}

/// Writes the final answer to standard output, ending it with one newline.
fn write_answer(outcome: &PromptOutput) -> Result<()> {
    let Some(text) = outcome.text.as_deref() else {
        return Ok(());
    };
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(text.as_bytes())
        .and_then(|()| {
            if text.ends_with('\n') {
                Ok(())
            } else {
                stdout.write_all(b"\n")
            }
        })
        .context("writing the answer")
}
