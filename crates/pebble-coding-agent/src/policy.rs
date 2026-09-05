//! Typed policy hooks for applications embedding a coding agent.
//!
//! Hooks run on the caller's Tokio runtime and may be called concurrently by
//! different agents. They must honor cancellation and be safe to drop. They
//! are runtime services: applications install them again when resuming a
//! record.

use async_trait::async_trait;
use lithos_llm::types::{Message as LlmMessage, Request};
use pebble_agent::{LifecycleError, ToolCatalog, TurnContext};
use tokio_util::sync::CancellationToken;

use crate::types::{Message, TokenUsage};
use crate::{CompactionReason, SessionScope};

/// The request view available to a context policy after compaction and
/// discovery.
pub struct ContextPreparation<'a> {
    /// Identity of the agent making this request.
    pub session: &'a SessionScope,
    /// Committed messages and the model-turn number.
    pub turn:    TurnContext<'a>,
    /// The tools visible after permission middleware ran.
    pub tools:   &'a ToolCatalog,
}

/// Prepares the messages for one model request without rewriting durable
/// history.
#[async_trait]
pub trait ContextPolicy: Send + Sync {
    /// Return `None` to use the default view, or a complete replacement view.
    ///
    /// The profile's system prompt remains in place. Pebble validates tool-call
    /// pairing before sending a replacement. An error ends the prompt before
    /// another model call; cancellation leaves the agent reusable.
    async fn prepare(
        &self,
        context: ContextPreparation<'_>,
        cancel: &CancellationToken,
    ) -> Result<Option<Vec<LlmMessage>>, LifecycleError>;
}

/// Inputs to an application-defined summarizer.
pub struct CompactionPreparation<'a> {
    /// Identity of the session being compacted.
    pub session_id:        &'a str,
    /// Why compaction started.
    pub reason:            CompactionReason,
    /// Complete typed turns selected for summarization.
    pub messages:          &'a [Message],
    /// Recent turns Pebble will retain under its normal history policy.
    /// Compaction clears stale usage estimates and non-replayable provider
    /// data.
    pub retained_messages: &'a [Message],
    /// Pebble's default summary request, including custom instructions.
    pub default_request:   &'a Request,
}

/// A summary and the usage the application spent producing it.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionSummary {
    /// Handoff text. Pebble rejects empty summaries and bounds the stored text.
    pub text:            String,
    /// Token usage of the summarization operation.
    pub usage:           TokenUsage,
    /// Cost in USD micros, when known.
    pub cost_usd_micros: Option<u64>,
}

/// Supplies summary generation while Pebble owns history replacement and
/// events.
#[async_trait]
pub trait CompactionPolicy: Send + Sync {
    /// Generates a summary. Failures and cancellation leave history unchanged.
    ///
    /// The application may use a separate client or model. It must report its
    /// usage and must not mutate the agent or execute workflow routing here.
    async fn summarize(
        &self,
        context: CompactionPreparation<'_>,
        cancel: &CancellationToken,
    ) -> Result<CompactionSummary, LifecycleError>;
}
