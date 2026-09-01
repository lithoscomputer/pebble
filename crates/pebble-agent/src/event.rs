//! The small lifecycle event stream emitted by a generic agent.

use std::sync::Arc;

use lithos_llm::types::{ErrorKind, Message, Response, ToolCall, ToolResult};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// The first model output observed in one stream attempt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FirstOutputKind {
    /// Assistant text.
    Text,
    /// Assistant reasoning.
    Reasoning,
    /// A tool call.
    ToolCall,
}

/// One observable step in an agent prompt.
///
/// These events describe the generic conversation lifecycle. A coding-agent
/// layer can wrap them in its own event type and add resource or environment
/// events.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AgentEvent {
    /// One prompt, including queued follow-up input, started.
    PromptStarted,
    /// A model turn started.
    TurnStarted {
        /// The zero-based turn number in this prompt.
        turn: usize,
    },
    /// A user message was committed to history.
    UserMessage {
        /// The committed message.
        message: Message,
    },
    /// Steering was committed before the next model turn.
    SteeringMessage {
        /// The committed message.
        message: Message,
    },
    /// A model request is about to open.
    ModelRequestStarted {
        /// The requested model selector.
        model: String,
    },
    /// A stream attempt produced its first output.
    FirstOutput {
        /// The kind of output.
        kind: FirstOutputKind,
    },
    /// A fragment of assistant text.
    TextDelta {
        /// The fragment.
        delta: String,
    },
    /// A fragment of assistant reasoning.
    ReasoningDelta {
        /// The fragment.
        delta: String,
    },
    /// Visible output from a failed attempt must be replaced.
    OutputReplaced,
    /// A failed response stream will be opened again.
    TurnReplay {
        /// The failed attempt, counted from one.
        failed_attempt: u32,
        /// The wait before the next attempt.
        delay_seconds:  f64,
        /// The model-layer error category.
        error_kind:     ErrorKind,
    },
    /// A complete assistant response was committed.
    AssistantMessage {
        /// The committed response.
        response: Response,
    },
    /// A tool call started.
    ToolStarted {
        /// The requested call.
        call: ToolCall,
    },
    /// A tool emitted incremental output for an observer.
    ToolOutputDelta {
        /// The provider's call identifier.
        tool_call_id: String,
        /// The fragment.
        delta:        String,
    },
    /// A tool call completed and its result was committed.
    ToolCompleted {
        /// The completed result.
        result: ToolResult,
    },
    /// A round was interrupted so queued steering can be applied.
    TurnInterrupted,
    /// The prompt and all queued follow-up input completed.
    PromptCompleted {
        /// The final response.
        response: Response,
    },
    /// The active prompt was aborted.
    PromptAborted,
    /// The agent was closed.
    AgentClosed,
}

/// Projects generic lifecycle events into an embedding layer's event model.
///
/// Projection is synchronous and ordered. It runs before the event is sent to
/// generic live subscribers. Implementations should enqueue durable work and
/// return promptly.
pub trait EventProjection: Send + Sync {
    /// Projects one event.
    fn project(&self, event: &AgentEvent);
}

impl<F> EventProjection for F
where
    F: Fn(&AgentEvent) + Send + Sync,
{
    fn project(&self, event: &AgentEvent) {
        self(event);
    }
}

#[derive(Clone)]
pub(crate) struct EventHub {
    live:       broadcast::Sender<AgentEvent>,
    projection: Option<Arc<dyn EventProjection>>,
}

impl EventHub {
    pub(crate) fn new(capacity: usize, projection: Option<Arc<dyn EventProjection>>) -> Self {
        let (live, _) = broadcast::channel(capacity);
        Self { live, projection }
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.live.subscribe()
    }

    pub(crate) fn emit(&self, event: AgentEvent) {
        if let Some(projection) = &self.projection {
            projection.project(&event);
        }
        let _ = self.live.send(event);
    }
}
