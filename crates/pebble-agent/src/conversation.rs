//! Explicit projection of committed conversation changes.

use lithos_llm::types::{Message, Response, ToolCall, ToolResult};
use serde_json::Value;

/// Projects canonical conversation commits into an embedding layer.
///
/// These callbacks run synchronously immediately after the generic agent
/// commits the same change. They are separate from lifecycle events so event
/// subscribers cannot become an accidental source of conversation state.
pub trait ConversationProjection: Send + Sync {
    /// A prompt, follow-up, or lifecycle-produced user message was committed.
    fn user_message_committed(&self, _message: &Message) {}

    /// A queued steering message was committed.
    fn steering_message_committed(&self, _message: &Message, _attribution: Option<&Value>) {}

    /// A complete assistant response was committed.
    fn assistant_message_committed(&self, _response: &Response) {}

    /// One ordered tool round was committed.
    fn tool_results_committed(
        &self,
        _calls: &[ToolCall],
        _results: &[ToolResult],
        _cancelled: bool,
    ) {
    }

    /// A lifecycle stage replaced the canonical conversation.
    fn conversation_replaced(&self, _messages: &[Message]) {}
}
