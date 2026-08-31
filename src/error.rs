//! Pebble's runtime error type and the serializable projection that rides the
//! event stream.
//!
//! The runtime [`Error`] keeps full fidelity: it wraps
//! [`lithos_llm::types::Error`], which is neither `Clone` nor serializable
//! because it owns a live source chain. [`ErrorData`] is the projection that
//! events carry instead — cloneable, serializable, and free of raw provider
//! payloads.

use std::error::Error as StdError;
use std::fmt;
use std::result::Result as StdResult;

use lithos_llm::types::{
    Error as LlmError, ErrorKind as LlmErrorKind, RequestBuildError, RetryClassification,
};
use serde::{Deserialize, Serialize};

use crate::event::EventSinkError;

/// Why a run was interrupted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InterruptReason {
    /// The session's wall-clock budget expired.
    WallClockTimeout,
    /// A caller cancelled the run.
    Cancelled,
}

impl InterruptReason {
    /// The human-readable rendering used in error messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WallClockTimeout => "wall clock timeout",
            Self::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for InterruptReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A failure raised while compacting conversation history.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CompactionError {
    /// The summarization request could not be built, which a session with an
    /// unusable model selector runs into before any call is made.
    #[error("summary request could not be built: {0}")]
    Request(#[source] RequestBuildError),

    /// The summarization request to the model failed.
    #[error("summary request failed: {0}")]
    Llm(#[source] LlmError),

    /// The model returned a summary that was empty once trimmed. History is
    /// left intact rather than replaced with nothing.
    #[error(
        "generated summary was empty after trimming; refused to replace \
         {summarized_turn_count} turns and left history intact"
    )]
    EmptySummary {
        /// How many turns the refused compaction would have replaced.
        summarized_turn_count: usize,
    },
}

/// A pebble runtime failure.
///
/// Variants are branch-oriented: callers recover differently from a closed
/// session, an interrupt, and a model failure. The underlying
/// [`lithos_llm::types::Error`] is retained as a source for logging;
/// [`ErrorData`] is the shape that reaches consumers over the event stream.
///
/// A variant that keeps a source also renders that source in its own message.
/// The projection carries [`ErrorData::message`] as a single self-contained
/// line, which is what most consumers display, and the price of that is a
/// repeated innermost message for anyone who walks the chain as well.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The model layer failed.
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// History compaction failed.
    #[error("context compaction failed: {0}")]
    Compaction(#[from] CompactionError),

    /// The session has been shut down.
    #[error("session is closed")]
    SessionClosed,

    /// The session was asked to do something its current state forbids.
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// A tool call failed in a way that stops the run.
    #[error("tool execution error: {0}")]
    ToolExecution(String),

    /// The run was interrupted before it finished.
    #[error("interrupted: {0}")]
    Interrupted(InterruptReason),

    /// The configured [`crate::EventSink`] refused an event.
    ///
    /// The event stream is the durable record of a run, so a session that
    /// cannot record what it did stops instead of continuing untracked.
    #[error("event sink failed: {0}")]
    EventSink(#[from] EventSinkError),
}

impl Error {
    /// The stable category of this failure.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Llm(_) => ErrorKind::Llm,
            Self::Compaction(_) => ErrorKind::Compaction,
            Self::SessionClosed => ErrorKind::SessionClosed,
            Self::InvalidState(_) => ErrorKind::InvalidState,
            Self::ToolExecution(_) => ErrorKind::ToolExecution,
            Self::Interrupted(_) => ErrorKind::Interrupted,
            Self::EventSink(_) => ErrorKind::EventSink,
        }
    }

    /// The model-layer failure underlying this error, when there is one.
    #[must_use]
    pub fn llm_source(&self) -> Option<&LlmError> {
        match self {
            Self::Llm(error) | Self::Compaction(CompactionError::Llm(error)) => Some(error),
            Self::Compaction(
                CompactionError::EmptySummary { .. } | CompactionError::Request(_),
            )
            | Self::SessionClosed
            | Self::InvalidState(_)
            | Self::ToolExecution(_)
            | Self::Interrupted(_)
            | Self::EventSink(_) => None,
        }
    }
}

/// The result of a fallible pebble operation.
pub type Result<T> = StdResult<T, Error>;

/// The stable category of a projected pebble failure.
///
/// Mirrors the [`Error`] variants so a consumer can branch on a serialized
/// failure the same way it would branch on the runtime error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorKind {
    /// The model layer failed.
    Llm,
    /// History compaction failed.
    Compaction,
    /// The session has been shut down.
    SessionClosed,
    /// The session was asked to do something its current state forbids.
    InvalidState,
    /// A tool call failed in a way that stopped the run.
    ToolExecution,
    /// The run was interrupted before it finished.
    Interrupted,
    /// The configured event sink refused an event.
    EventSink,
}

/// A cloneable, serializable projection of an [`Error`].
///
/// This is what every error-carrying event holds. It keeps the facts a
/// consumer can act on — category, safe message, retry advice, provider and
/// model context, and the rendered source chain — and deliberately drops the
/// raw provider response body that [`lithos_llm::types::ErrorData`] would
/// carry. Pebble never puts an unparsed provider payload on its event stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ErrorData {
    /// The stable category of the failure.
    pub kind: ErrorKind,

    /// The failure rendered for a human, safe to display.
    ///
    /// The whole error, cause included, so one line tells the whole story.
    /// A consumer that also renders `source_chain` will therefore see the
    /// innermost failure twice; show `message` alone, or the chain under a
    /// heading of its own.
    pub message: String,

    /// The model-layer category, when a model call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_kind: Option<LlmErrorKind>,

    /// Whether repeating the same model call is safe, when a model call
    /// failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryClassification>,

    /// The provider that produced the failure, when one was selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,

    /// The model that produced the failure, when one was selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The HTTP status, when the failure came from an HTTP response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,

    /// The provider's own error code, as reported on the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,

    /// The provider's advised wait in milliseconds, whatever the kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_retry_after_millis: Option<u64>,

    /// The `Display` text of each cause below `message`, outermost first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_chain: Vec<String>,
}

impl ErrorData {
    /// Builds a projection with only a category and a message.
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            llm_kind: None,
            retry: None,
            provider: None,
            model: None,
            status: None,
            provider_code: None,
            provider_retry_after_millis: None,
            source_chain: Vec::new(),
        }
    }

    /// Records the model this failure was produced for.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Records the provider this failure was produced for.
    ///
    /// A projection built from a model-layer error already carries the
    /// provider the call resolved to; this fills it in for failures raised
    /// before or outside a provider call.
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    fn fill_from_llm(&mut self, error: &LlmError) {
        self.llm_kind = Some(error.kind());
        self.retry = Some(error.retry_classification());
        self.provider = error
            .provider()
            .map(|provider| provider.as_str().to_owned());
        self.status = error.status();
        self.provider_code = error.provider_code().map(ToOwned::to_owned);
        self.provider_retry_after_millis = error
            .provider_retry_after()
            .map(|delay| u64::try_from(delay.as_millis()).unwrap_or(u64::MAX));
    }
}

impl From<&Error> for ErrorData {
    fn from(error: &Error) -> Self {
        let mut data = Self::new(error.kind(), error.to_string());
        data.source_chain = source_chain(error);
        if let Some(llm) = error.llm_source() {
            data.fill_from_llm(llm);
        }
        data
    }
}

impl From<&LlmError> for ErrorData {
    fn from(error: &LlmError) -> Self {
        let mut data = Self::new(ErrorKind::Llm, error.message().to_owned());
        data.source_chain = source_chain(error);
        data.fill_from_llm(error);
        data
    }
}

/// Renders every cause below `error`, outermost first.
fn source_chain(error: &(dyn StdError + 'static)) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current = error.source();
    while let Some(cause) = current {
        chain.push(cause.to_string());
        current = cause.source();
    }
    chain
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use lithos_llm::catalog::ProviderId;

    use super::*;

    fn network_error() -> LlmError {
        LlmError::new(LlmErrorKind::Network, "connection refused")
    }

    fn rate_limited() -> LlmError {
        LlmError::new(LlmErrorKind::RateLimit, "too fast")
            .with_provider(ProviderId::from("openai"))
            .with_status(429)
            .with_provider_code("rate_limit_exceeded")
            .with_retry(RetryClassification::after(Duration::from_secs(2)))
            .with_provider_retry_after(Duration::from_secs(2))
    }

    #[test]
    fn agent_error_from_llm_error() {
        let error = Error::from(network_error());
        assert!(matches!(error, Error::Llm(_)));
        assert!(error.to_string().contains("connection refused"));
    }

    #[test]
    fn compaction_error_preserves_llm_source_chain() {
        let error = Error::Compaction(CompactionError::Llm(network_error()));

        let chain = source_chain(&error);

        assert!(
            chain.len() >= 2,
            "expected the compaction and LLM errors below the top message: {chain:?}"
        );
        assert!(
            chain
                .last()
                .is_some_and(|cause| cause.contains("connection refused")),
            "underlying LLM failure missing from source chain: {chain:?}"
        );
    }

    #[test]
    fn empty_compaction_summary_display() {
        let error = Error::Compaction(CompactionError::EmptySummary {
            summarized_turn_count: 3,
        });
        assert_eq!(
            error.to_string(),
            "context compaction failed: generated summary was empty after trimming; \
             refused to replace 3 turns and left history intact"
        );
    }

    #[test]
    fn session_closed_display() {
        assert_eq!(Error::SessionClosed.to_string(), "session is closed");
    }

    #[test]
    fn invalid_state_display() {
        assert_eq!(
            Error::InvalidState("bad state".into()).to_string(),
            "invalid state: bad state"
        );
    }

    #[test]
    fn tool_execution_display() {
        assert_eq!(
            Error::ToolExecution("command failed".into()).to_string(),
            "tool execution error: command failed"
        );
    }

    #[test]
    fn interrupted_display() {
        assert_eq!(
            Error::Interrupted(InterruptReason::Cancelled).to_string(),
            "interrupted: cancelled"
        );
    }

    #[test]
    fn interrupted_wall_clock_timeout_display() {
        assert_eq!(
            Error::Interrupted(InterruptReason::WallClockTimeout).to_string(),
            "interrupted: wall clock timeout"
        );
    }

    #[test]
    fn every_variant_maps_to_its_kind() {
        let cases = [
            (Error::from(network_error()), ErrorKind::Llm),
            (
                Error::Compaction(CompactionError::EmptySummary {
                    summarized_turn_count: 1,
                }),
                ErrorKind::Compaction,
            ),
            (Error::SessionClosed, ErrorKind::SessionClosed),
            (Error::InvalidState("x".into()), ErrorKind::InvalidState),
            (Error::ToolExecution("x".into()), ErrorKind::ToolExecution),
            (
                Error::Interrupted(InterruptReason::Cancelled),
                ErrorKind::Interrupted,
            ),
        ];
        for (error, kind) in cases {
            assert_eq!(error.kind(), kind);
        }
    }

    #[test]
    fn projection_of_an_llm_error_carries_provider_context() {
        let data = ErrorData::from(&Error::from(rate_limited()));

        assert_eq!(data.kind, ErrorKind::Llm);
        assert_eq!(data.llm_kind, Some(LlmErrorKind::RateLimit));
        assert_eq!(data.provider.as_deref(), Some("openai"));
        assert_eq!(data.status, Some(429));
        assert_eq!(data.provider_code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(data.provider_retry_after_millis, Some(2_000));
        assert_eq!(
            data.retry,
            Some(RetryClassification::After { millis: 2_000 })
        );
        assert_eq!(data.source_chain, vec!["too fast".to_owned()]);
    }

    #[test]
    fn projection_of_a_compaction_error_keeps_the_llm_details() {
        let data = ErrorData::from(&Error::Compaction(CompactionError::Llm(rate_limited())));

        assert_eq!(data.kind, ErrorKind::Compaction);
        assert_eq!(data.llm_kind, Some(LlmErrorKind::RateLimit));
        assert_eq!(data.provider.as_deref(), Some("openai"));
        assert!(data.message.starts_with("context compaction failed:"));
    }

    #[test]
    fn projection_of_a_pebble_error_has_no_llm_context() {
        let data = ErrorData::from(&Error::ToolExecution("command failed".into()));

        assert_eq!(data.kind, ErrorKind::ToolExecution);
        assert_eq!(data.message, "tool execution error: command failed");
        assert_eq!(data.llm_kind, None);
        assert_eq!(data.retry, None);
        assert!(data.source_chain.is_empty());
    }

    #[test]
    fn projection_built_straight_from_an_llm_error_uses_the_provider_message() {
        let data = ErrorData::from(&rate_limited());

        assert_eq!(data.kind, ErrorKind::Llm);
        assert_eq!(data.message, "too fast");
        assert_eq!(data.llm_kind, Some(LlmErrorKind::RateLimit));
    }

    #[test]
    fn projection_never_carries_a_raw_provider_payload() {
        let error = LlmError::new(LlmErrorKind::Provider, "upstream failed")
            .with_raw_data(serde_json::json!({"secret": "leak me"}));
        let data = ErrorData::from(&error);

        let json = serde_json::to_string(&data).expect("projection serializes");
        assert!(
            !json.contains("leak me"),
            "raw provider payload leaked: {json}"
        );
        assert!(
            !json.contains("raw_data"),
            "raw payload field present: {json}"
        );
    }

    #[test]
    fn projection_round_trips_and_omits_absent_context() {
        let data = ErrorData::from(&Error::SessionClosed).with_model("claude-sonnet-5");

        let json = serde_json::to_string(&data).expect("projection serializes");
        assert!(!json.contains("provider_code"), "{json}");
        assert!(!json.contains("source_chain"), "{json}");
        assert!(json.contains(r#""model":"claude-sonnet-5""#), "{json}");

        let restored: ErrorData = serde_json::from_str(&json).expect("projection parses");
        assert_eq!(restored, data);
    }

    #[test]
    fn projection_kind_is_snake_case_on_the_wire() {
        let data = ErrorData::from(&Error::ToolExecution("x".into()));
        let value = serde_json::to_value(&data).expect("projection serializes");
        assert_eq!(value["kind"], "tool_execution");
    }

    #[test]
    fn interrupt_reason_is_snake_case_on_the_wire() {
        let value =
            serde_json::to_value(InterruptReason::WallClockTimeout).expect("reason serializes");
        assert_eq!(value, serde_json::json!("wall_clock_timeout"));
    }
}
