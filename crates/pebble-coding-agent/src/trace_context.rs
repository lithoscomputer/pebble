//! W3C trace context on the requests a session makes.
//!
//! Each model request and each tool call is a span. Pebble sends the span as
//! a [`traceparent`](https://www.w3.org/TR/trace-context/) and the session's
//! [`baggage`](https://www.w3.org/TR/baggage/) on every model request, puts
//! both in `params._meta` of every MCP `tools/call`, and records the span's
//! IDs on the events that describe the call. A proxy that stores the caller's
//! span as its parent can then name the exact Pebble call it served.
//!
//! An application places a session in its own trace with
//! [`CodingAgentBuilder::trace_context`](crate::CodingAgentBuilder::trace_context).
//! A session without one starts a trace of its own. Subagents inherit their
//! parent session's context.

use std::fmt::{self, Write as _};
use std::sync::Arc;

use thiserror::Error;
use uuid::Uuid;

/// The `traceparent` header and `_meta` key.
pub const TRACEPARENT: &str = "traceparent";

/// The `tracestate` header and `_meta` key.
pub const TRACESTATE: &str = "tracestate";

/// The `baggage` header and `_meta` key.
pub const BAGGAGE: &str = "baggage";

/// The baggage key that carries the acting session's ID.
pub const SESSION_ID_BAGGAGE_KEY: &str = "session_id";

/// The baggage key that carries a tool call's ID, on a tool call's span.
pub const TOOL_CALL_ID_BAGGAGE_KEY: &str = "tool_call_id";

/// The longest `tracestate` the W3C specification lets a vendor send.
const MAX_TRACESTATE_LEN: usize = 512;

/// The trace a session's spans belong to, and the baggage they carry.
///
/// Build one from the caller's `traceparent` with
/// [`from_traceparent`](Self::from_traceparent), so each span Pebble starts is
/// a child of the caller's span, or with [`new`](Self::new) to start a trace
/// with no parent. Add baggage entries with
/// [`with_baggage`](Self::with_baggage). Pebble adds `session_id` to every
/// span's baggage and `tool_call_id` to a tool call's, so those two keys are
/// refused here.
#[derive(Clone, PartialEq, Eq)]
pub struct TraceContext {
    trace_id:       [u8; 16],
    parent_span_id: Option<[u8; 8]>,
    flags:          u8,
    tracestate:     Option<String>,
    baggage:        Vec<(String, String)>,
}

impl TraceContext {
    /// A new trace with a random ID, no parent span, and the sampled flag set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trace_id:       random_id(),
            parent_span_id: None,
            flags:          1,
            tracestate:     None,
            baggage:        Vec::new(),
        }
    }

    /// The trace of the caller whose span `traceparent` names.
    ///
    /// Accepts version `00` exactly, and a later version by its first four
    /// fields, as the specification asks. Spans Pebble starts keep the
    /// caller's trace ID and flags and name the caller's span as their parent.
    ///
    /// # Errors
    ///
    /// Returns [`TraceContextError::InvalidTraceparent`] when the value is not
    /// a valid `traceparent`, including an all-zero trace or span ID.
    pub fn from_traceparent(traceparent: &str) -> Result<Self, TraceContextError> {
        let invalid = || TraceContextError::InvalidTraceparent(traceparent.to_owned());
        let value = traceparent.trim();
        let mut fields = value.split('-');
        let (Some(version), Some(trace_id), Some(span_id), Some(flags)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            return Err(invalid());
        };
        let version: [u8; 1] = parse_hex(version).ok_or_else(invalid)?;
        // Version 00 has exactly four fields; ff is forbidden; a later version
        // may append fields after a dash.
        if version[0] == 0xff || (version[0] == 0 && fields.next().is_some()) {
            return Err(invalid());
        }
        let trace_id: [u8; 16] = parse_hex(trace_id).ok_or_else(invalid)?;
        let span_id: [u8; 8] = parse_hex(span_id).ok_or_else(invalid)?;
        let [flags] = parse_hex::<1>(flags).ok_or_else(invalid)?;
        if trace_id == [0; 16] || span_id == [0; 8] {
            return Err(invalid());
        }
        Ok(Self {
            trace_id,
            parent_span_id: Some(span_id),
            flags,
            tracestate: None,
            baggage: Vec::new(),
        })
    }

    /// Sends `tracestate` unchanged with every span's `traceparent`.
    ///
    /// # Errors
    ///
    /// Returns [`TraceContextError::InvalidTracestate`] for a value longer
    /// than 512 bytes or one that is not printable ASCII.
    pub fn with_tracestate(mut self, tracestate: &str) -> Result<Self, TraceContextError> {
        let value = tracestate.trim();
        if value.is_empty()
            || value.len() > MAX_TRACESTATE_LEN
            || !value.bytes().all(|byte| (b' '..=b'~').contains(&byte))
        {
            return Err(TraceContextError::InvalidTracestate(tracestate.to_owned()));
        }
        self.tracestate = Some(value.to_owned());
        Ok(self)
    }

    /// Adds `key=value` to every span's baggage, replacing an earlier entry
    /// with the same key. The value is percent-encoded on the wire.
    ///
    /// # Errors
    ///
    /// Returns [`TraceContextError::InvalidBaggageKey`] for a key that is not
    /// an HTTP token, and [`TraceContextError::ReservedBaggageKey`] for
    /// `session_id` and `tool_call_id`, which Pebble sets itself.
    pub fn with_baggage(
        mut self,
        key: &str,
        value: impl Into<String>,
    ) -> Result<Self, TraceContextError> {
        if key == SESSION_ID_BAGGAGE_KEY || key == TOOL_CALL_ID_BAGGAGE_KEY {
            return Err(TraceContextError::ReservedBaggageKey(key.to_owned()));
        }
        if key.is_empty() || !key.bytes().all(is_token_byte) {
            return Err(TraceContextError::InvalidBaggageKey(key.to_owned()));
        }
        let value = value.into();
        match self
            .baggage
            .iter_mut()
            .find(|(existing, _)| existing == key)
        {
            Some(entry) => entry.1 = value,
            None => self.baggage.push((key.to_owned(), value)),
        }
        Ok(self)
    }

    /// The trace ID, as 32 lowercase hex digits.
    #[must_use]
    pub fn trace_id(&self) -> String {
        hex(&self.trace_id)
    }

    /// The caller's span, as 16 lowercase hex digits, when there is one.
    #[must_use]
    pub fn parent_span_id(&self) -> Option<String> {
        self.parent_span_id.as_ref().map(|id| hex(id))
    }

    /// The baggage entries the application added, in the order added.
    #[must_use]
    pub fn baggage(&self) -> &[(String, String)] {
        &self.baggage
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TraceContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TraceContext")
            .field("trace_id", &self.trace_id())
            .field("parent_span_id", &self.parent_span_id())
            .field("flags", &self.flags)
            .field("tracestate", &self.tracestate)
            .field("baggage", &self.baggage)
            .finish()
    }
}

/// Why a [`TraceContext`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TraceContextError {
    /// The value is not a valid W3C `traceparent`.
    #[error("invalid traceparent: {0:?}")]
    InvalidTraceparent(String),
    /// The value is not a sendable W3C `tracestate`.
    #[error("invalid tracestate: {0:?}")]
    InvalidTracestate(String),
    /// The baggage key is not an HTTP token.
    #[error("invalid baggage key: {0:?}")]
    InvalidBaggageKey(String),
    /// Pebble sets this baggage key itself.
    #[error("baggage key {0:?} is set by Pebble")]
    ReservedBaggageKey(String),
}

/// A session's trace context with the session it names.
#[derive(Debug)]
pub(crate) struct SessionTrace {
    context:    TraceContext,
    session_id: String,
}

impl SessionTrace {
    pub(crate) fn new(context: TraceContext, session_id: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            context,
            session_id: session_id.into(),
        })
    }

    /// Starts a span for a model request.
    pub(crate) fn model_span(self: &Arc<Self>) -> Span {
        self.span(None)
    }

    /// Starts a span for the tool call `tool_call_id`.
    pub(crate) fn tool_span(self: &Arc<Self>, tool_call_id: &str) -> Span {
        self.span(Some(tool_call_id.to_owned()))
    }

    fn span(self: &Arc<Self>, tool_call_id: Option<String>) -> Span {
        Span {
            session: Arc::clone(self),
            id: random_id(),
            tool_call_id,
        }
    }
}

/// One model request or tool call, as a child of the session's parent span.
#[derive(Clone, Debug)]
pub(crate) struct Span {
    session:      Arc<SessionTrace>,
    id:           [u8; 8],
    tool_call_id: Option<String>,
}

impl Span {
    /// The trace ID, as 32 lowercase hex digits.
    pub(crate) fn trace_id(&self) -> String {
        self.session.context.trace_id()
    }

    /// This span's ID, as 16 lowercase hex digits.
    pub(crate) fn span_id(&self) -> String {
        hex(&self.id)
    }

    /// The caller's span this span is a child of, when there is one.
    pub(crate) fn parent_span_id(&self) -> Option<String> {
        self.session.context.parent_span_id()
    }

    /// The trace context entries to send for this span, by name: the
    /// `traceparent`, the `tracestate` when the application gave one, and
    /// the `baggage`.
    pub(crate) fn entries(&self) -> Vec<(&'static str, String)> {
        let context = &self.session.context;
        let traceparent = format!(
            "00-{}-{}-{:02x}",
            context.trace_id(),
            self.span_id(),
            context.flags
        );
        let mut entries = vec![(TRACEPARENT, traceparent)];
        if let Some(tracestate) = &context.tracestate {
            entries.push((TRACESTATE, tracestate.clone()));
        }
        entries.push((BAGGAGE, self.baggage()));
        entries
    }

    fn baggage(&self) -> String {
        let pebble = [
            Some((SESSION_ID_BAGGAGE_KEY, self.session.session_id.as_str())),
            self.tool_call_id
                .as_deref()
                .map(|id| (TOOL_CALL_ID_BAGGAGE_KEY, id)),
        ];
        let mut baggage = String::new();
        let application = self.session.context.baggage.iter();
        for (key, value) in application
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .chain(pebble.into_iter().flatten())
        {
            if !baggage.is_empty() {
                baggage.push(',');
            }
            baggage.push_str(key);
            baggage.push('=');
            percent_encode(value, &mut baggage);
        }
        baggage
    }
}

fn random_id<const N: usize>() -> [u8; N] {
    loop {
        let mut id = [0; N];
        let random = Uuid::new_v4().into_bytes();
        id.copy_from_slice(&random[..N]);
        if id != [0; N] {
            return id;
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Parses exactly `2 * N` lowercase hex digits.
fn parse_hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    if text.len() != N * 2
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut out = [0; N];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// An RFC 7230 `tchar`, which is what a baggage key is.
fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

/// Writes `value` with every byte outside the W3C `baggage-octet` range, and
/// `%` itself, percent-encoded.
fn percent_encode(value: &str, out: &mut String) {
    for byte in value.bytes() {
        let plain = matches!(byte, 0x21 | 0x23..=0x2b | 0x2d..=0x3a | 0x3c..=0x5b | 0x5d..=0x7e)
            && byte != b'%';
        if plain {
            out.push(char::from(byte));
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn a_traceparent_is_parsed_into_the_callers_trace_and_span() {
        let context = TraceContext::from_traceparent(PARENT).expect("valid");

        assert_eq!(context.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(
            context.parent_span_id().as_deref(),
            Some("00f067aa0ba902b7")
        );
    }

    #[test]
    fn malformed_traceparents_are_refused() {
        for value in [
            "",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
        ] {
            assert!(
                TraceContext::from_traceparent(value).is_err(),
                "{value:?} should be refused"
            );
        }
        // A later version may append fields.
        assert!(TraceContext::from_traceparent(&format!("01{}-what", &PARENT[2..])).is_ok());
    }

    #[test]
    fn a_span_is_a_child_of_the_callers_span_in_the_same_trace() {
        let context = TraceContext::from_traceparent(PARENT)
            .and_then(|context| context.with_tracestate("vendor=opaque"))
            .and_then(|context| context.with_baggage("run_id", "run 1/α"))
            .and_then(|context| context.with_baggage("run_step", "3"))
            .expect("valid");
        let session = SessionTrace::new(context, "ses_1");
        let model = session.model_span();
        let tool = session.tool_span("call,1");

        assert_ne!(model.span_id(), tool.span_id());
        assert_ne!(model.span_id(), "00f067aa0ba902b7");
        assert_eq!(model.entries(), vec![
            (
                TRACEPARENT,
                format!("00-4bf92f3577b34da6a3ce929d0e0e4736-{}-01", model.span_id())
            ),
            (TRACESTATE, "vendor=opaque".to_owned()),
            (
                BAGGAGE,
                "run_id=run%201/%CE%B1,run_step=3,session_id=ses_1".to_owned()
            ),
        ]);
        assert_eq!(
            tool.entries()[2].1,
            "run_id=run%201/%CE%B1,run_step=3,session_id=ses_1,tool_call_id=call%2C1"
        );
    }

    #[test]
    fn pebbles_baggage_keys_are_reserved_and_keys_must_be_tokens() {
        let context = TraceContext::new();

        assert_eq!(
            context.clone().with_baggage("session_id", "x"),
            Err(TraceContextError::ReservedBaggageKey("session_id".into()))
        );
        assert_eq!(
            context.clone().with_baggage("tool_call_id", "x"),
            Err(TraceContextError::ReservedBaggageKey("tool_call_id".into()))
        );
        assert_eq!(
            context.with_baggage("run id", "x"),
            Err(TraceContextError::InvalidBaggageKey("run id".into()))
        );
    }

    #[test]
    fn a_new_trace_has_no_parent_and_a_fresh_id() {
        let first = TraceContext::new();
        let second = TraceContext::new();

        assert_eq!(first.parent_span_id(), None);
        assert_ne!(first.trace_id(), second.trace_id());
        assert_eq!(first.trace_id().len(), 32);
    }
}
