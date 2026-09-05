//! Application-owned storage for output that does not fit in a model request.
//!
//! References are opaque. Storage may live outside the workspace or on another
//! machine. The application owns authorization, retention, quotas, redaction,
//! and cleanup of unfinished writes. Reinstall the store when resuming a
//! session.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use pebble_agent::ToolArtifact;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::SessionScope;
use crate::environment::ExecOutcome;
use crate::tool::{RegisteredTool, ToolError};
use crate::types::ToolSource;

/// Maximum bytes returned by one `read_tool_output` call.
pub const MAX_OUTPUT_READ_BYTES: usize = 16 * 1024;

/// A storage operation must finish within this bound, including finalization
/// after a process was cancelled. A dropped operation must not publish a
/// partial object as complete.
pub(crate) const OUTPUT_STORE_TIMEOUT: Duration = Duration::from_secs(5);

/// The byte stream being captured. Ordering is guaranteed within each stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputStream {
    /// Standard output, or the merged output when streams cannot be separated.
    Stdout,
    /// Standard error. Empty when the environment merges streams.
    Stderr,
    /// Text produced by a tool other than a process.
    Result,
}

/// An unfinished capture. Implementations must support concurrent writes to
/// different streams and keep each stream's byte order.
#[async_trait]
pub trait ToolOutputWriter: Send + Sync {
    /// Stores raw bytes before preview truncation. The application applies any
    /// storage redaction policy here; Pebble's event redactor does not run
    /// here. Implementations must be safe to drop while pending.
    async fn append(&self, stream: OutputStream, bytes: &[u8]) -> Result<(), ToolError>;

    /// Commits immutable objects and returns references accepted by the store's
    /// `read` method. Called once after all writes. A cancelled command's
    /// capture contains only the bytes it produced before stopping. Return
    /// references only for committed objects; failed or abandoned captures
    /// need cleanup.
    async fn finish(&self) -> Result<Vec<ToolArtifact>, ToolError>;
}

/// A page of bytes from a stored output. Offsets count bytes, not characters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputPage {
    /// At most the requested number of bytes, starting at the requested offset.
    pub bytes:       Vec<u8>,
    /// Size of the immutable object. An offset past this size returns no bytes.
    pub total_bytes: u64,
}

/// Storage supplied by a native embedding application.
#[async_trait]
pub trait ToolOutputStore: Send + Sync {
    /// Begins a capture attributed to this session and model-native call ID.
    /// Pebble bounds and cancels this future; it must be safe to drop.
    async fn start(
        &self,
        session: &SessionScope,
        tool_call_id: &str,
    ) -> Result<Arc<dyn ToolOutputWriter>, ToolError>;

    /// Reads an immutable output after checking that `session` may access it.
    /// Never trust a model-supplied reference as authorization. Pebble bounds
    /// and cancels this future; it must be safe to drop. Returning more than
    /// `limit` bytes or inconsistent sizes is an error.
    async fn read(
        &self,
        session: &SessionScope,
        reference: &str,
        offset: u64,
        limit: usize,
    ) -> Result<OutputPage, ToolError>;
}

/// Checks that an environment forwards the bytes it reports observing.
pub(crate) struct CountedWriter {
    inner:  Arc<dyn ToolOutputWriter>,
    stdout: AtomicU64,
    stderr: AtomicU64,
}

impl CountedWriter {
    pub(crate) fn new(inner: Arc<dyn ToolOutputWriter>) -> Self {
        Self {
            inner,
            stdout: 0.into(),
            stderr: 0.into(),
        }
    }

    pub(crate) fn matches(&self, outcome: &ExecOutcome) -> bool {
        use std::sync::atomic::Ordering;
        self.stdout.load(Ordering::Relaxed) == outcome.stdout_capture.observed_bytes as u64
            && self.stderr.load(Ordering::Relaxed) == outcome.stderr_capture.observed_bytes as u64
    }
}

#[async_trait]
impl ToolOutputWriter for CountedWriter {
    async fn append(&self, stream: OutputStream, bytes: &[u8]) -> Result<(), ToolError> {
        self.inner.append(stream, bytes).await?;
        let count = match stream {
            OutputStream::Stdout => &self.stdout,
            OutputStream::Stderr => &self.stderr,
            OutputStream::Result => {
                return Err(ToolError::execution(
                    "Environment wrote to a non-process output stream",
                ));
            }
        };
        count.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    async fn finish(&self) -> Result<Vec<ToolArtifact>, ToolError> {
        self.inner.finish().await
    }
}

pub(crate) async fn storage_call<T>(
    cancel: &CancellationToken,
    operation: impl Future<Output = Result<T, ToolError>>,
) -> Result<T, ToolError> {
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err(ToolError::cancelled("Output storage cancelled")),
        result = timeout(OUTPUT_STORE_TIMEOUT, operation) => {
            result.map_err(|_| ToolError::execution("Output storage timed out"))?
        }
    }
}

pub(crate) fn read_output_tool(store: Arc<dyn ToolOutputStore>) -> RegisteredTool {
    RegisteredTool::function(
        "read_tool_output",
        "Read a page of saved tool output by its opaque reference. Offsets and limits count bytes. Use next_offset to continue.",
        json!({"type":"object", "properties": {
            "reference":{"type":"string"},
            "offset":{"type":"integer","minimum":0},
            "limit":{"type":"integer","minimum":1,"maximum":MAX_OUTPUT_READ_BYTES}
        },"required":["reference"],"additionalProperties":false}),
        move |context, args| {
            let store = Arc::clone(&store);
            async move {
                let reference = args["reference"].as_str().ok_or_else(|| ToolError::invalid_arguments("reference must be a string"))?;
                let offset = match args.get("offset") {
                    None => 0,
                    Some(value) => value.as_u64().ok_or_else(|| ToolError::invalid_arguments("offset must be a nonnegative integer"))?,
                };
                let limit = match args.get("limit") {
                    None => MAX_OUTPUT_READ_BYTES,
                    Some(value) => value.as_u64().and_then(|n| usize::try_from(n).ok())
                        .filter(|n| (1..=MAX_OUTPUT_READ_BYTES).contains(n))
                        .ok_or_else(|| ToolError::invalid_arguments("limit must be between 1 and 16384"))?,
                };
                let scope = context.session_scope().ok_or_else(|| ToolError::unavailable("Output retrieval requires a session"))?;
                let page = storage_call(context.cancel(), store.read(scope, reference, offset, limit)).await?;
                let expected = page.total_bytes.saturating_sub(offset).min(limit as u64);
                if page.bytes.len() as u64 != expected {
                    return Err(ToolError::execution("Output store returned an inconsistent page"));
                }
                let next_offset = offset.saturating_add(page.bytes.len() as u64);
                Ok(json!({
                    "reference":reference, "offset":offset,
                    "next_offset":next_offset, "total_bytes":page.total_bytes,
                    "eof":next_offset >= page.total_bytes,
                    "text":String::from_utf8_lossy(&page.bytes)
                }).to_string())
            }
        },
    ).with_source(ToolSource::Native)
}
