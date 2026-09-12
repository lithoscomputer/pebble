//! The older MCP HTTP transport: one server-sent event stream that names the
//! endpoint messages are posted to.
//!
//! `rmcp` no longer ships this as a client, so pebble carries it for the
//! servers that still speak it. The stream is bounded: an event larger than
//! [`MAX_SSE_MESSAGE_BYTES`] ends the stream, and an endpoint on another
//! origin than the stream's is refused, so a server cannot redirect the
//! client's posts elsewhere.

use std::collections::HashMap;
use std::future::Future;

use futures_util::{StreamExt as _, TryStreamExt as _};
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use rmcp::model::ServerJsonRpcMessage;
use rmcp::service::{RoleClient, RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::Transport;
use sse_stream::{Sse, SseStream};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use url::Url;

/// The largest server-sent event the transport accepts.
pub(super) const MAX_SSE_MESSAGE_BYTES: usize = 1024 * 1024;

/// A client transport over the SSE protocol.
pub(super) struct SseClientTransport {
    client:      reqwest::Client,
    headers:     HeaderMap,
    endpoint_rx: watch::Receiver<Option<String>>,
    messages_rx: mpsc::Receiver<ServerJsonRpcMessage>,
    stream_task: Option<JoinHandle<()>>,
}

impl SseClientTransport {
    /// Opens the event stream at `url`, carrying `headers` on every request.
    pub(super) fn new(
        url: &str,
        client: reqwest::Client,
        headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<Self, SseClientError> {
        let (endpoint_tx, endpoint_rx) = watch::channel(None);
        let (messages_tx, messages_rx) = mpsc::channel(64);
        let sse_url = Url::parse(url).map_err(|error| SseClientError::Url(error.to_string()))?;
        let headers: HeaderMap = headers.into_iter().collect();
        let stream_client = client.clone();
        let stream_headers = headers.clone();
        let stream_task = tokio::spawn(async move {
            if let Err(error) = read_sse_stream(
                stream_client,
                stream_headers,
                sse_url,
                endpoint_tx,
                messages_tx,
            )
            .await
            {
                tracing::warn!(error = %error, "SSE MCP stream ended");
            }
        });
        Ok(Self {
            client,
            headers,
            endpoint_rx,
            messages_rx,
            stream_task: Some(stream_task),
        })
    }
}

impl Drop for SseClientTransport {
    fn drop(&mut self) {
        if let Some(stream_task) = &self.stream_task {
            stream_task.abort();
        }
    }
}

impl Transport<RoleClient> for SseClientTransport {
    type Error = SseClientError;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let client = self.client.clone();
        let headers = self.headers.clone();
        let mut endpoint_rx = self.endpoint_rx.clone();
        async move {
            let endpoint = wait_for_endpoint(&mut endpoint_rx).await?;
            client
                .post(endpoint)
                .headers(headers)
                .header(CONTENT_TYPE, "application/json")
                .json(&item)
                .send()
                .await
                .map_err(|error| SseClientError::http(&error))?
                .error_for_status()
                .map_err(|error| SseClientError::http(&error))?;
            Ok(())
        }
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<RoleClient>>> + Send {
        self.messages_rx.recv()
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        if let Some(stream_task) = self.stream_task.take() {
            stream_task.abort();
        }
        Ok(())
    }
}

async fn wait_for_endpoint(
    endpoint_rx: &mut watch::Receiver<Option<String>>,
) -> Result<String, SseClientError> {
    loop {
        if let Some(endpoint) = endpoint_rx.borrow().clone() {
            return Ok(endpoint);
        }
        endpoint_rx
            .changed()
            .await
            .map_err(|_| SseClientError::EndpointUnavailable)?;
    }
}

async fn read_sse_stream(
    client: reqwest::Client,
    headers: HeaderMap,
    sse_url: Url,
    endpoint_tx: watch::Sender<Option<String>>,
    messages_tx: mpsc::Sender<ServerJsonRpcMessage>,
) -> Result<(), SseClientError> {
    let response = client
        .get(sse_url.clone())
        .headers(headers)
        .header(ACCEPT, "text/event-stream")
        .send()
        .await
        .map_err(|error| SseClientError::http(&error))?
        .error_for_status()
        .map_err(|error| SseClientError::http(&error))?;
    let mut size_guard = SseSizeGuard::default();
    let byte_stream = response.bytes_stream().map(move |chunk| {
        let chunk = chunk.map_err(|error| SseClientError::http(&error))?;
        size_guard.check_chunk(&chunk)?;
        Ok::<_, SseClientError>(chunk)
    });
    let mut stream = SseStream::from_bytes_stream(byte_stream);

    while let Some(event) = stream
        .try_next()
        .await
        .map_err(|error| SseClientError::Stream(error.to_string()))?
    {
        handle_sse_event(event, &sse_url, &endpoint_tx, &messages_tx).await?;
    }
    Ok(())
}

async fn handle_sse_event(
    event: Sse,
    sse_url: &Url,
    endpoint_tx: &watch::Sender<Option<String>>,
    messages_tx: &mpsc::Sender<ServerJsonRpcMessage>,
) -> Result<(), SseClientError> {
    let data = event.data.unwrap_or_default();
    match event.event.as_deref() {
        Some("endpoint") => {
            let endpoint = resolve_endpoint_url(sse_url, data.trim())?;
            let _ = endpoint_tx.send(Some(endpoint.to_string()));
        }
        None | Some("" | "message") => {
            if data.trim().is_empty() {
                return Ok(());
            }
            let message: ServerJsonRpcMessage = serde_json::from_str(&data)
                .map_err(|error| SseClientError::Message(error.to_string()))?;
            messages_tx
                .send(message)
                .await
                .map_err(|_| SseClientError::ReceiverClosed)?;
        }
        _ => {}
    }
    Ok(())
}

/// The endpoint the stream named, resolved against the stream's URL and
/// refused when it leaves the stream's origin.
fn resolve_endpoint_url(sse_url: &Url, endpoint: &str) -> Result<Url, SseClientError> {
    if endpoint.starts_with("//") {
        return Err(SseClientError::Endpoint(
            "the endpoint must not be protocol-relative".into(),
        ));
    }
    let resolved = if endpoint.starts_with('/') {
        let (path, query) = endpoint
            .split_once('?')
            .map_or((endpoint, None), |(path, query)| (path, Some(query)));
        let base_path = sse_url.path().trim_end_matches('/');
        let prefix = base_path.rsplit_once('/').map_or("", |(prefix, _)| prefix);
        let mut url = sse_url.clone();
        url.set_path(&format!("{prefix}{path}"));
        url.set_query(query);
        url
    } else {
        sse_url
            .join(endpoint)
            .map_err(|error| SseClientError::Endpoint(error.to_string()))?
    };
    if resolved.origin() != sse_url.origin() {
        return Err(SseClientError::Endpoint(
            "the endpoint's origin must match the stream's origin".into(),
        ));
    }
    Ok(resolved)
}

/// Ends a stream whose event outgrows the budget.
#[derive(Default)]
struct SseSizeGuard {
    current_event_bytes: usize,
    current_line_bytes:  usize,
}

impl SseSizeGuard {
    fn check_chunk(&mut self, chunk: &[u8]) -> Result<(), SseClientError> {
        for byte in chunk {
            self.current_event_bytes = self
                .current_event_bytes
                .checked_add(1)
                .filter(|bytes| *bytes <= MAX_SSE_MESSAGE_BYTES)
                .ok_or(SseClientError::MessageTooLarge {
                    max_bytes: MAX_SSE_MESSAGE_BYTES,
                })?;
            match *byte {
                b'\n' => {
                    if self.current_line_bytes == 0 {
                        self.current_event_bytes = 0;
                    }
                    self.current_line_bytes = 0;
                }
                b'\r' => {}
                _ => self.current_line_bytes += 1,
            }
        }
        Ok(())
    }
}

/// Why the SSE transport failed.
#[derive(Debug, thiserror::Error)]
pub(super) enum SseClientError {
    #[error("invalid SSE MCP URL: {0}")]
    Url(String),
    #[error("SSE MCP request failed: {0}")]
    Http(String),
    #[error("SSE MCP stream failed: {0}")]
    Stream(String),
    #[error("invalid SSE MCP endpoint: {0}")]
    Endpoint(String),
    #[error("invalid SSE MCP JSON-RPC message: {0}")]
    Message(String),
    #[error("SSE MCP endpoint was not received before the stream closed")]
    EndpointUnavailable,
    #[error("SSE MCP receiver closed")]
    ReceiverClosed,
    #[error("SSE MCP message exceeds maximum size of {max_bytes} bytes")]
    MessageTooLarge { max_bytes: usize },
}

impl SseClientError {
    fn http(error: &reqwest::Error) -> Self {
        Self::Http(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse_url() -> Url {
        Url::parse("https://srv.example.com/sse").expect("a valid URL")
    }

    #[test]
    fn a_relative_endpoint_with_a_query_resolves_beside_the_stream() {
        let resolved =
            resolve_endpoint_url(&sse_url(), "/messages?sessionId=abc").expect("resolves");
        assert_eq!(
            resolved.as_str(),
            "https://srv.example.com/messages?sessionId=abc"
        );
    }

    #[test]
    fn a_same_origin_absolute_endpoint_is_accepted() {
        let resolved = resolve_endpoint_url(&sse_url(), "https://srv.example.com/messages?s=1")
            .expect("resolves");
        assert_eq!(resolved.as_str(), "https://srv.example.com/messages?s=1");
        let default_port = resolve_endpoint_url(&sse_url(), "https://srv.example.com:443/messages")
            .expect("resolves");
        assert_eq!(default_port.origin(), sse_url().origin());
    }

    #[test]
    fn endpoints_off_the_streams_origin_are_refused() {
        for endpoint in [
            "https://evil.example/steal",
            "http://srv.example.com/messages",
            "https://srv.example.com:8443/messages",
        ] {
            let error = resolve_endpoint_url(&sse_url(), endpoint).expect_err("refused");
            assert!(error.to_string().contains("origin"), "{endpoint}: {error}");
        }
        let error = resolve_endpoint_url(&sse_url(), "//evil.example/steal").expect_err("refused");
        assert!(error.to_string().contains("protocol-relative"), "{error}");
    }

    #[test]
    fn the_size_guard_ends_an_oversized_event_and_resets_between_events() {
        // The blank line that ends an event counts against it, so the
        // largest event that fits is the budget less its two newlines.
        let mut guard = SseSizeGuard::default();
        guard
            .check_chunk(&vec![b'a'; MAX_SSE_MESSAGE_BYTES - 2])
            .expect("under the cap");
        guard.check_chunk(b"\n\n").expect("the event ends");
        guard
            .check_chunk(&vec![b'b'; MAX_SSE_MESSAGE_BYTES - 1])
            .expect("a new event starts from zero");
        assert!(matches!(
            guard.check_chunk(b"cc"),
            Err(SseClientError::MessageTooLarge { .. })
        ));
    }
}
