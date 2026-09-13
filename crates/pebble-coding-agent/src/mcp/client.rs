//! One server's process and connection, through the `rmcp` client.
//!
//! [`Connection::start`] launches or reaches the server, performs the MCP
//! handshake within the startup timeout, and lists its tools.
//! [`Connection::call`] forwards one call with the server's tool timeout and
//! the caller's cancellation, sending the protocol's `notifications/cancelled`
//! when either ends the wait. [`Connection::close`] ends the session, stops
//! the process the connection owns, and releases the route it opened to an
//! environment-hosted server's port.

use std::collections::{BTreeMap, HashMap};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use reqwest::header::{CONNECTION, HeaderMap, HeaderName, HeaderValue};
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CallToolResult, CancelledNotification,
    CancelledNotificationParam, ClientCapabilities, ClientInfo, ClientRequest, Implementation,
    ProtocolVersion, RawContent, RequestId, ServerResult,
};
use rmcp::service::{
    ClientInitializeError, Peer, PeerRequestOptions, RequestHandle, RoleClient, RunningService,
    ServiceError, serve_client_with_ct,
};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use serde_json::{Map, Value};
use tokio::io::{AsyncBufReadExt as _, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::error::Elapsed;
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use super::routes::PortRoutes;
use super::sse::SseClientTransport;
use super::{McpHttpProtocol, McpPlacement, McpServer};
use crate::environment::{Environment, ExecRequest};

/// How much of a server's own error output is kept for a failure message.
const STDERR_TAIL_BYTES: usize = 4096;
/// How often an environment-hosted server's port is probed while it starts.
const PORT_POLL: Duration = Duration::from_millis(100);
/// How long one readiness probe may take. A forward into a container accepts
/// the connection before anything listens inside, so readiness is an HTTP
/// answer, not an accepted connection.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// How long a launcher or a kill command in the environment may take.
const ENVIRONMENT_COMMAND_TIMEOUT_MS: u64 = 30_000;

/// One tool the server advertised.
pub(super) struct DiscoveredTool {
    pub(super) name:         String,
    pub(super) description:  String,
    pub(super) input_schema: Value,
}

/// How one forwarded call ended.
pub(super) enum CallOutcome {
    /// The server answered; the text the model reads.
    Ok(String),
    /// The server answered with `isError`; the text the model reads as the
    /// tool's error.
    ToolError(String),
    /// The call did not reach the server or came back malformed.
    Failed(String),
    /// The server did not answer within the tool timeout.
    Timeout(Duration),
    /// The caller cancelled the wait.
    Cancelled,
}

/// Why a server did not start.
#[derive(Debug, thiserror::Error)]
pub(super) enum StartError {
    #[error("could not launch `{program}`: {reason}")]
    Launch { program: String, reason: String },
    #[error("the server did not complete the MCP handshake within {}s{tail}", timeout.as_secs())]
    HandshakeTimeout { timeout: Duration, tail: String },
    #[error("the MCP handshake failed: {reason}{tail}")]
    Handshake { reason: String, tail: String },
    #[error("listing the server's tools failed: {reason}")]
    ListTools { reason: String },
    #[error("no route to port {port} in the environment: {reason}")]
    Route { port: u16, reason: String },
    #[error("{0}")]
    Unsupported(String),
}

/// The last bytes a server wrote to its error output.
#[derive(Default)]
struct Tail(Vec<u8>);

impl Tail {
    fn push(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
        let excess = self.0.len().saturating_sub(STDERR_TAIL_BYTES);
        self.0.drain(..excess);
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0).into_owned()
    }
}

type SharedTail = Arc<StdMutex<Tail>>;

fn tail_suffix(tail: Option<&SharedTail>) -> String {
    let text = tail
        .map(|tail| tail.lock().unwrap_or_else(PoisonError::into_inner).text())
        .unwrap_or_default();
    let text = text.trim();
    if text.is_empty() {
        String::new()
    } else {
        format!("; the server wrote: {text}")
    }
}

/// A server process launched in the environment: how to stop it and where
/// its output went.
struct EnvironmentProcess {
    environment: Arc<dyn Environment>,
    pid:         String,
    stderr_log:  String,
}

impl EnvironmentProcess {
    /// The last of what the server wrote to its error output, for a failure
    /// message. Best effort: an environment that cannot read it says nothing.
    async fn stderr_tail(&self) -> String {
        let command = format!(
            "tail -c {STDERR_TAIL_BYTES} {} 2>/dev/null",
            shell_quote(&self.stderr_log)
        );
        let outcome = self
            .environment
            .exec(ExecRequest {
                timeout_ms: Some(ENVIRONMENT_COMMAND_TIMEOUT_MS),
                ..ExecRequest::new(&command)
            })
            .await;
        let text = outcome
            .map(|outcome| outcome.result.stdout)
            .unwrap_or_default();
        let text = text.trim();
        if text.is_empty() {
            String::new()
        } else {
            format!("; the server wrote: {text}")
        }
    }

    /// `SIGTERM` to the process and its group, a second, then `SIGKILL`, and
    /// the log files removed. Best effort.
    async fn stop(&self) {
        let pid = &self.pid;
        let command = format!(
            "kill -TERM -{pid} 2>/dev/null; kill -TERM {pid} 2>/dev/null; sleep 1; kill -KILL -{pid} \
             2>/dev/null; kill -KILL {pid} 2>/dev/null; rm -f {out} {err}; true",
            out = shell_quote(&self.stderr_log.replace(".err", ".out")),
            err = shell_quote(&self.stderr_log),
        );
        if let Err(error) = self
            .environment
            .exec(ExecRequest {
                timeout_ms: Some(ENVIRONMENT_COMMAND_TIMEOUT_MS),
                ..ExecRequest::new(&command)
            })
            .await
        {
            tracing::warn!(pid = %pid, error = %error, "stopping the MCP server in the environment failed");
        }
    }
}

/// The route the application opened to an environment-hosted server's port,
/// released when the connection closes.
struct Route {
    routes: Arc<dyn PortRoutes>,
    port:   u16,
}

impl Route {
    /// Best effort: the sandbox's own release closes it anyway, so a failure
    /// is logged and nothing else.
    async fn release(&self) {
        if let Err(error) = self.routes.release(self.port).await {
            tracing::warn!(port = self.port, error = %error.detail(), "releasing the MCP server's route failed");
        }
    }
}

/// Everything a failed start has to put back.
struct Cleanup {
    process: Option<EnvironmentProcess>,
    route:   Option<Route>,
}

impl Cleanup {
    async fn run(self) {
        if let Some(process) = &self.process {
            process.stop().await;
        }
        if let Some(route) = &self.route {
            route.release().await;
        }
    }
}

/// The live connection to one server.
pub(super) struct Connection {
    peer:                Peer<RoleClient>,
    service:             Mutex<Option<RunningService<RoleClient, ClientInfo>>>,
    tool_timeout:        Duration,
    /// The process an environment-hosted server runs in.
    process:             Option<EnvironmentProcess>,
    /// The route to an environment-hosted server's port.
    route:               Option<Route>,
    /// What closed the connection, set once by the first call that observed
    /// the close; every later call fails without reaching the server.
    disconnect:          OnceLock<String>,
    /// Whether the close has been handed out for reporting.
    disconnect_reported: AtomicBool,
}

/// The protocol handshake, or the way it did not happen.
type Handshake =
    Result<Result<RunningService<RoleClient, ClientInfo>, ClientInitializeError>, Elapsed>;

impl Connection {
    /// Launches or reaches the server, completes the handshake, and lists its
    /// tools.
    pub(super) async fn start(
        server: &McpServer,
        environment: &Arc<dyn Environment>,
        routes: Option<&Arc<dyn PortRoutes>>,
    ) -> Result<(Self, Vec<DiscoveredTool>), StartError> {
        let startup = server.startup_timeout;
        let tool_timeout = server.tool_timeout;
        let info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("pebble", env!("CARGO_PKG_VERSION")),
        )
        .with_protocol_version(ProtocolVersion::V_2025_03_26);
        let cancel = CancellationToken::new();
        let began = Instant::now();
        let mut cleanup = Cleanup {
            process: None,
            route:   None,
        };
        let mut tail: Option<SharedTail> = None;
        let handshake: Handshake = match &server.placement {
            McpPlacement::Stdio {
                command,
                env,
                current_dir,
                clear_env,
            } => {
                let (program, args) = command.split_first().ok_or_else(|| StartError::Launch {
                    program: String::new(),
                    reason:  "the command is empty".into(),
                })?;
                let mut cmd = Command::new(program);
                cmd.args(args).kill_on_drop(true);
                if *clear_env {
                    cmd.env_clear();
                }
                cmd.envs(env);
                if let Some(dir) = current_dir {
                    cmd.current_dir(dir);
                }
                #[cfg(unix)]
                cmd.process_group(0);
                let (transport, stderr) = TokioChildProcess::builder(cmd)
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|error| StartError::Launch {
                        program: program.clone(),
                        reason:  error.to_string(),
                    })?;
                let shared: SharedTail = Arc::default();
                if let Some(stderr) = stderr {
                    let shared = Arc::clone(&shared);
                    let name = server.name.clone();
                    // Disposable: the server's own diagnostics, kept for a
                    // failure message and traced; the child owns the pipe.
                    tokio::spawn(async move {
                        let mut lines = BufReader::new(stderr).lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            tracing::debug!(server = %name, line = %line, "MCP server stderr");
                            let mut tail = shared.lock().unwrap_or_else(PoisonError::into_inner);
                            tail.push(line.as_bytes());
                            tail.push(b"\n");
                        }
                    });
                }
                tail = Some(shared);
                timeout(
                    startup,
                    serve_client_with_ct(info, transport, cancel.child_token()),
                )
                .await
            }
            McpPlacement::Http {
                url,
                headers,
                protocol,
            } => {
                // The server gets the startup timeout to start answering, as
                // an environment-hosted one does: an application often spawns
                // it just before building the agent.
                let deadline = began + startup;
                if let Err(error) = probe_until_ready(url, headers, deadline).await {
                    return Err(match error {
                        ProbeError::Build(reason) => StartError::Unsupported(reason),
                        ProbeError::Deadline => StartError::HandshakeTimeout {
                            timeout: startup,
                            tail:    String::new(),
                        },
                    });
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                connect_http(*protocol, url, headers, info, &cancel, remaining).await?
            }
            McpPlacement::Environment {
                command,
                port,
                env,
                protocol,
                path,
            } => {
                let process =
                    launch_in_environment(environment, &server.name, command, env).await?;
                cleanup.process = Some(process);
                // The address pebble reaches the port at is the application's
                // route to it, or the loopback address when the environment
                // shares the host's network. An application whose environment
                // routes to no port says so through the same error, and the
                // server is reported as having no route.
                let (url, headers) = match routes {
                    Some(routes) => match routes.route(*port).await {
                        Ok(route) => {
                            cleanup.route = Some(Route {
                                routes: Arc::clone(routes),
                                port:   *port,
                            });
                            (route.url, route.headers)
                        }
                        Err(error) => {
                            cleanup.run().await;
                            return Err(StartError::Route {
                                port:   *port,
                                reason: error.detail(),
                            });
                        }
                    },
                    None => (format!("http://127.0.0.1:{port}"), BTreeMap::new()),
                };
                let url = match path {
                    Some(path) => format!(
                        "{}/{}",
                        url.trim_end_matches('/'),
                        path.trim_start_matches('/')
                    ),
                    None => url,
                };
                let deadline = began + startup;
                if let Err(error) = probe_until_ready(&url, &headers, deadline).await {
                    let tail = match &cleanup.process {
                        Some(process) => process.stderr_tail().await,
                        None => String::new(),
                    };
                    cleanup.run().await;
                    return Err(match error {
                        ProbeError::Build(reason) => StartError::Route {
                            port: *port,
                            reason,
                        },
                        ProbeError::Deadline => StartError::HandshakeTimeout {
                            timeout: startup,
                            tail,
                        },
                    });
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                match connect_http(*protocol, &url, &headers, info, &cancel, remaining).await {
                    Ok(handshake) => handshake,
                    Err(error) => {
                        cleanup.run().await;
                        return Err(error);
                    }
                }
            }
        };
        let service = match handshake {
            Ok(Ok(service)) => service,
            Ok(Err(error)) => {
                let tail = environment_or_pipe_tail(cleanup.process.as_ref(), tail.as_ref()).await;
                cleanup.run().await;
                return Err(StartError::Handshake {
                    reason: error.to_string(),
                    tail,
                });
            }
            Err(_) => {
                let tail = environment_or_pipe_tail(cleanup.process.as_ref(), tail.as_ref()).await;
                cleanup.run().await;
                return Err(StartError::HandshakeTimeout {
                    timeout: startup,
                    tail,
                });
            }
        };
        if let Some(peer_info) = service.peer().peer_info() {
            tracing::info!(
                server = %server.name,
                server_name = %peer_info.server_info.name,
                server_version = %peer_info.server_info.version,
                "MCP server initialized"
            );
        }
        let remaining = (began + startup).saturating_duration_since(Instant::now());
        let tools = match timeout(remaining.max(PROBE_TIMEOUT), service.list_all_tools()).await {
            Ok(Ok(tools)) => tools,
            Ok(Err(error)) => {
                cleanup.run().await;
                return Err(StartError::ListTools {
                    reason: error.to_string(),
                });
            }
            Err(_) => {
                cleanup.run().await;
                return Err(StartError::ListTools {
                    reason: format!("no answer within {}s", startup.as_secs()),
                });
            }
        };
        let tools = tools
            .into_iter()
            .map(|tool| DiscoveredTool {
                name:         tool.name.to_string(),
                description:  tool.description.as_deref().unwrap_or("").to_owned(),
                input_schema: serde_json::to_value(&*tool.input_schema).unwrap_or_default(),
            })
            .collect();
        Ok((
            Self {
                peer: service.peer().clone(),
                service: Mutex::new(Some(service)),
                tool_timeout,
                process: cleanup.process,
                route: cleanup.route,
                disconnect: OnceLock::new(),
                disconnect_reported: AtomicBool::new(false),
            },
            tools,
        ))
    }

    /// Forwards one call. The server's tool timeout and the caller's
    /// cancellation both end the wait with a `notifications/cancelled` to the
    /// server.
    pub(super) async fn call(
        &self,
        tool: &str,
        arguments: Value,
        cancel: &CancellationToken,
    ) -> CallOutcome {
        if self.disconnect.get().is_some() {
            return CallOutcome::Failed("the server's connection is closed".into());
        }
        let arguments: Option<Map<String, Value>> = match arguments {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return CallOutcome::Failed(format!(
                    "MCP tool arguments must be a JSON object, got {other}"
                ));
            }
        };
        let mut params = CallToolRequestParams::new(tool.to_owned());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = match self
            .peer
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await
        {
            Ok(handle) => handle,
            Err(error) => return self.failed(&error),
        };
        let RequestHandle { rx, id, peer, .. } = handle;
        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                notify_cancelled(&peer, id, "cancelled by the agent").await;
                CallOutcome::Cancelled
            }
            () = sleep(self.tool_timeout) => {
                notify_cancelled(&peer, id, RequestHandle::<RoleClient>::REQUEST_TIMEOUT_REASON).await;
                CallOutcome::Timeout(self.tool_timeout)
            }
            response = rx => match response {
                Ok(Ok(ServerResult::CallToolResult(result))) => outcome_of(&result),
                Ok(Ok(_)) => CallOutcome::Failed("the server answered with an unexpected response type".into()),
                Ok(Err(error)) => self.failed(&error),
                Err(_) => self.failed(&ServiceError::TransportClosed),
            }
        }
    }

    fn failed(&self, error: &ServiceError) -> CallOutcome {
        if matches!(
            error,
            ServiceError::TransportClosed | ServiceError::TransportSend(_)
        ) {
            // Only the first close is kept: it is the one that closed the
            // connection, and the one the session is told about.
            let _ = self.disconnect.set(error.to_string());
        }
        CallOutcome::Failed(error.to_string())
    }

    /// What closed the connection, handed out once: `Some` to exactly one
    /// caller after the close was observed, `None` before it and to every
    /// caller after that one. The caller reports it on the session's stream.
    pub(super) fn unreported_disconnect(&self) -> Option<&str> {
        let error = self.disconnect.get()?;
        if self.disconnect_reported.swap(true, Ordering::AcqRel) {
            return None;
        }
        Some(error)
    }

    /// Ends the session, stops the owned process, and releases the route to
    /// its port. The protocol's close is bounded by `limit`.
    pub(super) async fn close(&self, limit: Duration) {
        let service = self.service.lock().await.take();
        if let Some(mut service) = service {
            match timeout(limit + PROBE_TIMEOUT, service.close_with_timeout(limit)).await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => {
                    tracing::warn!(error = %error, "MCP client did not close cleanly");
                }
                Err(_) => tracing::warn!("MCP client did not close in time"),
            }
        }
        if let Some(process) = &self.process {
            process.stop().await;
        }
        if let Some(route) = &self.route {
            route.release().await;
        }
    }
}

/// The failure detail for a handshake that did not complete: the environment
/// server's log when there is one, else the pipe tail a child process wrote.
async fn environment_or_pipe_tail(
    process: Option<&EnvironmentProcess>,
    tail: Option<&SharedTail>,
) -> String {
    match process {
        Some(process) => process.stderr_tail().await,
        None => tail_suffix(tail),
    }
}

/// Opens the HTTP transport `protocol` names to `url` and runs the handshake
/// within `within`.
async fn connect_http(
    protocol: McpHttpProtocol,
    url: &str,
    headers: &BTreeMap<String, String>,
    info: ClientInfo,
    cancel: &CancellationToken,
    within: Duration,
) -> Result<Handshake, StartError> {
    let headers = header_map(headers)?;
    Ok(match protocol {
        McpHttpProtocol::StreamableHttp => {
            let mut config = StreamableHttpClientTransportConfig::with_uri(url.to_owned());
            config.custom_headers = headers;
            let transport = StreamableHttpClientTransport::with_client(http_client()?, config);
            timeout(
                within,
                serve_client_with_ct(info, transport, cancel.child_token()),
            )
            .await
        }
        McpHttpProtocol::Sse => {
            let transport = SseClientTransport::new(url, http_client()?, headers)
                .map_err(|error| StartError::Unsupported(error.to_string()))?;
            timeout(
                within,
                serve_client_with_ct(info, transport, cancel.child_token()),
            )
            .await
        }
    })
}

fn http_client() -> Result<reqwest::Client, StartError> {
    reqwest::Client::builder()
        .build()
        .map_err(|error| StartError::Unsupported(format!("building the HTTP client: {error}")))
}

/// Why the readiness probe gave up.
enum ProbeError {
    Build(String),
    Deadline,
}

/// Asks `url` until it answers with anything at all or `deadline` passes.
///
/// Each probe closes its connection: a single-threaded server, or a forward's
/// bridge, must be free for the handshake. Any HTTP response means the server
/// is listening; a bare GET to an MCP endpoint is usually refused, and that is
/// enough.
async fn probe_until_ready(
    url: &str,
    headers: &BTreeMap<String, String>,
    deadline: Instant,
) -> Result<(), ProbeError> {
    let mut headers: HeaderMap = header_map(headers)
        .map_err(|error| ProbeError::Build(error.to_string()))?
        .into_iter()
        .collect();
    headers.insert(CONNECTION, HeaderValue::from_static("close"));
    let probe = reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .timeout(PROBE_TIMEOUT)
        .build()
        .map_err(|error| ProbeError::Build(format!("building the readiness probe: {error}")))?;
    loop {
        if probe.get(url).headers(headers.clone()).send().await.is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(ProbeError::Deadline);
        }
        sleep(PORT_POLL).await;
    }
}

/// Launches `command` in the environment as a detached process whose output
/// goes to log files, and answers with how to stop it.
///
/// The launcher returns as soon as the server is started, so the environment's
/// own drain of the launcher's output ends; the server's output goes to files
/// the failure message reads. A `setsid` puts the server in its own session
/// where one exists; elsewhere job control gives it its own process group, so
/// the environment ending the launcher's group does not end the server.
async fn launch_in_environment(
    environment: &Arc<dyn Environment>,
    server_name: &str,
    command: &[String],
    env: &BTreeMap<String, String>,
) -> Result<EnvironmentProcess, StartError> {
    let program = command.first().cloned().ok_or_else(|| StartError::Launch {
        program: String::new(),
        reason:  "the command is empty".into(),
    })?;
    let token = uuid::Uuid::new_v4().simple().to_string();
    let base = format!(
        "/tmp/pebble-mcp-{}-{token}",
        super::sanitize_name(server_name)
    );
    let stdout_log = format!("{base}.out");
    let stderr_log = format!("{base}.err");
    let inner = format!(
        "{} >{} 2>{}",
        command
            .iter()
            .map(|word| shell_quote(word))
            .collect::<Vec<_>>()
            .join(" "),
        shell_quote(&stdout_log),
        shell_quote(&stderr_log),
    );
    let launcher = format!(
        "if command -v setsid >/dev/null 2>&1; then setsid bash -c {inner} </dev/null >/dev/null \
         2>&1 & else set -m; bash -c {inner} </dev/null >/dev/null 2>&1 & fi\necho $!",
        inner = shell_quote(&inner),
    );
    let env_vars: HashMap<String, String> = env
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let outcome = environment
        .exec(ExecRequest {
            timeout_ms: Some(ENVIRONMENT_COMMAND_TIMEOUT_MS),
            env_vars: (!env_vars.is_empty()).then_some(&env_vars),
            ..ExecRequest::new(&launcher)
        })
        .await
        .map_err(|error| StartError::Launch {
            program: program.clone(),
            reason:  error.to_string(),
        })?;
    let pid = outcome.result.stdout.trim().to_owned();
    if !outcome.result.is_success() || pid.is_empty() || !pid.chars().all(|c| c.is_ascii_digit()) {
        return Err(StartError::Launch {
            program,
            reason: format!(
                "the launcher exited with {:?} and wrote {:?}{}",
                outcome.result.exit_code,
                outcome.result.stdout.trim(),
                if outcome.result.stderr.trim().is_empty() {
                    String::new()
                } else {
                    format!("; stderr: {}", outcome.result.stderr.trim())
                }
            ),
        });
    }
    tracing::info!(server = %server_name, pid = %pid, "MCP server launched in the environment");
    Ok(EnvironmentProcess {
        environment: Arc::clone(environment),
        pid,
        stderr_log,
    })
}

/// `text` as one single-quoted shell word.
pub(super) fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

async fn notify_cancelled(peer: &Peer<RoleClient>, id: RequestId, reason: &str) {
    let notification = CancelledNotification::new(CancelledNotificationParam {
        request_id: id,
        reason:     Some(reason.to_owned()),
    });
    let _ = peer.send_notification(notification.into()).await;
}

/// The text parts joined by newlines, a placeholder for every other part;
/// `isError` makes it the tool's error.
fn outcome_of(result: &CallToolResult) -> CallOutcome {
    let text = result
        .content
        .iter()
        .map(|part| match &part.raw {
            RawContent::Text(text) => text.text.clone(),
            RawContent::Image(_) => "[image content]".to_owned(),
            RawContent::Audio(_) => "[audio content]".to_owned(),
            RawContent::Resource(_) | RawContent::ResourceLink(_) => {
                "[resource content]".to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if result.is_error.unwrap_or(false) {
        CallOutcome::ToolError(text)
    } else {
        CallOutcome::Ok(text)
    }
}

/// The configured (or route-provided) headers, parsed, in the shape the
/// `rmcp` transport config takes.
fn header_map(
    headers: &BTreeMap<String, String>,
) -> Result<HashMap<HeaderName, HeaderValue>, StartError> {
    let mut map = HashMap::new();
    for (key, value) in headers {
        let name = HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
            StartError::Unsupported(format!("invalid header name `{key}`: {error}"))
        })?;
        let value = HeaderValue::from_str(value).map_err(|error| {
            StartError::Unsupported(format!("invalid header value for `{key}`: {error}"))
        })?;
        map.insert(name, value);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use rmcp::model::Content;

    use super::*;

    #[test]
    fn a_shell_word_survives_quotes_and_spaces() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's here"), "'it'\\''s here'");
    }

    #[test]
    fn call_results_join_text_and_mark_errors() {
        let ok = CallToolResult::success(vec![Content::text("line 1"), Content::text("line 2")]);
        assert!(matches!(outcome_of(&ok), CallOutcome::Ok(text) if text == "line 1\nline 2"));
        let error = CallToolResult::error(vec![Content::text("something failed")]);
        assert!(
            matches!(outcome_of(&error), CallOutcome::ToolError(text) if text == "something failed")
        );
        let image = CallToolResult::success(vec![Content::image("base64data", "image/png")]);
        assert!(matches!(outcome_of(&image), CallOutcome::Ok(text) if text == "[image content]"));
    }

    #[test]
    fn a_tail_keeps_only_the_last_bytes() {
        let mut tail = Tail::default();
        tail.push(&vec![b'a'; STDERR_TAIL_BYTES]);
        tail.push(b"tail");
        let text = tail.text();
        assert_eq!(text.len(), STDERR_TAIL_BYTES);
        assert!(text.ends_with("tail"));
    }

    #[test]
    fn header_maps_reject_invalid_names() {
        let mut headers = BTreeMap::new();
        headers.insert("x-ok".to_owned(), "yes".to_owned());
        assert_eq!(header_map(&headers).expect("valid headers").len(), 1);
        headers.insert("bad header".to_owned(), "no".to_owned());
        assert!(matches!(
            header_map(&headers),
            Err(StartError::Unsupported(_))
        ));
    }
}
