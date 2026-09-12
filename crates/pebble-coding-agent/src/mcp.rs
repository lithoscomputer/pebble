//! MCP servers as tool sources.
//!
//! An application names the servers a session may call
//! ([`CodingAgentBuilder::mcp_servers`](crate::CodingAgentBuilder::mcp_servers));
//! pebble starts them when the agent is built, registers every tool they
//! advertise as a [`RegisteredTool`] whose source is [`ToolSource::Mcp`] under
//! the name `mcp__{server}__{tool}`, and closes them when the agent shuts
//! down. Pebble's middleware, history, output policy, cancellation, and events
//! then apply to an MCP tool as to any other; the agent loop sees only tools.
//!
//! Three placements, as both embedders have today:
//! - [`McpPlacement::Stdio`]: a child process of the application, never of the
//!   sandbox.
//! - [`McpPlacement::Http`]: a server reached over HTTP, by the streamable HTTP
//!   transport or the older SSE one.
//! - [`McpPlacement::Environment`]: a server launched through
//!   [`Environment::exec`] and reached over HTTP through the environment's
//!   route to its port. The route is sandbox-driver's [`PreviewUrls`] facet,
//!   which the application hands over with
//!   [`CodingAgentBuilder::port_routes`](crate::CodingAgentBuilder::port_routes)
//!   and which every sandbox that forwards ports already provides, headers
//!   included. Without one, the port is reached on the loopback address, which
//!   is where a server in an environment that shares the host's network
//!   listens. Pebble owns the route's lifetime: it is released when the server
//!   fails to start and when the agent shuts down.
//!
//! A server that does not start (a launch error, a handshake that fails or
//! times out) is reported as [`McpServerFailed`](CodingEvent::McpServerFailed)
//! and skipped, and the session proceeds with the tools of the servers that
//! started, each reported as [`McpServerReady`](CodingEvent::McpServerReady).
//! A tool result the server marks `isError` reaches the model as the tool's
//! error text; a transport failure, a timeout, or a cancellation reaches it as
//! a failed call with a reason.

mod client;
mod sse;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use client::{CallOutcome, Connection, DiscoveredTool};
use lithos_llm::types::ToolDefinition;
pub use sandbox_driver::{PreviewUrl, PreviewUrls};
use serde_json::Value;

use crate::environment::Environment;
use crate::tool::{RegisteredTool, ToolContext, ToolError};
use crate::types::{CodingEvent, McpToolSummary, ToolSource};

/// How long a server gets to close after the agent ends.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// One MCP server a session may call.
#[derive(Clone, Debug)]
pub struct McpServer {
    name:            String,
    placement:       McpPlacement,
    startup_timeout: Duration,
    tool_timeout:    Duration,
}

impl McpServer {
    /// A server called `name`, placed as `placement`, with ten seconds to
    /// start and sixty to answer a call.
    pub fn new(name: impl Into<String>, placement: McpPlacement) -> Self {
        Self {
            name: name.into(),
            placement,
            startup_timeout: Duration::from_secs(10),
            tool_timeout: Duration::from_secs(60),
        }
    }

    /// Sets how long the server gets to complete the MCP handshake and list
    /// its tools, launch included.
    #[must_use]
    pub const fn with_startup_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }

    /// Sets how long one tool call may take before it fails as timed out.
    #[must_use]
    pub const fn with_tool_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = timeout;
        self
    }

    /// The server's configured name, which prefixes its tools.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where the server runs and how it is reached.
    #[must_use]
    pub const fn placement(&self) -> &McpPlacement {
        &self.placement
    }

    /// How long the server gets to start.
    #[must_use]
    pub const fn startup_timeout(&self) -> Duration {
        self.startup_timeout
    }

    /// How long one tool call may take.
    #[must_use]
    pub const fn tool_timeout(&self) -> Duration {
        self.tool_timeout
    }
}

/// Where an MCP server runs and how pebble reaches it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum McpPlacement {
    /// A child process of the embedding application, spoken to over its
    /// standard streams.
    Stdio {
        /// The program and its arguments.
        command:     Vec<String>,
        /// Variables layered over the application's environment, or the
        /// whole environment when `clear_env` is set.
        env:         BTreeMap<String, String>,
        /// The directory to start in; absent inherits the application's.
        current_dir: Option<PathBuf>,
        /// Whether the child sees only `env`, not the application's variables.
        clear_env:   bool,
    },
    /// A server reached over HTTP from the application.
    Http {
        /// The endpoint URL.
        url:      String,
        /// Headers every request carries, such as an authorization token.
        headers:  BTreeMap<String, String>,
        /// Which HTTP transport the server speaks.
        protocol: McpHttpProtocol,
    },
    /// A server launched in the session's environment and reached over HTTP
    /// through the environment's route to its port.
    Environment {
        /// The program and its arguments, run through the environment's shell.
        command:  Vec<String>,
        /// The port the server listens on inside the environment.
        port:     u16,
        /// Variables layered over the environment's for the server process.
        env:      BTreeMap<String, String>,
        /// Which HTTP transport the server speaks.
        protocol: McpHttpProtocol,
    },
}

/// Which HTTP transport an MCP server speaks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum McpHttpProtocol {
    /// The current transport: JSON-RPC over POST, with optional server
    /// streams.
    #[default]
    StreamableHttp,
    /// The older transport: one server-sent event stream that names the
    /// endpoint messages are posted to.
    Sse,
}

/// The name the model calls a server's tool by: `mcp__{server}__{tool}`,
/// with every character outside `[A-Za-z0-9_]` in either part replaced by
/// `_`.
#[must_use]
pub fn qualified_tool_name(server: &str, tool: &str) -> String {
    format!("mcp__{}__{}", sanitize_name(server), sanitize_name(tool))
}

/// The `(server, tool)` a qualified name was built from, or `None` when the
/// name is not in the qualified shape.
#[must_use]
pub fn parse_qualified_name(qualified: &str) -> Option<(String, String)> {
    let rest = qualified.strip_prefix("mcp__")?;
    let (server, tool) = rest.split_once("__")?;
    if server.is_empty() || tool.is_empty() {
        return None;
    }
    Some((server.to_owned(), tool.to_owned()))
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// What became of one configured server when the agent was built.
#[derive(Clone, Debug)]
pub(crate) enum McpServerOutcome {
    Ready {
        server: String,
        tools:  Vec<McpToolSummary>,
    },
    Failed {
        server: String,
        error:  String,
    },
}

impl McpServerOutcome {
    pub(crate) fn to_event(&self) -> CodingEvent {
        match self {
            Self::Ready { server, tools } => CodingEvent::McpServerReady {
                server: server.clone(),
                tools:  tools.clone(),
            },
            Self::Failed { server, error } => CodingEvent::McpServerFailed {
                server: server.clone(),
                error:  error.clone(),
            },
        }
    }
}

/// One agent's MCP servers: started, their tools registered, and closed with
/// the agent.
pub(crate) struct McpServers {
    connections: Vec<(String, Arc<Connection>)>,
    tools:       Vec<RegisteredTool>,
    outcomes:    Vec<McpServerOutcome>,
}

impl McpServers {
    /// Starts every configured server in order and discovers its tools. A
    /// server that fails is recorded and skipped.
    pub(crate) async fn start(
        servers: &[McpServer],
        environment: &Arc<dyn Environment>,
        routes: Option<&Arc<dyn PreviewUrls>>,
    ) -> Self {
        let mut started = Self {
            connections: Vec::with_capacity(servers.len()),
            tools:       Vec::new(),
            outcomes:    Vec::with_capacity(servers.len()),
        };
        for server in servers {
            // Boxed: the start future carries the readiness probe and the
            // route, and would otherwise weigh on every future above it.
            match Box::pin(Connection::start(server, environment, routes)).await {
                Ok((connection, tools)) => {
                    let connection = Arc::new(connection);
                    let mut summaries: Vec<McpToolSummary> = tools
                        .iter()
                        .map(|tool| McpToolSummary {
                            name:          qualified_tool_name(&server.name, &tool.name),
                            original_name: tool.name.clone(),
                        })
                        .collect();
                    summaries.sort_by(|left, right| left.name.cmp(&right.name));
                    for tool in tools {
                        started
                            .tools
                            .push(registered_tool(&connection, server, tool));
                    }
                    tracing::info!(server = %server.name, tools = summaries.len(), "MCP server ready");
                    started.outcomes.push(McpServerOutcome::Ready {
                        server: server.name.clone(),
                        tools:  summaries,
                    });
                    started.connections.push((server.name.clone(), connection));
                }
                Err(error) => {
                    let error = error.to_string();
                    tracing::error!(server = %server.name, error = %error, "MCP server failed to start");
                    started.outcomes.push(McpServerOutcome::Failed {
                        server: server.name.clone(),
                        error,
                    });
                }
            }
        }
        started
    }

    /// The tools to register with the agent, one per discovered tool, in
    /// server order then registered-name order.
    pub(crate) fn tools(&self) -> Vec<RegisteredTool> {
        self.tools.clone()
    }

    /// What became of each configured server, in configuration order.
    pub(crate) fn outcomes(&self) -> &[McpServerOutcome] {
        &self.outcomes
    }

    /// Closes every connection, stops every owned process, and releases every
    /// route. Runs after the agent shut down, so no call is in flight.
    pub(crate) async fn shutdown(&mut self) {
        for (name, connection) in self.connections.drain(..) {
            connection.close(SHUTDOWN_TIMEOUT).await;
            tracing::debug!(server = %name, "MCP server stopped");
        }
    }
}

/// The pebble tool that forwards one MCP tool to its server.
fn registered_tool(
    connection: &Arc<Connection>,
    server: &McpServer,
    tool: DiscoveredTool,
) -> RegisteredTool {
    let qualified = qualified_tool_name(&server.name, &tool.name);
    let definition = ToolDefinition::function(qualified, tool.description, tool.input_schema);
    let connection = Arc::clone(connection);
    let source = ToolSource::Mcp {
        server_name:   server.name.clone(),
        original_name: tool.name.clone(),
    };
    let server_name = server.name.clone();
    let original = tool.name;
    let executor = Arc::new(move |arguments: Value, context: ToolContext| {
        let connection = Arc::clone(&connection);
        let server_name = server_name.clone();
        let original = original.clone();
        Box::pin(async move {
            match connection
                .call(&original, arguments, context.cancel())
                .await
            {
                CallOutcome::Ok(text) => Ok(text),
                CallOutcome::ToolError(text) => Err(ToolError::execution(text)),
                CallOutcome::Failed(message) => Err(ToolError::unavailable(format!(
                    "MCP server `{server_name}` failed the call to `{original}`: {message}"
                ))),
                CallOutcome::Timeout(timeout) => Err(ToolError::execution(format!(
                    "MCP tool `{original}` on server `{server_name}` did not answer within {}s",
                    timeout.as_secs()
                ))),
                CallOutcome::Cancelled => Err(ToolError::cancelled(format!(
                    "MCP tool `{original}` on server `{server_name}` was cancelled"
                ))),
            }
        }) as _
    });
    RegisteredTool::new(definition, executor)
        .with_source(source)
        .allow_in_subagents()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_names_prefix_and_sanitize() {
        assert_eq!(
            qualified_tool_name("filesystem", "read_file"),
            "mcp__filesystem__read_file"
        );
        assert_eq!(
            qualified_tool_name("my-server", "read.file"),
            "mcp__my_server__read_file"
        );
        assert_eq!(
            qualified_tool_name("my_server", "read_file"),
            "mcp__my_server__read_file"
        );
    }

    #[test]
    fn qualified_names_parse_back() {
        let qualified = qualified_tool_name("filesystem", "read_file");
        assert_eq!(
            parse_qualified_name(&qualified),
            Some(("filesystem".to_owned(), "read_file".to_owned()))
        );
        assert_eq!(
            parse_qualified_name(&qualified_tool_name("my-server", "read.file")),
            Some(("my_server".to_owned(), "read_file".to_owned()))
        );
        assert_eq!(parse_qualified_name("not_mcp__server__tool"), None);
        assert_eq!(parse_qualified_name("mcp__serveronly"), None);
        assert_eq!(parse_qualified_name("mcp____tool"), None);
    }

    #[test]
    fn a_server_starts_with_the_documented_timeouts() {
        let server = McpServer::new("echo", McpPlacement::Http {
            url:      "http://127.0.0.1:1/mcp".into(),
            headers:  BTreeMap::new(),
            protocol: McpHttpProtocol::default(),
        });
        assert_eq!(server.name(), "echo");
        assert_eq!(server.startup_timeout(), Duration::from_secs(10));
        assert_eq!(server.tool_timeout(), Duration::from_secs(60));
        assert_eq!(McpHttpProtocol::default(), McpHttpProtocol::StreamableHttp);
    }
}
