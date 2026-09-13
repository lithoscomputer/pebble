//! The route from the application to a port inside its environment.
//!
//! An MCP server placed in the environment
//! ([`McpPlacement::Environment`](super::McpPlacement::Environment)) listens
//! on a port there, and pebble has to reach that port from where it runs: on
//! the host's own loopback address when the environment shares the host's
//! network, through a forward the sandbox opens into a container, or at a
//! remote provider's preview link with the header its requests need. That is
//! the application's knowledge, so the application supplies it as a
//! [`PortRoutes`] through
//! [`CodingAgentBuilder::port_routes`](crate::CodingAgentBuilder::port_routes);
//! pebble asks it once per server and releases what it opened.
//!
//! Pebble owns the trait so that an application implements it over whatever
//! its sandbox offers, with no crate shared between them: fabro over its run
//! sandbox's preview-URL facet, petri over its execution environment's
//! `preview_url`. A trait object crosses a crate boundary only when both
//! sides compile the same crate version, so a trait borrowed from a sandbox
//! crate would make every application pin that crate at pebble's revision.

use std::collections::BTreeMap;
use std::error::Error as StdError;

use async_trait::async_trait;

/// How pebble reaches a port inside the environment an MCP server was
/// launched in.
///
/// An implementor is shared across tasks and called from the agent's build
/// and shutdown paths. A call may take as long as opening a forward takes;
/// pebble bounds nothing here, so an implementor whose route can hang bounds
/// it itself. [`route`](Self::route) is asked once per server, before the
/// server's readiness probe; [`release`](Self::release) is called once for
/// every route that was opened, when the server fails to start and when the
/// agent shuts down.
#[async_trait]
pub trait PortRoutes: Send + Sync {
    /// Opens (or looks up) a route to `port` and returns where to connect.
    ///
    /// [`PortRouteError::Unsupported`] says the environment routes to none of
    /// its ports; pebble reports it as the reason the server has no route,
    /// and the server is skipped.
    async fn route(&self, port: u16) -> Result<PortRoute, PortRouteError>;

    /// Releases the route opened for `port`.
    ///
    /// Idempotent: a port never routed, or released twice, succeeds. The
    /// default releases nothing, for a route that holds nothing open. Pebble
    /// logs a failure and does nothing else with it.
    async fn release(&self, port: u16) -> Result<(), PortRouteError> {
        let _ = port;
        Ok(())
    }
}

/// Where a routed port is reached from the application.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortRoute {
    /// The URL that reaches the port. The path the server speaks at, when
    /// its placement names one, is appended to it.
    pub url:     String,
    /// Headers every request to `url` carries, such as a preview link's
    /// token.
    pub headers: BTreeMap<String, String>,
}

impl PortRoute {
    /// A route at `url` whose requests need no headers.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url:     url.into(),
            headers: BTreeMap::new(),
        }
    }
}

/// Why a route could not be opened or released.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PortRouteError {
    /// The environment routes to none of its ports.
    #[error("the environment does not route to its ports")]
    Unsupported,
    /// The environment routes to its ports, but this route failed.
    #[error("{message}")]
    Failed {
        /// What failed, written for the server's failure report.
        message: String,
        /// The underlying failure, when there is one.
        #[source]
        source:  Option<Box<dyn StdError + Send + Sync + 'static>>,
    },
}

impl PortRouteError {
    /// A failure with no underlying cause.
    #[must_use]
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed {
            message: message.into(),
            source:  None,
        }
    }

    /// A failure that keeps `source` as its cause.
    #[must_use]
    pub fn failed_with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::Failed {
            message: message.into(),
            source:  Some(Box::new(source)),
        }
    }

    /// The message followed by each of its causes, `: `-joined: the one-line
    /// reason a server's failure report carries.
    #[must_use]
    pub fn detail(&self) -> String {
        let mut rendered = self.to_string();
        let mut current = StdError::source(self);
        while let Some(cause) = current {
            rendered.push_str(": ");
            rendered.push_str(&cause.to_string());
            current = cause.source();
        }
        rendered
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn a_route_without_headers_is_built_from_its_url() {
        let route = PortRoute::new("http://127.0.0.1:8080");
        assert_eq!(route, PortRoute {
            url:     "http://127.0.0.1:8080".to_owned(),
            headers: BTreeMap::new(),
        });
    }

    #[test]
    fn an_unsupported_route_says_so() {
        assert_eq!(
            PortRouteError::Unsupported.detail(),
            "the environment does not route to its ports"
        );
    }

    #[test]
    fn a_failed_route_renders_its_causes_on_one_line() {
        let plain = PortRouteError::failed("opening the forward");
        assert_eq!(plain.detail(), "opening the forward");
        assert!(StdError::source(&plain).is_none());

        let caused = PortRouteError::failed_with_source(
            "opening the forward",
            io::Error::new(io::ErrorKind::ConnectionRefused, "connection refused"),
        );
        assert_eq!(caused.to_string(), "opening the forward");
        assert_eq!(caused.detail(), "opening the forward: connection refused");
        assert!(StdError::source(&caused).is_some());
    }
}
