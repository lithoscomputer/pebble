//! Pebble's port routing over the driver's preview URLs.

use std::sync::Arc;

use async_trait::async_trait;
use sandbox_driver::{PreviewUrls, Sandbox};

use crate::mcp::{PortRoute, PortRouteError, PortRoutes};

/// The route from the application to a port inside `handle`'s sandbox, as
/// pebble's MCP support takes it: [`PortRoutes`] over the driver's preview
/// URLs, when the provider has them. `None` for a provider without
/// forwarding, which is where pebble reaches the port on the loopback
/// address instead.
#[must_use]
pub fn port_routes(handle: &Arc<dyn Sandbox>) -> Option<Arc<dyn PortRoutes>> {
    handle.preview_urls()?;
    Some(Arc::new(SandboxPortRoutes(Arc::clone(handle))))
}

/// [`PortRoutes`] over a sandbox handle: the driver's preview-URL facet
/// answers with the URL and headers that reach a port.
pub struct SandboxPortRoutes(Arc<dyn Sandbox>);

impl SandboxPortRoutes {
    /// The driver's facet, present whenever [`port_routes`] handed this out.
    /// A missing facet is the environment routing to none of its ports.
    fn facet(&self) -> Result<&dyn PreviewUrls, PortRouteError> {
        self.0.preview_urls().ok_or(PortRouteError::Unsupported)
    }
}

#[async_trait]
impl PortRoutes for SandboxPortRoutes {
    async fn route(&self, port: u16) -> Result<PortRoute, PortRouteError> {
        let preview = self.facet()?.preview_url(port).await.map_err(|error| {
            PortRouteError::failed_with_source(
                format!("Failed to open a route to sandbox port {port}"),
                error,
            )
        })?;
        Ok(PortRoute {
            url:     preview.url,
            headers: preview.headers,
        })
    }

    async fn release(&self, port: u16) -> Result<(), PortRouteError> {
        self.facet()?
            .release_preview_url(port)
            .await
            .map_err(|error| {
                PortRouteError::failed_with_source(
                    format!("Failed to release the route to sandbox port {port}"),
                    error,
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use sandbox_driver::{SandboxProvider as _, SandboxSource, SandboxSpec};
    use sandbox_driver_host::HostProvider;

    use super::*;

    #[tokio::test]
    async fn port_routes_answer_pebble_with_the_access_facets_preview_url() {
        let dir = tempfile::tempdir().unwrap();
        let provider = HostProvider::new();
        let handle = provider
            .create(
                &SandboxSpec::new(SandboxSource::HostDirectory)
                    .working_directory(dir.path().display().to_string()),
                None,
            )
            .await
            .unwrap();
        let routes = port_routes(&handle).expect("the host provider routes to its ports");
        let route = routes.route(8080).await.unwrap();
        assert_eq!(route, PortRoute::new("http://127.0.0.1:8080"));
        routes.release(8080).await.unwrap();
        drop((dir, provider));
    }
}
