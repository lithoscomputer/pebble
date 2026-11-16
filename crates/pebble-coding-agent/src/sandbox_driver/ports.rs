//! Pebble's port routing over the driver's ports.

use std::num::NonZeroU16;
use std::sync::Arc;

use async_trait::async_trait;
use sandbox_driver::{Error as DriverError, Expose, PortSupport, Sandbox};

use crate::mcp::{PortRoute, PortRouteError, PortRoutes};

/// The route from the application to a port inside `handle`'s sandbox, as
/// pebble's MCP support takes it: [`PortRoutes`] over the driver's
/// [`Ports`](sandbox_driver::Ports), when the provider can expose ports.
/// `None` for a provider without forwarding, which is where pebble reaches
/// the port on the loopback address instead.
///
/// `support` is the provider descriptor's
/// [`ports`](sandbox_driver::Descriptor::ports). Parity gap: the old handle
/// answered this itself (its preview-URL facet was `None`); a new handle always
/// has `ports()`, and only the provider's descriptor, which the handle does not
/// carry, says whether they work.
#[must_use]
pub fn port_routes(
    handle: &Arc<dyn Sandbox>,
    support: &PortSupport,
) -> Option<Arc<dyn PortRoutes>> {
    if *support == PortSupport::None {
        return None;
    }
    Some(Arc::new(SandboxPortRoutes {
        handle:  Arc::clone(handle),
        support: support.clone(),
    }))
}

/// [`PortRoutes`] over a sandbox handle: the driver's ports answer with the
/// URL and headers that reach a port.
pub struct SandboxPortRoutes {
    handle:  Arc<dyn Sandbox>,
    support: PortSupport,
}

impl SandboxPortRoutes {
    /// The port as the driver takes it, when the provider can expose it.
    fn port(&self, port: u16) -> Result<NonZeroU16, PortRouteError> {
        let port = NonZeroU16::new(port)
            .ok_or_else(|| PortRouteError::failed("Port 0 is not a sandbox port"))?;
        match &self.support {
            PortSupport::Any => Ok(port),
            PortSupport::Selected(ports) if ports.contains(&port) => Ok(port),
            _ => Err(PortRouteError::Unsupported),
        }
    }
}

/// A refusal is the environment routing to none of its ports; anything
/// else is this route failing.
fn route_error(message: String, error: DriverError) -> PortRouteError {
    match error {
        DriverError::Unsupported { .. } => PortRouteError::Unsupported,
        error => PortRouteError::failed_with_source(message, error),
    }
}

#[async_trait]
impl PortRoutes for SandboxPortRoutes {
    async fn route(&self, port: u16) -> Result<PortRoute, PortRouteError> {
        let exposure = self
            .handle
            .ports()
            .expose(self.port(port)?, Expose {
                signed: false,
                ttl_ms: None,
            })
            .await
            .map_err(|error| {
                route_error(
                    format!("Failed to open a route to sandbox port {port}"),
                    error,
                )
            })?;
        Ok(PortRoute {
            url:     exposure.url,
            headers: exposure.headers,
        })
    }

    async fn release(&self, port: u16) -> Result<(), PortRouteError> {
        self.handle
            .ports()
            .release(self.port(port)?)
            .await
            .map_err(|error| {
                route_error(
                    format!("Failed to release the route to sandbox port {port}"),
                    error,
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox_driver::testing::host_sandbox;

    #[tokio::test]
    async fn port_routes_answer_pebble_with_the_providers_exposure() {
        let (_catalog, handle) = host_sandbox().await;
        let routes =
            port_routes(&handle, &PortSupport::Any).expect("the host provider routes to its ports");
        let route = routes.route(8080).await.unwrap();
        assert_eq!(route, PortRoute::new("http://127.0.0.1:8080"));
        routes.release(8080).await.unwrap();
    }

    #[tokio::test]
    async fn a_provider_without_ports_has_no_routes_and_unlisted_ports_are_refused() {
        let (_catalog, handle) = host_sandbox().await;
        assert!(port_routes(&handle, &PortSupport::None).is_none());
        let selected = PortSupport::Selected([NonZeroU16::new(3000).unwrap()].into());
        let routes = port_routes(&handle, &selected).unwrap();
        assert!(routes.route(3000).await.is_ok());
        assert!(matches!(
            routes.route(8080).await,
            Err(PortRouteError::Unsupported)
        ));
    }
}
