//! The injected model-stream seam.

use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::Client;
use lithos_llm::middleware::CallContext;
use lithos_llm::types::{Error, Request, ResponseStream};

/// Opens model response streams for an agent.
///
/// Implementations must not create an async runtime. They may apply provider
/// selection, credentials, middleware, and retries before visible output.
/// The returned stream must stop when `context.cancellation()` is cancelled.
#[async_trait]
pub trait ModelService: Send + Sync {
    /// Opens one response stream.
    async fn stream(
        &self,
        request: Request,
        context: CallContext,
    ) -> StdResult<ResponseStream, Error>;
}

#[async_trait]
impl ModelService for Client {
    async fn stream(
        &self,
        request: Request,
        context: CallContext,
    ) -> StdResult<ResponseStream, Error> {
        self.stream_with_context(request, context).await
    }
}

#[async_trait]
impl<T> ModelService for Arc<T>
where
    T: ModelService + ?Sized,
{
    async fn stream(
        &self,
        request: Request,
        context: CallContext,
    ) -> StdResult<ResponseStream, Error> {
        (**self).stream(request, context).await
    }
}
