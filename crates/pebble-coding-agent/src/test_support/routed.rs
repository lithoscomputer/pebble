//! A scripted model that keeps one script per session.
//!
//! [`ScriptedProvider`] answers calls from one queue in arrival order. A
//! parent agent and the children it spawns are separate sessions on separate
//! tasks, and which of them reaches the provider first is up to the scheduler,
//! so a queue they share hands answers to whichever session asks first.
//! [`RoutedProvider`] gives every session its own [`ScriptedProvider`] and
//! picks the script from the request itself.
//!
//! The rule is the first user message. A child's first user message is the
//! task its parent gave it, so a lane keyed on that task answers only that
//! child. The root session's requests, and any request whose first user
//! message names no lane, go to the root lane. A request whose first user
//! message names two lanes is answered with an error, so an ambiguous key
//! fails the test loudly rather than answering the wrong session.
//!
//! Non-streaming calls (compaction's summary call) route the same way: the
//! summary request's one user message is the rendered transcript, which opens
//! with the session's first prompt.
//!
//! ```
//! use pebble_coding_agent::test_support::{
//!     ScriptedCall, ScriptedProvider, routed_client, text_response,
//! };
//!
//! let (client, provider) = routed_client(
//!     ScriptedProvider::new(vec![ScriptedCall::response(text_response("root"))]),
//!     vec![(
//!         "child: review the diff",
//!         ScriptedProvider::new(vec![ScriptedCall::response(text_response("child"))]),
//!     )],
//! );
//! # let _ = (client, provider.root().call_count(), provider.lane("child: review the diff"));
//! ```

use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::Client;
use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::AdapterId;
use lithos_llm::client::ClientBuild;
use lithos_llm::types::{
    Error as LlmError, ErrorKind as LlmErrorKind, Response, ResponseStream, Role,
};

use super::scripted::{ScriptedProvider, message_text, test_catalog};

/// A provider that answers each session from that session's own script.
#[derive(Debug)]
pub struct RoutedProvider {
    id:    AdapterId,
    root:  Arc<ScriptedProvider>,
    lanes: Vec<(String, Arc<ScriptedProvider>)>,
}

impl RoutedProvider {
    /// The root session's script: every request no lane claims.
    #[must_use]
    pub fn root(&self) -> &ScriptedProvider {
        &self.root
    }

    /// The script of the lane keyed on `key`.
    ///
    /// # Panics
    ///
    /// Panics when no lane was keyed on `key`, which is a mistake in the test.
    #[must_use]
    pub fn lane(&self, key: &str) -> &ScriptedProvider {
        self.lanes
            .iter()
            .find(|(lane_key, _)| lane_key == key)
            .map_or_else(
                || panic!("no scripted lane is keyed on {key:?}"),
                |(_, lane)| lane.as_ref(),
            )
    }

    /// The lane a request belongs to, from its first user message.
    fn lane_for(&self, call: &ResolvedCall) -> StdResult<&ScriptedProvider, LlmError> {
        let opening = call
            .request()
            .messages()
            .iter()
            .find(|message| message.role() == Role::User)
            .map(message_text)
            .unwrap_or_default();
        let matched: Vec<&(String, Arc<ScriptedProvider>)> = self
            .lanes
            .iter()
            .filter(|(key, _)| opening.contains(key.as_str()))
            .collect();
        match matched.as_slice() {
            [] => Ok(&self.root),
            [(_, lane)] => Ok(lane),
            many => {
                let keys: Vec<&str> = many.iter().map(|(key, _)| key.as_str()).collect();
                Err(LlmError::new(
                    LlmErrorKind::Middleware,
                    format!(
                        "the request's first user message names {keys:?}; key each lane on \
                         text only its session's first prompt has"
                    ),
                ))
            }
        }
    }
}

#[async_trait]
impl ProviderAdapter for RoutedProvider {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> StdResult<Response, LlmError> {
        self.lane_for(call)?.complete(call).await
    }

    async fn stream(&self, call: &ResolvedCall) -> StdResult<ResponseStream, LlmError> {
        self.lane_for(call)?.stream(call).await
    }
}

/// A client whose root session answers from `root` and whose other sessions
/// answer from the lane keyed on their first user message.
///
/// A lane's key must be text that only that session's first prompt contains:
/// a request matching two keys is answered with an error.
///
/// # Panics
///
/// Panics if the client cannot be built, which is a bug in this crate.
#[must_use]
pub fn routed_client(
    root: ScriptedProvider,
    lanes: Vec<(&str, ScriptedProvider)>,
) -> (Client, Arc<RoutedProvider>) {
    let provider = Arc::new(RoutedProvider {
        id:    AdapterId::new("test-adapter"),
        root:  Arc::new(root),
        lanes: lanes
            .into_iter()
            .map(|(key, lane)| (key.to_owned(), Arc::new(lane)))
            .collect(),
    });
    let shared: Arc<dyn ProviderAdapter> = Arc::clone(&provider) as Arc<dyn ProviderAdapter>;
    let ClientBuild { client, .. } = Client::builder()
        .catalog(test_catalog())
        .adapter_arc("test", Arc::clone(&shared))
        .adapter_arc("bare", shared)
        .build()
        .expect("the routed client builds");
    (client, provider)
}
