//! A scripted model that keeps one script per session.
//!
//! [`ScriptedProvider`] answers calls from one queue in arrival order. A
//! parent agent and the children it spawns are separate sessions on separate
//! tasks, and which of them reaches the provider first is up to the scheduler,
//! so a queue they share hands answers to whichever session asks first.
//! [`RoutedProvider`] gives every session its own [`ScriptedProvider`] and
//! picks the script from the request itself.
//!
//! The rule is the opening line: the first line of the request's first user
//! message. A child's first user message is the task its parent gave it, so a
//! lane keyed on that task answers only that child. The root session's
//! requests, and any request whose opening line names no lane, go to the root
//! lane. A request whose opening line names two lanes is answered with an
//! error, so an ambiguous key fails the test loudly rather than answering the
//! wrong session.
//!
//! Non-streaming calls (compaction's summary call) route the same way: the
//! summary request's one user message is the rendered transcript, which opens
//! with the session's first prompt. Only that opening line is read, because
//! the rest of a parent's transcript quotes every `spawn_agent` call it made,
//! task and all, and a rule that read the whole message would hand the
//! parent's summary to a child's lane.
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
    Error as LlmError, ErrorKind as LlmErrorKind, Request, Response, ResponseStream, Role,
};

use super::scripted::{ScriptedProvider, message_text, test_catalog};
use crate::compaction::SUMMARY_TRANSCRIPT_PREAMBLE;

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

    /// The lane a request belongs to, from its opening line.
    fn lane_for(&self, call: &ResolvedCall) -> StdResult<&ScriptedProvider, LlmError> {
        let opening = opening_line(call.request());
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
                        "the request's opening line names {keys:?}; key each lane on text only \
                         its session's first prompt has"
                    ),
                ))
            }
        }
    }
}

/// The first line of the request's first user message, which is the line a
/// lane key is matched against.
///
/// A summarizing call's user message is the rendered transcript behind a fixed
/// preamble; the preamble is stepped over so the line read is the transcript's
/// first, which quotes the session's first prompt.
fn opening_line(request: &Request) -> String {
    let opening = request
        .messages()
        .iter()
        .find(|message| message.role() == Role::User)
        .map(message_text)
        .unwrap_or_default();
    opening
        .strip_prefix(SUMMARY_TRANSCRIPT_PREAMBLE)
        .unwrap_or(&opening)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_user(text: &str) -> Request {
        Request::builder()
            .model("test/model")
            .system("system")
            .user(text)
            .build()
            .expect("the request builds")
    }

    #[test]
    fn the_opening_line_is_the_first_line_of_the_first_user_message() {
        let request = request_with_user("child: count the files\nthen report");

        assert_eq!(opening_line(&request), "child: count the files");
    }

    #[test]
    fn a_summary_request_opens_with_the_transcripts_first_line() {
        let transcript = "User: delegate the count\n[Tool call: spawn_agent] {\"task\":\"child: \
                          count the files\"}\n";
        let request = request_with_user(&format!("{SUMMARY_TRANSCRIPT_PREAMBLE}{transcript}"));

        assert_eq!(opening_line(&request), "User: delegate the count");
    }

    #[test]
    fn a_request_without_a_user_message_has_an_empty_opening_line() {
        let request = Request::builder()
            .model("test/model")
            .system("system")
            .build()
            .expect("the request builds");

        assert_eq!(opening_line(&request), "");
    }
}
