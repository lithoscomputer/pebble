//! Searching the web, when the application can.
//!
//! Pebble does not talk to a search engine. It defines the seam — a
//! [`SearchProvider`] the application implements over whatever it has, an API
//! key it holds, a service it runs, a cache it prefers — and owns everything
//! the model sees: the tool, its schema, and the way results are written out.
//!
//! A session with no provider registers no `web_search` tool, so a model is
//! never told it can search and then refused.

use std::error::Error as StdError;

use async_trait::async_trait;

use crate::tool::ToolError;
use crate::types::ToolErrorKind;

/// One search the model asked for.
///
/// Built by pebble from the model's arguments and handed to the provider, so a
/// provider reads it rather than building one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SearchRequest {
    /// What to search for.
    pub query:       String,
    /// How many results to return at most.
    ///
    /// Pebble bounds what the model asked for to between 1 and 20 before
    /// building the request, so that is what a provider sees on a search a
    /// session ran. A request an application builds itself carries whatever it
    /// says, so a provider that cannot answer for any number should bound it
    /// again.
    pub max_results: u32,
}

impl SearchRequest {
    /// A request for `query`, with `max_results` results at most.
    #[must_use]
    pub fn new(query: impl Into<String>, max_results: u32) -> Self {
        Self {
            query: query.into(),
            max_results,
        }
    }
}

/// One result a search came back with.
///
/// The provider fills in what its engine returned. An empty string is the same
/// as nothing: the renderer says `(no title)` or `(no url)` rather than showing
/// a blank line the model has to interpret.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SearchResult {
    /// The page's title.
    pub title:        String,
    /// Where the page is.
    pub url:          String,
    /// The engine's summary of the page.
    pub snippet:      String,
    /// When the page was published, as the engine reported it. Rendered
    /// verbatim, so it is whatever the engine calls a date.
    pub published_at: Option<String>,
}

impl SearchResult {
    /// A result with no publication date.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
    ) -> Self {
        Self {
            title:        title.into(),
            url:          url.into(),
            snippet:      snippet.into(),
            published_at: None,
        }
    }

    /// The same result, dated.
    #[must_use]
    pub fn with_published_at(mut self, published_at: impl Into<String>) -> Self {
        self.published_at = Some(published_at.into());
        self
    }
}

/// Why a search did not produce results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SearchErrorKind {
    /// The provider cannot search at all right now — no credential, a service
    /// that is down, a quota that is spent.
    Unavailable,
    /// The provider refused this particular request, such as a query longer
    /// than the engine accepts.
    InvalidRequest,
    /// The search ran and failed.
    Execution,
}

/// A failed search.
///
/// The message reaches the model, so it says what happened without naming a
/// credential or a URL the application would rather keep to itself. The
/// underlying failure stays attached as the error's source, for logs.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct SearchError {
    kind:    SearchErrorKind,
    message: String,
    #[source]
    source:  Option<Box<dyn StdError + Send + Sync + 'static>>,
}

impl SearchError {
    /// Builds an error with no underlying cause.
    #[must_use]
    pub fn new(kind: SearchErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    /// Builds an error that keeps `source` as its cause.
    #[must_use]
    pub fn with_source(
        kind: SearchErrorKind,
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// The category of this failure.
    #[must_use]
    pub const fn kind(&self) -> SearchErrorKind {
        self.kind
    }

    /// The model-facing message, without its causes.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<SearchError> for ToolError {
    /// Carries a failed search to the model.
    ///
    /// The provider's message is already written for the model, so it is kept
    /// as-is. Only the kind is translated, and only what the search failed *of*
    /// travels on as the cause: wrapping the whole search error would repeat
    /// its message as the first line of every log entry, once as the message
    /// and once as its own first cause.
    fn from(error: SearchError) -> Self {
        let kind = match error.kind() {
            SearchErrorKind::Unavailable => ToolErrorKind::Unavailable,
            SearchErrorKind::InvalidRequest => ToolErrorKind::InvalidArguments,
            SearchErrorKind::Execution => ToolErrorKind::Execution,
        };
        let SearchError {
            message, source, ..
        } = error;
        match source {
            Some(cause) => Self::with_boxed_source(kind, message, cause),
            None => Self::new(kind, message),
        }
    }
}

/// Where a session's web searches go.
///
/// An application installs one with
/// [`SessionBuilder::search_provider`](crate::advanced::SessionBuilder::search_provider);
/// a session without one advertises no search tool.
///
/// Implementations run inside a tool call, so they hold the round open until
/// they answer. A provider that talks to a network service should bound its own
/// wait rather than relying on the session's.
#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Runs one search.
    ///
    /// Returning an empty list is a normal answer: the model is told the search
    /// found nothing, which is different from the search failing.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] when the search could not be run or failed.
    async fn search(&self, request: SearchRequest) -> Result<Vec<SearchResult>, SearchError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("connection reset")]
    struct Cause;

    #[test]
    fn a_result_is_built_from_its_three_required_parts() {
        let result = SearchResult::new("Rust", "https://rust-lang.org", "A systems language");

        assert_eq!(result.title, "Rust");
        assert_eq!(result.published_at, None);
        assert_eq!(
            result
                .with_published_at("2026-01-02")
                .published_at
                .as_deref(),
            Some("2026-01-02")
        );
    }

    #[test]
    fn a_failed_search_keeps_its_category_and_its_cause() {
        let error = SearchError::with_source(
            SearchErrorKind::Unavailable,
            "Search is not configured",
            Cause,
        );

        let tool_error = ToolError::from(error);
        assert_eq!(tool_error.kind(), ToolErrorKind::Unavailable);
        assert_eq!(tool_error.message(), "Search is not configured");
        // The message is carried over rather than wrapped, so the log-facing
        // chain says what happened once and then what it happened of.
        assert_eq!(
            tool_error.detail(),
            "Search is not configured\n  caused by: connection reset"
        );
    }

    #[test]
    fn a_failed_search_with_nothing_underneath_it_has_no_cause() {
        let tool_error = ToolError::from(SearchError::new(
            SearchErrorKind::Execution,
            "The engine answered with nothing",
        ));

        assert_eq!(tool_error.detail(), "The engine answered with nothing");
    }

    #[test]
    fn each_search_failure_maps_to_the_tool_failure_it_means() {
        for (kind, expected) in [
            (SearchErrorKind::Unavailable, ToolErrorKind::Unavailable),
            (
                SearchErrorKind::InvalidRequest,
                ToolErrorKind::InvalidArguments,
            ),
            (SearchErrorKind::Execution, ToolErrorKind::Execution),
        ] {
            let error = ToolError::from(SearchError::new(kind, "no"));
            assert_eq!(error.kind(), expected, "{kind:?}");
        }
    }
}
