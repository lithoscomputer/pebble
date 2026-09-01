//! Searching the web through whatever the application gave the session.

use std::fmt::Write as _;
use std::sync::Arc;

use lithos_llm::types::ToolDefinition;
use serde_json::Value;

use crate::search::{SearchProvider, SearchRequest, SearchResult};
use crate::tool::{NativeTool, RegisteredTool, required_str};
use crate::types::ToolSource;

/// How many results a call returns when the model names no number.
const DEFAULT_MAX_RESULTS: u32 = 5;

/// How many results a call returns at most, whatever the model asks for.
const MAX_RESULTS: u32 = 20;

/// Searches the web through `provider`.
///
/// The schema and the rendered results are pebble's, not the provider's: two
/// applications searching through different engines give their models the same
/// tool and the same output, which is what lets a prompt written for one work
/// on the other.
#[must_use]
pub fn make_web_search_tool(provider: Arc<dyn SearchProvider>) -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            NativeTool::WebSearch.canonical_name(),
            "Search the web when current external information is needed. Returns result titles, \
             URLs, and descriptions; use web_fetch for a specific URL.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results (default 5, max 20)"}
                },
                "required": ["query"]
            }),
        ),
        executor:   Arc::new(move |args, _ctx| {
            let provider = Arc::clone(&provider);
            Box::pin(async move {
                let query = required_str(&args, "query")?.to_owned();
                let request = SearchRequest::new(query, max_results_arg(&args));

                let results = provider.search(request).await?;
                Ok(format_results(&results))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// How many results this call asked for, bounded.
///
/// A model that asks for a hundred results gets twenty rather than an argument
/// error: the number is a preference, and refusing the call over it would cost
/// a round trip for nothing.
fn max_results_arg(args: &Value) -> u32 {
    args.get("max_results")
        .and_then(Value::as_u64)
        // A number past what a `u32` holds is past the bound anyway.
        .map_or(DEFAULT_MAX_RESULTS, |requested| {
            u32::try_from(requested).unwrap_or(MAX_RESULTS)
        })
        .clamp(1, MAX_RESULTS)
}

/// The results as the model reads them.
///
/// One numbered block per result — title, URL, summary, and a date when the
/// engine reported one — separated by blank lines. This text is contract: a
/// model reads it, so it does not change with the provider underneath.
pub(crate) fn format_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_owned();
    }

    let mut output = String::new();
    for (index, result) in results.iter().enumerate() {
        let _ = write!(
            output,
            "{}. {}\n   {}\n   {}\n",
            index + 1,
            present(&result.title, "(no title)"),
            present(&result.url, "(no url)"),
            result.snippet
        );
        if let Some(date) = present_option(result.published_at.as_deref()) {
            let _ = writeln!(output, "   {date}");
        }
        output.push('\n');
    }
    output
}

/// `value`, or `missing` when the engine returned nothing for it.
fn present<'a>(value: &'a str, missing: &'a str) -> &'a str {
    if value.is_empty() { missing } else { value }
}

/// The value, when there is one and it is not empty.
fn present_option(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, PoisonError};

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::search::{SearchError, SearchErrorKind};
    use crate::test_support::MockEnvironment;
    use crate::tool::ToolError;
    use crate::tools::testing::{context, schema_of};
    use crate::types::ToolErrorKind;

    /// A provider that answers with what it was built with, and records what it
    /// was asked.
    struct Recording {
        results:  Vec<SearchResult>,
        requests: Mutex<Vec<SearchRequest>>,
    }

    impl Recording {
        fn new(results: Vec<SearchResult>) -> Arc<Self> {
            Arc::new(Self {
                results,
                requests: Mutex::new(Vec::new()),
            })
        }

        fn requests(&self) -> Vec<SearchRequest> {
            self.requests
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    #[async_trait]
    impl SearchProvider for Recording {
        async fn search(&self, request: SearchRequest) -> Result<Vec<SearchResult>, SearchError> {
            self.requests
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(request);
            Ok(self.results.clone())
        }
    }

    /// A provider that cannot search.
    struct Down;

    #[async_trait]
    impl SearchProvider for Down {
        async fn search(&self, _request: SearchRequest) -> Result<Vec<SearchResult>, SearchError> {
            Err(SearchError::new(
                SearchErrorKind::Unavailable,
                "Search is temporarily unavailable",
            ))
        }
    }

    async fn call(provider: Arc<dyn SearchProvider>, args: Value) -> Result<String, ToolError> {
        let tool = make_web_search_tool(provider);
        (tool.executor)(args, context(MockEnvironment::default())).await
    }

    #[test]
    fn results_are_numbered_with_their_title_url_and_summary() {
        let output = format_results(&[
            SearchResult::new("Rust Lang", "https://rust-lang.org", "A systems language"),
            SearchResult::new(
                "Rust Book",
                "https://doc.rust-lang.org/book",
                "The Rust book",
            ),
        ]);

        assert_eq!(
            output,
            "1. Rust Lang\n   https://rust-lang.org\n   A systems language\n\n2. Rust Book\n   \
             https://doc.rust-lang.org/book\n   The Rust book\n\n"
        );
    }

    #[test]
    fn no_results_says_so() {
        assert_eq!(format_results(&[]), "No results found.");
    }

    #[test]
    fn a_date_is_rendered_when_the_engine_reported_one() {
        let output = format_results(&[SearchResult::new(
            "Rust Lang",
            "https://rust-lang.org",
            "A systems language",
        )
        .with_published_at("2026-01-02")]);

        assert!(output.contains("1. Rust Lang"));
        assert!(output.contains("https://rust-lang.org"));
        assert!(output.contains("A systems language"));
        assert!(output.contains("   2026-01-02\n"));
    }

    /// An engine that answers with a blank title or URL has told the model
    /// nothing; saying so is clearer than an empty line.
    #[test]
    fn a_missing_title_or_url_is_named_rather_than_left_blank() {
        let output = format_results(&[SearchResult {
            title:        String::new(),
            url:          String::new(),
            snippet:      "Something".to_owned(),
            published_at: Some(String::new()),
        }]);

        assert_eq!(output, "1. (no title)\n   (no url)\n   Something\n\n");
    }

    #[tokio::test]
    async fn a_search_asks_the_provider_and_renders_what_it_answered() {
        let provider = Recording::new(vec![SearchResult::new(
            "Rust Lang",
            "https://rust-lang.org",
            "A systems language",
        )]);

        let output = call(
            Arc::clone(&provider) as Arc<dyn SearchProvider>,
            json!({"query": "rust"}),
        )
        .await
        .expect("the search runs");

        assert!(output.contains("1. Rust Lang"));
        assert_eq!(provider.requests(), vec![SearchRequest::new("rust", 5)]);
    }

    #[tokio::test]
    async fn a_request_for_more_results_than_the_tool_allows_is_bounded() {
        let provider = Recording::new(Vec::new());

        for (asked, expected) in [(3_u64, 3_u32), (100, 20), (0, 1)] {
            call(
                Arc::clone(&provider) as Arc<dyn SearchProvider>,
                json!({"query": "rust", "max_results": asked}),
            )
            .await
            .expect("the search runs");
            let request = provider.requests().pop().expect("the provider was asked");
            assert_eq!(request.max_results, expected, "asked for {asked}");
        }
    }

    #[tokio::test]
    async fn web_search_missing_query_returns_error() {
        let error = call(Recording::new(Vec::new()), json!({}))
            .await
            .expect_err("a search needs something to search for");

        assert!(
            error.message().contains("query"),
            "the error should name the missing query, got: {}",
            error.message()
        );
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    #[tokio::test]
    async fn a_provider_that_cannot_search_says_so_in_its_own_words() {
        let error = call(Arc::new(Down), json!({"query": "rust"}))
            .await
            .expect_err("the provider is down");

        assert_eq!(error.message(), "Search is temporarily unavailable");
        assert_eq!(error.kind(), ToolErrorKind::Unavailable);
    }

    /// The schema belongs to pebble, not to whichever engine is underneath, so
    /// two applications give their models the same tool.
    #[test]
    fn the_schema_is_the_same_whichever_provider_is_underneath() {
        let first = make_web_search_tool(Recording::new(Vec::new()));
        let second = make_web_search_tool(Arc::new(Down));

        assert_eq!(schema_of(&first), schema_of(&second));
        assert_eq!(first.definition.description, second.definition.description);
        assert_eq!(first.definition.name, "web_search");
    }
}
