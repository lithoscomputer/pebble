//! The search providers pebble ships.
//!
//! [`Brave`] searches through the [Brave Search API]; [`Venice`] through
//! [Venice]'s search endpoint. Each implements [`SearchProvider`] and is built
//! from an API key and a [`reqwest::Client`] the application owns, so proxies,
//! TLS, and timeouts are the application's to set; [`default_client`] builds
//! one for an application with no client of its own. Both send the model's
//! query as it was written and hand back what the engine answered, in the
//! engine's order, bounded to the twenty results pebble's tool allows at most.
//!
//! Neither reads a credential from the environment, and neither is installed
//! by default: the application decides which engine it has a key for and
//! gives that provider to
//! [`CodingAgentBuilder::search_provider`](crate::CodingAgentBuilder::search_provider).
//!
//! What a failure says reaches the model, so a message names the engine and
//! what went wrong, never the key or the URL; the underlying `reqwest` error
//! stays attached as the [`SearchError`]'s source, for logs.
//!
//! ```
//! use std::sync::Arc;
//!
//! use pebble_coding_agent::extensions::SearchProvider;
//! use pebble_coding_agent::search::providers::{self, Brave};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let client = providers::default_client()?;
//! let search: Arc<dyn SearchProvider> = Arc::new(Brave::new("brave-api-key", client));
//! // CodingAgent::builder(model_client, environment).search_provider(search)
//! # Ok(())
//! # }
//! ```
//!
//! [Brave Search API]: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
//! [Venice]: https://docs.venice.ai/

use std::fmt;
use std::fmt::Write as _;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, Response, StatusCode};
use serde_json::Value;

use crate::search::seam::{
    SearchError, SearchErrorKind, SearchProvider, SearchRequest, SearchResult,
};

const BRAVE_SEARCH_URL: &str = "https://api.search.brave.com/res/v1/web/search";
const VENICE_SEARCH_URL: &str = "https://api.venice.ai/api/v1/augment/search";

/// The longest query Venice accepts.
const VENICE_QUERY_MAX_CHARS: usize = 400;

/// How long a Venice search is given, whatever the client allows: the endpoint
/// runs a search and then reads the pages it found, so it answers slower than
/// an engine that returns its index.
const VENICE_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// The most results either engine is asked for. It is also the most the
/// `web_search` tool passes on, so a request a session built is already within
/// it; one an application built is bounded here.
const MAX_RESULTS: u32 = 20;

/// How long [`default_client`] waits for an answer.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// A client for these providers, for an application with none of its own.
///
/// It verifies TLS with rustls and gives a search thirty seconds. An
/// application that has a client already — with its proxy, its certificate
/// store, its timeouts — passes that instead.
///
/// # Errors
///
/// Returns the error `reqwest` reports when the TLS backend cannot be set up.
pub fn default_client() -> Result<Client, reqwest::Error> {
    Client::builder().timeout(DEFAULT_TIMEOUT).build()
}

/// Brave Search's web search API.
///
/// One `GET` per search, authenticated with the subscription token, asking
/// for `count` results of `q`. A result's snippet is the engine's description
/// of the page; Brave reports no publication date pebble renders. A search
/// waits as long as the client allows, which for [`default_client`] is
/// thirty seconds.
#[derive(Clone)]
pub struct Brave {
    api_key:  String,
    endpoint: String,
    client:   Client,
}

impl Brave {
    /// A provider that authenticates with `api_key` and sends its searches
    /// through `client`.
    #[must_use]
    pub fn new(api_key: impl Into<String>, client: Client) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: BRAVE_SEARCH_URL.to_owned(),
            client,
        }
    }

    /// The same provider, sending its searches to `url` instead of the public
    /// API: a proxy, a relay, or a test double that speaks Brave's protocol.
    #[must_use]
    pub fn endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = url.into();
        self
    }
}

impl fmt::Debug for Brave {
    /// Names the endpoint and leaves the key out.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Brave")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SearchProvider for Brave {
    async fn search(&self, request: SearchRequest) -> Result<Vec<SearchResult>, SearchError> {
        let count = bounded(request.max_results).to_string();
        let response = self
            .client
            .get(&self.endpoint)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .query(&[("q", request.query.as_str()), ("count", count.as_str())])
            .send()
            .await
            .map_err(|error| request_failed(Service::Brave, error))?;

        let body = successful_json(Service::Brave, response).await?;
        Ok(brave_results(&body))
    }
}

/// Venice's search endpoint.
///
/// One `POST` per search, with a bearer token, asking Venice to run the query
/// through Brave and return `limit` results. Venice reads the pages it finds,
/// so a result's snippet is page content and most results carry the date
/// Venice found on the page. A query longer than four hundred characters is
/// refused before it is sent, as Venice would refuse it.
#[derive(Clone)]
pub struct Venice {
    api_key:  String,
    endpoint: String,
    client:   Client,
}

impl Venice {
    /// A provider that authenticates with `api_key` and sends its searches
    /// through `client`.
    #[must_use]
    pub fn new(api_key: impl Into<String>, client: Client) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: VENICE_SEARCH_URL.to_owned(),
            client,
        }
    }

    /// The same provider, sending its searches to `url` instead of the public
    /// API: a proxy, a relay, or a test double that speaks Venice's protocol.
    #[must_use]
    pub fn endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = url.into();
        self
    }
}

impl fmt::Debug for Venice {
    /// Names the endpoint and leaves the key out.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Venice")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SearchProvider for Venice {
    async fn search(&self, request: SearchRequest) -> Result<Vec<SearchResult>, SearchError> {
        if request.query.chars().count() > VENICE_QUERY_MAX_CHARS {
            return Err(SearchError::new(
                SearchErrorKind::InvalidRequest,
                format!(
                    "query exceeds Venice Search maximum of {VENICE_QUERY_MAX_CHARS} characters"
                ),
            ));
        }

        let response = self
            .client
            .post(&self.endpoint)
            .timeout(VENICE_REQUEST_TIMEOUT)
            .bearer_auth(&self.api_key)
            .header("Accept", "application/json")
            .json(&serde_json::json!({
                "query": request.query,
                "limit": bounded(request.max_results),
                "search_provider": "brave",
            }))
            .send()
            .await
            .map_err(|error| request_failed(Service::Venice, error))?;

        let body = successful_json(Service::Venice, response).await?;
        Ok(venice_results(&body))
    }
}

/// Which engine a failure is about, as the model reads it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Service {
    Brave,
    Venice,
}

impl fmt::Display for Service {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Brave => "Brave Search API",
            Self::Venice => "Venice Search API",
        })
    }
}

/// `max_results`, within what the engines accept.
fn bounded(max_results: u32) -> u32 {
    max_results.clamp(1, MAX_RESULTS)
}

/// A search that got no answer.
///
/// An engine that could not be reached is unavailable, and the model is told
/// so; a search that timed out or failed on the wire is a failed search, which
/// the model may try again.
fn request_failed(service: Service, error: reqwest::Error) -> SearchError {
    let (kind, what) = if error.is_timeout() {
        (SearchErrorKind::Execution, "timed out")
    } else if error.is_connect() {
        (SearchErrorKind::Unavailable, "could not be reached")
    } else {
        (SearchErrorKind::Execution, "request failed")
    };
    SearchError::with_source(kind, format!("{service} {what}"), error)
}

/// The answer's JSON body, or the failure its status means.
async fn successful_json(service: Service, response: Response) -> Result<Value, SearchError> {
    if !response.status().is_success() {
        return Err(status_failure(service, &response));
    }
    response.json().await.map_err(|error| {
        SearchError::with_source(
            SearchErrorKind::Execution,
            format!("Failed to parse {service} response"),
            error,
        )
    })
}

/// A status the engine refused with.
///
/// Venice says how much credit is left when it wants payment, in a header the
/// message repeats so the model, and whoever reads the log, know why.
fn status_failure(service: Service, response: &Response) -> SearchError {
    let status = response.status();
    let mut message = format!("{service} returned status {status}");
    if service == Service::Venice && status == StatusCode::PAYMENT_REQUIRED {
        if let Some(balance) = header_str(response, "x-venice-balance-usd") {
            let _ = write!(message, " (balance USD {balance})");
        } else if let Some(balance) = header_str(response, "x-venice-balance-diem") {
            let _ = write!(message, " (balance DIEM {balance})");
        }
    }
    SearchError::new(status_kind(status), message)
}

/// What a status says about whether the next search could work.
///
/// A rejected key, spent credit, and a rate limit will refuse the next search
/// too; a request the engine could not accept was this request's fault; and
/// anything else is the engine failing on this search.
fn status_kind(status: StatusCode) -> SearchErrorKind {
    match status {
        StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::PAYMENT_REQUIRED
        | StatusCode::TOO_MANY_REQUESTS => SearchErrorKind::Unavailable,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
            SearchErrorKind::InvalidRequest
        }
        _ => SearchErrorKind::Execution,
    }
}

/// The header's value, when it is there and is text.
fn header_str<'a>(response: &'a Response, name: &str) -> Option<&'a str> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
}

/// Brave's `web.results`, each with its title, URL, and description.
fn brave_results(body: &Value) -> Vec<SearchResult> {
    results_in(body.get("web").and_then(|web| web.get("results")))
        .map(|result| {
            SearchResult::new(
                text(result, "title"),
                text(result, "url"),
                text(result, "description"),
            )
        })
        .collect()
}

/// Venice's `results`, each with its title, URL, content, and the date when
/// Venice found one.
fn venice_results(body: &Value) -> Vec<SearchResult> {
    results_in(body.get("results"))
        .map(|result| {
            let hit = SearchResult::new(
                text(result, "title"),
                text(result, "url"),
                text(result, "content"),
            );
            match optional_text(result, "date") {
                Some(date) => hit.with_published_at(date),
                None => hit,
            }
        })
        .collect()
}

/// The results at `value`, or none when the engine sent no array there.
fn results_in(value: Option<&Value>) -> impl Iterator<Item = &Value> {
    value.and_then(Value::as_array).into_iter().flatten()
}

/// The text at `key`, or nothing when there is none.
///
/// An empty string is what [`SearchResult`] means by a missing field: the
/// renderer names a missing title or URL, so the provider does not.
fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// The text at `key`, when there is some.
fn optional_text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;
    use std::sync::{Arc, Mutex, PoisonError};

    use axum::Router;
    use axum::body::to_bytes;
    use axum::extract::{Request, State};
    use axum::http::{HeaderMap, HeaderValue, Method};
    use axum::response::{IntoResponse, Response};
    use serde_json::json;
    use tokio::net::TcpListener;

    use super::*;

    /// One request the fake engine received.
    #[derive(Debug, Clone)]
    struct Received {
        method:  Method,
        path:    String,
        query:   Option<String>,
        headers: HeaderMap,
        body:    String,
    }

    impl Received {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers.get(name).and_then(|value| value.to_str().ok())
        }

        fn json(&self) -> Value {
            serde_json::from_str(&self.body).expect("a JSON body")
        }
    }

    /// What the fake engine answers every request with.
    #[derive(Debug, Clone)]
    struct Answer {
        status:  StatusCode,
        headers: Vec<(&'static str, &'static str)>,
        body:    String,
    }

    impl Answer {
        fn json(body: &Value) -> Self {
            Self {
                status:  StatusCode::OK,
                headers: Vec::new(),
                body:    body.to_string(),
            }
        }

        fn status(status: StatusCode) -> Self {
            Self {
                status,
                headers: Vec::new(),
                body: String::new(),
            }
        }

        fn header(mut self, name: &'static str, value: &'static str) -> Self {
            self.headers.push((name, value));
            self
        }
    }

    type Recorded = Arc<Mutex<Vec<Received>>>;

    /// A search engine in the test process: it records what it is asked and
    /// answers with what it was started with.
    struct Engine {
        url:      String,
        received: Recorded,
    }

    impl Engine {
        async fn start(answer: Answer) -> Self {
            let received = Recorded::default();
            let app = Router::new()
                .fallback(record_and_answer)
                .with_state((answer, Arc::clone(&received)));
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind the fake engine");
            let url = format!(
                "http://{}",
                listener.local_addr().expect("the fake engine's address")
            );
            tokio::spawn(async move {
                axum::serve(listener, app)
                    .await
                    .expect("serve the fake engine");
            });
            Self { url, received }
        }

        fn endpoint(&self, path: &str) -> String {
            format!("{}{path}", self.url)
        }

        fn received(&self) -> Vec<Received> {
            self.received
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }

        fn only_request(&self) -> Received {
            let mut received = self.received();
            assert_eq!(received.len(), 1, "one request was expected: {received:?}");
            received.remove(0)
        }
    }

    async fn record_and_answer(
        State((answer, received)): State<(Answer, Recorded)>,
        request: Request,
    ) -> Response {
        let (parts, body) = request.into_parts();
        let body = to_bytes(body, usize::MAX)
            .await
            .expect("read the request body");
        received
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Received {
                method:  parts.method,
                path:    parts.uri.path().to_owned(),
                query:   parts.uri.query().map(str::to_owned),
                headers: parts.headers,
                body:    String::from_utf8(body.to_vec()).expect("a UTF-8 body"),
            });
        let mut response = (answer.status, answer.body).into_response();
        for (name, value) in answer.headers {
            response
                .headers_mut()
                .insert(name, HeaderValue::from_static(value));
        }
        response
    }

    /// A client that talks to the fake engine directly. Proxy discovery is
    /// for the public APIs, not the loopback address.
    fn client() -> Client {
        Client::builder()
            .no_proxy()
            .build()
            .expect("a client with no proxy")
    }

    /// The address of a listener nothing is listening on.
    async fn closed_port() -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a port to close");
        let url = format!(
            "http://{}/search",
            listener.local_addr().expect("the closed port's address")
        );
        drop(listener);
        url
    }

    fn brave(engine: &Engine) -> Brave {
        Brave::new("brave-key", client()).endpoint(engine.endpoint("/search"))
    }

    fn venice(engine: &Engine) -> Venice {
        Venice::new("venice-key", client()).endpoint(engine.endpoint("/augment"))
    }

    #[tokio::test]
    async fn brave_asks_for_the_query_with_its_token_and_keeps_the_engines_order() {
        let engine = Engine::start(Answer::json(&json!({
            "web": {"results": [
                {"title": "One", "url": "https://one", "description": "first"},
                {"title": "Two", "url": "https://two", "description": "second"}
            ]}
        })))
        .await;

        let results = brave(&engine)
            .search(SearchRequest::new("pebble agent", 2))
            .await
            .expect("the search runs");

        let request = engine.only_request();
        assert_eq!(request.method, Method::GET);
        assert_eq!(request.path, "/search");
        assert_eq!(request.query.as_deref(), Some("q=pebble+agent&count=2"));
        assert_eq!(request.header("x-subscription-token"), Some("brave-key"));
        assert_eq!(request.header("accept"), Some("application/json"));
        assert_eq!(results, vec![
            SearchResult::new("One", "https://one", "first"),
            SearchResult::new("Two", "https://two", "second"),
        ]);
    }

    #[tokio::test]
    async fn venice_posts_the_query_with_its_bearer_token_and_keeps_the_date() {
        let engine = Engine::start(Answer::json(&json!({
            "results": [
                {"title": "One", "url": "https://one", "content": "first", "date": "2026-01-01"},
                {"title": "Two", "url": "https://two", "content": "second"}
            ]
        })))
        .await;

        let results = venice(&engine)
            .search(SearchRequest::new("pebble", 5))
            .await
            .expect("the search runs");

        let request = engine.only_request();
        assert_eq!(request.method, Method::POST);
        assert_eq!(request.path, "/augment");
        assert_eq!(request.header("authorization"), Some("Bearer venice-key"));
        assert_eq!(request.header("accept"), Some("application/json"));
        assert_eq!(request.header("content-type"), Some("application/json"));
        assert_eq!(
            request.json(),
            json!({"query": "pebble", "limit": 5, "search_provider": "brave"})
        );
        assert_eq!(results, vec![
            SearchResult::new("One", "https://one", "first").with_published_at("2026-01-01"),
            SearchResult::new("Two", "https://two", "second"),
        ]);
    }

    #[tokio::test]
    async fn venice_refuses_a_query_longer_than_the_engine_accepts_without_sending_it() {
        let engine = Engine::start(Answer::json(&json!({"results": []}))).await;

        let error = venice(&engine)
            .search(SearchRequest::new(
                "x".repeat(VENICE_QUERY_MAX_CHARS + 1),
                5,
            ))
            .await
            .expect_err("the query is too long");

        assert_eq!(error.kind(), SearchErrorKind::InvalidRequest);
        assert_eq!(
            error.message(),
            "query exceeds Venice Search maximum of 400 characters"
        );
        assert!(engine.received().is_empty(), "nothing was sent");

        // Exactly the limit is still a query.
        venice(&engine)
            .search(SearchRequest::new("x".repeat(VENICE_QUERY_MAX_CHARS), 5))
            .await
            .expect("a query at the limit is sent");
        assert_eq!(engine.received().len(), 1);
    }

    #[tokio::test]
    async fn the_number_of_results_asked_for_is_bounded_to_what_the_engines_accept() {
        let engine = Engine::start(Answer::json(&json!({}))).await;

        for (asked, expected) in [(0, 1), (7, 7), (100, 20)] {
            brave(&engine)
                .search(SearchRequest::new("pebble", asked))
                .await
                .expect("the search runs");
            venice(&engine)
                .search(SearchRequest::new("pebble", asked))
                .await
                .expect("the search runs");

            let mut received = engine.received();
            let to_venice = received.pop().expect("Venice was asked");
            let to_brave = received.pop().expect("Brave was asked");
            assert_eq!(
                to_brave.query.as_deref(),
                Some(format!("q=pebble&count={expected}").as_str()),
                "asked Brave for {asked}"
            );
            assert_eq!(
                to_venice.json()["limit"],
                json!(expected),
                "asked Venice for {asked}"
            );
        }
    }

    /// The renderer names a missing title or URL; the provider leaves the
    /// field empty rather than deciding what to call it.
    #[tokio::test]
    async fn a_result_missing_a_field_leaves_it_empty_for_the_renderer() {
        let brave_engine = Engine::start(Answer::json(&json!({
            "web": {"results": [{"title": "", "description": null, "extra": 1}]}
        })))
        .await;
        let venice_engine = Engine::start(Answer::json(&json!({
            "results": [{"url": "https://one", "date": ""}]
        })))
        .await;

        let from_brave = brave(&brave_engine)
            .search(SearchRequest::new("pebble", 1))
            .await
            .expect("the search runs");
        let from_venice = venice(&venice_engine)
            .search(SearchRequest::new("pebble", 1))
            .await
            .expect("the search runs");

        assert_eq!(from_brave, vec![SearchResult::new("", "", "")]);
        assert_eq!(from_venice, vec![SearchResult::new("", "https://one", "")]);
        assert_eq!(from_venice[0].published_at, None);
    }

    #[tokio::test]
    async fn an_answer_with_no_results_is_an_empty_search_not_a_failure() {
        let brave_engine = Engine::start(Answer::json(&json!({"query": {}}))).await;
        let venice_engine = Engine::start(Answer::json(&json!({"results": null}))).await;

        let from_brave = brave(&brave_engine)
            .search(SearchRequest::new("pebble", 3))
            .await
            .expect("the search runs");
        let from_venice = venice(&venice_engine)
            .search(SearchRequest::new("pebble", 3))
            .await
            .expect("the search runs");

        assert!(from_brave.is_empty());
        assert!(from_venice.is_empty());
    }

    #[tokio::test]
    async fn a_status_says_whether_the_next_search_could_work() {
        let cases = [
            (StatusCode::UNAUTHORIZED, SearchErrorKind::Unavailable),
            (StatusCode::FORBIDDEN, SearchErrorKind::Unavailable),
            (StatusCode::PAYMENT_REQUIRED, SearchErrorKind::Unavailable),
            (StatusCode::TOO_MANY_REQUESTS, SearchErrorKind::Unavailable),
            (StatusCode::BAD_REQUEST, SearchErrorKind::InvalidRequest),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                SearchErrorKind::InvalidRequest,
            ),
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                SearchErrorKind::Execution,
            ),
            (StatusCode::SERVICE_UNAVAILABLE, SearchErrorKind::Execution),
        ];
        for (status, expected) in cases {
            let engine = Engine::start(Answer::status(status)).await;

            let from_brave = brave(&engine)
                .search(SearchRequest::new("pebble", 1))
                .await
                .expect_err("the engine refused");
            let from_venice = venice(&engine)
                .search(SearchRequest::new("pebble", 1))
                .await
                .expect_err("the engine refused");

            assert_eq!(from_brave.kind(), expected, "{status}");
            assert_eq!(
                from_brave.message(),
                format!("Brave Search API returned status {status}")
            );
            assert_eq!(from_venice.kind(), expected, "{status}");
            assert_eq!(
                from_venice.message(),
                format!("Venice Search API returned status {status}")
            );
        }
    }

    #[tokio::test]
    async fn venice_reports_the_balance_when_payment_is_required() {
        let cases = [
            (
                Answer::status(StatusCode::PAYMENT_REQUIRED)
                    .header("x-venice-balance-usd", "0.00")
                    .header("x-venice-balance-diem", "12.5"),
                "Venice Search API returned status 402 Payment Required (balance USD 0.00)",
            ),
            (
                Answer::status(StatusCode::PAYMENT_REQUIRED)
                    .header("x-venice-balance-diem", "12.5"),
                "Venice Search API returned status 402 Payment Required (balance DIEM 12.5)",
            ),
            (
                Answer::status(StatusCode::PAYMENT_REQUIRED),
                "Venice Search API returned status 402 Payment Required",
            ),
        ];
        for (answer, expected) in cases {
            let engine = Engine::start(answer).await;

            let error = venice(&engine)
                .search(SearchRequest::new("pebble", 1))
                .await
                .expect_err("payment is required");

            assert_eq!(error.kind(), SearchErrorKind::Unavailable);
            assert_eq!(error.message(), expected);
        }

        // The balance headers are Venice's; Brave's 402 is reported as it is.
        let engine = Engine::start(
            Answer::status(StatusCode::PAYMENT_REQUIRED).header("x-venice-balance-usd", "0.00"),
        )
        .await;
        let error = brave(&engine)
            .search(SearchRequest::new("pebble", 1))
            .await
            .expect_err("payment is required");
        assert_eq!(
            error.message(),
            "Brave Search API returned status 402 Payment Required"
        );
    }

    #[tokio::test]
    async fn an_answer_that_is_not_json_is_a_failed_search_with_the_cause_attached() {
        let engine = Engine::start(Answer {
            status:  StatusCode::OK,
            headers: Vec::new(),
            body:    "<html>upstream timeout</html>".to_owned(),
        })
        .await;

        let error = brave(&engine)
            .search(SearchRequest::new("pebble", 1))
            .await
            .expect_err("the body is not JSON");

        assert_eq!(error.kind(), SearchErrorKind::Execution);
        assert_eq!(error.message(), "Failed to parse Brave Search API response");
        assert!(error.source().is_some(), "the decode error is the cause");
    }

    #[tokio::test]
    async fn an_engine_that_cannot_be_reached_is_unavailable() {
        let url = closed_port().await;

        let error = Venice::new("venice-key", client())
            .endpoint(url)
            .search(SearchRequest::new("pebble", 1))
            .await
            .expect_err("nothing is listening");

        assert_eq!(error.kind(), SearchErrorKind::Unavailable);
        assert_eq!(error.message(), "Venice Search API could not be reached");
        assert!(
            error.source().is_some(),
            "the connection error is the cause"
        );
    }

    #[tokio::test]
    async fn a_search_that_runs_out_of_time_is_a_failed_search() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an engine that never answers");
        let url = format!(
            "http://{}/search",
            listener.local_addr().expect("the silent engine's address")
        );
        // Accept and hold the connection open without answering.
        tokio::spawn(async move {
            let mut held = Vec::new();
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                held.push(socket);
            }
        });
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(50))
            .build()
            .expect("a client with a short timeout");

        let error = Brave::new("brave-key", client)
            .endpoint(url)
            .search(SearchRequest::new("pebble", 1))
            .await
            .expect_err("the engine never answers");

        assert_eq!(error.kind(), SearchErrorKind::Execution);
        assert_eq!(error.message(), "Brave Search API timed out");
    }

    #[test]
    fn the_api_key_stays_out_of_debug_output() {
        let client = default_client().expect("the default client builds");
        let brave = Brave::new("brave-secret", client.clone());
        let venice = Venice::new("venice-secret", client).endpoint("https://relay.example/venice");

        let rendered = format!("{brave:?} {venice:?}");

        assert!(!rendered.contains("secret"), "{rendered}");
        assert!(rendered.contains(BRAVE_SEARCH_URL), "{rendered}");
        assert!(
            rendered.contains("https://relay.example/venice"),
            "{rendered}"
        );
    }
}
