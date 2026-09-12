//! Searching the web.
//!
//! Pebble owns everything the model sees of a search — the `web_search` tool,
//! its schema, and the way results are written out — and leaves the engine to
//! the application through one seam: a `SearchProvider`, implemented over
//! whatever the application has and installed with
//! [`CodingAgentBuilder::search_provider`](crate::CodingAgentBuilder::search_provider).
//! The seam's types are in [`extensions`](crate::extensions), with the other
//! services an application supplies. A session with no provider registers no
//! `web_search` tool.
//!
//! Two providers ship with pebble, for engines whose seam is a public HTTP
//! API: `providers::Brave` and `providers::Venice`, behind the
//! `search-providers` feature. Each is built from an API key and a
//! `reqwest::Client` the application owns. Which one a session gets, and
//! whether it gets one, stays the application's decision.

pub(crate) mod seam;

#[cfg(feature = "search-providers")]
pub mod providers;
