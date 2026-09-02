//! Test doubles for code that embeds pebble.
//!
//! These are the doubles pebble's own tests run against, published so an
//! application can drive its tools and hooks without a real machine
//! underneath. Enable the `test-util` feature to reach them:
//!
//! ```toml
//! [dev-dependencies]
//! pebble = { version = "0.1", features = ["test-util"] }
//! ```
//!
//! There are two of them. [`MockEnvironment`] stands in for the machine a
//! session works on, and [`ScriptedProvider`] stands in for the model it talks
//! to — registered on a real [`Client`](lithos_llm::Client), so everything
//! between the session and the provider is the code that runs in production.

mod scripted;

pub use self::scripted::{
    ScriptedCall, ScriptedCompletion, ScriptedFailure, ScriptedItem, ScriptedProvider,
    TEST_CATALOG, client_from, custom_tool_call_response, events_for, message_text,
    multi_tool_call_response, reasoning_delta_events, reasoning_response,
    responses_reasoning_response, scripted_client, scripted_client_builder, test_catalog,
    text_delta_events, text_response, tool_call_events, tool_call_response, with_cost,
    with_finish_reason, with_input_tokens, with_usage,
};
pub use crate::environment::mock::{MockEnvironment, MutableMockEnvironment};
