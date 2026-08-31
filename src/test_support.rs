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

pub use crate::environment::mock::{MockEnvironment, MutableMockEnvironment};
