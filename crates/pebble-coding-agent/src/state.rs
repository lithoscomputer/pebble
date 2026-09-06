//! Durable session state: the conversation and its stored form.
//!
//! [`History`] is the committed conversation of a live agent, as [`Message`]
//! turns. [`SessionRecord`] is its durable form, carrying [`StoredMessage`]
//! turns, a format version, and the exact route the session ran on; resume it
//! with [`CodingAgent::resume`](crate::CodingAgent::resume). When the event log
//! and record do not share a transaction, reconcile the record's event cursor
//! before resume.

pub use crate::history::History;
pub use crate::record::{SESSION_RECORD_FORMAT_VERSION, SessionRecord, StoredMessage};
pub use crate::types::{InputContent, Message};
