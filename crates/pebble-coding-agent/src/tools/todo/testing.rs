//! What the todo tools' own tests build a call out of.

use std::sync::{Arc, Mutex, PoisonError};

use crate::test_support::MockEnvironment;
use crate::tool::{CodingEventEmitter, ToolContext};
use crate::tools::testing::context;
use crate::types::CodingEvent;

/// An emitter that keeps what it was given, so a test can assert on the
/// changes a runtime announced.
#[derive(Default)]
pub(super) struct CollectingEmitter {
    events: Mutex<Vec<CodingEvent>>,
}

impl CollectingEmitter {
    /// Everything published so far.
    pub(super) fn events(&self) -> Vec<CodingEvent> {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl CodingEventEmitter for CollectingEmitter {
    fn emit(&self, event: CodingEvent) {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }
}

/// A call from `session`, whose tree is rooted at `root`.
///
/// The two identities are what decide which list a tool writes, so a test names
/// both.
pub(super) fn context_for(session: &str, root: &str) -> ToolContext {
    context(MockEnvironment::default())
        .with_session(session, root)
        .with_coding_event_emitter(Arc::new(CollectingEmitter::default()))
}

/// A call that publishes what it changes to `emitter`.
pub(super) fn context_emitting(emitter: Arc<CollectingEmitter>) -> ToolContext {
    context(MockEnvironment::default())
        .with_session("ses_a", "ses_a")
        .with_coding_event_emitter(emitter)
}
