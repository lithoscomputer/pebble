//! Coding-specific values carried through generic agent control.

use std::fmt;

use pebble_agent::{AgentControlHandle, CompletionLease, UserMessage};
use serde_json::Value;

use crate::types::Actor;

/// A hold that keeps natural completion open for steering.
#[must_use = "the lease parks completion only while it is held"]
pub struct SteeringLease {
    _lease: CompletionLease,
}

impl SteeringLease {
    pub(crate) fn acquire(control: &AgentControlHandle) -> Self {
        Self {
            _lease: control.hold_completion(),
        }
    }
}

impl fmt::Debug for SteeringLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SteeringLease")
            .finish_non_exhaustive()
    }
}

/// A steering message for the generic queue, with its coding-layer author as
/// opaque attribution.
pub(crate) fn steering_message(text: impl Into<String>, actor: Option<Actor>) -> UserMessage {
    let message = UserMessage::text(text);
    match actor.and_then(|actor| serde_json::to_value(actor).ok()) {
        Some(attribution) => message.with_attribution(attribution),
        None => message,
    }
}

/// The coding-layer author carried by generic message attribution.
pub(crate) fn actor_from_attribution(attribution: Option<&Value>) -> Option<Actor> {
    attribution.and_then(|value| serde_json::from_value(value.clone()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_author_survives_the_round_trip_through_an_attribution() {
        let actor = Actor::User {
            id:           Some("u_1".into()),
            display_name: Some("Ada".into()),
        };

        let message = steering_message("hello", Some(actor.clone()));

        assert_eq!(message.text_content(), "hello");
        assert_eq!(actor_from_attribution(message.attribution()), Some(actor));
        assert_eq!(actor_from_attribution(None), None);
    }
}
