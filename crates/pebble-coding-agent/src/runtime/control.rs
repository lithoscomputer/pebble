//! Coding-specific values carried through generic agent control.

use std::fmt;

use pebble_agent::{AgentControlHandle, CompletionLease, UserMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{Actor, InputContent, InputSource};

/// Coding facts carried through the generic message attribution seam.
#[derive(Debug, Default, Serialize, Deserialize)]
struct CodingAttribution {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    actor:  Option<Actor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<InputSource>,
}

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
pub(crate) fn steering_message(content: InputContent, actor: Option<Actor>) -> UserMessage {
    attributed_message(content, CodingAttribution {
        actor,
        source: None,
    })
}

/// An ordinary coding input for the generic queue or prompt operation.
pub(crate) fn input_message(content: InputContent, source: InputSource) -> UserMessage {
    attributed_message(content, CodingAttribution {
        actor:  None,
        source: Some(source),
    })
}

fn attributed_message(content: InputContent, attribution: CodingAttribution) -> UserMessage {
    let message = UserMessage::new(content.into_parts());
    match serde_json::to_value(attribution).ok() {
        Some(attribution) => message.with_attribution(attribution),
        None => message,
    }
}

/// The coding-layer author carried by generic message attribution.
pub(crate) fn actor_from_attribution(attribution: Option<&Value>) -> Option<Actor> {
    attribution.and_then(|value| {
        serde_json::from_value::<CodingAttribution>(value.clone())
            .ok()
            .and_then(|attribution| attribution.actor)
            // Read the attribution shape emitted before the envelope existed.
            .or_else(|| serde_json::from_value(value.clone()).ok())
    })
}

/// The coding input source carried by generic message attribution.
pub(crate) fn input_source_from_attribution(attribution: Option<&Value>) -> Option<InputSource> {
    attribution.and_then(|value| {
        serde_json::from_value::<CodingAttribution>(value.clone())
            .ok()
            .and_then(|attribution| attribution.source)
    })
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

        let message = steering_message("hello".into(), Some(actor.clone()));

        assert_eq!(message.text_content(), "hello");
        assert_eq!(actor_from_attribution(message.attribution()), Some(actor));
        assert_eq!(actor_from_attribution(None), None);
    }

    #[test]
    fn an_input_source_survives_the_round_trip_through_an_attribution() {
        let message = input_message("hello".into(), InputSource::FollowUp);

        assert_eq!(
            input_source_from_attribution(message.attribution()),
            Some(InputSource::FollowUp)
        );
    }
}
