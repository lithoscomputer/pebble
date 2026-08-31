//! Who authored an out-of-band message to a running session.

use serde::{Deserialize, Serialize};

/// The author of a steering message or other out-of-band input.
///
/// Pebble does not model identity itself. Every field is optional so an
/// embedder that has no identity system can still say *which kind* of party
/// spoke, and one that does can attach its own identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Actor {
    /// A human operator.
    User {
        /// The embedder's identifier for the person.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id:           Option<String>,
        /// A name suitable for display.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
    },
    /// Another agent, such as a parent session steering a child.
    Agent {
        /// The session or agent identifier.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// The embedding system itself — a scheduler, watchdog, or timeout.
    System,
    /// A party outside the system, reached through an integration.
    External {
        /// How to name the integration or party.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn user_serializes_with_a_kind_tag() {
        let actor = Actor::User {
            id:           Some("u_1".into()),
            display_name: Some("Ada".into()),
        };
        assert_eq!(
            serde_json::to_value(&actor).expect("serializes"),
            json!({"kind": "user", "id": "u_1", "display_name": "Ada"})
        );
    }

    #[test]
    fn absent_identity_is_omitted_and_still_parses() {
        let actor = Actor::User {
            id:           None,
            display_name: None,
        };
        let value = serde_json::to_value(&actor).expect("serializes");
        assert_eq!(value, json!({"kind": "user"}));
        assert_eq!(
            serde_json::from_value::<Actor>(value).expect("parses"),
            actor
        );
    }

    #[test]
    fn system_is_a_bare_tag() {
        assert_eq!(
            serde_json::to_value(Actor::System).expect("serializes"),
            json!({"kind": "system"})
        );
    }

    #[test]
    fn every_variant_round_trips() {
        let actors = vec![
            Actor::User {
                id:           Some("u_1".into()),
                display_name: None,
            },
            Actor::Agent {
                id: Some("ses_parent".into()),
            },
            Actor::System,
            Actor::External {
                label: Some("slack".into()),
            },
        ];
        let json = serde_json::to_string(&actors).expect("serializes");
        let restored: Vec<Actor> = serde_json::from_str(&json).expect("parses");
        assert_eq!(restored, actors);
    }
}
