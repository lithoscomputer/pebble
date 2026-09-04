//! Session identities, distinct from model-native tool-call identifiers.

use std::fmt;

use serde::{Deserialize, Serialize};

/// An opaque session identifier supplied by Pebble or a stored record.
///
/// No UUID format is required: applications may restore their own identifiers.
/// The serialized representation is the original string.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Wraps an opaque session identifier without changing its spelling.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The identifier's stored spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the underlying string for a storage or event boundary.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A session's identity and the root of the tree it belongs to.
///
/// Start a tree with [`root`](Self::root). Derive a child's identity with
/// [`child`](Self::child); descendants retain the same root automatically.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionIdentity {
    session_id:      SessionId,
    root_session_id: SessionId,
}

impl SessionIdentity {
    /// Starts a tree whose root is this session.
    #[must_use]
    pub fn root(session_id: SessionId) -> Self {
        Self {
            root_session_id: session_id.clone(),
            session_id,
        }
    }

    /// Places another session in this tree.
    #[must_use]
    pub fn child(&self, session_id: SessionId) -> Self {
        Self {
            session_id,
            root_session_id: self.root_session_id.clone(),
        }
    }

    /// The session whose tools and events carry this identity.
    #[must_use]
    pub const fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// The root shared by every session in the tree.
    #[must_use]
    pub const fn root_session_id(&self) -> &SessionId {
        &self.root_session_id
    }

    /// Whether this session is the root of its tree.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.session_id == self.root_session_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descendants_keep_the_original_root_and_ids_round_trip_as_strings() {
        let root_id: SessionId = serde_json::from_str("\"legacy/session\"").unwrap();
        let root = SessionIdentity::root(root_id);
        let child = root.child(SessionId::new("child"));
        let grandchild = child.child(SessionId::new("grandchild"));
        assert!(root.is_root());
        assert!(!child.is_root());
        assert!(!grandchild.is_root());
        assert_eq!(grandchild.root_session_id(), root.session_id());
        assert_eq!(grandchild.session_id().as_str(), "grandchild");
        assert_eq!(
            serde_json::to_string(grandchild.root_session_id()).unwrap(),
            "\"legacy/session\""
        );
    }
}
