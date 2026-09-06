//! Session identifiers and the scope shared by tools in a session tree.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An opaque session identifier supplied by Pebble or a stored record.
///
/// No UUID format is required: applications may restore their own identifiers.
/// The serialized representation is the original string.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Creates a fresh session identifier.
    #[must_use]
    pub fn fresh() -> Self {
        Self(format!("ses_{}", Uuid::new_v4()))
    }

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

/// The acting session, its root, its immediate parent, and its depth.
///
/// Session-scoped tools use the session ID. Root-scoped tools, such as the
/// shared task list, use the root session ID.
///
/// Start a tree with [`root`](Self::root). Derive a child's scope with
/// [`child`](Self::child); descendants retain the same root automatically.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StoredScope")]
pub struct SessionScope {
    session_id:        SessionId,
    root_session_id:   SessionId,
    parent_session_id: Option<SessionId>,
    depth:             usize,
}

/// Validates stored ancestry before exposing it to permission policies.
#[derive(Deserialize)]
struct StoredScope {
    session_id:        SessionId,
    root_session_id:   SessionId,
    parent_session_id: Option<SessionId>,
    depth:             usize,
}

impl TryFrom<StoredScope> for SessionScope {
    type Error = &'static str;

    fn try_from(stored: StoredScope) -> Result<Self, Self::Error> {
        let valid = match &stored.parent_session_id {
            None => stored.depth == 0 && stored.session_id == stored.root_session_id,
            Some(parent) => {
                stored.depth > 0
                    && stored.session_id != stored.root_session_id
                    && stored.session_id != *parent
                    && ((stored.depth == 1) == (*parent == stored.root_session_id))
            }
        };
        if !valid {
            return Err("session scope has inconsistent root, parent, or depth");
        }
        Ok(Self {
            session_id:        stored.session_id,
            root_session_id:   stored.root_session_id,
            parent_session_id: stored.parent_session_id,
            depth:             stored.depth,
        })
    }
}

impl SessionScope {
    /// Starts a tree whose root is this session.
    #[must_use]
    pub fn root(session_id: SessionId) -> Self {
        Self {
            root_session_id: session_id.clone(),
            parent_session_id: None,
            depth: 0,
            session_id,
        }
    }

    /// Places another session directly below this one.
    ///
    /// Use an identifier unique within this tree.
    #[must_use]
    pub fn child(&self, session_id: SessionId) -> Self {
        Self {
            session_id,
            root_session_id: self.root_session_id.clone(),
            parent_session_id: Some(self.session_id.clone()),
            depth: self.depth.saturating_add(1),
        }
    }

    /// The session acting in this scope.
    #[must_use]
    pub const fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// The root shared by every session in the tree.
    #[must_use]
    pub const fn root_session_id(&self) -> &SessionId {
        &self.root_session_id
    }

    /// The immediate parent, absent for a root.
    #[must_use]
    pub const fn parent_session_id(&self) -> Option<&SessionId> {
        self.parent_session_id.as_ref()
    }

    /// The number of parent links from this session to the root.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Whether this session is the root of its tree.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent_session_id.is_none()
    }
}

impl Default for SessionScope {
    /// Starts an independent tree with a fresh identifier.
    fn default() -> Self {
        Self::root(SessionId::fresh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descendants_keep_the_original_root_and_ids_round_trip_as_strings() {
        let root_id: SessionId = serde_json::from_str("\"legacy/session\"").unwrap();
        let root = SessionScope::root(root_id);
        let child = root.child(SessionId::new("child"));
        let grandchild = child.child(SessionId::new("grandchild"));
        assert!(root.is_root());
        assert!(!child.is_root());
        assert!(!grandchild.is_root());
        assert_eq!(root.parent_session_id(), None);
        assert_eq!(root.depth(), 0);
        assert_eq!(child.parent_session_id(), Some(root.session_id()));
        assert_eq!(child.depth(), 1);
        assert_eq!(grandchild.parent_session_id(), Some(child.session_id()));
        assert_eq!(grandchild.depth(), 2);
        assert_eq!(grandchild.root_session_id(), root.session_id());
        let restored: SessionScope =
            serde_json::from_str(&serde_json::to_string(&grandchild).unwrap()).unwrap();
        assert_eq!(restored, grandchild);
        assert_eq!(grandchild.session_id().as_str(), "grandchild");
        assert_eq!(
            serde_json::to_string(grandchild.root_session_id()).unwrap(),
            "\"legacy/session\""
        );
    }
    #[test]
    fn inconsistent_stored_ancestry_is_refused() {
        let grandchild = SessionScope::root(SessionId::new("root"))
            .child(SessionId::new("child"))
            .child(SessionId::new("grandchild"));
        let stored = serde_json::to_value(&grandchild).unwrap();
        for (field, value) in [
            ("parent_session_id", serde_json::Value::Null),
            ("depth", serde_json::json!(0)),
            ("depth", serde_json::json!(1)),
            ("root_session_id", serde_json::json!("grandchild")),
        ] {
            let mut invalid = stored.clone();
            invalid[field] = value;
            assert!(serde_json::from_value::<SessionScope>(invalid).is_err());
        }
        let mut omitted = stored;
        omitted.as_object_mut().unwrap().remove("parent_session_id");
        assert!(serde_json::from_value::<SessionScope>(omitted).is_err());
    }
}
