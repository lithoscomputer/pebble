//! The HTTP headers that name the session a request was made for.
//!
//! Every model request a session makes, and every request to an HTTP MCP
//! server the agent connects to, carries the session's identity, so a proxy
//! that records the exchange can match it to the session's events. The
//! values are the ones the event envelope carries as `session_id` and
//! `parent_session_id`, spelled exactly the same.

use crate::SessionScope;

/// The header that carries the acting session's ID.
pub const SESSION_ID_HEADER: &str = "x-pebble-session-id";

/// The header that carries the parent session's ID, sent by subagents only.
pub const PARENT_SESSION_ID_HEADER: &str = "x-pebble-parent-session-id";

/// The identity headers for a session in `scope`.
///
/// An ID that cannot be sent as a header value unchanged is left out rather
/// than rewritten: a rewritten value would no longer match the events it is
/// meant to find. Pebble's own IDs are always sendable; only an ID an
/// application restored in its own spelling can be left out.
pub(crate) fn session_headers(scope: &SessionScope) -> Vec<(&'static str, String)> {
    let parent = scope.parent_session_id();
    [
        Some((SESSION_ID_HEADER, scope.session_id())),
        parent.map(|parent| (PARENT_SESSION_ID_HEADER, parent)),
    ]
    .into_iter()
    .flatten()
    .filter(|(_, id)| is_header_value(id.as_str()))
    .map(|(name, id)| (name, id.to_string()))
    .collect()
}

/// Whether `value` is a valid HTTP header value as written: visible ASCII,
/// spaces, and tabs, with no surrounding whitespace.
fn is_header_value(value: &str) -> bool {
    !value.is_empty()
        && value.trim() == value
        && value
            .bytes()
            .all(|byte| byte == b'\t' || (b' '..=b'~').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SessionId;

    #[test]
    fn a_root_names_itself_and_a_child_also_names_its_parent() {
        let root = SessionScope::root(SessionId::new("ses_root"));
        let child = root.child(SessionId::new("ses_child"));

        assert_eq!(session_headers(&root), vec![(
            SESSION_ID_HEADER,
            "ses_root".to_owned()
        )]);
        assert_eq!(session_headers(&child), vec![
            (SESSION_ID_HEADER, "ses_child".to_owned()),
            (PARENT_SESSION_ID_HEADER, "ses_root".to_owned()),
        ]);
    }

    #[test]
    fn an_id_that_is_not_a_header_value_is_left_out() {
        let root = SessionScope::root(SessionId::new("sés\nroot"));
        let child = root.child(SessionId::new("ses_child"));

        assert!(session_headers(&root).is_empty());
        assert_eq!(session_headers(&child), vec![(
            SESSION_ID_HEADER,
            "ses_child".to_owned()
        )]);
    }
}
