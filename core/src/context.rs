//! Working-context assembly from persistent history (SPEC.md §5, layer 1,
//! EPIC 10.2).
//!
//! [`crate::llm::TurnBuffer`] (EPIC 4.3) trims an in-RAM `Vec<Message>` to
//! fit the context window, but says nothing about where those messages come
//! from on a cold start. [`recent_context`] is that source: it reads the
//! most recent turns for a session out of [`HistoryStore`] (EPIC 10.1) and
//! turns them back into [`Message`]s a fresh [`TurnBuffer`] can hold, so a
//! restart mid-session resumes with the same working context it had before
//! the daemon went down.

use crate::history::{HistoryError, HistoryStore, TurnRecord};
use crate::llm::{Message, Role};

/// Turns a stored role string back into a [`Role`].
///
/// History only ever persists turns the orchestrator (8.1) hands the write
/// actor — user, assistant, and tool messages — never [`Role::System`],
/// which is recompiled fresh every request (EPIC 4.2) and never belongs in
/// stored history.
fn role_from_str(role: &str) -> Option<Role> {
    match role {
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

/// Builds the working context for `session_id` from its most recent
/// `limit` persisted turns, oldest first, ready to seed a [`TurnBuffer`].
///
/// Only the single most recent turn, if interrupted, is kept as
/// "you were saying…" context (§2.5.1) — older interrupted partials are
/// dropped rather than replayed as if they were complete turns, since a cut
/// off assistant sentence read back verbatim later in the conversation
/// would read as a non sequitur.
///
/// A row whose role does not map to a conversational [`Role`] is skipped
/// rather than failing the whole assembly — defensive against rows written
/// by some future producer this function doesn't yet know about.
pub fn recent_context(
    store: &HistoryStore,
    session_id: &str,
    limit: usize,
) -> Result<Vec<Message>, HistoryError> {
    let turns = store.recent_turns(session_id, limit)?;
    let last_index = turns.len().saturating_sub(1);

    Ok(turns
        .into_iter()
        .enumerate()
        .filter(|(i, turn): &(usize, TurnRecord)| !turn.interrupted || *i == last_index)
        .filter_map(|(_, turn)| {
            role_from_str(&turn.role).map(|role| Message::new(role, turn.text))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::NewTurn;
    use crate::llm::Trust;

    fn turn(session_id: &str, role: &str, text: &str, interrupted: bool) -> NewTurn {
        NewTurn {
            session_id: session_id.to_string(),
            timestamp_ms: 1_000,
            role: role.to_string(),
            text: text.to_string(),
            provenance: Trust::User,
            interrupted,
        }
    }

    #[test]
    fn assembles_messages_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();
        store.log_turn(turn("s1", "user", "hi", false)).unwrap();
        store.log_turn(turn("s1", "assistant", "hello", false)).unwrap();

        let messages = recent_context(&store, "s1", 10).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].content, "hi");
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[1].content, "hello");
    }

    #[test]
    fn keeps_only_the_most_recent_interrupted_turn() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();
        store.log_turn(turn("s1", "assistant", "cut off earlier", true)).unwrap();
        store.log_turn(turn("s1", "user", "go on", false)).unwrap();
        store.log_turn(turn("s1", "assistant", "cut off now", true)).unwrap();

        let messages = recent_context(&store, "s1", 10).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "go on");
        assert_eq!(messages[1].content, "cut off now");
    }

    #[test]
    fn a_complete_final_turn_after_an_interrupted_one_keeps_both_dropping_only_the_stale_partial() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();
        store.log_turn(turn("s1", "assistant", "cut off", true)).unwrap();
        store.log_turn(turn("s1", "user", "finished thought", false)).unwrap();

        let messages = recent_context(&store, "s1", 10).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "finished thought");
    }

    #[test]
    fn scopes_to_the_requested_session_only() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();
        store.log_turn(turn("s1", "user", "in scope", false)).unwrap();
        store.log_turn(turn("s2", "user", "other session", false)).unwrap();

        let messages = recent_context(&store, "s1", 10).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "in scope");
    }
}
