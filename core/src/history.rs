//! Persistent per-turn conversation history in SQLite (SPEC.md §5, layer 2,
//! EPIC 10.1).
//!
//! History, the EPIC 10.4 summarizer, and EPIC 10.5 vector search all share
//! **one** database file, and SQLite allows only one writer at a time — two
//! independent connections both issuing writes will block each other or, in
//! the worst case with the wrong pragmas, corrupt the file. [`HistoryStore`]
//! owns exactly one write connection on a dedicated thread (the "write
//! actor") and funnels every write through it via a channel; readers instead
//! open their own short-lived connections, which WAL mode allows to proceed
//! concurrently with the writer without blocking on it.

use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::thread;

use rusqlite::{params, Connection, OptionalExtension};

use crate::llm::Trust;

/// Errors from opening or using the history store.
#[derive(Debug, thiserror::Error)]
pub enum HistoryError {
    /// The database file or its parent directory could not be prepared.
    #[error("failed to open history database at {path}: {source}")]
    Open {
        /// Path the database was opened at.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: rusqlite::Error,
    },
    /// The database's parent directory could not be created.
    #[error("failed to create parent directory for history database at {path}: {source}")]
    CreateDir {
        /// Path the database was opened at.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A query against the database failed.
    #[error("history database query failed: {0}")]
    Query(#[from] rusqlite::Error),
    /// The write actor thread hung up — it panicked or the store was
    /// dropped mid-request.
    #[error("history write actor is no longer running")]
    WriteActorGone,
}

/// Where a stored [`TurnRecord`] came from (SPEC.md §5.1).
///
/// A thin string mirror of [`Trust`] for the `provenance` column: SQLite has
/// no enum type, and round-tripping through [`Trust`] directly (rather than
/// a bespoke history-only enum) is what keeps the taint tag identical from
/// the moment a turn is logged through to EPIC 10.5 prompt injection — no
/// second enum for the two to silently drift apart.
fn provenance_to_str(trust: Trust) -> &'static str {
    match trust {
        Trust::User => "user",
        Trust::Assistant => "assistant",
        Trust::ToolUntrusted => "tool_untrusted",
    }
}

fn provenance_from_str(s: &str) -> Result<Trust, rusqlite::Error> {
    match s {
        "user" => Ok(Trust::User),
        "assistant" => Ok(Trust::Assistant),
        "tool_untrusted" => Ok(Trust::ToolUntrusted),
        other => Err(rusqlite::Error::InvalidColumnType(
            0,
            format!("unknown provenance tag {other:?}"),
            rusqlite::types::Type::Text,
        )),
    }
}

/// A turn to be persisted, handed to the write actor by the orchestrator
/// (EPIC 8.1) once a turn completes.
#[derive(Debug, Clone)]
pub struct NewTurn {
    /// Groups turns from the same conversation.
    pub session_id: String,
    /// Unix epoch milliseconds when the turn was logged.
    pub timestamp_ms: i64,
    /// Chat role, stored as-given (e.g. `"user"`, `"assistant"`).
    pub role: String,
    /// Turn text.
    pub text: String,
    /// Provenance taint — must survive unchanged into any downstream
    /// summary or retrieval (§5.1).
    pub provenance: Trust,
    /// Set when the turn was cut short (e.g. barge-in, §2.5.1) rather than
    /// completing normally.
    pub interrupted: bool,
}

/// A turn as read back from the database.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRecord {
    /// Row id assigned by SQLite.
    pub id: i64,
    /// Groups turns from the same conversation.
    pub session_id: String,
    /// Unix epoch milliseconds when the turn was logged.
    pub timestamp_ms: i64,
    /// Chat role, as stored.
    pub role: String,
    /// Turn text.
    pub text: String,
    /// Provenance taint (§5.1).
    pub provenance: Trust,
    /// Whether the turn was cut short.
    pub interrupted: bool,
}

fn row_to_turn(row: &rusqlite::Row) -> rusqlite::Result<TurnRecord> {
    let provenance_str: String = row.get(5)?;
    Ok(TurnRecord {
        id: row.get(0)?,
        session_id: row.get(1)?,
        timestamp_ms: row.get(2)?,
        role: row.get(3)?,
        text: row.get(4)?,
        provenance: provenance_from_str(&provenance_str)?,
        interrupted: row.get::<_, i64>(6)? != 0,
    })
}

const CREATE_TURNS_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS turns (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT    NOT NULL,
        timestamp_ms  INTEGER NOT NULL,
        role          TEXT    NOT NULL,
        text          TEXT    NOT NULL,
        provenance    TEXT    NOT NULL,
        interrupted   INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS turns_session_id_idx ON turns (session_id, id);
";

/// Opens `path` (creating it and its parent directory if needed), enables
/// WAL mode, and creates the `turns` table if absent.
///
/// WAL mode is a per-database-file setting that persists once written, but
/// every connection this module opens (write actor and each reader) sets it
/// again on connect anyway — cheap, and it means store setup never depends
/// on which connection happens to run first.
fn open_and_prepare(path: &Path) -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "busy_timeout", 5_000)?;
    conn.execute_batch(CREATE_TURNS_TABLE)?;
    Ok(conn)
}

/// A command sent to the write actor thread.
enum WriteCommand {
    LogTurn {
        turn: NewTurn,
        reply: std_mpsc::SyncSender<Result<i64, HistoryError>>,
    },
}

/// Handle to the persistent history store (EPIC 10.1).
///
/// Cloning is cheap ([`std_mpsc::Sender`] is `Clone`) and safe: every clone
/// shares the same write actor, so single-writer discipline holds no matter
/// how many callers hold a handle.
#[derive(Clone)]
pub struct HistoryStore {
    db_path: PathBuf,
    write_tx: std_mpsc::Sender<WriteCommand>,
}

impl HistoryStore {
    /// Opens (or creates) the history database at `path` and starts its
    /// write actor thread.
    ///
    /// `path`'s parent directory is created if missing — `[memory].db_path`
    /// defaults under `~/.marceline/`, which may not exist on first run.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, HistoryError> {
        let db_path = path.into();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| HistoryError::CreateDir {
                path: db_path.clone(),
                source,
            })?;
        }

        // Prepare the schema up front so `open` itself fails fast on a bad
        // path, rather than surfacing the error asynchronously on the first
        // write.
        open_and_prepare(&db_path).map_err(|source| HistoryError::Open {
            path: db_path.clone(),
            source,
        })?;

        let (write_tx, write_rx) = std_mpsc::channel::<WriteCommand>();
        let actor_path = db_path.clone();
        thread::spawn(move || {
            // The actor's own connection is separate from the one used to
            // prepare the schema above — that connection is dropped once
            // `open` returns.
            let conn = match open_and_prepare(&actor_path) {
                Ok(conn) => conn,
                Err(_) => return, // Already validated above; unreachable in practice.
            };

            for command in write_rx {
                match command {
                    WriteCommand::LogTurn { turn, reply } => {
                        let result = conn
                            .execute(
                                "INSERT INTO turns
                                    (session_id, timestamp_ms, role, text, provenance, interrupted)
                                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                                params![
                                    turn.session_id,
                                    turn.timestamp_ms,
                                    turn.role,
                                    turn.text,
                                    provenance_to_str(turn.provenance),
                                    turn.interrupted as i64,
                                ],
                            )
                            .map(|_| conn.last_insert_rowid())
                            .map_err(HistoryError::from);
                        let _ = reply.send(result);
                    }
                }
            }
        });

        Ok(Self { db_path, write_tx })
    }

    /// Persists a completed turn via the write actor, returning its row id.
    ///
    /// Blocks the calling thread until the actor replies. Call from within
    /// [`tokio::task::spawn_blocking`] when invoking from async code, so a
    /// slow disk doesn't stall a runtime worker thread.
    pub fn log_turn(&self, turn: NewTurn) -> Result<i64, HistoryError> {
        let (reply_tx, reply_rx) = std_mpsc::sync_channel(1);
        self.write_tx
            .send(WriteCommand::LogTurn {
                turn,
                reply: reply_tx,
            })
            .map_err(|_| HistoryError::WriteActorGone)?;
        reply_rx.recv().map_err(|_| HistoryError::WriteActorGone)?
    }

    /// Reads the most recent `limit` turns for `session_id`, oldest first.
    ///
    /// Opens its own connection rather than going through the write actor —
    /// WAL mode lets a reader proceed without waiting on in-flight writes,
    /// and readers never need single-writer discipline.
    pub fn recent_turns(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<TurnRecord>, HistoryError> {
        let conn = open_and_prepare(&self.db_path).map_err(|source| HistoryError::Open {
            path: self.db_path.clone(),
            source,
        })?;

        let mut stmt = conn.prepare(
            "SELECT id, session_id, timestamp_ms, role, text, provenance, interrupted
             FROM turns
             WHERE session_id = ?1
             ORDER BY id DESC
             LIMIT ?2",
        )?;
        let mut rows = stmt
            .query_map(params![session_id, limit as i64], row_to_turn)?
            .collect::<Result<Vec<_>, _>>()?;
        rows.reverse();
        Ok(rows)
    }

    /// Reads a single turn by row id, if it exists.
    pub fn get_turn(&self, id: i64) -> Result<Option<TurnRecord>, HistoryError> {
        let conn = open_and_prepare(&self.db_path).map_err(|source| HistoryError::Open {
            path: self.db_path.clone(),
            source,
        })?;

        conn.query_row(
            "SELECT id, session_id, timestamp_ms, role, text, provenance, interrupted
             FROM turns WHERE id = ?1",
            params![id],
            row_to_turn,
        )
        .optional()
        .map_err(HistoryError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(session_id: &str, role: &str, text: &str, provenance: Trust) -> NewTurn {
        NewTurn {
            session_id: session_id.to_string(),
            timestamp_ms: 1_000,
            role: role.to_string(),
            text: text.to_string(),
            provenance,
            interrupted: false,
        }
    }

    #[test]
    fn logs_and_reads_back_a_turn() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        let id = store
            .log_turn(turn("s1", "user", "hello", Trust::User))
            .unwrap();

        let record = store.get_turn(id).unwrap().unwrap();
        assert_eq!(record.session_id, "s1");
        assert_eq!(record.text, "hello");
        assert_eq!(record.provenance, Trust::User);
        assert!(!record.interrupted);
    }

    #[test]
    fn recent_turns_are_ordered_oldest_first_and_scoped_to_session() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        store.log_turn(turn("s1", "user", "first", Trust::User)).unwrap();
        store
            .log_turn(turn("s1", "assistant", "second", Trust::Assistant))
            .unwrap();
        store
            .log_turn(turn("s2", "user", "other session", Trust::User))
            .unwrap();

        let turns = store.recent_turns("s1", 10).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].text, "first");
        assert_eq!(turns[1].text, "second");
    }

    #[test]
    fn recent_turns_respects_the_limit_keeping_the_most_recent() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        for i in 0..5 {
            store
                .log_turn(turn("s1", "user", &format!("turn {i}"), Trust::User))
                .unwrap();
        }

        let turns = store.recent_turns("s1", 2).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].text, "turn 3");
        assert_eq!(turns[1].text, "turn 4");
    }

    #[test]
    fn provenance_survives_the_round_trip_for_every_variant() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        for provenance in [Trust::User, Trust::Assistant, Trust::ToolUntrusted] {
            let id = store
                .log_turn(turn("s1", "role", "text", provenance))
                .unwrap();
            assert_eq!(store.get_turn(id).unwrap().unwrap().provenance, provenance);
        }
    }

    #[test]
    fn an_interrupted_turn_keeps_its_flag() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        let mut t = turn("s1", "user", "cut off", Trust::User);
        t.interrupted = true;
        let id = store.log_turn(t).unwrap();

        assert!(store.get_turn(id).unwrap().unwrap().interrupted);
    }

    #[test]
    fn turns_persist_across_a_store_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("history.db");

        {
            let store = HistoryStore::open(&db_path).unwrap();
            store.log_turn(turn("s1", "user", "before restart", Trust::User)).unwrap();
        }

        let store = HistoryStore::open(&db_path).unwrap();
        let turns = store.recent_turns("s1", 10).unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].text, "before restart");
    }

    #[test]
    fn concurrent_writers_and_a_reader_do_not_error_or_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let store = store.clone();
                std::thread::spawn(move || {
                    store
                        .log_turn(turn("s1", "user", &format!("concurrent {i}"), Trust::User))
                        .unwrap()
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let turns = store.recent_turns("s1", 100).unwrap();
        assert_eq!(turns.len(), 8);
    }
}
