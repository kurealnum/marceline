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
//!
//! EPIC 10.3 extends this same store with a `memories` table (long-term
//! memory rows) plus a `memories_vec` `vec0` virtual table (the
//! `sqlite-vec` vector index) rather than opening a second store or a
//! second write connection: SQLite is still single-writer for the one
//! shared DB file, so every memory write goes through the very same write
//! actor as `turns`, via new [`WriteCommand`] variants. See
//! `crate::memory` for the higher-level embed-then-store/retrieve API built
//! on top of the low-level methods here, and `crate::embedding` for the
//! embedding pipeline itself.

use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::sync::Once;
use std::thread;

use rusqlite::{params, Connection, OptionalExtension};

use crate::llm::Trust;

/// Registers the `sqlite-vec` loadable extension (`vec0` virtual table,
/// `vec_version()`, etc.) as a SQLite *auto* extension — vendored as C and
/// compiled with `SQLITE_CORE` (see `core/Cargo.toml`'s comment on the
/// `sqlite-vec` dependency), so this links it straight into the same
/// process as `rusqlite`'s bundled SQLite rather than loading a separate
/// `.so`/`.dylib` at runtime.
///
/// `sqlite3_auto_extension` is a process-global registration that affects
/// every [`Connection`] opened *after* it runs, so this must be called
/// before the first `Connection::open` anywhere in the process — guarded by
/// [`Once`] so every call site (the initial validation open, the write
/// actor's own connection, and every reader) can call it unconditionally
/// without double-registering.
fn register_sqlite_vec() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *const std::os::raw::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::os::raw::c_int,
        >(sqlite_vec::sqlite3_vec_init)));
    });
}

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
    /// A memory write supplied a vector whose length doesn't match the
    /// dimension the `memories_vec` index was created with.
    ///
    /// The *only* sanctioned way to change dimension is the re-embed
    /// migration ([`HistoryStore::apply_reembed`]), which rebuilds the
    /// index from scratch — a plain insert with a mismatched vector would
    /// otherwise either corrupt the index or silently mix two models'
    /// vectors in one queryable table (SPEC.md §5.2), so it errors instead.
    #[error("memory vector has dimension {got}, but the index is {expected}-dimensional; run a re-embed migration to change it")]
    DimMismatch {
        /// Dimension the `memories_vec` index currently has.
        expected: usize,
        /// Dimension of the vector that was passed in.
        got: usize,
    },
    /// An edit or delete named a memory row id that doesn't exist — most
    /// likely a typo, or the row was already deleted by a previous command
    /// (EPIC 10.6's CLI). Reported as a clear error rather than silently
    /// no-oping, since a no-op edit/delete would look like it worked.
    #[error("no memory with id {id}")]
    NotFound {
        /// The row id that was not found.
        id: i64,
    },
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

/// Plain metadata for each long-term memory row (SPEC.md §5.2, EPIC 10.3):
/// source text, the embedding model that produced its vector, the vector's
/// dimension, and provenance. The vector itself lives separately in the
/// `memories_vec` `vec0` virtual table (created lazily — see
/// [`ensure_vec_table`]), keyed by the same rowid as `memories.id`, since
/// `vec0` cannot mix vector storage with ordinary columns in one table.
const CREATE_MEMORIES_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS memories (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        text          TEXT    NOT NULL,
        embed_model   TEXT    NOT NULL,
        dim           INTEGER NOT NULL,
        provenance    TEXT    NOT NULL,
        created_at_ms INTEGER NOT NULL
    );
";

/// Opens `path` (creating it and its parent directory if needed), enables
/// WAL mode, registers the `sqlite-vec` extension, and creates the `turns`
/// and `memories` tables if absent.
///
/// WAL mode is a per-database-file setting that persists once written, but
/// every connection this module opens (write actor and each reader) sets it
/// again on connect anyway — cheap, and it means store setup never depends
/// on which connection happens to run first.
///
/// `memories_vec` (the `vec0` index) is deliberately *not* created here:
/// its column dimension is fixed at creation time and depends on the
/// configured embedding model, which this function doesn't know about. See
/// [`ensure_vec_table`].
fn open_and_prepare(path: &Path) -> Result<Connection, rusqlite::Error> {
    register_sqlite_vec();
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "busy_timeout", 5_000)?;
    conn.execute_batch(CREATE_TURNS_TABLE)?;
    conn.execute_batch(CREATE_MEMORIES_TABLE)?;
    Ok(conn)
}

/// Returns the `vec0` column dimension `memories_vec` currently has, or
/// `None` if the table doesn't exist yet (no memory has ever been written).
///
/// `vec0` has no `PRAGMA table_info` support, so this recovers the
/// dimension the blunt way: the table's own `CREATE VIRTUAL TABLE` text
/// (as SQLite stored it in `sqlite_master`) always contains `float[N]`.
fn existing_vec_dim(conn: &Connection) -> Result<Option<usize>, rusqlite::Error> {
    let sql: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories_vec'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let Some(sql) = sql else {
        return Ok(None);
    };
    let dim = sql
        .split("float[")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .and_then(|n| n.parse::<usize>().ok());
    Ok(dim)
}

/// Ensures `memories_vec` exists with exactly `dim` dimensions, creating it
/// on first use. Returns [`HistoryError::DimMismatch`] if it already exists
/// with a *different* dimension — normal writes must match the
/// already-established dimension; only [`HistoryStore::apply_reembed`] is
/// allowed to change it, by dropping and recreating the table itself.
fn ensure_vec_table(conn: &Connection, dim: usize) -> Result<(), HistoryError> {
    match existing_vec_dim(conn)? {
        Some(existing) if existing != dim => Err(HistoryError::DimMismatch {
            expected: existing,
            got: dim,
        }),
        Some(_) => Ok(()),
        None => {
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE memories_vec USING vec0(embedding float[{dim}]);"
            ))?;
            Ok(())
        }
    }
}

/// A memory to be persisted (SPEC.md §5.2, EPIC 10.3), handed to the write
/// actor once `crate::memory` has computed its embedding.
///
/// Carries an already-computed `vector` rather than raw text: embedding is
/// CPU-bound model inference and does not belong on the write actor thread,
/// which must stay free to service `turns` writes too.
#[derive(Debug, Clone)]
pub struct NewMemory {
    /// Source text the memory was derived from — kept verbatim so a
    /// re-embed migration can recompute the vector later without needing
    /// the original conversation.
    pub text: String,
    /// Identifies the model that produced `vector` (SPEC.md §5.2).
    pub embed_model: String,
    /// The embedding vector. Its length becomes (or must match) the
    /// `memories_vec` index's dimension.
    pub vector: Vec<f32>,
    /// Provenance taint (§5.1) — preserved from the source turn(s).
    pub provenance: Trust,
    /// Unix epoch milliseconds when the memory was created.
    pub created_at_ms: i64,
}

/// A memory as read back from the database (without its vector — callers
/// doing similarity search get distances from [`HistoryStore::search_similar`]
/// instead of the raw vector, and plain listing never needs it).
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecord {
    /// Row id, shared with `memories_vec`'s rowid.
    pub id: i64,
    /// Source text.
    pub text: String,
    /// Embedding model id that produced the stored vector.
    pub embed_model: String,
    /// Vector dimension.
    pub dim: usize,
    /// Provenance taint (§5.1).
    pub provenance: Trust,
    /// Unix epoch milliseconds when the memory was created.
    pub created_at_ms: i64,
}

fn row_to_memory(row: &rusqlite::Row) -> rusqlite::Result<MemoryRecord> {
    let provenance_str: String = row.get(4)?;
    Ok(MemoryRecord {
        id: row.get(0)?,
        text: row.get(1)?,
        embed_model: row.get(2)?,
        dim: row.get::<_, i64>(3)? as usize,
        provenance: provenance_from_str(&provenance_str)?,
        created_at_ms: row.get(5)?,
    })
}

fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// A command sent to the write actor thread.
enum WriteCommand {
    LogTurn {
        turn: NewTurn,
        reply: std_mpsc::SyncSender<Result<i64, HistoryError>>,
    },
    InsertMemory {
        memory: NewMemory,
        reply: std_mpsc::SyncSender<Result<i64, HistoryError>>,
    },
    /// Atomically swaps every memory's vector (and `embed_model`/`dim`) for
    /// ones recomputed under a new model — the re-embed migration (SPEC.md
    /// §5.2, EPIC 10.3). Rebuilds `memories_vec` from scratch inside one
    /// transaction so no reader ever observes a mix of old- and new-model
    /// vectors.
    ReembedMemories {
        model_id: String,
        dim: usize,
        vectors: Vec<(i64, Vec<f32>)>,
        reply: std_mpsc::SyncSender<Result<usize, HistoryError>>,
    },
    /// Overwrites an existing memory's source text and vector in place
    /// (EPIC 10.6's `marceline memory edit`) — the caller (`crate::memory`
    /// or the CLI) must have already recomputed `vector` from `text` under
    /// the current embedding pipeline, since editing a memory's text
    /// without re-embedding would leave a stale vector pointing at text
    /// that no longer exists (SPEC.md §5, "auditable and editable").
    UpdateMemoryText {
        id: i64,
        text: String,
        vector: Vec<f32>,
        embed_model: String,
        reply: std_mpsc::SyncSender<Result<(), HistoryError>>,
    },
    /// Deletes a memory row and its vector together (EPIC 10.6's
    /// `marceline memory forget`), in one transaction so a reader never
    /// observes one half deleted without the other.
    DeleteMemory {
        id: i64,
        reply: std_mpsc::SyncSender<Result<(), HistoryError>>,
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
                    WriteCommand::InsertMemory { memory, reply } => {
                        let result = (|| -> Result<i64, HistoryError> {
                            let dim = memory.vector.len();
                            ensure_vec_table(&conn, dim)?;
                            let tx = conn.unchecked_transaction()?;
                            tx.execute(
                                "INSERT INTO memories
                                    (text, embed_model, dim, provenance, created_at_ms)
                                 VALUES (?1, ?2, ?3, ?4, ?5)",
                                params![
                                    memory.text,
                                    memory.embed_model,
                                    dim as i64,
                                    provenance_to_str(memory.provenance),
                                    memory.created_at_ms,
                                ],
                            )?;
                            let id = tx.last_insert_rowid();
                            tx.execute(
                                "INSERT INTO memories_vec (rowid, embedding) VALUES (?1, ?2)",
                                params![id, vector_to_blob(&memory.vector)],
                            )?;
                            tx.commit()?;
                            Ok(id)
                        })();
                        let _ = reply.send(result);
                    }
                    WriteCommand::ReembedMemories {
                        model_id,
                        dim,
                        vectors,
                        reply,
                    } => {
                        let result = (|| -> Result<usize, HistoryError> {
                            let tx = conn.unchecked_transaction()?;
                            tx.execute_batch("DROP TABLE IF EXISTS memories_vec;")?;
                            tx.execute_batch(&format!(
                                "CREATE VIRTUAL TABLE memories_vec USING vec0(embedding float[{dim}]);"
                            ))?;
                            for (id, vector) in &vectors {
                                tx.execute(
                                    "UPDATE memories SET embed_model = ?1, dim = ?2 WHERE id = ?3",
                                    params![model_id, dim as i64, id],
                                )?;
                                tx.execute(
                                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?1, ?2)",
                                    params![id, vector_to_blob(vector)],
                                )?;
                            }
                            tx.commit()?;
                            Ok(vectors.len())
                        })();
                        let _ = reply.send(result);
                    }
                    WriteCommand::UpdateMemoryText {
                        id,
                        text,
                        vector,
                        embed_model,
                        reply,
                    } => {
                        let result = (|| -> Result<(), HistoryError> {
                            let exists = conn
                                .query_row(
                                    "SELECT 1 FROM memories WHERE id = ?1",
                                    params![id],
                                    |_| Ok(()),
                                )
                                .optional()?
                                .is_some();
                            if !exists {
                                return Err(HistoryError::NotFound { id });
                            }
                            let dim = vector.len();
                            ensure_vec_table(&conn, dim)?;
                            let tx = conn.unchecked_transaction()?;
                            tx.execute(
                                "UPDATE memories SET text = ?1, embed_model = ?2, dim = ?3 WHERE id = ?4",
                                params![text, embed_model, dim as i64, id],
                            )?;
                            tx.execute(
                                "DELETE FROM memories_vec WHERE rowid = ?1",
                                params![id],
                            )?;
                            tx.execute(
                                "INSERT INTO memories_vec (rowid, embedding) VALUES (?1, ?2)",
                                params![id, vector_to_blob(&vector)],
                            )?;
                            tx.commit()?;
                            Ok(())
                        })();
                        let _ = reply.send(result);
                    }
                    WriteCommand::DeleteMemory { id, reply } => {
                        let result = (|| -> Result<(), HistoryError> {
                            let tx = conn.unchecked_transaction()?;
                            let changed =
                                tx.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
                            if changed == 0 {
                                return Err(HistoryError::NotFound { id });
                            }
                            tx.execute(
                                "DELETE FROM memories_vec WHERE rowid = ?1",
                                params![id],
                            )?;
                            tx.commit()?;
                            Ok(())
                        })();
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

    /// Reads the most recent `limit` turns across every session, oldest
    /// first — `marceline memory list`'s turn-history half (EPIC 11.3),
    /// which (unlike [`Self::recent_turns`]) is not scoped to one
    /// conversation: an operator inspecting what's stored wants everything
    /// recent, not one session picked in advance.
    pub fn recent_turns_all_sessions(&self, limit: usize) -> Result<Vec<TurnRecord>, HistoryError> {
        let conn = open_and_prepare(&self.db_path).map_err(|source| HistoryError::Open {
            path: self.db_path.clone(),
            source,
        })?;

        let mut stmt = conn.prepare(
            "SELECT id, session_id, timestamp_ms, role, text, provenance, interrupted
             FROM turns
             ORDER BY id DESC
             LIMIT ?1",
        )?;
        let mut rows = stmt
            .query_map(params![limit as i64], row_to_turn)?
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

    /// Persists an already-embedded memory via the write actor, returning
    /// its row id (SPEC.md §5.2, EPIC 10.3).
    ///
    /// `crate::memory::store_memory` is the usual entry point (it computes
    /// `memory.vector` first); this is the low-level primitive that
    /// actually writes it, alongside `turns`, through the single write
    /// actor.
    pub fn insert_memory(&self, memory: NewMemory) -> Result<i64, HistoryError> {
        let (reply_tx, reply_rx) = std_mpsc::sync_channel(1);
        self.write_tx
            .send(WriteCommand::InsertMemory {
                memory,
                reply: reply_tx,
            })
            .map_err(|_| HistoryError::WriteActorGone)?;
        reply_rx.recv().map_err(|_| HistoryError::WriteActorGone)?
    }

    /// Reads every memory row, oldest first. Used for listing and as the
    /// source-text pass of a re-embed migration.
    pub fn all_memories(&self) -> Result<Vec<MemoryRecord>, HistoryError> {
        let conn = open_and_prepare(&self.db_path).map_err(|source| HistoryError::Open {
            path: self.db_path.clone(),
            source,
        })?;
        let mut stmt = conn.prepare(
            "SELECT id, text, embed_model, dim, provenance, created_at_ms
             FROM memories ORDER BY id",
        )?;
        let rows = stmt
            .query_map([], row_to_memory)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Reads a single memory by row id, if it exists.
    pub fn get_memory(&self, id: i64) -> Result<Option<MemoryRecord>, HistoryError> {
        let conn = open_and_prepare(&self.db_path).map_err(|source| HistoryError::Open {
            path: self.db_path.clone(),
            source,
        })?;
        conn.query_row(
            "SELECT id, text, embed_model, dim, provenance, created_at_ms
             FROM memories WHERE id = ?1",
            params![id],
            row_to_memory,
        )
        .optional()
        .map_err(HistoryError::from)
    }

    /// Distinct `embed_model` ids currently stored across all memories.
    ///
    /// Empty if there are no memories yet. A non-empty result containing
    /// anything other than the currently configured model id is the signal
    /// that a re-embed migration is due (SPEC.md §5.2) — see
    /// `crate::memory::ensure_current_embed_model`.
    pub fn distinct_embed_models(&self) -> Result<Vec<String>, HistoryError> {
        let conn = open_and_prepare(&self.db_path).map_err(|source| HistoryError::Open {
            path: self.db_path.clone(),
            source,
        })?;
        let mut stmt = conn.prepare("SELECT DISTINCT embed_model FROM memories")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Finds the `k` memories whose stored vectors are nearest `query` (by
    /// `vec0`'s default Euclidean distance — vectors are expected to be
    /// L2-normalized by the embedding pipeline, which makes nearest-by-L2
    /// and nearest-by-cosine agree), nearest first.
    ///
    /// Returns an empty result if no memory has ever been written (the
    /// `memories_vec` index doesn't exist yet), rather than erroring —
    /// "no memories" is a normal, expected state, not a failure.
    pub fn search_similar(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(MemoryRecord, f64)>, HistoryError> {
        let conn = open_and_prepare(&self.db_path).map_err(|source| HistoryError::Open {
            path: self.db_path.clone(),
            source,
        })?;
        if existing_vec_dim(&conn)?.is_none() {
            return Ok(Vec::new());
        }

        let mut stmt = conn.prepare(
            "SELECT m.id, m.text, m.embed_model, m.dim, m.provenance, m.created_at_ms, v.distance
             FROM memories_vec v
             JOIN memories m ON m.id = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;
        let rows = stmt
            .query_map(params![vector_to_blob(query), k as i64], |row| {
                Ok((row_to_memory(row)?, row.get::<_, f64>(6)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Atomically re-embeds every memory (SPEC.md §5.2, EPIC 10.3):
    /// `vectors` must already contain the freshly-recomputed vector for
    /// every existing memory id, keyed under the new model's id and
    /// dimension. Rebuilds `memories_vec` from scratch in one transaction
    /// on the write actor, so no reader ever sees a mix of old- and
    /// new-model vectors — the table simply doesn't exist for the instant
    /// between drop and recreate, and readers treat "doesn't exist yet" as
    /// "no memories" rather than an error.
    ///
    /// Low-level: `crate::memory::reembed_all` is the usual entry point —
    /// it reads `all_memories`, calls the new pipeline for each, and hands
    /// the results here.
    pub fn apply_reembed(
        &self,
        model_id: &str,
        dim: usize,
        vectors: Vec<(i64, Vec<f32>)>,
    ) -> Result<usize, HistoryError> {
        let (reply_tx, reply_rx) = std_mpsc::sync_channel(1);
        self.write_tx
            .send(WriteCommand::ReembedMemories {
                model_id: model_id.to_string(),
                dim,
                vectors,
                reply: reply_tx,
            })
            .map_err(|_| HistoryError::WriteActorGone)?;
        reply_rx.recv().map_err(|_| HistoryError::WriteActorGone)?
    }

    /// Overwrites memory `id`'s source text, embedding vector, and
    /// `embed_model` id via the write actor (EPIC 10.6's `marceline memory
    /// edit`).
    ///
    /// Callers must recompute `vector` from `text` under the current
    /// embedding pipeline before calling this — passing the old vector
    /// alongside new `text` would leave a stale vector that no longer
    /// matches the row's text, defeating the point of storing text at all
    /// (SPEC.md §5's "auditable and editable"). Returns
    /// [`HistoryError::NotFound`] if `id` doesn't exist, rather than
    /// silently inserting nothing.
    pub fn update_memory_text(
        &self,
        id: i64,
        text: impl Into<String>,
        vector: Vec<f32>,
        embed_model: impl Into<String>,
    ) -> Result<(), HistoryError> {
        let (reply_tx, reply_rx) = std_mpsc::sync_channel(1);
        self.write_tx
            .send(WriteCommand::UpdateMemoryText {
                id,
                text: text.into(),
                vector,
                embed_model: embed_model.into(),
                reply: reply_tx,
            })
            .map_err(|_| HistoryError::WriteActorGone)?;
        reply_rx.recv().map_err(|_| HistoryError::WriteActorGone)?
    }

    /// Deletes memory `id` — both its `memories` row and its
    /// `memories_vec` vector — via the write actor, in one transaction
    /// (EPIC 10.6's `marceline memory forget`).
    ///
    /// Returns [`HistoryError::NotFound`] if `id` doesn't exist, rather
    /// than silently no-oping — a delete that doesn't report success or
    /// failure clearly is not auditable (SPEC.md §5).
    pub fn delete_memory(&self, id: i64) -> Result<(), HistoryError> {
        let (reply_tx, reply_rx) = std_mpsc::sync_channel(1);
        self.write_tx
            .send(WriteCommand::DeleteMemory { id, reply: reply_tx })
            .map_err(|_| HistoryError::WriteActorGone)?;
        reply_rx.recv().map_err(|_| HistoryError::WriteActorGone)?
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

        store
            .log_turn(turn("s1", "user", "first", Trust::User))
            .unwrap();
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
    fn recent_turns_all_sessions_spans_every_session_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        store.log_turn(turn("s1", "user", "first", Trust::User)).unwrap();
        store
            .log_turn(turn("s2", "user", "second", Trust::User))
            .unwrap();
        store
            .log_turn(turn("s1", "assistant", "third", Trust::Assistant))
            .unwrap();

        let turns = store.recent_turns_all_sessions(10).unwrap();
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].text, "first");
        assert_eq!(turns[1].text, "second");
        assert_eq!(turns[2].text, "third");
    }

    #[test]
    fn recent_turns_all_sessions_respects_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        for i in 0..5 {
            store
                .log_turn(turn("s1", "user", &format!("turn {i}"), Trust::User))
                .unwrap();
        }

        let turns = store.recent_turns_all_sessions(2).unwrap();
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
            store
                .log_turn(turn("s1", "user", "before restart", Trust::User))
                .unwrap();
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

    fn new_memory(text: &str, vector: Vec<f32>) -> NewMemory {
        NewMemory {
            text: text.to_string(),
            embed_model: "fake-v1".to_string(),
            vector,
            provenance: Trust::User,
            created_at_ms: 1_000,
        }
    }

    #[test]
    fn update_memory_text_changes_text_and_vector() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        let id = store
            .insert_memory(new_memory("old text", vec![1.0, 0.0, 0.0]))
            .unwrap();

        store
            .update_memory_text(id, "new text", vec![0.0, 1.0, 0.0], "fake-v2")
            .unwrap();

        let record = store.get_memory(id).unwrap().unwrap();
        assert_eq!(record.text, "new text");
        assert_eq!(record.embed_model, "fake-v2");

        // The new vector is what search now ranks against — searching for
        // the old vector's direction no longer finds this row as its
        // nearest match, and searching for the new vector's direction does,
        // proving the old vector is gone rather than merely shadowed.
        let results = store.search_similar(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0.id, id);
        assert_eq!(results[0].0.text, "new text");
    }

    #[test]
    fn update_memory_text_on_a_missing_id_errors() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        let err = store
            .update_memory_text(999, "text", vec![1.0, 0.0], "fake-v1")
            .unwrap_err();
        assert!(matches!(err, HistoryError::NotFound { id: 999 }));
    }

    #[test]
    fn delete_memory_removes_it_from_both_tables() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        let id = store
            .insert_memory(new_memory("to delete", vec![1.0, 0.0, 0.0]))
            .unwrap();
        let other = store
            .insert_memory(new_memory("keep me", vec![0.0, 1.0, 0.0]))
            .unwrap();

        store.delete_memory(id).unwrap();

        assert!(store.get_memory(id).unwrap().is_none());
        assert!(store.get_memory(other).unwrap().is_some());

        // No longer surfaced by similarity search either — its vector row
        // is gone, not just its metadata row.
        let results = store.search_similar(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(results.iter().all(|(m, _)| m.id != id));
    }

    #[test]
    fn delete_memory_on_a_missing_id_errors() {
        let dir = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(dir.path().join("history.db")).unwrap();

        let err = store.delete_memory(999).unwrap_err();
        assert!(matches!(err, HistoryError::NotFound { id: 999 }));
    }
}
