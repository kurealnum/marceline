//! Local control-plane IPC for the daemon lifecycle (SPEC.md §11, EPIC
//! 11.1): lets `marceline status` ask a running daemon how it's doing
//! without reaching into worker processes directly (§11's "thin clients...
//! over local IPC" constraint).
//!
//! This is deliberately a separate, much simpler transport from
//! `crate::ipc`'s gRPC-to-Python-worker channel: the control plane only
//! ever talks Rust-process-to-Rust-process (the CLI and the daemon it just
//! started), so a hand-rolled JSON-line protocol over a unix domain socket
//! is enough — no protobuf schema to keep in sync with a worker template.
//!
//! Shutdown itself does *not* go through this socket: per SPEC.md §11.1,
//! `marceline stop` sends SIGTERM directly to the daemon's pid (from the
//! pidfile below) and the daemon's own SIGTERM handler runs the graceful
//! ordering (§2.5.1). Keeping the control socket read-only (status queries
//! only) means a client that can merely connect can never trigger a
//! shutdown it wasn't otherwise allowed to (sending a signal to the pid
//! requires owning/permission over that process, same as any other unix
//! signal).

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::watch;

use crate::orchestrator::ConversationState;
use crate::supervisor::{HealthView, WorkerState};

/// Directory the daemon's runtime files (pidfile, control socket) live in.
///
/// Reuses `[memory].db_path`'s parent directory (`MemoryConfig::expanded_db_path`,
/// normally `~/.marceline/`) rather than inventing a second runtime-directory
/// convention.
pub fn runtime_dir(db_path: &Path) -> PathBuf {
    db_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Path to the daemon's pidfile within `dir` (see [`runtime_dir`]).
pub fn pidfile_path(dir: &Path) -> PathBuf {
    dir.join("marceline.pid")
}

/// Path to the daemon's control socket within `dir` (see [`runtime_dir`]).
pub fn control_socket_path(dir: &Path) -> PathBuf {
    dir.join("control.sock")
}

/// Writes `pid` to `path`, creating `path`'s parent directory if needed.
pub fn write_pidfile(path: &Path, pid: u32) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, pid.to_string())
}

/// Reads the pid stored at `path`, if the file exists and parses cleanly.
pub fn read_pidfile(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Removes the pidfile at `path`, ignoring a missing file (already gone is
/// the goal state, not an error).
pub fn remove_pidfile(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// One request the CLI can send over the control socket.
#[derive(Debug, Serialize, Deserialize)]
pub enum ControlRequest {
    /// `marceline status`: report per-stage health and the current
    /// conversation state.
    Status,
    /// `marceline config set stt.model`/`stt.backend` (EPIC 11.2): restart
    /// the STT worker on a new model/backend without a full daemon
    /// restart. `backend` is `None` when only the model id changes.
    SwapSttModel {
        /// The new model id to load.
        model: String,
        /// The new STT backend, if it's changing too.
        backend: Option<String>,
    },
}

/// Health of one supervised stage, as reported over the wire — a
/// serializable mirror of [`WorkerState`], which lives in `core::supervisor`
/// for the in-process health view and has no reason to derive `serde`
/// itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageHealth {
    /// Process launched, not yet confirmed healthy.
    Starting,
    /// Process is running and healthy.
    Up,
    /// Process exited; a restart is pending.
    Restarting,
    /// Supervisor is shutting down; the worker will not be restarted.
    Stopped,
}

impl From<WorkerState> for StageHealth {
    fn from(state: WorkerState) -> Self {
        match state {
            WorkerState::Starting => StageHealth::Starting,
            WorkerState::Up => StageHealth::Up,
            WorkerState::Restarting => StageHealth::Restarting,
            WorkerState::Stopped => StageHealth::Stopped,
        }
    }
}

/// A serializable mirror of [`ConversationState`] (SPEC.md §2.5): the real
/// enum lives in `core::orchestrator` for the state machine's own use and
/// has no reason to carry `serde` derives itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireConversationState {
    /// Waiting for a wake word; nothing in flight.
    Idle,
    /// Wake word fired; collecting an utterance.
    Listening,
    /// VAD ended the utterance; STT is transcribing it.
    Transcribing,
    /// Final transcript handed to the LLM; may loop on tool calls.
    Thinking,
    /// Streamed tokens are sentence-chunked into TTS and playing.
    Speaking,
    /// A stage failed or timed out.
    Error,
}

impl From<ConversationState> for WireConversationState {
    fn from(state: ConversationState) -> Self {
        match state {
            ConversationState::Idle => WireConversationState::Idle,
            ConversationState::Listening => WireConversationState::Listening,
            ConversationState::Transcribing => WireConversationState::Transcribing,
            ConversationState::Thinking => WireConversationState::Thinking,
            ConversationState::Speaking => WireConversationState::Speaking,
            ConversationState::Error => WireConversationState::Error,
        }
    }
}

/// Per-stage health plus the current conversation state — `marceline
/// status`'s whole payload (EPIC 11.1's "done when").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusReport {
    /// Each supervised worker's name (e.g. `"stt"`, `"tts"`) to its health,
    /// in no particular order.
    pub workers: Vec<(String, StageHealth)>,
    /// The orchestrator's current state.
    pub state: WireConversationState,
}

/// One response to a [`ControlRequest`].
#[derive(Debug, Serialize, Deserialize)]
pub enum ControlResponse {
    /// Reply to [`ControlRequest::Status`].
    Status(StatusReport),
    /// Reply to a successful [`ControlRequest::SwapSttModel`], carrying the
    /// model id the worker actually reports having loaded.
    Swapped {
        /// Model id reported by the freshly (re)loaded worker.
        model: String,
    },
    /// Reply to a failed [`ControlRequest::SwapSttModel`] — e.g. the
    /// worker never came back healthy on the new model/backend.
    SwapFailed {
        /// Human-readable reason the swap failed.
        reason: String,
    },
}

/// Errors talking to the control socket.
#[derive(Debug, thiserror::Error)]
pub enum ControlError {
    /// Connecting to the socket failed — most likely no daemon is running.
    #[error("could not reach the daemon's control socket at {path}: {source}")]
    Connect {
        /// Socket path that was dialed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// A read or write on the connected socket failed.
    #[error("control socket I/O failed: {0}")]
    Io(#[from] io::Error),
    /// The response (or request) did not parse as the expected JSON shape.
    #[error("malformed control message: {0}")]
    Malformed(#[from] serde_json::Error),
    /// The daemon closed the connection without sending a response.
    #[error("the daemon closed the control socket without responding")]
    NoResponse,
}

/// Sends one request to a running daemon's control socket at `socket_path`
/// and returns its response. Used by the CLI's `status` subcommand —
/// never by anything inside the daemon process itself.
pub async fn send_request(
    socket_path: &Path,
    request: &ControlRequest,
) -> Result<ControlResponse, ControlError> {
    let stream =
        UnixStream::connect(socket_path)
            .await
            .map_err(|source| ControlError::Connect {
                path: socket_path.to_path_buf(),
                source,
            })?;
    let (read_half, mut write_half) = stream.into_split();
    let mut line = serde_json::to_string(request)?;
    line.push('\n');
    write_half.write_all(line.as_bytes()).await?;
    write_half.flush().await?;

    let mut reader = BufReader::new(read_half);
    let mut response_line = String::new();
    let n = reader.read_line(&mut response_line).await?;
    if n == 0 {
        return Err(ControlError::NoResponse);
    }
    Ok(serde_json::from_str(response_line.trim_end())?)
}

/// Serves the control socket at `socket_path` forever, answering
/// [`ControlRequest::Status`] from `stt_health`/`tts_health` and `state`,
/// and [`ControlRequest::SwapSttModel`] via `stt` (when present — a `None`
/// `stt` answers every swap request with [`ControlResponse::SwapFailed`],
/// which should not happen in practice since the daemon always has one,
/// but keeps this function callable from tests that don't).
///
/// Meant to run as a background task for the lifetime of the daemon
/// process (spawned by `cli::converse`'s daemon mode); shutdown happens
/// out-of-band via SIGTERM (see module docs), not through this socket, so
/// this function has no normal return path — the whole process exits
/// around it.
///
/// Binds fresh each daemon start: a stale socket file left behind by an
/// unclean previous exit is removed first, since `UnixListener::bind`
/// fails with `AddrInUse` on an existing path even if nothing is listening
/// on it.
pub async fn serve_control(
    socket_path: &Path,
    stt_health: HealthView,
    tts_health: HealthView,
    state: watch::Receiver<ConversationState>,
    stt: Option<std::sync::Arc<crate::stt::SttManager>>,
) -> io::Result<()> {
    let _ = std::fs::remove_file(socket_path);
    let listener = tokio::net::UnixListener::bind(socket_path)?;

    loop {
        let (stream, _addr) = listener.accept().await?;
        let stt_health = stt_health.clone();
        let tts_health = tts_health.clone();
        let state = state.clone();
        let stt = stt.clone();
        tokio::spawn(async move {
            let (read_half, mut write_half) = stream.into_split();
            let mut reader = BufReader::new(read_half);
            let mut line = String::new();
            if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                return;
            }
            let Ok(request) = serde_json::from_str::<ControlRequest>(line.trim_end()) else {
                return;
            };

            let response = match request {
                ControlRequest::Status => {
                    let mut workers: Vec<(String, StageHealth)> = stt_health
                        .read()
                        .await
                        .iter()
                        .map(|(name, s)| (name.clone(), StageHealth::from(*s)))
                        .collect();
                    workers.extend(
                        tts_health
                            .read()
                            .await
                            .iter()
                            .map(|(name, s)| (name.clone(), StageHealth::from(*s))),
                    );
                    ControlResponse::Status(StatusReport {
                        workers,
                        state: WireConversationState::from(*state.borrow()),
                    })
                }
                ControlRequest::SwapSttModel { model, backend } => match stt {
                    Some(stt) => match stt.swap_model(&model, backend.as_deref()).await {
                        Ok(info) => ControlResponse::Swapped { model: info.name },
                        Err(err) => ControlResponse::SwapFailed {
                            reason: err.to_string(),
                        },
                    },
                    None => ControlResponse::SwapFailed {
                        reason: "no STT worker is running in this daemon".to_string(),
                    },
                },
            };

            let mut out =
                serde_json::to_string(&response).expect("ControlResponse always serializes");
            out.push('\n');
            let _ = write_half.write_all(out.as_bytes()).await;
            let _ = write_half.flush().await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[test]
    fn pidfile_round_trips_and_creates_its_parent_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = pidfile_path(&dir.path().join("nested"));

        write_pidfile(&path, 4242).unwrap();
        assert_eq!(read_pidfile(&path), Some(4242));

        remove_pidfile(&path);
        assert_eq!(read_pidfile(&path), None);
    }

    #[test]
    fn reading_a_missing_pidfile_is_none_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_pidfile(&pidfile_path(dir.path())), None);
    }

    #[tokio::test]
    async fn status_request_reports_worker_health_and_conversation_state() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = control_socket_path(dir.path());

        let stt_health: HealthView = Arc::new(RwLock::new(HashMap::from([(
            "stt".to_string(),
            WorkerState::Up,
        )])));
        let tts_health: HealthView = Arc::new(RwLock::new(HashMap::from([(
            "tts".to_string(),
            WorkerState::Restarting,
        )])));
        let (_state_tx, state_rx) = watch::channel(ConversationState::Listening);

        let socket_for_server = socket_path.clone();
        tokio::spawn(async move {
            let _ = serve_control(&socket_for_server, stt_health, tts_health, state_rx, None).await;
        });
        // Give the listener a moment to bind before dialing it.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let response = send_request(&socket_path, &ControlRequest::Status)
            .await
            .unwrap();
        let ControlResponse::Status(report) = response else {
            panic!("expected a Status response, got {response:?}");
        };

        assert_eq!(report.state, WireConversationState::Listening);
        assert!(report
            .workers
            .contains(&("stt".to_string(), StageHealth::Up)));
        assert!(report
            .workers
            .contains(&("tts".to_string(), StageHealth::Restarting)));
    }

    #[tokio::test]
    async fn connecting_to_a_socket_nobody_is_serving_fails_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = control_socket_path(dir.path());

        let err = send_request(&socket_path, &ControlRequest::Status)
            .await
            .unwrap_err();
        assert!(matches!(err, ControlError::Connect { .. }));
    }
}
