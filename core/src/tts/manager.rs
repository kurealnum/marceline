//! Launching a TTS worker and connecting to it (SPEC.md §2.2, §2.4).
//!
//! Mirrors [`crate::stt::manager`]'s launch path: publish a [`WorkerSpec`]
//! to a [`Supervisor`], which keeps the process alive (restart-on-crash)
//! until told to stop, then poll the health view until the worker reports
//! `Up` before connecting. No hot-swap here (unlike STT's model swap,
//! EPIC 3.4) — nothing in EPIC 5 needs a running worker's voice changed
//! without a restart, so it is not built ahead of that need.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::GrpcTtsEngine;
use crate::config::TtsConfig;
use crate::engine::EngineError;
use crate::supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};

/// Backend name used in errors raised here.
const BACKEND: &str = "tts";

/// Name the TTS worker is registered under in the health view.
pub const WORKER_NAME: &str = "tts";

/// How long to wait for a freshly launched worker to report healthy.
///
/// Kokoro/Piper are both light models (§10) — nowhere near Whisper
/// `large-v3`'s multi-gigabyte load — so this is well under STT's
/// `SWAP_TIMEOUT`.
pub const LAUNCH_TIMEOUT: Duration = Duration::from_secs(60);

/// Interval between readiness polls while waiting for the worker to come up.
const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// How many failed launches to tolerate before giving up.
///
/// The supervisor retries forever by design, but a caller waiting on first
/// connection should not stall for the full [`LAUNCH_TIMEOUT`] behind a
/// worker that cannot start at all — a missing venv or a bad voice id
/// fails the same way every time.
const MAX_LAUNCH_ATTEMPTS: u32 = 3;

/// Where a TTS worker lives on disk.
///
/// Not in `config.toml`: these are properties of the checkout, not of how
/// the user wants Marceline to run, and `[tts]` deliberately holds only the
/// latter (§3.1).
#[derive(Debug, Clone)]
pub struct TtsWorkerPaths {
    /// Python interpreter to run, normally the worker's venv.
    pub python: PathBuf,
    /// The worker entrypoint script.
    pub script: PathBuf,
    /// Unix domain socket the worker binds.
    pub socket_path: PathBuf,
}

impl TtsWorkerPaths {
    /// Paths for a backend name, relative to the repository root.
    ///
    /// `kokoro` and `piper` are separate worker directories, so the
    /// backend selects which script runs — the mechanism by which
    /// `[tts].backend` swaps implementations (EPIC 5.5).
    pub fn for_backend(backend: &str) -> Self {
        let dir = PathBuf::from("workers").join(backend_dir(backend));
        Self {
            python: dir.join(".venv/bin/python"),
            script: dir.join("worker.py"),
            socket_path: PathBuf::from("/tmp/marceline-tts.sock"),
        }
    }
}

/// Maps a `[tts].backend` value to its worker directory.
///
/// Unknown backends fall through to their own name, so adding a worker
/// directory is enough to add a backend — no match arm to remember.
fn backend_dir(backend: &str) -> &str {
    match backend {
        // The default Kokoro worker (EPIC 5.1) lives in `workers/tts`,
        // named for the stage rather than the model — `workers/kokoro`
        // would read oddly once Piper is also just a TTS backend.
        "kokoro" => "tts",
        other => other,
    }
}

/// Builds the worker spec for a `[tts]` config block.
///
/// `model_id` carries the voice id: [`WorkerSpec::command`] always passes
/// `--model-id`, per the template's convention every worker follows
/// (EPIC 0.4) — TTS workers read it as their default voice (see
/// `tts_service.parse_args`).
pub fn worker_spec(config: &TtsConfig, paths: &TtsWorkerPaths) -> WorkerSpec {
    WorkerSpec {
        name: WORKER_NAME.to_string(),
        python: paths.python.clone(),
        script: paths.script.clone(),
        socket_path: paths.socket_path.clone(),
        model_id: config.voice.clone(),
        device: config.device,
    }
}

/// Launches the TTS worker described by `config` and connects to it.
///
/// Spawns the supervisor as a background task; it keeps the worker alive
/// (restart-on-crash) until `shutdown` is set. Returns once the worker
/// reports healthy and the gRPC connection succeeds.
pub async fn launch(
    config: &TtsConfig,
    paths: TtsWorkerPaths,
    health: HealthView,
    shutdown: watch::Receiver<bool>,
    cancel: CancellationToken,
) -> Result<GrpcTtsEngine, EngineError> {
    let spec = worker_spec(config, &paths);
    let socket_path = spec.socket_path.clone();
    let (_spec_tx, spec_rx) = watch::channel(spec);

    tokio::spawn(Supervisor::new(spec_rx, Arc::clone(&health), shutdown).run());

    connect_when_ready(&socket_path, &health, cancel, LAUNCH_TIMEOUT).await
}

/// Waits for the worker to report healthy, then connects to it.
///
/// Polls rather than assuming: a connect that succeeds too early can land
/// on a worker that has not finished loading. The health view flipping to
/// `Up` is the signal that the model actually finished loading.
async fn connect_when_ready(
    socket_path: &std::path::Path,
    health: &HealthView,
    cancel: CancellationToken,
    timeout: Duration,
) -> Result<GrpcTtsEngine, EngineError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last_err = None;
    let mut restarts = 0;
    let mut previous = None;

    while tokio::time::Instant::now() < deadline {
        let state = health.read().await.get(WORKER_NAME).copied();

        if state == Some(WorkerState::Restarting) && previous != Some(WorkerState::Restarting) {
            restarts += 1;
            if restarts > MAX_LAUNCH_ATTEMPTS {
                return Err(last_err.unwrap_or(EngineError::Worker {
                    backend: BACKEND,
                    message: format!(
                        "worker failed to start {MAX_LAUNCH_ATTEMPTS} times; \
                         check the worker venv and voice id"
                    ),
                }));
            }
        }
        previous = state;

        if state == Some(WorkerState::Up) {
            match GrpcTtsEngine::connect(socket_path, cancel.clone()).await {
                Ok(engine) => return Ok(engine),
                Err(err) => last_err = Some(err),
            }
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }

    Err(last_err.unwrap_or(EngineError::Timeout {
        backend: BACKEND,
        elapsed_ms: timeout.as_millis() as u64,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_backend_maps_to_the_tts_worker_directory() {
        let paths = TtsWorkerPaths::for_backend("kokoro");
        assert_eq!(paths.script, PathBuf::from("workers/tts/worker.py"));
        assert_eq!(paths.python, PathBuf::from("workers/tts/.venv/bin/python"));
    }

    #[test]
    fn an_unknown_backend_uses_its_own_directory_name() {
        // Adding a worker directory is enough to add a backend.
        let paths = TtsWorkerPaths::for_backend("piper");
        assert_eq!(paths.script, PathBuf::from("workers/piper/worker.py"));
    }
}
