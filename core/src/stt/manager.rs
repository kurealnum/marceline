//! STT worker lifecycle and model hot-swap (SPEC.md §2.4, EPIC 3.4).
//!
//! Swapping the STT model is a **worker restart**, not an in-place model
//! reload: relaunch the Python process with a different `--model-id` and
//! reconnect. Deliberately blunt — it reuses the crash-isolation the
//! supervisor already provides (§2.2), and an in-place reload would mean
//! teaching the worker to unload CUDA memory safely mid-life for no gain.
//!
//! The Rust trait impl is unchanged across a swap. That is the whole
//! "hot-swappable via a config line" claim: [`GrpcSttEngine`] talks to
//! whatever is on the socket, so `whisper` -> `faster-whisper` (EPIC 3.5)
//! is a different worker script, not different client code.
//!
//! Two things this type exists to get right:
//!
//! * **No restart mid-transcription.** A swap waits for in-flight work.
//!   Killing the worker mid-inference would surface as a spurious error on
//!   a turn the user is waiting on.
//! * **Reconnect is transparent.** Callers hold an [`SttManager`], not an
//!   engine, so the connection replaced underneath them is not their
//!   problem.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{watch, Mutex};
use tokio_util::sync::CancellationToken;

use super::{GrpcSttEngine, SttEngine, SttInfo};
use crate::audio::AudioChunk;
use crate::config::SttConfig;
use crate::device::Device;
use crate::engine::EngineError;
use crate::supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};
use crate::transcribe::{transcribe_segment, Transcription};

/// Backend name used in errors raised here.
const BACKEND: &str = "stt";

/// Name the STT worker is registered under in the health view.
pub const WORKER_NAME: &str = "stt";

/// How long to wait for a swapped-in worker to report healthy and answer.
///
/// Generous: this covers a cold model load, which for `large-v3` means
/// reading gigabytes of weights onto the GPU.
pub const SWAP_TIMEOUT: Duration = Duration::from_secs(180);

/// Interval between readiness polls while waiting for a swap to land.
const SWAP_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// How many failed launches to tolerate before giving up on a worker.
///
/// The supervisor retries forever by design (a model that OOMs once may
/// load next time), but a caller *waiting* on first connection should not
/// stall for the full [`SWAP_TIMEOUT`] behind a worker that cannot start at
/// all — a missing venv or a bad model id fails the same way every time.
const MAX_LAUNCH_ATTEMPTS: u32 = 3;

/// Where the STT worker lives on disk.
///
/// Not in `config.toml`: these are properties of the checkout, not of how
/// the user wants Marceline to run, and `[stt]` deliberately holds only the
/// latter (§3.1).
#[derive(Debug, Clone)]
pub struct SttWorkerPaths {
    /// Python interpreter to run, normally the worker's venv.
    pub python: PathBuf,
    /// The worker entrypoint script.
    pub script: PathBuf,
    /// Unix domain socket the worker binds.
    pub socket_path: PathBuf,
}

impl SttWorkerPaths {
    /// Paths for a backend name, relative to the repository root.
    ///
    /// `whisper` and `faster-whisper` are separate worker directories, so
    /// the backend selects which script runs — the mechanism by which
    /// `[stt].backend` swaps implementations (EPIC 3.5).
    pub fn for_backend(backend: &str) -> Self {
        let dir = PathBuf::from("workers").join(backend_dir(backend));
        Self {
            python: dir.join(".venv/bin/python"),
            script: dir.join("worker.py"),
            socket_path: PathBuf::from("/tmp/marceline-stt.sock"),
        }
    }
}

/// Maps a `[stt].backend` value to its worker directory.
///
/// Unknown backends fall through to their own name, so adding a worker
/// directory is enough to add a backend — no match arm to remember.
fn backend_dir(backend: &str) -> &str {
    match backend {
        // The default HF whisper worker (EPIC 3.1).
        "whisper" => "stt",
        other => other,
    }
}

/// Owns the STT worker's lifecycle and the client connected to it.
///
/// Hand this to the orchestrator instead of an [`SttEngine`]: it transcribes
/// like an engine, and it can also swap the model underneath itself.
pub struct SttManager {
    /// The current connection. Replaced after a swap.
    ///
    /// Also the in-flight guard: [`transcribe`][SttManager::transcribe]
    /// holds this lock for the duration of a turn, so a swap cannot land
    /// mid-inference.
    engine: Arc<Mutex<GrpcSttEngine>>,
    cancel: CancellationToken,
    lang: String,
    /// Present only when this process launched the worker. `None` means
    /// there is nothing here to restart, which is why
    /// [`swap_model`][SttManager::swap_model] can fail up front rather than
    /// pretending.
    supervised: Option<Supervised>,
}

/// The pieces only a manager that owns its worker process has.
struct Supervised {
    /// Published to the supervisor; a new value restarts the worker.
    spec: watch::Sender<WorkerSpec>,
    health: HealthView,
    paths: SttWorkerPaths,
    device: Device,
}

impl SttManager {
    /// Launches the STT worker described by `config` and connects to it.
    ///
    /// Spawns the supervisor as a background task; it keeps the worker
    /// alive (restart-on-crash) until `shutdown` is set.
    pub async fn start(
        config: &SttConfig,
        paths: SttWorkerPaths,
        health: HealthView,
        shutdown: watch::Receiver<bool>,
        cancel: CancellationToken,
    ) -> Result<Self, EngineError> {
        let spec = worker_spec(config, &paths);
        let socket_path = spec.socket_path.clone();
        let (spec_tx, spec_rx) = watch::channel(spec);

        tokio::spawn(Supervisor::new(spec_rx, Arc::clone(&health), shutdown).run());

        let engine = connect_when_ready(&socket_path, &health, cancel.clone(), SWAP_TIMEOUT).await?;

        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            cancel,
            lang: config.lang.clone(),
            supervised: Some(Supervised {
                spec: spec_tx,
                health,
                paths,
                device: config.device,
            }),
        })
    }

    /// Attaches to a worker someone else is already running.
    ///
    /// For `marceline transcribe --socket <path>`: no supervisor, so no
    /// swap either — [`swap_model`][SttManager::swap_model] reports that
    /// rather than pretending to restart a process it does not own.
    pub async fn attach(
        socket_path: PathBuf,
        cancel: CancellationToken,
        lang: String,
    ) -> Result<Self, EngineError> {
        let engine = GrpcSttEngine::connect(&socket_path, cancel.clone()).await?;
        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            cancel,
            lang,
            supervised: None,
        })
    }

    /// Transcribes one segment on the current worker.
    ///
    /// Holds the engine lock for the whole turn, which is what stops a
    /// concurrent [`swap_model`][SttManager::swap_model] from killing the
    /// worker mid-inference.
    pub async fn transcribe(
        &self,
        segment: AudioChunk,
        timeout: Duration,
    ) -> Result<Transcription, EngineError> {
        let engine = self.engine.lock().await;
        transcribe_segment(&*engine, segment, timeout).await
    }

    /// Capabilities of the currently loaded model.
    pub async fn info(&self) -> SttInfo {
        self.engine.lock().await.info()
    }

    /// Restarts the worker on `model_id` (and optionally a new backend).
    ///
    /// Waits for any in-flight transcription first, then republishes the
    /// spec, waits for the new worker to report healthy, and reconnects.
    /// Returns the new worker's [`SttInfo`] — read from the worker, so a
    /// caller learns what actually loaded rather than what was asked for.
    ///
    /// A no-op swap (same backend and model) short-circuits: restarting to
    /// load identical weights would cost a minute of GPU time for nothing.
    pub async fn swap_model(
        &self,
        model_id: &str,
        backend: Option<&str>,
    ) -> Result<SttInfo, SwapError> {
        // Taken before anything else: the whole point is not to interrupt a
        // turn that is already running.
        let mut engine = self.engine.lock().await;

        let Some(supervised) = &self.supervised else {
            return Err(SwapError::NotSupervised);
        };

        let current = supervised.spec.borrow().clone();
        let paths = match backend {
            Some(backend) => SttWorkerPaths::for_backend(backend),
            None => supervised.paths.clone(),
        };

        if current.model_id == model_id && current.script == paths.script {
            // Restarting to load identical weights would cost a minute of
            // GPU time for no change.
            tracing::info!(model_id, "stt model already loaded, not restarting");
            return Ok(engine.info());
        }

        let next = WorkerSpec {
            name: WORKER_NAME.to_string(),
            python: paths.python.clone(),
            script: paths.script.clone(),
            socket_path: paths.socket_path.clone(),
            model_id: model_id.to_string(),
            device: supervised.device,
        };
        tracing::info!(
            from_model = %current.model_id,
            to_model = %model_id,
            script = %next.script.display(),
            "swapping stt model"
        );

        // Publishing the spec is what triggers the restart; the supervisor
        // kills the current worker and relaunches on the new id.
        supervised
            .spec
            .send(next)
            .map_err(|_| SwapError::NotSupervised)?;

        let reconnected = connect_when_ready(
            &paths.socket_path,
            &supervised.health,
            self.cancel.clone(),
            SWAP_TIMEOUT,
        )
        .await?;

        let info = reconnected.info();
        // Swap the connection under the lock, so nothing observes a
        // half-swapped manager.
        *engine = reconnected;
        tracing::info!(model = %info.name, "stt model swapped");
        Ok(info)
    }

    /// The language this manager launches workers with.
    pub fn lang(&self) -> &str {
        &self.lang
    }
}

/// Why a model swap could not be performed.
#[derive(Debug, thiserror::Error)]
pub enum SwapError {
    /// This manager attached to a worker it does not own, so there is no
    /// process for it to restart.
    #[error("cannot swap models: the stt worker is not supervised by this process")]
    NotSupervised,
    /// The new worker never came up, or could not be reached.
    #[error(transparent)]
    Engine(#[from] EngineError),
}

/// Builds the worker spec for an `[stt]` config block.
pub fn worker_spec(config: &SttConfig, paths: &SttWorkerPaths) -> WorkerSpec {
    WorkerSpec {
        name: WORKER_NAME.to_string(),
        python: paths.python.clone(),
        script: paths.script.clone(),
        socket_path: paths.socket_path.clone(),
        model_id: config.model.clone(),
        device: config.device,
    }
}

/// Waits for the worker to report healthy, then connects to it.
///
/// Polls rather than assuming: after a restart the old socket may still be
/// on disk for a moment, so a connect that succeeds too early can land on a
/// worker that is going away. The health view flipping to `Up` is the
/// signal that the *new* process finished loading its model.
async fn connect_when_ready(
    socket_path: &std::path::Path,
    health: &HealthView,
    cancel: CancellationToken,
    timeout: Duration,
) -> Result<GrpcSttEngine, EngineError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last_err = None;
    let mut restarts = 0;
    let mut previous = None;

    while tokio::time::Instant::now() < deadline {
        let state = health.read().await.get(WORKER_NAME).copied();

        // A worker that keeps dying is not going to become ready by being
        // waited on longer. Failing after a few cycles turns a silent
        // multi-minute stall into an error the user can act on — usually a
        // missing venv or a bad model id.
        if state == Some(WorkerState::Restarting) && previous != Some(WorkerState::Restarting) {
            restarts += 1;
            if restarts > MAX_LAUNCH_ATTEMPTS {
                return Err(last_err.unwrap_or(EngineError::Worker {
                    backend: BACKEND,
                    message: format!(
                        "worker failed to start {MAX_LAUNCH_ATTEMPTS} times; \
                         check the worker venv and model id"
                    ),
                }));
            }
        }
        previous = state;

        if state == Some(WorkerState::Up) {
            match GrpcSttEngine::connect(socket_path, cancel.clone()).await {
                Ok(engine) => return Ok(engine),
                Err(err) => last_err = Some(err),
            }
        }
        tokio::time::sleep(SWAP_POLL_INTERVAL).await;
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
    fn whisper_backend_maps_to_the_default_worker_directory() {
        let paths = SttWorkerPaths::for_backend("whisper");
        assert_eq!(paths.script, PathBuf::from("workers/stt/worker.py"));
        assert_eq!(paths.python, PathBuf::from("workers/stt/.venv/bin/python"));
    }

    #[test]
    fn an_unknown_backend_uses_its_own_directory_name() {
        // Adding a worker directory is enough to add a backend.
        let paths = SttWorkerPaths::for_backend("faster-whisper");
        assert_eq!(
            paths.script,
            PathBuf::from("workers/faster-whisper/worker.py")
        );
    }

    #[test]
    fn worker_spec_carries_the_configured_model_and_device() {
        let config = SttConfig {
            backend: "whisper".to_string(),
            model: "large-v3".to_string(),
            device: Device::Cuda,
            lang: "en".to_string(),
        };
        let spec = worker_spec(&config, &SttWorkerPaths::for_backend(&config.backend));

        assert_eq!(spec.name, WORKER_NAME);
        assert_eq!(spec.model_id, "large-v3");
        assert_eq!(spec.device, Device::Cuda);
        assert_eq!(spec.script, PathBuf::from("workers/stt/worker.py"));
    }
}
