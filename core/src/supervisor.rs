//! Launches, monitors, and restarts Python model workers (SPEC.md §2.2,
//! EPIC 0.6). A crashed model recovers via restart without taking the
//! daemon down. The health view built here is reused by the future
//! `marceline status` per-stage health report (EPIC 11.1).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::process::Command;
use tokio::sync::{watch, RwLock};
use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint, Uri};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;
use tower::service_fn;

/// Initial delay before the first restart attempt; doubles on each
/// consecutive crash up to [`MAX_BACKOFF`].
const INITIAL_BACKOFF: Duration = Duration::from_millis(200);
/// Ceiling on restart backoff so a persistently crashing worker still
/// gets retried at a bounded interval rather than backing off forever.
const MAX_BACKOFF: Duration = Duration::from_secs(10);
/// Interval between health-RPC polls while waiting for a worker to come up.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Maximum time to wait for a freshly spawned worker to report healthy.
const HEALTH_POLL_TIMEOUT: Duration = Duration::from_secs(10);

/// Static, launch-time description of one worker process, following the
/// standard CLI convention from the worker template (EPIC 0.4).
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    /// Human-readable worker name, used in logs and the health view (e.g. "stt").
    pub name: String,
    /// Path to the Python interpreter (typically the worker's venv) to run.
    pub python: PathBuf,
    /// Path to the worker's entrypoint script.
    pub script: PathBuf,
    /// Filesystem path of the unix domain socket the worker binds.
    pub socket_path: PathBuf,
    /// Model identifier passed to the worker.
    pub model_id: String,
    /// Compute device passed to the worker.
    pub device: String,
}

impl WorkerSpec {
    /// Builds the `Command` used to spawn this worker, per the template's
    /// `--socket-path`/`--model-id`/`--device` convention.
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.python);
        cmd.arg(&self.script)
            .arg("--socket-path")
            .arg(&self.socket_path)
            .arg("--model-id")
            .arg(&self.model_id)
            .arg("--device")
            .arg(&self.device);
        cmd
    }
}

/// Liveness of one supervised worker, as seen by other components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Process launched, not yet confirmed healthy.
    Starting,
    /// Process is running and its health RPC reports `SERVING`.
    Up,
    /// Process exited; a restart is pending (backoff).
    Restarting,
    /// Supervisor is shutting down; the worker will not be restarted.
    Stopped,
}

/// Shared, queryable health view: worker name -> current state. Other
/// components (e.g. `marceline status`, EPIC 11.1) read this without
/// depending on supervisor internals.
pub type HealthView = Arc<RwLock<HashMap<String, WorkerState>>>;

/// Supervises one worker: spawn, health-poll, restart-on-exit with
/// exponential backoff, until told to shut down.
pub struct Supervisor {
    spec: WorkerSpec,
    health: HealthView,
    shutdown: watch::Receiver<bool>,
}

impl Supervisor {
    /// Creates a supervisor for `spec`, sharing `health` with other
    /// supervised workers, and stopping (rather than restarting) once
    /// `shutdown` is set to `true`.
    pub fn new(spec: WorkerSpec, health: HealthView, shutdown: watch::Receiver<bool>) -> Self {
        Self {
            spec,
            health,
            shutdown,
        }
    }

    /// Runs the supervise loop until shutdown is signaled. Intended to be
    /// spawned as its own task per worker.
    pub async fn run(mut self) {
        let mut backoff = INITIAL_BACKOFF;
        let mut first_launch = true;

        loop {
            if *self.shutdown.borrow() {
                self.set_state(WorkerState::Stopped).await;
                return;
            }

            self.set_state(WorkerState::Starting).await;
            tracing::info!(worker = %self.spec.name, "spawning worker");

            let mut child = match self.spec.command().spawn() {
                Ok(child) => child,
                Err(err) => {
                    tracing::error!(worker = %self.spec.name, %err, "failed to spawn worker");
                    if self.wait_backoff_or_shutdown(&mut backoff).await {
                        self.set_state(WorkerState::Stopped).await;
                        return;
                    }
                    continue;
                }
            };

            if self.wait_healthy().await {
                if first_launch {
                    tracing::info!(worker = %self.spec.name, "worker up");
                } else {
                    tracing::info!(worker = %self.spec.name, "worker restarted");
                }
                first_launch = false;
                self.set_state(WorkerState::Up).await;
                backoff = INITIAL_BACKOFF;
            } else {
                tracing::warn!(worker = %self.spec.name, "worker never became healthy");
            }

            tokio::select! {
                status = child.wait() => {
                    match status {
                        Ok(status) => tracing::warn!(worker = %self.spec.name, %status, "worker exited"),
                        Err(err) => tracing::error!(worker = %self.spec.name, %err, "failed to wait on worker"),
                    }
                }
                _ = self.shutdown.changed() => {
                    if *self.shutdown.borrow() {
                        tracing::info!(worker = %self.spec.name, "shutting down worker");
                        let _ = child.kill().await;
                        self.set_state(WorkerState::Stopped).await;
                        return;
                    }
                }
            }

            if *self.shutdown.borrow() {
                self.set_state(WorkerState::Stopped).await;
                return;
            }

            tracing::info!(worker = %self.spec.name, backoff_ms = backoff.as_millis() as u64, "worker restarting");
            self.set_state(WorkerState::Restarting).await;
            if self.wait_backoff_or_shutdown(&mut backoff).await {
                self.set_state(WorkerState::Stopped).await;
                return;
            }
        }
    }

    /// Sleeps for the current backoff (doubling it up to [`MAX_BACKOFF`]),
    /// waking early if shutdown is signaled. Returns `true` if shutdown
    /// fired during the wait.
    async fn wait_backoff_or_shutdown(&mut self, backoff: &mut Duration) -> bool {
        tokio::select! {
            _ = sleep(*backoff) => {}
            _ = self.shutdown.changed() => {}
        }
        *backoff = (*backoff * 2).min(MAX_BACKOFF);
        *self.shutdown.borrow()
    }

    async fn set_state(&self, state: WorkerState) {
        self.health.write().await.insert(self.spec.name.clone(), state);
    }

    /// Polls the worker's standard gRPC health-check RPC (over its UDS)
    /// until it reports `SERVING` or [`HEALTH_POLL_TIMEOUT`] elapses.
    async fn wait_healthy(&self) -> bool {
        let deadline = tokio::time::Instant::now() + HEALTH_POLL_TIMEOUT;
        while tokio::time::Instant::now() < deadline {
            if let Ok(mut client) = connect_health_client(&self.spec.socket_path).await {
                if let Ok(resp) = client
                    .check(HealthCheckRequest {
                        service: String::new(),
                    })
                    .await
                {
                    if resp.into_inner().status == ServingStatus::Serving as i32 {
                        return true;
                    }
                }
            }
            sleep(HEALTH_POLL_INTERVAL).await;
        }
        false
    }
}

/// Connects a `HealthClient` to a worker over its unix domain socket. The
/// URI is a placeholder required by `tonic::transport::Endpoint`; the
/// connector below ignores it and always dials `socket_path`.
async fn connect_health_client(
    socket_path: &Path,
) -> Result<HealthClient<Channel>, tonic::transport::Error> {
    let socket_path = socket_path.to_path_buf();
    let channel = Endpoint::try_from("http://[::]:0")?
        .connect_with_connector(service_fn(move |_: Uri| {
            let socket_path = socket_path.clone();
            async move {
                let stream = tokio::net::UnixStream::connect(socket_path).await?;
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
            }
        }))
        .await?;
    Ok(HealthClient::new(channel))
}
