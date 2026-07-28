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
use tonic::transport::Channel;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

use crate::device::Device;

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
    /// Compute device passed to the worker. Routed through [`Device`] so no
    /// call site here hardcodes a device string (EPIC 0.7); only
    /// `Device::as_str` knows the wire representation.
    pub device: Device,
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
            .arg(self.device.as_str());
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
///
/// The spec arrives over a `watch` channel rather than being fixed at
/// construction, which is what makes model hot-swap (EPIC 3.4) a restart
/// rather than a code path: publish a new spec and this loop takes the
/// current worker down and brings it back up on the new model id.
pub struct Supervisor {
    spec: watch::Receiver<WorkerSpec>,
    /// Held only when this supervisor owns a spec nobody else will change
    /// ([`Supervisor::fixed`]). Keeping the sender alive keeps the channel
    /// open; a closed channel would make `changed()` resolve immediately
    /// and spin the run loop.
    _spec_owner: Option<watch::Sender<WorkerSpec>>,
    health: HealthView,
    shutdown: watch::Receiver<bool>,
}

impl Supervisor {
    /// Creates a supervisor whose worker follows `spec`, sharing `health`
    /// with other supervised workers, and stopping (rather than restarting)
    /// once `shutdown` is set to `true`.
    ///
    /// Publishing a new value on `spec` restarts the worker on it.
    pub fn new(
        spec: watch::Receiver<WorkerSpec>,
        health: HealthView,
        shutdown: watch::Receiver<bool>,
    ) -> Self {
        Self {
            spec,
            _spec_owner: None,
            health,
            shutdown,
        }
    }

    /// Creates a supervisor for a worker whose spec never changes.
    ///
    /// For callers with no hot-swap story (the daemon's stub worker); the
    /// returned supervisor owns the sending half, so the spec is fixed for
    /// the process's life.
    pub fn fixed(spec: WorkerSpec, health: HealthView, shutdown: watch::Receiver<bool>) -> Self {
        let (tx, rx) = watch::channel(spec);
        Self {
            spec: rx,
            _spec_owner: Some(tx),
            health,
            shutdown,
        }
    }

    /// Runs the supervise loop until shutdown is signaled. Intended to be
    /// spawned as its own task per worker.
    pub async fn run(mut self) {
        let mut backoff = INITIAL_BACKOFF;
        let mut first_launch = true;
        // Set once every spec sender is gone: the worker can no longer be
        // reconfigured, but must still be supervised and still stop on
        // shutdown.
        let mut spec_closed = false;

        loop {
            if *self.shutdown.borrow() {
                self.set_state(WorkerState::Stopped).await;
                return;
            }

            // Snapshot the spec for this launch. Re-read every iteration so
            // a spec published while the previous worker was running takes
            // effect on the relaunch.
            let spec = self.spec.borrow_and_update().clone();
            let name = spec.name.clone();

            self.set_state(WorkerState::Starting).await;
            tracing::info!(worker = %name, model_id = %spec.model_id, "spawning worker");

            let mut child = match spec.command().spawn() {
                Ok(child) => child,
                Err(err) => {
                    tracing::error!(worker = %name, %err, "failed to spawn worker");
                    if self.wait_backoff_or_shutdown(&mut backoff).await {
                        self.set_state(WorkerState::Stopped).await;
                        return;
                    }
                    continue;
                }
            };

            if self.wait_healthy(&spec).await {
                if first_launch {
                    tracing::info!(worker = %name, model_id = %spec.model_id, "worker up");
                } else {
                    tracing::info!(worker = %name, model_id = %spec.model_id, "worker restarted");
                }
                first_launch = false;
                self.set_state(WorkerState::Up).await;
                backoff = INITIAL_BACKOFF;
            } else {
                tracing::warn!(worker = %name, "worker never became healthy");
            }

            // A spec change is a deliberate restart, so it skips the
            // crash backoff below: the user is waiting on the new model,
            // and nothing is failing.
            let mut respawn_immediately = false;
            // Watch the child until it exits, shutdown fires, or a new spec
            // arrives. This is a loop because one wake-up — the spec channel
            // closing — means "keep watching this same child": the worker
            // can no longer be reconfigured, but it still has to be reaped
            // on exit and stopped on shutdown.
            loop {
                let spec_just_closed;

                tokio::select! {
                    status = child.wait() => {
                        match status {
                            Ok(status) => tracing::warn!(worker = %name, %status, "worker exited"),
                            Err(err) => tracing::error!(worker = %name, %err, "failed to wait on worker"),
                        }
                        spec_just_closed = false;
                    }
                    _ = self.shutdown.changed() => {
                        if *self.shutdown.borrow() {
                            tracing::info!(worker = %name, "shutting down worker");
                            let _ = child.kill().await;
                            self.set_state(WorkerState::Stopped).await;
                            return;
                        }
                        spec_just_closed = false;
                    }
                    closed = next_spec_change(&mut self.spec, spec_closed) => {
                        if closed {
                            tracing::debug!(worker = %name, "spec channel closed");
                            spec_closed = true;
                            spec_just_closed = true;
                        } else {
                            let next = self.spec.borrow_and_update().clone();
                            tracing::info!(
                                worker = %name,
                                from_model = %spec.model_id,
                                to_model = %next.model_id,
                                "spec changed, restarting worker"
                            );
                            self.set_state(WorkerState::Restarting).await;
                            // SIGKILL via `kill` is blunt, but the worker
                            // holds no state worth draining and the swap
                            // caller has already waited for in-flight work.
                            let _ = child.kill().await;
                            respawn_immediately = true;
                            spec_just_closed = false;
                        }
                    }
                }

                if !spec_just_closed {
                    break;
                }
            }

            if *self.shutdown.borrow() {
                self.set_state(WorkerState::Stopped).await;
                return;
            }

            if respawn_immediately {
                backoff = INITIAL_BACKOFF;
                continue;
            }

            tracing::info!(worker = %name, backoff_ms = backoff.as_millis() as u64, "worker restarting");
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
        let name = self.spec.borrow().name.clone();
        self.health.write().await.insert(name, state);
    }

    /// Polls the worker's standard gRPC health-check RPC (over its UDS)
    /// until it reports `SERVING` or [`HEALTH_POLL_TIMEOUT`] elapses.
    async fn wait_healthy(&self, spec: &WorkerSpec) -> bool {
        let deadline = tokio::time::Instant::now() + HEALTH_POLL_TIMEOUT;
        while tokio::time::Instant::now() < deadline {
            if let Ok(mut client) = connect_health_client(&spec.socket_path).await {
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

/// Waits for the next spec change, returning `true` when the channel closed.
///
/// A closed channel resolves immediately and forever, which would spin the
/// supervise loop; the caller latches `closed` and passes it back here so
/// this arm goes quiet instead, leaving the child-exit and shutdown arms to
/// do their jobs.
async fn next_spec_change(spec: &mut watch::Receiver<WorkerSpec>, closed: bool) -> bool {
    if closed {
        std::future::pending::<()>().await;
    }
    spec.changed().await.is_err()
}

/// Connects a `HealthClient` to a worker over its unix domain socket.
async fn connect_health_client(
    socket_path: &Path,
) -> Result<HealthClient<Channel>, tonic::transport::Error> {
    Ok(HealthClient::new(crate::ipc::connect_uds(socket_path).await?))
}
