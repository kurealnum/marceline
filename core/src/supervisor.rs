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
/// Consecutive failed launches tolerated before the supervisor gives up.
const MAX_CONSECUTIVE_LAUNCH_FAILURES: u32 = 20;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerState {
    /// Process launched, not yet confirmed healthy.
    Starting,
    /// Process is running and its health RPC reports `SERVING`.
    Up,
    /// Process was running but did not report healthy before the deadline.
    Unhealthy {
        /// The last reason the worker failed to become healthy.
        reason: String,
    },
    /// Process exited; a restart is pending (backoff).
    Restarting,
    /// The supervisor stopped retrying after repeated launch failures.
    Failed {
        /// The last failure that caused the supervisor to give up.
        reason: String,
    },
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
    health_poll_interval: Duration,
    health_poll_timeout: Duration,
    max_consecutive_launch_failures: u32,
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
        Self::with_options(
            spec,
            None,
            health,
            shutdown,
            HEALTH_POLL_INTERVAL,
            HEALTH_POLL_TIMEOUT,
            MAX_CONSECUTIVE_LAUNCH_FAILURES,
        )
    }

    /// Creates a supervisor for a worker whose spec never changes.
    ///
    /// For callers with no hot-swap story (the daemon's stub worker); the
    /// returned supervisor owns the sending half, so the spec is fixed for
    /// the process's life.
    pub fn fixed(spec: WorkerSpec, health: HealthView, shutdown: watch::Receiver<bool>) -> Self {
        let (tx, rx) = watch::channel(spec);
        Self::with_options(
            rx,
            Some(tx),
            health,
            shutdown,
            HEALTH_POLL_INTERVAL,
            HEALTH_POLL_TIMEOUT,
            MAX_CONSECUTIVE_LAUNCH_FAILURES,
        )
    }

    fn with_options(
        spec: watch::Receiver<WorkerSpec>,
        spec_owner: Option<watch::Sender<WorkerSpec>>,
        health: HealthView,
        shutdown: watch::Receiver<bool>,
        health_poll_interval: Duration,
        health_poll_timeout: Duration,
        max_consecutive_launch_failures: u32,
    ) -> Self {
        Self {
            spec,
            _spec_owner: spec_owner,
            health,
            shutdown,
            health_poll_interval,
            health_poll_timeout,
            max_consecutive_launch_failures: max_consecutive_launch_failures.max(1),
        }
    }

    #[cfg(test)]
    fn with_test_options(
        spec: watch::Receiver<WorkerSpec>,
        health: HealthView,
        shutdown: watch::Receiver<bool>,
        health_poll_interval: Duration,
        health_poll_timeout: Duration,
        max_consecutive_launch_failures: u32,
    ) -> Self {
        Self::with_options(
            spec,
            None,
            health,
            shutdown,
            health_poll_interval,
            health_poll_timeout,
            max_consecutive_launch_failures,
        )
    }

    /// Runs the supervise loop until shutdown is signaled. Intended to be
    /// spawned as its own task per worker.
    pub async fn run(mut self) {
        let mut backoff = INITIAL_BACKOFF;
        let mut first_launch = true;
        let mut consecutive_launch_failures = 0;
        // Set once every spec sender is gone: the worker can no longer be
        // reconfigured, but must still be supervised and still stop on
        // shutdown.
        let mut spec_closed = false;

        loop {
            if *self.shutdown.borrow() {
                let name = self.spec.borrow().name.clone();
                self.set_state(&name, WorkerState::Stopped).await;
                return;
            }

            // Snapshot the spec for this launch. Re-read every iteration so
            // a spec published while the previous worker was running takes
            // effect on the relaunch.
            let spec = self.spec.borrow_and_update().clone();
            let name = spec.name.clone();

            self.set_state(&name, WorkerState::Starting).await;
            tracing::info!(worker = %name, model_id = %spec.model_id, "spawning worker");

            let mut child = match spec.command().spawn() {
                Ok(child) => child,
                Err(err) => {
                    let reason = format!("failed to spawn worker: {err}");
                    tracing::warn!(worker = %name, reason = %reason, "worker launch failed");
                    if self
                        .record_launch_failure(
                            &name,
                            &mut consecutive_launch_failures,
                            reason,
                        )
                        .await
                    {
                        return;
                    }
                    if self.wait_backoff_or_shutdown(&mut backoff).await {
                        self.set_state(&name, WorkerState::Stopped).await;
                        return;
                    }
                    continue;
                }
            };

            match self.wait_healthy(&spec).await {
                Ok(()) => {
                    if first_launch {
                        tracing::info!(worker = %name, model_id = %spec.model_id, "worker up");
                    } else {
                        tracing::info!(worker = %name, model_id = %spec.model_id, "worker restarted");
                    }
                    first_launch = false;
                    self.set_state(&name, WorkerState::Up).await;
                    backoff = INITIAL_BACKOFF;
                    consecutive_launch_failures = 0;
                }
                Err(reason) => {
                    self.set_state(
                        &name,
                        WorkerState::Unhealthy {
                            reason: reason.clone(),
                        },
                    )
                    .await;
                    tracing::warn!(
                        worker = %name,
                        reason = %reason,
                        "worker never became healthy; killing worker"
                    );
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    if self
                        .record_launch_failure(
                            &name,
                            &mut consecutive_launch_failures,
                            reason,
                        )
                        .await
                    {
                        return;
                    }
                    if self.wait_backoff_or_shutdown(&mut backoff).await {
                        self.set_state(&name, WorkerState::Stopped).await;
                        return;
                    }
                    continue;
                }
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
                            self.set_state(&name, WorkerState::Stopped).await;
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
                            self.set_state(&name, WorkerState::Restarting).await;
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
                self.set_state(&name, WorkerState::Stopped).await;
                return;
            }

            if respawn_immediately {
                backoff = INITIAL_BACKOFF;
                continue;
            }

            tracing::info!(worker = %name, backoff_ms = backoff.as_millis() as u64, "worker restarting");
            self.set_state(&name, WorkerState::Restarting).await;
            if self.wait_backoff_or_shutdown(&mut backoff).await {
                self.set_state(&name, WorkerState::Stopped).await;
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

    async fn set_state(&self, name: &str, state: WorkerState) {
        self.health
            .write()
            .await
            .insert(name.to_string(), state);
    }

    async fn record_launch_failure(
        &self,
        name: &str,
        consecutive_failures: &mut u32,
        reason: String,
    ) -> bool {
        *consecutive_failures += 1;
        if *consecutive_failures >= self.max_consecutive_launch_failures {
            let reason = format!(
                "{reason} (after {consecutive_failures} consecutive launch failures)"
            );
            tracing::error!(
                worker = %name,
                attempts = *consecutive_failures,
                reason = %reason,
                "worker launch failed repeatedly; giving up"
            );
            self.set_state(name, WorkerState::Failed { reason }).await;
            true
        } else {
            self.set_state(name, WorkerState::Restarting).await;
            false
        }
    }

    /// Polls the worker's standard gRPC health-check RPC (over its UDS)
    /// until it reports `SERVING` or [`HEALTH_POLL_TIMEOUT`] elapses.
    async fn wait_healthy(&self, spec: &WorkerSpec) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + self.health_poll_timeout;
        while tokio::time::Instant::now() < deadline {
            if let Ok(mut client) = connect_health_client(&spec.socket_path).await {
                if let Ok(resp) = client
                    .check(HealthCheckRequest {
                        service: String::new(),
                    })
                    .await
                {
                    if resp.into_inner().status == ServingStatus::Serving as i32 {
                        return Ok(());
                    }
                }
            }
            sleep(self.health_poll_interval).await;
        }
        Err(format!(
            "worker did not report SERVING within {}ms",
            self.health_poll_timeout.as_millis()
        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use tokio::sync::{watch, RwLock};

    #[cfg(unix)]
    #[tokio::test]
    async fn unhealthy_worker_is_killed_and_terminal_failure_keeps_reason() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let pid_path = dir.path().join("worker.pid");
        let script_path = dir.path().join("stuck-worker.sh");
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\necho $$ > '{}'\nexec sleep 60\n",
                pid_path.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script_path, permissions).unwrap();

        let spec = WorkerSpec {
            name: "stuck".to_string(),
            python: PathBuf::from("/bin/sh"),
            script: script_path,
            socket_path: dir.path().join("worker.sock"),
            model_id: "test".to_string(),
            device: Device::Cpu,
        };
        let health: HealthView = Arc::new(RwLock::new(HashMap::new()));
        let (_spec_tx, spec_rx) = watch::channel(spec);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let supervisor = Supervisor::with_test_options(
            spec_rx,
            Arc::clone(&health),
            shutdown_rx,
            Duration::from_millis(1),
            Duration::from_millis(20),
            2,
        );
        let task = tokio::spawn(supervisor.run());

        let state = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(state) = health.read().await.get("stuck").cloned() {
                    if matches!(state, WorkerState::Failed { .. }) {
                        break state;
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("supervisor should reach its terminal failure state");
        task.await.unwrap();

        let WorkerState::Failed { reason } = state else {
            panic!("expected terminal failure, got {state:?}");
        };
        assert!(reason.contains("did not report SERVING"), "reason: {reason}");

        let pid = std::fs::read_to_string(pid_path)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        let alive = std::process::Command::new("/bin/sh")
            .args(["-c", "kill -0 \"$1\"", "kill-check", &pid.to_string()])
            .status()
            .unwrap()
            .success();
        assert!(!alive, "unhealthy worker process {pid} is still alive");
    }
}
