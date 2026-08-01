//! Integration test for the supervisor's crash auto-restart (SPEC.md §2.2,
//! §0.6, EPIC 12.5): a Python worker that dies is relaunched without
//! anything above the supervisor noticing beyond a `Restarting` health
//! blip. Runs the real `workers/template` worker as a real subprocess and
//! kills it with `SIGKILL` (an uncatchable crash, not a graceful exit) to
//! prove this for real rather than asserting it about mocked state.
//!
//! Requires `workers/template/.venv` to exist (`workers/template/setup.sh`
//! — pure-Python `grpcio`/`grpcio-health-checking` deps, no GPU/model
//! weights); skips itself with a clear message if it's missing rather than
//! failing a checkout that hasn't run that setup step.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use marceline_core::{Device, HealthView, Supervisor, WorkerSpec, WorkerState};
use tokio::sync::{watch, RwLock};

fn template_python() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../workers/template/.venv/bin/python")
}

fn unique_socket_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "marceline-supervisor-test-{}-{name}.sock",
        std::process::id()
    ))
}

/// Polls `health` until `name` reports `state`, or panics after `timeout`.
async fn wait_for_state(health: &HealthView, name: &str, state: WorkerState, timeout: Duration) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if health.read().await.get(name).copied() == Some(state) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for worker {name} to reach {state:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_killed_worker_is_restarted_and_becomes_healthy_again() {
    let python = template_python();
    if !python.exists() {
        eprintln!(
            "skipping: {} not found — run workers/template/setup.sh first",
            python.display()
        );
        return;
    }

    let socket_path = unique_socket_path("crash-restart");
    let _ = std::fs::remove_file(&socket_path);

    let spec = WorkerSpec {
        name: "template".to_string(),
        python,
        script: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../workers/template/worker.py"),
        socket_path: socket_path.clone(),
        model_id: "template".to_string(),
        device: Device::Cpu,
    };

    let health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let supervisor_task =
        tokio::spawn(Supervisor::fixed(spec, Arc::clone(&health), shutdown_rx).run());

    wait_for_state(&health, "template", WorkerState::Up, Duration::from_secs(20)).await;

    // Find the real OS process bound to this test's unique socket path and
    // SIGKILL it — an uncatchable crash, not a graceful shutdown, so this
    // proves the supervisor notices an unexpected exit, not just a clean
    // one.
    let pgrep = std::process::Command::new("pgrep")
        .arg("-f")
        .arg(socket_path.to_str().expect("socket path is valid utf8"))
        .output()
        .expect("run pgrep");
    let pid = String::from_utf8_lossy(&pgrep.stdout)
        .lines()
        .next()
        .unwrap_or_else(|| panic!("no process found matching socket path {}", socket_path.display()))
        .to_string();
    let killed = std::process::Command::new("kill")
        .arg("-KILL")
        .arg(&pid)
        .status()
        .expect("run kill");
    assert!(killed.success(), "failed to signal pid {pid}");

    wait_for_state(&health, "template", WorkerState::Restarting, Duration::from_secs(10)).await;
    wait_for_state(&health, "template", WorkerState::Up, Duration::from_secs(20)).await;

    supervisor_task.abort();
    let _ = std::fs::remove_file(&socket_path);
}
