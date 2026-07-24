//! Thin control surface binary (`marceline`) for the daemon.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use marceline_core::{HealthView, Supervisor, WorkerSpec};
use tokio::sync::{watch, RwLock};

fn main() {
    let args: Vec<String> = env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose");

    if args.get(1).map(String::as_str) == Some("--version") {
        println!("marceline {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    marceline_core::logging::init(verbose);

    let runtime = tokio::runtime::Runtime::new().expect("failed to start tokio runtime");
    runtime.block_on(run(verbose));
}

/// Reads a worker spec field from an environment variable, falling back to
/// `default` when unset. Lets the epic-0 demo script point at the worker
/// template without hardcoding paths in the binary.
fn env_or(var: &str, default: &str) -> String {
    env::var(var).unwrap_or_else(|_| default.to_string())
}

/// Boots the daemon: starts the worker supervisor for a stub worker and
/// runs until an interrupt/terminate signal is received, at which point
/// the worker is signaled to exit and the process shuts down cleanly.
async fn run(verbose: bool) {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "core up");
    tracing::debug!(verbose, "verbose logging enabled by --verbose flag");

    let spec = WorkerSpec {
        name: env_or("WORKER_NAME", "stub"),
        python: PathBuf::from(env_or(
            "WORKER_PYTHON",
            "workers/template/.venv/bin/python",
        )),
        script: PathBuf::from(env_or("WORKER_SCRIPT", "workers/template/worker.py")),
        socket_path: PathBuf::from(env_or("WORKER_SOCKET", "/tmp/marceline-worker.sock")),
        model_id: env_or("WORKER_MODEL_ID", "template"),
        device: env_or("WORKER_DEVICE", "cpu"),
    };

    let health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let supervisor_task = tokio::spawn(Supervisor::new(spec, health, shutdown_rx).run());

    wait_for_shutdown_signal().await;

    tracing::info!("shutdown signal received, stopping worker");
    let _ = shutdown_tx.send(true);
    let _ = supervisor_task.await;
}

/// Waits for Ctrl-C or SIGTERM, whichever arrives first.
#[cfg(unix)]
async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm =
        signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = sigterm.recv() => {}
    }
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
