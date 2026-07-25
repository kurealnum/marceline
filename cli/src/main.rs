//! Thin control surface binary (`marceline`) for the daemon.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use marceline_core::{Device, HealthView, Supervisor, WorkerSpec};
use tokio::sync::{watch, RwLock};

mod transcribe;

/// Default socket the STT worker binds, matching `workers/stt/README.md`.
const DEFAULT_STT_SOCKET: &str = "/tmp/marceline-stt.sock";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose");

    if args.get(1).map(String::as_str) == Some("--version") {
        println!("marceline {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    marceline_core::logging::init(verbose);

    let runtime = tokio::runtime::Runtime::new().expect("failed to start tokio runtime");

    match args.get(1).map(String::as_str) {
        Some("transcribe") => runtime.block_on(run_transcribe(&args)),
        Some("--help") | Some("-h") => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some(unknown) if !unknown.starts_with('-') => {
            eprintln!("unknown command: {unknown}");
            print_usage();
            ExitCode::FAILURE
        }
        // No subcommand: run the daemon, as before.
        _ => {
            runtime.block_on(run(verbose));
            ExitCode::SUCCESS
        }
    }
}

/// Prints the command surface. Kept hand-rolled to match the rest of this
/// binary; a real arg parser lands with the wider CLI epic (11).
fn print_usage() {
    eprintln!(
        "\
Usage:
  marceline                          Run the daemon
  marceline transcribe <file.wav>    Transcribe a wav file and print the text
  marceline --version                Print the version

Options:
  --socket <path>   STT worker socket (default {DEFAULT_STT_SOCKET})
  --verbose         Debug-level logging"
    );
}

/// Runs `marceline transcribe <file.wav>` (EPIC 3.3, 11.4).
///
/// Requires an already-running STT worker; launching one from `[stt]`
/// config is story 3.4. Exits non-zero on any failure, so a wedged worker
/// shows up as a visible error rather than as silence.
async fn run_transcribe(args: &[String]) -> ExitCode {
    let Some(path) = args.get(2).filter(|arg| !arg.starts_with('-')) else {
        eprintln!("transcribe requires a wav file path");
        print_usage();
        return ExitCode::FAILURE;
    };

    let socket = flag_value(args, "--socket").unwrap_or_else(|| DEFAULT_STT_SOCKET.to_string());

    match transcribe::transcribe_file(Path::new(path), Path::new(&socket)).await {
        Ok(transcription) => {
            // The transcript goes to stdout so it can be piped; everything
            // else about the run goes to stderr via tracing.
            println!("{}", transcription.text);
            tracing::info!(
                confidence = transcription.confidence,
                segments = transcription.segments,
                "transcription complete"
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("transcription failed: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Reads the value following `flag` in `args`, if present.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.get(index + 1).cloned()
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
        device: env_or("WORKER_DEVICE", "cpu")
            .parse::<Device>()
            .expect("WORKER_DEVICE must be a supported device"),
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
