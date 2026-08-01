//! Thin control surface binary (`marceline`) for the daemon.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use marceline_core::{Device, HealthView, Supervisor, WorkerSpec};
use tokio::sync::{watch, RwLock};

mod converse;
mod lifecycle;
mod memory;
mod say;
mod say_to_llm;
mod transcribe;

/// Default config file, relative to the working directory. The full
/// XDG-aware search path arrives with the config CLI (EPIC 11.2).
const DEFAULT_CONFIG: &str = "config.toml";

/// Config keys `marceline config set` accepts today.
///
/// An allowlist rather than "anything the file contains": a typo silently
/// adding a key the daemon ignores is worse than being told no.
const SETTABLE_KEYS: &[&str] = &["stt.model", "stt.backend"];

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
        Some("say") => runtime.block_on(run_say(&args)),
        Some("say-to-llm") => runtime.block_on(run_say_to_llm(&args)),
        Some("converse") => runtime.block_on(run_converse(&args)),
        Some("config") => run_config(&args),
        Some("memory") => runtime.block_on(memory::run_memory(&args)),
        Some(cmd @ ("start" | "stop" | "status")) => {
            runtime.block_on(lifecycle::run_lifecycle(cmd, &args))
        }
        // Hidden: the actual daemon body `start` spawns detached via
        // `setsid`. Not part of the documented command surface — an
        // operator drives the daemon through `start`/`stop`/`status`.
        Some("__daemon-run") => runtime.block_on(lifecycle::run_daemon_process(&args)),
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
  marceline                            Run the epic-0 stub worker (see `start` for the real daemon)
  marceline start                      Start the daemon in the background
  marceline stop                       Gracefully stop the running daemon
  marceline status                     Show per-stage health and state of a running daemon
  marceline transcribe <file.wav>      Transcribe a wav file and print the text
  marceline say <text>                 Speak text aloud, per [tts] config
  marceline say-to-llm <text>          Stream an LLM reply to text, per [llm] config
  marceline converse                   Run the full wake->listen->think->speak MVP loop
  marceline config get <key>           Print a config value
  marceline config set <key> <value>   Change a config value
  marceline memory list                List stored long-term memories
  marceline memory search <query>      Find memories similar to a query
  marceline memory edit <id> <text>    Replace a memory's text and re-embed it
  marceline memory forget <id>         Delete a memory by row id
  marceline --version                  Print the version

Settable keys: {keys}

Options:
  --config <path>     Config file (default {DEFAULT_CONFIG})
  --soul <path>       SOUL.md path for say-to-llm (default {soul_default})
  --wav <path>        Wav output path for say (default {wav_default})
  --socket <path>     Attach to an already-running STT worker instead of
                      launching one from config (e.g. /tmp/marceline-stt.sock)
  --model-dir <path>  Embedding model directory for memory edit/search
  --k <n>             Result count for memory search (default 5)
  --verbose           Debug-level logging",
        keys = SETTABLE_KEYS.join(", "),
        soul_default = say_to_llm::DEFAULT_SOUL,
        wav_default = say::DEFAULT_WAV,
    );
}

/// Runs `marceline transcribe <file.wav>` (EPIC 3.3, 3.4, 11.4).
///
/// Launches the STT worker described by `[stt]` config, unless `--socket`
/// points at one already running. Exits non-zero on any failure, so a
/// wedged worker shows up as a visible error rather than as silence.
async fn run_transcribe(args: &[String]) -> ExitCode {
    let Some(path) = args.get(2).filter(|arg| !arg.starts_with('-')) else {
        eprintln!("transcribe requires a wav file path");
        print_usage();
        return ExitCode::FAILURE;
    };

    // `--socket` wins when given: it is the explicit "use this worker"
    // request, and silently launching a second one would be surprising.
    let source = match flag_value(args, "--socket") {
        Some(socket) => transcribe::WorkerSource::Socket(PathBuf::from(socket)),
        None => transcribe::WorkerSource::Config(PathBuf::from(
            flag_value(args, "--config").unwrap_or_else(|| DEFAULT_CONFIG.to_string()),
        )),
    };

    match transcribe::transcribe_file(Path::new(path), source).await {
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

/// Runs `marceline say <text>` (EPIC 5, demoable).
///
/// Launches the TTS worker described by `[tts]` config, speaks `text`
/// through it, and writes what it spoke to `--wav` — the epic's demoable:
/// swap `[tts].backend` from `kokoro` to `piper` and rerun the same text.
async fn run_say(args: &[String]) -> ExitCode {
    let Some(text) = args.get(2).filter(|arg| !arg.starts_with('-')) else {
        eprintln!("say requires text to speak");
        print_usage();
        return ExitCode::FAILURE;
    };

    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| DEFAULT_CONFIG.to_string()));
    let wav_path = say::wav_path_from_args(args);

    match say::say(&config_path, &wav_path, text).await {
        Ok(()) => {
            println!("wrote {}", wav_path.display());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("say failed: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Runs `marceline say-to-llm <text>` (EPIC 4.1, 4.2).
///
/// Streams the reply to stdout as it arrives — the whole point of 4.1's
/// streaming contract is that a caller does not have to wait for the full
/// response before showing anything.
async fn run_say_to_llm(args: &[String]) -> ExitCode {
    let Some(text) = args.get(2).filter(|arg| !arg.starts_with('-')) else {
        eprintln!("say-to-llm requires text to send");
        print_usage();
        return ExitCode::FAILURE;
    };

    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| DEFAULT_CONFIG.to_string()));
    let soul_path = say_to_llm::soul_path_from_args(args);

    match say_to_llm::say_to_llm(&config_path, &soul_path, text).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(say_to_llm::SayToLlmError::Engine(err)) if err.is_guardrail_refused() => {
            // The graceful spoken message this refusal maps to on the
            // ERROR edge (§2.5, §9.11) once an orchestrator exists to
            // speak it; on the command line the equivalent is a plain,
            // non-panicked message rather than a raw error dump.
            println!("I can't do that right now — {err}");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("say-to-llm failed: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Runs `marceline converse` (EPIC 8.2, the MVP loop demo).
///
/// Runs forever: wake, listen, transcribe, think, speak, back to idle.
/// Exits non-zero only on a setup failure (a device or worker that never
/// came up) — a mid-turn stage failure is handled in-loop via the
/// orchestrator's `ERROR` edge and does not end the process.
async fn run_converse(args: &[String]) -> ExitCode {
    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| DEFAULT_CONFIG.to_string()));
    let soul_path = converse::soul_path_from_args(args);

    match converse::converse(&config_path, &soul_path).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("converse failed: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Runs `marceline config get|set` (EPIC 3.4; full config CLI is 11.2).
///
/// The `set` path is the CLI trigger for a model swap: it changes
/// `[stt].model` in the file, and the next worker launch picks it up. A
/// running daemon does not yet see the edit — that needs a control channel
/// into the daemon, which is EPIC 11.2's job; this is called out in the
/// output rather than left to surprise someone.
fn run_config(args: &[String]) -> ExitCode {
    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| DEFAULT_CONFIG.to_string()));

    match args.get(2).map(String::as_str) {
        Some("get") => {
            let Some(key) = args.get(3) else {
                eprintln!("config get requires a key");
                return ExitCode::FAILURE;
            };
            match marceline_core::config_edit::get_string(&config_path, key) {
                Ok(Some(value)) => {
                    println!("{value}");
                    ExitCode::SUCCESS
                }
                Ok(None) => {
                    eprintln!("{key} is not set in {}", config_path.display());
                    ExitCode::FAILURE
                }
                Err(err) => {
                    eprintln!("config get failed: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("set") => {
            let (Some(key), Some(value)) = (args.get(3), args.get(4)) else {
                eprintln!("config set requires a key and a value");
                print_usage();
                return ExitCode::FAILURE;
            };
            if !SETTABLE_KEYS.contains(&key.as_str()) {
                eprintln!(
                    "{key} is not settable; supported keys: {}",
                    SETTABLE_KEYS.join(", ")
                );
                return ExitCode::FAILURE;
            }

            match marceline_core::config_edit::set_string(&config_path, key, value) {
                Ok(previous) => {
                    let previous = previous.unwrap_or_else(|| "(unset)".to_string());
                    println!("{key}: {previous} -> {value}");
                    if key.starts_with("stt.") {
                        eprintln!(
                            "the stt worker will load this on its next launch; \
                             a running daemon keeps its current model until restarted"
                        );
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("config set failed: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("config takes `get <key>` or `set <key> <value>`");
            print_usage();
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

    let supervisor_task = tokio::spawn(Supervisor::fixed(spec, health, shutdown_rx).run());

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
