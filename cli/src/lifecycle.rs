//! `marceline start/stop/status` (EPIC 11.1) — the daemon lifecycle
//! commands. This is the primary entry point to the whole system: an
//! operator brings Marceline up, takes it down cleanly, and checks whether
//! it is healthy, all without log-diving.
//!
//! `start` launches a detached child process running the same
//! `converse_ex` loop `marceline converse` runs interactively, except with
//! a control socket attached ([`marceline_core::daemon::serve_status`]) so
//! `status` can ask it questions later. `stop` sends the child SIGTERM,
//! which the daemon's own handler turns into the graceful shutdown
//! ordering (SPEC.md §2.5.1, see `converse::converse_ex`'s doc comment for
//! the exact sequence) — this module's job is only to wait for the process
//! to actually exit afterward and hard-kill it if it doesn't within a
//! bound. `status` is a thin client of the control socket; per §11's "thin
//! clients... over local IPC" constraint, none of these three ever reaches
//! into a worker process directly.
//!
//! Detaching the child via `setsid` (rather than a plain `Command::spawn`)
//! keeps it running after the launching terminal session closes — a plain
//! spawned child stays in the same session and can still receive a SIGHUP
//! when that session ends.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::Duration;

use marceline_core::daemon::{self, ControlRequest, ControlResponse};
use marceline_core::Config;

use crate::converse;

/// How long `start` waits for the freshly spawned daemon to answer its
/// first `status` query before giving up (the process is still running
/// either way — this only bounds how long the CLI waits to confirm it).
const START_READY_TIMEOUT: Duration = Duration::from_secs(10);
/// How long `stop` waits for the daemon to exit after SIGTERM before
/// hard-killing it (SPEC.md §2.5.1 step 6).
const STOP_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
/// Interval between liveness/readiness polls in `start`/`stop`.
const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Runs `marceline start`, `stop`, or `status` (dispatch from `main.rs`).
pub async fn run_lifecycle(command: &str, args: &[String]) -> ExitCode {
    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| "config.toml".to_string()));
    let soul_path = converse::soul_path_from_args(args);

    let config = match Config::load(&config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("failed to load config: {err}");
            return ExitCode::FAILURE;
        }
    };
    let dir = daemon::runtime_dir(&config.memory.expanded_db_path());
    let pidfile = daemon::pidfile_path(&dir);
    let control_socket = daemon::control_socket_path(&dir);

    match command {
        "start" => run_start(&config_path, &soul_path, &dir, &pidfile, &control_socket).await,
        "stop" => run_stop(&pidfile, &control_socket).await,
        "status" => run_status(&pidfile, &control_socket).await,
        other => {
            eprintln!("unknown lifecycle command: {other}");
            ExitCode::FAILURE
        }
    }
}

/// Runs the daemon itself in the foreground — the hidden `__daemon-run`
/// subcommand `start` spawns detached via `setsid`. Never invoked
/// directly by an operator.
///
/// Writes its own pidfile using its own `std::process::id()` rather than
/// leaving `run_start` to guess it from `Command::spawn`'s return value:
/// `setsid` may or may not fork depending on whether the launching process
/// already leads its process group, so the pid `Command::spawn` reports
/// for the `setsid` invocation is not reliably the actual daemon pid. The
/// process that will actually receive `stop`'s SIGTERM is the only
/// reliable source of its own pid.
pub async fn run_daemon_process(args: &[String]) -> ExitCode {
    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| "config.toml".to_string()));
    let soul_path = converse::soul_path_from_args(args);

    let config = match Config::load(&config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("failed to load config: {err}");
            return ExitCode::FAILURE;
        }
    };
    let dir = daemon::runtime_dir(&config.memory.expanded_db_path());
    let control_socket = daemon::control_socket_path(&dir);
    let pidfile = daemon::pidfile_path(&dir);
    if let Err(err) = daemon::write_pidfile(&pidfile, std::process::id()) {
        eprintln!("failed to write pidfile at {}: {err}", pidfile.display());
        return ExitCode::FAILURE;
    }

    let result = converse::converse_ex(&config_path, &soul_path, Some(&control_socket)).await;
    daemon::remove_pidfile(&pidfile);
    let _ = std::fs::remove_file(&control_socket);

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("daemon exited with an error: {err}");
            ExitCode::FAILURE
        }
    }
}

/// `marceline start`: boots the daemon, or reports it's already running.
async fn run_start(
    config_path: &Path,
    soul_path: &Path,
    dir: &Path,
    pidfile: &Path,
    control_socket: &Path,
) -> ExitCode {
    if let Some(pid) = daemon::read_pidfile(pidfile) {
        if is_alive(pid) {
            println!("marceline is already running (pid {pid})");
            return ExitCode::SUCCESS;
        }
        // Stale pidfile from an unclean previous exit — clear it before
        // starting fresh so nothing downstream mistakes it for a live pid.
        daemon::remove_pidfile(pidfile);
    }

    if let Err(err) = std::fs::create_dir_all(dir) {
        eprintln!("failed to create runtime directory {}: {err}", dir.display());
        return ExitCode::FAILURE;
    }

    let current_exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("failed to resolve the current executable: {err}");
            return ExitCode::FAILURE;
        }
    };

    let log_path = dir.join("daemon.log");
    let log_file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        Ok(file) => file,
        Err(err) => {
            eprintln!("failed to open daemon log at {}: {err}", log_path.display());
            return ExitCode::FAILURE;
        }
    };
    let log_file_err = match log_file.try_clone() {
        Ok(file) => file,
        Err(err) => {
            eprintln!("failed to duplicate daemon log handle: {err}");
            return ExitCode::FAILURE;
        }
    };

    let child = Command::new("setsid")
        .arg(&current_exe)
        .arg("__daemon-run")
        .arg("--config")
        .arg(config_path)
        .arg("--soul")
        .arg(soul_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn();

    if let Err(err) = child {
        eprintln!("failed to start the daemon: {err}");
        return ExitCode::FAILURE;
    }
    // The daemon process itself writes `pidfile` with its own real pid
    // (see `run_daemon_process`'s doc comment for why this parent can't
    // reliably read it off `Command::spawn` instead) — poll for that and
    // for the control socket to come up before reporting success.
    let deadline = std::time::Instant::now() + START_READY_TIMEOUT;
    loop {
        if let Some(pid) = daemon::read_pidfile(pidfile) {
            if daemon::send_request(control_socket, &ControlRequest::Status)
                .await
                .is_ok()
            {
                println!("marceline started (pid {pid})");
                return ExitCode::SUCCESS;
            }
        }
        if std::time::Instant::now() >= deadline {
            println!(
                "marceline may still be starting up; it did not answer a status query within \
                 {}s — check {}",
                START_READY_TIMEOUT.as_secs(),
                log_path.display()
            );
            return ExitCode::SUCCESS;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// `marceline stop`: sends SIGTERM and waits for the daemon to exit,
/// following the shutdown ordering documented on `converse::converse_ex`.
async fn run_stop(pidfile: &Path, control_socket: &Path) -> ExitCode {
    let Some(pid) = daemon::read_pidfile(pidfile) else {
        println!("marceline is not running");
        return ExitCode::SUCCESS;
    };
    if !is_alive(pid) {
        println!("marceline is not running (stale pidfile removed)");
        daemon::remove_pidfile(pidfile);
        return ExitCode::SUCCESS;
    }

    if !send_signal(pid, "-TERM") {
        eprintln!("failed to signal pid {pid}");
        return ExitCode::FAILURE;
    }

    let deadline = std::time::Instant::now() + STOP_WAIT_TIMEOUT;
    while is_alive(pid) {
        if std::time::Instant::now() >= deadline {
            eprintln!("pid {pid} did not exit within {}s; sending SIGKILL", STOP_WAIT_TIMEOUT.as_secs());
            send_signal(pid, "-KILL");
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }

    daemon::remove_pidfile(pidfile);
    let _ = std::fs::remove_file(control_socket);
    println!("marceline stopped");
    ExitCode::SUCCESS
}

/// `marceline status`: reports per-stage health and the current
/// conversation state against a live daemon.
async fn run_status(pidfile: &Path, control_socket: &Path) -> ExitCode {
    let Some(pid) = daemon::read_pidfile(pidfile) else {
        println!("marceline is not running");
        return ExitCode::FAILURE;
    };
    if !is_alive(pid) {
        println!("marceline is not running (stale pidfile present)");
        return ExitCode::FAILURE;
    }

    match daemon::send_request(control_socket, &ControlRequest::Status).await {
        Ok(ControlResponse::Status(report)) => {
            println!("marceline is running (pid {pid})");
            println!("  state: {:?}", report.state);
            if report.workers.is_empty() {
                println!("  no supervised workers reported");
            }
            for (name, health) in &report.workers {
                println!("  {name}: {health:?}");
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("marceline process is running (pid {pid}) but did not answer status: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Whether a process with `pid` currently exists, via `kill -0` — no
/// dependency on `libc`/`nix` for one syscall's worth of behavior.
fn is_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Sends `signal` (e.g. `"-TERM"`, `"-KILL"`) to `pid` via the `kill`
/// command. Returns whether the signal was delivered.
fn send_signal(pid: u32, signal: &str) -> bool {
    Command::new("kill")
        .arg(signal)
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Reads the value following `flag` in `args`, if present.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.get(index + 1).cloned()
}
