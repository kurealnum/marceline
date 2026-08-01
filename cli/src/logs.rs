//! `marceline logs [--follow]` (EPIC 11.5): streams the daemon's
//! structured log output (SPEC.md §9.12, EPIC 0.3) live from the CLI.
//!
//! The daemon (`lifecycle::run_start`) redirects its stdout/stderr —
//! everything `core::logging::init`'s `tracing_subscriber::fmt()` layer
//! writes — to `daemon.log` in the same runtime directory as its pidfile
//! and control socket (`marceline_core::daemon::runtime_dir`). This module
//! is a plain reader of that file: it never talks to the daemon process
//! itself, so it works whether or not the daemon happens to be running
//! right now (a crashed daemon's last lines are still worth reading).
//!
//! Every log line already carries its own level (`INFO`/`DEBUG`/…) and
//! target module (`with_target(true)`, e.g. `marceline_core::stt::manager`)
//! — that target *is* this scope's "which stage/worker a line came from"
//! context, so nothing here needs to re-derive it. What this module adds
//! on top is `--follow` (tail-f semantics: print what's there, then keep
//! reading as more is appended, until ctrl-c) and a client-side `--verbose`
//! filter independent of the daemon's own `--verbose`: the daemon may have
//! been started verbose for its own reasons, but an operator watching the
//! stream can still choose to see only `INFO`-and-above unless they ask
//! for more.

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use marceline_core::daemon;
use marceline_core::Config;

/// How often `--follow` polls the log file for newly appended bytes.
/// `tracing_subscriber`'s writer has no notify-on-write hook exposed here,
/// so polling a plain file is the straightforward option; this interval
/// is fast enough that a fresh line feels live without busy-looping.
const FOLLOW_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Runs `marceline logs [--follow] [--verbose]`.
pub async fn run_logs(args: &[String]) -> ExitCode {
    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| "config.toml".to_string()));
    let follow = args.iter().any(|a| a == "--follow");
    let verbose = args.iter().any(|a| a == "--verbose");

    let config = match Config::load(&config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("failed to load config: {err}");
            return ExitCode::FAILURE;
        }
    };
    let dir = daemon::runtime_dir(&config.memory.expanded_db_path());
    let log_path = dir.join("daemon.log");

    let mut file = match std::fs::File::open(&log_path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!(
                "no daemon log at {} ({err}); has `marceline start` been run?",
                log_path.display()
            );
            return ExitCode::FAILURE;
        }
    };

    if let Err(err) = print_existing(&mut file, verbose) {
        eprintln!("failed to read {}: {err}", log_path.display());
        return ExitCode::FAILURE;
    }

    if !follow {
        return ExitCode::SUCCESS;
    }

    tokio::select! {
        result = follow_file(file, verbose) => {
            if let Err(err) = result {
                eprintln!("failed to read {}: {err}", log_path.display());
                return ExitCode::FAILURE;
            }
        }
        _ = tokio::signal::ctrl_c() => {}
    }
    ExitCode::SUCCESS
}

/// Prints every line currently in `file` that passes [`show_line`],
/// leaving the file cursor at EOF so a subsequent [`follow_file`] only
/// sees what's appended after this point.
fn print_existing(file: &mut std::fs::File, verbose: bool) -> std::io::Result<()> {
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    for line in contents.lines() {
        if show_line(line, verbose) {
            println!("{line}");
        }
    }
    Ok(())
}

/// Polls `file` for newly appended lines forever (tail -f), printing each
/// one that passes [`show_line`]. Returns only on a read error — the
/// caller races this against ctrl-c.
async fn follow_file(file: std::fs::File, verbose: bool) -> std::io::Result<()> {
    let mut reader = BufReader::new(file);
    loop {
        tokio::time::sleep(FOLLOW_POLL_INTERVAL).await;
        let mut line = String::new();
        loop {
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                // Caught up to EOF; a truncated/rotated log (rare — this
                // daemon only ever appends) would otherwise wedge this
                // loop reading nothing forever, so re-seek to the file's
                // current end whenever it's shorter than where we are.
                let pos = reader.stream_position()?;
                let len = reader.get_ref().metadata()?.len();
                if len < pos {
                    reader.seek(SeekFrom::Start(len))?;
                }
                break;
            }
            if line.ends_with('\n') {
                if show_line(line.trim_end_matches(['\n', '\r']), verbose) {
                    println!("{}", line.trim_end_matches(['\n', '\r']));
                }
                line.clear();
            }
        }
    }
}

/// Whether `line` should be shown given the client-side `--verbose`
/// filter: everything is shown when `verbose`, otherwise `DEBUG`/`TRACE`
/// lines from `tracing_subscriber::fmt()`'s default format (the level
/// token is the second whitespace-separated field, right after the
/// timestamp) are hidden. A line that doesn't parse as that format (e.g.
/// a panic message split across lines) is shown either way — hiding
/// something unrecognized is worse than showing an occasional extra line.
fn show_line(line: &str, verbose: bool) -> bool {
    if verbose {
        return true;
    }
    !matches!(line.split_whitespace().nth(1), Some("DEBUG") | Some("TRACE"))
}

/// Reads the value following `flag` in `args`, if present.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.get(index + 1).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hides_debug_and_trace_lines_unless_verbose() {
        let debug_line = "2026-01-01T00:00:00.000000Z DEBUG marceline_core::gate: tick";
        let trace_line = "2026-01-01T00:00:00.000000Z TRACE marceline_core::gate: tick";
        let info_line = "2026-01-01T00:00:00.000000Z  INFO marceline_core::gate: wake";

        assert!(!show_line(debug_line, false));
        assert!(!show_line(trace_line, false));
        assert!(show_line(info_line, false));

        assert!(show_line(debug_line, true));
        assert!(show_line(trace_line, true));
        assert!(show_line(info_line, true));
    }

    #[test]
    fn an_unrecognized_line_shape_is_shown_either_way() {
        assert!(show_line("thread 'main' panicked at src/foo.rs:1", false));
        assert!(show_line("", false));
    }
}
