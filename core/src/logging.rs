//! Structured logging setup (SPEC.md §9.12): without per-stage tracing a
//! realtime, multi-process system is undebuggable. This is the base layer
//! the later `--follow`/live-logs work (EPIC 11.5) builds on.

use tracing_subscriber::EnvFilter;

/// Environment variable that overrides the default log filter, same
/// convention as `RUST_LOG`.
const LOG_ENV_VAR: &str = "MARCELINE_LOG";

/// Initializes the global structured logging subscriber.
///
/// Default level is `info`; pass `verbose = true` (e.g. from a `--verbose`
/// CLI flag) to raise it to `debug`. `MARCELINE_LOG`/`RUST_LOG` still takes
/// precedence when set, so operators can target specific modules.
///
/// Safe to call more than once per process; subsequent calls are no-ops.
pub fn init(verbose: bool) {
    let default_level = if verbose { "debug" } else { "info" };

    let filter = EnvFilter::try_from_env(LOG_ENV_VAR)
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}
