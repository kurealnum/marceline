//! Locating the `workers/` tree on disk (EPIC 12.1): the one seam between
//! "run from a dev checkout" and "run from a packaged install".
//!
//! [`crate::stt::SttWorkerPaths`] and [`crate::tts::TtsWorkerPaths`] both
//! need to find a specific worker's `.venv`/`worker.py` under some
//! `workers/` root; before this story, that root was hardcoded as
//! `PathBuf::from("workers")` — correct only when the daemon's current
//! working directory happens to be the repo root, which is true for
//! `cargo run`/`cargo test` but not for an installed binary launched from
//! anywhere else. [`workers_root`] is the shared resolution [`worker_spec`]
//! (in both `stt::manager` and `tts::manager`) now goes through instead.
//!
//! Resolution order, each overriding the next:
//! 1. `MARCELINE_WORKERS_DIR` — an explicit override, for anyone who wants
//!    to point at a workers tree that lives somewhere unusual.
//! 2. A `workers/` directory next to the running executable
//!    (`<prefix>/bin/marceline` + `<prefix>/workers/...`) — the packaged
//!    install layout `scripts/package/build.sh` produces (EPIC 12.1).
//! 3. `workers/` relative to the current working directory — the existing
//!    dev-checkout behavior, unchanged so `cargo run`/`cargo test` keep
//!    working exactly as before from the repo root.

use std::path::PathBuf;

/// Environment variable that overrides where the `workers/` tree is found.
pub const WORKERS_DIR_ENV_VAR: &str = "MARCELINE_WORKERS_DIR";

/// Resolves the `workers/` root directory per the order documented on the
/// module (env override, then next to the current executable, then the
/// working directory).
pub fn workers_root() -> PathBuf {
    if let Ok(dir) = std::env::var(WORKERS_DIR_ENV_VAR) {
        return PathBuf::from(dir);
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            // Packaged layout: `<prefix>/bin/marceline` next to
            // `<prefix>/workers/`, i.e. `workers/` is a *sibling* of the
            // directory the binary lives in, not inside it.
            if let Some(prefix) = bin_dir.parent() {
                let candidate = prefix.join("workers");
                if candidate.is_dir() {
                    return candidate;
                }
            }
        }
    }

    PathBuf::from("workers")
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test, not two: both cases mutate the same process-global env
    // var, and `cargo test` runs tests on multiple threads by default, so
    // two separate tests race on it (one's `set_var` can land between the
    // other's `remove_var` and its assertion). Sequencing both cases
    // inside a single test removes the race entirely.
    #[test]
    fn env_override_wins_and_falls_back_to_relative_workers_without_it() {
        // SAFETY: `remove_var`/`set_var` are called back-to-back within
        // this one test body, with no `.await` or thread handoff between
        // a mutation and the assertion that depends on it.
        unsafe {
            std::env::remove_var(WORKERS_DIR_ENV_VAR);
        }
        // `cargo test`'s executable lives under `target/.../deps/`, which
        // has no `workers/` sibling, so this exercises the final
        // dev-checkout fallback exactly as `cargo run`/`cargo test` from
        // the repo root do today.
        assert_eq!(workers_root(), PathBuf::from("workers"));

        unsafe {
            std::env::set_var(WORKERS_DIR_ENV_VAR, "/tmp/marceline-workers-test-override");
        }
        assert_eq!(
            workers_root(),
            PathBuf::from("/tmp/marceline-workers-test-override")
        );

        unsafe {
            std::env::remove_var(WORKERS_DIR_ENV_VAR);
        }
    }
}
