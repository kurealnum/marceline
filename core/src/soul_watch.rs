//! Watches `SOUL.md` for changes and keeps a live, atomically-swapped
//! [`Persona`] so the daemon can hot-reload the persona without a restart
//! (SPEC.md §3.2, EPIC 9.2).
//!
//! Reload is read-only and polling-based: the watcher only ever reads
//! `SOUL.md`'s mtime and contents, never writes it, which is what lets
//! hot-reload coexist with the background summarizer (§9.15) — they touch
//! different files and can never race each other.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use tokio::time::{interval, Duration};
use tokio_util::sync::CancellationToken;

use crate::soul::Persona;

/// Holds the most recently, successfully loaded [`Persona`] for a watched
/// `SOUL.md` path, updated in the background by the task [`watch`] spawns.
pub struct SoulWatcher {
    current: Arc<RwLock<Persona>>,
}

impl SoulWatcher {
    /// Returns a clone of the persona as of the most recent successful
    /// reload (or the initial load at [`watch`] time if none has occurred).
    pub fn current(&self) -> Persona {
        self.current
            .read()
            .expect("soul watcher lock poisoned")
            .clone()
    }
}

/// Loads `path` once synchronously, then spawns a background task that
/// polls `path`'s mtime every `poll_interval` and reloads [`Persona`]
/// whenever it changes, until `cancel` fires.
///
/// A missing/unreadable file at start degrades to [`Persona::default`],
/// matching the "fresh install without SOUL.md yet is not a failure"
/// precedent from 9.1's callers. A read/parse failure on a later poll
/// leaves the last-good persona in place and logs a warning — a bad edit
/// must never crash the daemon (9.2's "done when").
pub fn watch(
    path: PathBuf,
    poll_interval: Duration,
    cancel: CancellationToken,
) -> (Arc<SoulWatcher>, tokio::task::JoinHandle<()>) {
    let initial = Persona::load(&path).unwrap_or_else(|err| {
        tracing::warn!(
            "failed to load SOUL.md at {}: {err}; starting from an empty persona",
            path.display()
        );
        Persona::default()
    });
    let current = Arc::new(RwLock::new(initial));
    let watcher = Arc::new(SoulWatcher {
        current: current.clone(),
    });

    let mut last_mtime = mtime(&path);
    let handle = tokio::spawn(async move {
        let mut ticker = interval(poll_interval);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = ticker.tick() => {
                    let mtime_now = mtime(&path);
                    if mtime_now == last_mtime {
                        continue;
                    }
                    last_mtime = mtime_now;
                    match Persona::load(&path) {
                        Ok(persona) => {
                            *current.write().expect("soul watcher lock poisoned") = persona;
                            tracing::info!("reloaded SOUL.md from {}", path.display());
                        }
                        Err(err) => {
                            tracing::warn!(
                                "failed to reload SOUL.md at {}: {err}; keeping previous persona",
                                path.display()
                            );
                        }
                    }
                }
            }
        }
    });

    (watcher, handle)
}

/// Returns `path`'s last-modified time, or `None` if it can't be read
/// (missing file, permissions) — treated as "changed" once it starts or
/// stops resolving, same as any other mtime transition.
fn mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn hot_reloads_persona_on_file_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("SOUL.md");
        fs::write(&path, "# Identity\n\nOriginal.\n").unwrap();

        let cancel = CancellationToken::new();
        let (watcher, handle) = watch(path.clone(), Duration::from_millis(10), cancel.clone());
        assert_eq!(watcher.current().identity.as_deref(), Some("Original."));

        tokio::time::sleep(Duration::from_millis(15)).await;
        fs::write(&path, "# Identity\n\nUpdated.\n").unwrap();

        let mut reloaded = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if watcher.current().identity.as_deref() == Some("Updated.") {
                reloaded = true;
                break;
            }
        }
        assert!(reloaded, "persona was not hot-reloaded within the timeout");

        cancel.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn keeps_last_good_persona_when_file_becomes_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("SOUL.md");
        fs::write(&path, "# Identity\n\nOriginal.\n").unwrap();

        let cancel = CancellationToken::new();
        let (watcher, handle) = watch(path.clone(), Duration::from_millis(10), cancel.clone());
        assert_eq!(watcher.current().identity.as_deref(), Some("Original."));

        tokio::time::sleep(Duration::from_millis(15)).await;
        fs::remove_file(&path).unwrap();

        // Give the watcher several polls to notice the removal; it must
        // never clear the last-good persona just because the read failed.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(watcher.current().identity.as_deref(), Some("Original."));

        cancel.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn missing_file_at_start_yields_default_persona() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.md");

        let cancel = CancellationToken::new();
        let (watcher, handle) = watch(path, Duration::from_millis(10), cancel.clone());
        assert_eq!(watcher.current(), Persona::default());

        cancel.cancel();
        handle.await.unwrap();
    }
}
