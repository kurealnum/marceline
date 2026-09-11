//! Resolves runtime file locations (worker scripts, model files) so the
//! daemon works from an installed binary, not only from a checkout
//! (issue #184).
//!
//! Every lookup here tries, in order:
//!
//! 1. An explicit override (an env var, for now — a config key/flag is a
//!    caller concern once one exists for a given resource).
//! 2. A path relative to [`std::env::current_exe`], for a self-contained
//!    install (the binary and its `workers/`/`models/` siblings copied
//!    together).
//! 3. `$XDG_DATA_HOME/marceline` (or `~/.local/share/marceline`), for a
//!    packaged install that puts data files in the standard user location.
//! 4. A system location (`/usr/local/share/marceline`, `/usr/share/marceline`),
//!    for a distro package.
//!
//! `CARGO_MANIFEST_DIR` never appears here — only in `#[cfg(test)]` code,
//! which is a real checkout by construction.

use std::path::{Path, PathBuf};

/// A resource (a worker directory, a model file) could not be found at
/// any candidate location.
///
/// `tried`, in the error message, is what turns "it doesn't work" into
/// "it's missing from here, here, and here" — the missing-model case is
/// the most likely first-run failure, so this is the whole difference
/// between a fixable error and a dead end.
#[derive(Debug, thiserror::Error)]
#[error("{name} not found; tried:\n{}", tried_list(.tried))]
pub struct MissingResourceError {
    /// Human-readable name of what was being resolved, e.g. `"workers root"`.
    pub name: String,
    /// Every candidate path tried, in the order they were tried.
    pub tried: Vec<PathBuf>,
}

fn tried_list(tried: &[PathBuf]) -> String {
    tried
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns the first candidate in `candidates` that exists on disk, or a
/// [`MissingResourceError`] naming all of them.
fn resolve_existing(name: &str, candidates: Vec<PathBuf>) -> Result<PathBuf, MissingResourceError> {
    match candidates.iter().find(|p| p.exists()) {
        Some(found) => Ok(found.clone()),
        None => Err(MissingResourceError {
            name: name.to_string(),
            tried: candidates,
        }),
    }
}

/// The directory [`std::env::current_exe`] lives in, if it can be resolved.
fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(Path::to_path_buf)
}

/// `$XDG_DATA_HOME/marceline`, falling back to `~/.local/share/marceline`
/// per the XDG base directory spec.
fn xdg_data_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir).join("marceline"));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".local/share/marceline"))
}

/// System-wide install locations, tried last.
fn system_data_dirs() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/usr/local/share/marceline"),
        PathBuf::from("/usr/share/marceline"),
    ]
}

/// Candidate roots for a data subdirectory (`workers`, `models`), in
/// priority order: an explicit override env var (pointing at the
/// resource directly, not at a data root to join `relative` onto), then
/// `relative` joined onto each fallback data root.
fn resource_candidates(override_env: &str, relative: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::var_os(override_env).filter(|v| !v.is_empty()) {
        candidates.push(PathBuf::from(dir));
    }
    let mut data_roots = Vec::new();
    data_roots.extend(exe_dir());
    data_roots.extend(xdg_data_dir());
    data_roots.extend(system_data_dirs());
    candidates.extend(data_roots.into_iter().map(|dir| dir.join(relative)));
    candidates
}

/// Resolves the `workers/` root directory (containing e.g. `workers/stt/`,
/// `workers/tts/`), so [`crate::stt::SttWorkerPaths::for_backend`] and
/// [`crate::tts::TtsWorkerPaths::for_backend`] don't hardcode a
/// checkout-relative `"workers"` path.
///
/// `MARCELINE_WORKERS_DIR` overrides everything else, for development —
/// pointing directly at the workers root, not at a data directory to
/// derive one from.
pub fn workers_root() -> Result<PathBuf, MissingResourceError> {
    resolve_existing("workers root", resource_candidates("MARCELINE_WORKERS_DIR", "workers"))
}

/// Resolves the directory `models/` files (the Silero VAD model, the
/// embedding model) live in.
///
/// `MARCELINE_MODELS_DIR` overrides everything else, for development.
pub fn models_root() -> Result<PathBuf, MissingResourceError> {
    resolve_existing("models root", resource_candidates("MARCELINE_MODELS_DIR", "models"))
}

/// Resolves the Silero VAD model file (`models/silero_vad.onnx`).
pub fn vad_model_path() -> Result<PathBuf, MissingResourceError> {
    let candidates = resource_candidates("MARCELINE_MODELS_DIR", "models")
        .into_iter()
        .map(|dir| dir.join("silero_vad.onnx"))
        .collect();
    resolve_existing("Silero VAD model (models/silero_vad.onnx)", candidates)
}

/// Resolves the embedding model directory (`models/all-MiniLM-L6-v2/`).
pub fn embed_model_dir() -> Result<PathBuf, MissingResourceError> {
    let candidates = resource_candidates("MARCELINE_MODELS_DIR", "models")
        .into_iter()
        .map(|dir| dir.join("all-MiniLM-L6-v2"))
        .collect();
    resolve_existing("embedding model (models/all-MiniLM-L6-v2)", candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_override_wins_over_every_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let workers = dir.path().join("custom-workers");
        std::fs::create_dir_all(&workers).unwrap();

        let previous = std::env::var_os("MARCELINE_WORKERS_DIR");
        std::env::set_var("MARCELINE_WORKERS_DIR", &workers);

        let resolved = workers_root().unwrap();
        assert_eq!(resolved, workers);

        match previous {
            Some(value) => std::env::set_var("MARCELINE_WORKERS_DIR", value),
            None => std::env::remove_var("MARCELINE_WORKERS_DIR"),
        }
    }

    #[test]
    fn a_missing_resource_names_every_path_it_tried() {
        let previous = std::env::var_os("MARCELINE_WORKERS_DIR");
        std::env::set_var("MARCELINE_WORKERS_DIR", "/definitely/not/a/real/path");

        let err = workers_root().unwrap_err();
        assert!(err.tried.contains(&PathBuf::from("/definitely/not/a/real/path")));
        let message = err.to_string();
        assert!(message.contains("/definitely/not/a/real/path"));

        match previous {
            Some(value) => std::env::set_var("MARCELINE_WORKERS_DIR", value),
            None => std::env::remove_var("MARCELINE_WORKERS_DIR"),
        }
    }
}
