//! `marceline setup` (EPIC 12.2): first-run scaffolding — write starter
//! `config.toml`/`SOUL.md` if missing, check disk space, and warm the
//! default STT/TTS models into their local cache so the first real
//! conversation turn isn't also the first (large, slow) model download.
//!
//! **Model download is not reimplemented here.** `workers/stt` loads its
//! model via `transformers.AutoModelForSpeechSeq2Seq.from_pretrained`
//! (Hugging Face Hub) and `workers/tts`'s Kokoro backend loads via
//! `KPipeline`, which does the same under the hood — both already fetch
//! and cache weights on first load, with their own progress bar (from
//! `huggingface_hub`'s downloader), entirely independent of this CLI.
//! [`crate::converse`]'s workers already inherit this process's
//! stdout/stderr (`core::supervisor::WorkerSpec::command` sets no
//! `Stdio` override, so `tokio::process::Command`'s default — inherit —
//! applies), so that progress bar is visible here for free. What this
//! module adds on top, which nothing else provides:
//!
//! - **Config/persona scaffolding.** Nothing else writes `config.toml`/
//!   `SOUL.md` — every other command assumes they already exist.
//! - **A disk-space check *before* triggering a multi-GB download**, so a
//!   short-on-space machine gets a clear refusal instead of a half-filled
//!   Hugging Face cache and a confusing downstream error.
//! - **Triggering the download proactively**, by launching each worker
//!   long enough for it to finish loading (which is the same as finishing
//!   its download) and then shutting it down — so `marceline start`
//!   afterward has both models already warm, not mid-download on someone's
//!   first spoken turn.
//!
//! Idempotent throughout: an existing `config.toml`/`SOUL.md` is never
//! overwritten, and a model already cached loads instantly (Hugging Face
//! Hub's own cache does that check, not this code) rather than
//! re-downloading.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use marceline_core::stt::SttWorkerPaths;
use marceline_core::tts::TtsWorkerPaths;
use marceline_core::{Config, HealthView, SttManager};
use tokio::sync::{watch, RwLock};
use tokio_util::sync::CancellationToken;

use crate::converse;

/// The starter `config.toml` this repo ships and tests against (SPEC.md
/// §3.1) — embedded at compile time so a packaged install (EPIC 12.1) has
/// it without needing the source tree around at runtime.
const DEFAULT_CONFIG_TOML: &str = include_str!("../../config.toml");

/// The starter `SOUL.md` template (SPEC.md §3.2's six suggested
/// sections, matching the exact heading text `core::soul::Section`
/// recognizes) — deliberately generic scaffolding, not a persona: SOUL.md
/// is user-authored (§3.2, "never written by the system" *after* this
/// first scaffold), so this is a fill-in-the-blanks starting point, not a
/// default identity the system should keep silently rewriting.
const DEFAULT_SOUL_MD: &str = r#"# Identity

Marceline. Friendly, direct, a little dry. Speaks in short sentences by
default; expands only when the question calls for it.

# Voice

voice: af_sky

Terse for quick facts and commands; more expansive when walking through
something step by step.

# Values / rules

- Never take an irreversible or destructive action without confirming first.
- Say "I don't know" rather than guessing.

# Knowledge about me (the user)

<!-- Standing facts worth Marceline always having in context: your name,
     timezone, current projects, preferences. Edit this section directly. -->

# Tools policy

<!-- Which tools are allowed to run automatically, which need confirmation,
     which are off entirely. Left blank uses each tool's own default
     safety class. -->

# Examples

<!-- Optional few-shot exchanges demonstrating the tone/behavior above. -->
"#;

/// Rough, deliberately conservative estimate of disk space the default
/// models need: Whisper `large-v3` (~3 GB in `safetensors`) + Kokoro
/// (~350 MB) plus headroom for the Hugging Face Hub cache's own bookkeeping
/// and partial-download temp files. Not a precise byte count — Hugging
/// Face Hub itself is the source of truth for exact sizes — just a sanity
/// floor so a machine that's clearly too short on space is told before it
/// starts, not partway through (this story's "refuse gracefully" bar).
const MIN_FREE_DISK_MB: u64 = 6_000;

/// Runs `marceline setup` (EPIC 12.2). Also called automatically by
/// `marceline start` (`lifecycle::run_start`) when `--config` doesn't
/// exist yet, so a genuinely first-ever `marceline start` on a fresh
/// install does the right thing without a separate manual step.
pub async fn run_setup(args: &[String]) -> ExitCode {
    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| "config.toml".to_string()));
    let soul_path = converse::soul_path_from_args(args);

    if let Err(err) = scaffold_config(&config_path) {
        eprintln!("failed to write starter config at {}: {err}", config_path.display());
        return ExitCode::FAILURE;
    }
    if let Err(err) = scaffold_soul(&soul_path) {
        eprintln!("failed to write starter SOUL.md at {}: {err}", soul_path.display());
        return ExitCode::FAILURE;
    }

    let config = match Config::load(&config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("failed to load {}: {err}", config_path.display());
            return ExitCode::FAILURE;
        }
    };

    let cache_dir = huggingface_cache_dir();
    match free_disk_mb(&cache_dir) {
        Ok(free_mb) if free_mb < MIN_FREE_DISK_MB => {
            eprintln!(
                "only {free_mb} MB free at {} (~{MIN_FREE_DISK_MB} MB estimated for the \
                 default STT+TTS models) — refusing to start a download that likely \
                 won't finish. Free up space and rerun `marceline setup`.",
                cache_dir.display()
            );
            return ExitCode::FAILURE;
        }
        Ok(free_mb) => {
            println!("disk check: {free_mb} MB free at {} — proceeding", cache_dir.display());
        }
        Err(err) => {
            eprintln!(
                "warning: could not check free disk space at {} ({err}); proceeding anyway",
                cache_dir.display()
            );
        }
    }

    println!("warming the STT model ({}/{})...", config.stt.backend, config.stt.model);
    if let Err(err) = warm_stt(&config).await {
        eprintln!(
            "warning: STT model warm-up failed ({err}); it will download on the next \
             `marceline start` instead"
        );
    }

    println!("warming the TTS model ({}/{})...", config.tts.backend, config.tts.voice);
    if let Err(err) = warm_tts(&config).await {
        eprintln!(
            "warning: TTS model warm-up failed ({err}); it will download on the next \
             `marceline start` instead"
        );
    }

    println!("setup complete.");
    ExitCode::SUCCESS
}

/// Writes [`DEFAULT_CONFIG_TOML`] to `path` unless a file is already
/// there — an existing config (however the user has since edited it) is
/// never overwritten.
fn scaffold_config(path: &Path) -> std::io::Result<()> {
    if path.exists() {
        println!("{} already exists, leaving it alone", path.display());
        return Ok(());
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, DEFAULT_CONFIG_TOML)?;
    println!("wrote starter config to {}", path.display());
    Ok(())
}

/// Writes [`DEFAULT_SOUL_MD`] to `path` unless a file is already there.
fn scaffold_soul(path: &Path) -> std::io::Result<()> {
    if path.exists() {
        println!("{} already exists, leaving it alone", path.display());
        return Ok(());
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, DEFAULT_SOUL_MD)?;
    println!("wrote starter SOUL.md to {}", path.display());
    Ok(())
}

/// Where `transformers`/`huggingface_hub` cache downloaded models —
/// `HF_HOME` if set (the library's own override, honored here so the
/// disk check looks at the same place the download will actually land),
/// otherwise its documented default of `~/.cache/huggingface`.
fn huggingface_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HF_HOME") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache/huggingface");
    }
    PathBuf::from(".cache/huggingface")
}

/// Free space, in MB, on the filesystem containing `dir` (which need not
/// exist yet — `df` reports on the nearest existing ancestor). Shells out
/// to `df -Pk` rather than pulling in a disk-space crate: one POSIX-format
/// `df` call is simpler than a new dependency for a single number.
fn free_disk_mb(dir: &Path) -> Result<u64, String> {
    let mut probe = dir.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(parent) => probe = parent.to_path_buf(),
            None => break,
        }
    }

    let output = std::process::Command::new("df")
        .arg("-Pk")
        .arg(&probe)
        .output()
        .map_err(|err| format!("failed to run `df`: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "`df` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let data_line = stdout
        .lines()
        .nth(1)
        .ok_or_else(|| "unexpected `df` output (no data line)".to_string())?;
    let available_kb: u64 = data_line
        .split_whitespace()
        .nth(3)
        .ok_or_else(|| "unexpected `df` output (missing available-space column)".to_string())?
        .parse()
        .map_err(|err| format!("could not parse `df`'s available-space column: {err}"))?;
    Ok(available_kb / 1024)
}

/// How long warm-up waits for each worker to report healthy — generous,
/// since "healthy" here means "finished downloading and loading a
/// multi-gigabyte model on whatever this machine's link speed is", not a
/// quick local health check.
///
/// This is also, today, the *only* backstop against a worker that can
/// never come up at all (e.g. a missing/misconfigured Python venv path):
/// `core::supervisor::Supervisor::run`'s own restart-attempt cap
/// (`MAX_LAUNCH_ATTEMPTS` in `stt::manager`) only counts restarts after a
/// successful spawn whose process later exits — a `spawn()` call that
/// fails outright (bad executable path) retries with backoff forever
/// without ever incrementing that counter. Fixing that fast-fail gap
/// belongs to the supervisor (EPIC 0.6), not to this one caller of it;
/// 10 minutes here just keeps `setup`'s own worst case bounded until it
/// does.
const WARMUP_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Launches the STT worker described by `config.stt` and waits for it to
/// report healthy (which — see module docs — means its model finished
/// downloading and loading), then shuts it down. Errors are the caller's
/// to treat as non-fatal: a failed warm-up just means the download still
/// happens later, on the next real `marceline start`.
async fn warm_stt(config: &Config) -> Result<(), String> {
    let paths = SttWorkerPaths::for_backend(&config.stt.backend);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let health: HealthView = Arc::new(RwLock::new(HashMap::new()));

    let manager = tokio::time::timeout(
        WARMUP_TIMEOUT,
        SttManager::start(&config.stt, paths, health, shutdown_rx, CancellationToken::new()),
    )
    .await
    .map_err(|_| "timed out waiting for the stt worker to become healthy".to_string())?
    .map_err(|err| err.to_string())?;
    drop(manager);

    let _ = shutdown_tx.send(true);
    Ok(())
}

/// Launches the TTS worker described by `config.tts` and waits for it to
/// report healthy, then shuts it down. See [`warm_stt`] for why failures
/// here are non-fatal to the overall `setup` run.
async fn warm_tts(config: &Config) -> Result<(), String> {
    let paths = TtsWorkerPaths::for_backend(&config.tts.backend);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let health: HealthView = Arc::new(RwLock::new(HashMap::new()));

    let engine = tokio::time::timeout(
        WARMUP_TIMEOUT,
        marceline_core::launch_tts_worker(&config.tts, paths, health, shutdown_rx, CancellationToken::new()),
    )
    .await
    .map_err(|_| "timed out waiting for the tts worker to become healthy".to_string())?
    .map_err(|err| err.to_string())?;
    drop(engine);

    let _ = shutdown_tx.send(true);
    Ok(())
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
    fn scaffolds_config_and_soul_only_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let soul_path = dir.path().join("SOUL.md");

        scaffold_config(&config_path).unwrap();
        scaffold_soul(&soul_path).unwrap();
        assert_eq!(std::fs::read_to_string(&config_path).unwrap(), DEFAULT_CONFIG_TOML);
        assert_eq!(std::fs::read_to_string(&soul_path).unwrap(), DEFAULT_SOUL_MD);

        // A second run must not clobber user edits.
        std::fs::write(&config_path, "version = 1\n# edited by hand\n").unwrap();
        scaffold_config(&config_path).unwrap();
        assert_eq!(
            std::fs::read_to_string(&config_path).unwrap(),
            "version = 1\n# edited by hand\n"
        );
    }

    #[test]
    fn scaffolding_creates_missing_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("nested/config.toml");
        scaffold_config(&config_path).unwrap();
        assert!(config_path.exists());
    }

    #[test]
    fn the_embedded_config_template_is_loadable_as_a_real_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, DEFAULT_CONFIG_TOML).unwrap();
        marceline_core::Config::load(&config_path).expect("starter config.toml must parse");
    }

    #[test]
    fn the_embedded_soul_template_parses_into_every_section() {
        let persona = marceline_core::Persona::parse(DEFAULT_SOUL_MD);
        assert!(persona.identity.is_some());
        assert!(persona.voice.is_some());
        assert!(persona.values_rules.is_some());
        assert!(persona.knowledge.is_some());
        assert!(persona.tools_policy.is_some());
    }

    #[test]
    fn free_disk_mb_reports_a_plausible_value_for_a_real_directory() {
        let dir = tempfile::tempdir().unwrap();
        let mb = free_disk_mb(dir.path()).expect("df should succeed against a real directory");
        assert!(mb > 0, "expected nonzero free space, got {mb}");
    }
}
