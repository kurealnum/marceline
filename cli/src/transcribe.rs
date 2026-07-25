//! The `marceline transcribe <file.wav>` path (EPIC 3.3, 3.4, 11.4).
//!
//! A wav file stands in for a gate-emitted segment, which makes the whole
//! audio→text path runnable without a microphone or a wake word — and makes
//! the epic's demo checkable: change `[stt].model`, rerun the same file, and
//! it still transcribes, on the new model.
//!
//! Two ways to get a worker:
//!
//! * **From config** (default) — read `[stt]`, launch the worker via the
//!   supervisor, transcribe, shut it down. This is the path where changing
//!   config changes which model runs.
//! * **`--socket`** — attach to a worker someone else is running. Faster to
//!   iterate against, and the only option if the worker lives elsewhere.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use marceline_core::stt::SttWorkerPaths;
use marceline_core::transcribe::{TranscribeOutcome, Transcription, DEFAULT_TIMEOUT};
use marceline_core::{read_wav, Config, HealthView, SttManager};
use tokio::sync::{watch, RwLock};
use tokio_util::sync::CancellationToken;

/// Anything that can go wrong transcribing a file from the command line.
#[derive(Debug, thiserror::Error)]
pub enum TranscribeFileError {
    /// The wav file could not be read.
    #[error(transparent)]
    Wav(#[from] marceline_core::WavReadError),
    /// The config file could not be loaded.
    #[error(transparent)]
    Config(#[from] marceline_core::ConfigError),
    /// The STT worker was unreachable, failed, or timed out.
    #[error(transparent)]
    Engine(#[from] marceline_core::EngineError),
    /// The transcript was rejected before reaching the LLM (EPIC 3.6).
    ///
    /// Reported as a failure rather than as empty output because it is the
    /// empty-transcript ERROR edge (§2.5): in the daemon this speaks a
    /// graceful message and returns to IDLE, and on the command line the
    /// equivalent is a message and a non-zero exit — never a blank line that
    /// looks like success.
    #[error("no usable speech in {path}: {reason}")]
    Rejected {
        /// File that produced no usable transcript.
        path: String,
        /// Which check rejected it, and the measurement behind it.
        reason: String,
    },
}

/// How to reach an STT worker.
pub enum WorkerSource {
    /// Launch one from the `[stt]` block of the config at this path.
    Config(PathBuf),
    /// Attach to a worker already listening on this socket.
    Socket(PathBuf),
}

/// Reads `path`, transcribes it, and returns the committed text.
///
/// Empty output is an error rather than an empty success: a caller asked for
/// a transcript, and printing a blank line while exiting 0 would look like
/// the file was silent when it might mean the worker misbehaved.
pub async fn transcribe_file(
    path: &Path,
    source: WorkerSource,
) -> Result<Transcription, TranscribeFileError> {
    let segment = read_wav(path)?;
    tracing::info!(
        file = %path.display(),
        samples = segment.pcm.len(),
        sample_rate = segment.sample_rate,
        channels = segment.channels,
        "read audio segment"
    );

    // A one-shot CLI run has nothing to barge in on, but the token is what
    // ctrl-c will fire once the orchestrator owns this path (§2.5.1), so it
    // is threaded through from the start rather than retrofitted.
    let cancel = CancellationToken::new();
    // Held for the whole run: dropping the sender would tell the supervisor
    // to stop the worker we are about to use.
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let manager = match source {
        WorkerSource::Socket(socket) => {
            tracing::info!(socket = %socket.display(), "attaching to a running stt worker");
            SttManager::attach(socket, cancel, "en".to_string()).await?
        }
        WorkerSource::Config(config_path) => {
            let config = Config::load(&config_path)?;
            tracing::info!(
                config = %config_path.display(),
                backend = %config.stt.backend,
                model = %config.stt.model,
                "launching stt worker from config"
            );
            let paths = SttWorkerPaths::for_backend(&config.stt.backend);
            let health: HealthView = Arc::new(RwLock::new(HashMap::new()));
            SttManager::start(&config.stt, paths, health, shutdown_rx, cancel).await?
        }
    };

    let info = manager.info().await;
    tracing::info!(
        model = %info.name,
        partials = info.partials,
        "using stt backend"
    );

    let result = manager.transcribe(segment, DEFAULT_TIMEOUT).await;

    // Stop a worker we launched before reporting, so the process does not
    // outlive the command that started it.
    let _ = shutdown_tx.send(true);

    match result? {
        TranscribeOutcome::Committed(transcription) => Ok(transcription),
        TranscribeOutcome::Rejected(rejection) => Err(TranscribeFileError::Rejected {
            path: path.display().to_string(),
            reason: rejection.reason(),
        }),
    }
}
