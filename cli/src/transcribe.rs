//! The `marceline transcribe <file.wav>` path (EPIC 3.3, 11.4).
//!
//! A wav file stands in for a gate-emitted segment, which makes the whole
//! audio→text path runnable without a microphone or a wake word — and makes
//! "change `[stt].model`, rerun the same file, still transcribes" (the
//! epic's demo) a thing you can actually check.

use std::path::Path;

use marceline_core::transcribe::{transcribe_segment, Transcription, DEFAULT_TIMEOUT};
use marceline_core::{read_wav, GrpcSttEngine, SttEngine};
use tokio_util::sync::CancellationToken;

/// Anything that can go wrong transcribing a file from the command line.
#[derive(Debug, thiserror::Error)]
pub enum TranscribeFileError {
    /// The wav file could not be read.
    #[error(transparent)]
    Wav(#[from] marceline_core::WavReadError),
    /// The STT worker was unreachable, failed, or timed out.
    #[error(transparent)]
    Engine(#[from] marceline_core::EngineError),
    /// The worker answered, but committed no text.
    #[error("no speech recognized in {path}")]
    NoSpeech {
        /// File that produced no transcript.
        path: String,
    },
}

/// Reads `path`, transcribes it via the worker at `socket`, and returns the
/// committed text.
///
/// Empty output is an error rather than an empty success: a caller asked
/// for a transcript, and printing a blank line while exiting 0 would look
/// like the file was silent when it might mean the worker misbehaved.
pub async fn transcribe_file(
    path: &Path,
    socket: &Path,
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
    let engine = GrpcSttEngine::connect(socket, CancellationToken::new()).await?;
    let info = engine.info();
    tracing::info!(
        model = %info.name,
        partials = info.partials,
        "using stt backend"
    );

    let transcription = transcribe_segment(&engine, segment, DEFAULT_TIMEOUT).await?;
    if transcription.is_empty() {
        return Err(TranscribeFileError::NoSpeech {
            path: path.display().to_string(),
        });
    }
    Ok(transcription)
}
