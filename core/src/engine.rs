//! Types shared by the three plugin contracts (SPEC.md §2.4, §2.4.1).
//!
//! The `SttEngine` / `LlmEngine` / `TtsEngine` traits each get their own
//! module and their own `*Info` type — the three stages report different
//! capabilities, and one shared `EngineInfo` would carry wrong or empty
//! fields per stage. What they *do* share lives here: the error type
//! every stream item carries, and the audio stream both mic-in and
//! TTS-out use.
//!
//! Invariant 1 of §2.4.1 is the reason [`EngineError`] exists at all:
//! every stream item is a `Result`, so a worker OOM at chunk 40 or an LLM
//! 500 mid-token propagates in-band, mid-stream, instead of silently
//! truncating a stream that looks like it ended normally.

use std::pin::Pin;

use futures::Stream;

use crate::audio::AudioChunk;

/// A failure surfaced by a plugin backend, in-band on its stream.
///
/// Deliberately coarse: callers act on the *kind* of failure (retry the
/// worker, abandon the turn, stay quiet because the user interrupted),
/// not on backend-specific detail, which belongs in the message and the
/// logs.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The backend could not be reached at all — worker not up yet, socket
    /// missing, connection refused. Distinct from [`EngineError::Worker`]
    /// because the supervisor (EPIC 0.6), not the turn, is what fixes it.
    #[error("failed to reach {backend} backend: {source}")]
    Transport {
        /// Backend that could not be reached, e.g. `"stt"`.
        backend: &'static str,
        /// Underlying transport error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The backend was reached but reported a failure, typically
    /// mid-stream (CUDA OOM during inference, a model in a bad state).
    #[error("{backend} backend failed: {message}")]
    Worker {
        /// Backend that failed, e.g. `"stt"`.
        backend: &'static str,
        /// Message as reported by the backend.
        message: String,
    },
    /// The backend sent something the contract does not allow. Surfaced
    /// rather than papered over: a worker emitting off-contract messages
    /// is a bug that silent tolerance would hide until it corrupted a
    /// transcript.
    #[error("{backend} backend violated the stream contract: {message}")]
    Protocol {
        /// Backend that misbehaved, e.g. `"stt"`.
        backend: &'static str,
        /// What was wrong with the message.
        message: String,
    },
    /// The backend accepted the work but produced nothing in time.
    ///
    /// Distinct from [`EngineError::Worker`] because a wedged worker is
    /// indistinguishable from a slow one at the protocol level, and the
    /// TRANSCRIBING error edge (§2.5) needs *something* to route on rather
    /// than waiting forever.
    #[error("{backend} backend timed out after {elapsed_ms}ms")]
    Timeout {
        /// Backend that timed out, e.g. `"stt"`.
        backend: &'static str,
        /// How long it was given, in milliseconds.
        elapsed_ms: u64,
    },
    /// The run's cancellation token fired (§2.5.1) — barge-in, ctrl-c, or
    /// a restart. Not a failure: it means "stop, and do not use whatever
    /// you had so far".
    #[error("{backend} stream cancelled")]
    Cancelled {
        /// Backend whose stream was cancelled, e.g. `"stt"`.
        backend: &'static str,
    },
}

impl EngineError {
    /// True when this error is a cooperative cancel rather than a fault.
    ///
    /// Callers use this to stay quiet instead of surfacing an error to the
    /// user: the user interrupting is not something to apologize for.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, EngineError::Cancelled { .. })
    }
}

/// Streamed PCM audio, shared by mic-in and TTS-out (SPEC.md §2.4.1).
///
/// Sample rate and channel count travel with each [`AudioChunk`] rather
/// than being agreed out of band, and `seq` lets a consumer detect
/// dropped or reordered chunks.
pub type AudioStream = Pin<Box<dyn Stream<Item = Result<AudioChunk, EngineError>> + Send>>;
