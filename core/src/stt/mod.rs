//! The speech-to-text plugin contract (SPEC.md §2.4, §2.4.1, EPIC 3.2).
//!
//! [`SttEngine`] is the seam that makes STT models hot-swappable from
//! config: audio frames in, transcripts out, capabilities advertised.
//! Swapping HF `whisper` for `faster-whisper` (EPIC 3.5) is a different
//! `[stt].backend` value, not a rewrite — which only holds if the stream
//! contract below stays honest about what backends actually do.
//!
//! The one place that honesty matters most is partials. Live partial
//! transcripts are a *backend capability, not a guarantee*: the default
//! HF `whisper` is chunk-based and effectively final-only, so
//! [`SttInfo::partials`] advertises whether a backend emits real ones.
//! **v1 ships final-only**, and consumers must not assume otherwise.

pub mod grpc;
pub mod guard;
pub mod manager;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::engine::{AudioStream, EngineError};

pub use grpc::GrpcSttEngine;
pub use guard::{GuardConfig, Rejection, SpeechGuard};
pub use manager::{SttManager, SttWorkerPaths, SwapError};

/// One transcript item from an STT backend.
///
/// Provisional and committed text are distinguishable *by type*, which is
/// the whole point: modeling STT output as a plain `String` stream loses
/// the distinction and leaks half-words like `"helo"` into the LLM prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum Transcript {
    /// Revisable text (`"helo"` later corrected to `"hello"`). For UI,
    /// debug, and endpointing tuning (§9.3) only — **never** forwarded to
    /// the LLM. Only backends advertising [`SttInfo::partials`] emit these.
    Partial(String),
    /// Committed text. The only variant that reaches the LLM.
    Final {
        /// Recognized text.
        text: String,
        /// Backend-reported confidence in `[0, 1]`.
        confidence: f32,
        /// Signals the hallucination guard gates on (EPIC 3.6).
        signals: SpeechSignals,
    },
}

/// Backend-reported evidence that a segment really contained speech.
///
/// Both fields are `Option` because a backend that cannot measure a signal
/// must stay distinguishable from one reporting a confident value: `0.0`
/// no-speech probability and `0.0` average log-prob are each the most
/// confident value in their range, so treating "unknown" as either would
/// silently disarm the guard (EPIC 3.6).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpeechSignals {
    /// Probability the model assigned to this segment holding no speech.
    /// High values behind plausible text are the signature of Whisper
    /// inventing words on silence.
    pub no_speech_prob: Option<f32>,
    /// Mean per-token log probability (`<= 0`; nearer 0 is more confident).
    pub avg_logprob: Option<f32>,
}

impl Transcript {
    /// Returns the committed text, or `None` for a [`Transcript::Partial`].
    ///
    /// The ergonomic way to honor "only `Final` goes to the LLM" at a call
    /// site: filter on this rather than matching and hoping the `Partial`
    /// arm was handled correctly.
    pub fn final_text(&self) -> Option<&str> {
        match self {
            Transcript::Final { text, .. } => Some(text),
            Transcript::Partial(_) => None,
        }
    }
}

/// Streamed transcripts, errors propagating in-band (invariant 1).
pub type TranscriptStream = Pin<Box<dyn Stream<Item = Result<Transcript, EngineError>> + Send>>;

/// Capabilities of a loaded STT backend (SPEC.md §2.4).
///
/// Reported by the backend based on what it actually loaded, not on what
/// config asked for — a worker that fell back to a different model must
/// not be able to claim otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttInfo {
    /// Backend-qualified model name, e.g. `"whisper:openai/whisper-large-v3"`.
    pub name: String,
    /// Language codes this backend is configured to recognize. v1 is
    /// English-only, so this holds exactly the configured `[stt].lang`.
    pub langs: Vec<String>,
    /// Sample rate (Hz) the backend wants its audio in. Informational:
    /// the backend resamples what it is given rather than making callers
    /// pre-match this.
    pub input_sample_rate: u32,
    /// Whether this backend emits real [`Transcript::Partial`] items.
    /// False for chunk-based Whisper, and false throughout v1.
    pub partials: bool,
}

/// A speech-to-text backend (SPEC.md §2.4).
///
/// Implementors are usually "talk to a Python worker over gRPC"
/// ([`GrpcSttEngine`]); the trait exists so the orchestrator never learns
/// which.
#[async_trait]
pub trait SttEngine: Send + Sync {
    /// Streams `audio` to the backend and returns its transcript stream.
    ///
    /// Returns a stream rather than a `Result<Stream>` on purpose: a
    /// failure to even start is delivered as the stream's first `Err`
    /// item, so callers have exactly one error path instead of two
    /// (invariant 1, §2.4.1).
    async fn transcribe(&self, audio: AudioStream) -> TranscriptStream;

    /// Reports what this backend can do. Synchronous and cheap —
    /// implementors resolve capabilities when they connect, so call sites
    /// can check `partials` without awaiting.
    fn info(&self) -> SttInfo;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_text_exposes_only_committed_transcripts() {
        let committed = Transcript::Final {
            text: "what time is it".to_string(),
            confidence: 0.9,
            signals: SpeechSignals::default(),
        };
        assert_eq!(committed.final_text(), Some("what time is it"));
        assert_eq!(Transcript::Partial("what tim".to_string()).final_text(), None);
    }
}
