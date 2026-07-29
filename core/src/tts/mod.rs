//! The text-to-speech plugin contract (SPEC.md §2.4, §2.4.1, EPIC 5.2).
//!
//! [`TtsEngine`] is the seam that makes TTS models hot-swappable from
//! config: already-segmented text in, streamed PCM audio out, capabilities
//! advertised. Swapping Kokoro for Piper (EPIC 5.5) is a different
//! `[tts].backend` value, not a rewrite — which only holds if the stream
//! contract below stays honest about what backends actually do.
//!
//! Sentence-chunking streamed LLM tokens into spans is the *caller's* job
//! (§5.3, EPIC 5.3): this trait always receives already-segmented text, so
//! a Kokoro-vs-Piper difference in preferred chunk granularity never leaks
//! into the caller.

pub mod chunker;
pub mod grpc;
pub mod manager;
pub mod playback;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::engine::{AudioStream, EngineError};

pub use chunker::sentence_chunk;
pub use grpc::GrpcTtsEngine;
pub use manager::{launch, TtsWorkerPaths, WORKER_NAME};
pub use playback::{play, PlaybackSink};

/// Already-segmented text streamed into a [`TtsEngine`] (SPEC.md §2.4.1).
///
/// Errors propagate in-band (invariant 1): a failure upstream of TTS (the
/// LLM stream erroring mid-answer) is delivered as an `Err` item rather
/// than a stream that just stops, so a backend can half-close cleanly
/// instead of hanging on a caller that is never coming back.
pub type TextStream = Pin<Box<dyn Stream<Item = Result<String, EngineError>> + Send>>;

/// A backend-specific voice identifier, e.g. `"af_sky"` for Kokoro's fixed
/// voice set (SPEC.md §3.1's `[tts].voice`).
///
/// A thin wrapper rather than a bare `String` so a call site cannot
/// accidentally pass raw text where a voice id is expected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceId(pub String);

impl From<String> for VoiceId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for VoiceId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl std::fmt::Display for VoiceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Capabilities of a loaded TTS backend (SPEC.md §2.4).
///
/// Reported by the backend based on what it actually loaded, not on what
/// config asked for — a worker that fell back to a different voice set
/// must not be able to claim otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsInfo {
    /// Backend-qualified model name, e.g. `"kokoro:82M"`.
    pub name: String,
    /// Voice ids this backend can synthesize (Kokoro's fixed voice set).
    pub voices: Vec<String>,
    /// Sample rate (Hz) the backend actually emits. Declared here rather
    /// than assumed, so a consumer never guesses it — the chipmunk-voice
    /// failure mode §2.4.1 warns about.
    pub output_sample_rate: u32,
}

/// A text-to-speech backend (SPEC.md §2.4).
///
/// Implementors are usually "talk to a Python worker over gRPC"
/// ([`GrpcTtsEngine`]); the trait exists so the orchestrator never learns
/// which.
#[async_trait]
pub trait TtsEngine: Send + Sync {
    /// Streams `text` to the backend with `voice` selected, and returns its
    /// synthesized audio stream.
    ///
    /// Returns a stream rather than a `Result<Stream>` on purpose: a
    /// failure to even start is delivered as the stream's first `Err`
    /// item, so callers have exactly one error path instead of two
    /// (invariant 1, §2.4.1).
    async fn synthesize(&self, text: TextStream, voice: VoiceId) -> AudioStream;

    /// Reports what this backend can do. Synchronous and cheap —
    /// implementors resolve capabilities when they connect, so call sites
    /// can check `voices` without awaiting.
    fn info(&self) -> TtsInfo;
}

/// Resolves SOUL.md's Voice preference (§3.2, EPIC 9.4) against what the
/// connected backend actually advertises.
///
/// `preferred` is the voice id parsed out of SOUL.md, if any
/// (`crate::soul::Persona::voice_preference`). An unknown/unavailable id —
/// or no preference at all — falls back to `default` (`config.toml`'s
/// `[tts].voice`, §3.1) rather than failing the turn; per §10, Kokoro's
/// voice set is fixed, so a stale or typo'd SOUL preference is exactly the
/// kind of thing that must degrade gracefully, not crash a reply.
pub fn resolve_voice(preferred: Option<&str>, info: &TtsInfo, default: &VoiceId) -> VoiceId {
    match preferred {
        Some(id) if info.voices.iter().any(|v| v == id) => VoiceId::from(id),
        Some(id) => {
            tracing::warn!(
                requested = id,
                available = ?info.voices,
                "SOUL.md's requested TTS voice is not in the backend's voice set; falling back to the configured default"
            );
            default.clone()
        }
        None => default.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_id_converts_from_str_and_string() {
        assert_eq!(VoiceId::from("af_sky"), VoiceId("af_sky".to_string()));
        assert_eq!(
            VoiceId::from("af_sky".to_string()),
            VoiceId("af_sky".to_string())
        );
        assert_eq!(VoiceId::from("af_sky").to_string(), "af_sky");
    }

    fn info(voices: &[&str]) -> TtsInfo {
        TtsInfo {
            name: "test".to_string(),
            voices: voices.iter().map(|v| v.to_string()).collect(),
            output_sample_rate: 24_000,
        }
    }

    #[test]
    fn a_valid_preferred_voice_is_used() {
        let resolved = resolve_voice(
            Some("af_bella"),
            &info(&["af_sky", "af_bella"]),
            &VoiceId::from("af_sky"),
        );
        assert_eq!(resolved, VoiceId::from("af_bella"));
    }

    #[test]
    fn an_unavailable_preferred_voice_falls_back_to_the_default() {
        let resolved = resolve_voice(
            Some("nonexistent_voice"),
            &info(&["af_sky", "af_bella"]),
            &VoiceId::from("af_sky"),
        );
        assert_eq!(resolved, VoiceId::from("af_sky"));
    }

    #[test]
    fn no_preference_uses_the_default() {
        let resolved = resolve_voice(None, &info(&["af_sky", "af_bella"]), &VoiceId::from("af_sky"));
        assert_eq!(resolved, VoiceId::from("af_sky"));
    }
}
