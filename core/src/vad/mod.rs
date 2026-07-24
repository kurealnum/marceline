//! Endpointing signal via Silero VAD (SPEC.md §2.6, §9.3, EPIC 2.2): tells
//! speech from silence, so the gate (EPIC 2.3) can tell when an utterance
//! ends. Bad endpointing feels worse than bad STT, so this is real
//! inference against the actual Silero VAD ONNX model (`models/
//! silero_vad.onnx`) — no placeholder needed here, unlike the wake word
//! (blocked on EPIC 13's model export).
//!
//! This story provides the raw per-frame speech/no-speech signal and the
//! primitives (speech-onset, silence duration) the gate needs. The
//! `silence_ms`/`min_utterance_ms`/`max_utterance_ms` *threshold knobs*
//! themselves are wired from `[vad]` config and tuned in EPIC 2.5 — this
//! module exposes a raw threshold parameter, not config plumbing.

pub mod model;

pub use model::{SileroVad, VadError, FRAME_SAMPLES};

/// Silero VAD's own commonly-recommended default speech probability
/// threshold. The tuned, config-driven value lands in EPIC 2.5.
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.5;

/// Milliseconds represented by one [`FRAME_SAMPLES`]-length frame at
/// [`model::SAMPLE_RATE`].
const FRAME_MS: u64 = (FRAME_SAMPLES as u64 * 1000) / model::SAMPLE_RATE as u64;

/// Tracks speech/silence state across consecutive VAD frames: whether
/// speech is ongoing, whether it just started this frame, and how long
/// it's been silent since the last speech frame. The gate (2.3) drives
/// its `silence_ms`-based end-of-utterance decision from
/// [`VadEndpointer::silence_ms`].
pub struct VadEndpointer {
    vad: SileroVad,
    threshold: f32,
    speaking: bool,
    speech_just_started: bool,
    silence_ms: u64,
}

impl VadEndpointer {
    /// Wraps a loaded [`SileroVad`] with a fire threshold (probability
    /// above which a frame counts as speech).
    pub fn new(vad: SileroVad, threshold: f32) -> Self {
        Self {
            vad,
            threshold,
            speaking: false,
            speech_just_started: false,
            silence_ms: 0,
        }
    }

    /// Resets to a fresh IDLE-like state: no speech, zero silence
    /// duration, and clears the model's recurrent state. Call this
    /// between utterances so a prior one's state/history doesn't bias the
    /// next.
    pub fn reset(&mut self) {
        self.vad.reset();
        self.speaking = false;
        self.speech_just_started = false;
        self.silence_ms = 0;
    }

    /// Feeds one [`FRAME_SAMPLES`]-length, 16kHz mono frame. Returns the
    /// raw speech probability from the model; updates the speaking/
    /// silence-duration state queryable via the other methods.
    pub fn process_frame(&mut self, frame: &[f32]) -> Result<f32, VadError> {
        let prob = self.vad.infer(frame)?;
        let is_speech = prob >= self.threshold;

        self.speech_just_started = is_speech && !self.speaking;
        if is_speech {
            self.speaking = true;
            self.silence_ms = 0;
        } else {
            self.speaking = false;
            self.silence_ms += FRAME_MS;
        }

        Ok(prob)
    }

    /// Whether the most recently processed frame was speech.
    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    /// Whether speech onset happened on the most recently processed frame
    /// (i.e. the prior frame was silence and this one wasn't).
    pub fn speech_just_started(&self) -> bool {
        self.speech_just_started
    }

    /// Milliseconds of continuous silence since the last speech frame (or
    /// since [`reset`](Self::reset)/construction if speech hasn't
    /// happened yet).
    pub fn silence_ms(&self) -> u64 {
        self.silence_ms
    }
}
