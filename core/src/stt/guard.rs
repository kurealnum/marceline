//! The silence/hallucination guard (SPEC.md §2.5, EPIC 3.6).
//!
//! Whisper invents plausible text on near-silence and non-speech — its most
//! famous failure mode. VAD endpointing (EPIC 2) reduces it but does not
//! eliminate it: a door closing or a cough can end up as a confident-looking
//! sentence. Left alone, Marceline answers a question nobody asked, out
//! loud, which is worse than saying nothing.
//!
//! So this guard sits exactly at the boundary §2.4.1 defines — the one place
//! a `Final` transcript becomes something the LLM sees — and a rejection
//! routes through the empty-transcript ERROR edge (§2.5): speak a graceful
//! message, return to IDLE. Never a silent hang, and never a hallucination
//! read back to the user.
//!
//! Three independent checks, because no single one is sufficient:
//!
//! 1. **Duration.** Audio too short to contain a word cannot have produced a
//!    real transcript, whatever the model says. Checked before inference, so
//!    the obvious case costs no GPU time.
//! 2. **No-speech probability.** The model's own verdict on whether it heard
//!    speech. The most direct signal, and unavailable from some backends —
//!    which is why the other two exist.
//! 3. **Average log probability.** Invented text tends to be low-confidence
//!    even when it reads fluently.
//!
//! Thresholds are tuning knobs with deliberately cautious defaults; real
//! values are an EPIC 8.3 concern, and they are config-adjustable so tuning
//! does not mean editing code.

use std::time::Duration;

use crate::audio::AudioChunk;
use crate::stt::SpeechSignals;

/// Default minimum speech duration.
///
/// 250ms is about the shortest a real spoken word gets. Anything below it is
/// a click, a breath, or a truncated segment.
pub const DEFAULT_MIN_SPEECH_MS: u32 = 250;

/// Default ceiling on the backend's no-speech probability.
///
/// 0.6 rather than 0.5: the model saying "probably silence" is worth acting
/// on, but a marginal call should not throw away a real utterance. Dropping
/// something the user actually said is the more annoying failure of the two.
pub const DEFAULT_MAX_NO_SPEECH_PROB: f32 = 0.6;

/// Default floor on average per-token log probability.
///
/// -1.0 is the threshold OpenAI's own reference implementation uses for
/// treating a Whisper segment as unreliable.
pub const DEFAULT_MIN_AVG_LOGPROB: f32 = -1.0;

/// Tunable thresholds for the guard (`[stt.guard]` config, EPIC 8.3).
///
/// Every field defaults independently, so a config can override one
/// threshold without restating the others.
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(default)]
pub struct GuardConfig {
    /// Minimum segment duration in milliseconds.
    pub min_speech_ms: u32,
    /// Reject when the backend's no-speech probability exceeds this.
    pub max_no_speech_prob: f32,
    /// Reject when the backend's average log probability falls below this.
    pub min_avg_logprob: f32,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            min_speech_ms: DEFAULT_MIN_SPEECH_MS,
            max_no_speech_prob: DEFAULT_MAX_NO_SPEECH_PROB,
            min_avg_logprob: DEFAULT_MIN_AVG_LOGPROB,
        }
    }
}

/// Why a transcript was not passed on to the LLM.
///
/// Carries the measurement that triggered it: when a real utterance gets
/// dropped, the logs have to say which threshold to loosen, or tuning
/// (EPIC 8.3) is guesswork.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rejection {
    /// The segment was too short to hold speech.
    TooShort {
        /// Measured segment duration, milliseconds.
        duration_ms: u64,
        /// Threshold it failed.
        min_speech_ms: u32,
    },
    /// The backend reported the audio probably contained no speech.
    NoSpeech {
        /// Reported no-speech probability.
        no_speech_prob: f32,
        /// Threshold it failed.
        max_no_speech_prob: f32,
    },
    /// The transcript's tokens were too improbable to trust.
    LowConfidence {
        /// Reported average log probability.
        avg_logprob: f32,
        /// Threshold it failed.
        min_avg_logprob: f32,
    },
    /// The backend committed no text at all — plain silence, correctly
    /// recognized. Routes through the same ERROR edge (§2.5): there is
    /// nothing for the LLM to answer either way.
    Empty,
}

impl Rejection {
    /// One-line explanation, for logs and the graceful spoken message.
    pub fn reason(&self) -> String {
        match self {
            Rejection::TooShort {
                duration_ms,
                min_speech_ms,
            } => format!("segment too short: {duration_ms}ms < {min_speech_ms}ms"),
            Rejection::NoSpeech {
                no_speech_prob,
                max_no_speech_prob,
            } => format!(
                "backend reported no speech: {no_speech_prob:.2} > {max_no_speech_prob:.2}"
            ),
            Rejection::LowConfidence {
                avg_logprob,
                min_avg_logprob,
            } => format!(
                "transcript confidence too low: {avg_logprob:.2} < {min_avg_logprob:.2}"
            ),
            Rejection::Empty => "no speech recognized".to_string(),
        }
    }
}

/// Gates transcripts on their way to the LLM.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeechGuard {
    config: GuardConfig,
}

impl SpeechGuard {
    /// Builds a guard with the given thresholds.
    pub fn new(config: GuardConfig) -> Self {
        Self { config }
    }

    /// The thresholds this guard applies.
    pub fn config(&self) -> GuardConfig {
        self.config
    }

    /// Checks a segment before inference runs.
    ///
    /// Cheap and worth doing first: audio too short to hold a word cannot
    /// produce a real transcript, so there is no reason to spend GPU time
    /// finding out what the model would invent for it.
    pub fn check_segment(&self, segment: &AudioChunk) -> Option<Rejection> {
        let duration_ms = segment_duration(segment).as_millis() as u64;
        if duration_ms < u64::from(self.config.min_speech_ms) {
            return Some(Rejection::TooShort {
                duration_ms,
                min_speech_ms: self.config.min_speech_ms,
            });
        }
        None
    }

    /// Checks a committed transcript against the backend's own signals.
    ///
    /// Missing signals are *not* treated as failures. A backend that cannot
    /// report `no_speech_prob` (the HF `whisper` worker) would otherwise
    /// have every transcript rejected, and the duration check plus whatever
    /// signals it does provide still apply.
    pub fn check_transcript(&self, text: &str, signals: SpeechSignals) -> Option<Rejection> {
        if text.trim().is_empty() {
            return Some(Rejection::Empty);
        }

        if let Some(no_speech_prob) = signals.no_speech_prob {
            if no_speech_prob > self.config.max_no_speech_prob {
                return Some(Rejection::NoSpeech {
                    no_speech_prob,
                    max_no_speech_prob: self.config.max_no_speech_prob,
                });
            }
        }

        if let Some(avg_logprob) = signals.avg_logprob {
            if avg_logprob < self.config.min_avg_logprob {
                return Some(Rejection::LowConfidence {
                    avg_logprob,
                    min_avg_logprob: self.config.min_avg_logprob,
                });
            }
        }

        None
    }
}

/// Duration of one audio chunk, from the rate and channel count it carries.
///
/// Derived from the chunk's own declared format rather than an assumed rate
/// (§2.4.1 invariant 2) — assume 16 kHz here and a 48 kHz segment looks
/// three times longer than it is, which would disarm the duration check on
/// exactly the short blips it exists to catch.
fn segment_duration(segment: &AudioChunk) -> Duration {
    let channels = u32::from(segment.channels.max(1));
    let frames = segment.pcm.len() as u64 / u64::from(channels);
    if segment.sample_rate == 0 {
        return Duration::ZERO;
    }
    Duration::from_micros(frames * 1_000_000 / u64::from(segment.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(samples: usize, sample_rate: u32, channels: u16) -> AudioChunk {
        AudioChunk {
            seq: 0,
            pcm: vec![0.1; samples],
            sample_rate,
            channels,
        }
    }

    fn signals(no_speech: Option<f32>, logprob: Option<f32>) -> SpeechSignals {
        SpeechSignals {
            no_speech_prob: no_speech,
            avg_logprob: logprob,
        }
    }

    #[test]
    fn accepts_a_segment_long_enough_to_hold_speech() {
        // 1 second at 16 kHz mono.
        let guard = SpeechGuard::default();
        assert_eq!(guard.check_segment(&segment(16_000, 16_000, 1)), None);
    }

    #[test]
    fn rejects_a_segment_too_short_to_hold_speech() {
        // 100ms, below the 250ms floor.
        let guard = SpeechGuard::default();
        let rejection = guard
            .check_segment(&segment(1_600, 16_000, 1))
            .expect("100ms should be rejected");

        assert!(matches!(rejection, Rejection::TooShort { .. }));
        assert!(rejection.reason().contains("100ms"));
    }

    #[test]
    fn duration_accounts_for_sample_rate_and_channels() {
        // 8000 stereo samples at 48 kHz is 4000 frames ~= 83ms, not 500ms.
        // Reading the chunk's declared format wrong here would disarm the
        // check on exactly the blips it exists to catch.
        let guard = SpeechGuard::default();
        assert!(guard.check_segment(&segment(8_000, 48_000, 2)).is_some());
        // The same frame count at 8 kHz really is 500ms, and passes.
        assert_eq!(guard.check_segment(&segment(8_000, 8_000, 2)), None);
    }

    #[test]
    fn a_zero_rate_segment_does_not_divide_by_zero() {
        let guard = SpeechGuard::default();
        assert!(guard.check_segment(&segment(1_600, 0, 1)).is_some());
    }

    #[test]
    fn accepts_a_genuine_utterance() {
        // The story's second `Done when`: a real transcript passes through.
        let guard = SpeechGuard::default();
        assert_eq!(
            guard.check_transcript("what time is it", signals(Some(0.02), Some(-0.15))),
            None
        );
    }

    #[test]
    fn rejects_high_no_speech_probability() {
        // The hallucination signature: fluent text over probable silence.
        let guard = SpeechGuard::default();
        let rejection = guard
            .check_transcript("Thank you for watching!", signals(Some(0.93), Some(-0.2)))
            .expect("probable silence should be rejected");

        assert!(matches!(rejection, Rejection::NoSpeech { .. }));
        assert!(rejection.reason().contains("0.93"));
    }

    #[test]
    fn rejects_low_average_logprob() {
        let guard = SpeechGuard::default();
        let rejection = guard
            .check_transcript("mumble mumble", signals(Some(0.1), Some(-2.4)))
            .expect("improbable text should be rejected");

        assert!(matches!(rejection, Rejection::LowConfidence { .. }));
    }

    #[test]
    fn rejects_empty_and_whitespace_only_text() {
        let guard = SpeechGuard::default();
        assert_eq!(
            guard.check_transcript("", signals(None, None)),
            Some(Rejection::Empty)
        );
        assert_eq!(
            guard.check_transcript("   \n", signals(None, None)),
            Some(Rejection::Empty)
        );
    }

    #[test]
    fn missing_signals_do_not_reject_on_their_own() {
        // The HF whisper backend cannot report no_speech_prob. If absence
        // counted as failure it would reject every transcript that backend
        // ever produced.
        let guard = SpeechGuard::default();
        assert_eq!(guard.check_transcript("hello there", signals(None, None)), None);
        assert_eq!(
            guard.check_transcript("hello there", signals(None, Some(-0.3))),
            None
        );
    }

    #[test]
    fn thresholds_are_adjustable() {
        // Tuning is EPIC 8.3's job; it must not require editing code.
        let strict = SpeechGuard::new(GuardConfig {
            min_speech_ms: 1_000,
            max_no_speech_prob: 0.1,
            min_avg_logprob: -0.1,
        });
        assert!(strict.check_segment(&segment(8_000, 16_000, 1)).is_some());
        assert!(strict
            .check_transcript("hi", signals(Some(0.2), None))
            .is_some());
        assert!(strict
            .check_transcript("hi", signals(None, Some(-0.5)))
            .is_some());

        let permissive = SpeechGuard::new(GuardConfig {
            min_speech_ms: 0,
            max_no_speech_prob: 1.0,
            min_avg_logprob: f32::NEG_INFINITY,
        });
        assert_eq!(permissive.check_segment(&segment(16, 16_000, 1)), None);
        assert_eq!(
            permissive.check_transcript("hi", signals(Some(0.99), Some(-9.0))),
            None
        );
    }

    #[test]
    fn checks_run_in_cheapest_first_order() {
        // Empty text short-circuits before the signal checks, so a rejection
        // names the plainest reason rather than an incidental threshold.
        let guard = SpeechGuard::default();
        assert_eq!(
            guard.check_transcript("", signals(Some(0.99), Some(-9.0))),
            Some(Rejection::Empty)
        );
    }
}
