//! Per-stage conversation latency (SPEC.md §9.2, §9.12, EPIC 12.3).
//!
//! Perceived responsiveness is wake→first-audio: the time from the wake
//! word firing to the first synthesized audio chunk reaching playback.
//! The hard target is **≤1.5s** (SPEC.md §10), treated as a regression-test
//! threshold once EPIC 12.4's canned-audio harness exists to drive a
//! deterministic timed run in CI — [`MAX_WAKE_TO_FIRST_AUDIO`] and
//! [`TurnLatencyMs::meets_target`] are the pieces that assertion needs;
//! wiring them into an actual CI job is 12.4's job, not this module's.
//!
//! [`TurnLatencyMs`] is a pure value type: `cli::converse`'s driving loop
//! records five [`std::time::Instant`]s at the state-machine boundaries it
//! already crosses every turn (wake fired, VAD end, transcript ready, LLM
//! first sentence, TTS first chunk) and calls [`TurnLatencyMs::from_instants`]
//! once the turn reaches SPEAKING — recording an `Instant` is a handful of
//! nanoseconds, so this adds no measurable overhead to the streaming path
//! it instruments.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// The hard wake→first-audio target (SPEC.md §10): perceived
/// responsiveness beyond this reads as sluggish.
pub const MAX_WAKE_TO_FIRST_AUDIO_MS: u64 = 1_500;

/// Per-stage durations for one completed conversation turn, from the wake
/// word firing through the first TTS audio chunk reaching playback.
///
/// Every field is a plain millisecond count (not a [`Duration`]) so this
/// serializes cleanly over the control socket (`core::daemon`'s
/// `StatusReport`, EPIC 11.1) and into a structured log line without a
/// custom (de)serialization impl.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnLatencyMs {
    /// Wake word fired → VAD detected end of utterance (the gate's tail,
    /// SPEC.md §2.5).
    pub vad_tail_ms: u64,
    /// End of utterance → STT produced a final transcript.
    pub stt_ms: u64,
    /// Transcript ready → the LLM's first sentence-chunked span is ready to
    /// speak (SPEC.md §2.5's `THINKING -> SPEAKING` trigger; the
    /// architecture chunks by sentence, not raw token, so this is that
    /// boundary rather than a true first-token timestamp).
    pub llm_first_sentence_ms: u64,
    /// First sentence ready → TTS produced its first audio chunk.
    pub tts_first_chunk_ms: u64,
    /// Wake word fired → first TTS audio chunk: the end-to-end
    /// wake→first-audio total this story's ≤1.5s target is measured
    /// against.
    pub total_ms: u64,
}

impl TurnLatencyMs {
    /// Builds a report from the five stage-boundary instants a turn
    /// crosses, in order: `wake`, `vad_end`, `transcript_ready`,
    /// `llm_first_sentence`, `tts_first_chunk`.
    pub fn from_instants(
        wake: Instant,
        vad_end: Instant,
        transcript_ready: Instant,
        llm_first_sentence: Instant,
        tts_first_chunk: Instant,
    ) -> Self {
        let ms = |d: Duration| d.as_millis() as u64;
        Self {
            vad_tail_ms: ms(vad_end.saturating_duration_since(wake)),
            stt_ms: ms(transcript_ready.saturating_duration_since(vad_end)),
            llm_first_sentence_ms: ms(llm_first_sentence.saturating_duration_since(transcript_ready)),
            tts_first_chunk_ms: ms(tts_first_chunk.saturating_duration_since(llm_first_sentence)),
            total_ms: ms(tts_first_chunk.saturating_duration_since(wake)),
        }
    }

    /// Whether this turn's wake→first-audio total meets the SPEC.md §10
    /// target ([`MAX_WAKE_TO_FIRST_AUDIO_MS`]).
    pub fn meets_target(&self) -> bool {
        self.total_ms <= MAX_WAKE_TO_FIRST_AUDIO_MS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_each_stage_gap_and_the_end_to_end_total() {
        let base = Instant::now();
        let wake = base;
        let vad_end = base + Duration::from_millis(300);
        let transcript_ready = vad_end + Duration::from_millis(200);
        let llm_first_sentence = transcript_ready + Duration::from_millis(150);
        let tts_first_chunk = llm_first_sentence + Duration::from_millis(100);

        let report = TurnLatencyMs::from_instants(
            wake,
            vad_end,
            transcript_ready,
            llm_first_sentence,
            tts_first_chunk,
        );

        assert_eq!(report.vad_tail_ms, 300);
        assert_eq!(report.stt_ms, 200);
        assert_eq!(report.llm_first_sentence_ms, 150);
        assert_eq!(report.tts_first_chunk_ms, 100);
        assert_eq!(report.total_ms, 750);
        assert!(report.meets_target());
    }

    #[test]
    fn a_total_over_the_threshold_does_not_meet_target() {
        let base = Instant::now();
        let wake = base;
        let tts_first_chunk = base + Duration::from_millis(MAX_WAKE_TO_FIRST_AUDIO_MS + 1);

        let report = TurnLatencyMs::from_instants(wake, wake, wake, wake, tts_first_chunk);
        assert_eq!(report.total_ms, MAX_WAKE_TO_FIRST_AUDIO_MS + 1);
        assert!(!report.meets_target());
    }

    #[test]
    fn exactly_at_the_threshold_meets_target() {
        let base = Instant::now();
        let wake = base;
        let tts_first_chunk = base + Duration::from_millis(MAX_WAKE_TO_FIRST_AUDIO_MS);

        let report = TurnLatencyMs::from_instants(wake, wake, wake, wake, tts_first_chunk);
        assert!(report.meets_target());
    }
}
