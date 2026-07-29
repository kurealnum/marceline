//! Barge-in intent gate (SPEC.md §2.5.1, EPIC 7.2): commits the interrupt
//! when a wake word fires while THINKING/SPEAKING (EPIC 7.1) — cancels the
//! in-flight run, flushes buffered playback, and lands back in LISTENING
//! with the follow-up utterance seeded from pre-roll (§2.6) so a
//! same-breath command isn't lost.
//!
//! The wake-word model (already running on every frame per 7.1), not full
//! STT, is the intent gate: any other sound during THINKING/SPEAKING is
//! ignored outright — playback keeps going and nothing reaches the LLM.
//! Debouncing a spurious/one-off wake fire is EPIC 7.3's job, not this
//! module's.

use tokio_util::sync::CancellationToken;

use crate::tts::playback::PlaybackSink;
use crate::AudioChunk;

use super::{Gate, GateOutput};

/// Given the `output` of `Gate::process_chunk` while THINKING/SPEAKING,
/// commits the barge-in if `output` is a wake detection: fires `cancel`,
/// flushes `sink`, and re-arms `gate` into LISTENING seeded from `chunk`
/// and `preroll`. Returns whether a barge-in was committed — any other
/// output is a no-op, since non-wake audio during playback must never stop
/// playback or reach the LLM.
pub fn commit_on_wake(
    gate: &mut Gate,
    output: &GateOutput,
    chunk: &AudioChunk,
    preroll: &AudioChunk,
    cancel: &CancellationToken,
    sink: &impl PlaybackSink,
) -> bool {
    if !matches!(output, GateOutput::WakeDetected) {
        return false;
    }

    cancel.cancel();
    sink.flush();
    gate.begin_listening(chunk, preroll);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{VadConfig, WakeConfig};
    use crate::{EnergyWakeDetector, GateState, SileroVad, VadEndpointer, WakeEngine};
    use std::sync::{Arc, Mutex};

    const SAMPLE_RATE: u32 = 16_000;

    #[derive(Default, Clone)]
    struct FakeSink {
        flushed: Arc<Mutex<u32>>,
    }

    impl PlaybackSink for FakeSink {
        fn push(&self, _chunk: &AudioChunk) {}
        fn flush(&self) {
            *self.flushed.lock().unwrap() += 1;
        }
    }

    fn chunk(pcm: Vec<f32>) -> AudioChunk {
        AudioChunk {
            seq: 0,
            pcm,
            sample_rate: SAMPLE_RATE,
            channels: 1,
        }
    }

    fn build_gate() -> Gate {
        let wake_config = WakeConfig {
            words: vec!["marceline".to_string()],
            sensitivity: 0.6,
        };
        let detector = EnergyWakeDetector::new(wake_config.sensitivity, SAMPLE_RATE, 320);
        let wake = WakeEngine::new(&wake_config, Box::new(detector));
        let model_path = format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"));
        let vad = SileroVad::load(model_path).expect("failed to load Silero VAD model");
        let endpointer = VadEndpointer::new(vad, crate::DEFAULT_SPEECH_THRESHOLD);
        let vad_config = VadConfig {
            silence_ms: 700,
            min_utterance_ms: 300,
            max_utterance_ms: 15_000,
        };
        Gate::new(wake, endpointer, &vad_config)
    }

    #[test]
    fn a_wake_detection_cancels_flushes_and_re_arms_listening() {
        let mut gate = build_gate();
        gate.enter_speaking();
        let preroll = chunk(vec![]);
        let cancel = CancellationToken::new();
        let sink = FakeSink::default();

        let committed = commit_on_wake(
            &mut gate,
            &GateOutput::WakeDetected,
            &chunk(vec![0.1, 0.2]),
            &preroll,
            &cancel,
            &sink,
        );

        assert!(committed);
        assert!(cancel.is_cancelled());
        assert_eq!(*sink.flushed.lock().unwrap(), 1);
        assert_eq!(gate.state(), GateState::Listening);
    }

    #[test]
    fn non_wake_output_never_commits_a_barge_in() {
        let mut gate = build_gate();
        gate.enter_speaking();
        let preroll = chunk(vec![]);
        let cancel = CancellationToken::new();
        let sink = FakeSink::default();

        let committed = commit_on_wake(
            &mut gate,
            &GateOutput::None,
            &chunk(vec![0.1, 0.2]),
            &preroll,
            &cancel,
            &sink,
        );

        assert!(!committed);
        assert!(!cancel.is_cancelled());
        assert_eq!(*sink.flushed.lock().unwrap(), 0);
        assert_eq!(gate.state(), GateState::Speaking);
    }
}
