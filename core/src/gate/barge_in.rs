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

use crate::audio::resample;
use crate::tts::playback::PlaybackSink;
use crate::vad::{VadEndpointer, FRAME_SAMPLES};
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

/// Debounces a raw wake detection into a barge-in (EPIC 7.3): cancelling
/// is expensive (kills in-flight GPU work, drops the LLM stream, §2.5.1),
/// so a bare `GateOutput::WakeDetected` isn't enough — this requires
/// `confirm_ms` of continuous VAD-confirmed speech to follow before
/// actually committing via [`commit_on_wake`]. A cough or one-off false
/// positive that doesn't hold breaks the confirmation window and playback
/// keeps going untouched.
pub struct Debounce {
    vad: VadEndpointer,
    confirm_ms: u64,
    confirming: bool,
    confirmed_ms: u64,
    pending: Vec<f32>,
}

impl Debounce {
    /// Wraps a fresh [`VadEndpointer`] (its own instance, independent of
    /// the one `Gate` uses for utterance endpointing) with the
    /// `[vad].barge_in_confirm_ms` threshold.
    pub fn new(vad: VadEndpointer, confirm_ms: u32) -> Self {
        Self {
            vad,
            confirm_ms: confirm_ms as u64,
            confirming: false,
            confirmed_ms: 0,
            pending: Vec::new(),
        }
    }

    /// Feeds one THINKING/SPEAKING-armed `chunk` alongside the `output`
    /// `Gate::process_chunk` returned for it. A raw `GateOutput::WakeDetected`
    /// opens (or continues) the confirmation window; any other output is
    /// only relevant if a window is already open (still checking whether
    /// speech continues). Once `confirm_ms` of continuous VAD-confirmed
    /// speech has accumulated, commits the barge-in and returns `true`;
    /// broken confirmation or a window that hasn't closed yet returns
    /// `false` and playback continues untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn on_output(
        &mut self,
        gate: &mut Gate,
        output: &GateOutput,
        chunk: &AudioChunk,
        preroll: &AudioChunk,
        cancel: &CancellationToken,
        sink: &impl PlaybackSink,
    ) -> bool {
        if !self.confirming {
            if !matches!(output, GateOutput::WakeDetected) {
                return false;
            }
            self.confirming = true;
            self.confirmed_ms = 0;
            self.vad.reset();
            self.pending.clear();
        }

        let chunk_ms = if chunk.sample_rate == 0 || chunk.channels == 0 {
            0
        } else {
            (chunk.pcm.len() as u64 * 1000) / (chunk.sample_rate as u64 * chunk.channels as u64)
        };

        self.pending.extend(resample::resample(
            &chunk.pcm,
            chunk.sample_rate,
            chunk.channels,
            crate::vad::model::SAMPLE_RATE as u32,
            1,
        ));

        let mut saw_speech = false;
        let mut saw_silence = false;
        while self.pending.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = self.pending.drain(..FRAME_SAMPLES).collect();
            if self.vad.process_frame(&frame).is_ok() {
                if self.vad.is_speaking() {
                    saw_speech = true;
                } else {
                    saw_silence = true;
                }
            }
        }

        if saw_silence && !saw_speech {
            self.confirming = false;
            return false;
        }

        if saw_speech {
            self.confirmed_ms += chunk_ms;
        }

        if self.confirmed_ms < self.confirm_ms {
            return false;
        }

        self.confirming = false;
        commit_on_wake(gate, output, chunk, preroll, cancel, sink)
    }
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

    fn silence_chunk(len: usize) -> AudioChunk {
        chunk(vec![0.0; len])
    }

    fn load_speech_sample() -> Vec<f32> {
        let fixture = format!(
            "{}/tests/fixtures/speech_sample.wav",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut reader = hound::WavReader::open(&fixture).expect("failed to open speech fixture");
        reader
            .samples::<i16>()
            .map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32)
            .collect()
    }

    fn build_debounce(confirm_ms: u32) -> Debounce {
        let model_path = format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"));
        let vad = SileroVad::load(model_path).expect("failed to load Silero VAD model");
        let endpointer = VadEndpointer::new(vad, crate::DEFAULT_SPEECH_THRESHOLD);
        Debounce::new(endpointer, confirm_ms)
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
            barge_in_confirm_ms: 300,
        no_speech_timeout_ms: 3_000,
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

    #[test]
    fn sustained_speech_confirms_and_commits_the_barge_in() {
        let mut gate = build_gate();
        gate.enter_speaking();
        let preroll = chunk(vec![]);
        let cancel = CancellationToken::new();
        let sink = FakeSink::default();
        let mut debounce = build_debounce(100);

        let speech = load_speech_sample();
        let mut committed = false;
        for (i, frame) in speech.chunks(1600).enumerate() {
            let output = if i == 0 {
                GateOutput::WakeDetected
            } else {
                GateOutput::None
            };
            if debounce.on_output(
                &mut gate,
                &output,
                &chunk(frame.to_vec()),
                &preroll,
                &cancel,
                &sink,
            ) {
                committed = true;
                break;
            }
        }

        assert!(committed, "sustained speech should confirm the barge-in");
        assert!(cancel.is_cancelled());
        assert_eq!(*sink.flushed.lock().unwrap(), 1);
        assert_eq!(gate.state(), GateState::Listening);
    }

    #[test]
    fn silence_after_the_initial_detection_never_commits_a_barge_in() {
        let mut gate = build_gate();
        gate.enter_speaking();
        let preroll = chunk(vec![]);
        let cancel = CancellationToken::new();
        let sink = FakeSink::default();
        let mut debounce = build_debounce(300);

        let mut committed = false;
        for i in 0..20 {
            let output = if i == 0 {
                GateOutput::WakeDetected
            } else {
                GateOutput::None
            };
            if debounce.on_output(
                &mut gate,
                &output,
                &silence_chunk(1600),
                &preroll,
                &cancel,
                &sink,
            ) {
                committed = true;
            }
        }

        assert!(!committed, "a spurious detection with no follow-up speech must not commit");
        assert!(!cancel.is_cancelled());
        assert_eq!(*sink.flushed.lock().unwrap(), 0);
        assert_eq!(gate.state(), GateState::Speaking);
    }
}
