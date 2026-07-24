//! Wires config, resampling, and a [`WakeDetector`] into something that
//! consumes capture-rate [`AudioChunk`]s and emits [`WakeEvent`]s
//! (SPEC.md EPIC 2.1).

use super::detector::WakeDetector;
use super::{WakeEvent, WAKE_SAMPLE_RATE};
use crate::audio::{resample, AudioChunk};
use crate::config::WakeConfig;

/// Consumes capture-rate [`AudioChunk`]s, resamples each to 16kHz mono
/// (the format openWakeWord — and the placeholder — expect), and runs
/// them through a [`WakeDetector`], logging and returning any fired
/// [`WakeEvent`].
pub struct WakeEngine {
    words: Vec<String>,
    detector: Box<dyn WakeDetector>,
}

impl WakeEngine {
    /// Builds an engine from `[wake]` config and a detector implementation.
    /// Pass `Box::new(EnergyWakeDetector::new(..))` for the placeholder, or
    /// a real ONNX-backed detector once EPIC 13.2 lands — nothing else
    /// here changes.
    pub fn new(config: &WakeConfig, detector: Box<dyn WakeDetector>) -> Self {
        Self {
            words: config.words.clone(),
            detector,
        }
    }

    /// Feeds one capture-rate chunk through the pipeline: resample to
    /// 16kHz mono, run the detector, and log + return an event if it fired.
    ///
    /// Every call logs the detector's current score at debug level —
    /// fired or not — so sensitivity tuning (EPIC 2.4) is data-driven
    /// instead of guesswork; a fire additionally logs at info level.
    pub fn process_chunk(&mut self, chunk: &AudioChunk) -> Option<WakeEvent> {
        let frame = resample::resample(
            &chunk.pcm,
            chunk.sample_rate,
            chunk.channels,
            WAKE_SAMPLE_RATE,
            1,
        );
        let fire = self.detector.process(&frame);
        tracing::debug!(
            score = self.detector.current_score(),
            fired = fire.is_some(),
            "wake score"
        );

        let (word_index, score) = fire?;
        let word = self
            .words
            .get(word_index)
            .cloned()
            .unwrap_or_else(|| format!("word#{word_index}"));

        tracing::info!(word = %word, score, "wake word fired");

        Some(WakeEvent {
            word,
            score,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// The detector's current per-frame score after the most recent
    /// [`process_chunk`](Self::process_chunk) call, fired or not. Lets
    /// tuning tools (EPIC 2.4) track near-miss scores without parsing logs.
    pub fn last_score(&self) -> f32 {
        self.detector.current_score()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wake::EnergyWakeDetector;

    fn config(words: &[&str]) -> WakeConfig {
        WakeConfig {
            words: words.iter().map(|s| s.to_string()).collect(),
            sensitivity: 0.6,
        }
    }

    fn tone_chunk(len: usize, amplitude: f32, sample_rate: u32) -> AudioChunk {
        AudioChunk {
            seq: 0,
            pcm: (0..len)
                .map(|i| amplitude * (i as f32 * 0.3).sin())
                .collect(),
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn fires_event_with_configured_word_on_loud_input() {
        let cfg = config(&["marceline", "marcy"]);
        let detector = EnergyWakeDetector::new(cfg.sensitivity, WAKE_SAMPLE_RATE, 320);
        let mut engine = WakeEngine::new(&cfg, Box::new(detector));

        let chunk = tone_chunk(320, 0.9, WAKE_SAMPLE_RATE);
        let mut fired = None;
        for _ in 0..20 {
            if let Some(event) = engine.process_chunk(&chunk) {
                fired = Some(event);
                break;
            }
        }
        let event = fired.expect("sustained loud audio should fire");
        assert_eq!(event.word, "marceline");
    }

    #[test]
    fn resamples_from_capture_rate_before_detecting() {
        let cfg = config(&["marceline"]);
        let detector = EnergyWakeDetector::new(cfg.sensitivity, WAKE_SAMPLE_RATE, 320);
        let mut engine = WakeEngine::new(&cfg, Box::new(detector));

        // Capture-rate chunk (44100 Hz) — engine must resample to 16kHz
        // internally rather than feeding the raw rate to the detector.
        let chunk = tone_chunk(1024, 0.9, 44_100);
        let mut fired = false;
        for _ in 0..40 {
            if engine.process_chunk(&chunk).is_some() {
                fired = true;
                break;
            }
        }
        assert!(fired, "should still fire after internal resampling");
    }

    #[test]
    fn sensitivity_measurably_shifts_fire_behavior() {
        // A borderline-quiet tone: too quiet to cross the low-sensitivity
        // threshold, loud enough to cross the high-sensitivity one.
        let borderline = tone_chunk(320, 0.2, WAKE_SAMPLE_RATE);

        let low_cfg = WakeConfig {
            words: vec!["marceline".into()],
            sensitivity: 0.3,
        };
        let mut low_engine = WakeEngine::new(
            &low_cfg,
            Box::new(EnergyWakeDetector::new(low_cfg.sensitivity, WAKE_SAMPLE_RATE, 320)),
        );
        let low_fired = (0..30).any(|_| low_engine.process_chunk(&borderline).is_some());

        let high_cfg = WakeConfig {
            words: vec!["marceline".into()],
            sensitivity: 0.9,
        };
        let mut high_engine = WakeEngine::new(
            &high_cfg,
            Box::new(EnergyWakeDetector::new(high_cfg.sensitivity, WAKE_SAMPLE_RATE, 320)),
        );
        let high_fired = (0..30).any(|_| high_engine.process_chunk(&borderline).is_some());

        assert!(!low_fired, "low sensitivity should not fire on a borderline-quiet tone");
        assert!(high_fired, "high sensitivity should fire on the same tone");
    }

    #[test]
    fn silence_never_fires() {
        let cfg = config(&["marceline"]);
        let detector = EnergyWakeDetector::new(cfg.sensitivity, WAKE_SAMPLE_RATE, 320);
        let mut engine = WakeEngine::new(&cfg, Box::new(detector));

        let chunk = AudioChunk {
            seq: 0,
            pcm: vec![0.0; 320],
            sample_rate: WAKE_SAMPLE_RATE,
            channels: 1,
        };
        for _ in 0..50 {
            assert!(engine.process_chunk(&chunk).is_none());
        }
    }
}
