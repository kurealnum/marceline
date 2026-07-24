//! The [`WakeDetector`] seam (SPEC.md EPIC 2.1): the interface a real
//! ONNX-backed openWakeWord model (EPIC 13.2) will implement, and
//! [`EnergyWakeDetector`], the placeholder that stands in until then.

/// Runs wake-word inference over 16kHz mono frames.
///
/// Implementations are fed consecutive frames via [`process`](Self::process)
/// and decide internally whether/when enough evidence has accumulated to
/// fire. A real model runs its own acoustic classifier per call; the
/// placeholder tracks a running energy envelope instead.
pub trait WakeDetector: Send {
    /// Feeds one frame of 16kHz mono f32 PCM. Returns `Some((word_index,
    /// score))` if this frame completes a fire, where `word_index` indexes
    /// into the detector's configured word list.
    fn process(&mut self, frame: &[f32]) -> Option<(usize, f32)>;

    /// The detector's current per-frame score, whether or not it fired.
    /// Lets callers log near-misses (SPEC.md EPIC 2.4: "log wake scores,
    /// fired and near-miss, so tuning is data-driven") without waiting
    /// for an actual fire.
    fn current_score(&self) -> f32;
}

/// Placeholder [`WakeDetector`]: fires when a leaky RMS envelope stays
/// above a sensitivity-derived threshold for a minimum sustained duration,
/// then enters a cooldown so one utterance doesn't fire repeatedly.
///
/// This is **not** real wake-word discrimination — it reacts to loudness,
/// not content — but it exercises every real piece of the pipeline around
/// it (config-driven sensitivity, resampling, event emission) so that
/// dropping in a real ONNX model later is a pure implementation swap.
pub struct EnergyWakeDetector {
    /// Envelope threshold derived from config sensitivity; higher
    /// sensitivity means a lower (easier to cross) threshold.
    threshold: f32,
    /// Leaky envelope decay per sample, in `(0, 1)`.
    decay: f32,
    /// Frames of continuous above-threshold energy required to fire.
    sustain_frames_required: u32,
    /// Frames to suppress new fires after one, so a single loud utterance
    /// doesn't fire repeatedly.
    cooldown_frames: u32,
    envelope: f32,
    sustain_frames: u32,
    cooldown_remaining: u32,
}

impl EnergyWakeDetector {
    /// Builds a detector from `[wake].sensitivity` (`0.0..=1.0`, higher =
    /// more sensitive = fires more easily) and frame-rate-derived sustain
    /// duration.
    ///
    /// `sample_rate` and `frame_len` size the sustain/cooldown windows in
    /// real time (defaults: ~150ms sustain, ~1.5s cooldown) regardless of
    /// how the caller chunks frames.
    pub fn new(sensitivity: f64, sample_rate: u32, frame_len: usize) -> Self {
        let sensitivity = sensitivity.clamp(0.0, 1.0) as f32;
        // Sensitivity 0.6 (the config default) -> threshold ~0.16; higher
        // sensitivity lowers the bar to cross.
        let threshold = (1.0 - sensitivity) * 0.4;
        let frames_per_sec = if frame_len == 0 {
            1.0
        } else {
            sample_rate as f32 / frame_len as f32
        };
        Self {
            threshold: threshold.max(0.01),
            decay: 0.3,
            sustain_frames_required: (frames_per_sec * 0.15).ceil().max(1.0) as u32,
            cooldown_frames: (frames_per_sec * 1.5).ceil().max(1.0) as u32,
            envelope: 0.0,
            sustain_frames: 0,
            cooldown_remaining: 0,
        }
    }
}

impl WakeDetector for EnergyWakeDetector {
    fn process(&mut self, frame: &[f32]) -> Option<(usize, f32)> {
        if frame.is_empty() {
            return None;
        }
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        // Leaky envelope: jump up instantly on loud frames, decay slowly
        // on quiet ones, so brief dips mid-word don't reset sustain.
        self.envelope = if rms > self.envelope {
            rms
        } else {
            self.envelope * (1.0 - self.decay) + rms * self.decay
        };

        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            return None;
        }

        if self.envelope >= self.threshold {
            self.sustain_frames += 1;
        } else {
            self.sustain_frames = 0;
        }

        if self.sustain_frames >= self.sustain_frames_required {
            self.sustain_frames = 0;
            self.cooldown_remaining = self.cooldown_frames;
            return Some((0, self.envelope.min(1.0)));
        }
        None
    }

    fn current_score(&self) -> f32 {
        self.envelope.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_frame(len: usize, amplitude: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amplitude * (i as f32 * 0.3).sin())
            .collect()
    }

    #[test]
    fn fires_on_sustained_loud_frames() {
        let mut detector = EnergyWakeDetector::new(0.6, 16_000, 320); // 20ms frames
        let loud = tone_frame(320, 0.9);
        let mut fired = None;
        for _ in 0..20 {
            if let Some(event) = detector.process(&loud) {
                fired = Some(event);
                break;
            }
        }
        assert!(fired.is_some(), "sustained loud audio should fire");
    }

    #[test]
    fn does_not_fire_on_silence() {
        let mut detector = EnergyWakeDetector::new(0.6, 16_000, 320);
        let silence = vec![0.0f32; 320];
        for _ in 0..50 {
            assert!(detector.process(&silence).is_none());
        }
    }

    #[test]
    fn enters_cooldown_after_firing() {
        let mut detector = EnergyWakeDetector::new(0.6, 16_000, 320);
        let loud = tone_frame(320, 0.9);
        // Fewer frames than one cooldown window (~75 frames here), so a
        // single continuous utterance must fire exactly once, not repeatedly.
        let mut fires = 0;
        for _ in 0..50 {
            if detector.process(&loud).is_some() {
                fires += 1;
            }
        }
        assert_eq!(fires, 1, "one continuous loud utterance should fire once");
    }

    #[test]
    fn higher_sensitivity_lowers_threshold() {
        let low = EnergyWakeDetector::new(0.1, 16_000, 320);
        let high = EnergyWakeDetector::new(0.9, 16_000, 320);
        assert!(high.threshold < low.threshold);
    }
}
