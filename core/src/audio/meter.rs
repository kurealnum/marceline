//! Live audio level metering (SPEC.md §9.12, EPIC 1.4): peak + RMS over
//! each [`super::AudioChunk`], rendered as a simple console bar so
//! "is the mic even working / why does it sound wrong" is visible at a
//! glance in a realtime, multi-process system that's otherwise blind.

/// Peak/RMS level meter, updated one [`super::AudioChunk`] at a time.
/// Not smoothed across chunks — each `update` reflects only the samples
/// just given it; callers wanting a decaying meter can blend consecutive
/// `peak()`/`rms()` readings themselves.
#[derive(Debug, Default, Clone, Copy)]
pub struct LevelMeter {
    peak: f32,
    rms: f32,
}

impl LevelMeter {
    /// Recomputes peak and RMS from `samples`.
    pub fn update(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            self.peak = 0.0;
            self.rms = 0.0;
            return;
        }
        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f32;
        for &s in samples {
            peak = peak.max(s.abs());
            sum_sq += s * s;
        }
        self.peak = peak;
        self.rms = (sum_sq / samples.len() as f32).sqrt();
    }

    /// Peak absolute sample value from the last `update` (in `[0, 1]` for
    /// well-behaved PCM; may exceed 1 on clipping input).
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// RMS level from the last `update`.
    pub fn rms(&self) -> f32 {
        self.rms
    }

    /// Renders a fixed-`width` console bar driven by peak level, plus
    /// peak/RMS in dBFS.
    pub fn render(&self, width: usize) -> String {
        let filled = (self.peak.clamp(0.0, 1.0) * width as f32).round() as usize;
        let bar: String = (0..width).map(|i| if i < filled { '#' } else { '-' }).collect();
        format!(
            "[{bar}] peak={:>6.1}dB rms={:>6.1}dB",
            to_dbfs(self.peak),
            to_dbfs(self.rms)
        )
    }
}

/// Converts a linear amplitude to dBFS (0 dBFS = full scale, `-inf` at silence).
fn to_dbfs(linear: f32) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_reads_zero() {
        let mut meter = LevelMeter::default();
        meter.update(&[0.0; 100]);
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.rms(), 0.0);
    }

    #[test]
    fn full_scale_square_wave_has_unit_peak_and_rms() {
        let mut meter = LevelMeter::default();
        let samples: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        meter.update(&samples);
        assert!((meter.peak() - 1.0).abs() < 1e-6);
        assert!((meter.rms() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_chunk_resets_to_zero() {
        let mut meter = LevelMeter::default();
        meter.update(&[1.0, -1.0]);
        meter.update(&[]);
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.rms(), 0.0);
    }

    #[test]
    fn render_is_full_width_at_full_scale() {
        let mut meter = LevelMeter::default();
        meter.update(&[1.0]);
        let rendered = meter.render(10);
        assert!(rendered.starts_with("[##########]"));
    }
}
