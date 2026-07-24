//! Resampling owned by the audio-out stage (SPEC.md §2.4.1, EPIC 1.3):
//! without it, a 22050 Hz TTS worker feeding a 48000 Hz sink produces a
//! chipmunk voice. v1 uses linear interpolation for rate conversion and a
//! simple duplicate/average/cycle strategy for channel count — cheap,
//! branch-free, and good enough for voice; swap for a proper
//! band-limited resampler (e.g. `rubato`) later if quality demands it.

/// Converts interleaved `pcm` from `(from_rate, from_channels)` to
/// `(to_rate, to_channels)`. A no-op (returns `pcm` unchanged, copied)
/// when both already match.
pub fn resample(
    pcm: &[f32],
    from_rate: u32,
    from_channels: u16,
    to_rate: u32,
    to_channels: u16,
) -> Vec<f32> {
    let channel_matched = reconcile_channels(pcm, from_channels, to_channels);
    resample_rate(&channel_matched, to_channels, from_rate, to_rate)
}

/// Reconciles channel count by duplicating (mono -> N), averaging
/// (N -> mono), or cycling source channels (arbitrary N -> M).
fn reconcile_channels(pcm: &[f32], from: u16, to: u16) -> Vec<f32> {
    if from == to || from == 0 || to == 0 {
        return pcm.to_vec();
    }
    let from = from as usize;
    let to = to as usize;
    let frames = pcm.len() / from;
    let mut out = Vec::with_capacity(frames * to);
    for frame_idx in 0..frames {
        let frame = &pcm[frame_idx * from..frame_idx * from + from];
        if to == 1 {
            out.push(frame.iter().sum::<f32>() / from as f32);
        } else if from == 1 {
            out.extend(std::iter::repeat_n(frame[0], to));
        } else {
            for c in 0..to {
                out.push(frame[c % from]);
            }
        }
    }
    out
}

/// Linearly resamples `pcm` (already at `channels` channel count) from
/// `from_rate` to `to_rate`.
fn resample_rate(pcm: &[f32], channels: u16, from_rate: u32, to_rate: u32) -> Vec<f32> {
    let channels = channels as usize;
    if from_rate == to_rate || channels == 0 {
        return pcm.to_vec();
    }
    let in_frames = pcm.len() / channels;
    if in_frames == 0 {
        return Vec::new();
    }
    let out_frames = ((in_frames as u64 * to_rate as u64) / from_rate as u64) as usize;
    let mut out = Vec::with_capacity(out_frames * channels);
    for i in 0..out_frames {
        let src_pos = i as f64 * from_rate as f64 / to_rate as f64;
        let idx0 = (src_pos.floor() as usize).min(in_frames - 1);
        let idx1 = (idx0 + 1).min(in_frames - 1);
        let frac = (src_pos - idx0 as f64) as f32;
        for c in 0..channels {
            let s0 = pcm[idx0 * channels + c];
            let s1 = pcm[idx1 * channels + c];
            out.push(s0 + (s1 - s0) * frac);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_when_rate_and_channels_already_match() {
        let pcm = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(resample(&pcm, 16_000, 1, 16_000, 1), pcm);
    }

    #[test]
    fn upsamples_2x_via_linear_interpolation() {
        let pcm = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample(&pcm, 8_000, 1, 16_000, 1);
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[2], 1.0);
        // Midpoint between 0.0 and 1.0 should be linearly interpolated.
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn downsamples_by_half() {
        let pcm = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = resample(&pcm, 12_000, 1, 6_000, 1);
        assert_eq!(out.len(), 3);
        assert_eq!(out, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn duplicates_mono_to_stereo() {
        let pcm = vec![0.5, -0.5];
        let out = resample(&pcm, 16_000, 1, 16_000, 2);
        assert_eq!(out, vec![0.5, 0.5, -0.5, -0.5]);
    }

    #[test]
    fn averages_stereo_to_mono() {
        let pcm = vec![1.0, 0.0, 0.0, 1.0];
        let out = resample(&pcm, 16_000, 2, 16_000, 1);
        assert_eq!(out, vec![0.5, 0.5]);
    }

    #[test]
    fn combines_channel_and_rate_conversion() {
        let pcm = vec![0.0, 1.0]; // mono, 2 frames @ 8kHz
        let out = resample(&pcm, 8_000, 1, 16_000, 2);
        // Channel duplication first -> [0,0,1,1] stereo @ 8kHz, then 2x
        // upsample per-channel -> 4 stereo frames.
        assert_eq!(out.len(), 8);
    }
}
