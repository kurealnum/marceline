//! Fixed-duration pre-roll ring buffer (SPEC.md §2.6).
//!
//! The capture ring retains the last ~1-2s of mic frames at all times, so
//! utterance capture seeded from it on wake word (or barge-in) doesn't
//! lose same-breath speech spoken during the ~300ms state flip to
//! LISTENING.

use std::collections::VecDeque;

/// Ring buffer over interleaved f32 samples, capped to hold at most
/// `capacity_samples` samples; pushing past capacity drops the oldest
/// samples first.
#[derive(Debug)]
pub struct PreRollRing {
    buf: VecDeque<f32>,
    capacity_samples: usize,
}

impl PreRollRing {
    /// Creates a ring sized to hold `seconds` of audio at `sample_rate` Hz
    /// across `channels` interleaved channels.
    pub fn with_duration(seconds: f32, sample_rate: u32, channels: u16) -> Self {
        let capacity_samples =
            ((seconds * sample_rate as f32) as usize) * (channels.max(1) as usize);
        Self {
            buf: VecDeque::with_capacity(capacity_samples),
            capacity_samples,
        }
    }

    /// Appends interleaved samples, trimming the oldest samples if the
    /// ring would otherwise exceed its capacity.
    pub fn push(&mut self, samples: &[f32]) {
        self.buf.extend(samples.iter().copied());
        let excess = self.buf.len().saturating_sub(self.capacity_samples);
        if excess > 0 {
            self.buf.drain(..excess);
        }
    }

    /// Returns a copy of the ring's current contents, oldest sample first.
    pub fn snapshot(&self) -> Vec<f32> {
        self.buf.iter().copied().collect()
    }

    /// Number of samples currently retained.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the ring currently holds no samples.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Capacity of the ring, in samples.
    pub fn capacity(&self) -> usize {
        self.capacity_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_at_least_the_requested_duration() {
        // 1s @ 16kHz mono = 16000 samples.
        let ring = PreRollRing::with_duration(1.0, 16_000, 1);
        assert_eq!(ring.capacity(), 16_000);
    }

    #[test]
    fn drops_oldest_samples_past_capacity() {
        let mut ring = PreRollRing::with_duration(1.0, 4, 1); // capacity = 4 samples
        ring.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ring.snapshot(), vec![1.0, 2.0, 3.0, 4.0]);

        ring.push(&[5.0, 6.0]);
        assert_eq!(ring.snapshot(), vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(ring.len(), 4);
    }

    #[test]
    fn starts_empty() {
        let ring = PreRollRing::with_duration(2.0, 48_000, 2);
        assert!(ring.is_empty());
        assert_eq!(ring.snapshot(), Vec::<f32>::new());
    }
}
