//! Internal audio pipeline (SPEC.md §2.4.1, §2.6): capture into a
//! pre-roll ring buffer today; playback and device/resampling handling
//! land in later EPIC 1 stories.

pub mod capture;
pub mod device_select;
pub mod playback;
pub mod resample;
pub mod ring;

pub use capture::{Capture, CaptureError};
pub use playback::{Playback, PlaybackError};
pub use ring::PreRollRing;

/// Self-describing PCM audio chunk (SPEC.md §2.4.1 invariant 2): sample
/// rate and channel count travel with the data, so no consumer branches
/// on an out-of-band format assumption.
///
/// This is the Rust-internal pipeline type. It mirrors the wire
/// `AudioChunk` message in `protocol::common`, but conversion between the
/// two only happens at the worker gRPC boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    /// Monotonic sequence number; lets a consumer detect dropped/reordered chunks.
    pub seq: u64,
    /// Interleaved f32 PCM samples.
    pub pcm: Vec<f32>,
    /// Sample rate in Hz, as reported by the capture device.
    pub sample_rate: u32,
    /// Channel count, as reported by the capture device.
    pub channels: u16,
}
