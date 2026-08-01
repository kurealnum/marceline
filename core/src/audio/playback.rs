//! PCM playback via `cpal` (SPEC.md §1.2, §2.6): the audio-out side of the
//! pipeline. Feeds a playback ring so output can start as soon as the
//! first [`AudioChunk`] arrives, rather than waiting for a full utterance.
//! `flush` drops all buffered audio immediately — barge-in (§2.5.1) needs
//! this so Marceline doesn't keep talking over the user after cancel.
//!
//! Resampling to the device rate is explicitly deferred to story 1.3: this
//! stage assumes each chunk's declared rate already matches the device.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, Stream, StreamConfig};

use super::device_select::resolve;
use super::resample;
use super::AudioChunk;

/// How often [`Playback::null`]'s background task drains the ring —
/// frequent enough that `while buffered_samples() > 0 { sleep }` callers
/// (e.g. `cli::converse`'s run loop) don't wait long, without spinning.
const NULL_DRAIN_INTERVAL: Duration = Duration::from_millis(10);

/// Errors that can occur while starting playback.
#[derive(Debug, thiserror::Error)]
pub enum PlaybackError {
    /// No default output device is available on this host.
    #[error("no default output device available")]
    NoOutputDevice,
    /// Querying the device's default output config failed.
    #[error("failed to query output config: {0}")]
    UnsupportedConfig(#[from] cpal::DefaultStreamConfigError),
    /// The device's default sample format isn't one this build converts.
    #[error("unsupported output sample format: {0:?}")]
    UnsupportedSampleFormat(SampleFormat),
    /// Building the cpal output stream failed.
    #[error("failed to build output stream: {0}")]
    BuildStream(#[from] cpal::BuildStreamError),
    /// Starting playback of the output stream failed.
    #[error("failed to start output stream: {0}")]
    PlayStream(#[from] cpal::PlayStreamError),
}

/// Playback ring: a plain FIFO of interleaved f32 samples. Underruns
/// (reader outpaces writer) are handled by the output callback padding
/// with silence, not by anything in here.
type PlaybackRing = Arc<Mutex<VecDeque<f32>>>;

/// Live PCM playback: owns the `cpal` output stream and a playback ring
/// fed by [`Playback::push`]. Dropping this stops playback.
pub struct Playback {
    // `None` for `Playback::null` (EPIC 12.4) — a background drain task
    // stands in for the `cpal` output callback instead.
    _stream: Option<Stream>,
    ring: PlaybackRing,
    sample_rate: u32,
    channels: u16,
}

impl Playback {
    /// Opens the output device named by `output_device` (falling back to
    /// the system default if absent/empty/unrecognized, EPIC 1.3) and
    /// starts the output stream.
    pub fn start(output_device: Option<&str>) -> Result<Self, PlaybackError> {
        let host = cpal::default_host();
        let devices = host
            .output_devices()
            .map_err(|_| PlaybackError::NoOutputDevice)?;
        let device = resolve(devices, output_device, "output", || {
            host.default_output_device()
        })
        .ok_or(PlaybackError::NoOutputDevice)?;
        let supported = device.default_output_config()?;
        let sample_rate = supported.sample_rate().0;
        let channels = supported.channels();
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();

        let ring: PlaybackRing = Arc::new(Mutex::new(VecDeque::new()));
        let stream = build_output_stream(&device, &config, sample_format, Arc::clone(&ring))?;
        stream.play()?;

        Ok(Self {
            _stream: Some(stream),
            ring,
            sample_rate,
            channels,
        })
    }

    /// Builds a [`Playback`] with no real audio device behind it (EPIC
    /// 12.4's canned-audio integration harness, and any other headless
    /// test): a background task drains the ring every
    /// [`NULL_DRAIN_INTERVAL`], standing in for the `cpal` output
    /// callback that would otherwise consume it in real time. `push`,
    /// `flush`, and `buffered_samples` all behave identically to a real
    /// [`Playback`] from the caller's point of view — the only observable
    /// difference is that no sound plays anywhere.
    pub fn null(sample_rate: u32, channels: u16) -> Self {
        let ring: PlaybackRing = Arc::new(Mutex::new(VecDeque::new()));
        let drain_ring = Arc::clone(&ring);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(NULL_DRAIN_INTERVAL).await;
                drain_ring.lock().expect("playback ring lock poisoned").clear();
            }
        });

        Self {
            _stream: None,
            ring,
            sample_rate,
            channels,
        }
    }

    /// Device output sample rate, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Device output channel count.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Appends `chunk`'s PCM to the playback ring, resampling from the
    /// chunk's declared rate/channels to the device's if they differ
    /// (EPIC 1.3) — this stage owns resampling (SPEC.md §2.4.1), driven
    /// by each chunk's self-describing rate.
    pub fn push(&self, chunk: &AudioChunk) {
        let pcm = if chunk.sample_rate == self.sample_rate && chunk.channels == self.channels {
            chunk.pcm.clone()
        } else {
            resample::resample(
                &chunk.pcm,
                chunk.sample_rate,
                chunk.channels,
                self.sample_rate,
                self.channels,
            )
        };
        let mut ring = self.ring.lock().expect("playback ring lock poisoned");
        ring.extend(pcm);
    }

    /// Drops all currently buffered audio immediately, so nothing already
    /// queued keeps playing after this call returns. Used by barge-in
    /// (§2.5.1) alongside firing the run cancellation token.
    pub fn flush(&self) {
        self.ring.lock().expect("playback ring lock poisoned").clear();
    }

    /// Number of samples currently buffered and not yet played.
    pub fn buffered_samples(&self) -> usize {
        self.ring.lock().expect("playback ring lock poisoned").len()
    }
}

/// Builds the output stream for `sample_format`, converting buffered f32
/// samples to the device's native format on write; pads with silence on
/// underrun rather than stuttering or panicking.
fn build_output_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    ring: PlaybackRing,
) -> Result<Stream, PlaybackError> {
    let err_fn = |err| tracing::error!(%err, "audio playback stream error");

    macro_rules! build_converting {
        ($sample_ty:ty) => {{
            let ring = ring.clone();
            device.build_output_stream(
                config,
                move |out: &mut [$sample_ty], _| fill_output(out, &ring),
                err_fn,
                None,
            )?
        }};
    }

    let stream = match sample_format {
        SampleFormat::F32 => build_converting!(f32),
        SampleFormat::I16 => build_converting!(i16),
        SampleFormat::U16 => build_converting!(u16),
        SampleFormat::I8 => build_converting!(i8),
        SampleFormat::U8 => build_converting!(u8),
        SampleFormat::I32 => build_converting!(i32),
        SampleFormat::U32 => build_converting!(u32),
        other => return Err(PlaybackError::UnsupportedSampleFormat(other)),
    };
    Ok(stream)
}

/// Runs in the `cpal` output callback: drains buffered f32 samples into
/// `out`, converting to `T`, and pads any shortfall with silence so an
/// underrun is inaudible silence rather than a stutter or panic.
fn fill_output<T>(out: &mut [T], ring: &PlaybackRing)
where
    T: Sample + FromSample<f32> + Default,
{
    let mut ring = ring.lock().expect("playback ring lock poisoned");
    for sample in out.iter_mut() {
        *sample = match ring.pop_front() {
            Some(value) => T::from_sample(value),
            None => T::default(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn null_playback_accepts_pushes_and_drains_them_in_the_background() {
        let playback = Playback::null(16_000, 1);
        assert_eq!(playback.sample_rate(), 16_000);
        assert_eq!(playback.channels(), 1);

        playback.push(&AudioChunk {
            seq: 0,
            pcm: vec![0.1; 400],
            sample_rate: 16_000,
            channels: 1,
        });
        assert!(playback.buffered_samples() > 0);

        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        while playback.buffered_samples() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(playback.buffered_samples(), 0, "null playback must drain on its own");
    }

    #[tokio::test]
    async fn null_playback_flush_clears_the_ring_immediately() {
        let playback = Playback::null(16_000, 1);
        playback.push(&AudioChunk {
            seq: 0,
            pcm: vec![0.1; 400],
            sample_rate: 16_000,
            channels: 1,
        });
        playback.flush();
        assert_eq!(playback.buffered_samples(), 0);
    }
}
