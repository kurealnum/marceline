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

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, Stream, StreamConfig};

use super::device_select::resolve;
use super::resample;
use super::AudioChunk;

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
    _stream: Stream,
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
            _stream: stream,
            ring,
            sample_rate,
            channels,
        })
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
