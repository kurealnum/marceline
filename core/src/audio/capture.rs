//! Mic capture via `cpal` into a stream of [`AudioChunk`]s, feeding a
//! shared pre-roll ring (SPEC.md §1.1, §2.6). This is the entry point for
//! all audio-in: the `cpal` callback stays allocation/lock-light and hands
//! off immediately; downstream stages (wake word, VAD, STT) drain the
//! returned channel without touching the audio thread.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream, StreamConfig};
use crossbeam_channel::{Receiver, Sender};

use super::ring::PreRollRing;
use super::AudioChunk;

/// Errors that can occur while starting mic capture.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// No default input device is available on this host.
    #[error("no default input device available")]
    NoInputDevice,
    /// Querying the device's default input config failed.
    #[error("failed to query input config: {0}")]
    UnsupportedConfig(#[from] cpal::DefaultStreamConfigError),
    /// The device's default sample format isn't one this build converts.
    #[error("unsupported input sample format: {0:?}")]
    UnsupportedSampleFormat(SampleFormat),
    /// Building the cpal input stream failed.
    #[error("failed to build input stream: {0}")]
    BuildStream(#[from] cpal::BuildStreamError),
    /// Starting playback of the input stream failed.
    #[error("failed to start input stream: {0}")]
    PlayStream(#[from] cpal::PlayStreamError),
}

/// Live mic capture: owns the `cpal` input stream and hands off
/// [`AudioChunk`]s to a channel a consumer drains, while also feeding a
/// pre-roll ring for same-breath command capture (§2.6).
pub struct Capture {
    // Keeping the stream alive keeps capture running; dropping it stops it.
    _stream: Stream,
    receiver: Receiver<AudioChunk>,
    preroll: Arc<Mutex<PreRollRing>>,
    sample_rate: u32,
    channels: u16,
}

impl Capture {
    /// Opens the default input device and starts streaming mic frames.
    /// `preroll_seconds` sizes the retained pre-roll window (SPEC.md §2.6
    /// recommends ~1-2s).
    pub fn start(preroll_seconds: f32) -> Result<Self, CaptureError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(CaptureError::NoInputDevice)?;
        let supported = device.default_input_config()?;
        let sample_rate = supported.sample_rate().0;
        let channels = supported.channels();
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();

        let preroll = Arc::new(Mutex::new(PreRollRing::with_duration(
            preroll_seconds,
            sample_rate,
            channels,
        )));
        let (tx, rx) = crossbeam_channel::unbounded();
        let seq = Arc::new(AtomicU64::new(0));

        let ctx = CallbackCtx {
            tx,
            preroll: Arc::clone(&preroll),
            seq: Arc::clone(&seq),
            sample_rate,
            channels,
        };
        let stream = build_input_stream(&device, &config, sample_format, ctx)?;
        stream.play()?;

        Ok(Self {
            _stream: stream,
            receiver: rx,
            preroll,
            sample_rate,
            channels,
        })
    }

    /// Channel downstream stages (wake word, VAD, STT) drain continuously;
    /// each item is one [`AudioChunk`] handed off from the capture callback.
    pub fn chunks(&self) -> &Receiver<AudioChunk> {
        &self.receiver
    }

    /// Snapshot of the last ~`preroll_seconds` of audio, as a single
    /// [`AudioChunk`]. `seq` is `0`: this is a synthesized window over the
    /// ring, not a discrete callback chunk.
    pub fn preroll(&self) -> AudioChunk {
        let pcm = self
            .preroll
            .lock()
            .expect("preroll ring lock poisoned")
            .snapshot();
        AudioChunk {
            seq: 0,
            pcm,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }
}

/// State shared by every invocation of the `cpal` audio callback,
/// bundled to keep [`build_input_stream`] under clippy's argument-count
/// lint while still being cheap to `Clone` per sample-format arm.
#[derive(Clone)]
struct CallbackCtx {
    tx: Sender<AudioChunk>,
    preroll: Arc<Mutex<PreRollRing>>,
    seq: Arc<AtomicU64>,
    sample_rate: u32,
    channels: u16,
}

/// Builds the input stream for `sample_format`, converting every incoming
/// buffer to f32 at this boundary (SPEC.md §2.4.1 invariant 2) before it
/// reaches [`handle_frames`].
fn build_input_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    ctx: CallbackCtx,
) -> Result<Stream, CaptureError> {
    let err_fn = |err| tracing::error!(%err, "audio capture stream error");

    macro_rules! build_converting {
        ($sample_ty:ty) => {{
            let ctx = ctx.clone();
            device.build_input_stream(
                config,
                move |data: &[$sample_ty], _| {
                    let converted: Vec<f32> = data.iter().map(|s| s.to_sample::<f32>()).collect();
                    handle_frames(&converted, &ctx);
                },
                err_fn,
                None,
            )?
        }};
    }

    let stream = match sample_format {
        SampleFormat::F32 => {
            let ctx = ctx.clone();
            device.build_input_stream(
                config,
                move |data: &[f32], _| handle_frames(data, &ctx),
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => build_converting!(i16),
        SampleFormat::U16 => build_converting!(u16),
        SampleFormat::I8 => build_converting!(i8),
        SampleFormat::U8 => build_converting!(u8),
        SampleFormat::I32 => build_converting!(i32),
        SampleFormat::U32 => build_converting!(u32),
        other => return Err(CaptureError::UnsupportedSampleFormat(other)),
    };
    Ok(stream)
}

/// Runs in the `cpal` audio callback: feeds the pre-roll ring, wraps the
/// (already f32) frames as an [`AudioChunk`], and hands it to the consumer
/// channel. The `Vec<f32>` copy out of the transient callback buffer is
/// the one unavoidable allocation; the channel send never blocks on a
/// slow/absent consumer.
fn handle_frames(data: &[f32], ctx: &CallbackCtx) {
    if let Ok(mut ring) = ctx.preroll.lock() {
        ring.push(data);
    }
    let chunk = AudioChunk {
        seq: ctx.seq.fetch_add(1, Ordering::Relaxed),
        pcm: data.to_vec(),
        sample_rate: ctx.sample_rate,
        channels: ctx.channels,
    };
    let _ = ctx.tx.send(chunk);
}
