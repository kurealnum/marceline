//! Write-to-wav debug tap (SPEC.md §9.12, EPIC 1.4): serializes an
//! [`super::AudioChunk`] stream to a playable `.wav` file. Attach it to
//! either the capture or the playback side simply by calling
//! [`WavTap::write_chunk`] wherever that stream's chunks are already
//! being drained — it does not subscribe to anything on its own, so it
//! can never stall or drop the real audio path.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use super::AudioChunk;

/// Errors from creating or writing a [`WavTap`].
#[derive(Debug, thiserror::Error)]
pub enum WavTapError {
    /// Creating the `.wav` file failed.
    #[error("failed to create wav file: {0}")]
    Create(#[source] hound::Error),
    /// Writing samples to the `.wav` file failed.
    #[error("failed to write wav samples: {0}")]
    Write(#[source] hound::Error),
    /// Finalizing (closing) the `.wav` file failed.
    #[error("failed to finalize wav file: {0}")]
    Finalize(#[source] hound::Error),
    /// A chunk's format doesn't match the tap's fixed format. A single
    /// `.wav` file has one sample rate/channel count for its whole
    /// duration, so every chunk written to a given tap must agree.
    #[error(
        "chunk format {found_rate}Hz/{found_channels}ch doesn't match \
         this tap's {expected_rate}Hz/{expected_channels}ch"
    )]
    FormatMismatch {
        /// Sample rate the tap was created with.
        expected_rate: u32,
        /// Channel count the tap was created with.
        expected_channels: u16,
        /// Sample rate found on the offending chunk.
        found_rate: u32,
        /// Channel count found on the offending chunk.
        found_channels: u16,
    },
}

/// Writes an [`AudioChunk`] stream to a 32-bit float PCM `.wav` file.
pub struct WavTap {
    writer: hound::WavWriter<BufWriter<File>>,
    sample_rate: u32,
    channels: u16,
}

impl WavTap {
    /// Creates the `.wav` file at `path`, fixed to `sample_rate`/`channels`
    /// (typically the format of whichever stream you're about to tap).
    pub fn create(
        path: impl AsRef<Path>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, WavTapError> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let writer = hound::WavWriter::create(path, spec).map_err(WavTapError::Create)?;
        Ok(Self {
            writer,
            sample_rate,
            channels,
        })
    }

    /// Appends `chunk`'s PCM. Returns [`WavTapError::FormatMismatch`] if
    /// the chunk's rate/channels don't match this tap's fixed format.
    pub fn write_chunk(&mut self, chunk: &AudioChunk) -> Result<(), WavTapError> {
        if chunk.sample_rate != self.sample_rate || chunk.channels != self.channels {
            return Err(WavTapError::FormatMismatch {
                expected_rate: self.sample_rate,
                expected_channels: self.channels,
                found_rate: chunk.sample_rate,
                found_channels: chunk.channels,
            });
        }
        for &sample in &chunk.pcm {
            self.writer
                .write_sample(sample)
                .map_err(WavTapError::Write)?;
        }
        Ok(())
    }

    /// Finalizes the `.wav` file, writing the correct header/length.
    /// Happens automatically on drop if not called explicitly; call this
    /// when you want to observe write errors instead of silently ignoring
    /// them at drop time.
    pub fn finalize(self) -> Result<(), WavTapError> {
        self.writer.finalize().map_err(WavTapError::Finalize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch_path(name: &str) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!(
            "marceline-wav-tap-test-{}-{n}-{name}",
            std::process::id()
        ))
    }

    #[test]
    fn writes_a_wav_with_matching_duration_and_format() {
        let path = scratch_path("basic.wav");
        let mut tap = WavTap::create(&path, 16_000, 1).unwrap();
        let chunk = AudioChunk {
            seq: 0,
            pcm: vec![0.0; 16_000], // 1 second @ 16kHz mono
            sample_rate: 16_000,
            channels: 1,
        };
        tap.write_chunk(&chunk).unwrap();
        tap.finalize().unwrap();

        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.channels, 1);
        assert_eq!(reader.duration(), 16_000);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_chunk_with_mismatched_format() {
        let path = scratch_path("mismatch.wav");
        let mut tap = WavTap::create(&path, 16_000, 1).unwrap();
        let chunk = AudioChunk {
            seq: 0,
            pcm: vec![0.0; 100],
            sample_rate: 48_000,
            channels: 2,
        };
        let err = tap.write_chunk(&chunk).unwrap_err();
        assert!(matches!(err, WavTapError::FormatMismatch { .. }));
        drop(tap);
        std::fs::remove_file(&path).ok();
    }
}
