//! Reading a wav file into the internal audio type.
//!
//! Backs `marceline transcribe <file>` (EPIC 3.3, 11.4): a file stands in
//! for a gate-emitted segment, which makes the STT path testable and
//! demoable without a microphone or a wake word.
//!
//! Everything converts to f32 on the way in (§2.4.1 invariant 2), so no
//! consumer downstream branches on sample format.

use std::path::{Path, PathBuf};

use super::AudioChunk;

/// Errors that can occur while reading a wav file.
#[derive(Debug, thiserror::Error)]
pub enum WavReadError {
    /// The file could not be opened or is not readable wav.
    #[error("failed to open wav file {path}: {source}")]
    Open {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying decoder error.
        #[source]
        source: hound::Error,
    },
    /// A sample could not be decoded.
    #[error("failed to decode wav file {path}: {source}")]
    Decode {
        /// Path being decoded.
        path: PathBuf,
        /// Underlying decoder error.
        #[source]
        source: hound::Error,
    },
    /// The file's sample format or bit depth is not supported.
    #[error("unsupported wav format in {path}: {format:?} {bits} bit")]
    UnsupportedFormat {
        /// Path with the unsupported format.
        path: PathBuf,
        /// Sample format the file declares.
        format: hound::SampleFormat,
        /// Bit depth the file declares.
        bits: u16,
    },
    /// The file holds no audio.
    #[error("wav file {path} contains no samples")]
    Empty {
        /// Path that turned out to be empty.
        path: PathBuf,
    },
}

/// Reads `path` into a single [`AudioChunk`] at the file's native format.
///
/// The chunk keeps the file's own sample rate and channel count rather
/// than being normalized here: rate and channels travel with the data, and
/// the STT backend is what resamples (invariant 2). `seq` is 0 — the whole
/// file is one segment, exactly as the gate emits one segment per
/// utterance.
pub fn read_wav(path: &Path) -> Result<AudioChunk, WavReadError> {
    let mut reader = hound::WavReader::open(path).map_err(|source| WavReadError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    let spec = reader.spec();

    let pcm = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|source| WavReadError::Decode {
                path: path.to_path_buf(),
                source,
            })?,
        // Integer PCM scales by the format's full-scale value, so a
        // full-amplitude sample lands at 1.0 regardless of bit depth.
        (hound::SampleFormat::Int, bits @ (16 | 24 | 32)) => {
            let scale = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|s| s as f32 / scale))
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|source| WavReadError::Decode {
                    path: path.to_path_buf(),
                    source,
                })?
        }
        (format, bits) => {
            return Err(WavReadError::UnsupportedFormat {
                path: path.to_path_buf(),
                format,
                bits,
            })
        }
    };

    if pcm.is_empty() {
        return Err(WavReadError::Empty {
            path: path.to_path_buf(),
        });
    }

    Ok(AudioChunk {
        seq: 0,
        pcm,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("marceline-wav-read-test-{}-{name}", std::process::id()))
    }

    fn write_wav(path: &Path, spec: hound::WavSpec, samples: &[f32]) {
        let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
        match spec.sample_format {
            hound::SampleFormat::Float => {
                for &sample in samples {
                    writer.write_sample(sample).expect("write f32 sample");
                }
            }
            hound::SampleFormat::Int => {
                for &sample in samples {
                    writer
                        .write_sample((sample * i16::MAX as f32) as i16)
                        .expect("write i16 sample");
                }
            }
        }
        writer.finalize().expect("finalize wav");
    }

    #[test]
    fn reads_f32_wav_preserving_format() {
        let path = temp_path("f32.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        write_wav(&path, spec, &[0.25, -0.5, 0.75]);

        let chunk = read_wav(&path).expect("read wav");
        assert_eq!(chunk.seq, 0);
        assert_eq!(chunk.sample_rate, 16_000);
        assert_eq!(chunk.channels, 1);
        assert_eq!(chunk.pcm, vec![0.25, -0.5, 0.75]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn converts_16_bit_int_wav_to_f32() {
        let path = temp_path("i16.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        write_wav(&path, spec, &[1.0, -1.0, 0.5, -0.5]);

        let chunk = read_wav(&path).expect("read wav");
        assert_eq!(chunk.sample_rate, 48_000);
        assert_eq!(chunk.channels, 2);
        assert_eq!(chunk.pcm.len(), 4);
        // Round-trip through i16 is lossy by up to one LSB.
        assert!((chunk.pcm[0] - 1.0).abs() < 1e-4);
        assert!((chunk.pcm[1] + 1.0).abs() < 1e-4);
        assert!((chunk.pcm[2] - 0.5).abs() < 1e-4);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_an_empty_wav() {
        let path = temp_path("empty.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        write_wav(&path, spec, &[]);

        let err = read_wav(&path).expect_err("an empty wav has nothing to transcribe");
        assert!(matches!(err, WavReadError::Empty { .. }), "got {err:?}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reports_a_missing_file_clearly() {
        let err = read_wav(&temp_path("absent.wav")).expect_err("missing file must fail");
        assert!(matches!(err, WavReadError::Open { .. }), "got {err:?}");
    }
}
