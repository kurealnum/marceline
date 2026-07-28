//! Playback wiring: `TtsEngine`'s `AudioStream` → audio-out (SPEC.md
//! §1.2, §2.4.1, EPIC 5.4).
//!
//! This is the last hop of the TTS pipeline: it drains synthesized
//! [`AudioChunk`]s into the speaker path as they arrive, so playback can
//! start on the first chunk rather than waiting for the whole answer.
//! Resampling to the output device's rate is already the audio-out
//! stage's job ([`Playback::push`][crate::audio::Playback::push],
//! EPIC 1.3) — this module only has to hand chunks over as they stream
//! in and react to cancellation and in-band errors.

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::audio::AudioChunk;
use crate::engine::{AudioStream, EngineError};

/// Backend name used in [`EngineError`] messages and logs.
const BACKEND: &str = "tts";

/// The audio-out operations playback driving needs.
///
/// Implemented by [`crate::audio::Playback`]; a separate trait so tests
/// can drive the loop below against a fake sink instead of opening a real
/// `cpal` output device, which is not available in CI.
pub trait PlaybackSink {
    /// Queues `chunk`'s PCM for playback, resampling if its declared
    /// rate/channels differ from the device's.
    fn push(&self, chunk: &AudioChunk);
    /// Drops all buffered-but-unplayed audio immediately (barge-in,
    /// §2.5.1).
    fn flush(&self);
}

impl PlaybackSink for crate::audio::Playback {
    fn push(&self, chunk: &AudioChunk) {
        crate::audio::Playback::push(self, chunk)
    }

    fn flush(&self) {
        crate::audio::Playback::flush(self)
    }
}

/// Drains `audio` into `sink` chunk by chunk until it ends, errors, or
/// `cancel` fires.
///
/// Cancellation is raced against the next chunk rather than only handled
/// via the stream's own `Err(EngineError::Cancelled)` item: the run token
/// firing must flush buffered playback *immediately*, not whenever the
/// upstream `TtsEngine` gets around to noticing its own cancel and ending
/// the stream (§2.5.1) — otherwise Marceline keeps talking over the user
/// for however long that takes.
///
/// Returns `Ok(())` when the stream ends normally, or the first
/// `EngineError` it hits (including [`EngineError::Cancelled`] when
/// `cancel` fired) — one error path, matching invariant 1 (§2.4.1).
pub async fn play(
    mut audio: AudioStream,
    sink: &impl PlaybackSink,
    cancel: CancellationToken,
) -> Result<(), EngineError> {
    loop {
        tokio::select! {
            biased;

            _ = cancel.cancelled() => {
                tracing::debug!("run cancelled, flushing tts playback");
                sink.flush();
                return Err(EngineError::Cancelled { backend: BACKEND });
            }

            next = audio.next() => match next {
                Some(Ok(chunk)) => sink.push(&chunk),
                Some(Err(err)) => {
                    // A cancel the caller already asked for can also
                    // surface as a stream error; either way, stop playing
                    // whatever is still buffered rather than letting it
                    // finish speaking (§2.5's ERROR edge / §2.5.1 barge-in).
                    sink.flush();
                    return Err(err);
                }
                None => return Ok(()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default, Clone)]
    struct FakeSink {
        pushed: Arc<Mutex<Vec<AudioChunk>>>,
        flushed: Arc<Mutex<u32>>,
    }

    impl PlaybackSink for FakeSink {
        fn push(&self, chunk: &AudioChunk) {
            self.pushed.lock().unwrap().push(chunk.clone());
        }
        fn flush(&self) {
            *self.flushed.lock().unwrap() += 1;
        }
    }

    fn chunk(seq: u64) -> AudioChunk {
        AudioChunk {
            seq,
            pcm: vec![0.1, 0.2],
            sample_rate: 24_000,
            channels: 1,
        }
    }

    fn audio_stream(chunks: Vec<Result<AudioChunk, EngineError>>) -> AudioStream {
        Box::pin(futures::stream::iter(chunks))
    }

    #[tokio::test]
    async fn pushes_every_chunk_in_order_and_returns_ok_when_the_stream_ends() {
        let sink = FakeSink::default();
        let audio = audio_stream(vec![Ok(chunk(0)), Ok(chunk(1)), Ok(chunk(2))]);

        let result = play(audio, &sink, CancellationToken::new()).await;

        assert!(result.is_ok());
        let pushed = sink.pushed.lock().unwrap();
        assert_eq!(pushed.iter().map(|c| c.seq).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(*sink.flushed.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn a_mid_stream_error_flushes_and_propagates() {
        let sink = FakeSink::default();
        let audio = audio_stream(vec![
            Ok(chunk(0)),
            Err(EngineError::Worker {
                backend: "tts",
                message: "model exploded".to_string(),
            }),
        ]);

        let result = play(audio, &sink, CancellationToken::new()).await;

        let err = result.expect_err("expected the worker error to propagate");
        assert!(matches!(err, EngineError::Worker { .. }));
        assert_eq!(*sink.flushed.lock().unwrap(), 1);
        assert_eq!(sink.pushed.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn cancelling_flushes_immediately_without_waiting_for_the_stream() {
        // A chunk stream that never ends on its own: the only way `play`
        // returns is the cancel token, and it must not have to wait for
        // an upstream cancel to notice — it flushes as soon as the token
        // fires, races against the next chunk rather than after it.
        let sink = FakeSink::default();
        let cancel = CancellationToken::new();

        let audio: AudioStream = Box::pin(futures::stream::unfold(0u64, |seq| async move {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            Some((Ok(chunk(seq)), seq + 1))
        }));

        let sink_for_task = sink.clone();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(async move { play(audio, &sink_for_task, cancel_for_task).await });

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        cancel.cancel();

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
            .await
            .expect("play must return promptly after cancel")
            .expect("task must not panic");

        let err = result.expect_err("expected cancellation");
        assert!(err.is_cancelled(), "got {err:?}");
        assert_eq!(*sink.flushed.lock().unwrap(), 1);
    }
}
