//! The TRANSCRIBING stage (SPEC.md §2.5, EPIC 3.3): gate segment in,
//! committed text out.
//!
//! This is the join between EPIC 2's gate and EPIC 3's STT backend, and it
//! is where the "only `Final` reaches the LLM" rule from §2.4.1 is actually
//! enforced. `Partial` items are logged and dropped here rather than being
//! passed along hopefully — one place makes the rule auditable, whereas
//! leaving it to each caller is how half-words end up in a prompt.
//!
//! It is also where the state machine's TRANSCRIBING **error edge** gets
//! something to route on. A worker that is down, that dies mid-inference,
//! or that simply stops answering must all produce an error rather than a
//! stalled turn — a voice assistant that hangs silently is worse than one
//! that admits it failed.

use std::time::Duration;

use futures::StreamExt;

use crate::audio::AudioChunk;
use crate::engine::{AudioStream, EngineError};
use crate::stt::guard::{Rejection, SpeechGuard};
use crate::stt::{SpeechSignals, SttEngine, Transcript};

/// Backend name used in errors raised by this stage.
const BACKEND: &str = "stt";

/// Audio sent per request-stream chunk, in milliseconds.
///
/// The gate hands over one whole utterance, but it is streamed to the
/// worker in slices rather than as a single message: it keeps request
/// messages a sane size, and it means a future streaming-capable backend
/// gets incremental audio without this stage changing.
pub const CHUNK_MS: u64 = 100;

/// Default ceiling on one segment's transcription.
///
/// Generous on purpose — it is a hang detector, not a latency budget. A
/// GPU transcribing a 30-second window finishes in a small number of
/// seconds; anything approaching this means the worker is wedged.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// What the TRANSCRIBING stage produced for one segment.
///
/// Three outcomes, not two, because "the worker failed" and "there was
/// nothing worth saying" call for different handling even though both stop
/// the turn: the first is a fault, the second is the empty-transcript ERROR
/// edge (§2.5) — speak a graceful message, return to IDLE.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscribeOutcome {
    /// Text the LLM should act on.
    Committed(Transcription),
    /// Nothing usable: silence, a blip, or a suspected hallucination. Routes
    /// through the ERROR edge; never forwarded to the LLM.
    Rejected(Rejection),
}

impl TranscribeOutcome {
    /// The committed transcription, or `None` when the segment was rejected.
    pub fn committed(&self) -> Option<&Transcription> {
        match self {
            TranscribeOutcome::Committed(transcription) => Some(transcription),
            TranscribeOutcome::Rejected(_) => None,
        }
    }

    /// Why the segment was rejected, if it was.
    pub fn rejection(&self) -> Option<Rejection> {
        match self {
            TranscribeOutcome::Rejected(rejection) => Some(*rejection),
            TranscribeOutcome::Committed(_) => None,
        }
    }
}

/// The committed result of transcribing one segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcription {
    /// All committed text, in order, joined by single spaces.
    pub text: String,
    /// The *lowest* confidence among the committed segments.
    ///
    /// Conservative by design: a turn is only as trustworthy as its least
    /// certain part, and averaging would let one confident segment mask a
    /// garbled one.
    pub confidence: f32,
    /// How many `Final` items the backend emitted. Usually 1; more means
    /// the segment was long enough for the backend to split it.
    pub segments: usize,
    /// The least reassuring signals seen across the committed segments.
    /// Retained after the guard has passed them so downstream logging and
    /// tuning (EPIC 8.3) can see how close a turn came to being rejected.
    pub signals: SpeechSignals,
}

impl Transcription {
    /// True when the backend committed no text at all.
    ///
    /// Silence, or a cancelled decode. Callers must not send an empty
    /// transcription to the LLM — there is nothing to answer.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Transcribes one gate-emitted segment into committed text.
///
/// Forwards only [`Transcript::Final`]; a `Partial` is logged and dropped
/// (§2.4.1). Errors from the backend arrive in-band and are returned as
/// soon as they appear, which is the TRANSCRIBING error edge (§2.5) — the
/// caller gets an error, never a hang.
///
/// `timeout` bounds the whole operation, including connecting and
/// inference. Use [`DEFAULT_TIMEOUT`] unless there is a reason not to.
pub async fn transcribe_segment(
    engine: &dyn SttEngine,
    segment: AudioChunk,
    timeout: Duration,
) -> Result<TranscribeOutcome, EngineError> {
    transcribe_segment_guarded(engine, segment, timeout, SpeechGuard::default()).await
}

/// Transcribes one segment, gating the result with `guard`.
///
/// The guard is a parameter rather than a global so thresholds can come from
/// config (EPIC 8.3) and so tests can pin them.
pub async fn transcribe_segment_guarded(
    engine: &dyn SttEngine,
    segment: AudioChunk,
    timeout: Duration,
    guard: SpeechGuard,
) -> Result<TranscribeOutcome, EngineError> {
    // Cheapest check first: audio too short to hold a word cannot have
    // produced a real transcript, so there is no reason to spend GPU time
    // finding out what the model would invent for it (EPIC 3.6).
    if let Some(rejection) = guard.check_segment(&segment) {
        tracing::info!(reason = %rejection.reason(), "rejected segment before transcribing");
        return Ok(TranscribeOutcome::Rejected(rejection));
    }

    let samples = segment.pcm.len();
    let audio = segment_stream(segment);

    match tokio::time::timeout(timeout, collect_finals(engine, audio, guard)).await {
        Ok(result) => result,
        Err(_) => {
            tracing::error!(
                timeout_ms = timeout.as_millis() as u64,
                samples,
                "stt did not produce a transcript in time"
            );
            Err(EngineError::Timeout {
                backend: BACKEND,
                elapsed_ms: timeout.as_millis() as u64,
            })
        }
    }
}

/// Drives the backend and accumulates its committed transcripts.
async fn collect_finals(
    engine: &dyn SttEngine,
    audio: AudioStream,
    guard: SpeechGuard,
) -> Result<TranscribeOutcome, EngineError> {
    let mut transcripts = engine.transcribe(audio).await;

    let mut texts: Vec<String> = Vec::new();
    let mut confidence = f32::INFINITY;
    let mut worst_signals = SpeechSignals::default();
    let mut rejection = None;

    while let Some(item) = transcripts.next().await {
        match item? {
            Transcript::Final {
                text,
                confidence: c,
                signals,
            } => {
                // Guarded per segment, not just on the joined text: one
                // hallucinated stretch inside an otherwise good utterance
                // still must not reach the LLM.
                if let Some(reason) = guard.check_transcript(&text, signals) {
                    // An empty segment between real ones is not itself a
                    // reason to discard the turn, so `Empty` is only recorded
                    // as the rejection if nothing else survives.
                    if !matches!(reason, Rejection::Empty) {
                        tracing::info!(
                            reason = %reason.reason(),
                            "rejected transcript segment"
                        );
                        rejection = Some(reason);
                    }
                    continue;
                }
                let text = text.trim().to_string();
                if !text.is_empty() {
                    texts.push(text);
                }
                confidence = confidence.min(c);
                worst_signals = worst_of(worst_signals, signals);
            }
            // Never forwarded downstream: revisable text is for UI, debug,
            // and endpointing tuning only (§2.4.1, §9.3).
            Transcript::Partial(text) => {
                tracing::debug!(partial = %text, "dropping partial transcript");
            }
        }
    }

    if texts.is_empty() {
        // Either everything was rejected, or the backend heard nothing.
        // Both route through the same ERROR edge (§2.5).
        return Ok(TranscribeOutcome::Rejected(
            rejection.unwrap_or(Rejection::Empty),
        ));
    }

    let segments = texts.len();
    Ok(TranscribeOutcome::Committed(Transcription {
        text: texts.join(" "),
        // No committed segment means there is no confidence to report;
        // 0.0 reads as "no signal" rather than as certainty.
        confidence: if confidence.is_finite() { confidence } else { 0.0 },
        segments,
        signals: worst_signals,
    }))
}

/// Keeps the least reassuring of two signal sets.
///
/// A turn is only as trustworthy as its weakest segment, so the reported
/// signals are the highest no-speech probability and the lowest log-prob
/// seen — the same reasoning as `Transcription::confidence` taking the
/// minimum.
fn worst_of(a: SpeechSignals, b: SpeechSignals) -> SpeechSignals {
    SpeechSignals {
        no_speech_prob: match (a.no_speech_prob, b.no_speech_prob) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (found, None) | (None, found) => found,
        },
        avg_logprob: match (a.avg_logprob, b.avg_logprob) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (found, None) | (None, found) => found,
        },
    }
}

/// Slices one segment into a stream of [`CHUNK_MS`]-sized chunks.
///
/// Each chunk carries the segment's own rate and channel count and its own
/// `seq`, so the receiving end can detect a drop or a reorder (invariant 2).
pub fn segment_stream(segment: AudioChunk) -> AudioStream {
    let frames_per_chunk = (segment.sample_rate as u64 * CHUNK_MS / 1_000).max(1) as usize;
    let samples_per_chunk = frames_per_chunk * segment.channels.max(1) as usize;

    let AudioChunk {
        pcm,
        sample_rate,
        channels,
        ..
    } = segment;

    // `chunks` never yields an empty slice, and an empty segment yields no
    // chunks at all — a half-closed stream with no audio, which the worker
    // answers with no transcript.
    let chunks: Vec<AudioChunk> = pcm
        .chunks(samples_per_chunk)
        .enumerate()
        .map(|(index, samples)| AudioChunk {
            seq: index as u64,
            pcm: samples.to_vec(),
            sample_rate,
            channels,
        })
        .collect();

    Box::pin(futures::stream::iter(chunks.into_iter().map(Ok)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(samples: usize, sample_rate: u32, channels: u16) -> AudioChunk {
        AudioChunk {
            seq: 0,
            pcm: vec![0.1; samples],
            sample_rate,
            channels,
        }
    }

    #[tokio::test]
    async fn slices_a_segment_into_chunk_ms_sized_pieces() {
        // 1 second of 16 kHz mono at 100ms per chunk -> 10 chunks.
        let chunks: Vec<_> = segment_stream(segment(16_000, 16_000, 1))
            .map(|item| item.expect("no errors in a sliced segment"))
            .collect()
            .await;

        assert_eq!(chunks.len(), 10);
        assert!(chunks.iter().all(|chunk| chunk.pcm.len() == 1_600));
        // Sequence numbers are monotonic from zero.
        assert_eq!(
            chunks.iter().map(|chunk| chunk.seq).collect::<Vec<_>>(),
            (0..10).collect::<Vec<u64>>()
        );
        // Format travels with every chunk.
        assert!(chunks
            .iter()
            .all(|chunk| chunk.sample_rate == 16_000 && chunk.channels == 1));
    }

    #[tokio::test]
    async fn keeps_stereo_frames_whole() {
        // Interleaved stereo must split on frame boundaries, or the
        // channels swap partway through the stream.
        let chunks: Vec<_> = segment_stream(segment(16_000, 16_000, 2))
            .map(|item| item.unwrap())
            .collect()
            .await;

        assert!(chunks
            .iter()
            .all(|chunk| chunk.pcm.len() % 2 == 0 && chunk.channels == 2));
    }

    #[tokio::test]
    async fn a_trailing_partial_chunk_is_kept() {
        // 150ms at 16 kHz -> one full 100ms chunk plus a 50ms remainder;
        // the remainder must not be dropped.
        let chunks: Vec<_> = segment_stream(segment(2_400, 16_000, 1))
            .map(|item| item.unwrap())
            .collect()
            .await;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].pcm.len(), 1_600);
        assert_eq!(chunks[1].pcm.len(), 800);
    }

    #[tokio::test]
    async fn an_empty_segment_yields_no_chunks() {
        let empty = AudioChunk {
            seq: 0,
            pcm: Vec::new(),
            sample_rate: 16_000,
            channels: 1,
        };
        let chunks: Vec<_> = segment_stream(empty).collect().await;
        assert!(chunks.is_empty());
    }

    #[test]
    fn is_empty_treats_whitespace_only_text_as_nothing_said() {
        let blank = Transcription {
            text: "   ".to_string(),
            confidence: 0.9,
            segments: 1,
            signals: SpeechSignals::default(),
        };
        assert!(blank.is_empty());
    }
}
