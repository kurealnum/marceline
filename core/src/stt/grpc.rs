//! gRPC client backend for the Python STT worker (SPEC.md §2.3, EPIC 3.2).
//!
//! This is the [`SttEngine`] implementation behind `[stt].backend =
//! "whisper"` and `"faster-whisper"` alike: both are Python workers
//! speaking the same `marceline.stt.Stt` contract over a unix domain
//! socket, so swapping between them changes which worker the supervisor
//! launched, not the code in here. That is the whole "hot-swappable via a
//! config line" promise (§2.4).
//!
//! Cancellation is cooperative and explicit (§2.5.1). Firing the run's
//! `CancellationToken` sends a `Cancel` *message* on the request stream
//! rather than dropping the connection, because a Whisper inference kernel
//! does not stop just because a socket closed — the worker's generate loop
//! has to see the flag and return early.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use marceline_protocol::stt::stt_client::SttClient;
use marceline_protocol::stt::{stt_request, stt_response, SttInfoRequest, SttRequest};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use super::{SpeechSignals, SttEngine, SttInfo, Transcript, TranscriptStream};
use crate::audio::AudioChunk;
use crate::engine::{AudioStream, EngineError};

/// Backend name used in [`EngineError`] messages and logs.
const BACKEND: &str = "stt";

/// How long to keep a cancelled RPC alive waiting for the worker to notice
/// the `Cancel` and end its stream.
///
/// Long enough for the worker to finish the decode step it is inside
/// (§2.5.1 has it check the flag between steps), short enough that a
/// misbehaving worker cannot stall the next turn.
const CANCEL_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// An STT backend backed by a Python worker on a unix domain socket.
///
/// Capabilities are fetched once at [`connect`][GrpcSttEngine::connect]
/// time and cached, which is what lets [`SttEngine::info`] stay
/// synchronous.
#[derive(Debug)]
pub struct GrpcSttEngine {
    channel: Channel,
    socket_path: PathBuf,
    info: SttInfo,
    cancel: CancellationToken,
}

impl GrpcSttEngine {
    /// Connects to the worker at `socket_path` and reads its capabilities.
    ///
    /// `cancel` is the run's cancellation token (§2.5.1), cloned into
    /// every stage; firing it makes in-flight transcriptions send the
    /// worker an explicit cancel.
    ///
    /// Fails if the worker is not reachable or will not report its
    /// capabilities — both mean there is nothing usable to hand the
    /// orchestrator, and the supervisor (EPIC 0.6) is what resolves it.
    pub async fn connect(
        socket_path: &Path,
        cancel: CancellationToken,
    ) -> Result<Self, EngineError> {
        let channel = crate::ipc::connect_uds(socket_path)
            .await
            .map_err(|err| EngineError::Transport {
                backend: BACKEND,
                source: Box::new(err),
            })?;

        let reported = SttClient::new(channel.clone())
            .get_info(SttInfoRequest {})
            .await
            .map_err(|status| EngineError::Worker {
                backend: BACKEND,
                message: format!("GetInfo failed: {}", status.message()),
            })?
            .into_inner();

        let info = SttInfo {
            name: reported.name,
            langs: reported.langs,
            input_sample_rate: reported.input_sample_rate,
            partials: reported.partials,
        };
        tracing::info!(
            socket = %socket_path.display(),
            model = %info.name,
            partials = info.partials,
            "connected to stt worker"
        );

        Ok(Self {
            channel,
            socket_path: socket_path.to_path_buf(),
            info,
            cancel,
        })
    }

    /// Path of the worker socket this backend talks to.
    ///
    /// Read by the hot-swap path (EPIC 3.4), which restarts the worker
    /// behind this socket with a different model id.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

#[async_trait]
impl SttEngine for GrpcSttEngine {
    async fn transcribe(&self, audio: AudioStream) -> TranscriptStream {
        let requests = request_stream(audio, self.cancel.clone());
        let mut client = SttClient::new(self.channel.clone());

        let responses = match client.transcribe(requests).await {
            Ok(response) => response.into_inner(),
            Err(status) => {
                // Failing to *open* the stream is still delivered in-band,
                // so callers have one error path rather than two.
                return single_error(status_to_error(status));
            }
        };

        Box::pin(response_stream(responses, self.cancel.clone()))
    }

    fn info(&self) -> SttInfo {
        self.info.clone()
    }
}

/// Maps the worker's response stream to [`Transcript`] items, discarding
/// anything committed after the run was cancelled.
///
/// Two things have to hold at once here, and they pull in opposite
/// directions:
///
/// 1. **The `Cancel` message must actually reach the worker.** Ending this
///    stream drops the whole RPC, request half included, so bailing out the
///    instant the token fires would throw away the very message §2.5.1
///    exists to deliver — leaving the worker burning GPU on audio nobody
///    wants. So after cancellation we keep the RPC alive and keep reading
///    until the worker stops on its own.
/// 2. **A transcript committed after cancel must not be used.** The worker
///    may already have sent a `final` before it noticed the cancel; per
///    §2.5.1's partial-state policy a cancelled turn's output is discarded,
///    or Marceline answers a question the user interrupted.
///
/// So: drain, drop, then report cancellation once.
fn response_stream(
    responses: tonic::Streaming<marceline_protocol::stt::SttResponse>,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<Transcript, EngineError>> + Send {
    // `None` as the state ends the stream, so cancellation is reported
    // exactly once.
    futures::stream::unfold(Some((responses, cancel)), |state| async move {
        let (mut responses, cancel) = state?;

        loop {
            let cancelled = cancel.is_cancelled();

            let next = if cancelled {
                // Already cancelled: give the worker a bounded grace period
                // to observe the cancel and end the stream. Bounded because
                // a worker that ignores cancel must not hang the caller
                // forever.
                match tokio::time::timeout(CANCEL_GRACE, responses.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        tracing::warn!(
                            grace_ms = CANCEL_GRACE.as_millis() as u64,
                            "stt worker did not end its stream after cancel"
                        );
                        return Some((Err(EngineError::Cancelled { backend: BACKEND }), None));
                    }
                }
            } else {
                tokio::select! {
                    biased;

                    // Re-enter the loop as cancelled, which switches to
                    // draining rather than yielding.
                    _ = cancel.cancelled() => {
                        tracing::debug!("run cancelled, draining stt stream without committing");
                        continue;
                    }

                    next = responses.next() => next,
                }
            };

            match next {
                Some(Ok(response)) => {
                    if cancelled {
                        // Discard rather than commit; see (2) above.
                        continue;
                    }
                    match transcript_from(response) {
                        Some(item) => return Some((item, Some((responses, cancel)))),
                        // Nothing usable in that message; keep reading
                        // rather than ending the stream early.
                        None => continue,
                    }
                }
                Some(Err(status)) => {
                    // A cancel we asked for can also surface as a stream
                    // error; report it as cancellation, not as a fault, so
                    // the orchestrator stays quiet instead of apologizing
                    // for the user having interrupted.
                    let err = if cancel.is_cancelled() {
                        EngineError::Cancelled { backend: BACKEND }
                    } else {
                        status_to_error(status)
                    };
                    return Some((Err(err), None));
                }
                None => {
                    return if cancelled {
                        Some((Err(EngineError::Cancelled { backend: BACKEND }), None))
                    } else {
                        None
                    }
                }
            }
        }
    })
}

/// Builds the request stream: audio chunks, then an explicit `Cancel` if
/// the run token fires.
///
/// The select is what makes cancel *prompt*: waiting for the audio stream
/// to finish first would leave the worker decoding audio nobody wants,
/// which is exactly the GPU-burn §2.5.1 exists to prevent.
fn request_stream(
    audio: AudioStream,
    cancel: CancellationToken,
) -> impl Stream<Item = SttRequest> + Send {
    // `None` as the state means "the stream is finished"; the cancel arm
    // yields its message and then sets that, so exactly one `Cancel` is
    // ever sent.
    futures::stream::unfold(Some((audio, cancel)), |state| async move {
        let (mut audio, cancel) = state?;

        tokio::select! {
            biased;

            // Cancel wins a tie: once the run is cancelled there is no
            // point sending another chunk of audio first.
            _ = cancel.cancelled() => {
                tracing::debug!("run cancelled, sending explicit cancel to stt worker");
                Some((cancel_request(), None))
            }

            next = audio.next() => match next {
                Some(Ok(chunk)) => Some((audio_request(chunk), Some((audio, cancel)))),
                // An error upstream of STT (the capture path failing
                // mid-utterance) is not something the worker can act on.
                // Half-close so it transcribes what it already has
                // instead of waiting on a stream that is over.
                Some(Err(err)) => {
                    tracing::warn!(%err, "audio stream failed mid-utterance, half-closing to stt worker");
                    None
                }
                None => None,
            },
        }
    })
}

/// Wraps one audio chunk as a request-stream message.
fn audio_request(chunk: AudioChunk) -> SttRequest {
    SttRequest {
        payload: Some(stt_request::Payload::Audio(wire_chunk(chunk))),
    }
}

/// The explicit cooperative-cancel message (§2.5.1).
fn cancel_request() -> SttRequest {
    SttRequest {
        payload: Some(stt_request::Payload::Cancel(marceline_protocol::common::Cancel {})),
    }
}

/// Converts an internal [`AudioChunk`] to its wire form.
///
/// The only place the two representations meet: `channels` widens from
/// `u16` to the wire's `u32`, and f32 PCM passes through unchanged
/// (invariant 2 — sample *format* is fixed by the type, rate and channels
/// travel with the data).
fn wire_chunk(chunk: AudioChunk) -> marceline_protocol::common::AudioChunk {
    marceline_protocol::common::AudioChunk {
        seq: chunk.seq,
        pcm: chunk.pcm,
        sample_rate: chunk.sample_rate,
        channels: u32::from(chunk.channels),
    }
}

/// Maps one worker response to a [`Transcript`] item.
///
/// Returns `None` for a response carrying no transcript at all, which is
/// treated as a no-op rather than an error: an empty oneof is most likely
/// a newer worker sending a variant this build does not know, and dropping
/// it degrades more gracefully than failing the turn.
fn transcript_from(
    response: marceline_protocol::stt::SttResponse,
) -> Option<Result<Transcript, EngineError>> {
    match response.transcript {
        Some(stt_response::Transcript::Final(final_transcript)) => Some(Ok(Transcript::Final {
            text: final_transcript.text,
            confidence: final_transcript.confidence,
            // Absent on the wire stays absent here; see `SpeechSignals`.
            signals: SpeechSignals {
                no_speech_prob: final_transcript.no_speech_prob,
                avg_logprob: final_transcript.avg_logprob,
            },
        })),
        Some(stt_response::Transcript::Partial(text)) => Some(Ok(Transcript::Partial(text))),
        None => {
            tracing::warn!("stt worker sent a response with no transcript, ignoring");
            None
        }
    }
}

/// Classifies a gRPC status from the worker into an [`EngineError`].
///
/// `INVALID_ARGUMENT` means we sent the worker something off-contract, so
/// it maps to [`EngineError::Protocol`]; anything else is treated as the
/// worker failing.
fn status_to_error(status: tonic::Status) -> EngineError {
    match status.code() {
        tonic::Code::InvalidArgument => EngineError::Protocol {
            backend: BACKEND,
            message: status.message().to_string(),
        },
        tonic::Code::Cancelled => EngineError::Cancelled { backend: BACKEND },
        _ => EngineError::Worker {
            backend: BACKEND,
            message: format!("{}: {}", status.code(), status.message()),
        },
    }
}

/// A transcript stream carrying exactly one error, for failures that
/// happen before the worker's stream exists.
fn single_error(err: EngineError) -> TranscriptStream {
    Box::pin(futures::stream::once(async move { Err(err) }))
}
