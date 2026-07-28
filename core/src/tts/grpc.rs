//! gRPC client backend for the Python TTS worker (SPEC.md §2.3, EPIC 5.2).
//!
//! This is the [`TtsEngine`] implementation behind `[tts].backend =
//! "kokoro"` and `"piper"` alike (EPIC 5.5): both are Python workers
//! speaking the same `marceline.tts.Tts` contract over a unix domain
//! socket, so swapping between them changes which worker the supervisor
//! launched, not the code in here. That is the whole "hot-swappable via a
//! config line" promise (§2.4).
//!
//! Cancellation is cooperative and explicit (§2.5.1), mirroring the STT
//! gRPC client ([`crate::stt::grpc`]): firing the run's `CancellationToken`
//! sends a `Cancel` *message* on the request stream rather than dropping
//! the connection, because Kokoro synthesis does not stop just because a
//! socket closed — the worker's generate loop has to see the flag and
//! return early between sub-utterances.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use marceline_protocol::tts::tts_client::TtsClient;
use marceline_protocol::tts::{tts_request, TtsInfoRequest, TtsRequest};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use super::{TextStream, TtsEngine, TtsInfo, VoiceId};
use crate::audio::AudioChunk;
use crate::engine::{AudioStream, EngineError};

/// Backend name used in [`EngineError`] messages and logs.
const BACKEND: &str = "tts";

/// How long to keep a cancelled RPC alive waiting for the worker to notice
/// the `Cancel` and end its stream.
///
/// Long enough for the worker to finish the sub-utterance it is inside
/// (§2.5.1 has it check the flag between synthesis steps), short enough
/// that a misbehaving worker cannot stall the next turn.
const CANCEL_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// A TTS backend backed by a Python worker on a unix domain socket.
///
/// Capabilities are fetched once at [`connect`][GrpcTtsEngine::connect]
/// time and cached, which is what lets [`TtsEngine::info`] stay
/// synchronous.
#[derive(Debug)]
pub struct GrpcTtsEngine {
    channel: Channel,
    socket_path: PathBuf,
    info: TtsInfo,
    cancel: CancellationToken,
}

impl GrpcTtsEngine {
    /// Connects to the worker at `socket_path` and reads its capabilities.
    ///
    /// `cancel` is the run's cancellation token (§2.5.1), cloned into
    /// every stage; firing it makes in-flight synthesis send the worker an
    /// explicit cancel.
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

        let reported = TtsClient::new(channel.clone())
            .get_info(TtsInfoRequest {})
            .await
            .map_err(|status| EngineError::Worker {
                backend: BACKEND,
                message: format!("GetInfo failed: {}", status.message()),
            })?
            .into_inner();

        let info = TtsInfo {
            name: reported.name,
            voices: reported.voices,
            output_sample_rate: reported.output_sample_rate,
        };
        tracing::info!(
            socket = %socket_path.display(),
            model = %info.name,
            voices = info.voices.len(),
            "connected to tts worker"
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
    /// Read by the hot-swap path, which restarts the worker behind this
    /// socket with a different voice/backend.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

#[async_trait]
impl TtsEngine for GrpcTtsEngine {
    async fn synthesize(&self, text: TextStream, voice: VoiceId) -> AudioStream {
        let requests = request_stream(text, voice, self.cancel.clone());
        let mut client = TtsClient::new(self.channel.clone());

        let responses = match client.synthesize(requests).await {
            Ok(response) => response.into_inner(),
            Err(status) => {
                // Failing to *open* the stream is still delivered in-band,
                // so callers have one error path rather than two.
                return single_error(status_to_error(status));
            }
        };

        Box::pin(response_stream(responses, self.cancel.clone()))
    }

    fn info(&self) -> TtsInfo {
        self.info.clone()
    }
}

/// Maps the worker's response stream to [`AudioChunk`] items, discarding
/// anything synthesized after the run was cancelled.
///
/// Mirrors [`crate::stt::grpc`]'s `response_stream`: two things have to
/// hold at once, pulling in opposite directions.
///
/// 1. **The `Cancel` message must actually reach the worker.** Ending this
///    stream drops the whole RPC, request half included, so bailing out the
///    instant the token fires would throw away the very message §2.5.1
///    exists to deliver — leaving the worker synthesizing audio nobody
///    wants. So after cancellation we keep the RPC alive and keep reading
///    until the worker stops on its own.
/// 2. **Audio produced after cancel must not be played.** The worker may
///    already have sent a chunk before it noticed the cancel; per §2.5.1's
///    partial-state policy a cancelled turn's output is discarded, or
///    Marceline keeps speaking an answer the user interrupted.
fn response_stream(
    responses: tonic::Streaming<marceline_protocol::tts::TtsResponse>,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<AudioChunk, EngineError>> + Send {
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
                            "tts worker did not end its stream after cancel"
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
                        tracing::debug!("run cancelled, draining tts stream without playing");
                        continue;
                    }

                    next = responses.next() => next,
                }
            };

            match next {
                Some(Ok(response)) => {
                    if cancelled {
                        // Discard rather than play; see (2) above.
                        continue;
                    }
                    match chunk_from(response) {
                        Some(chunk) => return Some((Ok(chunk), Some((responses, cancel)))),
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

/// Request-stream state: the voice has not been sent yet, or has.
///
/// Modeled explicitly rather than a `bool` flag so the unfold body cannot
/// accidentally send the voice message twice or skip it.
enum RequestState {
    /// The stream has not sent its leading `voice` message yet.
    Voice(TextStream, VoiceId, CancellationToken),
    /// The voice has been sent; now streaming `text` (or cancel).
    Text(TextStream, CancellationToken),
}

/// Builds the request stream: one leading `voice` message, then text
/// spans, then an explicit `Cancel` if the run token fires.
///
/// The select on the text arm is what makes cancel *prompt*: waiting for
/// the text stream to finish first would leave the worker synthesizing
/// audio nobody wants, which is exactly the compute-burn §2.5.1 exists to
/// prevent.
fn request_stream(
    text: TextStream,
    voice: VoiceId,
    cancel: CancellationToken,
) -> impl Stream<Item = TtsRequest> + Send {
    // `None` as the state means "the stream is finished"; the cancel arm
    // yields its message and then sets that, so exactly one `Cancel` is
    // ever sent.
    futures::stream::unfold(
        Some(RequestState::Voice(text, voice, cancel)),
        |state| async move {
            match state? {
                RequestState::Voice(text, voice, cancel) => {
                    Some((voice_request(voice), Some(RequestState::Text(text, cancel))))
                }
                RequestState::Text(mut text, cancel) => {
                    tokio::select! {
                        biased;

                        // Cancel wins a tie: once the run is cancelled
                        // there is no point sending another text span first.
                        _ = cancel.cancelled() => {
                            tracing::debug!("run cancelled, sending explicit cancel to tts worker");
                            Some((cancel_request(), None))
                        }

                        next = text.next() => match next {
                            Some(Ok(span)) => {
                                Some((text_request(span), Some(RequestState::Text(text, cancel))))
                            }
                            // An error upstream of TTS (the LLM stream
                            // erroring mid-answer) is not something the
                            // worker can act on. Half-close so it finishes
                            // synthesizing what it already has instead of
                            // waiting on a stream that is over.
                            Some(Err(err)) => {
                                tracing::warn!(%err, "text stream failed mid-answer, half-closing to tts worker");
                                None
                            }
                            None => None,
                        },
                    }
                }
            }
        },
    )
}

/// Wraps the leading voice selection as a request-stream message.
fn voice_request(voice: VoiceId) -> TtsRequest {
    TtsRequest {
        payload: Some(tts_request::Payload::Voice(voice.0)),
    }
}

/// Wraps one text span as a request-stream message.
fn text_request(text: String) -> TtsRequest {
    TtsRequest {
        payload: Some(tts_request::Payload::Text(text)),
    }
}

/// The explicit cooperative-cancel message (§2.5.1).
fn cancel_request() -> TtsRequest {
    TtsRequest {
        payload: Some(tts_request::Payload::Cancel(marceline_protocol::common::Cancel {})),
    }
}

/// Converts one worker response to an [`AudioChunk`].
///
/// Returns `None` for a response carrying no audio at all, which is
/// treated as a no-op rather than an error: an unset field is most likely
/// a newer worker sending something this build does not know how to read,
/// and dropping it degrades more gracefully than failing the turn.
fn chunk_from(response: marceline_protocol::tts::TtsResponse) -> Option<AudioChunk> {
    match response.audio {
        Some(chunk) => Some(AudioChunk {
            seq: chunk.seq,
            pcm: chunk.pcm,
            sample_rate: chunk.sample_rate,
            // The wire's `channels` is `u32`; the internal type narrows to
            // `u16`, matching every real channel count that exists.
            channels: chunk.channels as u16,
        }),
        None => {
            tracing::warn!("tts worker sent a response with no audio, ignoring");
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

/// An audio stream carrying exactly one error, for failures that happen
/// before the worker's stream exists.
fn single_error(err: EngineError) -> AudioStream {
    Box::pin(futures::stream::once(async move { Err(err) }))
}
