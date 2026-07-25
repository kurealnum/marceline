//! Integration tests for the gRPC STT client backend (EPIC 3.2).
//!
//! These run a fake `marceline.stt.Stt` server on a real unix domain
//! socket and drive [`GrpcSttEngine`] against it. The transport and the
//! bidirectional streaming shape are most of what this backend *is*, so
//! stubbing them out would leave the interesting parts untested; what is
//! faked is the model behind the contract, not the contract.
//!
//! The Python worker's own behavior is covered by its `unittest` suite
//! (`workers/stt/tests`); these tests cover the Rust half of the same
//! contract.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use marceline_core::audio::AudioChunk;
use marceline_core::engine::{AudioStream, EngineError};
use marceline_core::stt::{GrpcSttEngine, SttEngine, Transcript};
use marceline_protocol::stt::stt_server::{Stt, SttServer};
use marceline_protocol::stt::{
    stt_request, stt_response, FinalTranscript, SttInfo, SttInfoRequest, SttRequest, SttResponse,
};
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, Streaming};

/// Counter making each test's socket path unique within this process.
static SOCKET_SEQ: AtomicU32 = AtomicU32::new(0);

fn unique_socket_path(name: &str) -> PathBuf {
    let n = SOCKET_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "marceline-stt-test-{}-{n}-{name}.sock",
        std::process::id()
    ))
}

/// What a fake worker should do when it gets a `Transcribe` stream.
#[derive(Clone)]
enum Behavior {
    /// Consume the request stream, then emit one `final`.
    FinalAfterHalfClose { text: String, confidence: f32 },
    /// Emit a `partial` before the `final`, as a partials-capable backend
    /// would. Used to prove the client maps both variants.
    PartialThenFinal { partial: String, text: String },
    /// Fail mid-stream with the given gRPC status, as a worker hitting
    /// CUDA OOM at chunk 40 does.
    FailMidStream(tonic::Code, String),
    /// Wait for an explicit `Cancel` message and end the stream without
    /// emitting a transcript, as the real worker does on cancel.
    AwaitCancel,
    /// Emit a `final` even after being cancelled, modelling a worker that
    /// had already committed a transcript before it noticed the cancel.
    FinalDespiteCancel { text: String },
}

/// Fake `Stt` server standing in for the Python worker.
struct FakeWorker {
    behavior: Behavior,
    info: SttInfo,
    /// Audio chunks received, so tests can assert what was actually sent.
    received: Arc<Mutex<Vec<marceline_protocol::common::AudioChunk>>>,
    /// Set once an explicit `Cancel` message arrives on the request stream.
    saw_cancel: Arc<Mutex<bool>>,
}

#[tonic::async_trait]
impl Stt for FakeWorker {
    type TranscribeStream = ReceiverStream<Result<SttResponse, Status>>;

    async fn transcribe(
        &self,
        request: Request<Streaming<SttRequest>>,
    ) -> Result<Response<Self::TranscribeStream>, Status> {
        let mut inbound = request.into_inner();
        let behavior = self.behavior.clone();
        let received = Arc::clone(&self.received);
        let saw_cancel = Arc::clone(&self.saw_cancel);
        let (tx, rx) = mpsc::channel(8);

        tokio::spawn(async move {
            if let Behavior::FailMidStream(code, message) = &behavior {
                // Fail without waiting for half-close, which is what
                // "mid-stream" means from the client's point of view.
                let _ = tx.send(Err(Status::new(*code, message.clone()))).await;
                return;
            }

            while let Some(request) = inbound.next().await {
                let Ok(request) = request else { return };
                match request.payload {
                    Some(stt_request::Payload::Audio(chunk)) => {
                        received.lock().unwrap().push(chunk);
                    }
                    Some(stt_request::Payload::Cancel(_)) => {
                        *saw_cancel.lock().unwrap() = true;
                        if let Behavior::FinalDespiteCancel { text } = &behavior {
                            let _ = tx
                                .send(Ok(SttResponse {
                                    transcript: Some(stt_response::Transcript::Final(
                                        FinalTranscript {
                                            text: text.clone(),
                                            confidence: 1.0,
                                        },
                                    )),
                                }))
                                .await;
                        }
                        // Real worker behavior: stop, emit nothing.
                        return;
                    }
                    None => return,
                }
            }

            match behavior {
                Behavior::FinalAfterHalfClose { text, confidence } => {
                    let _ = tx
                        .send(Ok(SttResponse {
                            transcript: Some(stt_response::Transcript::Final(FinalTranscript {
                                text,
                                confidence,
                            })),
                        }))
                        .await;
                }
                Behavior::PartialThenFinal { partial, text } => {
                    let _ = tx
                        .send(Ok(SttResponse {
                            transcript: Some(stt_response::Transcript::Partial(partial)),
                        }))
                        .await;
                    let _ = tx
                        .send(Ok(SttResponse {
                            transcript: Some(stt_response::Transcript::Final(FinalTranscript {
                                text,
                                confidence: 1.0,
                            })),
                        }))
                        .await;
                }
                // Half-close with no cancel: emit nothing, as the client
                // asked for nothing.
                Behavior::AwaitCancel | Behavior::FinalDespiteCancel { .. } => {}
                Behavior::FailMidStream(..) => unreachable!("handled above"),
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn get_info(&self, _: Request<SttInfoRequest>) -> Result<Response<SttInfo>, Status> {
        Ok(Response::new(self.info.clone()))
    }
}

/// A fake worker running on its own socket for the duration of a test.
struct Harness {
    socket_path: PathBuf,
    received: Arc<Mutex<Vec<marceline_protocol::common::AudioChunk>>>,
    saw_cancel: Arc<Mutex<bool>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Harness {
    async fn start(name: &str, behavior: Behavior) -> Self {
        Self::start_with_info(
            name,
            behavior,
            SttInfo {
                name: "whisper:openai/whisper-large-v3".to_string(),
                langs: vec!["en".to_string()],
                input_sample_rate: 16_000,
                partials: false,
            },
        )
        .await
    }

    async fn start_with_info(name: &str, behavior: Behavior, info: SttInfo) -> Self {
        let socket_path = unique_socket_path(name);
        let _ = std::fs::remove_file(&socket_path);

        let received = Arc::new(Mutex::new(Vec::new()));
        let saw_cancel = Arc::new(Mutex::new(false));
        let worker = FakeWorker {
            behavior,
            info,
            received: Arc::clone(&received),
            saw_cancel: Arc::clone(&saw_cancel),
        };

        let listener = UnixListener::bind(&socket_path).expect("bind test socket");
        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(SttServer::new(worker))
                .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        Self {
            socket_path,
            received,
            saw_cancel,
            shutdown: Some(shutdown),
        }
    }

    fn path(&self) -> &Path {
        &self.socket_path
    }

    fn received_chunks(&self) -> Vec<marceline_protocol::common::AudioChunk> {
        self.received.lock().unwrap().clone()
    }

    fn saw_cancel(&self) -> bool {
        *self.saw_cancel.lock().unwrap()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Builds an `AudioStream` of `count` chunks of `samples` each.
fn audio_stream(count: u64, samples: usize) -> AudioStream {
    Box::pin(futures::stream::iter((0..count).map(move |seq| {
        Ok(AudioChunk {
            seq,
            pcm: vec![0.05; samples],
            sample_rate: 16_000,
            channels: 1,
        })
    })))
}

#[tokio::test]
async fn transcribe_streams_audio_and_yields_a_final_transcript() {
    let harness = Harness::start(
        "final",
        Behavior::FinalAfterHalfClose {
            text: "what time is it".to_string(),
            confidence: 0.82,
        },
    )
    .await;

    let engine = GrpcSttEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");

    let items: Vec<_> = engine.transcribe(audio_stream(3, 160)).await.collect().await;

    assert_eq!(items.len(), 1);
    match items[0].as_ref().expect("transcript item") {
        Transcript::Final { text, confidence } => {
            assert_eq!(text, "what time is it");
            assert!((confidence - 0.82).abs() < 1e-6);
        }
        other => panic!("expected a final transcript, got {other:?}"),
    }

    // The audio really crossed the wire, self-describing (invariant 2).
    let chunks = harness.received_chunks();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].seq, 0);
    assert_eq!(chunks[0].pcm.len(), 160);
    assert_eq!(chunks[0].sample_rate, 16_000);
    assert_eq!(chunks[0].channels, 1);
}

#[tokio::test]
async fn info_reports_the_model_name_and_no_partials() {
    let harness = Harness::start(
        "info",
        Behavior::FinalAfterHalfClose {
            text: String::new(),
            confidence: 0.0,
        },
    )
    .await;

    let engine = GrpcSttEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");
    let info = engine.info();

    assert_eq!(info.name, "whisper:openai/whisper-large-v3");
    assert_eq!(info.langs, vec!["en".to_string()]);
    assert_eq!(info.input_sample_rate, 16_000);
    assert!(
        !info.partials,
        "v1 must advertise final-only transcription (§2.4.1)"
    );
}

#[tokio::test]
async fn maps_partials_when_a_backend_advertises_them() {
    // No v1 backend emits partials, but the client must not drop or
    // mislabel them the day one does — that is what `partials` on
    // `SttInfo` is for.
    let harness = Harness::start_with_info(
        "partials",
        Behavior::PartialThenFinal {
            partial: "what tim".to_string(),
            text: "what time is it".to_string(),
        },
        SttInfo {
            name: "fake:partial-capable".to_string(),
            langs: vec!["en".to_string()],
            input_sample_rate: 16_000,
            partials: true,
        },
    )
    .await;

    let engine = GrpcSttEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");

    assert!(engine.info().partials);
    let items: Vec<_> = engine.transcribe(audio_stream(1, 160)).await.collect().await;
    let transcripts: Vec<_> = items.into_iter().map(|item| item.unwrap()).collect();

    assert_eq!(
        transcripts,
        vec![
            Transcript::Partial("what tim".to_string()),
            Transcript::Final {
                text: "what time is it".to_string(),
                confidence: 1.0,
            },
        ]
    );
    // Only `Final` is forwardable to the LLM.
    let committed: Vec<_> = transcripts
        .iter()
        .filter_map(Transcript::final_text)
        .collect();
    assert_eq!(committed, vec!["what time is it"]);
}

#[tokio::test]
async fn firing_the_run_token_sends_an_explicit_cancel_to_the_worker() {
    // The heart of §2.5.1: cancellation is a *message*, not a dropped
    // socket, because a dropped socket does not stop a GPU kernel.
    let harness = Harness::start("cancel", Behavior::AwaitCancel).await;
    let cancel = CancellationToken::new();

    let engine = GrpcSttEngine::connect(harness.path(), cancel.clone())
        .await
        .expect("connect to fake worker");

    // An audio stream that never ends on its own, so the only way this
    // transcription terminates is the cancel.
    let audio: AudioStream = Box::pin(futures::stream::unfold(0u64, |seq| async move {
        // Yield to the runtime so the cancel arm of the request stream's
        // select gets a chance to win.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        Some((
            Ok(AudioChunk {
                seq,
                pcm: vec![0.0; 160],
                sample_rate: 16_000,
                channels: 1,
            }),
            seq + 1,
        ))
    }));

    let mut transcripts = engine.transcribe(audio).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancel.cancel();

    // The stream must actually end — a cancel that leaves the caller
    // awaiting forever is worse than no cancel at all — and it must end
    // with cancellation rather than a transcript or a fault.
    let remaining: Vec<Result<Transcript, EngineError>> =
        tokio::time::timeout(std::time::Duration::from_secs(5), transcripts.by_ref().collect())
            .await
            .expect("transcript stream must terminate after cancel");

    assert_eq!(
        remaining.len(),
        1,
        "expected exactly one cancellation item, got {remaining:?}"
    );
    let err = remaining[0].as_ref().expect_err("expected cancellation");
    assert!(err.is_cancelled(), "got {err:?}");
    assert!(
        harness.saw_cancel(),
        "the worker never received an explicit Cancel message"
    );
}

#[tokio::test]
async fn a_transcript_committed_after_cancel_is_discarded() {
    // §2.5.1 partial-state policy: a cancelled turn's output is not used.
    // The worker may already have committed a `final` before it noticed
    // our cancel; that transcript must never reach the caller, or
    // Marceline answers a question the user interrupted.
    let harness = Harness::start(
        "cancel-race",
        Behavior::FinalDespiteCancel {
            text: "answering something you no longer asked".to_string(),
        },
    )
    .await;
    let cancel = CancellationToken::new();

    let engine = GrpcSttEngine::connect(harness.path(), cancel.clone())
        .await
        .expect("connect to fake worker");

    let audio: AudioStream = Box::pin(futures::stream::unfold(0u64, |seq| async move {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        Some((
            Ok(AudioChunk {
                seq,
                pcm: vec![0.0; 160],
                sample_rate: 16_000,
                channels: 1,
            }),
            seq + 1,
        ))
    }));

    let transcripts = engine.transcribe(audio).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancel.cancel();

    let items: Vec<Result<Transcript, EngineError>> =
        tokio::time::timeout(std::time::Duration::from_secs(5), transcripts.collect())
            .await
            .expect("transcript stream must terminate after cancel");

    assert!(
        items.iter().all(|item| item.is_err()),
        "a transcript committed after cancel leaked to the caller: {items:?}"
    );
    assert!(items
        .iter()
        .all(|item| item.as_ref().err().is_some_and(EngineError::is_cancelled)));
}

#[tokio::test]
async fn worker_failure_arrives_in_band_as_a_stream_error() {
    // Invariant 1 (§2.4.1): a worker OOM mid-stream must surface as an
    // `Err` item, not as a stream that quietly ends looking successful.
    let harness = Harness::start(
        "oom",
        Behavior::FailMidStream(tonic::Code::Internal, "CUDA out of memory".to_string()),
    )
    .await;

    let engine = GrpcSttEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");
    let items: Vec<_> = engine.transcribe(audio_stream(2, 160)).await.collect().await;

    assert_eq!(items.len(), 1);
    let err = items[0].as_ref().expect_err("expected an in-band error");
    assert!(matches!(err, EngineError::Worker { .. }), "got {err:?}");
    assert!(err.to_string().contains("CUDA out of memory"));
    assert!(!err.is_cancelled());
}

#[tokio::test]
async fn client_protocol_violations_are_distinguished_from_worker_faults() {
    // The worker rejects what *we* sent (a format change mid-stream, say)
    // with INVALID_ARGUMENT. That is our bug, not a model failure, and the
    // error type has to say so.
    let harness = Harness::start(
        "invalid",
        Behavior::FailMidStream(
            tonic::Code::InvalidArgument,
            "audio format changed mid-stream".to_string(),
        ),
    )
    .await;

    let engine = GrpcSttEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");
    let items: Vec<_> = engine.transcribe(audio_stream(1, 160)).await.collect().await;

    let err = items[0].as_ref().expect_err("expected an in-band error");
    assert!(matches!(err, EngineError::Protocol { .. }), "got {err:?}");
}

#[tokio::test]
async fn connecting_to_a_missing_worker_fails_as_transport() {
    // Nothing is listening: the supervisor (EPIC 0.6), not the turn, is
    // what fixes this, so it must not look like a model failure.
    let missing = unique_socket_path("absent");
    let err = GrpcSttEngine::connect(&missing, CancellationToken::new())
        .await
        .expect_err("connecting to a missing socket must fail");

    assert!(matches!(err, EngineError::Transport { .. }), "got {err:?}");
    assert!(!err.is_cancelled());
}
