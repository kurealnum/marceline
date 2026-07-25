//! Shared test scaffolding: a fake `marceline.stt.Stt` server.
//!
//! Used by the STT client tests (EPIC 3.2) and the transcribe-pipeline
//! tests (EPIC 3.3). It runs on a real unix domain socket, because the
//! transport and the bidirectional streaming shape are most of what the
//! code under test *is* — what is faked here is the model behind the
//! contract, not the contract.
//!
//! The Python worker's own behavior is covered by its `unittest` suite
//! (`workers/stt/tests`).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use marceline_protocol::stt::stt_server::{Stt, SttServer};
use marceline_protocol::stt::{
    stt_request, stt_response, FinalTranscript, SttInfo, SttInfoRequest, SttRequest, SttResponse,
};
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{Request, Response, Status, Streaming};

/// Counter making each test's socket path unique within this process.
static SOCKET_SEQ: AtomicU32 = AtomicU32::new(0);

pub fn unique_socket_path(name: &str) -> PathBuf {
    let n = SOCKET_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "marceline-stt-test-{}-{n}-{name}.sock",
        std::process::id()
    ))
}

/// What a fake worker should do when it gets a `Transcribe` stream.
#[derive(Clone)]
pub enum Behavior {
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
    /// Emit several `final`s, as a backend does when a segment is long
    /// enough that it splits into more than one window.
    MultipleFinals(Vec<(String, f32)>),
    /// Accept the audio and then never answer, modelling a wedged worker.
    /// Nothing but a timeout gets the caller out of this.
    Hang,
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
                Behavior::MultipleFinals(finals) => {
                    for (text, confidence) in finals {
                        let _ = tx
                            .send(Ok(SttResponse {
                                transcript: Some(stt_response::Transcript::Final(
                                    FinalTranscript { text, confidence },
                                )),
                            }))
                            .await;
                    }
                }
                // Hold the response stream open forever without answering.
                // Dropping `tx` here instead would end the stream, which is
                // the opposite of the wedged-worker case under test.
                Behavior::Hang => std::future::pending::<()>().await,
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
pub struct Harness {
    socket_path: PathBuf,
    received: Arc<Mutex<Vec<marceline_protocol::common::AudioChunk>>>,
    saw_cancel: Arc<Mutex<bool>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Harness {
    pub async fn start(name: &str, behavior: Behavior) -> Self {
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

    pub async fn start_with_info(name: &str, behavior: Behavior, info: SttInfo) -> Self {
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

    pub fn path(&self) -> &Path {
        &self.socket_path
    }

    pub fn received_chunks(&self) -> Vec<marceline_protocol::common::AudioChunk> {
        self.received.lock().unwrap().clone()
    }

    pub fn saw_cancel(&self) -> bool {
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

