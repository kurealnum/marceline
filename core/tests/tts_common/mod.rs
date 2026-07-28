//! Shared test scaffolding: a fake `marceline.tts.Tts` server.
//!
//! Used by the TTS client tests (EPIC 5.2). It runs on a real unix domain
//! socket, because the transport and the bidirectional streaming shape are
//! most of what the code under test *is* — what is faked here is the model
//! behind the contract, not the contract.
//!
//! The Python worker's own behavior is covered by its `unittest` suite
//! (`workers/tts/tests`, `python/marceline_worker/tests`).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use marceline_protocol::common::AudioChunk;
use marceline_protocol::tts::tts_server::{Tts, TtsServer};
use marceline_protocol::tts::{
    tts_request, TtsInfo, TtsInfoRequest, TtsRequest, TtsResponse,
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
        "marceline-tts-test-{}-{n}-{name}.sock",
        std::process::id()
    ))
}

/// Builds one `AudioChunk` response carrying `samples` at `sample_rate`.
fn audio_response(seq: u64, samples: usize, sample_rate: u32) -> TtsResponse {
    TtsResponse {
        audio: Some(AudioChunk {
            seq,
            pcm: vec![0.1; samples],
            sample_rate,
            channels: 1,
        }),
    }
}

/// What a fake worker should do when it gets a `Synthesize` stream.
#[derive(Clone)]
pub enum Behavior {
    /// Consume the request stream, then emit `chunks` audio chunks.
    ChunksAfterHalfClose { chunks: u64, samples: usize },
    /// Fail mid-stream with the given gRPC status, as a worker hitting a
    /// model fault mid-synthesis does.
    FailMidStream(tonic::Code, String),
    /// Wait for an explicit `Cancel` message and end the stream without
    /// emitting audio, as the real worker does on cancel.
    AwaitCancel,
    /// Emit a chunk even after being cancelled, modelling a worker that had
    /// already produced audio before it noticed the cancel.
    ChunkDespiteCancel,
    /// Accept the request and then never answer, modelling a wedged worker.
    Hang,
}

/// Fake `Tts` server standing in for the Python worker.
struct FakeWorker {
    behavior: Behavior,
    info: TtsInfo,
    /// `(text, voice-at-the-time)` for every text span received, so tests
    /// can assert what was actually sent and in which voice.
    received: Arc<Mutex<Vec<(String, String)>>>,
    /// Set once an explicit `Cancel` message arrives on the request stream.
    saw_cancel: Arc<Mutex<bool>>,
}

#[tonic::async_trait]
impl Tts for FakeWorker {
    type SynthesizeStream = ReceiverStream<Result<TtsResponse, Status>>;

    async fn synthesize(
        &self,
        request: Request<Streaming<TtsRequest>>,
    ) -> Result<Response<Self::SynthesizeStream>, Status> {
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

            let mut voice = String::new();
            while let Some(request) = inbound.next().await {
                let Ok(request) = request else { return };
                match request.payload {
                    Some(tts_request::Payload::Voice(v)) => voice = v,
                    Some(tts_request::Payload::Text(text)) => {
                        received.lock().unwrap().push((text, voice.clone()));
                    }
                    Some(tts_request::Payload::Cancel(_)) => {
                        *saw_cancel.lock().unwrap() = true;
                        if let Behavior::ChunkDespiteCancel = &behavior {
                            let _ = tx.send(Ok(audio_response(0, 4, 24_000))).await;
                        }
                        // Real worker behavior: stop, emit nothing more.
                        return;
                    }
                    None => return,
                }
            }

            match behavior {
                Behavior::ChunksAfterHalfClose { chunks, samples } => {
                    for seq in 0..chunks {
                        let _ = tx.send(Ok(audio_response(seq, samples, 24_000))).await;
                    }
                }
                // Hold the response stream open forever without answering.
                // Dropping `tx` here instead would end the stream, which is
                // the opposite of the wedged-worker case under test.
                Behavior::Hang => std::future::pending::<()>().await,
                // Half-close with no cancel: emit nothing, as the client
                // asked for nothing.
                Behavior::AwaitCancel | Behavior::ChunkDespiteCancel => {}
                Behavior::FailMidStream(..) => unreachable!("handled above"),
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn get_info(&self, _: Request<TtsInfoRequest>) -> Result<Response<TtsInfo>, Status> {
        Ok(Response::new(self.info.clone()))
    }
}

/// A fake worker running on its own socket for the duration of a test.
pub struct Harness {
    socket_path: PathBuf,
    received: Arc<Mutex<Vec<(String, String)>>>,
    saw_cancel: Arc<Mutex<bool>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Harness {
    pub async fn start(name: &str, behavior: Behavior) -> Self {
        Self::start_with_info(
            name,
            behavior,
            TtsInfo {
                name: "kokoro:82M".to_string(),
                voices: vec!["af_sky".to_string(), "am_adam".to_string()],
                output_sample_rate: 24_000,
            },
        )
        .await
    }

    pub async fn start_with_info(name: &str, behavior: Behavior, info: TtsInfo) -> Self {
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
                .add_service(TtsServer::new(worker))
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

    pub fn received_text(&self) -> Vec<(String, String)> {
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
