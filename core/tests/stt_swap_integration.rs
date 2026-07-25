//! Integration tests for STT model hot-swap (EPIC 3.4).
//!
//! These spawn a **real worker process** — a tiny Python script that speaks
//! the `Stt` contract and reports whatever `--model-id` it was handed. The
//! restart is the whole mechanism here: an in-process fake could not show
//! that publishing a new spec takes one process down and brings another up
//! on the new model, which is exactly the claim being tested.
//!
//! The script needs only `grpcio` and the generated stubs, no ML stack. If
//! `MARCELINE_TEST_PYTHON` is not set to such an interpreter, these tests
//! skip rather than fail: the suite must stay runnable on a box without the
//! worker venv built.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use marceline_core::audio::AudioChunk;
use marceline_core::config::SttConfig;
use marceline_core::device::Device;
use marceline_core::stt::{SttManager, SttWorkerPaths, SwapError};
use marceline_core::supervisor::{HealthView, WorkerState};
use tokio::sync::{watch, RwLock};
use tokio_util::sync::CancellationToken;

/// Counter keeping each test's socket path unique.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// Interpreter able to `import grpc` and `marceline_protocol`.
///
/// Set by the developer/CI to the STT worker's venv:
/// `MARCELINE_TEST_PYTHON=workers/stt/.venv/bin/python cargo test`.
fn test_python() -> Option<PathBuf> {
    let python = PathBuf::from(std::env::var_os("MARCELINE_TEST_PYTHON")?);
    python.exists().then_some(python)
}

/// Socket paths must stay well under the ~107 byte `sun_path` limit, so
/// these live directly in the temp dir with short names.
fn socket_path(name: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("mstt-swap-{}-{n}-{name}.sock", std::process::id()))
}

/// Writes the echo worker script and returns its path.
///
/// It answers `GetInfo` with its own `--model-id`, which is how a test can
/// tell *which* process is currently serving.
fn write_echo_worker(dir: &std::path::Path) -> PathBuf {
    let repo_python = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("python");
    let script = dir.join("echo_worker.py");
    let source = format!(
        r#"
import argparse, os, sys
from concurrent import futures

sys.path.insert(0, {repo_python:?})

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from marceline_protocol import stt_pb2, stt_pb2_grpc


class Echo(stt_pb2_grpc.SttServicer):
    def __init__(self, model_id):
        self._model_id = model_id

    def GetInfo(self, request, context):
        return stt_pb2.SttInfo(
            name=self._model_id, langs=["en"], input_sample_rate=16000, partials=False
        )

    def Transcribe(self, request_iterator, context):
        samples = 0
        for request in request_iterator:
            if request.WhichOneof("payload") == "audio":
                samples += len(request.audio.pcm)
        yield stt_pb2.SttResponse(
            final=stt_pb2.FinalTranscript(
                text=f"{{self._model_id}} heard {{samples}}", confidence=0.5
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    stt_pb2_grpc.add_SttServicer_to_server(Echo(args.model_id), server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    if os.path.exists(args.socket_path):
        os.unlink(args.socket_path)
    server.add_insecure_port(f"unix://{{args.socket_path}}")
    server.start()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("stt", health_pb2.HealthCheckResponse.SERVING)
    server.wait_for_termination()


main()
"#
    );
    std::fs::write(&script, source).expect("write echo worker");
    script
}

/// Everything one swap test needs, cleaned up on drop.
struct Fixture {
    dir: PathBuf,
    socket: PathBuf,
    health: HealthView,
    shutdown: watch::Sender<bool>,
}

impl Fixture {
    /// Stops the worker and waits for the supervisor to confirm it.
    ///
    /// Every test must call this. Relying on `Drop` alone leaks worker
    /// processes: the send happens, but the runtime can shut down before the
    /// supervisor task wakes to kill its child, leaving a stray Python
    /// process holding a socket — and the test binary's stdout — after the
    /// suite has finished.
    async fn shutdown(self) {
        let _ = self.shutdown.send(true);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while tokio::time::Instant::now() < deadline {
            if matches!(
                self.health.read().await.get("stt"),
                Some(&WorkerState::Stopped)
            ) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the supervisor never reported the worker stopped");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Backstop only; `shutdown` above is what actually reaps the child.
        let _ = self.shutdown.send(true);
        let _ = std::fs::remove_file(&self.socket);
        // `MARCELINE_KEEP_FIXTURE=1` leaves the generated worker script on
        // disk, so a failing test can be reproduced by hand.
        if std::env::var_os("MARCELINE_KEEP_FIXTURE").is_none() {
            let _ = std::fs::remove_dir_all(&self.dir);
        } else {
            eprintln!("keeping fixture at {}", self.dir.display());
        }
    }
}

fn fixture(name: &str) -> (Fixture, SttWorkerPaths, watch::Receiver<bool>) {
    let dir = std::env::temp_dir().join(format!("mstt-swap-dir-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let script = write_echo_worker(&dir);
    let socket = socket_path(name);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let paths = SttWorkerPaths {
        python: test_python().expect("checked by caller"),
        script,
        socket_path: socket.clone(),
    };
    (
        Fixture {
            dir,
            socket,
            health: Arc::new(RwLock::new(HashMap::new())),
            shutdown: shutdown_tx,
        },
        paths,
        shutdown_rx,
    )
}

fn stt_config(model: &str) -> SttConfig {
    SttConfig {
        backend: "whisper".to_string(),
        model: model.to_string(),
        device: Device::Cpu,
        lang: "en".to_string(),
    }
}

fn segment(samples: usize) -> AudioChunk {
    AudioChunk {
        seq: 0,
        pcm: vec![0.1; samples],
        sample_rate: 16_000,
        channels: 1,
    }
}

#[tokio::test]
async fn swapping_the_model_restarts_the_worker_on_the_new_id() {
    let Some(_python) = test_python() else {
        eprintln!("skipping: set MARCELINE_TEST_PYTHON to an interpreter with grpcio");
        return;
    };
    let (fixture, paths, shutdown_rx) = fixture("swap");

    let manager = SttManager::start(
        &stt_config("large-v3"),
        paths,
        Arc::clone(&fixture.health),
        shutdown_rx,
        CancellationToken::new(),
    )
    .await
    .expect("start the initial worker");

    // The worker reports the model id it was launched with, so this is
    // evidence about the running process rather than about config.
    assert_eq!(manager.info().await.name, "large-v3");
    let before = manager
        .transcribe(segment(1_600), Duration::from_secs(10))
        .await
        .expect("transcribe on the original model");
    assert_eq!(before.text, "large-v3 heard 1600");

    // The swap itself.
    let info = manager
        .swap_model("small.en", None)
        .await
        .expect("swap should succeed");
    assert_eq!(info.name, "small.en");
    assert_eq!(manager.info().await.name, "small.en");

    // And the client reconnected transparently: transcription works on the
    // new worker without the caller touching the connection.
    let after = manager
        .transcribe(segment(3_200), Duration::from_secs(10))
        .await
        .expect("transcribe on the swapped model");
    assert_eq!(after.text, "small.en heard 3200");

    assert_eq!(
        fixture.health.read().await.get("stt"),
        Some(&WorkerState::Up),
        "the worker should be healthy again after the swap"
    );

    drop(manager);
    fixture.shutdown().await;
}

#[tokio::test]
async fn swapping_to_the_same_model_does_not_restart() {
    let Some(_python) = test_python() else {
        eprintln!("skipping: set MARCELINE_TEST_PYTHON to an interpreter with grpcio");
        return;
    };
    let (fixture, paths, shutdown_rx) = fixture("noop");

    let manager = SttManager::start(
        &stt_config("large-v3"),
        paths,
        Arc::clone(&fixture.health),
        shutdown_rx,
        CancellationToken::new(),
    )
    .await
    .expect("start the initial worker");

    // A no-op swap must be cheap: restarting to load identical weights
    // would cost a minute of GPU time and interrupt nothing usefully.
    let started = std::time::Instant::now();
    let info = manager
        .swap_model("large-v3", None)
        .await
        .expect("a no-op swap should succeed");

    assert_eq!(info.name, "large-v3");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a no-op swap should not restart anything, took {:?}",
        started.elapsed()
    );

    drop(manager);
    fixture.shutdown().await;
}

#[tokio::test]
async fn an_attached_worker_cannot_be_swapped() {
    let Some(_python) = test_python() else {
        eprintln!("skipping: set MARCELINE_TEST_PYTHON to an interpreter with grpcio");
        return;
    };
    let (fixture, paths, shutdown_rx) = fixture("attached");

    // Launch one worker, then attach a *second* manager to its socket. The
    // attached manager does not own the process, so it must refuse rather
    // than pretend to restart something it cannot.
    let _owner = SttManager::start(
        &stt_config("large-v3"),
        paths.clone(),
        Arc::clone(&fixture.health),
        shutdown_rx,
        CancellationToken::new(),
    )
    .await
    .expect("start the worker");

    let attached = SttManager::attach(
        paths.socket_path.clone(),
        CancellationToken::new(),
        "en".to_string(),
    )
    .await
    .expect("attach to the running worker");

    let err = attached
        .swap_model("small.en", None)
        .await
        .expect_err("an attached manager must not claim to swap");
    assert!(matches!(err, SwapError::NotSupervised), "got {err:?}");

    drop(attached);
    drop(_owner);
    fixture.shutdown().await;
}
