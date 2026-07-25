//! Integration tests for the gRPC STT client backend (EPIC 3.2).
//!
//! These drive [`GrpcSttEngine`] against the fake `Stt` server in
//! `common`, over a real unix domain socket.

mod common;

use common::{unique_socket_path, Behavior, Harness};
use futures::StreamExt;
use marceline_core::audio::AudioChunk;
use marceline_core::engine::{AudioStream, EngineError};
use marceline_core::stt::{GrpcSttEngine, SttEngine, Transcript};
use marceline_protocol::stt::SttInfo;
use tokio_util::sync::CancellationToken;

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
