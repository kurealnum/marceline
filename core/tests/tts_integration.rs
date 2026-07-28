//! Integration tests for the gRPC TTS client backend (EPIC 5.2).
//!
//! These drive [`GrpcTtsEngine`] against the fake `Tts` server in
//! `tts_common`, over a real unix domain socket.

mod tts_common;

use futures::StreamExt;
use marceline_core::engine::EngineError;
use marceline_core::tts::{GrpcTtsEngine, TextStream, TtsEngine, VoiceId};
use tokio_util::sync::CancellationToken;
use tts_common::{unique_socket_path, Behavior, Harness};

/// Builds a `TextStream` of the given spans.
fn text_stream(spans: &[&str]) -> TextStream {
    let owned: Vec<String> = spans.iter().map(|s| s.to_string()).collect();
    Box::pin(futures::stream::iter(owned.into_iter().map(Ok)))
}

#[tokio::test]
async fn synthesize_streams_text_and_yields_audio_chunks() {
    let harness = Harness::start(
        "final",
        Behavior::ChunksAfterHalfClose {
            chunks: 3,
            samples: 480,
        },
    )
    .await;

    let engine = GrpcTtsEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");

    let items: Vec<_> = engine
        .synthesize(text_stream(&["hello there"]), VoiceId::from("af_sky"))
        .await
        .collect()
        .await;

    assert_eq!(items.len(), 3);
    for (i, item) in items.iter().enumerate() {
        let chunk = item.as_ref().expect("audio chunk");
        assert_eq!(chunk.seq, i as u64);
        assert_eq!(chunk.pcm.len(), 480);
        assert_eq!(chunk.sample_rate, 24_000);
        assert_eq!(chunk.channels, 1);
    }

    // The text and the selected voice really crossed the wire.
    let received = harness.received_text();
    assert_eq!(received, vec![("hello there".to_string(), "af_sky".to_string())]);
}

#[tokio::test]
async fn multiple_text_spans_all_carry_the_selected_voice() {
    let harness = Harness::start(
        "multi-span",
        Behavior::ChunksAfterHalfClose {
            chunks: 0,
            samples: 0,
        },
    )
    .await;

    let engine = GrpcTtsEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");

    let _: Vec<_> = engine
        .synthesize(text_stream(&["first", "second"]), VoiceId::from("am_adam"))
        .await
        .collect()
        .await;

    assert_eq!(
        harness.received_text(),
        vec![
            ("first".to_string(), "am_adam".to_string()),
            ("second".to_string(), "am_adam".to_string()),
        ]
    );
}

#[tokio::test]
async fn info_reports_the_model_name_voices_and_sample_rate() {
    let harness = Harness::start(
        "info",
        Behavior::ChunksAfterHalfClose {
            chunks: 0,
            samples: 0,
        },
    )
    .await;

    let engine = GrpcTtsEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");
    let info = engine.info();

    assert_eq!(info.name, "kokoro:82M");
    assert_eq!(
        info.voices,
        vec!["af_sky".to_string(), "am_adam".to_string()]
    );
    assert_eq!(info.output_sample_rate, 24_000);
}

#[tokio::test]
async fn firing_the_run_token_sends_an_explicit_cancel_to_the_worker() {
    // The heart of §2.5.1: cancellation is a *message*, not a dropped
    // socket, because a dropped socket does not stop synthesis in progress.
    let harness = Harness::start("cancel", Behavior::AwaitCancel).await;
    let cancel = CancellationToken::new();

    let engine = GrpcTtsEngine::connect(harness.path(), cancel.clone())
        .await
        .expect("connect to fake worker");

    // A text stream that never ends on its own, so the only way this
    // synthesis terminates is the cancel.
    let text: TextStream = Box::pin(futures::stream::unfold(0u64, |seq| async move {
        // Yield to the runtime so the cancel arm of the request stream's
        // select gets a chance to win.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        Some((Ok(format!("span {seq}")), seq + 1))
    }));

    let mut audio = engine.synthesize(text, VoiceId::from("af_sky")).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancel.cancel();

    // The stream must actually end — a cancel that leaves the caller
    // awaiting forever is worse than no cancel at all — and it must end
    // with cancellation rather than audio or a fault.
    let remaining: Vec<Result<_, EngineError>> =
        tokio::time::timeout(std::time::Duration::from_secs(5), audio.by_ref().collect())
            .await
            .expect("audio stream must terminate after cancel");

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
async fn audio_produced_after_cancel_is_discarded() {
    // §2.5.1 partial-state policy: a cancelled turn's output is not used.
    // The worker may already have produced a chunk before it noticed our
    // cancel; that audio must never reach the caller, or Marceline keeps
    // speaking an answer the user interrupted.
    let harness = Harness::start("cancel-race", Behavior::ChunkDespiteCancel).await;
    let cancel = CancellationToken::new();

    let engine = GrpcTtsEngine::connect(harness.path(), cancel.clone())
        .await
        .expect("connect to fake worker");

    let text: TextStream = Box::pin(futures::stream::unfold(0u64, |seq| async move {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        Some((Ok(format!("span {seq}")), seq + 1))
    }));

    let audio = engine.synthesize(text, VoiceId::from("af_sky")).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancel.cancel();

    let items: Vec<Result<_, EngineError>> =
        tokio::time::timeout(std::time::Duration::from_secs(5), audio.collect())
            .await
            .expect("audio stream must terminate after cancel");

    assert!(
        items.iter().all(|item| item.is_err()),
        "audio produced after cancel leaked to the caller: {items:?}"
    );
    assert!(items
        .iter()
        .all(|item| item.as_ref().err().is_some_and(EngineError::is_cancelled)));
}

#[tokio::test]
async fn worker_failure_arrives_in_band_as_a_stream_error() {
    // Invariant 1 (§2.4.1): a worker fault mid-stream must surface as an
    // `Err` item, not as a stream that quietly ends looking successful.
    let harness = Harness::start(
        "fault",
        Behavior::FailMidStream(tonic::Code::Internal, "model exploded".to_string()),
    )
    .await;

    let engine = GrpcTtsEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");
    let items: Vec<_> = engine
        .synthesize(text_stream(&["hi"]), VoiceId::from("af_sky"))
        .await
        .collect()
        .await;

    assert_eq!(items.len(), 1);
    let err = items[0].as_ref().expect_err("expected an in-band error");
    assert!(matches!(err, EngineError::Worker { .. }), "got {err:?}");
    assert!(err.to_string().contains("model exploded"));
    assert!(!err.is_cancelled());
}

#[tokio::test]
async fn client_protocol_violations_are_distinguished_from_worker_faults() {
    // The worker rejects what *we* sent with INVALID_ARGUMENT. That is our
    // bug, not a model failure, and the error type has to say so.
    let harness = Harness::start(
        "invalid",
        Behavior::FailMidStream(
            tonic::Code::InvalidArgument,
            "unknown voice id".to_string(),
        ),
    )
    .await;

    let engine = GrpcTtsEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker");
    let items: Vec<_> = engine
        .synthesize(text_stream(&["hi"]), VoiceId::from("af_sky"))
        .await
        .collect()
        .await;

    let err = items[0].as_ref().expect_err("expected an in-band error");
    assert!(matches!(err, EngineError::Protocol { .. }), "got {err:?}");
}

#[tokio::test]
async fn connecting_to_a_missing_worker_fails_as_transport() {
    // Nothing is listening: the supervisor (EPIC 0.6), not the turn, is
    // what fixes this, so it must not look like a model failure.
    let missing = unique_socket_path("absent");
    let err = GrpcTtsEngine::connect(&missing, CancellationToken::new())
        .await
        .expect_err("connecting to a missing socket must fail");

    assert!(matches!(err, EngineError::Transport { .. }), "got {err:?}");
    assert!(!err.is_cancelled());
}
