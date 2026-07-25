//! Integration tests for the TRANSCRIBING stage (EPIC 3.3).
//!
//! The point of these is the *join*: a segment that the real [`Gate`] state
//! machine emitted, streamed to a worker over a real socket, coming back as
//! committed text. Unit tests either side of that seam can both pass while
//! the seam itself is wrong, which is the failure mode worth spending an
//! integration test on.
//!
//! They also pin the error edge (§2.5): worker-down, worker-fails, and
//! worker-hangs must each surface an error rather than a silent stall.

mod common;

use std::time::Duration;

use common::{unique_socket_path, Behavior, FakeSignals, Harness};
use marceline_core::audio::AudioChunk;
use marceline_core::config::{VadConfig, WakeConfig};
use marceline_core::engine::EngineError;
use marceline_core::stt::GrpcSttEngine;
use marceline_core::stt::{GuardConfig, Rejection, SpeechGuard};
use marceline_core::transcribe::{transcribe_segment, transcribe_segment_guarded, DEFAULT_TIMEOUT};
use marceline_core::{
    EnergyWakeDetector, Gate, GateOutput, SileroVad, VadEndpointer, WakeEngine,
    DEFAULT_SPEECH_THRESHOLD,
};
use tokio_util::sync::CancellationToken;

/// Capture format the gate tests use: 16 kHz mono, matching the mic path.
const SAMPLE_RATE: u32 = 16_000;

fn chunk(pcm: Vec<f32>) -> AudioChunk {
    AudioChunk {
        seq: 0,
        pcm,
        sample_rate: SAMPLE_RATE,
        channels: 1,
    }
}

fn silence(len: usize) -> AudioChunk {
    chunk(vec![0.0; len])
}

/// Loud tone the placeholder energy wake detector fires on.
fn loud_tone(len: usize) -> AudioChunk {
    chunk((0..len).map(|i| 0.9 * (i as f32 * 0.3).sin()).collect())
}

/// Builds the same gate the EPIC 2.3 tests drive: real Silero VAD behind
/// the endpointer, placeholder energy wake detector.
fn build_gate() -> Gate {
    let vad_config = VadConfig {
        silence_ms: 700,
        min_utterance_ms: 300,
        max_utterance_ms: 15_000,
    };
    let wake_config = WakeConfig {
        words: vec!["marceline".to_string()],
        sensitivity: 0.6,
    };
    let detector = EnergyWakeDetector::new(wake_config.sensitivity, SAMPLE_RATE, 320);
    let wake = WakeEngine::new(&wake_config, Box::new(detector));
    let model_path = format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"));
    let vad = SileroVad::load(model_path).expect("failed to load Silero VAD model");
    Gate::new(
        wake,
        VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD),
        &vad_config,
    )
}

/// The real speech fixture the VAD tests use — a tone fires the energy wake
/// detector but Silero will not call it speech, so an actual utterance is
/// required to get the gate all the way to emitting.
fn load_speech_sample() -> Vec<f32> {
    let fixture = format!(
        "{}/tests/fixtures/speech_sample.wav",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut reader = hound::WavReader::open(&fixture).expect("failed to open speech fixture");
    reader
        .samples::<i16>()
        .map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32)
        .collect()
}

/// Drives the real gate through wake -> speech -> trailing silence and
/// returns the segment it emits, exactly as a live utterance would.
fn gate_emitted_segment() -> AudioChunk {
    let mut gate = build_gate();
    let preroll = chunk(vec![0.42; 1_600]);

    // Sustained loud audio fires the placeholder wake detector.
    let mut fired = false;
    for _ in 0..20 {
        if let GateOutput::Wake = gate.process_chunk(&loud_tone(320), &preroll) {
            fired = true;
            break;
        }
    }
    assert!(fired, "sustained loud audio should fire wake");

    // A continuous slice of real speech, short of the silence threshold.
    let speech = load_speech_sample();
    for frame in speech[..24_000.min(speech.len())].chunks(1_600) {
        match gate.process_chunk(&chunk(frame.to_vec()), &preroll) {
            GateOutput::None => {}
            other => panic!("unexpected gate output during speech: {other:?}"),
        }
    }

    // Trailing silence ends the utterance.
    for _ in 0..100 {
        if let GateOutput::Segment(segment) = gate.process_chunk(&silence(1_600), &preroll) {
            return segment;
        }
    }
    panic!("gate never emitted a segment");
}

async fn engine_for(harness: &Harness) -> GrpcSttEngine {
    GrpcSttEngine::connect(harness.path(), CancellationToken::new())
        .await
        .expect("connect to fake worker")
}

#[tokio::test]
async fn a_gate_emitted_segment_becomes_a_final_transcript() {
    let harness = Harness::start(
        "pipeline",
        Behavior::FinalAfterHalfClose {
            text: "  marceline what time is it  ".to_string(),
            confidence: 0.91,
        },
    )
    .await;

    let segment = gate_emitted_segment();
    assert!(
        !segment.pcm.is_empty(),
        "the gate must emit actual audio to transcribe"
    );

    let engine = engine_for(&harness).await;
    let transcription = transcribe_segment(&engine, segment.clone(), DEFAULT_TIMEOUT)
        .await
        .expect("transcription should succeed")
        .committed()
        .cloned()
        .expect("a genuine utterance must not be rejected");

    // Text is trimmed on the way out; a leading space in a prompt is noise.
    assert_eq!(transcription.text, "marceline what time is it");
    assert!((transcription.confidence - 0.91).abs() < 1e-6);
    assert_eq!(transcription.segments, 1);
    assert!(!transcription.is_empty());

    // The whole segment reached the worker, sliced but not lost.
    let received: usize = harness
        .received_chunks()
        .iter()
        .map(|chunk| chunk.pcm.len())
        .sum();
    assert_eq!(received, segment.pcm.len());

    // Chunks arrive in order, each self-describing (invariant 2).
    let chunks = harness.received_chunks();
    assert!(chunks.len() > 1, "the segment should be streamed, not sent whole");
    assert_eq!(
        chunks.iter().map(|chunk| chunk.seq).collect::<Vec<_>>(),
        (0..chunks.len() as u64).collect::<Vec<u64>>()
    );
    assert!(chunks
        .iter()
        .all(|chunk| chunk.sample_rate == SAMPLE_RATE && chunk.channels == 1));
}

#[tokio::test]
async fn only_final_transcripts_reach_the_caller() {
    // A partials-capable backend's revisable text must not end up in the
    // committed transcript — that is how half-words leak into an LLM prompt
    // (§2.4.1).
    let harness = Harness::start_with_info(
        "partials-dropped",
        Behavior::PartialThenFinal {
            partial: "what tim".to_string(),
            text: "what time is it".to_string(),
        },
        marceline_protocol::stt::SttInfo {
            name: "fake:partial-capable".to_string(),
            langs: vec!["en".to_string()],
            input_sample_rate: SAMPLE_RATE,
            partials: true,
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    let transcription = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect("transcription should succeed")
        .committed()
        .cloned()
        .expect("should not be rejected");

    assert_eq!(transcription.text, "what time is it");
    assert_eq!(transcription.segments, 1, "the partial must not count as a segment");
}

#[tokio::test]
async fn a_worker_failure_mid_transcription_surfaces_as_an_error() {
    // The `Done when` case: an error, not a silent hang.
    let harness = Harness::start(
        "mid-stream-failure",
        Behavior::FailMidStream(tonic::Code::Internal, "CUDA out of memory".to_string()),
    )
    .await;

    let engine = engine_for(&harness).await;
    let err = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect_err("a failing worker must produce an error");

    assert!(matches!(err, EngineError::Worker { .. }), "got {err:?}");
    assert!(err.to_string().contains("CUDA out of memory"));
}

#[tokio::test]
async fn a_worker_that_is_not_running_surfaces_as_an_error() {
    let missing = unique_socket_path("no-worker");
    let err = GrpcSttEngine::connect(&missing, CancellationToken::new())
        .await
        .expect_err("connecting with no worker running must fail");

    assert!(matches!(err, EngineError::Transport { .. }), "got {err:?}");
}

#[tokio::test]
async fn a_wedged_worker_times_out_instead_of_hanging() {
    // A worker that accepts the audio and then never answers is the case a
    // naive implementation hangs on forever. The stage must give up.
    let harness = Harness::start("wedged", Behavior::Hang).await;

    let engine = engine_for(&harness).await;
    let started = std::time::Instant::now();
    let err = transcribe_segment(
        &engine,
        silence(SAMPLE_RATE as usize),
        Duration::from_millis(300),
    )
    .await
    .expect_err("a wedged worker must time out");

    assert!(matches!(err, EngineError::Timeout { .. }), "got {err:?}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the timeout should fire promptly, took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn silence_is_rejected_rather_than_treated_as_an_engine_failure() {
    // A worker that heard nothing and says so is not a fault — it is the
    // "nothing to answer" case, which routes through the ERROR edge (§2.5)
    // rather than surfacing as a broken worker.
    let harness = Harness::start(
        "no-speech",
        Behavior::FinalAfterHalfClose {
            text: String::new(),
            confidence: 0.0,
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    let outcome = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect("empty speech is not an engine error");

    // Silence correctly recognized as silence routes through the same
    // empty-transcript ERROR edge as a rejection (§2.5): either way there is
    // nothing for the LLM to answer.
    assert_eq!(outcome.rejection(), Some(Rejection::Empty));
    assert!(outcome.committed().is_none());
}

#[tokio::test]
async fn multiple_final_segments_are_joined_and_scored_conservatively() {
    // A long segment can come back as several finals; the text joins in
    // order and the confidence reports the worst part, not the average.
    let harness = Harness::start(
        "multi-final",
        Behavior::MultipleFinals(vec![
            ("what time".to_string(), 0.9),
            ("is it".to_string(), 0.4),
        ]),
    )
    .await;

    let engine = engine_for(&harness).await;
    let transcription = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect("transcription should succeed")
        .committed()
        .cloned()
        .expect("should not be rejected");

    assert_eq!(transcription.text, "what time is it");
    assert_eq!(transcription.segments, 2);
    assert!((transcription.confidence - 0.4).abs() < 1e-6);
}

#[tokio::test]
async fn a_preexisting_wav_file_transcribes_the_same_way() {
    // Backs `marceline transcribe <file>` (EPIC 11.4): a file stands in for
    // a gate segment, so the demo path and the live path share one code
    // path rather than drifting.
    let fixture = format!(
        "{}/tests/fixtures/speech_sample.wav",
        env!("CARGO_MANIFEST_DIR")
    );
    let segment = marceline_core::read_wav(std::path::Path::new(&fixture)).expect("read fixture");

    let harness = Harness::start(
        "wav",
        Behavior::FinalAfterHalfClose {
            text: "transcribed from a file".to_string(),
            confidence: 0.7,
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    let transcription = transcribe_segment(&engine, segment.clone(), DEFAULT_TIMEOUT)
        .await
        .expect("transcription should succeed")
        .committed()
        .cloned()
        .expect("should not be rejected");

    assert_eq!(transcription.text, "transcribed from a file");
    let received: usize = harness
        .received_chunks()
        .iter()
        .map(|chunk| chunk.pcm.len())
        .sum();
    assert_eq!(received, segment.pcm.len());
}

#[tokio::test]
async fn a_hallucinated_transcript_on_near_silence_is_rejected() {
    // The story's first `Done when`, and the failure mode this guard exists
    // for: Whisper answers near-silence with fluent, plausible text — its
    // training data's most common filler. The text alone looks fine; only
    // `no_speech_prob` gives it away. This must never reach the LLM, or
    // Marceline says "Thank you for watching!" out loud unprompted.
    let harness = Harness::start(
        "hallucination",
        Behavior::FinalWithSignals {
            text: "Thank you for watching!".to_string(),
            signals: FakeSignals {
                no_speech_prob: Some(0.94),
                avg_logprob: Some(-0.3),
            },
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    let outcome = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect("a rejection is not an engine error");

    assert!(
        outcome.committed().is_none(),
        "a hallucination must not be committed: {outcome:?}"
    );
    let rejection = outcome.rejection().expect("expected a rejection");
    assert!(matches!(rejection, Rejection::NoSpeech { .. }), "got {rejection:?}");
    // The reason has to name the measurement, or tuning (EPIC 8.3) is blind.
    assert!(rejection.reason().contains("0.94"), "{}", rejection.reason());
}

#[tokio::test]
async fn a_genuine_utterance_passes_the_guard_unaffected() {
    // The story's second `Done when`. Same code path, real signals.
    let harness = Harness::start(
        "genuine",
        Behavior::FinalWithSignals {
            text: "marceline what time is it".to_string(),
            signals: FakeSignals {
                no_speech_prob: Some(0.02),
                avg_logprob: Some(-0.18),
            },
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    let transcription = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect("transcription should succeed")
        .committed()
        .cloned()
        .expect("a genuine utterance must not be rejected");

    assert_eq!(transcription.text, "marceline what time is it");
    // Signals survive onto the committed transcription, so a turn that
    // nearly tripped the guard is visible downstream.
    assert_eq!(transcription.signals.no_speech_prob, Some(0.02));
    assert_eq!(transcription.signals.avg_logprob, Some(-0.18));
}

#[tokio::test]
async fn a_low_confidence_transcript_is_rejected() {
    let harness = Harness::start(
        "low-confidence",
        Behavior::FinalWithSignals {
            text: "shuffling papers and a door".to_string(),
            signals: FakeSignals {
                no_speech_prob: Some(0.2),
                avg_logprob: Some(-2.8),
            },
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    let outcome = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect("a rejection is not an engine error");

    assert!(matches!(
        outcome.rejection(),
        Some(Rejection::LowConfidence { .. })
    ));
}

#[tokio::test]
async fn a_backend_reporting_no_signals_is_still_trusted() {
    // The HF whisper worker cannot report no_speech_prob. If missing signals
    // counted as failure, that backend's every transcript would be dropped —
    // so absence must not reject on its own. The duration check still applies.
    let harness = Harness::start(
        "no-signals",
        Behavior::FinalWithSignals {
            text: "what time is it".to_string(),
            signals: FakeSignals {
                no_speech_prob: None,
                avg_logprob: None,
            },
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    let transcription = transcribe_segment(&engine, silence(SAMPLE_RATE as usize), DEFAULT_TIMEOUT)
        .await
        .expect("transcription should succeed")
        .committed()
        .cloned()
        .expect("missing signals must not cause a rejection");

    assert_eq!(transcription.text, "what time is it");
    assert_eq!(transcription.signals.no_speech_prob, None);
}

#[tokio::test]
async fn a_too_short_segment_is_rejected_without_calling_the_backend() {
    // The duration check runs before inference, so a blip costs no GPU time.
    // Asserted by the worker never receiving any audio at all.
    let harness = Harness::start(
        "too-short",
        Behavior::FinalWithSignals {
            text: "you".to_string(),
            signals: FakeSignals::default(),
        },
    )
    .await;

    let engine = engine_for(&harness).await;
    // 100ms, under the 250ms floor.
    let outcome = transcribe_segment(&engine, silence(1_600), DEFAULT_TIMEOUT)
        .await
        .expect("a rejection is not an engine error");

    assert!(matches!(
        outcome.rejection(),
        Some(Rejection::TooShort { .. })
    ));
    assert!(
        harness.received_chunks().is_empty(),
        "no audio should have been sent to the worker"
    );
}

#[tokio::test]
async fn guard_thresholds_are_configurable() {
    // Tuning is EPIC 8.3's job, so the same audio and signals must be able
    // to pass or fail purely on config.
    let harness = Harness::start(
        "configurable",
        Behavior::FinalWithSignals {
            text: "borderline".to_string(),
            signals: FakeSignals {
                no_speech_prob: Some(0.5),
                avg_logprob: Some(-0.8),
            },
        },
    )
    .await;
    let engine = engine_for(&harness).await;
    let segment = silence(SAMPLE_RATE as usize);

    // Default thresholds accept it.
    let accepted = transcribe_segment(&engine, segment.clone(), DEFAULT_TIMEOUT)
        .await
        .expect("should not error");
    assert!(accepted.committed().is_some(), "{accepted:?}");

    // A stricter guard rejects the very same result.
    let strict = SpeechGuard::new(GuardConfig {
        min_speech_ms: 250,
        max_no_speech_prob: 0.3,
        min_avg_logprob: -1.0,
    });
    let rejected = transcribe_segment_guarded(&engine, segment, DEFAULT_TIMEOUT, strict)
        .await
        .expect("should not error");
    assert!(matches!(
        rejected.rejection(),
        Some(Rejection::NoSpeech { .. })
    ));
}
