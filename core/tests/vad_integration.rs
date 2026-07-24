//! Integration test for real Silero VAD inference (EPIC 2.2): loads the
//! vendored ONNX model and runs it against both silence and a real
//! speech sample, verifying the model actually discriminates the two
//! (unlike the wake-word placeholder, this is genuine acoustic inference).
#![allow(clippy::chunks_exact_to_as_chunks)]

use marceline_core::vad::model::FRAME_SAMPLES;
use marceline_core::{SileroVad, VadEndpointer, DEFAULT_SPEECH_THRESHOLD};

fn model_path() -> String {
    format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn silence_never_registers_as_speech() {
    let vad = SileroVad::load(model_path()).expect("failed to load Silero VAD model");
    let mut endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);

    let silence = vec![0.0f32; FRAME_SAMPLES];
    for _ in 0..50 {
        let prob = endpointer
            .process_frame(&silence)
            .expect("inference should succeed");
        assert!(prob < DEFAULT_SPEECH_THRESHOLD, "silence scored {prob}");
    }
    assert!(!endpointer.is_speaking());
    assert!(endpointer.silence_ms() > 0);
}

#[test]
fn real_speech_sample_is_mostly_detected_as_speech() {
    let vad = SileroVad::load(model_path()).expect("failed to load Silero VAD model");
    let mut endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);

    let fixture = format!("{}/tests/fixtures/speech_sample.wav", env!("CARGO_MANIFEST_DIR"));
    let mut reader = hound::WavReader::open(&fixture).expect("failed to open speech fixture");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.channels, 1);

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32)
        .collect();

    let mut speech_frames = 0u32;
    let mut total_frames = 0u32;
    let mut speech_onsets = 0u32;
    for frame in samples.chunks_exact(FRAME_SAMPLES) {
        endpointer.process_frame(frame).expect("inference should succeed");
        total_frames += 1;
        if endpointer.is_speaking() {
            speech_frames += 1;
        }
        if endpointer.speech_just_started() {
            speech_onsets += 1;
        }
    }

    let ratio = speech_frames as f32 / total_frames as f32;
    assert!(
        ratio > 0.5,
        "expected most of a continuous speech sample to be detected as speech, got ratio={ratio}"
    );
    assert!(speech_onsets >= 1, "expected at least one speech onset");
}

#[test]
fn silence_to_speech_transition_reports_onset_and_resets_silence_ms() {
    let vad = SileroVad::load(model_path()).expect("failed to load Silero VAD model");
    let mut endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);

    let silence = vec![0.0f32; FRAME_SAMPLES];
    for _ in 0..30 {
        endpointer.process_frame(&silence).expect("inference should succeed");
    }
    let silence_ms_before_speech = endpointer.silence_ms();
    assert!(silence_ms_before_speech > 0);
    assert!(!endpointer.speech_just_started());

    let fixture = format!(
        "{}/tests/fixtures/speech_sample.wav",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut reader = hound::WavReader::open(&fixture).expect("failed to open speech fixture");
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.expect("failed to read sample") as f32 / i16::MAX as f32)
        .collect();

    let mut onset_seen = false;
    for frame in samples.chunks_exact(FRAME_SAMPLES) {
        endpointer.process_frame(frame).expect("inference should succeed");
        if endpointer.speech_just_started() {
            onset_seen = true;
            assert_eq!(endpointer.silence_ms(), 0, "silence_ms resets on speech onset");
            break;
        }
    }
    assert!(onset_seen, "expected a speech onset after feeding real speech following silence");
}
