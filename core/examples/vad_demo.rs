//! Manual smoke test for Silero VAD endpointing (EPIC 2.2). Not part of
//! the crate's public surface; run with `cargo run -p core --example
//! vad_demo`. Talk near the mic to see the speech/silence signal move.
use std::time::{Duration, Instant};

use marceline_core::{Capture, SileroVad, VadEndpointer, DEFAULT_SPEECH_THRESHOLD};

fn main() {
    let model_path = format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"));
    let vad = SileroVad::load(&model_path).expect("failed to load Silero VAD model");
    let mut endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);

    let capture = Capture::start(1.5, None).expect("failed to start capture");

    println!("listening for 5s (need {} 16kHz-mono samples per VAD frame)", marceline_core::FRAME_SAMPLES);
    let deadline = Instant::now() + Duration::from_secs(5);
    // Resample capture-rate chunks down to a running 16kHz mono buffer,
    // then slice off fixed-size VAD frames as enough samples accumulate.
    let mut pending: Vec<f32> = Vec::new();
    while Instant::now() < deadline {
        let Ok(chunk) = capture.chunks().recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        let resampled = marceline_core::audio::resample::resample(
            &chunk.pcm,
            chunk.sample_rate,
            chunk.channels,
            16_000,
            1,
        );
        pending.extend(resampled);

        while pending.len() >= marceline_core::FRAME_SAMPLES {
            let frame: Vec<f32> = pending.drain(..marceline_core::FRAME_SAMPLES).collect();
            let prob = endpointer.process_frame(&frame).expect("inference failed");
            let marker = if endpointer.is_speaking() { "SPEECH" } else { "..." };
            println!(
                "{marker:6} prob={prob:.3} silence_ms={}",
                endpointer.silence_ms()
            );
        }
    }
}
