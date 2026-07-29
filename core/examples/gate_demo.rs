//! Manual smoke test for the wake+VAD gate (EPIC 2.3) — the epic's own
//! demo, run against live hardware: make a loud noise near the mic
//! (placeholder wake detector — see core::wake docs for why it's not the
//! real word yet), console prints `WAKE`, then talk; once you go quiet
//! for ~700ms the captured segment (with pre-roll audio at its head) is
//! written to a `.wav` file. Not part of the crate's public surface; run
//! with `cargo run -p core --example gate_demo`.
use std::env::temp_dir;
use std::time::{Duration, Instant};

use marceline_core::config::{VadConfig, WakeConfig};
use marceline_core::{
    Capture, EnergyWakeDetector, Gate, GateOutput, SileroVad, VadEndpointer, WakeEngine, WavTap,
    DEFAULT_SPEECH_THRESHOLD,
};

fn main() {
    let capture = Capture::start(1.5, None).expect("failed to start capture");

    let wake_config = WakeConfig {
        words: vec!["marceline".to_string(), "marcy".to_string()],
        sensitivity: 0.6,
    };
    let detector = EnergyWakeDetector::new(wake_config.sensitivity, 16_000, 1600);
    let wake = WakeEngine::new(&wake_config, Box::new(detector));

    let model_path = format!("{}/../models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"));
    let vad = SileroVad::load(&model_path).expect("failed to load Silero VAD model");
    let endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);

    let vad_config = VadConfig {
        silence_ms: 700,
        min_utterance_ms: 300,
        max_utterance_ms: 15_000,
        barge_in_confirm_ms: 300,
        no_speech_timeout_ms: 3_000,
    };
    let mut gate = Gate::new(wake, endpointer, &vad_config);

    println!("IDLE. Make a loud sound to fire wake (placeholder detector), then speak.");
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut segment_written = false;

    while Instant::now() < deadline && !segment_written {
        let Ok(chunk) = capture.chunks().recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        let preroll = capture.preroll();

        match gate.process_chunk(&chunk, &preroll) {
            GateOutput::Wake => println!("WAKE"),
            GateOutput::NoSpeechTimeout => println!("(no speech followed wake, back to IDLE)"),
            GateOutput::Segment(segment) => {
                let path = temp_dir().join("marceline-gate-demo-segment.wav");
                let mut tap = WavTap::create(&path, segment.sample_rate, segment.channels)
                    .expect("failed to create wav tap");
                tap.write_chunk(&segment).expect("failed to write segment");
                tap.finalize().expect("failed to finalize wav");
                println!(
                    "SEGMENT written to {path:?}: {} samples @ {}Hz/{}ch",
                    segment.pcm.len(),
                    segment.sample_rate,
                    segment.channels
                );
                segment_written = true;
            }
            GateOutput::TooShort => println!("(utterance discarded: too short)"),
            GateOutput::WakeDetected => println!("(wake detected during THINKING/SPEAKING)"),
            GateOutput::None => {}
        }
    }

    if !segment_written {
        println!("timed out waiting for a full wake -> speech -> silence cycle");
    }
}
