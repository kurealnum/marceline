//! False-trigger tuning pass (EPIC 2.4). Not part of the crate's public
//! surface; run with `cargo run -p core --example wake_tuning_demo`.
//!
//! Runs the placeholder wake detector against live ambient mic audio at
//! two different `[wake].sensitivity` values back to back, reporting
//! fire counts and near-miss scores for each — the sensitivity/false-
//! trigger data this story calls for. Since the real word-discrimination
//! model (EPIC 13.2) doesn't exist yet, this measures spurious fires on
//! *loudness* during ambient conditions, not "does it ignore ordinary
//! conversation but fire on 'Marceline'" — that claim needs the real
//! model.
use std::time::{Duration, Instant};

use marceline_core::config::WakeConfig;
use marceline_core::{Capture, EnergyWakeDetector, WakeEngine};

fn run_pass(capture: &Capture, sensitivity: f64, duration: Duration) {
    let config = WakeConfig {
        words: vec!["marceline".to_string(), "marcy".to_string()],
        sensitivity,
    };
    let detector = EnergyWakeDetector::new(sensitivity, 16_000, 1600);
    let mut engine = WakeEngine::new(&config, Box::new(detector));

    let mut fires = 0u32;
    let mut frames = 0u32;
    let mut max_score = 0.0f32;
    let mut score_sum = 0.0f32;

    let deadline = Instant::now() + duration;
    while Instant::now() < deadline {
        let Ok(chunk) = capture.chunks().recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        if let Some(event) = engine.process_chunk(&chunk) {
            fires += 1;
            println!("  WAKE word={} score={:.3}", event.word, event.score);
        }
        let score = engine.last_score();
        max_score = max_score.max(score);
        score_sum += score;
        frames += 1;
    }

    let mean = if frames > 0 { score_sum / frames as f32 } else { 0.0 };
    println!(
        "sensitivity={sensitivity:.1}: fires={fires} frames={frames} max_score={max_score:.3} mean_score={mean:.3}"
    );
}

fn main() {
    let capture = Capture::start(1.5, None).expect("failed to start capture");
    let pass_duration = Duration::from_secs(5);

    println!("--- pass 1: default sensitivity 0.6, ambient audio ---");
    run_pass(&capture, 0.6, pass_duration);

    println!("--- pass 2: high sensitivity 0.95, same ambient conditions ---");
    run_pass(&capture, 0.95, pass_duration);

    println!(
        "expect pass 2 to fire at least as often as pass 1 on the same ambient audio \
         (higher sensitivity = lower threshold = easier to cross)"
    );
}
