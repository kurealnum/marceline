//! Manual smoke test for wake-word detection (EPIC 2.1). Not part of the
//! crate's public surface; run with `cargo run -p core --example
//! wake_demo`. Make some noise near the mic to see it fire — this uses
//! the placeholder EnergyWakeDetector (see core::wake module docs), so
//! it reacts to loudness, not the actual word "Marceline".
use std::time::{Duration, Instant};

use marceline_core::{Capture, EnergyWakeDetector, WakeEngine};

fn main() {
    let capture = Capture::start(1.5, None).expect("failed to start capture");
    let config = marceline_core::config::WakeConfig {
        words: vec!["marceline".to_string(), "marcy".to_string()],
        sensitivity: 0.6,
    };
    let detector = EnergyWakeDetector::new(config.sensitivity, 16_000, 1600);
    let mut engine = WakeEngine::new(&config, Box::new(detector));

    println!("listening for 5s (placeholder detector: reacts to loud sound, not real words)");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut fires = 0;
    while Instant::now() < deadline {
        if let Ok(chunk) = capture.chunks().recv_timeout(Duration::from_millis(200)) {
            if let Some(event) = engine.process_chunk(&chunk) {
                fires += 1;
                println!(
                    "WAKE word={} score={:.2} at {:?}",
                    event.word, event.score, event.timestamp
                );
            }
        }
    }
    println!("done, fires={fires}");
}
