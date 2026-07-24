//! Manual smoke test for level metering + wav tap (EPIC 1.4). Not part of
//! the crate's public surface; run with
//! `cargo run -p core --example meter_wav_demo`. Talk/make noise near the
//! mic while it runs to see the meter move.
use std::env::temp_dir;
use std::time::{Duration, Instant};

use marceline_core::{Capture, LevelMeter, WavTap};

fn main() {
    let capture = Capture::start(1.5, None).expect("failed to start capture");
    let rate = {
        // Peek the format off the first chunk rather than exposing a
        // getter solely for this demo.
        let first = capture
            .chunks()
            .recv_timeout(Duration::from_secs(2))
            .expect("expected at least one chunk within 2s");
        let path = temp_dir().join("marceline-meter-wav-demo.wav");
        let mut tap =
            WavTap::create(&path, first.sample_rate, first.channels).expect("failed to create wav tap");
        let mut meter = LevelMeter::default();
        let mut total_samples = 0u64;

        meter.update(&first.pcm);
        tap.write_chunk(&first).expect("failed to write chunk");
        total_samples += first.pcm.len() as u64;
        println!("{}", meter.render(30));

        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Ok(chunk) = capture.chunks().recv_timeout(Duration::from_millis(200)) {
                meter.update(&chunk.pcm);
                tap.write_chunk(&chunk).expect("failed to write chunk");
                total_samples += chunk.pcm.len() as u64;
                println!("{}", meter.render(30));
            }
        }

        tap.finalize().expect("failed to finalize wav");
        println!(
            "wrote {path:?}: {total_samples} samples @ {}Hz/{}ch",
            first.sample_rate, first.channels
        );

        let reader = hound::WavReader::open(&path).expect("failed to reopen wav");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, first.sample_rate);
        assert_eq!(spec.channels, first.channels);
        assert_eq!(
            reader.duration() as u64,
            total_samples / first.channels as u64,
            "wav frame count should match frames captured"
        );
        first.sample_rate
    };

    println!("OK sample_rate={rate}");
}
