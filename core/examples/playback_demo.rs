//! Manual smoke test for PCM playback (EPIC 1.2). Not part of the crate's
//! public surface; run with `cargo run -p core --example playback_demo`.
use std::f32::consts::PI;
use std::thread;
use std::time::Duration;

use marceline_core::{AudioChunk, Playback};

fn sine_chunk(seconds: f32, freq: f32, sample_rate: u32, channels: u16) -> AudioChunk {
    let n = (seconds * sample_rate as f32) as usize;
    let mut pcm = Vec::with_capacity(n * channels as usize);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * PI * freq * t).sin() * 0.2;
        for _ in 0..channels {
            pcm.push(sample);
        }
    }
    AudioChunk {
        seq: 0,
        pcm,
        sample_rate,
        channels,
    }
}

fn main() {
    let playback = Playback::start(None).expect("failed to start playback");
    let rate = playback.sample_rate();
    let channels = playback.channels();
    println!("device sample_rate={rate} channels={channels}");

    let chunk = sine_chunk(1.0, 440.0, rate, channels);
    playback.push(&chunk);
    println!("pushed buffered_samples={}", playback.buffered_samples());

    thread::sleep(Duration::from_millis(300));
    let mid = playback.buffered_samples();
    println!("after 300ms buffered_samples={mid}");
    assert!(
        mid < chunk.pcm.len(),
        "playback should have drained some samples"
    );

    thread::sleep(Duration::from_millis(900));
    let drained = playback.buffered_samples();
    println!("after 1200ms total buffered_samples={drained}");
    assert_eq!(drained, 0, "1s chunk should be fully drained by 1.2s");

    // Flush test: push a chunk, then flush shortly after — buffer must be
    // empty near-immediately rather than draining the rest naturally.
    let chunk2 = sine_chunk(2.0, 440.0, rate, channels);
    playback.push(&chunk2);
    thread::sleep(Duration::from_millis(50));
    playback.flush();
    let after_flush = playback.buffered_samples();
    println!("after flush buffered_samples={after_flush}");
    assert_eq!(after_flush, 0, "flush should drop all buffered audio");

    println!("OK");
}
