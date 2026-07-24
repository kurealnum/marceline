//! Manual smoke test for device-rate resampling (EPIC 1.3). Not part of
//! the crate's public surface; run with
//! `cargo run -p core --example resample_demo`.
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
    AudioChunk { seq: 0, pcm, sample_rate, channels }
}

fn main() {
    // Bad device name: must warn and fall back to default rather than fail.
    let playback = Playback::start(Some("definitely-not-a-real-device"))
        .expect("should fall back to default on unknown device name");
    let rate = playback.sample_rate();
    let channels = playback.channels();
    println!("device sample_rate={rate} channels={channels}");

    // Chunk declares a rate that (almost certainly) differs from the
    // device's — proves push() resamples rather than assuming a match.
    let chunk_rate = 22_050u32;
    let chunk = sine_chunk(1.0, 440.0, chunk_rate, 1);
    println!("pushing chunk_rate={chunk_rate} chunk_channels=1 samples={}", chunk.pcm.len());
    playback.push(&chunk);

    let expected_device_samples =
        (chunk.pcm.len() as u64 * rate as u64 / chunk_rate as u64) as usize * channels as usize;
    let buffered = playback.buffered_samples();
    println!("buffered_samples={buffered} expected~={expected_device_samples}");
    let tolerance = (expected_device_samples / 100).max(channels as usize);
    assert!(
        buffered.abs_diff(expected_device_samples) <= tolerance,
        "resampled length should match device rate/channels scaling"
    );

    thread::sleep(Duration::from_millis(1200));
    assert_eq!(playback.buffered_samples(), 0, "resampled chunk should fully drain");
    println!("OK");
}
