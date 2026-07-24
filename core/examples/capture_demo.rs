//! Manual smoke test for mic capture (EPIC 1.1). Not part of the crate's
//! public surface; run with `cargo run -p core --example capture_demo`.
use std::time::{Duration, Instant};

fn main() {
    let capture = marceline_core::Capture::start(1.5).expect("failed to start capture");
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut chunks = 0u64;
    let mut samples = 0u64;
    let mut last_seq: Option<u64> = None;
    let mut gaps = 0u64;
    let mut rate = 0u32;
    let mut channels = 0u16;

    while Instant::now() < deadline {
        if let Ok(chunk) = capture.chunks().recv_timeout(Duration::from_millis(200)) {
            if let Some(prev) = last_seq {
                if chunk.seq != prev + 1 {
                    gaps += 1;
                }
            }
            last_seq = Some(chunk.seq);
            chunks += 1;
            samples += chunk.pcm.len() as u64;
            rate = chunk.sample_rate;
            channels = chunk.channels;
        }
    }

    let preroll = capture.preroll();
    println!(
        "chunks={chunks} samples={samples} sample_rate={rate} channels={channels} seq_gaps={gaps} preroll_samples={}",
        preroll.pcm.len()
    );
}
