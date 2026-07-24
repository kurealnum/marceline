//! Shared types and generated gRPC/protobuf stubs, consumed by `core`.
//!
//! The canonical schema lives in `proto/*.proto` (SPEC.md §2.4.1) and is
//! compiled to Rust at build time by `build.rs`, and to Python stubs for
//! the worker template (EPIC 0.4) by `scripts/gen-python-proto.sh`.

/// Types shared between the `stt` and `tts` contracts: `AudioChunk`
/// (self-describing PCM) and `Cancel` (cooperative cancel signal, §2.5.1).
pub mod common {
    tonic::include_proto!("marceline.common");
}

/// Streaming speech-to-text contract: audio frames in, transcripts out.
#[allow(clippy::result_large_err)]
pub mod stt {
    tonic::include_proto!("marceline.stt");
}

/// Streaming text-to-speech contract: segmented text in, audio chunks out.
#[allow(clippy::result_large_err)]
pub mod tts {
    tonic::include_proto!("marceline.tts");
}
