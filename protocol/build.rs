//! Compiles the `.proto` contracts in `proto/` to Rust client + server
//! stubs at build time (SPEC.md §2.4.1, EPIC 0.5).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(
            &["proto/common.proto", "proto/stt.proto", "proto/tts.proto"],
            &["proto"],
        )?;
    Ok(())
}
