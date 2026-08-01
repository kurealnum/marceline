//! The `marceline say <text>` path (EPIC 5, demoable; EPIC 11.4's
//! isolated-TTS-stage-test half of the CLI control surface).
//!
//! Speaks `text` aloud through the configured `[tts]` backend and writes
//! what it spoke to a `.wav` file, so the epic's demo is checkable from the
//! command line: change `[tts].backend` from `kokoro` to `piper`, rerun the
//! same text, and it still speaks — on the new backend, no code change.
//! This is the same [`TtsEngine`] trait impl the daemon's conversation
//! loop drives (`cli::converse`), so a passing `say` run against a given
//! `[tts]` config means that stage works in the real pipeline too — no
//! separate test-only code path to fall out of sync with it.
//!
//! Unlike `say-to-llm`, this does not touch the LLM: `text` goes straight
//! to TTS as a single already-segmented span, the same shape sentence-
//! chunking (EPIC 5.3) would hand it one sentence at a time.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use marceline_core::tts::TtsWorkerPaths;
use marceline_core::{Config, HealthView, Playback, TtsEngine, VoiceId, WavTap};
use std::collections::HashMap;
use tokio::sync::{watch, RwLock};
use tokio_util::sync::CancellationToken;

/// Anything that can go wrong speaking text from the command line.
#[derive(Debug, thiserror::Error)]
pub enum SayError {
    /// The config file could not be loaded.
    #[error(transparent)]
    Config(#[from] marceline_core::ConfigError),
    /// The TTS worker was unreachable, failed, or timed out.
    #[error(transparent)]
    Engine(#[from] marceline_core::EngineError),
    /// Opening the speaker output failed.
    #[error(transparent)]
    Playback(#[from] marceline_core::PlaybackError),
    /// Creating or writing the `.wav` file failed.
    #[error(transparent)]
    WavTap(#[from] marceline_core::WavTapError),
}

/// Speaks `text` through the `[tts]` backend named in the config at
/// `config_path`, playing it live and writing it to `wav_path`.
///
/// A one-shot CLI run has nothing to barge in on, but the cancellation
/// token is threaded through from the start (§2.5.1) rather than
/// retrofitted, same as `transcribe_file`.
pub async fn say(config_path: &Path, wav_path: &Path, text: &str) -> Result<(), SayError> {
    let config = Config::load(config_path)?;
    tracing::info!(
        config = %config_path.display(),
        backend = %config.tts.backend,
        voice = %config.tts.voice,
        "launching tts worker from config"
    );

    let cancel = CancellationToken::new();
    // Held for the whole run: dropping the sender would tell the
    // supervisor to stop the worker we are about to use.
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let paths = TtsWorkerPaths::for_backend(&config.tts.backend);

    let engine = marceline_core::launch_tts_worker(
        &config.tts,
        paths,
        health,
        shutdown_rx,
        cancel.clone(),
    )
    .await;
    let engine = match engine {
        Ok(engine) => engine,
        Err(err) => {
            let _ = shutdown_tx.send(true);
            return Err(err.into());
        }
    };

    let info = engine.info();
    tracing::info!(model = %info.name, sample_rate = info.output_sample_rate, "using tts backend");

    let owned_text = text.to_string();
    let text_stream: marceline_core::TextStream =
        Box::pin(futures::stream::once(async move { Ok(owned_text) }));
    let voice = VoiceId::from(config.tts.voice.as_str());
    let mut audio = engine.synthesize(text_stream, voice).await;

    let playback = Playback::start(config.audio.output_device.as_deref())?;
    let mut wav = WavTap::create(wav_path, info.output_sample_rate, 1)?;

    let result = loop {
        match audio.next().await {
            Some(Ok(chunk)) => {
                playback.push(&chunk);
                if let Err(err) = wav.write_chunk(&chunk) {
                    break Err(err.into());
                }
            }
            Some(Err(err)) => break Err(err.into()),
            None => break Ok(()),
        }
    };

    if let Err(err) = wav.finalize() {
        tracing::warn!(%err, "failed to finalize wav file");
    }

    // Let playback drain the ring before tearing the stream down, so the
    // last chunk is actually heard rather than dropped with the process.
    while playback.buffered_samples() > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // Stop a worker we launched before returning, so the process does not
    // outlive the command that started it.
    let _ = shutdown_tx.send(true);

    result
}

/// Default `.wav` output path when `--wav` is not given.
pub const DEFAULT_WAV: &str = "say.wav";

/// Resolves `--wav <path>` from CLI args, falling back to [`DEFAULT_WAV`].
pub fn wav_path_from_args(args: &[String]) -> PathBuf {
    let index = args.iter().position(|arg| arg == "--wav");
    let path = index
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| DEFAULT_WAV.to_string());
    PathBuf::from(path)
}
