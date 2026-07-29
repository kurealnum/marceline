//! The `marceline converse` path (EPIC 8.2) — the MVP loop.
//!
//! Wires the real stages behind the 8.1 orchestrator state machine so a
//! spoken question produces a spoken answer, unattended:
//! `IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> IDLE`.
//!
//! The [`marceline_core::Orchestrator`] tracks and validates state (and
//! logs every transition, per 8.1); the actual stage work — gate, STT,
//! LLM, TTS — runs here in the driving loop rather than inside
//! [`marceline_core::Stages`] hooks. That split exists because
//! `THINKING -> SPEAKING` fires on the *first streamed TTS chunk*, a point
//! reached midway through the LLM+TTS pipeline, not at the moment
//! `THINKING` is entered — the orchestrator's `apply` needs `&mut self`,
//! so the party best placed to call it the instant that chunk arrives is
//! this loop, not a `&self` hook. [`NoopStages`] fulfills the trait so the
//! orchestrator still has somewhere to log from.
//!
//! No tools, no memory, no barge-in yet (out of scope per the issue) — one
//! provider per stage, exactly the MVP bar.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use marceline_core::stt::SttWorkerPaths;
use marceline_core::transcribe::{TranscribeOutcome, DEFAULT_TIMEOUT};
use marceline_core::tts::TtsWorkerPaths;
use marceline_core::{
    compile_system_prompt, sentence_chunk, ChatRequest, Config, ConversationEvent,
    EnergyWakeDetector, Gate, GateOutput, HealthView, LlmEngine, Message, OpenAiCompatibleEngine,
    Orchestrator, Playback, Role, SileroVad, SttManager, Stages, TtsEngine, VadEndpointer,
    VoiceId, DEFAULT_SPEECH_THRESHOLD,
};
use marceline_core::audio::Capture;
use tokio::sync::{watch, RwLock};
use tokio_util::sync::CancellationToken;

/// Anything that can go wrong running the full conversation loop.
#[derive(Debug, thiserror::Error)]
pub enum ConverseError {
    /// The config file could not be loaded.
    #[error(transparent)]
    Config(#[from] marceline_core::ConfigError),
    /// A worker or backend engine failed.
    #[error(transparent)]
    Engine(#[from] marceline_core::EngineError),
    /// Opening the mic or speaker failed.
    #[error(transparent)]
    Capture(#[from] marceline_core::audio::CaptureError),
    /// Opening the speaker output failed.
    #[error(transparent)]
    Playback(#[from] marceline_core::PlaybackError),
    /// The wake/VAD gate could not load its VAD model.
    #[error(transparent)]
    Vad(#[from] marceline_core::VadError),
}

/// A [`Stages`] impl that does nothing: this story's real stage work runs
/// in [`converse`]'s driving loop instead (see module docs for why), but
/// the orchestrator still needs a `Stages` to log transitions from.
struct NoopStages;

#[async_trait]
impl Stages for NoopStages {
    async fn on_enter_listening(&self, _run: &CancellationToken) {}
    async fn on_enter_transcribing(&self, _run: &CancellationToken) {}
    async fn on_enter_thinking(&self, _run: &CancellationToken) {}
    async fn on_enter_speaking(&self, _run: &CancellationToken) {}
    async fn on_enter_error(&self, reason: &str) {
        tracing::error!(reason, "conversation turn failed");
    }
}

/// How long `LISTENING` waits for a wake word before polling again.
/// Mirrors the gate's own placeholder timeout (`core/src/gate/mod.rs`);
/// concrete values are a tuning knob (EPIC 8.3), not this story's job.
const WAKE_POLL_TIMEOUT: Duration = Duration::from_millis(200);

/// Runs the MVP loop forever: wake, listen, transcribe, think, speak,
/// back to idle. Returns only on an unrecoverable setup failure (a worker
/// or device that never came up) — a mid-turn stage failure routes
/// through the orchestrator's `ERROR` edge and the loop keeps running.
pub async fn converse(config_path: &Path, soul_path: &Path) -> Result<(), ConverseError> {
    let config = Config::load(config_path)?;
    let soul = std::fs::read_to_string(soul_path).unwrap_or_default();
    let system_prompt = compile_system_prompt(&soul, &[]);

    let capture = Capture::start(1.5, config.audio.input_device.as_deref())?;
    let detector = EnergyWakeDetector::new(config.wake.sensitivity, 16_000, 1600);
    let wake = marceline_core::WakeEngine::new(&config.wake, Box::new(detector));
    let model_path = format!("{}/models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"));
    let vad = SileroVad::load(&model_path)?;
    let endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);
    let mut gate = Gate::new(wake, endpointer, &config.vad);

    let playback = Playback::start(config.audio.output_device.as_deref())?;

    // Workers are launched once and reused across every turn; held here so
    // the shutdown senders outlive the whole loop and stop them on return.
    let (stt_shutdown_tx, stt_shutdown_rx) = watch::channel(false);
    let stt_health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let stt = SttManager::start(
        &config.stt,
        SttWorkerPaths::for_backend(&config.stt.backend),
        stt_health,
        stt_shutdown_rx,
        CancellationToken::new(),
    )
    .await?;

    let (tts_shutdown_tx, tts_shutdown_rx) = watch::channel(false);
    let tts_health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let tts = marceline_core::launch_tts_worker(
        &config.tts,
        TtsWorkerPaths::for_backend(&config.tts.backend),
        tts_health,
        tts_shutdown_rx,
        CancellationToken::new(),
    )
    .await?;
    let voice = VoiceId::from(config.tts.voice.as_str());

    let result = run_loop(
        &mut gate,
        &capture,
        &playback,
        &stt,
        &tts,
        &voice,
        &config,
        &system_prompt,
    )
    .await;

    let _ = stt_shutdown_tx.send(true);
    let _ = tts_shutdown_tx.send(true);
    result
}

/// The actual `IDLE -> ... -> IDLE`, repeated forever loop, split out of
/// [`converse`] so worker setup/teardown above stays uncluttered.
#[allow(clippy::too_many_arguments)]
async fn run_loop(
    gate: &mut Gate,
    capture: &Capture,
    playback: &Playback,
    stt: &SttManager,
    tts: &dyn TtsEngine,
    voice: &VoiceId,
    config: &Config,
    system_prompt: &str,
) -> Result<(), ConverseError> {
    let llm = OpenAiCompatibleEngine::new(&config.llm, CancellationToken::new())?;
    let mut orchestrator = Orchestrator::new(NoopStages);

    'turn: loop {
        // IDLE: poll the mic for the wake word.
        loop {
            let Ok(chunk) = capture.chunks().recv_timeout(WAKE_POLL_TIMEOUT) else {
                continue;
            };
            let preroll = capture.preroll();
            if matches!(gate.process_chunk(&chunk, &preroll), GateOutput::Wake) {
                orchestrator
                    .apply(ConversationEvent::WakeWord)
                    .await
                    .expect("Idle always accepts WakeWord");
                break;
            }
        }

        // LISTENING: collect the utterance.
        let segment = loop {
            let Ok(chunk) = capture.chunks().recv_timeout(WAKE_POLL_TIMEOUT) else {
                continue;
            };
            let preroll = capture.preroll();
            match gate.process_chunk(&chunk, &preroll) {
                GateOutput::Segment(segment) => break segment,
                GateOutput::NoSpeechTimeout | GateOutput::TooShort => {
                    // Nothing worth transcribing; the gate is already back
                    // in IDLE internally, so mirror that in the orchestrator
                    // (no ERROR — this is a normal empty-turn, not a fault)
                    // and start the next turn over from IDLE.
                    let _ = orchestrator
                        .apply(ConversationEvent::StageError("no speech captured".into()))
                        .await;
                    let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                    continue 'turn;
                }
                _ => continue,
            }
        };

        orchestrator
            .apply(ConversationEvent::VadEnd)
            .await
            .expect("Listening always accepts VadEnd");

        // TRANSCRIBING.
        let transcript = match stt.transcribe(segment, DEFAULT_TIMEOUT).await {
            Ok(TranscribeOutcome::Committed(t)) => t.text,
            Ok(TranscribeOutcome::Rejected(rejection)) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError(rejection.reason()))
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            Err(err) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError(err.to_string()))
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
        };

        orchestrator
            .apply(ConversationEvent::FinalTranscript)
            .await
            .expect("Transcribing always accepts FinalTranscript");

        // THINKING: only `Final` transcripts ever reach here (§2.4.1).
        let messages = vec![
            Message::new(Role::System, system_prompt),
            Message::new(Role::User, transcript),
        ];
        let events = llm
            .chat(ChatRequest {
                messages,
                tools: Vec::new(),
                max_tokens: config.llm.max_tokens_per_turn,
            })
            .await;
        let mut sentences = sentence_chunk(events);

        // First sentence pulled eagerly: this is the "first TTS chunk"
        // trigger, so the transition into Speaking is driven by actually
        // having something to say, not by entering Thinking.
        let first_sentence = match sentences.next().await {
            Some(Ok(text)) => text,
            Some(Err(err)) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError(err.to_string()))
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            None => {
                // The model returned no text at all; nothing to speak.
                let _ = orchestrator
                    .apply(ConversationEvent::StageError("empty llm response".into()))
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
        };
        let rest: marceline_core::TextStream = Box::pin(sentences);
        let text_stream: marceline_core::TextStream = Box::pin(
            futures::stream::once(async move { Ok(first_sentence) }).chain(rest),
        );

        let mut audio = tts.synthesize(text_stream, voice.clone()).await;
        let first_chunk = match audio.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(err)) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError(err.to_string()))
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            None => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError("tts produced no audio".into()))
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
        };

        // SPEAKING: the first chunk arriving is what flips the state.
        orchestrator
            .apply(ConversationEvent::FirstTtsChunk)
            .await
            .expect("Thinking always accepts FirstTtsChunk");
        playback.push(&first_chunk);
        while let Some(chunk) = audio.next().await {
            match chunk {
                Ok(chunk) => playback.push(&chunk),
                Err(err) => {
                    let _ = orchestrator
                        .apply(ConversationEvent::StageError(err.to_string()))
                        .await;
                    let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                    break;
                }
            }
        }
        while playback.buffered_samples() > 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        orchestrator
            .apply(ConversationEvent::PlaybackDone)
            .await
            .ok();
    }
}

/// Default SOUL.md path for `converse`, mirroring `say-to-llm`.
pub const DEFAULT_SOUL: &str = "SOUL.md";

/// Resolves `--soul <path>` from CLI args, or [`DEFAULT_SOUL`].
pub fn soul_path_from_args(args: &[String]) -> PathBuf {
    let index = args.iter().position(|arg| arg == "--soul");
    match index.and_then(|i| args.get(i + 1)) {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(DEFAULT_SOUL),
    }
}
