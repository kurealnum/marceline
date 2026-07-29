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
//! this loop, not a `&self` hook. [`ErrorSpeaker`]'s happy-path hooks are
//! no-ops for the same reason; only its `on_enter_error` does real work
//! (EPIC 8.3's graceful spoken failure message).
//!
//! No tools, no memory, no barge-in yet (out of scope per the issue) — one
//! provider per stage, exactly the MVP bar.
//!
//! **Cancellation (EPIC 8.4, SPEC.md §2.5.1):** one run [`CancellationToken`]
//! per turn, minted the moment `WakeWord` fires, is what every stage's
//! client connection for that turn is built with. STT and TTS workers stay
//! up across turns (relaunching the model per turn would be absurd), but
//! each turn opens a *fresh client connection* to the already-running
//! worker carrying that turn's token — cheap (a socket connect, not a
//! model load) and it's what lets firing the token actually reach a
//! specific turn's in-flight gRPC call rather than being fixed at
//! worker-launch time. `ctrl-c` fires whatever turn is currently in
//! flight (or exits immediately if idle) — the same path barge-in (EPIC 7)
//! will fire later.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use marceline_core::stt::SttWorkerPaths;
use marceline_core::transcribe::{TranscribeOutcome, DEFAULT_TIMEOUT};
use marceline_core::tts::TtsWorkerPaths;
use marceline_core::{
    compile_system_prompt, sentence_chunk, ChatRequest, Config, ConversationEvent,
    EnergyWakeDetector, FailedStage, Gate, GateOutput, GrpcTtsEngine, HealthView, LlmEngine,
    Message, OpenAiCompatibleEngine, Orchestrator, Playback, Role, SileroVad, SttManager, Stages,
    TtsEngine, VadEndpointer, VoiceId, DEFAULT_SPEECH_THRESHOLD,
};
use marceline_core::audio::Capture;
use tokio::sync::{watch, RwLock};
use tokio_util::sync::CancellationToken;

/// The one run token currently in flight, shared with the `ctrl-c`
/// watcher (EPIC 8.4, §2.5.1) — `None` while `IDLE`. A plain `std::sync`
/// mutex is enough: it is only ever held for the instant it takes to
/// clone or replace the token, never across an `.await`.
type CurrentRun = Arc<Mutex<Option<CancellationToken>>>;

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

/// A short, fixed message spoken on any non-TTS stage failure (SPEC.md
/// §9.11, EPIC 8.3). v1 does not attempt to tailor this per failure —
/// just make sure the user hears *something* rather than silence.
const GRACEFUL_ERROR_MESSAGE: &str = "Sorry, I ran into a problem with that. Please try again.";

/// A [`Stages`] impl whose only real work is the `ERROR` edge (SPEC.md
/// §2.5, EPIC 8.3): every other hook is a no-op because the happy-path
/// stage work runs in [`converse`]'s driving loop instead (see module
/// docs for why — `THINKING -> SPEAKING` fires mid-stage, not at entry).
///
/// On error, speaks [`GRACEFUL_ERROR_MESSAGE`] through a fresh connection
/// to the already-running TTS worker — unless the failed stage *is* TTS,
/// in which case no spoken message is possible and this only logs
/// (§9.11's accepted exception).
///
/// A *fresh* connection, not the turn's own (now-cancelled or faulted)
/// one: that token is either already fired or belongs to the stage that
/// just failed, and reusing it here would make the graceful message
/// cancel itself before a single chunk plays.
struct ErrorSpeaker<'a> {
    tts_socket: &'a Path,
    playback: &'a Playback,
    voice: &'a VoiceId,
}

#[async_trait(?Send)]
impl<'a> Stages for ErrorSpeaker<'a> {
    async fn on_enter_listening(&self, _run: &CancellationToken) {}
    async fn on_enter_transcribing(&self, _run: &CancellationToken) {}
    async fn on_enter_thinking(&self, _run: &CancellationToken) {}
    async fn on_enter_speaking(&self, _run: &CancellationToken) {}

    async fn on_enter_error(&self, stage: FailedStage, reason: &str) {
        tracing::error!(?stage, reason, "conversation turn failed");
        if stage == FailedStage::Tts {
            // Can't speak a TTS failure through the TTS that just failed
            // — log only and return to IDLE silently (§9.11).
            return;
        }

        let tts = match GrpcTtsEngine::connect(self.tts_socket, CancellationToken::new()).await {
            Ok(tts) => tts,
            Err(err) => {
                tracing::error!(%err, "could not reach tts worker to speak graceful error message");
                return;
            }
        };
        let text_stream: marceline_core::TextStream = Box::pin(futures::stream::once(async {
            Ok(GRACEFUL_ERROR_MESSAGE.to_string())
        }));
        let mut audio = tts.synthesize(text_stream, self.voice.clone()).await;
        while let Some(chunk) = audio.next().await {
            match chunk {
                Ok(chunk) => self.playback.push(&chunk),
                Err(err) => {
                    // The graceful message itself failed to speak; nothing
                    // more we can do here without recursing into ERROR.
                    tracing::error!(%err, "failed to speak graceful error message");
                    break;
                }
            }
        }
        while self.playback.buffered_samples() > 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
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

    // Workers are launched once and stay up across every turn (relaunching
    // the model per turn would be absurd); each turn instead opens its own
    // client connection to these sockets, carrying that turn's own
    // cancellation token (see module docs, EPIC 8.4). The `SttManager`/
    // engine values returned here exist only to prove the worker came up —
    // no further calls go through them.
    let stt_paths = SttWorkerPaths::for_backend(&config.stt.backend);
    let stt_socket = stt_paths.socket_path.clone();
    let (stt_shutdown_tx, stt_shutdown_rx) = watch::channel(false);
    let stt_health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let _stt_launch = SttManager::start(
        &config.stt,
        stt_paths,
        stt_health,
        stt_shutdown_rx,
        CancellationToken::new(),
    )
    .await?;

    let tts_paths = TtsWorkerPaths::for_backend(&config.tts.backend);
    let tts_socket = tts_paths.socket_path.clone();
    let (tts_shutdown_tx, tts_shutdown_rx) = watch::channel(false);
    let tts_health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let _tts_launch = marceline_core::launch_tts_worker(
        &config.tts,
        tts_paths,
        tts_health,
        tts_shutdown_rx,
        CancellationToken::new(),
    )
    .await?;
    let voice = VoiceId::from(config.tts.voice.as_str());

    // Fired by the `ctrl-c` watcher below and set/cleared by the loop as
    // turns start and finish — the one shared handle onto "the run
    // currently in flight" (§2.5.1).
    let current_run: CurrentRun = Arc::new(Mutex::new(None));
    let ctrlc_run = Arc::clone(&current_run);
    tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            match ctrlc_run.lock().expect("current_run lock poisoned").clone() {
                // A turn is in flight: cancel it (same path barge-in will
                // ride, EPIC 7) rather than killing the process outright.
                Some(token) => token.cancel(),
                // Idle: nothing to cancel, so ctrl-c means "exit".
                None => std::process::exit(0),
            }
        }
    });

    let result = run_loop(
        &mut gate,
        &capture,
        &playback,
        &stt_socket,
        &tts_socket,
        &config.stt.lang,
        &voice,
        &config,
        &system_prompt,
        &current_run,
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
    stt_socket: &Path,
    tts_socket: &Path,
    lang: &str,
    voice: &VoiceId,
    config: &Config,
    system_prompt: &str,
    current_run: &CurrentRun,
) -> Result<(), ConverseError> {
    let transcribe_timeout = Duration::from_millis(config.orchestrator.transcribe_timeout_ms);
    let think_timeout = Duration::from_millis(config.orchestrator.think_timeout_ms);
    let speak_timeout = Duration::from_millis(config.orchestrator.speak_timeout_ms);
    let mut orchestrator = Orchestrator::new(ErrorSpeaker {
        tts_socket,
        playback,
        voice,
    });

    'turn: loop {
        // IDLE: poll the mic for the wake word. No run token exists yet —
        // clear the shared slot so a stray ctrl-c while idle just exits
        // (handled by the watcher) instead of cancelling nothing.
        *current_run.lock().expect("current_run lock poisoned") = None;
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

        // One run token for the whole turn (§2.5.1): minted by `apply`
        // above, published here so `ctrl-c` can reach it, and cloned into
        // every stage's client connection below.
        let run_token = orchestrator
            .run_token()
            .expect("Listening always has a run token")
            .clone();
        *current_run.lock().expect("current_run lock poisoned") = Some(run_token.clone());

        let stt = match SttManager::attach(stt_socket.to_path_buf(), run_token.clone(), lang.to_string())
            .await
        {
            Ok(stt) => stt,
            Err(err) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Stt,
                        reason: err.to_string(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue 'turn;
            }
        };
        let tts = match GrpcTtsEngine::connect(tts_socket, run_token.clone()).await {
            Ok(tts) => tts,
            Err(err) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Tts,
                        reason: err.to_string(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue 'turn;
            }
        };
        // Cheap to build per turn: no persistent connection, just an HTTP
        // client config, so it carries this turn's own token rather than
        // one fixed for the process lifetime.
        let llm = OpenAiCompatibleEngine::new(&config.llm, run_token.clone())?;

        // LISTENING: collect the utterance. The gate's own no-speech
        // timeout (`[vad].no_speech_timeout_ms`, EPIC 8.3) covers the
        // "nobody spoke after the wake word" edge internally.
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
                        .apply(ConversationEvent::StageError {
                            stage: FailedStage::Gate,
                            reason: "no speech captured".into(),
                        })
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

        // TRANSCRIBING: worker-down surfaces as either a timeout here or
        // an `EngineError` from `transcribe` itself; both route through
        // the same `StageError`.
        let transcript = match tokio::time::timeout(
            transcribe_timeout,
            stt.transcribe(segment, DEFAULT_TIMEOUT),
        )
        .await
        {
            Ok(Ok(TranscribeOutcome::Committed(t))) => t.text,
            Ok(Ok(TranscribeOutcome::Rejected(rejection))) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Stt,
                        reason: rejection.reason(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            Ok(Err(err)) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Stt,
                        reason: err.to_string(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            Err(_elapsed) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Stt,
                        reason: "stt timed out".into(),
                    })
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

        // First sentence pulled eagerly, under `think_timeout`: this is
        // the "first TTS chunk" trigger, so the transition into Speaking
        // is driven by actually having something to say, not by entering
        // Thinking. A stuck/dead LLM shows up here as a timeout.
        let first_sentence = match tokio::time::timeout(think_timeout, sentences.next()).await {
            Ok(Some(Ok(text))) => text,
            Ok(Some(Err(err))) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Llm,
                        reason: err.to_string(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            Ok(None) => {
                // The model returned no text at all; nothing to speak.
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Llm,
                        reason: "empty llm response".into(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            Err(_elapsed) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Llm,
                        reason: "llm timed out".into(),
                    })
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
        let first_chunk = match tokio::time::timeout(speak_timeout, audio.next()).await {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(err))) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Tts,
                        reason: err.to_string(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            Ok(None) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Tts,
                        reason: "tts produced no audio".into(),
                    })
                    .await;
                let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                continue;
            }
            Err(_elapsed) => {
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Tts,
                        reason: "tts timed out".into(),
                    })
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
        let mut cancelled = false;
        while let Some(chunk) = audio.next().await {
            match chunk {
                Ok(chunk) => playback.push(&chunk),
                Err(err) => {
                    // A cancel (ctrl-c/barge-in) surfaces here as a stream
                    // error (§2.5.1) — flush rather than waiting for the
                    // ring to drain, or Marceline talks over the user for
                    // the length of whatever was already buffered.
                    cancelled = run_token.is_cancelled();
                    if cancelled {
                        // Partial-state policy (§2.5.1): until EPIC 10's
                        // history store exists, marking the turn
                        // `interrupted` means logging it — there is
                        // nowhere else to record it yet.
                        tracing::info!(interrupted = true, "turn cancelled mid-speech");
                    }
                    let _ = orchestrator
                        .apply(ConversationEvent::StageError {
                            stage: FailedStage::Tts,
                            reason: err.to_string(),
                        })
                        .await;
                    let _ = orchestrator.apply(ConversationEvent::ErrorHandled).await;
                    break;
                }
            }
        }
        if cancelled {
            playback.flush();
        } else {
            while playback.buffered_samples() > 0 {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
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
