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
use std::time::{Duration, Instant};

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

/// How often the SOUL.md hot-reload watcher checks the file's mtime
/// (EPIC 9.2). Fast enough that a save feels live, cheap enough to poll
/// forever in the background.
const SOUL_WATCH_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Runs the MVP loop forever: wake, listen, transcribe, think, speak,
/// back to idle. Returns only on an unrecoverable setup failure (a worker
/// or device that never came up) — a mid-turn stage failure routes
/// through the orchestrator's `ERROR` edge and the loop keeps running.
pub async fn converse(config_path: &Path, soul_path: &Path) -> Result<(), ConverseError> {
    let config = Config::load(config_path)?;

    // Hot-reload SOUL.md in the background (EPIC 9.2): each turn below
    // recompiles the system prompt from the watcher's latest persona, so an
    // edit takes effect on the next turn with no restart.
    let soul_watch_cancel = CancellationToken::new();
    let (soul_watcher, soul_watch_handle) = marceline_core::soul_watch::watch(
        soul_path.to_path_buf(),
        SOUL_WATCH_POLL_INTERVAL,
        soul_watch_cancel.clone(),
    );

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
        &soul_watcher,
        &current_run,
    )
    .await;

    let _ = stt_shutdown_tx.send(true);
    let _ = tts_shutdown_tx.send(true);
    soul_watch_cancel.cancel();
    let _ = soul_watch_handle.await;
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
    soul_watcher: &marceline_core::soul_watch::SoulWatcher,
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
        // Wake→first-audio latency (SPEC.md §9.2, §10, EPIC 12.3) is
        // timed from here — the wake word firing is the perceived start
        // of a turn, the moment the target's "≤1.5s" is measured against.
        let wake_at = Instant::now();

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
        let vad_end_at = Instant::now();

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
        let transcript_ready_at = Instant::now();

        // THINKING: only `Final` transcripts ever reach here (§2.4.1).
        // Recompiled from the watcher's latest persona every turn, so a
        // SOUL.md save takes effect on the very next turn (EPIC 9.2).
        let persona = soul_watcher.current();
        let system_prompt = compile_system_prompt(&persona.render(), &[]);
        let messages = vec![
            Message::new(Role::System, &system_prompt),
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
        // Every other arm of this match `continue`s the turn loop, so this
        // is guaranteed assigned by the time it's read below (SPEC.md
        // §12.3's `llm_first_sentence_ms` boundary).
        let llm_first_sentence_at;
        let first_sentence = match tokio::time::timeout(think_timeout, sentences.next()).await {
            Ok(Some(Ok(text))) => {
                llm_first_sentence_at = Instant::now();
                text
            }
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

        // Resolved fresh every turn from the watcher's latest persona
        // (EPIC 9.4): a SOUL.md voice change takes effect on the next
        // reply, same as a persona/tool-policy edit (EPIC 9.2/9.3), and an
        // unavailable request falls back to the config default rather
        // than failing the turn.
        let resolved_voice = marceline_core::resolve_voice(
            persona.voice_preference().voice_id.as_deref(),
            &tts.info(),
            voice,
        );
        let mut audio = tts.synthesize(text_stream, resolved_voice).await;
        // Same guarantee as `llm_first_sentence_at` above: every other arm
        // `continue`s the turn loop.
        let tts_first_chunk_at;
        let first_chunk = match tokio::time::timeout(speak_timeout, audio.next()).await {
            Ok(Some(Ok(chunk))) => {
                tts_first_chunk_at = Instant::now();
                chunk
            }
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

        // Wake→first-audio latency (SPEC.md §9.2, §10, EPIC 12.3): every
        // instant here was captured at a state-machine boundary this turn
        // already crossed, so this is a few `Instant::now()` calls' worth
        // of overhead added to the whole turn, not something that stalls
        // the streaming path it measures. Logged as a structured event
        // (queryable via `RUST_LOG`/`MARCELINE_LOG` target filtering, and
        // over `marceline logs`, EPIC 11.5) — nothing yet asserts the
        // ≤1.5s target itself; that's EPIC 12.4's canned-audio harness.
        let latency = marceline_core::TurnLatencyMs::from_instants(
            wake_at,
            vad_end_at,
            transcript_ready_at,
            llm_first_sentence_at,
            tts_first_chunk_at,
        );
        tracing::info!(
            vad_tail_ms = latency.vad_tail_ms,
            stt_ms = latency.stt_ms,
            llm_first_sentence_ms = latency.llm_first_sentence_ms,
            tts_first_chunk_ms = latency.tts_first_chunk_ms,
            total_ms = latency.total_ms,
            meets_target = latency.meets_target(),
            "turn latency"
        );

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

/// Canned-audio integration test (EPIC 12.4): drives `run_loop` — the
/// exact function `marceline converse` runs — end to end against a
/// pre-recorded wav file instead of a live mic, with fake STT/TTS/LLM
/// servers standing in for the real Python workers and external LLM.
///
/// What's real: the wake/VAD gate (`Gate`, real Silero ONNX inference),
/// `Capture::from_wav_file` feeding it, `Playback::null` draining
/// synthesized audio, and every bit of orchestration in between —
/// `run_loop` itself is not special-cased for testing at all. What's
/// faked: the STT/TTS workers (real gRPC servers over real unix sockets,
/// speaking the exact `marceline_protocol` contract the Python workers
/// do — see `core/tests/{common,tts_common}` for the precedent this
/// mirrors) and the LLM (a real HTTP server speaking the OpenAI SSE
/// format, mirroring `core/tests/llm_integration.rs`). No CUDA, no
/// Python venv, no network — runnable headlessly in CI (this story's
/// "Done when").
#[cfg(test)]
mod canned_audio_tests {
    use super::*;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicU32, Ordering};

    use bytes::Bytes;
    use http_body_util::{BodyExt, StreamBody};
    use hyper::body::Frame;
    use hyper::service::service_fn;
    use hyper::Response as HyperResponse;
    use hyper_util::rt::TokioIo;
    use marceline_protocol::common::AudioChunk as ProtoAudioChunk;
    use marceline_protocol::stt::stt_server::{Stt, SttServer};
    use marceline_protocol::stt::{
        stt_response, FinalTranscript, SttInfo, SttInfoRequest, SttRequest, SttResponse,
    };
    use marceline_protocol::tts::tts_server::{Tts, TtsServer};
    use marceline_protocol::tts::{TtsInfo, TtsInfoRequest, TtsRequest, TtsResponse};
    use tokio::net::{TcpListener, UnixListener};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
    use tonic::{Request, Response, Status, Streaming};

    static SOCKET_SEQ: AtomicU32 = AtomicU32::new(0);

    fn unique_socket_path(name: &str) -> PathBuf {
        let n = SOCKET_SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "marceline-canned-audio-test-{}-{n}-{name}.sock",
            std::process::id()
        ))
    }

    /// Fake `Stt` worker: consumes whatever audio it's sent and always
    /// answers with one fixed final transcript — the content of the
    /// canned wav doesn't need to be a real recognizable sentence for
    /// this harness to prove the pipeline wiring works end to end;
    /// `core/tests/vad_integration.rs`'s real-speech fixture only needs
    /// to be *speech-like enough* to drive the wake/VAD gate for real.
    struct FakeStt {
        transcript: String,
    }

    #[tonic::async_trait]
    impl Stt for FakeStt {
        type TranscribeStream = ReceiverStream<Result<SttResponse, Status>>;

        async fn transcribe(
            &self,
            request: Request<Streaming<SttRequest>>,
        ) -> Result<Response<Self::TranscribeStream>, Status> {
            let mut inbound = request.into_inner();
            let transcript = self.transcript.clone();
            let (tx, rx) = mpsc::channel(4);
            tokio::spawn(async move {
                while let Some(Ok(_msg)) = inbound.next().await {}
                let _ = tx
                    .send(Ok(SttResponse {
                        transcript: Some(stt_response::Transcript::Final(FinalTranscript {
                            text: transcript,
                            confidence: 0.98,
                            no_speech_prob: Some(0.01),
                            avg_logprob: Some(-0.1),
                        })),
                    }))
                    .await;
            });
            Ok(Response::new(ReceiverStream::new(rx)))
        }

        async fn get_info(&self, _: Request<SttInfoRequest>) -> Result<Response<SttInfo>, Status> {
            Ok(Response::new(SttInfo {
                name: "fake-stt".to_string(),
                langs: vec!["en".to_string()],
                input_sample_rate: 16_000,
                partials: false,
            }))
        }
    }

    /// Starts a fake STT worker on its own unix socket, returning the
    /// socket path `SttManager::attach` dials.
    async fn start_fake_stt(transcript: &str) -> PathBuf {
        let path = unique_socket_path("stt");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind fake stt socket");
        let worker = FakeStt {
            transcript: transcript.to_string(),
        };
        tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(SttServer::new(worker))
                .serve_with_incoming(UnixListenerStream::new(listener))
                .await;
        });
        path
    }

    /// Fake `Tts` worker: answers every synthesize request with one fixed
    /// audio chunk, and records every text span it was asked to speak so
    /// the test can assert the LLM's reply actually reached TTS.
    struct FakeTts {
        received: Arc<Mutex<Vec<String>>>,
    }

    #[tonic::async_trait]
    impl Tts for FakeTts {
        type SynthesizeStream = ReceiverStream<Result<TtsResponse, Status>>;

        async fn synthesize(
            &self,
            request: Request<Streaming<TtsRequest>>,
        ) -> Result<Response<Self::SynthesizeStream>, Status> {
            let mut inbound = request.into_inner();
            let received = Arc::clone(&self.received);
            let (tx, rx) = mpsc::channel(4);
            tokio::spawn(async move {
                while let Some(Ok(msg)) = inbound.next().await {
                    if let Some(marceline_protocol::tts::tts_request::Payload::Text(text)) =
                        msg.payload
                    {
                        received.lock().expect("received lock poisoned").push(text);
                    }
                }
                let _ = tx
                    .send(Ok(TtsResponse {
                        audio: Some(ProtoAudioChunk {
                            seq: 0,
                            pcm: vec![0.1; 800],
                            sample_rate: 24_000,
                            channels: 1,
                        }),
                    }))
                    .await;
            });
            Ok(Response::new(ReceiverStream::new(rx)))
        }

        async fn get_info(&self, _: Request<TtsInfoRequest>) -> Result<Response<TtsInfo>, Status> {
            Ok(Response::new(TtsInfo {
                name: "fake-tts".to_string(),
                voices: vec!["af_sky".to_string()],
                output_sample_rate: 24_000,
            }))
        }
    }

    /// Starts a fake TTS worker on its own unix socket, returning the
    /// socket path and the shared log of text spans it received.
    async fn start_fake_tts() -> (PathBuf, Arc<Mutex<Vec<String>>>) {
        let path = unique_socket_path("tts");
        let _ = std::fs::remove_file(&path);
        let received = Arc::new(Mutex::new(Vec::new()));
        let listener = UnixListener::bind(&path).expect("bind fake tts socket");
        let worker = FakeTts {
            received: Arc::clone(&received),
        };
        tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(TtsServer::new(worker))
                .serve_with_incoming(UnixListenerStream::new(listener))
                .await;
        });
        (path, received)
    }

    /// Starts a fake OpenAI-compatible SSE server (mirroring
    /// `core/tests/llm_integration.rs`'s `start_fake_server`) that always
    /// replies with one fixed sentence, and returns its base URL.
    async fn start_fake_llm(reply: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let line = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{reply:?}}},\"finish_reason\":null}}]}}\n\n\
             data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
             data: [DONE]\n\n"
        );

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(accepted) => accepted,
                    Err(_) => return,
                };
                let io = TokioIo::new(stream);
                let line = line.clone();
                tokio::spawn(async move {
                    let service = service_fn(move |_req| {
                        let body = Bytes::from(line.clone());
                        async move {
                            let stream = futures::stream::once(async move {
                                Ok::<_, Infallible>(Frame::data(body))
                            });
                            Ok::<_, Infallible>(
                                HyperResponse::builder()
                                    .status(200)
                                    .header("content-type", "text/event-stream")
                                    .body(BodyExt::boxed(StreamBody::new(stream)))
                                    .expect("build response"),
                            )
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, service)
                        .await;
                });
            }
        });

        format!("http://{addr}/v1")
    }

    /// Builds a canned wav: the repo's real-speech VAD fixture
    /// (`core/tests/fixtures/speech_sample.wav`, 16kHz mono — real enough
    /// audio to drive the wake/VAD gate for real, per
    /// `core/tests/vad_integration.rs`) followed by enough trailing
    /// silence for the VAD to endpoint the utterance
    /// (`[vad].silence_ms`).
    fn build_canned_wav(dest: &Path) {
        let fixture = format!(
            "{}/../core/tests/fixtures/speech_sample.wav",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut reader = hound::WavReader::open(&fixture).expect("open speech fixture");
        let mut samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("decode sample") as f32 / i16::MAX as f32)
            .collect();
        // 1.5s of trailing silence: comfortably past `[vad].silence_ms`
        // (700ms default) so the gate reliably endpoints the utterance.
        samples.extend(std::iter::repeat_n(0.0f32, 16_000 * 3 / 2));

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(dest, spec).expect("create canned wav");
        for s in samples {
            writer.write_sample(s).expect("write sample");
        }
        writer.finalize().expect("finalize canned wav");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_canned_audio_turn_produces_a_transcript_and_a_spoken_response() {
        let stt_socket = start_fake_stt("what's the weather like").await;
        let (tts_socket, tts_received) = start_fake_tts().await;
        let llm_base_url = start_fake_llm("It's sunny today.").await;

        let env_var = "MARCELINE_CANNED_AUDIO_TEST_LLM_KEY";
        std::env::set_var(env_var, "test-key");

        let config_dir = tempfile::tempdir().unwrap();
        let config_path = config_dir.path().join("config.toml");
        std::fs::write(&config_path, crate::setup::DEFAULT_CONFIG_TOML).unwrap();
        marceline_core::config_edit::set_string(&config_path, "llm.base_url", &llm_base_url)
            .unwrap();
        marceline_core::config_edit::set_string(&config_path, "llm.api_key_env", env_var).unwrap();
        let mut config = Config::load(&config_path).expect("load canned-audio test config");
        // Generous but bounded: real timeouts, just wide enough that a
        // loaded CI runner never trips them on a fake server that answers
        // near-instantly anyway.
        config.orchestrator.transcribe_timeout_ms = 10_000;
        config.orchestrator.think_timeout_ms = 10_000;
        config.orchestrator.speak_timeout_ms = 10_000;

        let detector = EnergyWakeDetector::new(config.wake.sensitivity, 16_000, 1600);
        let wake = marceline_core::WakeEngine::new(&config.wake, Box::new(detector));
        let model_path = format!("{}/models/silero_vad.onnx", env!("CARGO_MANIFEST_DIR"));
        let vad = SileroVad::load(&model_path).expect("load silero vad model");
        let endpointer = VadEndpointer::new(vad, DEFAULT_SPEECH_THRESHOLD);
        let mut gate = Gate::new(wake, endpointer, &config.vad);

        let wav_path = config_dir.path().join("canned.wav");
        build_canned_wav(&wav_path);
        let capture = marceline_core::audio::Capture::from_wav_file(&wav_path, 1.5, 1600)
            .expect("build canned Capture");
        let playback = marceline_core::Playback::null(24_000, 1);

        let soul_path = config_dir.path().join("SOUL.md");
        std::fs::write(&soul_path, "# Identity\n\nTest persona for the canned-audio harness.\n")
            .unwrap();
        let soul_watch_cancel = CancellationToken::new();
        let (soul_watcher, soul_watch_handle) = marceline_core::soul_watch::watch(
            soul_path,
            Duration::from_secs(3600),
            soul_watch_cancel.clone(),
        );

        let current_run: CurrentRun = Arc::new(Mutex::new(None));
        let voice = VoiceId::from("af_sky");

        // EPIC 12.3's real per-turn wake→first-audio timestamps are
        // captured *inside* `run_loop` and only ever leave it via a
        // tracing log line — there is no return value or channel this
        // test can read them from. `turn_start` is a proxy measured from
        // outside instead: real-world clock time from just before the
        // gate starts consuming the canned wav to the fake TTS server
        // receiving text. It is not exactly wake→first-audio (it also
        // includes however long the gate takes to consume the fixture's
        // leading frames before `EnergyWakeDetector` fires), but it is
        // the same order of magnitude and regresses the same way a real
        // slowdown would, which is what a CI assertion needs.
        let turn_start = Instant::now();

        // `run_loop` is `?Send` throughout (its `Stages` impl uses
        // `#[async_trait(?Send)]`, per this file's module docs), so it
        // cannot be `tokio::spawn`ed — `select!` polls it and the
        // assertion below within this same task instead, which needs no
        // `Send` bound at all. `run_loop` never returns on its own (it's
        // `'turn: loop { ... }`), so whichever branch finishes first is
        // always the assertion; the dropped `run_loop` future is simply
        // never polled again, same effect as an abort.
        let wait_for_tts = async {
            // Poll for the fake TTS server to have received the LLM's
            // reply — this is the harness's "a response was produced"
            // assertion: it can only have that text if wake fired, VAD
            // endpointed the utterance, the fake STT's transcript reached
            // the LLM, and the LLM's streamed reply reached TTS, in that
            // order, through the real `run_loop`.
            let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
            loop {
                if !tts_received.lock().expect("received lock poisoned").is_empty() {
                    return;
                }
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "timed out waiting for a turn to reach TTS"
                );
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };

        tokio::select! {
            result = run_loop(
                &mut gate,
                &capture,
                &playback,
                &stt_socket,
                &tts_socket,
                "en",
                &voice,
                &config,
                &soul_watcher,
                &current_run,
            ) => {
                panic!("run_loop returned unexpectedly: {result:?}");
            }
            _ = wait_for_tts => {}
        }

        let spoken = tts_received.lock().expect("received lock poisoned").clone();
        assert!(
            spoken.iter().any(|text| text.contains("sunny")),
            "expected the LLM's reply to reach TTS, got {spoken:?}"
        );

        // EPIC 12.3's ≤1.5s wake→first-audio target, asserted against the
        // proxy timing documented above — this is the CI check that
        // story's own doc comment named as EPIC 12.4's job. A fake
        // STT/TTS/LLM stack answering near-instantly should clear the
        // real target with room to spare; if this ever starts flaking on
        // CI hardware, the target is what needs revisiting, not this
        // assertion.
        let elapsed_ms = turn_start.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms <= marceline_core::MAX_WAKE_TO_FIRST_AUDIO_MS,
            "canned-audio turn took {elapsed_ms}ms, over the {}ms wake→first-audio target",
            marceline_core::MAX_WAKE_TO_FIRST_AUDIO_MS
        );

        soul_watch_cancel.cancel();
        let _ = soul_watch_handle.await;
    }
}
