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
    compile_prompt_with_retrieval, compile_system_prompt, ensure_current_embed_model,
    recent_context, register_mcp_tools, resolve_max_iterations, sentence_chunk, think, ChatEvent,
    ChatEventStream, Config, ConversationEvent, ConversationState, DeclineAll, EmbedError,
    EnergyWakeDetector, FailedStage, Gate, GateOutput, GetTimeTool, GrpcTtsEngine, HealthView,
    HistoryError, HistoryStore, LlmEngine, LlmSummarizer, ListDirTool, MemoryError, MiniLmEmbedder,
    NewTurn, OpenAiCompatibleEngine, Orchestrator, Playback, ReadFileTool, SileroVad, SttManager,
    Stages, Summarizer, ToolBroker, TtsEngine, TurnBuffer, Trust, VadEndpointer, VoiceId,
    WebSearchTool, DEFAULT_SPEECH_THRESHOLD,
};
use marceline_core::audio::Capture;
use tokio::sync::{watch, Mutex as AsyncMutex, RwLock};
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
    /// Opening the history store failed.
    #[error(transparent)]
    History(#[from] HistoryError),
    /// Loading the embedding model failed.
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// Checking/re-embedding the long-term memory index at startup failed.
    #[error(transparent)]
    Memory(#[from] MemoryError),
    /// The `read_file`/`list_dir` sandbox root could not be resolved.
    #[error("failed to resolve the tool sandbox root: {0}")]
    ToolSandboxRoot(#[from] std::io::Error),
}

/// A short, fixed message spoken on any non-TTS stage failure (SPEC.md
/// §9.11, EPIC 8.3). v1 does not attempt to tailor this per failure —
/// just make sure the user hears *something* rather than silence.
const GRACEFUL_ERROR_MESSAGE: &str = "Sorry, I ran into a problem with that. Please try again.";

/// The `StageError` `reason` used when [`SessionGuard`] refuses a request
/// (`EngineError::GuardrailRefused`, SPEC.md §4.5) — an exact-match marker
/// rather than the error's own `to_string()`, so [`ErrorSpeaker`] can tell
/// this apart from a genuine fault and speak [`SESSION_LIMIT_MESSAGE`]
/// instead of [`GRACEFUL_ERROR_MESSAGE`].
const SESSION_LIMIT_REASON: &str = "session request limit reached";

/// Spoken in place of [`GRACEFUL_ERROR_MESSAGE`] when the LLM stage failed
/// specifically because [`SessionGuard`] hit `max_requests_per_session` —
/// a budget cap, not a fault, so it says so rather than the generic
/// "I ran into a problem" (SPEC.md §4.5, EPIC 4.5).
const SESSION_LIMIT_MESSAGE: &str =
    "I've reached my request limit for this session. Please try again in a bit.";

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
        let message = if reason == SESSION_LIMIT_REASON {
            SESSION_LIMIT_MESSAGE
        } else {
            GRACEFUL_ERROR_MESSAGE
        };
        let text_stream: marceline_core::TextStream =
            Box::pin(futures::stream::once(async move { Ok(message.to_string()) }));
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

/// How long `[llm].max_requests_per_session` counts against before the
/// count rolls over on its own (SPEC.md §4.5) — see
/// `converse_ex`'s `session_guard_state`.
const SESSION_GUARD_WINDOW: Duration = Duration::from_secs(60 * 60);

/// The one session every `converse` turn belongs to (EPIC 10.2/10.4). v1 has
/// no notion of multiple concurrent conversations — a single fixed id (not
/// one minted fresh per process start) is what lets a restarted daemon find
/// its own prior turns again via [`recent_context`].
const DEFAULT_SESSION_ID: &str = "default";

/// How many of the most recent persisted turns seed a fresh [`TurnBuffer`]
/// on daemon startup (EPIC 10.2).
const RECENT_CONTEXT_TURN_LIMIT: usize = 50;

/// How many long-term memories [`compile_prompt_with_retrieval`] pulls in
/// per turn (EPIC 10.5).
const MEMORY_RETRIEVAL_K: usize = 5;

/// How many recent turns the background summarizer distills per run
/// (EPIC 10.4).
const SUMMARY_TURN_LIMIT: usize = 20;

/// Token cap on the summarizer's own distillation response — a short
/// standing fact, not a full transcript.
const SUMMARIZER_MAX_TOKENS: u32 = 200;

/// Default directory `MiniLmEmbedder::load` reads `model.onnx` +
/// `tokenizer.json` from, relative to this crate — mirrors `memory.rs`'s
/// identical helper and `converse.rs`'s own `models/silero_vad.onnx`
/// convention.
fn default_embed_model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/all-MiniLM-L6-v2")
}

/// Root `read_file`/`list_dir` are confined to (issue #185): explicit and
/// small rather than the whole filesystem. `MARCELINE_TOOLS_ROOT`
/// overrides it; otherwise it's the current working directory — the
/// directory the operator actually launched the daemon from, not
/// something implicit like `$HOME`.
fn tool_sandbox_root() -> std::io::Result<PathBuf> {
    match std::env::var_os("MARCELINE_TOOLS_ROOT") {
        Some(dir) => Ok(PathBuf::from(dir)),
        None => std::env::current_dir(),
    }
}

/// Current Unix epoch milliseconds, for [`NewTurn::timestamp_ms`].
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_millis() as i64
}

/// Persists `turn` through [`HistoryStore`]'s write actor from a
/// [`tokio::task::spawn_blocking`] task, per `log_turn`'s own doc comment —
/// logs and swallows a failure rather than propagating one, since a history
/// write failing should never take down the live conversation turn.
async fn log_turn_async(store: &HistoryStore, turn: NewTurn) {
    let store = store.clone();
    match tokio::task::spawn_blocking(move || store.log_turn(turn)).await {
        Ok(Ok(_id)) => {}
        Ok(Err(err)) => tracing::error!(%err, "failed to log turn to history"),
        Err(err) => tracing::error!(%err, "log_turn task panicked"),
    }
}

/// Runs the MVP loop forever: wake, listen, transcribe, think, speak,
/// back to idle. Returns only on an unrecoverable setup failure (a worker
/// or device that never came up) — a mid-turn stage failure routes
/// through the orchestrator's `ERROR` edge and the loop keeps running.
///
/// Equivalent to `converse_ex(config_path, soul_path, None)` — the plain
/// interactive path with no control socket and no daemon-style SIGTERM
/// ordering (`ctrl-c` alone drives shutdown, as before).
pub async fn converse(config_path: &Path, soul_path: &Path) -> Result<(), ConverseError> {
    converse_ex(config_path, soul_path, None).await
}

/// How long a daemon-mode shutdown waits for playback to drain and workers
/// to exit before giving up and returning anyway (SPEC.md §2.5.1 step 6).
/// The supervisor's own `kill()` (EPIC 0.6) is what actually reclaims a
/// straggling worker process; this just bounds how long this function
/// waits to observe that happening before it returns.
const SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// [`converse`], with an optional `control_socket` (EPIC 11.1's daemon
/// mode, driven by `marceline start`/`stop`/`status`).
///
/// When `control_socket` is `Some`, this additionally:
/// - serves `marceline status` queries over that socket
///   ([`marceline_core::daemon::serve_status`]) for the process's lifetime;
/// - on SIGTERM (`marceline stop`, SPEC.md §11.1), runs the graceful
///   shutdown ordering in the exact sequence the story specifies: (1) fire
///   the run cancel token; (2) each side-effecting tool's own kill logic
///   already rides that same cancellation (EPIC 6's tool broker propagates
///   it, nothing further to do here); (3) flush + stop audio out; (4)
///   checkpoint memory/history — a no-op, since every turn's write already
///   commits synchronously through `HistoryStore`'s write actor as it
///   happens (EPIC 10.1/10.2);
///   (5) signal the STT/TTS workers to exit; (6) wait (bounded by
///   [`SHUTDOWN_DRAIN_TIMEOUT`]) for that to land, then return so the
///   process can exit — a still-running child is hard-killed by the
///   supervisor's own shutdown path (EPIC 0.6), not by this function.
pub async fn converse_ex(
    config_path: &Path,
    soul_path: &Path,
    control_socket: Option<&Path>,
) -> Result<(), ConverseError> {
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

    // Shared across every turn's `SessionGuard` (SPEC.md §4.5) so
    // `max_requests_per_session` is enforced across the daemon's whole
    // run rather than resetting to zero every turn (a fresh engine, and so
    // a fresh guard, is built each turn to carry that turn's own
    // cancellation token — see `run_loop`'s `SessionGuard::with_state`
    // call). A rolling window, not a permanent counter: once
    // `SESSION_GUARD_WINDOW` has elapsed since the cap was last hit, the
    // count resets on its own, so the daemon recovers without a restart.
    let session_guard_state = Arc::new(marceline_core::SessionGuardState::with_rolling_window(
        Some(SESSION_GUARD_WINDOW),
    ));

    // Built once at daemon start, exactly as `say_to_llm` does (EPIC 6/6.4):
    // v1's read-only built-ins plus whatever MCP servers are configured.
    // Shared (`Arc`) across every turn's `think` call below rather than
    // rebuilt per turn — an MCP server's connection is not cheap to reopen,
    // and there is no reason to.
    let mut broker = ToolBroker::new();
    broker
        .register(Arc::new(GetTimeTool))
        .expect("get_time is the first registration");
    let tool_sandbox_root = tool_sandbox_root()?;
    broker
        .register(Arc::new(ReadFileTool::new(marceline_core::tools::sandbox::Sandbox::new(
            &tool_sandbox_root,
        )?)))
        .expect("read_file is the first registration");
    broker
        .register(Arc::new(ListDirTool::new(marceline_core::tools::sandbox::Sandbox::new(
            &tool_sandbox_root,
        )?)))
        .expect("list_dir is the first registration");
    broker
        .register(Arc::new(WebSearchTool::new()?))
        .expect("web_search is the first registration");
    for skipped in register_mcp_tools(&mut broker, &config.mcp).await {
        tracing::warn!(server = %skipped, "mcp server unavailable, continuing without it");
    }
    let broker = Arc::new(broker);

    // History/memory (EPIC 10): opened once here and threaded through
    // `run_loop`, so the conversation loop actually persists what it hears
    // and says instead of forgetting it on the next turn or a restart.
    let history_store = {
        let db_path = config.memory.expanded_db_path();
        tokio::task::spawn_blocking(move || HistoryStore::open(db_path))
            .await
            .expect("history store open task panicked")?
    };
    let mut turn_buffer = {
        let store = history_store.clone();
        tokio::task::spawn_blocking(move || {
            recent_context(&store, DEFAULT_SESSION_ID, RECENT_CONTEXT_TURN_LIMIT)
        })
        .await
        .expect("recent_context task panicked")
        .map(TurnBuffer::from_turns)?
    };
    // Only built when `[memory].longterm` is on (EPIC 10.5) — a missing
    // model directory then fails startup with a clear error rather than
    // silently running without long-term memory.
    let embed_pipeline: Option<Arc<AsyncMutex<MiniLmEmbedder>>> = if config.memory.longterm {
        let model_dir = default_embed_model_dir();
        let mut pipeline = MiniLmEmbedder::load(&model_dir, config.memory.embed_model.clone())?;
        let store_for_check = history_store.clone();
        tokio::task::spawn_blocking(move || -> Result<MiniLmEmbedder, MemoryError> {
            ensure_current_embed_model(&store_for_check, &mut pipeline)?;
            Ok(pipeline)
        })
        .await
        .expect("ensure_current_embed_model task panicked")
        .map(|pipeline| Some(Arc::new(AsyncMutex::new(pipeline))))?
    } else {
        None
    };

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
    let stt_health_for_status = Arc::clone(&stt_health);
    // Wrapped in `Arc` so daemon mode's control socket (EPIC 11.2's live
    // `config set stt.model`/`stt.backend` swap) can hold its own handle
    // onto the same manager `run_loop` connects fresh clients to each
    // turn — `swap_model` takes `&self`, so both sides can call it
    // concurrently without extra locking here.
    let stt_manager = Arc::new(
        SttManager::start(
            &config.stt,
            stt_paths,
            stt_health,
            stt_shutdown_rx,
            CancellationToken::new(),
        )
        .await?,
    );

    let tts_paths = TtsWorkerPaths::for_backend(&config.tts.backend);
    let tts_socket = tts_paths.socket_path.clone();
    let (tts_shutdown_tx, tts_shutdown_rx) = watch::channel(false);
    let tts_health: HealthView = Arc::new(RwLock::new(HashMap::new()));
    let tts_health_for_status = Arc::clone(&tts_health);
    let _tts_launch = marceline_core::launch_tts_worker(
        &config.tts,
        tts_paths,
        tts_health,
        tts_shutdown_rx,
        CancellationToken::new(),
    )
    .await?;
    let voice = VoiceId::from(config.tts.voice.as_str());

    // Reported by `run_loop` on every state transition; `marceline status`
    // (over `control_socket`, when daemon mode is on) reads the receiver
    // side. Starts at `Idle` — `run_loop`'s first iteration republishes it
    // immediately anyway.
    let (state_tx, state_rx) = watch::channel(ConversationState::Idle);
    let control_task = control_socket.map(|socket_path| {
        let socket_path = socket_path.to_path_buf();
        let stt_manager_for_status = Arc::clone(&stt_manager);
        tokio::spawn(async move {
            if let Err(err) = marceline_core::daemon::serve_control(
                &socket_path,
                stt_health_for_status,
                tts_health_for_status,
                state_rx,
                Some(stt_manager_for_status),
            )
            .await
            {
                tracing::error!(%err, "control socket stopped serving");
            }
        })
    });

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

    // SIGTERM drives the graceful ordering below (SPEC.md §2.5.1, EPIC
    // 11.1's `marceline stop`); ctrl-c above stays the interactive
    // "cancel this turn, or exit if idle" shortcut it always was.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to install SIGTERM handler");

    let result = tokio::select! {
        res = run_loop(
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
            &state_tx,
            &session_guard_state,
            &broker,
            &history_store,
            embed_pipeline.as_ref(),
            &mut turn_buffer,
        ) => res,
        _ = sigterm.recv() => {
            tracing::info!("SIGTERM received; running graceful shutdown ordering (SPEC.md §2.5.1)");
            // (1) fire the run cancel token for whatever turn is in flight
            // — idle means there is nothing to cancel.
            if let Some(token) = current_run.lock().expect("current_run lock poisoned").clone() {
                token.cancel();
            }
            // (2) each side-effecting tool's own kill logic rides that
            // same cancellation (EPIC 6's tool broker), so there is
            // nothing further to do here.
            // (3) flush + stop audio out.
            while playback.buffered_samples() > 0 {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Ok(())
        }
    };

    // (4) checkpoint memory/history to SQLite: a no-op here — every turn's
    // history write already commits synchronously through `HistoryStore`'s
    // write actor as it happens (EPIC 10.1/10.2), so there is nothing left
    // to flush at shutdown.
    // (5) signal the STT/TTS workers to exit.
    let _ = stt_shutdown_tx.send(true);
    let _ = tts_shutdown_tx.send(true);
    // (6) wait, bounded, for that to land; a still-running child is
    // hard-killed by the supervisor's own shutdown path (EPIC 0.6), not by
    // this function.
    let _ = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, tokio::time::sleep(Duration::from_millis(200))).await;
    soul_watch_cancel.cancel();
    let _ = soul_watch_handle.await;
    if let Some(task) = control_task {
        task.abort();
    }
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
    state_tx: &watch::Sender<ConversationState>,
    session_guard_state: &Arc<marceline_core::SessionGuardState>,
    broker: &Arc<ToolBroker>,
    history_store: &HistoryStore,
    embed_pipeline: Option<&Arc<AsyncMutex<MiniLmEmbedder>>>,
    turn_buffer: &mut TurnBuffer,
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
        state_tx.send_replace(orchestrator.state());
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
                state_tx.send_replace(orchestrator.state());
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
        // one fixed for the process lifetime. Wrapped in `SessionGuard`
        // (§4.5) every turn too — a fresh guard each time, since the guard
        // itself is generic over the (per-turn) engine it wraps, but
        // sharing `session_guard_state` is what makes
        // `max_requests_per_session` actually count across turns instead
        // of resetting to zero along with the engine.
        let llm = marceline_core::SessionGuard::with_state(
            OpenAiCompatibleEngine::new(&config.llm, run_token.clone())?,
            config.llm.max_tokens_per_turn,
            config.llm.max_requests_per_session,
            Arc::clone(session_guard_state),
        );

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
        state_tx.send_replace(orchestrator.state());

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
        state_tx.send_replace(orchestrator.state());

        // THINKING: only `Final` transcripts ever reach here (§2.4.1).
        // Recompiled from the watcher's latest persona every turn, so a
        // SOUL.md save takes effect on the very next turn (EPIC 9.2). When
        // `[memory].longterm` is on, long-term memory is retrieved and
        // folded in too (EPIC 10.5); a retrieval failure falls back to the
        // plain persona prompt rather than failing the whole turn.
        let persona = soul_watcher.current();
        let system_prompt = match embed_pipeline {
            Some(pipeline) => {
                let mut pipeline = pipeline.lock().await;
                match compile_prompt_with_retrieval(
                    history_store,
                    &mut *pipeline,
                    &persona.render(),
                    &transcript,
                    MEMORY_RETRIEVAL_K,
                ) {
                    Ok(prompt) => prompt,
                    Err(err) => {
                        tracing::error!(%err, "memory retrieval failed; falling back to plain system prompt");
                        compile_system_prompt(&persona.render(), &[])
                    }
                }
            }
            None => compile_system_prompt(&persona.render(), &[]),
        };

        // Persist the user's turn (EPIC 10.1/10.2) and fold it into the
        // working context before building the request — a restart later
        // resumes with this same turn via `recent_context`.
        turn_buffer.push_user(transcript.clone());
        log_turn_async(
            history_store,
            NewTurn {
                session_id: DEFAULT_SESSION_ID.to_string(),
                timestamp_ms: now_ms(),
                role: "user".to_string(),
                text: transcript.clone(),
                provenance: Trust::User,
                interrupted: false,
            },
        )
        .await;

        let context_window = llm.info().context_window;
        let messages = turn_buffer.messages_for_request(&system_prompt, context_window);
        let policy = persona.tool_policy();
        let tools = broker.catalog();
        let max_iterations = resolve_max_iterations(config.llm.max_tool_iterations_per_turn);
        let max_tokens = config.llm.max_tokens_per_turn;

        // `think` (EPIC 6.3) drives the whole tool-call loop itself instead
        // of just returning a `ChatEventStream`, so it runs on its own
        // task; its `on_text` callback feeds an mpsc channel wrapped as a
        // `ChatEventStream` — the same seam [`sentence_chunk`] already
        // consumes — so speech still starts on the first sentence rather
        // than waiting for the whole tool loop (and every tool call in it)
        // to finish. The channel's sender lives only inside `think`'s own
        // closure, so it drops (ending the stream, flushing any trailing
        // partial sentence) the instant `think` returns.
        let (text_tx, mut text_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatEvent, marceline_core::EngineError>>();
        let think_broker = Arc::clone(broker);
        let think_cancel = run_token.clone();
        let think_task = tokio::spawn(async move {
            think(
                &llm,
                &think_broker,
                messages,
                tools,
                &policy,
                max_tokens,
                max_iterations,
                think_cancel,
                // No real voice-confirmation path exists yet (EPIC 6.5/9.3
                // built the seam, nothing speaks a prompt yet); every real
                // tool today is `ReadOnly` (§10) so this is never actually
                // consulted — fail closed if that ever changes.
                &DeclineAll,
                move |delta: &str| {
                    let _ = text_tx.send(Ok(ChatEvent::TextDelta(delta.to_string())));
                },
            )
            .await
        });
        let events: ChatEventStream =
            Box::pin(futures::stream::poll_fn(move |cx| text_rx.poll_recv(cx)));
        let mut sentences = sentence_chunk(events);

        // First sentence pulled eagerly, under `think_timeout`: this is
        // the "first TTS chunk" trigger, so the transition into Speaking
        // is driven by actually having something to say, not by entering
        // Thinking. A stuck/dead LLM shows up here as a timeout.
        let first_sentence = match tokio::time::timeout(think_timeout, sentences.next()).await {
            Ok(Some(Ok(text))) => text,
            Ok(Some(Err(err))) => {
                // A `SessionGuard` refusal (§4.5) is a budget cap, not a
                // fault — spoken as such rather than the generic error
                // message, via the exact-match reason `ErrorSpeaker` looks
                // for.
                let reason = if err.is_guardrail_refused() {
                    SESSION_LIMIT_REASON.to_string()
                } else {
                    err.to_string()
                };
                let _ = orchestrator
                    .apply(ConversationEvent::StageError {
                        stage: FailedStage::Llm,
                        reason,
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
        // Tees each sentence into `full_reply` as it is pulled for TTS, so
        // the final (possibly partial, if cancelled mid-speech) reply text
        // is available to log below without buffering it twice.
        let full_reply = Arc::new(Mutex::new(first_sentence.clone()));
        let full_reply_for_stream = Arc::clone(&full_reply);
        let rest: marceline_core::TextStream = Box::pin(sentences.inspect(move |item| {
            if let Ok(text) = item {
                full_reply_for_stream
                    .lock()
                    .expect("full_reply lock poisoned")
                    .push_str(text);
            }
        }));
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
        state_tx.send_replace(orchestrator.state());
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
                        // Partial-state policy (§2.5.1): the reply logged
                        // below is marked `interrupted` rather than dropped.
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
        state_tx.send_replace(orchestrator.state());

        // The tool loop finished speaking its text well before this point
        // (it only needs to have produced *some* text, not have returned),
        // so this join is normally instant — it exists to surface an
        // iteration-cap warning or a `think` failure that never showed up
        // as a `ChatEvent` (e.g. a cancellation between tool-call rounds).
        match think_task.await {
            Ok(Ok((outcome, _messages))) if outcome.iteration_cap_hit => {
                tracing::warn!(max_iterations, "tool iteration cap hit; forced a final answer");
            }
            Ok(Ok(_)) => {}
            Ok(Err(err)) => tracing::error!(%err, "thinking loop failed"),
            Err(err) => tracing::error!(%err, "think task panicked"),
        }

        // Persist the assistant's turn (EPIC 10.1/10.2), interrupted or
        // not, and fold it into the working context for the next turn.
        let reply_text = full_reply
            .lock()
            .expect("full_reply lock poisoned")
            .clone();
        if !reply_text.is_empty() {
            turn_buffer.push_assistant(reply_text.clone());
            log_turn_async(
                history_store,
                NewTurn {
                    session_id: DEFAULT_SESSION_ID.to_string(),
                    timestamp_ms: now_ms(),
                    role: "assistant".to_string(),
                    text: reply_text,
                    provenance: Trust::Assistant,
                    interrupted: cancelled,
                },
            )
            .await;

            // Distill this session's recent turns into a durable memory in
            // the background (EPIC 10.4) — off the turn path, so it never
            // adds latency to a spoken reply.
            if let Some(pipeline) = embed_pipeline.cloned() {
                let store = history_store.clone();
                let llm_config = config.llm.clone();
                tokio::spawn(async move {
                    let engine = match OpenAiCompatibleEngine::new(&llm_config, CancellationToken::new())
                    {
                        Ok(engine) => engine,
                        Err(err) => {
                            tracing::error!(%err, "background summarizer could not build an llm engine");
                            return;
                        }
                    };
                    let summarizer = LlmSummarizer::new(engine, SUMMARIZER_MAX_TOKENS);
                    // Not `summarize_session` directly: that function holds
                    // its `&mut dyn EmbeddingPipeline` argument across its
                    // own internal `.await`, which makes the resulting
                    // future `!Send` and unspawnable — inlined here instead,
                    // so the concrete `MiniLmEmbedder` guard is only
                    // touched by the synchronous `store_memory` call, never
                    // held across an `.await` point.
                    let turns = match store.recent_turns(DEFAULT_SESSION_ID, SUMMARY_TURN_LIMIT) {
                        Ok(turns) => turns,
                        Err(err) => {
                            tracing::error!(%err, "background summarizer could not read recent turns");
                            return;
                        }
                    };
                    if turns.is_empty() {
                        return;
                    }
                    let provenance = marceline_core::derive_provenance(&turns);
                    let summary = match summarizer.summarize(&turns).await {
                        Ok(summary) => summary,
                        Err(err) => {
                            tracing::error!(%err, "background summarization failed");
                            return;
                        }
                    };
                    let mut pipeline = pipeline.lock().await;
                    if let Err(err) =
                        marceline_core::store_memory(&store, &mut *pipeline, summary, provenance, now_ms())
                    {
                        tracing::error!(%err, "failed to store distilled memory");
                    }
                });
            }
        }
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
