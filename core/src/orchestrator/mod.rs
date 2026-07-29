//! The conversation orchestrator state machine (SPEC.md §2.5, EPIC 8.1).
//!
//! Sequences one conversation turn across every stage: `IDLE -> LISTENING
//! -> TRANSCRIBING -> THINKING -> SPEAKING -> IDLE`, with an `ERROR` edge
//! reachable from any state and a barge-in edge from `THINKING`/`SPEAKING`
//! back to `LISTENING`. There is deliberately no separate `RESPONDING`
//! state: the first streamed TTS chunk is what flips `THINKING ->
//! SPEAKING`, since "generating" and "playing" are the same state once
//! tokens stream continuously into playback (§2.5).
//!
//! This story only builds the skeleton: the state enum, the legal
//! transition table, and a place for each transition to invoke its stage
//! ([`Stages`]) plus the per-turn [`CancellationToken`] (EPIC 8.4) and the
//! `ERROR` routing (EPIC 8.3). Real end-to-end wiring lands in 8.2; this
//! only needs to prove the machine can be driven through a full cycle and
//! that illegal transitions are rejected rather than silently ignored.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

/// A state the conversation can be in (SPEC.md §2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState {
    /// Waiting for a wake word; nothing in flight.
    Idle,
    /// Wake word fired; collecting an utterance via the wake/VAD gate (EPIC 2).
    Listening,
    /// VAD ended the utterance; STT (EPIC 3) is transcribing it.
    Transcribing,
    /// Final transcript handed to the LLM (EPIC 4); may loop on tool calls
    /// (EPIC 6) until a final answer streams out.
    Thinking,
    /// Streamed tokens are sentence-chunked into TTS (EPIC 5) and playing.
    Speaking,
    /// A stage failed or timed out (EPIC 8.3); routes to a graceful spoken
    /// message (or silence, if TTS itself failed) then back to `Idle`.
    Error,
}

/// Which stage failed or timed out (SPEC.md §2.5, EPIC 8.3).
///
/// Distinguished from a plain string reason because [`Stages::on_enter_error`]
/// must special-case [`FailedStage::Tts`]: every other failure gets a
/// spoken graceful message, but if TTS itself is what failed, no spoken
/// message is possible, so `ERROR` logs and returns to `Idle` silently
/// (accepted, §9.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailedStage {
    /// The wake/VAD gate (EPIC 2) — e.g. `no-speech` timeout in `Listening`.
    Gate,
    /// Speech-to-text (EPIC 3) — worker-down, timeout, or empty/failed transcript.
    Stt,
    /// The LLM (EPIC 4) — request error or tool timeout in `Thinking`.
    Llm,
    /// Text-to-speech (EPIC 5) — worker-down or timeout in `Speaking`. The
    /// one stage whose own failure cannot be spoken.
    Tts,
}

/// An event that drives a transition between [`ConversationState`]s.
#[derive(Debug, Clone)]
pub enum ConversationEvent {
    /// Wake word detected (gate, EPIC 2.3).
    WakeWord,
    /// VAD ended the utterance (gate, EPIC 2.5).
    VadEnd,
    /// STT produced a final (non-empty) transcript (EPIC 3.3).
    FinalTranscript,
    /// The first TTS chunk is ready to play (EPIC 5.4).
    FirstTtsChunk,
    /// Playback of the reply finished.
    PlaybackDone,
    /// Wake/VAD gate detected barge-in speech while `Thinking`/`Speaking`
    /// (§2.5.1); cancels the in-flight stage via the run's
    /// [`CancellationToken`] and jumps back to `Listening`.
    BargeIn,
    /// A stage failed or timed out (EPIC 8.3). Carries which stage failed
    /// (so `ERROR` knows whether a spoken message is possible) and a
    /// human-readable reason for logging.
    StageError {
        /// Which stage failed or timed out.
        stage: FailedStage,
        /// Human-readable reason, for logging.
        reason: String,
    },
    /// The `ERROR` state finished handling the failure (message spoken, or
    /// logged silently if TTS itself was the failed stage, §9.11) and is
    /// ready to return to `Idle`.
    ErrorHandled,
}

/// A transition that was requested but is not legal from the current state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IllegalTransition {
    pub from: ConversationState,
    pub event: &'static str,
}

impl std::fmt::Display for IllegalTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "illegal transition: {:?} does not accept {}",
            self.from, self.event
        )
    }
}

impl std::error::Error for IllegalTransition {}

/// The per-stage hooks a transition invokes. Stubbed here; EPIC 8.2 wires
/// the real STT/LLM/TTS engines behind these.
///
/// A single `run` [`CancellationToken`] (EPIC 8.4) is threaded through
/// every call so barge-in / cancellation has one place to fire from and
/// every stage has one place to observe it.
///
/// `?Send`: a real implementation (EPIC 8.3) holds a reference to the
/// audio [`Playback`][crate::audio::Playback] sink to speak `ERROR`'s
/// graceful message, and `cpal`'s stream handle is not `Send` — the
/// orchestrator is driven from one task, never spawned across threads, so
/// nothing requires it.
#[async_trait(?Send)]
pub trait Stages {
    /// Entering `Listening`: the gate (EPIC 2) starts collecting an utterance.
    async fn on_enter_listening(&self, run: &CancellationToken);
    /// Entering `Transcribing`: hand the collected segment to STT (EPIC 3).
    async fn on_enter_transcribing(&self, run: &CancellationToken);
    /// Entering `Thinking`: hand the final transcript to the LLM (EPIC 4),
    /// including any tool-call loop (EPIC 6).
    async fn on_enter_thinking(&self, run: &CancellationToken);
    /// Entering `Speaking`: streamed tokens are already sentence-chunked
    /// into TTS (EPIC 5) by the time this fires.
    async fn on_enter_speaking(&self, run: &CancellationToken);
    /// Entering `Error`: speak a graceful message (EPIC 8.3), unless
    /// `stage` is [`FailedStage::Tts`], in which case this logs only (§9.11).
    async fn on_enter_error(&self, stage: FailedStage, reason: &str);
}

/// Drives one conversation loop through [`ConversationState`]s.
///
/// Owns the current state and the run's [`CancellationToken`] (created
/// fresh on every `Idle -> Listening` transition, per EPIC 8.4's
/// "one run token per turn" contract, §2.5.1).
pub struct Orchestrator<S: Stages> {
    state: ConversationState,
    stages: S,
    run_token: Option<CancellationToken>,
}

impl<S: Stages> Orchestrator<S> {
    /// Builds a new orchestrator, starting `Idle`.
    pub fn new(stages: S) -> Self {
        Self {
            state: ConversationState::Idle,
            stages,
            run_token: None,
        }
    }

    /// The current state.
    pub fn state(&self) -> ConversationState {
        self.state
    }

    /// The current turn's cancellation token, if a turn is in flight.
    pub fn run_token(&self) -> Option<&CancellationToken> {
        self.run_token.as_ref()
    }

    /// Feeds one event to the machine. Logs the transition on success;
    /// returns [`IllegalTransition`] instead of silently ignoring an event
    /// that doesn't apply to the current state (SPEC.md EPIC 8.1 "done when").
    pub async fn apply(
        &mut self,
        event: ConversationEvent,
    ) -> Result<ConversationState, IllegalTransition> {
        use ConversationEvent as E;
        use ConversationState as St;

        // The ERROR edge is reachable from every state except ERROR itself
        // (a second failure while already handling one just stays put).
        if let E::StageError { stage, reason } = &event {
            if self.state == St::Error {
                return Err(IllegalTransition {
                    from: self.state,
                    event: "StageError",
                });
            }
            if let Some(t) = self.run_token.take() {
                t.cancel();
            }
            let next = St::Error;
            self.stages.on_enter_error(*stage, reason).await;
            self.transition(next, "StageError");
            return Ok(next);
        }

        let next = match (self.state, &event) {
            (St::Idle, E::WakeWord) => St::Listening,
            (St::Listening, E::VadEnd) => St::Transcribing,
            (St::Transcribing, E::FinalTranscript) => St::Thinking,
            (St::Thinking, E::FirstTtsChunk) => St::Speaking,
            (St::Speaking, E::PlaybackDone) => St::Idle,
            (St::Thinking, E::BargeIn) | (St::Speaking, E::BargeIn) => St::Listening,
            (St::Error, E::ErrorHandled) => St::Idle,
            (from, event) => {
                return Err(IllegalTransition {
                    from,
                    event: event_name(event),
                })
            }
        };

        match (&event, next) {
            (E::WakeWord, St::Listening) => {
                let run = CancellationToken::new();
                self.stages.on_enter_listening(&run).await;
                self.run_token = Some(run);
            }
            (E::VadEnd, St::Transcribing) => {
                let run = self.run_token.clone().unwrap_or_default();
                self.stages.on_enter_transcribing(&run).await;
            }
            (E::FinalTranscript, St::Thinking) => {
                let run = self.run_token.clone().unwrap_or_default();
                self.stages.on_enter_thinking(&run).await;
            }
            (E::FirstTtsChunk, St::Speaking) => {
                let run = self.run_token.clone().unwrap_or_default();
                self.stages.on_enter_speaking(&run).await;
            }
            (E::BargeIn, St::Listening) => {
                // Barge-in fires the outgoing run's token (EPIC 8.4:
                // cancel in-flight stage + flush audio) before a fresh
                // one is minted for the re-armed utterance.
                if let Some(t) = self.run_token.take() {
                t.cancel();
            }
                let run = CancellationToken::new();
                self.stages.on_enter_listening(&run).await;
                self.run_token = Some(run);
            }
            (E::PlaybackDone, St::Idle) => {
                self.run_token = None;
            }
            _ => {}
        }

        self.transition(next, event_name(&event));
        Ok(next)
    }

    fn transition(&mut self, next: ConversationState, event: &str) {
        tracing::info!(from = ?self.state, to = ?next, event, "conversation state transition");
        self.state = next;
    }
}

fn event_name(event: &ConversationEvent) -> &'static str {
    match event {
        ConversationEvent::WakeWord => "WakeWord",
        ConversationEvent::VadEnd => "VadEnd",
        ConversationEvent::FinalTranscript => "FinalTranscript",
        ConversationEvent::FirstTtsChunk => "FirstTtsChunk",
        ConversationEvent::PlaybackDone => "PlaybackDone",
        ConversationEvent::BargeIn => "BargeIn",
        ConversationEvent::StageError { .. } => "StageError",
        ConversationEvent::ErrorHandled => "ErrorHandled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct FakeStages {
        listening: AtomicUsize,
        transcribing: AtomicUsize,
        thinking: AtomicUsize,
        speaking: AtomicUsize,
        error: AtomicUsize,
    }

    #[async_trait(?Send)]
    impl Stages for FakeStages {
        async fn on_enter_listening(&self, _run: &CancellationToken) {
            self.listening.fetch_add(1, Ordering::SeqCst);
        }
        async fn on_enter_transcribing(&self, _run: &CancellationToken) {
            self.transcribing.fetch_add(1, Ordering::SeqCst);
        }
        async fn on_enter_thinking(&self, _run: &CancellationToken) {
            self.thinking.fetch_add(1, Ordering::SeqCst);
        }
        async fn on_enter_speaking(&self, _run: &CancellationToken) {
            self.speaking.fetch_add(1, Ordering::SeqCst);
        }
        async fn on_enter_error(&self, _stage: FailedStage, _reason: &str) {
            self.error.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn drives_a_full_idle_to_idle_cycle() {
        let mut orch = Orchestrator::new(FakeStages::default());
        assert_eq!(orch.state(), ConversationState::Idle);

        assert_eq!(
            orch.apply(ConversationEvent::WakeWord).await.unwrap(),
            ConversationState::Listening
        );
        assert!(orch.run_token().is_some());

        assert_eq!(
            orch.apply(ConversationEvent::VadEnd).await.unwrap(),
            ConversationState::Transcribing
        );
        assert_eq!(
            orch.apply(ConversationEvent::FinalTranscript).await.unwrap(),
            ConversationState::Thinking
        );
        assert_eq!(
            orch.apply(ConversationEvent::FirstTtsChunk).await.unwrap(),
            ConversationState::Speaking
        );
        assert_eq!(
            orch.apply(ConversationEvent::PlaybackDone).await.unwrap(),
            ConversationState::Idle
        );
        assert!(orch.run_token().is_none());

        assert_eq!(orch.stages.listening.load(Ordering::SeqCst), 1);
        assert_eq!(orch.stages.transcribing.load(Ordering::SeqCst), 1);
        assert_eq!(orch.stages.thinking.load(Ordering::SeqCst), 1);
        assert_eq!(orch.stages.speaking.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn illegal_transitions_are_rejected_not_ignored() {
        let mut orch = Orchestrator::new(FakeStages::default());

        // Can't VadEnd from Idle.
        let err = orch.apply(ConversationEvent::VadEnd).await.unwrap_err();
        assert_eq!(err.from, ConversationState::Idle);
        assert_eq!(err.event, "VadEnd");
        // State unchanged.
        assert_eq!(orch.state(), ConversationState::Idle);

        // Can't barge-in from Idle either.
        let err = orch.apply(ConversationEvent::BargeIn).await.unwrap_err();
        assert_eq!(err.from, ConversationState::Idle);
        assert_eq!(orch.state(), ConversationState::Idle);
    }

    #[tokio::test]
    async fn barge_in_cancels_the_outgoing_run_token_and_returns_to_listening() {
        let mut orch = Orchestrator::new(FakeStages::default());
        orch.apply(ConversationEvent::WakeWord).await.unwrap();
        orch.apply(ConversationEvent::VadEnd).await.unwrap();
        orch.apply(ConversationEvent::FinalTranscript).await.unwrap();

        let stale_token = orch.run_token().unwrap().clone();
        assert!(!stale_token.is_cancelled());

        assert_eq!(
            orch.apply(ConversationEvent::BargeIn).await.unwrap(),
            ConversationState::Listening
        );
        assert!(stale_token.is_cancelled());
        assert!(!orch.run_token().unwrap().is_cancelled());
    }

    #[tokio::test]
    async fn stage_error_routes_through_error_then_back_to_idle() {
        let mut orch = Orchestrator::new(FakeStages::default());
        orch.apply(ConversationEvent::WakeWord).await.unwrap();
        let run = orch.run_token().unwrap().clone();

        assert_eq!(
            orch.apply(ConversationEvent::StageError {
                stage: FailedStage::Stt,
                reason: "stt worker down".into(),
            })
            .await
            .unwrap(),
            ConversationState::Error
        );
        assert!(run.is_cancelled());
        assert_eq!(orch.stages.error.load(Ordering::SeqCst), 1);

        assert_eq!(
            orch.apply(ConversationEvent::ErrorHandled).await.unwrap(),
            ConversationState::Idle
        );
    }

    #[tokio::test]
    async fn error_while_already_in_error_is_illegal() {
        let mut orch = Orchestrator::new(FakeStages::default());
        orch.apply(ConversationEvent::WakeWord).await.unwrap();
        orch.apply(ConversationEvent::StageError {
            stage: FailedStage::Llm,
            reason: "boom".into(),
        })
        .await
        .unwrap();

        let err = orch
            .apply(ConversationEvent::StageError {
                stage: FailedStage::Llm,
                reason: "boom again".into(),
            })
            .await
            .unwrap_err();
        assert_eq!(err.from, ConversationState::Error);
    }
}
