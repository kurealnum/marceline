//! The wake+VAD gate state machine (SPEC.md §2.1, §2.5, §2.6, EPIC 2.3):
//! `IDLE -> LISTENING -> (collect utterance) -> emit audio segment`. Turns
//! always-on mic frames into clean, bounded audio segments the
//! orchestrator (EPIC 8) and STT (EPIC 3) consume directly.
//!
//! `THINKING`/`SPEAKING` (EPIC 7.1) are orchestrator-driven states entered
//! via `enter_thinking`/`enter_speaking`: the gate keeps running the wake
//! detector on every frame so it never goes deaf mid-turn, but doesn't
//! collect an utterance or run VAD/STT until back in IDLE/LISTENING. A
//! wake detection in these states is surfaced as `GateOutput::WakeDetected`
//! for EPIC 7.2 to act on later; this story only keeps the gate listening.
//!
//! Utterance capture is seeded from the capture pre-roll ring (§2.6): the
//! wake-firing chunk is prepended with whatever the pre-roll already
//! holds, so a same-breath command isn't lost to the ~300ms IDLE ->
//! LISTENING flip.
//!
//! Endpointing thresholds (`silence_ms`, `min_utterance_ms`,
//! `max_utterance_ms`) are read from `[vad]` config (EPIC 2.5) rather
//! than hardcoded: `silence_ms` ends the utterance once speech has been
//! heard; `min_utterance_ms` discards blips too short to be real speech;
//! `max_utterance_ms` force-emits a stuck/over-long segment rather than
//! collecting forever.

pub mod barge_in;

use crate::audio::resample;
use crate::config::VadConfig;
use crate::vad::FRAME_SAMPLES;
use crate::wake::WakeEngine;
use crate::{AudioChunk, VadEndpointer};

/// Which state the gate is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Not listening; running the wake detector on every frame.
    Idle,
    /// Collecting an utterance following a wake event.
    Listening,
    /// Orchestrator (EPIC 8) is running the LLM turn; the gate stays
    /// armed and keeps running the wake detector on every frame (EPIC
    /// 7.1), but does not collect an utterance or run VAD/STT.
    Thinking,
    /// TTS is playing back the reply; the gate stays armed and keeps
    /// running the wake detector on every frame (EPIC 7.1) so a
    /// same-breath wake word can barge in (EPIC 7.2). No VAD/STT here.
    Speaking,
}

/// What happened as a result of feeding one chunk to the gate.
#[derive(Debug)]
pub enum GateOutput {
    /// Nothing notable this chunk.
    None,
    /// A wake word just fired; the gate transitioned IDLE -> LISTENING.
    Wake,
    /// Wake fired but no speech followed within the no-speech timeout;
    /// the gate discarded the utterance and returned to IDLE.
    NoSpeechTimeout,
    /// An utterance was collected end-to-end; the gate emitted it as one
    /// segment (pre-roll audio at its head) and returned to IDLE.
    Segment(AudioChunk),
    /// Speech was heard but the whole utterance (speech onset to
    /// silence-triggered end) was shorter than `min_utterance_ms` — too
    /// short to be real speech. Discarded; the gate returned to IDLE.
    TooShort,
    /// A wake word fired while the gate was THINKING or SPEAKING (EPIC
    /// 7.1). This story only surfaces the detection for 7.2 to consume
    /// later; it does not itself collect an utterance or change state.
    WakeDetected,
}

/// The gate state machine: wake detection plus VAD-driven utterance
/// collection.
pub struct Gate {
    state: GateState,
    wake: WakeEngine,
    vad: VadEndpointer,
    /// `silence_ms`/`min_utterance_ms`/`max_utterance_ms`, as `u64` for
    /// comparison against the millisecond counters below.
    silence_end_ms: u64,
    min_utterance_ms: u64,
    max_utterance_ms: u64,
    /// How long LISTENING waits for speech to begin (after wake fires)
    /// before giving up and returning to IDLE (`[vad].no_speech_timeout_ms`,
    /// EPIC 8.3's "no-speech" edge — a tuning knob, not hardcoded).
    no_speech_timeout_ms: u64,
    /// Chunks collected for the utterance in progress, at capture's
    /// native sample_rate/channels (not resampled) so the emitted segment
    /// is full quality for STT.
    utterance: Vec<AudioChunk>,
    /// 16kHz mono samples accumulated but not yet long enough for a full
    /// VAD frame.
    vad_pending: Vec<f32>,
    /// Whether any speech has been detected yet in the utterance in
    /// progress — distinguishes "never spoke, timed out" from "spoke,
    /// then went silent."
    speech_seen: bool,
    /// Milliseconds elapsed in LISTENING with no speech yet.
    no_speech_elapsed_ms: u64,
    /// Total milliseconds elapsed since LISTENING began (wake fire),
    /// including any pre-roll-seeded head — used for `max_utterance_ms`.
    total_listening_ms: u64,
    /// `total_listening_ms` at the moment speech first began this
    /// utterance — used to compute the speech-onset-to-end span checked
    /// against `min_utterance_ms`.
    speech_onset_ms: Option<u64>,
}

impl Gate {
    /// Builds a gate from a wake engine, a VAD endpointer, and the
    /// `[vad]` endpointing thresholds (EPIC 2.5).
    pub fn new(wake: WakeEngine, vad: VadEndpointer, vad_config: &VadConfig) -> Self {
        Self {
            state: GateState::Idle,
            wake,
            vad,
            silence_end_ms: vad_config.silence_ms as u64,
            min_utterance_ms: vad_config.min_utterance_ms as u64,
            max_utterance_ms: vad_config.max_utterance_ms as u64,
            no_speech_timeout_ms: vad_config.no_speech_timeout_ms,
            utterance: Vec::new(),
            vad_pending: Vec::new(),
            speech_seen: false,
            no_speech_elapsed_ms: 0,
            total_listening_ms: 0,
            speech_onset_ms: None,
        }
    }

    /// Current state.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Called by the orchestrator (EPIC 8) when the LLM turn starts, so
    /// the gate keeps running the wake detector instead of sitting idle
    /// (EPIC 7.1). Does not touch the capture/ring-buffer path — those
    /// keep feeding `process_chunk` exactly as they do in IDLE/LISTENING.
    pub fn enter_thinking(&mut self) {
        self.state = GateState::Thinking;
    }

    /// Called by the orchestrator (EPIC 8) when TTS playback starts, so
    /// the gate keeps running the wake detector during playback (EPIC
    /// 7.1) as the precondition for barge-in (EPIC 7.2).
    pub fn enter_speaking(&mut self) {
        self.state = GateState::Speaking;
    }

    /// Called by the orchestrator (EPIC 8) when a turn completes
    /// (THINKING/SPEAKING done, no barge-in fired) to return the gate to
    /// its normal wake-from-IDLE behavior.
    pub fn enter_idle(&mut self) {
        self.state = GateState::Idle;
    }

    /// Feeds one capture-rate chunk through the gate. `preroll` is the
    /// capture's current pre-roll snapshot (SPEC.md §2.6) — only consulted
    /// when a wake event fires this call, to seed the utterance buffer.
    pub fn process_chunk(&mut self, chunk: &AudioChunk, preroll: &AudioChunk) -> GateOutput {
        match self.state {
            GateState::Idle => self.process_idle(chunk, preroll),
            GateState::Listening => self.process_listening(chunk),
            GateState::Thinking | GateState::Speaking => self.process_armed_background(chunk),
        }
    }

    /// THINKING/SPEAKING background pass (EPIC 7.1): keeps the wake
    /// detector running on every frame so the gate never goes deaf during
    /// a turn, but does not collect an utterance or run VAD/STT — those
    /// stay LISTENING-only. A detection here is surfaced to the caller
    /// (EPIC 7.2 will act on it); this story does not.
    fn process_armed_background(&mut self, chunk: &AudioChunk) -> GateOutput {
        if self.wake.process_chunk(chunk).is_some() {
            GateOutput::WakeDetected
        } else {
            GateOutput::None
        }
    }

    fn process_idle(&mut self, chunk: &AudioChunk, preroll: &AudioChunk) -> GateOutput {
        if self.wake.process_chunk(chunk).is_none() {
            return GateOutput::None;
        }

        self.seed_utterance(chunk, preroll);
        GateOutput::Wake
    }

    /// Resets utterance-collection state and transitions straight to
    /// LISTENING, seeding the utterance from `preroll` (§2.6) then
    /// `chunk` — shared by a fresh IDLE wake (`process_idle`) and a
    /// barge-in commit (`begin_listening`, EPIC 7.2), which skip IDLE
    /// because the wake word already fired.
    fn seed_utterance(&mut self, chunk: &AudioChunk, preroll: &AudioChunk) {
        self.vad.reset();
        self.vad_pending.clear();
        self.speech_seen = false;
        self.no_speech_elapsed_ms = 0;
        self.total_listening_ms = 0;
        self.speech_onset_ms = None;
        self.utterance.clear();
        if !preroll.pcm.is_empty() {
            self.utterance.push(preroll.clone());
        }
        self.utterance.push(chunk.clone());
        self.state = GateState::Listening;
    }

    /// Commits a barge-in (EPIC 7.2): called once the caller has
    /// confirmed `GateOutput::WakeDetected` fired while THINKING/SPEAKING.
    /// Seeds the follow-up utterance from `chunk`/`preroll` exactly as a
    /// fresh IDLE wake would, and transitions straight to LISTENING.
    pub fn begin_listening(&mut self, chunk: &AudioChunk, preroll: &AudioChunk) {
        self.seed_utterance(chunk, preroll);
    }

    fn process_listening(&mut self, chunk: &AudioChunk) -> GateOutput {
        self.utterance.push(chunk.clone());

        let chunk_ms = if chunk.sample_rate == 0 || chunk.channels == 0 {
            0
        } else {
            (chunk.pcm.len() as u64 * 1000)
                / (chunk.sample_rate as u64 * chunk.channels as u64)
        };

        self.vad_pending.extend(resample::resample(
            &chunk.pcm,
            chunk.sample_rate,
            chunk.channels,
            crate::vad::model::SAMPLE_RATE as u32,
            1,
        ));

        let mut any_speech_this_chunk = false;
        while self.vad_pending.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = self.vad_pending.drain(..FRAME_SAMPLES).collect();
            if self.vad.process_frame(&frame).is_ok() && self.vad.is_speaking() {
                any_speech_this_chunk = true;
            }
        }

        let speech_just_began = any_speech_this_chunk && !self.speech_seen;
        if any_speech_this_chunk {
            self.speech_seen = true;
        }
        if speech_just_began {
            self.speech_onset_ms = Some(self.total_listening_ms);
        }

        self.total_listening_ms += chunk_ms;

        if !self.speech_seen {
            self.no_speech_elapsed_ms += chunk_ms;
            if self.no_speech_elapsed_ms >= self.no_speech_timeout_ms {
                self.state = GateState::Idle;
                self.utterance.clear();
                return GateOutput::NoSpeechTimeout;
            }
            return GateOutput::None;
        }

        // Hard cap: force-emit rather than let a stuck/noisy segment run
        // forever, regardless of the trailing-silence state.
        if self.total_listening_ms >= self.max_utterance_ms {
            return self.end_utterance();
        }

        if self.vad.silence_ms() >= self.silence_end_ms {
            return self.end_utterance();
        }

        GateOutput::None
    }

    /// Ends the in-progress utterance: emits the collected segment unless
    /// its speech-onset-to-end span is under `min_utterance_ms`, in which
    /// case it's discarded as too short to be real speech. Either way,
    /// returns the gate to IDLE.
    fn end_utterance(&mut self) -> GateOutput {
        let span_ms = self
            .speech_onset_ms
            .map(|onset| self.total_listening_ms.saturating_sub(onset))
            .unwrap_or(0);

        self.state = GateState::Idle;
        if span_ms < self.min_utterance_ms {
            self.utterance.clear();
            return GateOutput::TooShort;
        }

        let segment = merge_chunks(&self.utterance);
        self.utterance.clear();
        GateOutput::Segment(segment)
    }
}

/// Concatenates same-format chunks into one, preserving the first
/// chunk's `seq`/`sample_rate`/`channels`.
fn merge_chunks(chunks: &[AudioChunk]) -> AudioChunk {
    let (sample_rate, channels) = chunks
        .first()
        .map(|c| (c.sample_rate, c.channels))
        .unwrap_or((0, 0));
    let pcm = chunks.iter().flat_map(|c| c.pcm.iter().copied()).collect();
    AudioChunk {
        seq: chunks.first().map(|c| c.seq).unwrap_or(0),
        pcm,
        sample_rate,
        channels,
    }
}
