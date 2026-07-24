//! The wake+VAD gate state machine (SPEC.md §2.1, §2.5, §2.6, EPIC 2.3):
//! `IDLE -> LISTENING -> (collect utterance) -> emit audio segment`. Turns
//! always-on mic frames into clean, bounded audio segments the
//! orchestrator (EPIC 8) and STT (EPIC 3) consume directly.
//!
//! Utterance capture is seeded from the capture pre-roll ring (§2.6): the
//! wake-firing chunk is prepended with whatever the pre-roll already
//! holds, so a same-breath command isn't lost to the ~300ms IDLE ->
//! LISTENING flip.
//!
//! Endpointing thresholds here are placeholder constants matching the
//! `[vad]` config defaults established in EPIC 0 — reading them from
//! config and tuning them on real speech is EPIC 2.5's job, not this
//! story's.

use crate::audio::resample;
use crate::vad::FRAME_SAMPLES;
use crate::wake::WakeEngine;
use crate::{AudioChunk, VadEndpointer};

/// Silence duration that ends an utterance once speech has been heard.
/// Matches the `[vad].silence_ms` default from EPIC 0's config.
const SILENCE_END_MS: u64 = 700;

/// How long LISTENING waits for speech to begin (after wake fires) before
/// giving up and returning to IDLE. Concrete timeout values are a tuning
/// knob (SPEC.md EPIC 8.3); this is a reasonable placeholder.
const NO_SPEECH_TIMEOUT_MS: u64 = 3_000;

/// Which state the gate is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Not listening; running the wake detector on every frame.
    Idle,
    /// Collecting an utterance following a wake event.
    Listening,
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
}

/// The gate state machine: wake detection plus VAD-driven utterance
/// collection.
pub struct Gate {
    state: GateState,
    wake: WakeEngine,
    vad: VadEndpointer,
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
}

impl Gate {
    /// Builds a gate from a wake engine and a VAD endpointer.
    pub fn new(wake: WakeEngine, vad: VadEndpointer) -> Self {
        Self {
            state: GateState::Idle,
            wake,
            vad,
            utterance: Vec::new(),
            vad_pending: Vec::new(),
            speech_seen: false,
            no_speech_elapsed_ms: 0,
        }
    }

    /// Current state.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Feeds one capture-rate chunk through the gate. `preroll` is the
    /// capture's current pre-roll snapshot (SPEC.md §2.6) — only consulted
    /// when a wake event fires this call, to seed the utterance buffer.
    pub fn process_chunk(&mut self, chunk: &AudioChunk, preroll: &AudioChunk) -> GateOutput {
        match self.state {
            GateState::Idle => self.process_idle(chunk, preroll),
            GateState::Listening => self.process_listening(chunk),
        }
    }

    fn process_idle(&mut self, chunk: &AudioChunk, preroll: &AudioChunk) -> GateOutput {
        if self.wake.process_chunk(chunk).is_none() {
            return GateOutput::None;
        }

        self.vad.reset();
        self.vad_pending.clear();
        self.speech_seen = false;
        self.no_speech_elapsed_ms = 0;
        self.utterance.clear();
        if !preroll.pcm.is_empty() {
            self.utterance.push(preroll.clone());
        }
        self.utterance.push(chunk.clone());
        self.state = GateState::Listening;
        GateOutput::Wake
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

        if any_speech_this_chunk {
            self.speech_seen = true;
        }

        if !self.speech_seen {
            self.no_speech_elapsed_ms += chunk_ms;
            if self.no_speech_elapsed_ms >= NO_SPEECH_TIMEOUT_MS {
                self.state = GateState::Idle;
                self.utterance.clear();
                return GateOutput::NoSpeechTimeout;
            }
            return GateOutput::None;
        }

        if self.vad.silence_ms() >= SILENCE_END_MS {
            let segment = merge_chunks(&self.utterance);
            self.utterance.clear();
            self.state = GateState::Idle;
            return GateOutput::Segment(segment);
        }

        GateOutput::None
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
