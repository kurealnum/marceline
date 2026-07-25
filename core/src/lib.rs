//! Marceline daemon core: orchestrator, audio, IPC, tools, memory.

pub mod audio;
pub mod config;
pub mod device;
pub mod engine;
pub mod gate;
pub mod ipc;
pub mod logging;
pub mod stt;
pub mod supervisor;
pub mod vad;
pub mod wake;

pub use audio::{
    AudioChunk, Capture, CaptureError, LevelMeter, Playback, PlaybackError, WavTap, WavTapError,
};
pub use config::{Config, ConfigError};
pub use device::Device;
pub use engine::{AudioStream, EngineError};
pub use gate::{Gate, GateOutput, GateState};
pub use stt::{GrpcSttEngine, SttEngine, SttInfo, Transcript, TranscriptStream};
pub use supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};
pub use vad::{SileroVad, VadEndpointer, VadError, DEFAULT_SPEECH_THRESHOLD, FRAME_SAMPLES};
pub use wake::{EnergyWakeDetector, WakeDetector, WakeEngine, WakeEvent};
