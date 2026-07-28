//! Marceline daemon core: orchestrator, audio, IPC, tools, memory.

pub mod audio;
pub mod config;
pub mod config_edit;
pub mod device;
pub mod engine;
pub mod gate;
pub mod ipc;
pub mod llm;
pub mod logging;
pub mod stt;
pub mod supervisor;
pub mod transcribe;
pub mod vad;
pub mod wake;

pub use audio::{
    read_wav, AudioChunk, Capture, CaptureError, LevelMeter, Playback, PlaybackError, WavReadError,
    WavTap, WavTapError,
};
pub use config::{Config, ConfigError};
pub use config_edit::ConfigEditError;
pub use device::Device;
pub use engine::{AudioStream, EngineError};
pub use gate::{Gate, GateOutput, GateState};
pub use llm::{
    compile_system_prompt, ChatEvent, ChatEventStream, ChatRequest, FinishReason, LlmEngine,
    LlmInfo, MemoryEntry, Message, OpenAiCompatibleEngine, Role, ToolSpec, Trust,
};
pub use stt::{
    GrpcSttEngine, SttEngine, SttInfo, SttManager, SttWorkerPaths, SwapError, Transcript,
    TranscriptStream,
};
pub use supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};
pub use transcribe::{transcribe_segment, Transcription};
pub use vad::{SileroVad, VadEndpointer, VadError, DEFAULT_SPEECH_THRESHOLD, FRAME_SAMPLES};
pub use wake::{EnergyWakeDetector, WakeDetector, WakeEngine, WakeEvent};
