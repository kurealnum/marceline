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
pub mod mcp;
pub mod stt;
pub mod supervisor;
pub mod thinking;
pub mod tools;
pub mod transcribe;
pub mod tts;
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
    compile_system_prompt, ChatEvent, ChatEventStream, ChatRequest, DropOldestTurn, FinishReason,
    LlmEngine, LlmInfo, MemoryEntry, Message, OpenAiCompatibleEngine, Role, SessionGuard,
    ToolCallRequest, ToolSpec, TrimPolicy, Trust, TurnBuffer,
};
pub use mcp::{register_mcp_tools, McpCallOutcome, McpClient, McpError, McpTool, McpToolInfo};
pub use stt::{
    GrpcSttEngine, SttEngine, SttInfo, SttManager, SttWorkerPaths, SwapError, Transcript,
    TranscriptStream,
};
pub use supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};
pub use thinking::{resolve_max_iterations, think, ThinkingOutcome, MAX_TOOL_ITERS_ENV};
pub use tools::{
    DuplicateToolError, GetTimeTool, ListDirTool, ReadFileTool, SafetyClass, Tool, ToolBroker,
    ToolResult, WebSearchTool,
};
pub use transcribe::{transcribe_segment, Transcription};
pub use tts::{
    launch as launch_tts_worker, play, sentence_chunk, GrpcTtsEngine, PlaybackSink, TextStream,
    TtsEngine, TtsInfo, TtsWorkerPaths, VoiceId,
};
pub use vad::{SileroVad, VadEndpointer, VadError, DEFAULT_SPEECH_THRESHOLD, FRAME_SAMPLES};
pub use wake::{EnergyWakeDetector, WakeDetector, WakeEngine, WakeEvent};
