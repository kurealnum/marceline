//! Marceline daemon core: orchestrator, audio, IPC, tools, memory.

pub mod audio;
pub mod config;
pub mod config_edit;
pub mod context;
pub mod device;
pub mod embedding;
pub mod engine;
pub mod gate;
pub mod history;
pub mod ipc;
pub mod llm;
pub mod logging;
pub mod mcp;
pub mod memory;
pub mod orchestrator;
pub mod soul;
pub mod soul_watch;
pub mod stt;
pub mod summarizer;
pub mod supervisor;
pub mod thinking;
pub mod tools;
pub mod transcribe;
pub mod tts;
pub mod vad;
pub mod wake;
pub mod worker_paths;

pub use audio::{
    read_wav, AudioChunk, Capture, CaptureError, LevelMeter, Playback, PlaybackError, WavReadError,
    WavTap, WavTapError,
};
pub use config::{Config, ConfigError};
pub use config_edit::ConfigEditError;
pub use context::recent_context;
pub use device::Device;
pub use embedding::{EmbedError, EmbeddingPipeline, MiniLmEmbedder, MINILM_DIM};
pub use engine::{AudioStream, EngineError};
pub use gate::{Gate, GateOutput, GateState};
pub use history::{HistoryError, HistoryStore, MemoryRecord, NewMemory, NewTurn, TurnRecord};
pub use llm::{
    compile_system_prompt, ChatEvent, ChatEventStream, ChatRequest, DropOldestTurn, FinishReason,
    LlmEngine, LlmInfo, MemoryEntry, Message, OpenAiCompatibleEngine, Role, SessionGuard,
    ToolCallRequest, ToolSpec, TrimPolicy, Trust, TurnBuffer,
};
pub use mcp::{register_mcp_tools, McpCallOutcome, McpClient, McpError, McpTool, McpToolInfo};
pub use memory::{
    compile_prompt_with_retrieval, ensure_current_embed_model, reembed_all, retrieve_similar,
    store_memory, MemoryError,
};
pub use orchestrator::{
    ConversationEvent, ConversationState, FailedStage, IllegalTransition, Orchestrator, Stages,
};
pub use soul::{Persona, SoulError, ToolDecision, ToolPolicy, VoicePreference};
pub use stt::{
    GrpcSttEngine, SttEngine, SttInfo, SttManager, SttWorkerPaths, SwapError, Transcript,
    TranscriptStream,
};
pub use summarizer::{
    derive_provenance, summarize_session, LlmSummarizer, SummarizeError, Summarizer,
    SummarizerError,
};
pub use supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};
pub use thinking::{
    resolve_max_iterations, think, Confirm, DeclineAll, ThinkingOutcome, MAX_TOOL_ITERS_ENV,
};
pub use tools::{
    DuplicateToolError, GetTimeTool, ListDirTool, ReadFileTool, SafetyClass, Tool, ToolBroker,
    ToolResult, WebSearchTool,
};
pub use transcribe::{transcribe_segment, Transcription};
pub use tts::{
    launch as launch_tts_worker, play, resolve_voice, sentence_chunk, GrpcTtsEngine, PlaybackSink,
    TextStream, TtsEngine, TtsInfo, TtsWorkerPaths, VoiceId,
};
pub use vad::{SileroVad, VadEndpointer, VadError, DEFAULT_SPEECH_THRESHOLD, FRAME_SAMPLES};
pub use wake::{EnergyWakeDetector, WakeDetector, WakeEngine, WakeEvent};
pub use worker_paths::{workers_root, WORKERS_DIR_ENV_VAR};
