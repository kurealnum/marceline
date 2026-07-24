//! Marceline daemon core: orchestrator, audio, IPC, tools, memory.

pub mod audio;
pub mod config;
pub mod device;
pub mod logging;
pub mod supervisor;

pub use audio::{AudioChunk, Capture, CaptureError, Playback, PlaybackError};
pub use config::{Config, ConfigError};
pub use device::Device;
pub use supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};
