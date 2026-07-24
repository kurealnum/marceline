//! Marceline daemon core: orchestrator, audio, IPC, tools, memory.

pub mod config;
pub mod logging;
pub mod supervisor;

pub use config::{Config, ConfigError};
pub use supervisor::{HealthView, Supervisor, WorkerSpec, WorkerState};
