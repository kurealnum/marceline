//! The `marceline say-to-llm <text>` path (EPIC 4.1, 4.2).
//!
//! Demoable proof that swapping LLM providers is a config change: run this
//! against `[llm]` pointed at LM Studio, edit `base_url` / `model` /
//! `api_key_env`, rerun unchanged, and it streams against the new backend.
//! Also the first place a compiled system prompt (SOUL.md + memory, §3.2)
//! actually reaches an LLM request.

use std::io::Write;
use std::path::{Path, PathBuf};

use futures::StreamExt;
use marceline_core::{
    compile_system_prompt, ChatEvent, ChatRequest, Config, LlmEngine, OpenAiCompatibleEngine,
    TurnBuffer,
};
use tokio_util::sync::CancellationToken;

/// Anything that can go wrong sending one turn to the configured LLM.
#[derive(Debug, thiserror::Error)]
pub enum SayToLlmError {
    /// The config file could not be loaded.
    #[error(transparent)]
    Config(#[from] marceline_core::ConfigError),
    /// The backend could not be reached, failed, or violated the stream
    /// contract.
    #[error(transparent)]
    Engine(#[from] marceline_core::EngineError),
}

/// Reads `[llm]` from `config_path`, compiles the system prompt from
/// `soul_path`, and streams a reply to `text` to stdout as it arrives.
///
/// A missing SOUL.md degrades to an empty persona (§3.2's compiler consumes
/// whatever SOUL.md compiles to; a fresh install without one yet is not a
/// failure) rather than refusing to run.
pub async fn say_to_llm(
    config_path: &Path,
    soul_path: &Path,
    text: &str,
) -> Result<(), SayToLlmError> {
    let config = Config::load(config_path)?;
    let soul = std::fs::read_to_string(soul_path).unwrap_or_default();
    let system_prompt = compile_system_prompt(&soul, &[]);

    let cancel = CancellationToken::new();
    let engine = OpenAiCompatibleEngine::new(&config.llm, cancel)?;
    let info = engine.info();

    // One-shot CLI run: a single turn, but routed through `TurnBuffer` so
    // the context-window trimming this session would eventually need is
    // exercised the same way the daemon's multi-turn conversations are.
    let mut turns = TurnBuffer::new();
    turns.push_user(text);
    let messages = turns.messages_for_request(&system_prompt, info.context_window);

    let request = ChatRequest {
        messages,
        tools: vec![],
        max_tokens: config.llm.max_tokens_per_turn,
    };

    let mut stream = engine.chat(request).await;
    let mut stdout = std::io::stdout();
    let mut reply = String::new();

    while let Some(event) = stream.next().await {
        match event? {
            ChatEvent::TextDelta(delta) => {
                let _ = stdout.write_all(delta.as_bytes());
                let _ = stdout.flush();
                reply.push_str(&delta);
            }
            ChatEvent::ToolCallDelta { .. } | ChatEvent::ToolCallDone { .. } => {
                // No tool broker wired up to `say-to-llm` yet (EPIC 6); the
                // event still has to be drained rather than left unhandled,
                // since dropping the stream mid-tool-call would look like a
                // truncated response.
            }
            ChatEvent::Done { .. } => break,
        }
    }
    println!();
    turns.push_assistant(reply);

    Ok(())
}

/// Default SOUL.md path, relative to the working directory.
pub const DEFAULT_SOUL: &str = "SOUL.md";

/// Resolves `--soul <path>` from CLI args, or [`DEFAULT_SOUL`].
pub fn soul_path_from_args(args: &[String]) -> PathBuf {
    let index = args.iter().position(|arg| arg == "--soul");
    match index.and_then(|i| args.get(i + 1)) {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(DEFAULT_SOUL),
    }
}
