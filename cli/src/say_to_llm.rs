//! The `marceline say-to-llm <text>` path (EPIC 4.1, 4.2, 6.3).
//!
//! Demoable proof that swapping LLM providers is a config change: run this
//! against `[llm]` pointed at LM Studio, edit `base_url` / `model` /
//! `api_key_env`, rerun unchanged, and it streams against the new backend.
//! Also the first place a compiled system prompt (SOUL.md + memory, §3.2)
//! actually reaches an LLM request, and — since EPIC 6.3 — the first place
//! the THINKING tool-call loop runs for real: ask "what time is it" and the
//! model can actually get an answer via `get_time` (§7's demoable).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use marceline_core::soul::Persona;
use marceline_core::{
    compile_system_prompt, register_mcp_tools, resolve_max_iterations, think, Config, DeclineAll,
    GetTimeTool, LlmEngine, ListDirTool, OpenAiCompatibleEngine, ReadFileTool, SessionGuard,
    ToolBroker, TurnBuffer, WebSearchTool,
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
    let persona = Persona::load(soul_path).unwrap_or_default();
    let system_prompt = compile_system_prompt(&persona.render(), &[]);

    let cancel = CancellationToken::new();
    let engine = OpenAiCompatibleEngine::new(&config.llm, cancel.clone())?;
    // Cost/rate guardrail (§4.5): caps this run to the configured
    // per-turn token budget and per-session request count, refusing
    // rather than calling the backend once either is exhausted.
    let engine = SessionGuard::new(
        engine,
        config.llm.max_tokens_per_turn,
        config.llm.max_requests_per_session,
    );
    let info = engine.info();

    // v1 built-ins (EPIC 6.2), read-only per the epic's key constraint,
    // plus whatever MCP servers are configured (EPIC 6.4) — merged into
    // the same broker, namespaced, so the THINKING loop below never has
    // to know which is which.
    let mut broker = ToolBroker::new();
    broker.register(Arc::new(GetTimeTool)).expect("get_time is the first registration");
    broker.register(Arc::new(ReadFileTool)).expect("read_file is the first registration");
    broker.register(Arc::new(ListDirTool)).expect("list_dir is the first registration");
    broker
        .register(Arc::new(WebSearchTool::new()?))
        .expect("web_search is the first registration");
    for skipped in register_mcp_tools(&mut broker, &config.mcp).await {
        tracing::warn!(server = %skipped, "mcp server unavailable, continuing without it");
    }

    // One-shot CLI run: a single turn, but routed through `TurnBuffer` so
    // the context-window trimming this session would eventually need is
    // exercised the same way the daemon's multi-turn conversations are.
    let mut turns = TurnBuffer::new();
    turns.push_user(text);
    let messages = turns.messages_for_request(&system_prompt, info.context_window);

    let mut stdout = std::io::stdout();
    let max_iterations = resolve_max_iterations(config.llm.max_tool_iterations_per_turn);
    let policy = persona.tool_policy();

    let (outcome, _messages) = think(
        &engine,
        &broker,
        messages,
        broker.catalog(),
        &policy,
        config.llm.max_tokens_per_turn,
        max_iterations,
        cancel,
        // No real voice-confirmation path exists yet (EPIC 6.5 built the
        // seam, nothing wires it up); v1 also registers nothing above
        // ReadOnly (§10), so this is never actually consulted.
        &DeclineAll,
        |delta: &str| {
            let _ = stdout.write_all(delta.as_bytes());
            let _ = stdout.flush();
        },
    )
    .await?;
    println!();

    if outcome.iteration_cap_hit {
        tracing::warn!(max_iterations, "tool iteration cap hit; forced a final answer");
    }
    turns.push_assistant(outcome.text);

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
