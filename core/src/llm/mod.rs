//! The LLM plugin contract (SPEC.md §2.4, §2.4.1, §2.5.1, EPIC 4.1).
//!
//! [`LlmEngine`] is the seam that makes the model backend hot-swappable
//! from config: swapping LM Studio for a hosted provider is a
//! `[llm].base_url` / `model` / `api_key_env` change, not a rewrite — which
//! only holds if every provider difference lives behind this trait.
//!
//! [`ChatEvent`] is a tagged event enum rather than a token string on
//! purpose: tool calls must be first-class events, or the THINKING loop
//! (EPIC 6) cannot be built on top of this without a rewrite. Every stream
//! item is a `Result` (invariant 1, §2.4.1), so a mid-stream HTTP 500 or a
//! malformed chunk propagates in-band instead of silently truncating a
//! stream that looks like it ended normally.

pub mod openai;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::engine::EngineError;

pub use openai::OpenAiCompatibleEngine;

/// One entry in a chat conversation sent to the backend.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// Who produced this message.
    pub role: Role,
    /// Message text. Empty for an assistant message that only carries tool
    /// calls.
    pub content: String,
}

/// Role of a [`Message`] in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The compiled system prompt (SOUL.md + memory, EPIC 4.2).
    System,
    /// Text from the user, e.g. an STT [`crate::Transcript::Final`].
    User,
    /// A prior model response, replayed as conversation history.
    Assistant,
    /// The result of a tool call, replayed as conversation history.
    Tool,
}

/// A tool the model may call, described in the backend's expected shape.
///
/// Kept intentionally opaque (JSON in, JSON out): the tool broker (EPIC 6)
/// owns tool schemas, this client only has to pass them through to an
/// OpenAI-compatible `tools` array unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    /// Tool name as the model will reference it in a call.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: serde_json::Value,
}

/// A request to start (or continue) a chat turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// Full conversation so far, oldest first, system prompt included.
    pub messages: Vec<Message>,
    /// Tools the model may call this turn. Empty when no tools apply.
    pub tools: Vec<ToolSpec>,
    /// Upper bound on tokens generated this turn (`[llm].max_tokens_per_turn`,
    /// §4.5) — the guardrail is enforced by the caller populating this from
    /// config, not by this client.
    pub max_tokens: u32,
}

/// Why a chat stream ended, as reported by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model produced a complete response.
    Stop,
    /// `max_tokens` was hit before the model finished.
    Length,
    /// The model stopped to make one or more tool calls.
    ToolCalls,
    /// The backend reported a reason this client does not recognize.
    Other,
}

/// One item of a streamed chat response (SPEC.md §2.4.1).
///
/// Text and tool calls are first-class, distinct variants: sentence-chunking
/// for TTS (§5.3) consumes [`ChatEvent::TextDelta`] only, and the tool
/// broker (EPIC 6) drives off `ToolCall*` events. Collapsing this to a
/// single string would make neither consumer buildable.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    /// The next chunk of assistant text.
    TextDelta(String),
    /// The next chunk of one tool call's arguments.
    ///
    /// `name` is `Some` only on the first delta for a given `id` — OpenAI-
    /// compatible backends send the tool name once per call, then stream
    /// `args_delta` as raw JSON fragments to be concatenated per `id`.
    ToolCallDelta {
        /// Identifies which in-flight tool call this delta belongs to.
        id: String,
        /// The tool's name, present on the first delta only.
        name: Option<String>,
        /// The next fragment of this call's JSON arguments.
        args_delta: String,
    },
    /// The named tool call's arguments are complete.
    ToolCallDone {
        /// The tool call that finished accumulating arguments.
        id: String,
    },
    /// The stream is finished.
    Done {
        /// Why the backend stopped generating.
        finish_reason: FinishReason,
    },
}

/// A streamed chat response, errors propagating in-band (invariant 1).
pub type ChatEventStream = Pin<Box<dyn Stream<Item = Result<ChatEvent, EngineError>> + Send>>;

/// Capabilities of a configured LLM backend (SPEC.md §2.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmInfo {
    /// Backend-qualified model name, e.g. `"openai-compatible:gpt-4o"`.
    pub name: String,
    /// Context window size in tokens, as configured/reported.
    pub context_window: u32,
    /// Whether this backend accepts a `tools` array.
    pub supports_tools: bool,
    /// Whether this backend streams (`stream: true`). Always `true` for
    /// [`OpenAiCompatibleEngine`]; the field exists so the trait does not
    /// assume every future backend can.
    pub streaming: bool,
}

/// An LLM backend (SPEC.md §2.4).
///
/// One implementation ships in v1: [`OpenAiCompatibleEngine`], a plain HTTP
/// client against any `/v1/chat/completions`-compatible endpoint. The trait
/// exists so the orchestrator never has to learn which provider is behind
/// it — that is the whole "hot-swappable via a config line" promise.
#[async_trait]
pub trait LlmEngine: Send + Sync {
    /// Sends `req` and returns the backend's streamed response.
    ///
    /// Returns a stream rather than a `Result<Stream>` on purpose: a
    /// failure to even open the request is delivered as the stream's first
    /// `Err` item, so callers have exactly one error path instead of two
    /// (invariant 1, §2.4.1).
    async fn chat(&self, req: ChatRequest) -> ChatEventStream;

    /// Reports what this backend can do. Synchronous and cheap.
    fn info(&self) -> LlmInfo;
}
