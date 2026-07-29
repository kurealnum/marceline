//! OpenAI-compatible HTTP client backend (SPEC.md §2.3, §2.4, EPIC 4.1).
//!
//! Plain HTTP against `/v1/chat/completions` with `stream: true` — no
//! worker, no IPC, just a client. Every provider difference (LM Studio,
//! Ollama, OpenAI, an Anthropic proxy) lives in [`crate::config::LlmConfig`]
//! (`base_url`, `model`, `api_key_env`); this file must never branch on
//! which provider it is talking to.
//!
//! Cancellation (§2.5.1) is simpler here than for STT: there is no
//! cooperative "please stop" message in the OpenAI streaming contract, so
//! firing the run's `CancellationToken` just drops the in-flight HTTP
//! stream. Most providers stop billing further tokens the moment the
//! connection closes.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::{
    ChatEvent, ChatEventStream, ChatRequest, FinishReason, LlmEngine, LlmInfo, Message, Role,
    ToolSpec,
};
use crate::config::LlmConfig;
use crate::engine::EngineError;

/// Backend name used in [`EngineError`] messages and logs.
const BACKEND: &str = "llm";

/// An LLM backend talking to any OpenAI-compatible `/v1/chat/completions`
/// endpoint.
#[derive(Debug)]
pub struct OpenAiCompatibleEngine {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
    info: LlmInfo,
    cancel: CancellationToken,
}

impl OpenAiCompatibleEngine {
    /// Builds a client from `[llm]` config.
    ///
    /// `cancel` is the run's cancellation token (§2.5.1), cloned into every
    /// stage; firing it drops the in-flight HTTP stream for any turn this
    /// client currently has open.
    ///
    /// Fails only if the API key env var named by `config.api_key_env` is
    /// unset — building the HTTP client itself cannot fail with the
    /// `rustls-tls` backend this crate uses.
    pub fn new(config: &LlmConfig, cancel: CancellationToken) -> Result<Self, EngineError> {
        let api_key = config.resolve_api_key().map_err(|err| EngineError::Worker {
            backend: BACKEND,
            message: err.to_string(),
        })?;

        let client = reqwest::Client::builder()
            .build()
            .map_err(|err| EngineError::Transport {
                backend: BACKEND,
                source: Box::new(err),
            })?;

        Ok(Self {
            client,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            model: config.model.clone(),
            api_key,
            info: LlmInfo {
                name: format!("{}:{}", config.backend, config.model),
                context_window: 0,
                supports_tools: true,
                streaming: true,
            },
            cancel,
        })
    }
}

#[async_trait]
impl LlmEngine for OpenAiCompatibleEngine {
    async fn chat(&self, req: ChatRequest) -> ChatEventStream {
        let body = WireRequest::from_request(&self.model, &req);
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await;

        let response = match response {
            Ok(response) => response,
            Err(err) => return single_error(transport_error(err)),
        };

        if !response.status().is_success() {
            let status = response.status();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "<no response body>".to_string());
            return single_error(EngineError::Worker {
                backend: BACKEND,
                message: format!("{status}: {message}"),
            });
        }

        Box::pin(event_stream(response, self.cancel.clone()))
    }

    fn info(&self) -> LlmInfo {
        self.info.clone()
    }
}

/// Maps a cancellable SSE byte stream into [`ChatEvent`] items.
///
/// Errors and cancellation both end the stream after exactly one item —
/// invariant 1 (§2.4.1) — and any tool calls still open when `Done` arrives
/// get a synthetic [`ChatEvent::ToolCallDone`] first, so a caller never sees
/// `Done` while a tool call looks unfinished.
fn event_stream(
    response: reqwest::Response,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<ChatEvent, EngineError>> + Send {
    let state = SseState {
        bytes: Box::pin(response.bytes_stream()),
        buf: String::new(),
        pending: Vec::new(),
        current_tool_call: None,
        done: false,
    };

    futures::stream::unfold(Some((state, cancel)), |state| async move {
        let (mut state, cancel) = state?;

        loop {
            if !state.pending.is_empty() {
                let event = state.pending.remove(0);
                state.done = state.done || matches!(event, ChatEvent::Done { .. });
                let next = if state.done { None } else { Some((state, cancel)) };
                return Some((Ok(event), next));
            }

            if let Some(line) = state.take_buffered_line() {
                if let Some(Err(err)) =
                    parse_sse_line(&line, &mut state.current_tool_call, &mut state.pending)
                {
                    return Some((Err(err), None));
                }
                continue;
            }

            let chunk = tokio::select! {
                biased;

                _ = cancel.cancelled() => {
                    return Some((Err(EngineError::Cancelled { backend: BACKEND }), None));
                }

                chunk = state.bytes.next() => chunk,
            };

            match chunk {
                Some(Ok(bytes)) => {
                    state.buf.push_str(&String::from_utf8_lossy(&bytes));
                }
                Some(Err(err)) => return Some((Err(transport_error(err)), None)),
                None => {
                    // The connection closed without a terminal `[DONE]` or
                    // `finish_reason` — the contract was violated, not
                    // merely ended.
                    return Some((
                        Err(EngineError::Protocol {
                            backend: BACKEND,
                            message: "stream ended before a finish reason was sent".to_string(),
                        }),
                        None,
                    ));
                }
            }
        }
    })
}

/// Per-stream state carried across `unfold` steps.
struct SseState {
    bytes: Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buf: String,
    /// Events parsed but not yet yielded — a single SSE line can imply more
    /// than one [`ChatEvent`] (e.g. closing the previous tool call before
    /// opening the next, or closing the last tool call before `Done`).
    pending: Vec<ChatEvent>,
    /// `(index, id)` of the tool call currently accumulating arguments, per
    /// the wire format's `tool_calls[].index` — OpenAI-compatible backends
    /// stream one tool call to completion before starting the next.
    current_tool_call: Option<(u32, String)>,
    done: bool,
}

impl SseState {
    /// Pops one complete `\n`-terminated line out of the buffer, if any.
    fn take_buffered_line(&mut self) -> Option<String> {
        let idx = self.buf.find('\n')?;
        let line = self.buf[..idx].trim_end_matches('\r').to_string();
        self.buf.drain(..=idx);
        Some(line)
    }
}

/// Parses one raw SSE line, pushing zero or more events onto `pending`.
///
/// `current_tool_call` tracks the in-flight call's wire `index` and id: the
/// OpenAI wire format has no explicit "this tool call is done" message, so
/// a new index (or the stream's `finish_reason`/`Done`) is what implies the
/// previous call finished, and that implied [`ChatEvent::ToolCallDone`] is
/// pushed before whatever triggered it.
fn parse_sse_line(
    line: &str,
    current_tool_call: &mut Option<(u32, String)>,
    pending: &mut Vec<ChatEvent>,
) -> Option<Result<(), EngineError>> {
    let line = line.trim();
    if line.is_empty() || !line.starts_with("data:") {
        return None;
    }
    let payload = line["data:".len()..].trim();
    if payload == "[DONE]" {
        close_current_tool_call(current_tool_call, pending);
        pending.push(ChatEvent::Done {
            finish_reason: FinishReason::Stop,
        });
        return Some(Ok(()));
    }

    let chunk: WireChunk = match serde_json::from_str(payload) {
        Ok(chunk) => chunk,
        Err(err) => {
            return Some(Err(EngineError::Protocol {
                backend: BACKEND,
                message: format!("malformed stream chunk: {err}"),
            }))
        }
    };

    let Some(choice) = chunk.choices.into_iter().next() else {
        return Some(Ok(()));
    };

    if let Some(finish_reason) = choice.finish_reason {
        close_current_tool_call(current_tool_call, pending);
        pending.push(ChatEvent::Done {
            finish_reason: finish_reason_from_wire(&finish_reason),
        });
        return Some(Ok(()));
    }

    if let Some(content) = choice.delta.content {
        if !content.is_empty() {
            pending.push(ChatEvent::TextDelta(content));
        }
    }

    if let Some(call) = choice.delta.tool_calls.and_then(|calls| calls.into_iter().next()) {
        let index = call.index;
        let id = call.id.unwrap_or_default();

        let is_new_call = match current_tool_call {
            Some((current_index, _)) if *current_index == index => false,
            _ => {
                close_current_tool_call(current_tool_call, pending);
                true
            }
        };

        let id = if is_new_call {
            *current_tool_call = Some((index, id.clone()));
            id
        } else {
            current_tool_call.as_ref().map(|(_, id)| id.clone()).unwrap_or(id)
        };

        let name = if is_new_call {
            call.function.as_ref().and_then(|f| f.name.clone())
        } else {
            None
        };
        let args_delta = call.function.and_then(|f| f.arguments).unwrap_or_default();
        pending.push(ChatEvent::ToolCallDelta {
            id,
            name,
            args_delta,
        });
    }

    Some(Ok(()))
}

/// Pushes [`ChatEvent::ToolCallDone`] for the in-flight tool call, if any,
/// and clears it.
fn close_current_tool_call(current_tool_call: &mut Option<(u32, String)>, pending: &mut Vec<ChatEvent>) {
    if let Some((_, id)) = current_tool_call.take() {
        pending.push(ChatEvent::ToolCallDone { id });
    }
}

/// Maps a backend-reported `finish_reason` string to [`FinishReason`].
///
/// Strict OpenAI only sends `stop` / `length` / `tool_calls` /
/// `function_call`, but EPIC 4.4 (verifying LM Studio and a hosted provider
/// by config swap only) found "OpenAI-compatible" backends are not all
/// literal about it: `max_tokens` and `end_turn` are an Anthropic-style
/// proxy's spellings of the same two outcomes. Recognizing them here, in
/// the one place that maps wire strings to [`FinishReason`], is what keeps
/// that variance from leaking into call sites as a provider-specific branch.
fn finish_reason_from_wire(raw: &str) -> FinishReason {
    match raw {
        "stop" | "end_turn" | "eos" => FinishReason::Stop,
        "length" | "max_tokens" => FinishReason::Length,
        "tool_calls" | "function_call" | "tool_use" => FinishReason::ToolCalls,
        _ => FinishReason::Other,
    }
}

/// Wraps a transport-level `reqwest` failure as an [`EngineError`].
fn transport_error(err: reqwest::Error) -> EngineError {
    EngineError::Transport {
        backend: BACKEND,
        source: Box::new(err),
    }
}

/// A response stream carrying exactly one error, for failures that happen
/// before the backend's stream exists.
fn single_error(err: EngineError) -> ChatEventStream {
    Box::pin(futures::stream::once(async move { Err(err) }))
}

/// The `/v1/chat/completions` request body.
#[derive(Debug, Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    stream: bool,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
}

impl<'a> WireRequest<'a> {
    fn from_request(model: &'a str, req: &'a ChatRequest) -> Self {
        Self {
            model,
            messages: req.messages.iter().map(WireMessage::from).collect(),
            stream: true,
            max_tokens: req.max_tokens,
            tools: req.tools.iter().map(WireTool::from).collect(),
        }
    }
}

#[derive(Debug, Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
    /// Only an assistant message that requested tools carries these
    /// (EPIC 6.3); omitted entirely for every other message so a strict
    /// backend does not see a stray empty array on a plain turn.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireOutToolCall<'a>>,
    /// Only a `Role::Tool` message carries this, linking it back to the
    /// assistant `tool_calls` entry it answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

impl<'a> From<&'a Message> for WireMessage<'a> {
    fn from(message: &'a Message) -> Self {
        Self {
            role: match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            },
            content: &message.content,
            tool_calls: message.tool_calls.iter().map(WireOutToolCall::from).collect(),
            tool_call_id: message.tool_call_id.as_deref(),
        }
    }
}

/// One tool call on the wire, as replayed on an assistant message that
/// requested tools (EPIC 6.3) — the outbound mirror of [`WireToolCall`],
/// which parses the *inbound* streamed shape.
#[derive(Debug, Serialize)]
struct WireOutToolCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireOutToolCallFunction<'a>,
}

#[derive(Debug, Serialize)]
struct WireOutToolCallFunction<'a> {
    name: &'a str,
    arguments: &'a str,
}

impl<'a> From<&'a super::ToolCallRequest> for WireOutToolCall<'a> {
    fn from(call: &'a super::ToolCallRequest) -> Self {
        Self {
            id: &call.id,
            kind: "function",
            function: WireOutToolCallFunction {
                name: &call.name,
                arguments: &call.arguments,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct WireTool<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireToolFunction<'a>,
}

#[derive(Debug, Serialize)]
struct WireToolFunction<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
}

impl<'a> From<&'a ToolSpec> for WireTool<'a> {
    fn from(spec: &'a ToolSpec) -> Self {
        Self {
            kind: "function",
            function: WireToolFunction {
                name: &spec.name,
                description: &spec.description,
                parameters: &spec.parameters,
            },
        }
    }
}

/// One `data:` chunk of a streamed `/v1/chat/completions` response.
#[derive(Debug, Deserialize)]
struct WireChunk {
    #[serde(default)]
    choices: Vec<WireChoice>,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Debug, Deserialize)]
struct WireToolCall {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireToolCallFunction>,
}

#[derive(Debug, Default, Deserialize)]
struct WireToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
