//! The THINKING tool-call loop (SPEC.md §2.5, §4, §2.5.1, EPIC 6.3).
//!
//! This is what turns "the LLM can describe a tool" (EPIC 6.1) into "the
//! LLM can act": stream the model, and whenever it emits a tool call,
//! dispatch it through the [`ToolBroker`] (EPIC 6.1), feed the result back
//! as a linked tool message, and let the model continue — repeating until
//! it produces a final text answer or the configured iteration cap is hit.
//!
//! Bounded by `max_iterations` (`[llm].max_tool_iterations_per_turn`,
//! §3.1, overridable via `MARCELINE_MAX_TOOL_ITERS`) so a model that keeps
//! requesting tools cannot spin the run forever: on breach, the tools it
//! just asked for are told the budget is gone instead of being run, and
//! one more request with no tools offered forces a final answer.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::engine::EngineError;
use crate::llm::{ChatEvent, ChatEventStream, ChatRequest, FinishReason, LlmEngine, Message, ToolCallRequest, ToolSpec};
use crate::tools::{SafetyClass, ToolBroker, ToolResult};

/// Message content fed back to the model when a tool call was gated by
/// [`Confirm`] and the user declined (or nothing declined on its behalf,
/// via [`DeclineAll`]).
const DECLINED_NOTE: &str = "user declined to confirm this action; it was not run";

/// Speaks a confirmation prompt and captures the user's spoken yes/no —
/// the in-voice confirmation gate [`SafetyClass::SideEffecting`]/
/// [`SafetyClass::Dangerous`] tools require before they may run (SPEC.md
/// §2.5.1, §4, EPIC 6.5).
///
/// No real implementation exists yet: it needs an orchestrator that owns
/// both TTS (to speak the prompt) and STT (to hear the answer), and
/// nothing in this codebase assembles those together yet. `SOUL.md` tool
/// policy (§3.2, EPIC 9.3) is what will decide *which* classes actually
/// reach this gate — today every real tool is `ReadOnly` (§10), so
/// [`think`] never calls this in practice. The seam exists now so that
/// wiring a real voice confirmation later — or registering the first
/// side-effecting tool (EPIC 14.3) — is a new [`Confirm`] impl, not a
/// rewrite of this loop.
#[async_trait]
pub trait Confirm: Send + Sync {
    /// Speaks `prompt` and returns `true` only on an affirmative spoken
    /// reply.
    async fn confirm(&self, prompt: &str) -> bool;
}

/// A [`Confirm`] that declines every prompt.
///
/// The safe default while no real voice-confirmation path exists: a
/// side-effecting tool that somehow reached this gate must fail closed
/// (not run) rather than auto-approve just because nothing real was
/// wired in yet.
pub struct DeclineAll;

#[async_trait]
impl Confirm for DeclineAll {
    async fn confirm(&self, _prompt: &str) -> bool {
        false
    }
}

/// True when `class` requires [`Confirm`] before the tool may run.
///
/// A stand-in for SOUL.md tool policy (§3.2), which will make this
/// decision per-tool once EPIC 9.3 wires it up; until then every class
/// above `ReadOnly` always requires confirmation, and an unregistered
/// name (`None`) requires nothing since [`ToolBroker::dispatch`] will
/// reject it as unknown anyway.
fn requires_confirmation(class: Option<SafetyClass>) -> bool {
    matches!(class, Some(SafetyClass::SideEffecting) | Some(SafetyClass::Dangerous))
}

/// Environment variable overriding
/// [`crate::config::LlmConfig::max_tool_iterations_per_turn`] (§3.1).
pub const MAX_TOOL_ITERS_ENV: &str = "MARCELINE_MAX_TOOL_ITERS";

/// Message content fed back to the model in place of running a tool call
/// once the iteration cap has already been hit.
const BUDGET_EXHAUSTED_NOTE: &str = "tool budget exhausted for this turn; answer with what you have";

/// Resolves the effective iteration cap: [`MAX_TOOL_ITERS_ENV`] if set to a
/// valid non-negative integer, else `configured` (the config file's
/// `max_tool_iterations_per_turn`).
///
/// Read fresh on every call rather than cached, so a config-reload-free
/// env override (handy for tuning during development) takes effect on the
/// very next turn.
pub fn resolve_max_iterations(configured: u32) -> u32 {
    std::env::var(MAX_TOOL_ITERS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(configured)
}

/// The result of running THINKING to completion for one user turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkingOutcome {
    /// The model's final text answer — the only thing downstream
    /// (sentence-chunking/TTS, §5.3) needs from this turn once it is over.
    pub text: String,
    /// Why the *final* chat stream ended, as the backend reported it.
    pub finish_reason: FinishReason,
    /// True when the loop stopped only because the iteration cap was
    /// reached, not because the model reached a natural final answer.
    /// Lets a caller log or otherwise surface that the cap actually bit,
    /// rather than that being indistinguishable from a normal stop.
    pub iteration_cap_hit: bool,
}

/// One text delta or tool-call round of one chat stream, collected for the
/// loop to act on.
struct StreamOutcome {
    text: String,
    tool_calls: Vec<ToolCallRequest>,
    finish_reason: FinishReason,
}

/// Runs THINKING for one turn against `engine`, dispatching any tool calls
/// through `broker`.
///
/// `messages` is the full conversation so far (system prompt included, per
/// [`ChatRequest::messages`]) and is both consumed and grown in place with
/// each tool round, so a caller that wants the updated history back (to
/// feed into [`crate::llm::TurnBuffer`], say) gets it via the returned
/// value — this function does not own history persistence, only one
/// turn's worth of round-tripping.
///
/// `on_text` is called with every [`ChatEvent::TextDelta`] as it streams —
/// across every round, not just the final one — so a caller wired to
/// sentence-chunking/TTS (§5.3) sees text the moment the model produces
/// it rather than only once this returns. In v1, tool-requesting rounds
/// rarely carry user-facing text alongside the calls, but nothing here
/// assumes that.
///
/// `confirm` gates any [`SafetyClass::SideEffecting`]/`Dangerous` tool
/// call (EPIC 6.5) — pass [`DeclineAll`] where no real voice confirmation
/// exists yet, which is every call site today since v1 registers nothing
/// above `ReadOnly` (§10).
#[allow(clippy::too_many_arguments)]
pub async fn think<E>(
    engine: &E,
    broker: &ToolBroker,
    mut messages: Vec<Message>,
    tools: Vec<ToolSpec>,
    max_tokens: u32,
    max_iterations: u32,
    cancel: CancellationToken,
    confirm: &dyn Confirm,
    mut on_text: impl FnMut(&str),
) -> Result<(ThinkingOutcome, Vec<Message>), EngineError>
where
    E: LlmEngine + ?Sized,
{
    let mut iterations: u32 = 0;

    loop {
        let stream = engine
            .chat(ChatRequest {
                messages: messages.clone(),
                tools: tools.clone(),
                max_tokens,
            })
            .await;
        let round = collect_round(stream, &mut on_text).await?;

        if round.tool_calls.is_empty() {
            return Ok((
                ThinkingOutcome {
                    text: round.text,
                    finish_reason: round.finish_reason,
                    iteration_cap_hit: false,
                },
                messages,
            ));
        }

        messages.push(Message::assistant_tool_calls(round.text, round.tool_calls.clone()));

        if iterations >= max_iterations {
            // Budget already spent: tell the model rather than running
            // what it just asked for, then force a final, tool-free
            // answer — the whole point of the cap is that the loop
            // cannot be spun forever.
            for call in &round.tool_calls {
                messages.push(Message::tool_result(call.id.clone(), BUDGET_EXHAUSTED_NOTE));
            }

            let final_stream = engine
                .chat(ChatRequest {
                    messages: messages.clone(),
                    tools: Vec::new(),
                    max_tokens,
                })
                .await;
            let final_round = collect_round(final_stream, &mut on_text).await?;

            return Ok((
                ThinkingOutcome {
                    text: final_round.text,
                    finish_reason: final_round.finish_reason,
                    iteration_cap_hit: true,
                },
                messages,
            ));
        }

        for call in &round.tool_calls {
            if requires_confirmation(broker.safety_class(&call.name)) {
                let prompt = format!("Should I run {}? Say yes to confirm.", call.name);
                if !confirm.confirm(&prompt).await {
                    messages.push(Message::tool_result(call.id.clone(), DECLINED_NOTE));
                    continue;
                }
            }

            let args = serde_json::from_str(&call.arguments).unwrap_or(serde_json::Value::Null);
            let result = broker.dispatch(&call.name, args, cancel.clone()).await;
            messages.push(Message::tool_result(call.id.clone(), tool_result_content(result)));
        }

        iterations += 1;

        // A barge-in firing mid-tool-dispatch should stop the loop before
        // it starts another round trip, rather than doing one more full
        // request the user has already interrupted.
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled { backend: "llm" });
        }
    }
}

/// Renders a [`ToolResult`] as the text content of a `Role::Tool` message.
///
/// Both variants become a JSON string: `Ok` because the model expects a
/// JSON-ish tool result body, `Err` wrapped in `{"error": ...}` rather
/// than left as bare text, so the model can distinguish "the tool told me
/// this" from "the tool failed and here's why" without guessing from
/// prose.
fn tool_result_content(result: ToolResult) -> String {
    match result {
        ToolResult::Ok(value) => value.to_string(),
        ToolResult::Err(message) => serde_json::json!({ "error": message }).to_string(),
    }
}

/// Drains one chat stream into its text, requested tool calls, and finish
/// reason.
///
/// Tool call fragments are accumulated per `id` across
/// [`ChatEvent::ToolCallDelta`] events — the wire protocol streams a call
/// in pieces the same way it streams text — and [`ChatEvent::ToolCallDone`]
/// is consumed as a no-op marker: nothing here needs it, since the
/// accumulator already reflects a call's args incrementally and
/// completeness is implied by the stream reaching `Done`.
async fn collect_round(
    mut stream: ChatEventStream,
    on_text: &mut impl FnMut(&str),
) -> Result<StreamOutcome, EngineError> {
    let mut text = String::new();
    let mut pending: HashMap<String, PendingCall> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut finish_reason = FinishReason::Other;

    while let Some(event) = stream.next().await {
        match event? {
            ChatEvent::TextDelta(delta) => {
                on_text(&delta);
                text.push_str(&delta);
            }
            ChatEvent::ToolCallDelta {
                id,
                name,
                args_delta,
            } => {
                if !pending.contains_key(&id) {
                    order.push(id.clone());
                }
                let entry = pending.entry(id).or_default();
                if let Some(name) = name {
                    entry.name = name;
                }
                entry.arguments.push_str(&args_delta);
            }
            ChatEvent::ToolCallDone { .. } => {}
            ChatEvent::Done { finish_reason: reason } => {
                finish_reason = reason;
                break;
            }
        }
    }

    let tool_calls = order
        .into_iter()
        .map(|id| {
            let call = pending.remove(&id).unwrap_or_default();
            ToolCallRequest {
                id,
                name: call.name,
                arguments: call.arguments,
            }
        })
        .collect();

    Ok(StreamOutcome {
        text,
        tool_calls,
        finish_reason,
    })
}

/// Accumulator for one tool call's streamed `name`/`arguments` fragments.
#[derive(Default)]
struct PendingCall {
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override_wins_when_set_and_parseable() {
        std::env::set_var(MAX_TOOL_ITERS_ENV, "3");
        assert_eq!(resolve_max_iterations(8), 3);
        std::env::remove_var(MAX_TOOL_ITERS_ENV);
    }

    #[test]
    fn configured_value_is_used_when_env_is_unset() {
        std::env::remove_var(MAX_TOOL_ITERS_ENV);
        assert_eq!(resolve_max_iterations(8), 8);
    }

    #[test]
    fn an_unparseable_env_value_falls_back_to_configured() {
        std::env::set_var(MAX_TOOL_ITERS_ENV, "not-a-number");
        assert_eq!(resolve_max_iterations(8), 8);
        std::env::remove_var(MAX_TOOL_ITERS_ENV);
    }

    #[test]
    fn tool_result_content_renders_ok_as_the_json_value() {
        let content = tool_result_content(ToolResult::Ok(serde_json::json!({"time": "noon"})));
        assert_eq!(content, r#"{"time":"noon"}"#);
    }

    #[test]
    fn tool_result_content_wraps_err_so_it_is_distinguishable() {
        let content = tool_result_content(ToolResult::Err("file not found".to_string()));
        assert_eq!(content, r#"{"error":"file not found"}"#);
    }
}
