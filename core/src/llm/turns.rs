//! Conversation turn management + context-window trimming (SPEC.md §5,
//! §9.10, EPIC 4.3).
//!
//! Working context is the current run's turns held in RAM, trimmed to fit
//! the model's context window. Basic trimming — drop the oldest turn until
//! it fits — is MVP-mandatory: a long conversation blows the window long
//! before the EPIC 10.4 background summarizer exists, so this can't wait
//! for that to land. [`TrimPolicy`] is the seam 10.4 replaces ("summarize
//! oldest" instead of "drop oldest") without touching [`TurnBuffer`]'s
//! interface.

use crate::llm::{Message, Role};

/// How [`TurnBuffer`] makes room when the conversation no longer fits the
/// context window.
///
/// A trait rather than a fixed function so EPIC 10.4 can swap "drop oldest"
/// for "summarize oldest" behind the same [`TurnBuffer::messages_for_request`]
/// call site.
pub trait TrimPolicy: Send + Sync {
    /// Shrinks `turns` in place until `estimate_tokens(turns) +
    /// system_prompt_tokens` fits `context_window`, or no more turns can be
    /// removed.
    ///
    /// Must never touch the system prompt — that is passed only as a token
    /// count, not as a `Message`, so a policy has nothing to drop it from.
    fn trim(&self, turns: &mut Vec<Message>, system_prompt_tokens: u32, context_window: u32);
}

/// The MVP-mandatory policy (§9.10): drop the oldest turn, repeat until the
/// conversation fits.
///
/// Deliberately coarse — no partial-turn truncation, no summarization. Good
/// enough to keep a long-running conversation from erroring out, and simple
/// enough that 10.4 has a clear, narrow seam to replace.
#[derive(Debug, Default, Clone, Copy)]
pub struct DropOldestTurn;

impl TrimPolicy for DropOldestTurn {
    fn trim(&self, turns: &mut Vec<Message>, system_prompt_tokens: u32, context_window: u32) {
        // A context window of 0 means the backend didn't report one
        // ([`crate::llm::LlmInfo::context_window`] is best-effort) — with
        // no real bound to trim to, trimming would just delete history for
        // no reason, so it is skipped rather than guessed at.
        if context_window == 0 {
            return;
        }

        while !turns.is_empty()
            && system_prompt_tokens + estimate_tokens(turns) > context_window
        {
            turns.remove(0);
        }
    }
}

/// A rough token-count estimate for a set of messages.
///
/// No tokenizer is wired in (that would tie this crate to a specific
/// provider's vocabulary, defeating the point of being OpenAI-standard
/// generic). ~4 chars/token undercounts badly for non-English text (CJK in
/// particular can run close to 1 char/token), so this uses ~2 chars/token —
/// conservative enough that trimming happens a little early rather than a
/// request exceeding the window because the estimate undercounted.
fn estimate_tokens(turns: &[Message]) -> u32 {
    turns
        .iter()
        .map(|m| (m.content.len() as u32) / 2 + 4)
        .sum()
}

/// The running conversation for one session: prior user/assistant turns,
/// trimmed to fit the model's context window before each request.
///
/// The system prompt (§3.2, EPIC 4.2) is not stored here — it is recompiled
/// per request and passed in fresh, so a `TurnBuffer` never has to know
/// about SOUL.md or memory retrieval.
pub struct TurnBuffer {
    turns: Vec<Message>,
    trim_policy: Box<dyn TrimPolicy>,
}

impl Default for TurnBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnBuffer {
    /// A turn buffer using the MVP [`DropOldestTurn`] policy.
    pub fn new() -> Self {
        Self::with_policy(Box::new(DropOldestTurn))
    }

    /// A turn buffer using a caller-supplied trim policy — the seam EPIC
    /// 10.4's summarizer plugs into.
    pub fn with_policy(trim_policy: Box<dyn TrimPolicy>) -> Self {
        Self {
            turns: Vec::new(),
            trim_policy,
        }
    }

    /// A turn buffer pre-seeded with `turns` (e.g. from
    /// [`crate::context::recent_context`], EPIC 10.2) instead of starting
    /// empty — how a restarted daemon resumes mid-session with the working
    /// context it had before going down.
    pub fn from_turns(turns: Vec<Message>) -> Self {
        Self {
            turns,
            trim_policy: Box::new(DropOldestTurn),
        }
    }

    /// Appends a user turn.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.turns.push(Message::new(Role::User, text));
    }

    /// Appends the model's final assistant turn.
    ///
    /// Only the *final* turn belongs here — intermediate tool-call
    /// round-trips are the tool broker's concern (EPIC 6), not turn
    /// management's.
    pub fn push_assistant(&mut self, text: impl Into<String>) {
        self.turns.push(Message::new(Role::Assistant, text));
    }

    /// Trims the buffer to fit `context_window` alongside `system_prompt`,
    /// then returns the full message list for [`crate::llm::ChatRequest`]:
    /// the system prompt first, followed by the (possibly trimmed) turns.
    ///
    /// The system prompt is never dropped — trimming only ever removes
    /// entries from `turns`, never the message built from `system_prompt`.
    pub fn messages_for_request(&mut self, system_prompt: &str, context_window: u32) -> Vec<Message> {
        let system_prompt_tokens = (system_prompt.len() as u32) / 4 + 4;
        self.trim_policy
            .trim(&mut self.turns, system_prompt_tokens, context_window);

        let mut messages = Vec::with_capacity(self.turns.len() + 1);
        messages.push(Message::new(Role::System, system_prompt));
        messages.extend(self.turns.iter().cloned());
        messages
    }

    /// The turns currently held, oldest first, system prompt excluded.
    pub fn turns(&self) -> &[Message] {
        &self.turns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long_turn(role: Role, chars: usize) -> Message {
        Message::new(role, "a".repeat(chars))
    }

    #[test]
    fn keeps_all_turns_when_they_fit_the_window() {
        let mut buffer = TurnBuffer::new();
        buffer.push_user("hello");
        buffer.push_assistant("hi there");

        let messages = buffer.messages_for_request("You are Marceline.", 10_000);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].content, "hello");
        assert_eq!(messages[2].content, "hi there");
    }

    #[test]
    fn drops_oldest_turns_first_until_it_fits() {
        let mut buffer = TurnBuffer::new();
        buffer.turns.push(long_turn(Role::User, 400));
        buffer.turns.push(long_turn(Role::Assistant, 400));
        buffer.turns.push(long_turn(Role::User, 20));

        // Small enough that only the most recent turn fits alongside a
        // negligible system prompt.
        let messages = buffer.messages_for_request("sys", 20);

        assert_eq!(messages.len(), 2, "system prompt + the one surviving turn");
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].content.len(), 20);
    }

    #[test]
    fn never_drops_the_system_prompt() {
        let mut buffer = TurnBuffer::new();
        buffer.turns.push(long_turn(Role::User, 10_000));

        // A window far too small for anything to fit.
        let messages = buffer.messages_for_request("You are Marceline.", 1);

        assert_eq!(messages.len(), 1, "every turn dropped, system prompt kept");
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are Marceline.");
    }

    #[test]
    fn a_second_turns_request_carries_the_first_turns_messages() {
        // Mirrors how `cli::converse::run_loop` drives a `TurnBuffer` turn
        // by turn (EPIC 10): push the user text, build the request, then
        // push the assistant's reply once it's known — the second turn's
        // request must still see the first turn's exchange.
        let mut buffer = TurnBuffer::new();

        buffer.push_user("my name is Oscar");
        let first_request = buffer.messages_for_request("You are Marceline.", 10_000);
        assert_eq!(first_request.len(), 2, "system prompt + the first user turn");
        buffer.push_assistant("Nice to meet you, Oscar!");

        buffer.push_user("what is my name?");
        let second_request = buffer.messages_for_request("You are Marceline.", 10_000);

        assert_eq!(
            second_request.len(),
            4,
            "system prompt + first user/assistant turn + the new user turn"
        );
        assert_eq!(second_request[0].role, Role::System);
        assert_eq!(second_request[1].content, "my name is Oscar");
        assert_eq!(second_request[2].content, "Nice to meet you, Oscar!");
        assert_eq!(second_request[3].content, "what is my name?");
    }

    #[test]
    fn a_turn_buffer_seeded_past_the_window_trims_via_messages_for_request() {
        // Mirrors the path `cli::converse::run_loop` drives: a buffer
        // pre-seeded from history (`TurnBuffer::from_turns`), then trimmed
        // to a small configured window on the next request.
        let mut buffer = TurnBuffer::from_turns(vec![
            long_turn(Role::User, 400),
            long_turn(Role::Assistant, 400),
            long_turn(Role::User, 20),
        ]);

        let messages = buffer.messages_for_request("sys", 20);

        assert_eq!(messages.len(), 2, "system prompt + the one surviving turn");
        assert_eq!(messages[1].content.len(), 20);
    }

    #[test]
    fn a_zero_context_window_skips_trimming() {
        let mut buffer = TurnBuffer::new();
        buffer.turns.push(long_turn(Role::User, 10_000));
        buffer.turns.push(long_turn(Role::Assistant, 10_000));

        let messages = buffer.messages_for_request("sys", 0);

        assert_eq!(messages.len(), 3, "no reported window means nothing gets dropped");
    }
}
