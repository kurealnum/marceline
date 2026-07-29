//! Background summarizer: distills recent conversation turns into a durable
//! memory entry (SPEC.md §5.1, §5.2, EPIC 10.4).
//!
//! This sits off the main turn loop on purpose — the orchestrator (EPIC
//! 8.1) drives real-time chat, and running an extra LLM round-trip plus a
//! blocking SQLite write on that path would add latency to every spoken
//! turn for no benefit the user notices immediately. [`summarize_session`]
//! is a plain async function with no dependency on the orchestrator; wiring
//! it into a periodic/idle trigger is later work (EPIC 8.1/10.6), out of
//! scope here.
//!
//! ## Blocking I/O: caller spawns, this module doesn't `spawn_blocking`
//!
//! [`HistoryStore::recent_turns`] and [`crate::memory::store_memory`] are
//! synchronous rusqlite calls — normally exactly what
//! [`tokio::task::spawn_blocking`] is for. That doesn't fit here: this
//! function's second argument is `&mut dyn EmbeddingPipeline`, a borrow
//! with this call's lifetime, and `spawn_blocking`'s closure must be
//! `'static`. Making the pipeline `'static` (an owned `Arc`/`Box`) would
//! ripple that requirement into every other `store_memory` caller for no
//! reason specific to summarization.
//!
//! Instead, [`summarize_session`] is documented as blocking-inside-async
//! and the *caller* is responsible for running it on its own
//! `tokio::spawn`ed task (this is the "off the main loop" requirement from
//! the issue) rather than `.await`ing it inline on a worker thread shared
//! with latency-sensitive work like STT/TTS. This matches how
//! `HistoryStore::log_turn`'s own doc comment already puts the
//! spawn-or-spawn_blocking choice on the caller.

use crate::embedding::EmbeddingPipeline;
use crate::engine::EngineError;
use crate::history::{HistoryError, HistoryStore, TurnRecord};
use crate::llm::{ChatEvent, ChatRequest, LlmEngine, Message, Role, Trust};
use crate::memory::{store_memory, MemoryError};

use async_trait::async_trait;
use futures::StreamExt;

/// Errors from running the summarization step itself (as opposed to the
/// storage step, see [`SummarizerError`]).
#[derive(Debug, thiserror::Error)]
pub enum SummarizeError {
    /// The backend errored mid-stream, or the stream never opened.
    #[error("LLM backend error while summarizing: {0}")]
    Engine(#[from] EngineError),
}

/// Errors from [`summarize_session`] end-to-end.
#[derive(Debug, thiserror::Error)]
pub enum SummarizerError {
    /// Reading recent turns failed.
    #[error(transparent)]
    History(#[from] HistoryError),
    /// Distilling the turns into a summary failed.
    #[error(transparent)]
    Summarize(#[from] SummarizeError),
    /// Embedding or storing the resulting memory failed.
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// Turns a run of conversation turns into a short standing-fact summary.
///
/// A trait rather than calling [`LlmEngine`] directly so tests can stub the
/// distillation step without a real model backend (see
/// [`LlmSummarizer`] for the real implementation, and this module's tests
/// for the fake used elsewhere in this crate for `LlmEngine` itself,
/// e.g. `crate::llm::guard`'s `RecordingEngine`).
#[async_trait]
pub trait Summarizer {
    /// Distills `turns` (oldest first, as returned by
    /// [`HistoryStore::recent_turns`]) into one summary string.
    ///
    /// Never sees provenance: taint derivation is [`derive_provenance`]'s
    /// job, kept entirely separate from the text of the summary so it
    /// can't be swayed by anything the LLM writes.
    async fn summarize(&self, turns: &[TurnRecord]) -> Result<String, SummarizeError>;
}

/// Real [`Summarizer`] backed by any [`LlmEngine`] — the production path.
///
/// Asks the model to distill the given turns into a short, standing fact
/// or summary, then collects [`ChatEvent::TextDelta`] chunks from the
/// response stream into one `String`. Tool calls are not expected for this
/// prompt, so [`ChatEvent::ToolCallDelta`]/[`ChatEvent::ToolCallDone`]
/// events are ignored rather than treated as an error — a backend that
/// ignores the empty `tools` list and calls one anyway shouldn't crash
/// summarization, it should just contribute nothing to the summary text.
pub struct LlmSummarizer<E> {
    engine: E,
    max_tokens: u32,
}

impl<E: LlmEngine> LlmSummarizer<E> {
    /// Wraps `engine`, capping the summary response at `max_tokens` —
    /// summaries are meant to be short standing facts, not full
    /// transcripts, so this is deliberately small relative to
    /// `[llm].max_tokens_per_turn`.
    pub fn new(engine: E, max_tokens: u32) -> Self {
        Self { engine, max_tokens }
    }
}

/// Renders `turns` as `role: text` lines for the summarization prompt.
fn render_turns(turns: &[TurnRecord]) -> String {
    turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.text))
        .collect::<Vec<_>>()
        .join("\n")
}

const SUMMARIZE_SYSTEM_PROMPT: &str = "You distill a conversation excerpt into one short, \
standing fact or summary worth remembering long-term. Respond with only the distilled \
fact/summary text, no preamble, no quotes, no bullet points.";

#[async_trait]
impl<E: LlmEngine + Send + Sync> Summarizer for LlmSummarizer<E> {
    async fn summarize(&self, turns: &[TurnRecord]) -> Result<String, SummarizeError> {
        let request = ChatRequest {
            messages: vec![
                Message::new(Role::System, SUMMARIZE_SYSTEM_PROMPT),
                Message::new(Role::User, render_turns(turns)),
            ],
            tools: Vec::new(),
            max_tokens: self.max_tokens,
        };

        let mut stream = self.engine.chat(request).await;
        let mut summary = String::new();
        while let Some(event) = stream.next().await {
            if let ChatEvent::TextDelta(chunk) = event? {
                summary.push_str(&chunk);
            }
        }
        Ok(summary)
    }
}

/// Derives the memory entry's provenance from the [`Trust`] of the turns it
/// was distilled from (SPEC.md §5.1).
///
/// The rule, in priority order:
/// 1. Any source turn is [`Trust::ToolUntrusted`] → the result is
///    [`Trust::ToolUntrusted`], unconditionally. This is the load-bearing
///    rule: an LLM summary can freely blend a `ToolUntrusted` turn's
///    content into fluent prose that reads exactly like a trusted fact,
///    and once written to memory that text gets re-injected into future
///    system prompts (EPIC 10.5) — laundering its taint away here would
///    turn a one-off untrusted tool/web response into a persistent,
///    repeatedly-injected prompt-injection vector. Nothing overrides this,
///    not even a majority of `User` turns in the same batch.
/// 2. Otherwise, any source turn is [`Trust::User`] → [`Trust::User`]: a
///    user-originated fact is the most valuable of the two remaining tags,
///    so a summary blending user and assistant turns keeps the stronger
///    one.
/// 3. Otherwise (every turn is [`Trust::Assistant`]) → [`Trust::Assistant`].
///
/// An empty slice falls through to [`Trust::Assistant`] as an arbitrary but
/// harmless default; [`summarize_session`] never actually calls this with
/// an empty slice since it returns early before reaching this point.
pub fn derive_provenance(turns: &[TurnRecord]) -> Trust {
    if turns.iter().any(|t| t.provenance == Trust::ToolUntrusted) {
        Trust::ToolUntrusted
    } else if turns.iter().any(|t| t.provenance == Trust::User) {
        Trust::User
    } else {
        Trust::Assistant
    }
}

/// Reads the most recent `limit` turns of `session_id`, distills them with
/// `summarizer`, and stores the result as a new long-term memory via
/// [`store_memory`] — the EPIC 10.4 background summarizer.
///
/// Returns `Ok(None)` (not an error) when there are no turns for
/// `session_id` yet: a session that hasn't produced any history is not a
/// failure, there is simply nothing to distill. Otherwise returns
/// `Ok(Some(id))` for the newly inserted memory row.
///
/// See this module's doc comment for why the blocking `recent_turns`/
/// `store_memory` calls inside this `async fn` are the caller's
/// responsibility to isolate (run this whole function via `tokio::spawn`),
/// rather than this function wrapping them in `spawn_blocking` itself.
pub async fn summarize_session(
    store: &HistoryStore,
    pipeline: &mut dyn EmbeddingPipeline,
    summarizer: &(impl Summarizer + Sync),
    session_id: &str,
    limit: usize,
    created_at_ms: i64,
) -> Result<Option<i64>, SummarizerError> {
    let turns = store.recent_turns(session_id, limit)?;
    if turns.is_empty() {
        return Ok(None);
    }

    let provenance = derive_provenance(&turns);
    let summary = summarizer.summarize(&turns).await?;
    let id = store_memory(store, pipeline, summary, provenance, created_at_ms)?;
    Ok(Some(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::fake::FakeEmbedder;
    use crate::history::NewTurn;

    fn store() -> HistoryStore {
        let dir = tempfile::tempdir().unwrap();
        // Leak the tempdir so the DB file outlives this helper, matching
        // memory.rs's test helper.
        let path = dir.path().join("summarizer.db");
        std::mem::forget(dir);
        HistoryStore::open(path).unwrap()
    }

    fn turn(role: &str, text: &str, provenance: Trust) -> TurnRecord {
        TurnRecord {
            id: 0,
            session_id: "s1".to_string(),
            timestamp_ms: 1_000,
            role: role.to_string(),
            text: text.to_string(),
            provenance,
            interrupted: false,
        }
    }

    /// A [`Summarizer`] stub that returns a fixed string, so tests don't
    /// need a real `LlmEngine` backend (there is no ONNX/HTTP dependency
    /// available in this sandbox or CI, same reasoning as
    /// `embedding::fake::FakeEmbedder` for the embedding side).
    struct FixedSummarizer(String);

    #[async_trait]
    impl Summarizer for FixedSummarizer {
        async fn summarize(&self, _turns: &[TurnRecord]) -> Result<String, SummarizeError> {
            Ok(self.0.clone())
        }
    }

    // --- derive_provenance -------------------------------------------------

    #[test]
    fn derive_provenance_all_user_is_user() {
        let turns = vec![
            turn("user", "a", Trust::User),
            turn("user", "b", Trust::User),
        ];
        assert_eq!(derive_provenance(&turns), Trust::User);
    }

    #[test]
    fn derive_provenance_all_assistant_is_assistant() {
        let turns = vec![
            turn("assistant", "a", Trust::Assistant),
            turn("assistant", "b", Trust::Assistant),
        ];
        assert_eq!(derive_provenance(&turns), Trust::Assistant);
    }

    #[test]
    fn derive_provenance_mixed_user_and_assistant_is_user() {
        let turns = vec![
            turn("assistant", "a", Trust::Assistant),
            turn("user", "b", Trust::User),
        ];
        assert_eq!(derive_provenance(&turns), Trust::User);
    }

    #[test]
    fn derive_provenance_any_untrusted_forces_untrusted_even_with_user_turns() {
        let turns = vec![
            turn("user", "a", Trust::User),
            turn("tool", "b", Trust::ToolUntrusted),
            turn("assistant", "c", Trust::Assistant),
        ];
        assert_eq!(derive_provenance(&turns), Trust::ToolUntrusted);
    }

    #[test]
    fn derive_provenance_single_untrusted_turn_is_untrusted() {
        let turns = vec![turn("tool", "a", Trust::ToolUntrusted)];
        assert_eq!(derive_provenance(&turns), Trust::ToolUntrusted);
    }

    #[test]
    fn derive_provenance_empty_defaults_to_assistant() {
        assert_eq!(derive_provenance(&[]), Trust::Assistant);
    }

    // --- summarize_session ---------------------------------------------------

    #[tokio::test]
    async fn no_turns_found_is_a_noop_not_an_error() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);
        let summarizer = FixedSummarizer("should not be called".to_string());

        let result = summarize_session(&store, &mut pipeline, &summarizer, "missing", 20, 1_000)
            .await
            .unwrap();

        assert_eq!(result, None);
        assert!(store.all_memories().unwrap().is_empty());
    }

    #[tokio::test]
    async fn stores_a_memory_with_provenance_derived_from_the_source_turns() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);
        let summarizer = FixedSummarizer("the user prefers dark mode".to_string());

        store
            .log_turn(NewTurn {
                session_id: "s1".to_string(),
                timestamp_ms: 1_000,
                role: "user".to_string(),
                text: "I prefer dark mode".to_string(),
                provenance: Trust::User,
                interrupted: false,
            })
            .unwrap();
        store
            .log_turn(NewTurn {
                session_id: "s1".to_string(),
                timestamp_ms: 1_001,
                role: "assistant".to_string(),
                text: "Got it, dark mode it is.".to_string(),
                provenance: Trust::Assistant,
                interrupted: false,
            })
            .unwrap();

        let source_turns = store.recent_turns("s1", 20).unwrap();
        let expected_provenance = derive_provenance(&source_turns);

        let id = summarize_session(&store, &mut pipeline, &summarizer, "s1", 20, 2_000)
            .await
            .unwrap()
            .expect("turns existed, should have stored a memory");

        let memories = store.all_memories().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].id, id);
        assert_eq!(memories[0].text, "the user prefers dark mode");
        assert_eq!(memories[0].provenance, expected_provenance);
        assert_eq!(expected_provenance, Trust::User);
    }

    #[tokio::test]
    async fn a_single_untrusted_source_turn_forces_untrusted_memory_provenance() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);
        let summarizer = FixedSummarizer("some fact pulled from a web page".to_string());

        store
            .log_turn(NewTurn {
                session_id: "s2".to_string(),
                timestamp_ms: 1_000,
                role: "user".to_string(),
                text: "look this up for me".to_string(),
                provenance: Trust::User,
                interrupted: false,
            })
            .unwrap();
        store
            .log_turn(NewTurn {
                session_id: "s2".to_string(),
                timestamp_ms: 1_001,
                role: "tool".to_string(),
                text: "<web page content>".to_string(),
                provenance: Trust::ToolUntrusted,
                interrupted: false,
            })
            .unwrap();

        summarize_session(&store, &mut pipeline, &summarizer, "s2", 20, 2_000)
            .await
            .unwrap();

        let memories = store.all_memories().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].provenance, Trust::ToolUntrusted);
    }
}
