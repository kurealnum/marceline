//! Cost/rate guardrail for cloud LLM endpoints (SPEC.md §3.1, §4.5).
//!
//! Local endpoints are free, but the same caps apply to them harmlessly —
//! the risk this guards against is a metered provider getting hit by a
//! retry storm or a barge-in loop and running up unbounded cost. Both caps
//! come straight from `[llm]` config: `max_tokens_per_turn` and
//! `max_requests_per_session`.

use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;

use super::{ChatEventStream, ChatRequest, LlmEngine, LlmInfo};
use crate::engine::EngineError;

/// Backend name used in [`EngineError`] messages and logs.
const BACKEND: &str = "llm";

/// Wraps an [`LlmEngine`] with the two `[llm]`-configured caps: a per-turn
/// token cap and a per-session request cap.
///
/// A breach of either cap **refuses the request outright** — the wrapped
/// engine is never called — rather than calling it and hoping it errors
/// out cheaply. That is the difference between "cost stayed bounded" and
/// "cost was bounded by whatever the provider decided to charge for the
/// attempt".
pub struct SessionGuard<E> {
    inner: E,
    max_tokens_per_turn: u32,
    max_requests_per_session: u32,
    requests_made: AtomicU32,
}

impl<E: LlmEngine> SessionGuard<E> {
    /// Wraps `inner`, enforcing `max_tokens_per_turn` and
    /// `max_requests_per_session` (`[llm]` config, §3.1) on every call.
    pub fn new(inner: E, max_tokens_per_turn: u32, max_requests_per_session: u32) -> Self {
        Self {
            inner,
            max_tokens_per_turn,
            max_requests_per_session,
            requests_made: AtomicU32::new(0),
        }
    }

    /// How many requests this guard has let through (or refused) so far.
    ///
    /// Exposed for callers that want to surface "N of M requests used this
    /// session" rather than only a hard cutoff.
    pub fn requests_made(&self) -> u32 {
        self.requests_made.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl<E: LlmEngine> LlmEngine for SessionGuard<E> {
    async fn chat(&self, mut req: ChatRequest) -> ChatEventStream {
        // Counted (not compare-and-swap'd) before the cap check: every
        // call attempt counts against the session, refused or not, so a
        // caller that ignores refusals and keeps calling still can't quietly
        // reset the counter by racing it.
        let ordinal = self.requests_made.fetch_add(1, Ordering::SeqCst) + 1;
        if ordinal > self.max_requests_per_session {
            return refusal(format!(
                "session request cap reached ({} of {} requests used)",
                ordinal - 1,
                self.max_requests_per_session
            ));
        }

        // The cap is authoritative regardless of what the caller asked
        // for — clamping here means a caller forgetting to read config
        // still can't overrun the turn budget.
        req.max_tokens = req.max_tokens.min(self.max_tokens_per_turn);

        self.inner.chat(req).await
    }

    fn info(&self) -> LlmInfo {
        self.inner.info()
    }
}

/// A chat stream carrying exactly one guardrail refusal, so a caller has
/// the same single error path whether the backend failed or was never
/// called at all (invariant 1, §2.4.1).
fn refusal(message: String) -> ChatEventStream {
    Box::pin(futures::stream::once(async move {
        Err(EngineError::GuardrailRefused {
            backend: BACKEND,
            message,
        })
    }))
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::llm::{ChatEvent, FinishReason, Message, Role, ToolSpec};

    /// A stub backend that reports what `max_tokens` it was actually
    /// called with, so tests can assert the guard clamps it rather than
    /// trusting the caller.
    struct RecordingEngine {
        last_max_tokens: std::sync::Mutex<Option<u32>>,
    }

    impl RecordingEngine {
        fn new() -> Self {
            Self {
                last_max_tokens: std::sync::Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl LlmEngine for RecordingEngine {
        async fn chat(&self, req: ChatRequest) -> ChatEventStream {
            *self.last_max_tokens.lock().unwrap() = Some(req.max_tokens);
            Box::pin(futures::stream::once(async move {
                Ok(ChatEvent::Done {
                    finish_reason: FinishReason::Stop,
                })
            }))
        }

        fn info(&self) -> LlmInfo {
            LlmInfo {
                name: "stub".to_string(),
                context_window: 1000,
                supports_tools: false,
                streaming: true,
            }
        }
    }

    fn request(max_tokens: u32) -> ChatRequest {
        ChatRequest {
            messages: vec![Message::new(Role::User, "hi")],
            tools: Vec::<ToolSpec>::new(),
            max_tokens,
        }
    }

    #[tokio::test]
    async fn clamps_max_tokens_to_the_per_turn_cap() {
        let guard = SessionGuard::new(RecordingEngine::new(), 100, 10);
        let mut stream = guard.chat(request(5_000)).await;
        assert!(stream.next().await.unwrap().is_ok());
        assert_eq!(*guard.inner.last_max_tokens.lock().unwrap(), Some(100));
    }

    #[tokio::test]
    async fn leaves_max_tokens_alone_when_already_under_the_cap() {
        let guard = SessionGuard::new(RecordingEngine::new(), 2048, 10);
        let mut stream = guard.chat(request(64)).await;
        assert!(stream.next().await.unwrap().is_ok());
        assert_eq!(*guard.inner.last_max_tokens.lock().unwrap(), Some(64));
    }

    #[tokio::test]
    async fn refuses_once_the_session_request_cap_is_exhausted() {
        let guard = SessionGuard::new(RecordingEngine::new(), 100, 2);

        for _ in 0..2 {
            let mut stream = guard.chat(request(10)).await;
            assert!(stream.next().await.unwrap().is_ok());
        }

        let mut stream = guard.chat(request(10)).await;
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.is_guardrail_refused());
    }

    #[tokio::test]
    async fn a_refused_request_never_reaches_the_wrapped_engine() {
        let guard = SessionGuard::new(RecordingEngine::new(), 100, 0);

        let mut stream = guard.chat(request(10)).await;
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.is_guardrail_refused());
        assert_eq!(*guard.inner.last_max_tokens.lock().unwrap(), None);
    }
}
