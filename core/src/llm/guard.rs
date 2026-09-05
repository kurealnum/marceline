//! Cost/rate guardrail for cloud LLM endpoints (SPEC.md §3.1, §4.5).
//!
//! Local endpoints are free, but the same caps apply to them harmlessly —
//! the risk this guards against is a metered provider getting hit by a
//! retry storm or a barge-in loop and running up unbounded cost. Both caps
//! come straight from `[llm]` config: `max_tokens_per_turn` and
//! `max_requests_per_session`.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use super::{ChatEventStream, ChatRequest, LlmEngine, LlmInfo};
use crate::engine::EngineError;

/// Backend name used in [`EngineError`] messages and logs.
const BACKEND: &str = "llm";

/// The request counter [`SessionGuard`] enforces `max_requests_per_session`
/// against, split out from the guard itself so it can outlive any one
/// wrapped engine.
///
/// A one-shot caller (`marceline say-to-llm`) needs nothing more than
/// [`SessionGuard::new`]'s implicit per-instance counter — the process
/// exits after one request either way. A long-running daemon is different:
/// each turn opens a *fresh* engine carrying that turn's own
/// [`tokio_util::sync::CancellationToken`] (so barge-in can cancel just
/// that turn's in-flight call), which means a new [`SessionGuard`] wraps a
/// new engine every turn — but the request count it enforces must survive
/// across those turns, or the cap does nothing. Building one
/// `SessionGuardState` at daemon startup and handing every turn's guard a
/// clone (via [`SessionGuard::with_state`]) is how the count outlives the
/// per-turn engine while cancellation still doesn't.
pub struct SessionGuardState {
    requests_made: AtomicU32,
    /// `None` never resets — the original one-request-counter-per-process
    /// behavior. `Some(window)` makes the cap a rolling window instead of a
    /// permanent one: once `window` has elapsed since the count last reset,
    /// the next request resets it back to zero rather than staying refused
    /// forever, so a long-running daemon recovers on its own instead of
    /// needing a restart.
    window: Option<Duration>,
    window_start: Mutex<Instant>,
}

impl SessionGuardState {
    /// A counter that never resets on its own — matches the original
    /// behavior, appropriate for a one-shot process.
    pub fn new() -> Self {
        Self::with_rolling_window(None)
    }

    /// A counter that resets back to zero once `window` has elapsed since
    /// it was last reset (or created) — SPEC.md §4.5's cost cap made safe
    /// for a process that runs all day, per the module doc's daemon note.
    pub fn with_rolling_window(window: Option<Duration>) -> Self {
        Self {
            requests_made: AtomicU32::new(0),
            window,
            window_start: Mutex::new(Instant::now()),
        }
    }

    /// Rolls the window over if it has elapsed, then increments and
    /// returns the new ordinal (1-based) for this request.
    fn next_ordinal(&self) -> u32 {
        if let Some(window) = self.window {
            let mut window_start = self.window_start.lock().expect("window_start lock poisoned");
            if window_start.elapsed() >= window {
                *window_start = Instant::now();
                self.requests_made.store(0, Ordering::SeqCst);
            }
        }
        self.requests_made.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn requests_made(&self) -> u32 {
        self.requests_made.load(Ordering::SeqCst)
    }
}

impl Default for SessionGuardState {
    fn default() -> Self {
        Self::new()
    }
}

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
    state: Arc<SessionGuardState>,
}

impl<E: LlmEngine> SessionGuard<E> {
    /// Wraps `inner`, enforcing `max_tokens_per_turn` and
    /// `max_requests_per_session` (`[llm]` config, §3.1) on every call,
    /// with its own private, never-resetting counter — right for a
    /// one-shot process like `marceline say-to-llm`.
    pub fn new(inner: E, max_tokens_per_turn: u32, max_requests_per_session: u32) -> Self {
        Self::with_state(
            inner,
            max_tokens_per_turn,
            max_requests_per_session,
            Arc::new(SessionGuardState::new()),
        )
    }

    /// Wraps `inner`, enforcing the same caps against a [`SessionGuardState`]
    /// shared with other `SessionGuard`s (typically one per turn, each
    /// wrapping its own freshly-cancellable engine) — how a long-running
    /// daemon keeps one request count across turns without pinning every
    /// turn to the same `CancellationToken`. See [`SessionGuardState`]'s
    /// doc comment.
    pub fn with_state(
        inner: E,
        max_tokens_per_turn: u32,
        max_requests_per_session: u32,
        state: Arc<SessionGuardState>,
    ) -> Self {
        Self {
            inner,
            max_tokens_per_turn,
            max_requests_per_session,
            state,
        }
    }

    /// How many requests this guard has let through (or refused) so far.
    ///
    /// Exposed for callers that want to surface "N of M requests used this
    /// session" rather than only a hard cutoff.
    pub fn requests_made(&self) -> u32 {
        self.state.requests_made()
    }
}

#[async_trait]
impl<E: LlmEngine> LlmEngine for SessionGuard<E> {
    async fn chat(&self, mut req: ChatRequest) -> ChatEventStream {
        // Counted (not compare-and-swap'd) before the cap check: every
        // call attempt counts against the session, refused or not, so a
        // caller that ignores refusals and keeps calling still can't quietly
        // reset the counter by racing it.
        let ordinal = self.state.next_ordinal();
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

    #[tokio::test]
    async fn a_shared_state_enforces_the_cap_across_separate_guard_instances() {
        // Mirrors the daemon: a fresh `SessionGuard` (wrapping a fresh
        // engine, for a fresh per-turn cancellation token) each "turn",
        // but sharing one `SessionGuardState` — the cap must still bite
        // across those separate `SessionGuard` values.
        let state = Arc::new(SessionGuardState::new());
        for _ in 0..2 {
            let guard = SessionGuard::with_state(RecordingEngine::new(), 100, 2, Arc::clone(&state));
            let mut stream = guard.chat(request(10)).await;
            assert!(stream.next().await.unwrap().is_ok());
        }

        let guard = SessionGuard::with_state(RecordingEngine::new(), 100, 2, Arc::clone(&state));
        let mut stream = guard.chat(request(10)).await;
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.is_guardrail_refused());
    }

    #[tokio::test]
    async fn a_rolling_window_resets_the_count_once_it_elapses() {
        let state = Arc::new(SessionGuardState::with_rolling_window(Some(
            Duration::from_millis(20),
        )));
        let guard = SessionGuard::with_state(RecordingEngine::new(), 100, 1, Arc::clone(&state));

        assert!(guard.chat(request(10)).await.next().await.unwrap().is_ok());
        let err = guard
            .chat(request(10))
            .await
            .next()
            .await
            .unwrap()
            .unwrap_err();
        assert!(err.is_guardrail_refused());

        tokio::time::sleep(Duration::from_millis(30)).await;

        assert!(
            guard.chat(request(10)).await.next().await.unwrap().is_ok(),
            "the window elapsed, so the cap should have reset without a restart"
        );
    }
}
