//! Integration tests for [`OpenAiCompatibleEngine`] (EPIC 4.1) against a
//! local, hand-rolled OpenAI-compatible SSE server — real bytes over a real
//! TCP socket, not a mocked HTTP client, so the SSE line-framing and
//! tool-call-index bookkeeping in `core/src/llm/openai.rs` gets exercised
//! for real.

use std::convert::Infallible;
use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use http_body_util::{combinators::BoxBody, BodyExt, StreamBody};
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::Response;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use marceline_core::config::LlmConfig;
use marceline_core::{ChatEvent, ChatRequest, FinishReason, LlmEngine, Message, OpenAiCompatibleEngine, Role};

/// Starts a fake OpenAI-compatible server that streams `chunks` (each a raw
/// SSE `data: ...\n\n` line, optionally delayed) as its response body to
/// every request it receives, and returns its base URL.
async fn start_fake_server(chunks: Vec<(Duration, String)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");

    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(accepted) => accepted,
                Err(_) => return,
            };
            let io = TokioIo::new(stream);
            let chunks = chunks.clone();

            tokio::spawn(async move {
                let service = service_fn(move |_req| {
                    let chunks = chunks.clone();
                    async move { Ok::<_, Infallible>(sse_response(chunks)) }
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, service)
                    .await;
            });
        }
    });

    format!("http://{addr}/v1")
}

/// Builds a streaming `text/event-stream` response out of `chunks`.
fn sse_response(chunks: Vec<(Duration, String)>) -> Response<BoxBody<Bytes, Infallible>> {
    let stream = futures::stream::unfold(chunks.into_iter(), |mut remaining| async move {
        let (delay, line) = remaining.next()?;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        Some((Ok::<_, Infallible>(Frame::data(Bytes::from(line))), remaining))
    });

    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(BodyExt::boxed(StreamBody::new(stream)))
        .expect("build response")
}

fn test_config(base_url: String) -> LlmConfig {
    std::env::set_var("MARCELINE_TEST_LLM_KEY", "test-key");
    LlmConfig {
        backend: "openai-compatible".to_string(),
        base_url,
        model: "test-model".to_string(),
        api_key_env: "MARCELINE_TEST_LLM_KEY".to_string(),
        max_tokens_per_turn: 512,
        max_requests_per_session: 100,
        max_tool_iterations_per_turn: 4,
    }
}

fn user_request(text: &str) -> ChatRequest {
    ChatRequest {
        messages: vec![Message {
            role: Role::User,
            content: text.to_string(),
        }],
        tools: vec![],
        max_tokens: 512,
    }
}

#[tokio::test]
async fn streams_ordered_text_deltas_then_done() {
    let chunks = vec![
        (Duration::ZERO, sse_line(r#"{"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#)),
        (Duration::ZERO, sse_line(r#"{"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#)),
        (Duration::ZERO, sse_line(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#)),
        (Duration::ZERO, "data: [DONE]\n\n".to_string()),
    ];
    let base_url = start_fake_server(chunks).await;
    let config = test_config(base_url);
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");

    let events: Vec<_> = engine
        .chat(user_request("hello"))
        .await
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|item| item.expect("no stream error"))
        .collect();

    assert_eq!(
        events,
        vec![
            ChatEvent::TextDelta("Hel".to_string()),
            ChatEvent::TextDelta("lo".to_string()),
            ChatEvent::Done {
                finish_reason: FinishReason::Stop
            },
        ]
    );
}

#[tokio::test]
async fn tool_call_chunks_interleave_delta_and_done_events() {
    let chunks = vec![
        (
            Duration::ZERO,
            sse_line(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
            ),
        ),
        (
            Duration::ZERO,
            sse_line(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"nyc\"}"}}]},"finish_reason":null}]}"#,
            ),
        ),
        (Duration::ZERO, sse_line(r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#)),
    ];
    let base_url = start_fake_server(chunks).await;
    let config = test_config(base_url);
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");

    let events: Vec<_> = engine
        .chat(user_request("what's the weather in nyc"))
        .await
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|item| item.expect("no stream error"))
        .collect();

    assert_eq!(
        events,
        vec![
            ChatEvent::ToolCallDelta {
                id: "call_1".to_string(),
                name: Some("get_weather".to_string()),
                args_delta: "{\"city\":".to_string(),
            },
            ChatEvent::ToolCallDelta {
                id: "call_1".to_string(),
                name: None,
                args_delta: "\"nyc\"}".to_string(),
            },
            ChatEvent::ToolCallDone {
                id: "call_1".to_string(),
            },
            ChatEvent::Done {
                finish_reason: FinishReason::ToolCalls,
            },
        ]
    );
}

#[tokio::test]
async fn cancellation_ends_the_stream_without_a_worker_error() {
    let chunks = vec![
        (Duration::ZERO, sse_line(r#"{"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#)),
        // Long enough that the test's cancel always wins the race.
        (Duration::from_millis(500), sse_line(r#"{"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#)),
    ];
    let base_url = start_fake_server(chunks).await;
    let config = test_config(base_url);
    let cancel = CancellationToken::new();
    let engine = OpenAiCompatibleEngine::new(&config, cancel.clone()).expect("engine");

    let mut stream = engine.chat(user_request("hello")).await;
    assert_eq!(stream.next().await.unwrap().unwrap(), ChatEvent::TextDelta("Hel".to_string()));

    cancel.cancel();
    let next = stream.next().await.expect("stream ends with an item, not silently");
    let err = next.expect_err("cancellation surfaces as an error, not a silent end");
    assert!(err.is_cancelled(), "expected a cancelled error, got {err:?}");
    assert!(stream.next().await.is_none(), "stream must end after reporting cancellation");
}

fn sse_line(json: &str) -> String {
    format!("data: {json}\n\n")
}
