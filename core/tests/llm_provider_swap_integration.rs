//! EPIC 4.4 — proves the OpenAI-compatible client is provider-agnostic by
//! running the *same* `OpenAiCompatibleEngine` code against two fake
//! backends shaped like real-world quirks found while verifying against
//! LM Studio and a hosted provider: differing `finish_reason` spellings,
//! SSE keep-alive comment lines, and a `data:` frame split across two raw
//! TCP writes. Only `[llm].base_url` differs between the two runs in this
//! test — never a code path — which is the whole promise SPEC.md §2.4
//! makes testable.

use std::convert::Infallible;

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

/// Starts a fake OpenAI-compatible server streaming raw `chunks` as its
/// response body, and returns its base URL. Each chunk is written as its
/// own TCP frame, so a chunk that is not itself a complete `data:` line
/// exercises the client's cross-chunk buffering.
async fn start_fake_server(chunks: Vec<String>) -> String {
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

fn sse_response(chunks: Vec<String>) -> Response<BoxBody<Bytes, Infallible>> {
    let stream = futures::stream::unfold(chunks.into_iter(), |mut remaining| async move {
        let chunk = remaining.next()?;
        Some((Ok::<_, Infallible>(Frame::data(Bytes::from(chunk))), remaining))
    });

    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(BodyExt::boxed(StreamBody::new(stream)))
        .expect("build response")
}

fn config_for(base_url: String, env_var: &'static str) -> LlmConfig {
    std::env::set_var(env_var, "test-key");
    LlmConfig {
        backend: "openai-compatible".to_string(),
        base_url,
        model: "test-model".to_string(),
        api_key_env: env_var.to_string(),
        max_tokens_per_turn: 512,
        max_requests_per_session: 100,
        max_tool_iterations_per_turn: 4,
    }
}

async fn run_hello(base_url: String, env_var: &'static str) -> Vec<ChatEvent> {
    let config = config_for(base_url, env_var);
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");
    let request = ChatRequest {
        messages: vec![Message::new(Role::User, "hello")],
        tools: vec![],
        max_tokens: 512,
    };
    engine
        .chat(request)
        .await
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|item| item.expect("no stream error"))
        .collect()
}

/// Stands in for a local LM Studio endpoint: strict OpenAI framing, one
/// SSE line per TCP chunk, `finish_reason: "stop"`.
#[tokio::test]
async fn streams_hello_from_an_lm_studio_shaped_backend() {
    let chunks = vec![
        sse_line(r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#),
        sse_line(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
        "data: [DONE]\n\n".to_string(),
    ];
    let base_url = start_fake_server(chunks).await;

    let events = run_hello(base_url, "MARCELINE_TEST_LM_STUDIO_KEY").await;

    assert_eq!(
        events,
        vec![
            ChatEvent::TextDelta("Hello".to_string()),
            ChatEvent::Done {
                finish_reason: FinishReason::Stop
            },
        ]
    );
}

/// Stands in for a hosted OpenAI-compatible proxy in front of a different
/// underlying model family: SSE keep-alive comment lines interspersed, a
/// `data:` frame split mid-line across two TCP writes, and an
/// Anthropic-style `finish_reason: "end_turn"` instead of `"stop"`.
#[tokio::test]
async fn streams_hello_from_a_hosted_provider_shaped_backend() {
    let chunks = vec![
        ": keep-alive\n\n".to_string(),
        // Split mid-JSON across two chunks — the client must buffer
        // across chunk boundaries rather than assume one line per read.
        r#"data: {"choices":[{"delta":{"content":"Hel"#.to_string(),
        r#"lo"},"finish_reason":null}]}"#.to_string() + "\n\n",
        ": keep-alive\n\n".to_string(),
        sse_line(r#"{"choices":[{"delta":{},"finish_reason":"end_turn"}]}"#),
    ];
    let base_url = start_fake_server(chunks).await;

    let events = run_hello(base_url, "MARCELINE_TEST_HOSTED_KEY").await;

    assert_eq!(
        events,
        vec![
            ChatEvent::TextDelta("Hello".to_string()),
            ChatEvent::Done {
                finish_reason: FinishReason::Stop
            },
        ]
    );
}

/// Slow-path sanity check that both fake backends can be reached from the
/// same wall-clock run, i.e. nothing about one server's setup leaks into
/// the other — the two tests above are the real assertion.
#[tokio::test]
async fn base_url_is_the_only_thing_that_differs_between_runs() {
    let a = start_fake_server(vec![
        sse_line(r#"{"choices":[{"delta":{"content":"A"},"finish_reason":null}]}"#),
        sse_line(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
    ])
    .await;
    let b = start_fake_server(vec![
        sse_line(r#"{"choices":[{"delta":{"content":"B"},"finish_reason":null}]}"#),
        sse_line(r#"{"choices":[{"delta":{},"finish_reason":"max_tokens"}]}"#),
    ])
    .await;

    let mut config = config_for(a, "MARCELINE_TEST_SWAP_KEY");
    let events_a = run_hello(config.base_url.clone(), "MARCELINE_TEST_SWAP_KEY").await;

    // Only base_url changes for the second run; model/api_key_env/backend
    // are untouched, matching a real config.toml edit (§3.1).
    config.base_url = b;
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");
    let events_b: Vec<_> = engine
        .chat(ChatRequest {
            messages: vec![Message::new(Role::User, "hello")],
            tools: vec![],
            max_tokens: 512,
        })
        .await
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|item| item.expect("no stream error"))
        .collect();

    assert_eq!(events_a[0], ChatEvent::TextDelta("A".to_string()));
    assert_eq!(
        events_a[1],
        ChatEvent::Done {
            finish_reason: FinishReason::Stop
        }
    );
    assert_eq!(events_b[0], ChatEvent::TextDelta("B".to_string()));
    assert_eq!(
        events_b[1],
        ChatEvent::Done {
            finish_reason: FinishReason::Length
        }
    );
}

fn sse_line(json: &str) -> String {
    format!("data: {json}\n\n")
}
