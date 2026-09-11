//! Regression test for a multi-byte UTF-8 character split across two raw
//! HTTP chunks (see `core/src/llm/openai.rs`'s `SseState`). Chunks here are
//! raw `Bytes`, not `String`, so the split point can land mid-character —
//! something a `String`-typed chunk can never represent.

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
use marceline_core::{ChatEvent, ChatRequest, LlmEngine, Message, OpenAiCompatibleEngine, Role};

async fn start_fake_server(chunks: Vec<Bytes>) -> String {
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

fn sse_response(chunks: Vec<Bytes>) -> Response<BoxBody<Bytes, Infallible>> {
    let stream = futures::stream::unfold(chunks.into_iter(), |mut remaining| async move {
        let chunk = remaining.next()?;
        Some((Ok::<_, Infallible>(Frame::data(chunk)), remaining))
    });

    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(BodyExt::boxed(StreamBody::new(stream)))
        .expect("build response")
}

fn test_config(base_url: String) -> LlmConfig {
    std::env::set_var("MARCELINE_TEST_UTF8_KEY", "test-key");
    LlmConfig {
        backend: "openai-compatible".to_string(),
        base_url,
        model: "test-model".to_string(),
        api_key_env: "MARCELINE_TEST_UTF8_KEY".to_string(),
        max_tokens_per_turn: 512,
        max_requests_per_session: 100,
        max_tool_iterations_per_turn: 4,
    }
}

/// A multi-byte character split mid-sequence across two byte chunks must
/// still be decoded correctly rather than replaced/corrupted.
#[tokio::test]
async fn a_multibyte_character_split_across_chunks_is_not_corrupted() {
    let text = "caf\u{e9} \u{1f600}"; // "café 😀" — 2-byte and 4-byte chars
    let line = format!(r#"data: {{"choices":[{{"delta":{{"content":"{text}"}},"finish_reason":null}}]}}"#);
    let full = format!("{line}\n\ndata: [DONE]\n\n");
    // Split mid-way through the é (a 2-byte sequence) and mid-way through
    // the 😀 (a 4-byte sequence) by cutting one byte into each.
    let cut1 = full.find('é').unwrap() + 1;
    let cut2 = full.find('😀').unwrap() + 1;
    let bytes = full.into_bytes();

    let chunks = vec![
        Bytes::copy_from_slice(&bytes[..cut1]),
        Bytes::copy_from_slice(&bytes[cut1..cut2]),
        Bytes::copy_from_slice(&bytes[cut2..]),
    ];

    let base_url = start_fake_server(chunks).await;
    let config = test_config(base_url);
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");

    let events: Vec<_> = engine
        .chat(ChatRequest {
            messages: vec![Message::new(Role::User, "hi")],
            tools: vec![],
            max_tokens: 512,
        })
        .await
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|item| item.expect("no stream error"))
        .collect();

    assert_eq!(events[0], ChatEvent::TextDelta(text.to_string()));
}
