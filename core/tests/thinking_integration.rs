//! Integration tests for the THINKING tool-call loop (EPIC 6.3).
//!
//! Drives [`marceline_core::think`] against a real
//! [`OpenAiCompatibleEngine`] and a local, hand-rolled OpenAI-compatible SSE
//! server that serves a scripted sequence of responses (one per request) —
//! real bytes over a real socket, same as `llm_integration.rs`, but serving
//! a *sequence* so a tool-call round followed by a final-answer round can
//! be exercised for real, including verifying what the second request
//! actually contained (the tool result, linked by id).

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use http_body_util::{combinators::BoxBody, BodyExt, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use marceline_core::config::LlmConfig;
use marceline_core::{
    resolve_max_iterations, think, FinishReason, Message, OpenAiCompatibleEngine, Role,
    SafetyClass, Tool, ToolBroker, ToolResult, MAX_TOOL_ITERS_ENV,
};

/// One canned SSE response: a sequence of `data: ...\n\n` lines.
type ScriptedResponse = Vec<String>;

/// Starts a fake OpenAI-compatible server serving `responses[0]` to the
/// first request it receives, `responses[1]` to the second, and so on
/// (clamped to the last entry if more requests arrive than scripted).
///
/// Every request body is captured, in order, into the returned `Vec` —
/// this is what lets a test assert the *second* request actually carried
/// the tool result linked to the first response's tool call, not just
/// that some second request happened.
async fn start_scripted_server(
    responses: Vec<ScriptedResponse>,
) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let responses = Arc::new(responses);
    let call_index = Arc::new(AtomicUsize::new(0));
    let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let bodies_for_server = Arc::clone(&bodies);
    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(accepted) => accepted,
                Err(_) => return,
            };
            let io = TokioIo::new(stream);
            let responses = Arc::clone(&responses);
            let call_index = Arc::clone(&call_index);
            let bodies = Arc::clone(&bodies_for_server);

            tokio::spawn(async move {
                let service = service_fn(move |req: Request<Incoming>| {
                    let responses = Arc::clone(&responses);
                    let call_index = Arc::clone(&call_index);
                    let bodies = Arc::clone(&bodies);
                    async move {
                        let body = req.into_body().collect().await.map(|b| b.to_bytes());
                        if let Ok(bytes) = body {
                            bodies
                                .lock()
                                .unwrap()
                                .push(String::from_utf8_lossy(&bytes).to_string());
                        }

                        let index = call_index.fetch_add(1, Ordering::SeqCst);
                        let script = &responses[index.min(responses.len() - 1)];
                        Ok::<_, Infallible>(sse_response(script.clone()))
                    }
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, service)
                    .await;
            });
        }
    });

    (format!("http://{addr}/v1"), bodies)
}

/// Builds a streaming `text/event-stream` response out of `lines`.
fn sse_response(lines: ScriptedResponse) -> Response<BoxBody<Bytes, Infallible>> {
    let stream = futures::stream::unfold(lines.into_iter(), |mut remaining| async move {
        let line = remaining.next()?;
        Some((Ok::<_, Infallible>(Frame::data(Bytes::from(line))), remaining))
    });

    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(BodyExt::boxed(StreamBody::new(stream)))
        .expect("build response")
}

fn sse_line(json: &str) -> String {
    format!("data: {json}\n\n")
}

fn test_config(base_url: String) -> LlmConfig {
    std::env::set_var("MARCELINE_TEST_THINKING_KEY", "test-key");
    LlmConfig {
        backend: "openai-compatible".to_string(),
        base_url,
        model: "test-model".to_string(),
        api_key_env: "MARCELINE_TEST_THINKING_KEY".to_string(),
        max_tokens_per_turn: 512,
        max_requests_per_session: 100,
        max_tool_iterations_per_turn: 8,
    }
}

/// A stub tool recording every call it receives, for asserting dispatch
/// happened (or didn't) without depending on a real built-in's output.
struct RecordingTool {
    calls: Arc<Mutex<Vec<serde_json::Value>>>,
}

#[async_trait]
impl Tool for RecordingTool {
    fn name(&self) -> &str {
        "get_time"
    }
    fn description(&self) -> &str {
        "stub for tests"
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    async fn call(&self, args: serde_json::Value, _cancel: CancellationToken) -> ToolResult {
        self.calls.lock().unwrap().push(args);
        ToolResult::Ok(serde_json::json!({"time": "12:00 PM"}))
    }
    fn cancel(&self) {}
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
}

/// The tool-call round: the model asks for `get_time`.
fn tool_call_round() -> ScriptedResponse {
    vec![
        sse_line(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_time","arguments":"{}"}}]},"finish_reason":null}]}"#,
        ),
        sse_line(r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
    ]
}

/// A final-answer round with no tool calls.
fn final_round(text: &str) -> ScriptedResponse {
    vec![
        sse_line(&format!(
            r#"{{"choices":[{{"delta":{{"content":"{text}"}},"finish_reason":null}}]}}"#
        )),
        sse_line(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
    ]
}

#[tokio::test]
async fn a_tool_call_is_run_and_its_result_feeds_a_final_answer() {
    let (base_url, bodies) = start_scripted_server(vec![
        tool_call_round(),
        final_round("It is 12:00 PM."),
    ])
    .await;
    let config = test_config(base_url);
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");

    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut broker = ToolBroker::new();
    broker
        .register(Arc::new(RecordingTool {
            calls: Arc::clone(&calls),
        }))
        .unwrap();

    let messages = vec![Message::new(Role::User, "what time is it")];
    let mut seen_text = String::new();

    let (outcome, _messages) = think(
        &engine,
        &broker,
        messages,
        broker.catalog(),
        512,
        8,
        CancellationToken::new(),
        |delta| seen_text.push_str(delta),
    )
    .await
    .expect("thinking loop succeeds");

    assert_eq!(outcome.text, "It is 12:00 PM.");
    assert_eq!(outcome.finish_reason, FinishReason::Stop);
    assert!(!outcome.iteration_cap_hit);
    assert_eq!(seen_text, "It is 12:00 PM.", "on_text must see the final text too");
    assert_eq!(calls.lock().unwrap().len(), 1, "the tool must actually have run");

    // The second request must carry the tool result, linked by id — not
    // just "some second request happened".
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    assert!(bodies[1].contains(r#""role":"tool""#));
    assert!(bodies[1].contains(r#""tool_call_id":"call_1""#));
    assert!(bodies[1].contains(r#"time\":\"12:00 PM"#));
}

#[tokio::test]
async fn hitting_the_iteration_cap_forces_a_final_answer_without_running_the_tool() {
    let (base_url, bodies) = start_scripted_server(vec![
        tool_call_round(),
        final_round("Here's what I have."),
    ])
    .await;
    let config = test_config(base_url);
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");

    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut broker = ToolBroker::new();
    broker
        .register(Arc::new(RecordingTool {
            calls: Arc::clone(&calls),
        }))
        .unwrap();

    let messages = vec![Message::new(Role::User, "what time is it")];

    // max_iterations = 0: the very first tool-call round already exceeds
    // the budget, so it must be told "budget exhausted" instead of run.
    let (outcome, _messages) = think(
        &engine,
        &broker,
        messages,
        broker.catalog(),
        512,
        0,
        CancellationToken::new(),
        |_| {},
    )
    .await
    .expect("thinking loop succeeds");

    assert_eq!(outcome.text, "Here's what I have.");
    assert!(outcome.iteration_cap_hit);
    assert!(
        calls.lock().unwrap().is_empty(),
        "the tool must not run once the budget is already spent"
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "one tool-call round, one forced final round");
    assert!(bodies[1].contains("tool budget exhausted"));
    // The forced final round must not offer tools, or the model could
    // just ask for another one.
    assert!(!bodies[1].contains(r#""tools":[{"#));
}

#[tokio::test]
async fn a_final_answer_with_no_tool_calls_is_a_single_round() {
    let (base_url, bodies) = start_scripted_server(vec![final_round("Just chatting.")]).await;
    let config = test_config(base_url);
    let engine = OpenAiCompatibleEngine::new(&config, CancellationToken::new()).expect("engine");
    let broker = ToolBroker::new();

    let messages = vec![Message::new(Role::User, "hi")];
    let (outcome, _messages) = think(
        &engine,
        &broker,
        messages,
        broker.catalog(),
        512,
        8,
        CancellationToken::new(),
        |_| {},
    )
    .await
    .expect("thinking loop succeeds");

    assert_eq!(outcome.text, "Just chatting.");
    assert!(!outcome.iteration_cap_hit);
    assert_eq!(bodies.lock().unwrap().len(), 1);
}

#[test]
fn env_override_takes_precedence_over_configured_cap() {
    std::env::set_var(MAX_TOOL_ITERS_ENV, "2");
    assert_eq!(resolve_max_iterations(8), 2);
    std::env::remove_var(MAX_TOOL_ITERS_ENV);
}
