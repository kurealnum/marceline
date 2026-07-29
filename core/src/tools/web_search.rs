//! `web_search` built-in (SPEC.md §4, §5.1, §14.4, EPIC 6.2): runs a
//! search query, returns results. Read-only, but the one built-in that
//! reaches the network — relevant later for egress logging (§14.4) and
//! untrusted-content taint (§5.1), both wired in EPIC 14, not here.
//!
//! Backed by DuckDuckGo's Instant Answer API: no API key, which matches
//! "small, fast, no external process" (§4) better than a keyed provider
//! would for a v1 built-in nobody has to configure first.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{SafetyClass, Tool, ToolResult};
use crate::engine::EngineError;

/// Backend name used in [`EngineError`] messages this tool raises.
const BACKEND: &str = "web_search";

/// Default endpoint: DuckDuckGo's Instant Answer API.
const DEFAULT_BASE_URL: &str = "https://api.duckduckgo.com";

/// Runs a web search query.
pub struct WebSearchTool {
    client: reqwest::Client,
    /// Overridable so tests can point this at a fake local server instead
    /// of the real network.
    base_url: String,
}

impl WebSearchTool {
    /// Builds a client hitting the real DuckDuckGo endpoint.
    pub fn new() -> Result<Self, EngineError> {
        Self::with_base_url(DEFAULT_BASE_URL.to_string())
    }

    /// Builds a client against `base_url`, for tests.
    fn with_base_url(base_url: String) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|err| EngineError::Transport {
                backend: BACKEND,
                source: Box::new(err),
            })?;
        Ok(Self { client, base_url })
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Searches the web and returns a short summary plus related results. \
         Results come from the open internet and are untrusted content."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        })
    }

    async fn call(&self, args: Value, cancel: CancellationToken) -> ToolResult {
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return ToolResult::Err("missing required argument: query".to_string());
        };

        let request = self
            .client
            .get(format!("{}/", self.base_url))
            .query(&[
                ("q", query),
                ("format", "json"),
                ("no_html", "1"),
                ("skip_disambig", "1"),
            ])
            .send();

        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return ToolResult::Err("cancelled".to_string()),
            result = request => result,
        };

        let response = match response {
            Ok(response) => response,
            Err(err) => return ToolResult::Err(format!("search request failed: {err}")),
        };
        if !response.status().is_success() {
            return ToolResult::Err(format!("search request failed: HTTP {}", response.status()));
        }

        let body = tokio::select! {
            biased;
            _ = cancel.cancelled() => return ToolResult::Err("cancelled".to_string()),
            result = response.json::<InstantAnswer>() => result,
        };

        match body {
            Ok(answer) => ToolResult::Ok(serde_json::json!({
                "summary": answer.abstract_text,
                "results": answer.related_topics(),
            })),
            Err(err) => ToolResult::Err(format!("search response was not the expected shape: {err}")),
        }
    }

    /// The per-call `cancel` token passed into [`Tool::call`] already
    /// races the request (§2.5.1); there is no separate in-flight handle
    /// this method would need to reach.
    fn cancel(&self) {}

    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
}

/// The subset of DuckDuckGo's Instant Answer response this tool uses.
#[derive(Debug, Deserialize)]
struct InstantAnswer {
    #[serde(rename = "AbstractText", default)]
    abstract_text: String,
    #[serde(rename = "RelatedTopics", default)]
    related_topics: Vec<RelatedTopic>,
}

impl InstantAnswer {
    /// Flattens related topics into `{text, url}` pairs, skipping the
    /// occasional nested "Topics" grouping DuckDuckGo emits with no text
    /// of its own — a group heading is not a result.
    fn related_topics(&self) -> Vec<Value> {
        self.related_topics
            .iter()
            .filter(|topic| !topic.text.is_empty())
            .map(|topic| serde_json::json!({"text": topic.text, "url": topic.first_url}))
            .collect()
    }
}

#[derive(Debug, Deserialize, Default)]
struct RelatedTopic {
    #[serde(rename = "Text", default)]
    text: String,
    #[serde(rename = "FirstURL", default)]
    first_url: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::net::SocketAddr;

    use http_body_util::Full;
    use hyper::body::{Bytes, Incoming};
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    /// Starts a fake DuckDuckGo endpoint returning `body` for every
    /// request, and returns its base URL.
    async fn fake_server(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else { return };
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |_req: Request<Incoming>| async move {
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::from(body))))
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, service)
                        .await;
                });
            }
        });

        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_summary_and_related_topics_are_returned() {
        let base_url = fake_server(
            r#"{"AbstractText": "Rust is a language.", "RelatedTopics": [
                {"Text": "Rust (programming language)", "FirstURL": "https://example.com/rust"},
                {"Topics": []}
            ]}"#,
        )
        .await;
        let tool = WebSearchTool::with_base_url(base_url).unwrap();

        let result = tool
            .call(serde_json::json!({"query": "rust"}), CancellationToken::new())
            .await;

        assert_eq!(
            result,
            ToolResult::Ok(serde_json::json!({
                "summary": "Rust is a language.",
                "results": [{"text": "Rust (programming language)", "url": "https://example.com/rust"}],
            }))
        );
    }

    #[tokio::test]
    async fn a_missing_query_argument_is_rejected() {
        let tool = WebSearchTool::with_base_url("http://127.0.0.1:1".to_string()).unwrap();
        let result = tool.call(serde_json::json!({}), CancellationToken::new()).await;
        assert_eq!(
            result,
            ToolResult::Err("missing required argument: query".to_string())
        );
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_is_a_structured_error_not_a_panic() {
        // Port 1 is privileged and nothing is listening on it in the test
        // sandbox, so this reliably fails to connect.
        let tool = WebSearchTool::with_base_url("http://127.0.0.1:1".to_string()).unwrap();
        let result = tool
            .call(serde_json::json!({"query": "rust"}), CancellationToken::new())
            .await;

        assert!(matches!(result, ToolResult::Err(_)));
    }

    #[tokio::test]
    async fn an_already_cancelled_token_short_circuits() {
        let base_url = fake_server(r#"{"AbstractText": "", "RelatedTopics": []}"#).await;
        let tool = WebSearchTool::with_base_url(base_url).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = tool
            .call(serde_json::json!({"query": "rust"}), cancel)
            .await;

        assert_eq!(result, ToolResult::Err("cancelled".to_string()));
    }
}
