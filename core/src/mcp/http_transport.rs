//! HTTP MCP transport (SPEC.md §2.3, EPIC 6.4): one JSON-RPC request per
//! plain HTTP POST, no persistent connection.
//!
//! Deliberately the simpler half of MCP's "Streamable HTTP" transport:
//! no SSE stream back, no server-initiated push. `initialize`,
//! `tools/list`, and `tools/call` are all plain request/response, which
//! is everything v1 needs — a server that actually requires the
//! streaming half of the spec is out of scope until something needs it.

use async_trait::async_trait;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

use super::transport::{McpError, McpTransport};
use super::wire::WireResponse;

/// Speaks JSON-RPC to one HTTP endpoint.
pub struct HttpTransport {
    server_name: String,
    client: reqwest::Client,
    url: String,
    next_id: AtomicU64,
}

impl HttpTransport {
    /// Builds a client posting JSON-RPC requests to `url`.
    pub fn new(server_name: String, url: String) -> Result<Self, McpError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|err| McpError::Transport {
                server: server_name.clone(),
                message: err.to_string(),
            })?;
        Ok(Self {
            server_name,
            client,
            url,
            next_id: AtomicU64::new(1),
        })
    }
}

#[async_trait]
impl McpTransport for HttpTransport {
    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let response = self
            .client
            .post(&self.url)
            .json(&payload)
            .send()
            .await
            .map_err(|err| McpError::Transport {
                server: self.server_name.clone(),
                message: err.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(McpError::Transport {
                server: self.server_name.clone(),
                message: format!("HTTP {}", response.status()),
            });
        }

        let body: WireResponse = response.json().await.map_err(|err| McpError::Protocol {
            server: self.server_name.clone(),
            message: format!("response was not valid JSON-RPC: {err}"),
        })?;

        match body.error {
            Some(err) => Err(McpError::Protocol {
                server: self.server_name.clone(),
                message: err.message,
            }),
            None => Ok(body.result.unwrap_or(Value::Null)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::net::SocketAddr;

    use http_body_util::Full;
    use hyper::body::Bytes;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    /// Starts a fake MCP HTTP server returning `body` for every POST.
    async fn fake_server(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else { return };
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |_req: Request<hyper::body::Incoming>| async move {
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
    async fn a_successful_response_returns_its_result() {
        let url = fake_server(r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#).await;
        let transport = HttpTransport::new("fake".to_string(), url).unwrap();

        let result = transport
            .request("tools/list", serde_json::json!({}))
            .await
            .expect("request succeeds");

        assert_eq!(result, serde_json::json!({"tools": []}));
    }

    #[tokio::test]
    async fn a_jsonrpc_error_object_is_reported_as_protocol_error() {
        let url = fake_server(r#"{"jsonrpc":"2.0","id":1,"error":{"message":"boom"}}"#).await;
        let transport = HttpTransport::new("fake".to_string(), url).unwrap();

        let err = transport
            .request("tools/call", serde_json::json!({}))
            .await
            .expect_err("expected a protocol error");

        assert!(matches!(err, McpError::Protocol { .. }));
        assert!(err.to_string().contains("boom"));
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_is_a_transport_error() {
        let transport = HttpTransport::new("fake".to_string(), "http://127.0.0.1:1".to_string()).unwrap();

        let err = transport
            .request("tools/list", serde_json::json!({}))
            .await
            .expect_err("connecting to nothing must fail");

        assert!(matches!(err, McpError::Transport { .. }));
    }
}
