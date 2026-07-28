//! Stdio MCP transport (SPEC.md §2.3, EPIC 6.4): launch the server as a
//! child process and speak line-delimited JSON-RPC over its stdin/stdout —
//! the common case for a local MCP server, mirroring how the Python
//! STT/TTS workers are launched as subprocesses elsewhere in this codebase.
//!
//! One line, one JSON-RPC message — MCP's stdio transport is
//! newline-delimited JSON, not the `Content-Length:`-framed shape LSP
//! uses. Requests are correlated to responses by numeric `id`: a
//! background task reads the child's stdout and dispatches each response
//! to the [`oneshot`] sender a concurrent [`request`][StdioTransport::request]
//! call is waiting on.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex};

use super::transport::{McpError, McpTransport};
use super::wire::WireResponse;

/// Requests awaiting a response, keyed by the id they were sent with.
type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, McpError>>>>>;

/// A running MCP server reached over its stdin/stdout.
///
/// Killed (`kill_on_drop`) when this is dropped — an MCP server process
/// must not outlive the broker that launched it, the same lifecycle
/// discipline the worker supervisor already applies to STT/TTS workers.
pub struct StdioTransport {
    server_name: String,
    stdin: Mutex<ChildStdin>,
    next_id: AtomicU64,
    pending: PendingMap,
    // Kept alive only to hold the kill-on-drop process handle; never read.
    _child: Child,
}

impl StdioTransport {
    /// Launches `command args...` and prepares to speak JSON-RPC over its
    /// stdio.
    ///
    /// Fails only if the process could not be spawned at all (missing
    /// executable, permissions) — a server that starts but never answers
    /// fails individual [`request`][Self::request] calls instead, so a
    /// hung server does not block discovery forever without a caller
    /// choosing to wait that long.
    pub async fn spawn(server_name: String, command: &str, args: &[String]) -> Result<Self, McpError> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|err| McpError::Transport {
                server: server_name.clone(),
                message: format!("failed to launch {command}: {err}"),
            })?;

        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let pending: PendingMap =
            Arc::new(Mutex::new(HashMap::new()));

        spawn_reader(server_name.clone(), stdout, Arc::clone(&pending));

        Ok(Self {
            server_name,
            stdin: Mutex::new(stdin),
            next_id: AtomicU64::new(1),
            pending,
            _child: child,
        })
    }
}

/// Reads newline-delimited JSON-RPC responses from `stdout` and dispatches
/// each to the pending request waiting on its `id`, for as long as the
/// process keeps producing output.
///
/// On EOF (the server exited) every still-pending request is failed
/// rather than left to hang forever — a dead server must surface as an
/// error to whoever is waiting, not as a stuck future.
fn spawn_reader(
    server_name: String,
    stdout: tokio::process::ChildStdout,
    pending: PendingMap,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    dispatch_line(&server_name, &line, &pending).await;
                }
                Ok(None) => {
                    fail_all_pending(&server_name, "server closed its stdout", &pending).await;
                    return;
                }
                Err(err) => {
                    tracing::warn!(server = %server_name, %err, "failed to read mcp server stdout");
                    fail_all_pending(&server_name, "failed to read server stdout", &pending).await;
                    return;
                }
            }
        }
    });
}

/// Parses one line as a [`WireResponse`] and resolves the pending request
/// it answers, if any.
///
/// A response with no matching pending id (already timed out and given
/// up, or an id this client never sent) and a line that fails to parse at
/// all are both logged and otherwise ignored — one bad line must not take
/// down the reader task and every other in-flight call with it.
async fn dispatch_line(
    server_name: &str,
    line: &str,
    pending: &PendingMap,
) {
    let response: WireResponse = match serde_json::from_str(line) {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!(server = %server_name, %err, "mcp server sent a malformed line, ignoring");
            return;
        }
    };

    let Some(id) = response.id else {
        // A notification (no id) — v1 has nothing that acts on
        // server-initiated notifications.
        return;
    };

    let Some(sender) = pending.lock().await.remove(&id) else {
        return;
    };

    let result = match response.error {
        Some(err) => Err(McpError::Protocol {
            server: server_name.to_string(),
            message: err.message,
        }),
        None => Ok(response.result.unwrap_or(Value::Null)),
    };
    let _ = sender.send(result);
}

/// Fails every request still waiting for a response with `reason`.
async fn fail_all_pending(
    server_name: &str,
    reason: &str,
    pending: &PendingMap,
) {
    for (_, sender) in pending.lock().await.drain() {
        let _ = sender.send(Err(McpError::Transport {
            server: server_name.to_string(),
            message: reason.to_string(),
        }));
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut line = payload.to_string();
        line.push('\n');

        {
            let mut stdin = self.stdin.lock().await;
            if let Err(err) = stdin.write_all(line.as_bytes()).await {
                self.pending.lock().await.remove(&id);
                return Err(McpError::Transport {
                    server: self.server_name.clone(),
                    message: format!("failed to write request: {err}"),
                });
            }
            if let Err(err) = stdin.flush().await {
                self.pending.lock().await.remove(&id);
                return Err(McpError::Transport {
                    server: self.server_name.clone(),
                    message: format!("failed to flush request: {err}"),
                });
            }
        }

        rx.await.unwrap_or(Err(McpError::Transport {
            server: self.server_name.clone(),
            message: "response channel closed before a reply arrived".to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> String {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fake_mcp_stdio_server.py").to_string()
    }

    #[tokio::test]
    async fn a_request_reaches_the_server_and_its_response_comes_back() {
        let transport = StdioTransport::spawn(
            "fake".to_string(),
            "python3",
            &[fixture_path()],
        )
        .await
        .expect("spawn fake server");

        let result = transport
            .request("initialize", serde_json::json!({}))
            .await
            .expect("initialize succeeds");

        assert_eq!(result["serverInfo"]["name"], "fake-mcp-server");
    }

    #[tokio::test]
    async fn concurrent_requests_get_matched_to_their_own_response() {
        let transport = Arc::new(
            StdioTransport::spawn("fake".to_string(), "python3", &[fixture_path()])
                .await
                .expect("spawn fake server"),
        );

        let a = {
            let transport = Arc::clone(&transport);
            tokio::spawn(async move {
                transport
                    .request("tools/call", serde_json::json!({"name": "add", "arguments": {"a": 1, "b": 2}}))
                    .await
            })
        };
        let b = {
            let transport = Arc::clone(&transport);
            tokio::spawn(async move {
                transport
                    .request("tools/call", serde_json::json!({"name": "add", "arguments": {"a": 10, "b": 20}}))
                    .await
            })
        };

        let result_a = a.await.unwrap().expect("call a succeeds");
        let result_b = b.await.unwrap().expect("call b succeeds");

        assert_eq!(result_a["content"][0]["text"], "3");
        assert_eq!(result_b["content"][0]["text"], "30");
    }

    #[tokio::test]
    async fn a_missing_executable_fails_to_spawn_rather_than_hanging() {
        let result = StdioTransport::spawn(
            "missing".to_string(),
            "definitely-not-a-real-executable-xyz",
            &[],
        )
        .await;

        let Err(err) = result else {
            panic!("spawning a missing executable must fail");
        };
        assert!(matches!(err, McpError::Transport { .. }));
    }

    #[tokio::test]
    async fn the_server_exiting_fails_pending_requests_instead_of_hanging_forever() {
        let transport = StdioTransport::spawn(
            "fake".to_string(),
            "python3",
            &[fixture_path(), "--exit-immediately".to_string()],
        )
        .await
        .expect("spawn fake server");

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            transport.request("initialize", serde_json::json!({})),
        )
        .await
        .expect("request must not hang once the server has exited");

        assert!(result.is_err());
    }
}
