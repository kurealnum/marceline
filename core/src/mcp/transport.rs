//! The transport seam an [`McpClient`][super::client::McpClient] talks
//! through (SPEC.md §2.3, EPIC 6.4): "stdio or HTTP transport per server."
//!
//! One trait, two implementations
//! ([`stdio`][super::stdio_transport]/[`http`][super::http_transport]), so
//! `McpClient` never branches on which kind of server it is talking to —
//! the same shape every plugin contract in this codebase uses.

use async_trait::async_trait;
use serde_json::Value;

/// A failure talking to an MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The server process/endpoint could not be reached at all.
    #[error("failed to reach mcp server {server}: {message}")]
    Transport {
        /// Configured server name.
        server: String,
        /// What went wrong.
        message: String,
    },
    /// The server was reached but returned a JSON-RPC error object, or a
    /// response that did not match what was requested.
    #[error("mcp server {server} returned an error: {message}")]
    Protocol {
        /// Configured server name.
        server: String,
        /// The server's error message (or a description of the
        /// malformed response).
        message: String,
    },
}

/// One JSON-RPC round trip to an MCP server: `method` + `params` in, the
/// `result` value out (or an [`McpError`]).
///
/// Deliberately minimal — just enough for `initialize`, `tools/list`, and
/// `tools/call` (everything v1 needs). Notifications, cancellation
/// (`notifications/cancelled`), and server-initiated requests are not
/// modeled here; a per-call [`tokio_util::sync::CancellationToken`] races
/// the call instead (matching the built-in tools, §2.5.1) — protocol-level
/// cancel wiring is EPIC 6.6's job, not this one's.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Sends a JSON-RPC request and returns its `result` value.
    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError>;
}
