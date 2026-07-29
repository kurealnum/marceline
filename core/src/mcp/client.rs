//! An MCP client bound to one server (SPEC.md §2.3, §4, EPIC 6.4): the
//! handshake plus the two calls v1 needs, layered over whichever
//! [`McpTransport`] the server's config selected.

use serde_json::Value;

use super::transport::{McpError, McpTransport};

/// Protocol version this client claims during `initialize`.
///
/// Pinned to one known-good date rather than "latest": an MCP server that
/// only understands an older/newer protocol should fail the handshake
/// visibly rather than this client silently assuming compatibility it
/// hasn't verified.
const PROTOCOL_VERSION: &str = "2024-11-05";

/// One tool as reported by a server's `tools/list`.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolInfo {
    /// The tool's name, as the server knows it (not yet namespaced).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's arguments, as the server declared it.
    pub parameters: Value,
}

/// The result of one `tools/call`.
#[derive(Debug, Clone, PartialEq)]
pub struct McpCallOutcome {
    /// True when the server reports this call as a tool-level failure
    /// (distinct from a transport/protocol failure, which surfaces as
    /// `Err` instead).
    pub is_error: bool,
    /// The call's `content`, passed through as-is — typically an array of
    /// `{type: "text", text: "..."}` items per the MCP spec, but this
    /// client does not assume that shape; rendering it is the caller's
    /// job ([`super::tool::McpTool`]).
    pub content: Value,
}

/// A client bound to one configured MCP server.
pub struct McpClient {
    server_name: String,
    transport: Box<dyn McpTransport>,
}

impl McpClient {
    /// Wraps `transport` for `server_name`.
    pub fn new(server_name: String, transport: Box<dyn McpTransport>) -> Self {
        Self {
            server_name,
            transport,
        }
    }

    /// The server name this client is bound to (the namespace prefix for
    /// its tools).
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Performs the MCP `initialize` handshake.
    ///
    /// The real spec also has clients send a follow-up
    /// `notifications/initialized`; v1's transport seam (§2.5.1's note:
    /// deeper cancel/notification wiring is EPIC 6.6's job) does not model
    /// fire-and-forget notifications, and every server tested against
    /// tolerates its absence, so it is skipped rather than adding
    /// notification support for a step nothing here depends on.
    pub async fn initialize(&self) -> Result<(), McpError> {
        self.transport
            .request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "marceline",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                }),
            )
            .await?;
        Ok(())
    }

    /// Lists the tools this server offers.
    ///
    /// A malformed entry (missing/empty `name`) is skipped rather than
    /// failing the whole list — one bad entry in an otherwise-useful
    /// server's catalog should not cost every other tool it offers.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>, McpError> {
        let result = self.transport.request("tools/list", serde_json::json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        Ok(tools
            .into_iter()
            .filter_map(|tool| {
                let name = tool.get("name")?.as_str()?.to_string();
                if name.is_empty() {
                    return None;
                }
                let description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let parameters = tool
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
                Some(McpToolInfo {
                    name,
                    description,
                    parameters,
                })
            })
            .collect())
    }

    /// Calls `name` with `arguments`.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<McpCallOutcome, McpError> {
        let result = self
            .transport
            .request("tools/call", serde_json::json!({"name": name, "arguments": arguments}))
            .await?;

        Ok(McpCallOutcome {
            is_error: result.get("isError").and_then(Value::as_bool).unwrap_or(false),
            content: result.get("content").cloned().unwrap_or(Value::Null),
        })
    }
}
