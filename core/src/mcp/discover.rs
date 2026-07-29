//! Connects every configured MCP server and registers its tools into the
//! broker, namespaced (SPEC.md §2.3, §4, EPIC 6.4).
//!
//! A server that fails to start, fails its handshake, or returns nothing
//! useful is skipped with a logged warning — built-ins (and every other
//! configured server) must still work regardless of one server being
//! down.

use std::sync::Arc;

use super::client::McpClient;
use super::http_transport::HttpTransport;
use super::stdio_transport::StdioTransport;
use super::tool::McpTool;
use super::transport::{McpError, McpTransport};
use crate::config::{McpServerConfig, McpTransportConfig};
use crate::tools::ToolBroker;

/// Connects every server in `servers`, discovers its tools, and registers
/// each into `broker` under `serverName.toolName`.
///
/// Returns the names of servers that were skipped, so a caller (or a
/// test) can observe what didn't come up without depending on captured
/// log output — the actual reason is still logged via `tracing::warn` at
/// the point of failure.
pub async fn register_mcp_tools(broker: &mut ToolBroker, servers: &[McpServerConfig]) -> Vec<String> {
    let mut skipped = Vec::new();
    for server in servers {
        if let Err(err) = register_one(broker, server).await {
            tracing::warn!(server = %server.name, %err, "skipping mcp server");
            skipped.push(server.name.clone());
        }
    }
    skipped
}

/// Connects, initializes, discovers, and registers tools for one server.
async fn register_one(broker: &mut ToolBroker, server: &McpServerConfig) -> Result<(), McpError> {
    let transport: Box<dyn McpTransport> = match &server.transport {
        McpTransportConfig::Stdio { command, args } => {
            Box::new(StdioTransport::spawn(server.name.clone(), command, args).await?)
        }
        McpTransportConfig::Http { url } => Box::new(HttpTransport::new(server.name.clone(), url.clone())?),
    };

    let client = Arc::new(McpClient::new(server.name.clone(), transport));
    client.initialize().await?;
    let tools = client.list_tools().await?;

    for tool in tools {
        let namespaced = format!("{}.{}", server.name, tool.name);
        let mcp_tool = Arc::new(McpTool::new(
            namespaced.clone(),
            tool.name,
            tool.description,
            tool.parameters,
            Arc::clone(&client),
        ));
        if let Err(err) = broker.register(mcp_tool) {
            // A name collision (two servers whose namespace prefixes
            // still collided, or a built-in reusing an MCP server's
            // name) is this one tool's problem, not the whole server's —
            // the rest of the server's tools still register.
            tracing::warn!(tool = %namespaced, %err, "skipping duplicate mcp tool name");
        }
    }

    Ok(())
}
