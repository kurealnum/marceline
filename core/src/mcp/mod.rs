//! MCP client: discover and call configured MCP servers' tools (SPEC.md
//! §2.3, §4, EPIC 6.4).
//!
//! The Rust core is an MCP *client*, never a server: each entry in
//! `[[mcp]]` config ([`crate::config::McpServerConfig`]) names a server to
//! connect to (stdio or HTTP, §2.3), whose tools are discovered at
//! startup and registered into the [`crate::tools::ToolBroker`] namespaced
//! `serverName.toolName` (§4) — indistinguishable from a built-in once
//! they are in the broker, which is the whole point of both going through
//! the same [`crate::tools::Tool`] trait.

pub mod client;
pub mod discover;
pub mod http_transport;
pub mod stdio_transport;
pub mod tool;
pub mod transport;
mod wire;

pub use client::{McpCallOutcome, McpClient, McpToolInfo};
pub use discover::register_mcp_tools;
pub use http_transport::HttpTransport;
pub use stdio_transport::StdioTransport;
pub use tool::McpTool;
pub use transport::{McpError, McpTransport};
