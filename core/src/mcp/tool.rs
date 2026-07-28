//! Wraps one discovered MCP tool as a [`Tool`] impl (SPEC.md §2.5.1,
//! §4, EPIC 6.4) — the seam that lets an MCP tool sit in
//! [`ToolBroker`][crate::tools::ToolBroker]'s registry indistinguishably
//! from a built-in.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::client::McpClient;
use crate::tools::{SafetyClass, Tool, ToolResult};

/// One MCP server's tool, registered under its namespaced name
/// (`serverName.toolName`, §4).
pub struct McpTool {
    /// The namespaced name this tool is registered under.
    namespaced_name: String,
    /// The name the *server* knows this tool by — what actually goes on
    /// the wire in `tools/call`.
    inner_name: String,
    description: String,
    parameters: Value,
    client: Arc<McpClient>,
}

impl McpTool {
    /// Builds an `McpTool` for one entry of `client`'s `tools/list`.
    pub fn new(
        namespaced_name: String,
        inner_name: String,
        description: String,
        parameters: Value,
        client: Arc<McpClient>,
    ) -> Self {
        Self {
            namespaced_name,
            inner_name,
            description,
            parameters,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.namespaced_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn call(&self, args: Value, cancel: CancellationToken) -> ToolResult {
        let call = self.client.call_tool(&self.inner_name, args);
        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => return ToolResult::Err("cancelled".to_string()),
            result = call => result,
        };

        match outcome {
            Ok(outcome) if outcome.is_error => ToolResult::Err(render_content(&outcome.content)),
            Ok(outcome) => ToolResult::Ok(outcome.content),
            Err(err) => ToolResult::Err(err.to_string()),
        }
    }

    /// The per-call `cancel` token passed into [`Tool::call`] already
    /// races the request (§2.5.1); protocol-level cancellation
    /// (`notifications/cancelled`) is EPIC 6.6's job, not this one's.
    fn cancel(&self) {}

    /// v1 only ever *uses* MCP tools read-only; the confirmation-gated
    /// default for third-party MCP tools (§14.2) is a later, separate
    /// enforcement pass — this class is what that pass will read.
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
}

/// Renders an MCP call's `content` as text for [`ToolResult::Err`].
///
/// MCP content is conventionally an array of `{type: "text", text}`
/// items; this joins every text part it finds. A server that returns
/// something else entirely still gets *something* legible back — the raw
/// JSON — rather than this silently producing an empty message.
fn render_content(content: &Value) -> String {
    if let Some(items) = content.as_array() {
        let joined: String = items
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if !joined.is_empty() {
            return joined;
        }
    }
    content.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_content_joins_text_items() {
        let content = serde_json::json!([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]);
        assert_eq!(render_content(&content), "first\nsecond");
    }

    #[test]
    fn render_content_falls_back_to_raw_json_for_an_unexpected_shape() {
        let content = serde_json::json!({"unexpected": true});
        assert_eq!(render_content(&content), r#"{"unexpected":true}"#);
    }
}
