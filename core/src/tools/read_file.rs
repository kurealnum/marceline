//! `read_file` built-in (SPEC.md §4, §5.1, EPIC 6.2): read a file path,
//! return its contents. Read-only.
//!
//! Content this returns is **untrusted** (§5.1) — it entered the prompt
//! from outside the model's control, same as `web_search` results.
//! Injection/taint handling on that content is EPIC 14's job; this tool
//! only has to fetch it and say so isn't itself dangerous, which is why it
//! stays `SafetyClass::ReadOnly` — the danger is downstream trust, not
//! this call.

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{SafetyClass, Tool, ToolResult};

/// Reads a file's contents as UTF-8 text.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads a file's contents as text. The path must exist and be readable."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path of the file to read.",
                },
            },
            "required": ["path"],
        })
    }

    async fn call(&self, args: Value, cancel: CancellationToken) -> ToolResult {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return ToolResult::Err("missing required argument: path".to_string());
        };

        let read = tokio::fs::read(path);
        let bytes = tokio::select! {
            biased;
            _ = cancel.cancelled() => return ToolResult::Err("cancelled".to_string()),
            result = read => result,
        };

        match bytes {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => ToolResult::Ok(serde_json::json!({ "content": text })),
                // Surfaced to the model rather than lossily replacing
                // invalid bytes: a tool silently mangling a binary file's
                // "contents" into garbage text is worse than telling the
                // model this isn't a text file.
                Err(_) => ToolResult::Err(format!("{path} is not valid UTF-8 text")),
            },
            Err(err) => ToolResult::Err(format!("failed to read {path}: {err}")),
        }
    }

    /// The per-call `cancel` token passed into [`Tool::call`] already
    /// races the read (§2.5.1); there is no separate in-flight handle
    /// this method would need to reach.
    fn cancel(&self) {}

    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_file(contents: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("create temp file");
        file.write_all(contents).expect("write temp file");
        file
    }

    #[tokio::test]
    async fn reads_a_text_files_contents() {
        let file = tmp_file(b"hello from disk");
        let tool = ReadFileTool;

        let result = tool
            .call(
                serde_json::json!({"path": file.path().to_str().unwrap()}),
                CancellationToken::new(),
            )
            .await;

        assert_eq!(
            result,
            ToolResult::Ok(serde_json::json!({"content": "hello from disk"}))
        );
    }

    #[tokio::test]
    async fn a_missing_path_is_a_structured_error_not_a_panic() {
        let tool = ReadFileTool;
        let result = tool
            .call(
                serde_json::json!({"path": "/definitely/does/not/exist"}),
                CancellationToken::new(),
            )
            .await;

        let ToolResult::Err(message) = result else {
            panic!("expected Err, got {result:?}");
        };
        assert!(message.contains("/definitely/does/not/exist"));
    }

    #[tokio::test]
    async fn a_missing_path_argument_is_rejected() {
        let tool = ReadFileTool;
        let result = tool.call(serde_json::json!({}), CancellationToken::new()).await;
        assert_eq!(
            result,
            ToolResult::Err("missing required argument: path".to_string())
        );
    }

    #[tokio::test]
    async fn invalid_utf8_is_reported_rather_than_mangled() {
        let file = tmp_file(&[0xFF, 0xFE, 0x00, 0x80]);
        let tool = ReadFileTool;

        let result = tool
            .call(
                serde_json::json!({"path": file.path().to_str().unwrap()}),
                CancellationToken::new(),
            )
            .await;

        assert!(matches!(result, ToolResult::Err(msg) if msg.contains("not valid UTF-8")));
    }

    #[tokio::test]
    async fn an_already_cancelled_token_short_circuits() {
        let file = tmp_file(b"irrelevant");
        let tool = ReadFileTool;
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = tool
            .call(
                serde_json::json!({"path": file.path().to_str().unwrap()}),
                cancel,
            )
            .await;

        assert_eq!(result, ToolResult::Err("cancelled".to_string()));
    }
}
