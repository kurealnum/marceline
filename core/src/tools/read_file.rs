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
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

use super::sandbox::Sandbox;
use super::{SafetyClass, Tool, ToolResult};

/// Largest file this tool will return in full. A single read past this
/// cap is truncated rather than loaded whole: without a limit a large
/// file becomes gigabytes of allocation that then gets cloned on every
/// subsequent tool iteration this turn (`thinking.rs`'s message list).
pub const MAX_READ_BYTES: usize = 256 * 1024;

/// Reads a file's contents as UTF-8 text, confined to a configured root
/// (see [`Sandbox`]) — "read-only" describes the effect on the
/// filesystem, not the effect on the user, and this tool is auto-run by
/// policy (§4), so what it can reach has to be bounded explicitly rather
/// than left as "anything the process can open".
pub struct ReadFileTool {
    sandbox: Sandbox,
}

impl ReadFileTool {
    /// Confines every `read_file` call to `root`.
    pub fn new(sandbox: Sandbox) -> Self {
        Self { sandbox }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads a file's contents as text. The path must be relative to, and \
         stay inside, the configured sandbox root."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read, relative to the sandbox root.",
                },
            },
            "required": ["path"],
        })
    }

    async fn call(&self, args: Value, cancel: CancellationToken) -> ToolResult {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return ToolResult::Err("missing required argument: path".to_string());
        };

        let resolved = match self.sandbox.resolve(path) {
            Ok(resolved) => resolved,
            Err(err) => return ToolResult::Err(err),
        };

        let read = read_capped(resolved, MAX_READ_BYTES);
        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => return ToolResult::Err("cancelled".to_string()),
            result = read => result,
        };

        match outcome {
            // A truncated read may have cut a multi-byte character in
            // half at the cap boundary, which isn't the "not valid UTF-8
            // text" case below — that's about the *file*, not about where
            // this tool happened to stop reading it — so truncation uses
            // a lossy decode instead of the strict check.
            Ok((bytes, true)) => {
                let text = String::from_utf8_lossy(&bytes).into_owned();
                ToolResult::Ok(serde_json::json!({
                    "content": text,
                    "truncated": true,
                    "truncated_at_bytes": MAX_READ_BYTES,
                }))
            }
            Ok((bytes, false)) => match String::from_utf8(bytes) {
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

/// Reads at most `max_bytes` of `path`, returning whether the file had
/// more left. Reads one byte past the cap to distinguish "exactly
/// `max_bytes` long" from "longer than the cap" without needing a second
/// `stat` call.
async fn read_capped(path: std::path::PathBuf, max_bytes: usize) -> std::io::Result<(Vec<u8>, bool)> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; max_bytes + 1];
    let mut total = 0;
    loop {
        let n = file.read(&mut buf[total..]).await?;
        if n == 0 {
            break;
        }
        total += n;
        if total > max_bytes {
            break;
        }
    }
    let truncated = total > max_bytes;
    buf.truncate(total.min(max_bytes));
    Ok((buf, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_in(dir: &std::path::Path) -> ReadFileTool {
        ReadFileTool::new(Sandbox::new(dir).expect("root exists"))
    }

    #[tokio::test]
    async fn reads_a_text_files_contents() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello from disk").unwrap();
        let tool = tool_in(dir.path());

        let result = tool
            .call(serde_json::json!({"path": "a.txt"}), CancellationToken::new())
            .await;

        assert_eq!(
            result,
            ToolResult::Ok(serde_json::json!({"content": "hello from disk"}))
        );
    }

    #[tokio::test]
    async fn a_missing_path_is_a_structured_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        let result = tool
            .call(serde_json::json!({"path": "does-not-exist.txt"}), CancellationToken::new())
            .await;

        let ToolResult::Err(message) = result else {
            panic!("expected Err, got {result:?}");
        };
        assert!(message.contains("does-not-exist.txt"));
    }

    #[tokio::test]
    async fn a_missing_path_argument_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool_in(dir.path());
        let result = tool.call(serde_json::json!({}), CancellationToken::new()).await;
        assert_eq!(
            result,
            ToolResult::Err("missing required argument: path".to_string())
        );
    }

    #[tokio::test]
    async fn invalid_utf8_is_reported_rather_than_mangled() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("bin"), [0xFF, 0xFE, 0x00, 0x80]).unwrap();
        let tool = tool_in(dir.path());

        let result = tool
            .call(serde_json::json!({"path": "bin"}), CancellationToken::new())
            .await;

        assert!(matches!(result, ToolResult::Err(msg) if msg.contains("not valid UTF-8")));
    }

    #[tokio::test]
    async fn a_path_outside_the_sandbox_root_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(outside.path(), b"secret").unwrap();
        let tool = tool_in(dir.path());

        let escape = format!("../{}", outside.path().file_name().unwrap().to_str().unwrap());
        let result = tool
            .call(serde_json::json!({"path": escape}), CancellationToken::new())
            .await;

        assert!(matches!(result, ToolResult::Err(_)));
    }

    #[tokio::test]
    async fn a_file_larger_than_the_cap_is_truncated_with_a_note() {
        let dir = tempfile::tempdir().unwrap();
        let contents = vec![b'a'; MAX_READ_BYTES + 100];
        std::fs::write(dir.path().join("big.txt"), &contents).unwrap();
        let tool = tool_in(dir.path());

        let result = tool
            .call(serde_json::json!({"path": "big.txt"}), CancellationToken::new())
            .await;

        let ToolResult::Ok(payload) = result else {
            panic!("expected Ok, got {result:?}");
        };
        assert_eq!(payload["content"].as_str().unwrap().len(), MAX_READ_BYTES);
        assert_eq!(payload["truncated"], serde_json::Value::Bool(true));
    }

    #[tokio::test]
    async fn an_already_cancelled_token_short_circuits() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"irrelevant").unwrap();
        let tool = tool_in(dir.path());
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = tool
            .call(serde_json::json!({"path": "a.txt"}), cancel)
            .await;

        assert_eq!(result, ToolResult::Err("cancelled".to_string()));
    }
}
