//! `list_dir` built-in (SPEC.md §4, EPIC 6.2): lists a directory's
//! entries. Read-only.

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{SafetyClass, Tool, ToolResult};

/// Lists the entries of a directory.
pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "Lists the names and types (file/directory) of a directory's entries."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path of the directory to list.",
                },
            },
            "required": ["path"],
        })
    }

    async fn call(&self, args: Value, cancel: CancellationToken) -> ToolResult {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return ToolResult::Err("missing required argument: path".to_string());
        };

        tokio::select! {
            biased;
            _ = cancel.cancelled() => ToolResult::Err("cancelled".to_string()),
            result = list(path) => result,
        }
    }

    /// The per-call `cancel` token passed into [`Tool::call`] already
    /// races the listing (§2.5.1); there is no separate in-flight handle
    /// this method would need to reach.
    fn cancel(&self) {}

    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
}

/// Reads every entry of `path` into a JSON array of `{name, is_dir}`.
async fn list(path: &str) -> ToolResult {
    let mut read_dir = match tokio::fs::read_dir(path).await {
        Ok(read_dir) => read_dir,
        Err(err) => return ToolResult::Err(format!("failed to list {path}: {err}")),
    };

    let mut entries = Vec::new();
    loop {
        match read_dir.next_entry().await {
            Ok(Some(entry)) => {
                let is_dir = entry.file_type().await.map(|ft| ft.is_dir()).unwrap_or(false);
                entries.push(serde_json::json!({
                    "name": entry.file_name().to_string_lossy(),
                    "is_dir": is_dir,
                }));
            }
            Ok(None) => break,
            Err(err) => return ToolResult::Err(format!("failed to read an entry of {path}: {err}")),
        }
    }

    ToolResult::Ok(serde_json::json!({ "entries": entries }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lists_files_and_subdirectories() {
        let dir = tempfile::tempdir().expect("create temp dir");
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let tool = ListDirTool;
        let result = tool
            .call(
                serde_json::json!({"path": dir.path().to_str().unwrap()}),
                CancellationToken::new(),
            )
            .await;

        let ToolResult::Ok(payload) = result else {
            panic!("expected Ok, got {result:?}");
        };
        let mut names: Vec<(String, bool)> = payload["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e["name"].as_str().unwrap().to_string(),
                    e["is_dir"].as_bool().unwrap(),
                )
            })
            .collect();
        names.sort();

        assert_eq!(
            names,
            vec![("a.txt".to_string(), false), ("sub".to_string(), true)]
        );
    }

    #[tokio::test]
    async fn an_empty_directory_reports_no_entries() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let tool = ListDirTool;

        let result = tool
            .call(
                serde_json::json!({"path": dir.path().to_str().unwrap()}),
                CancellationToken::new(),
            )
            .await;

        assert_eq!(result, ToolResult::Ok(serde_json::json!({"entries": []})));
    }

    #[tokio::test]
    async fn a_missing_path_is_a_structured_error_not_a_panic() {
        let tool = ListDirTool;
        let result = tool
            .call(
                serde_json::json!({"path": "/definitely/does/not/exist"}),
                CancellationToken::new(),
            )
            .await;

        assert!(matches!(result, ToolResult::Err(_)));
    }

    #[tokio::test]
    async fn a_missing_path_argument_is_rejected() {
        let tool = ListDirTool;
        let result = tool.call(serde_json::json!({}), CancellationToken::new()).await;
        assert_eq!(
            result,
            ToolResult::Err("missing required argument: path".to_string())
        );
    }

    #[tokio::test]
    async fn an_already_cancelled_token_short_circuits() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let tool = ListDirTool;
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = tool
            .call(
                serde_json::json!({"path": dir.path().to_str().unwrap()}),
                cancel,
            )
            .await;

        assert_eq!(result, ToolResult::Err("cancelled".to_string()));
    }
}
