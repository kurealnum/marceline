//! `get_time` built-in (SPEC.md §4, EPIC 6.2): current local time/date,
//! no args.
//!
//! The epic's demo tool ("what time is it" → a `get_time` call → the
//! answer spoken back) — atomic and instant, so it is also the simplest
//! possible exercise of the broker end to end.

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{SafetyClass, Tool, ToolResult};

/// Reports the current local date and time. Takes no arguments.
pub struct GetTimeTool;

#[async_trait]
impl Tool for GetTimeTool {
    fn name(&self) -> &str {
        "get_time"
    }

    fn description(&self) -> &str {
        "Returns the current local date and time."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
        })
    }

    /// Nothing to cancel: reading the clock cannot block, so `cancel` is
    /// checked only in the trivial "already fired" sense — there is no
    /// window during the call where firing it would matter.
    async fn call(&self, _args: Value, cancel: CancellationToken) -> ToolResult {
        if cancel.is_cancelled() {
            return ToolResult::Err("cancelled".to_string());
        }
        let now = chrono::Local::now();
        ToolResult::Ok(serde_json::json!({
            "iso8601": now.to_rfc3339(),
            "formatted": now.format("%A, %B %-d, %Y at %-I:%M %p").to_string(),
        }))
    }

    /// Atomic and instant — nothing in-flight to interrupt (§2.5.1).
    fn cancel(&self) {}

    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reports_a_well_formed_iso8601_timestamp() {
        let tool = GetTimeTool;
        let result = tool.call(Value::Null, CancellationToken::new()).await;

        let ToolResult::Ok(payload) = result else {
            panic!("expected Ok, got {result:?}");
        };
        let iso = payload["iso8601"].as_str().expect("iso8601 field");
        chrono::DateTime::parse_from_rfc3339(iso).expect("valid rfc3339 timestamp");
    }

    #[tokio::test]
    async fn an_already_cancelled_token_short_circuits() {
        let tool = GetTimeTool;
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = tool.call(Value::Null, cancel).await;
        assert_eq!(result, ToolResult::Err("cancelled".to_string()));
    }

    #[test]
    fn advertises_readonly_and_an_empty_schema() {
        let tool = GetTimeTool;
        assert_eq!(tool.name(), "get_time");
        assert_eq!(tool.safety_class(), SafetyClass::ReadOnly);
        assert_eq!(
            tool.parameters(),
            serde_json::json!({"type": "object", "properties": {}})
        );
    }
}
