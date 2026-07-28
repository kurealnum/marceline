//! The tool broker: built-ins + MCP merged into one catalog (SPEC.md §4,
//! §2.5.1, EPIC 6.1).
//!
//! Without this, the model has no idea what it can do and the THINKING
//! loop (EPIC 6.3) has nothing to dispatch to. [`ToolBroker`] owns every
//! callable [`Tool`] and hands the LLM client a single merged
//! [`ToolSpec`][crate::llm::ToolSpec] list — the shape
//! [`ChatRequest`][crate::llm::ChatRequest]`.tools` already expects, so a
//! broker producing the wrong thing here would be caught by the existing
//! OpenAI-wire serialization rather than needing its own.
//!
//! **v1 registers read-only tools only** (§4, §10). The [`Tool`] trait,
//! [`SafetyClass`], and the cancel contract all exist from day one so a
//! side-effecting tool (EPIC 6.5) is a new impl, not a trait rewrite —
//! but nothing dangerous is registered until the security pass (EPIC 14)
//! signs off.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::llm::ToolSpec;

/// How much latitude a tool has to run without a human in the loop
/// (SPEC.md §2.5.1, §9.4/6.5).
///
/// Read entirely as documentation in v1: nothing in [`ToolBroker`] enforces
/// a class yet — that enforcement is EPIC 6.5's and 14.3's job. The enum
/// exists now so a `Tool` impl states its class from day one instead of
/// every existing tool needing a retrofit when enforcement lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyClass {
    /// Cancellable, no side effects — aborting mid-call and discarding the
    /// result is always safe (`web_search`, `read_file`, `get_time`).
    ReadOnly,
    /// Cannot un-ring the bell once started. `Tool::cancel` may be a
    /// no-op for these; per §2.5.1, cancellation instead means "don't feed
    /// the result back," not "undo it." Requires voice confirmation
    /// (EPIC 6.5) before it may run at all.
    SideEffecting,
}

/// The outcome of one [`Tool::call`], fed back to the LLM as a `Tool`
/// [`Message`][crate::llm::Message].
///
/// A dedicated enum rather than `Result<Value, EngineError>`: a tool
/// failure is not a plugin-transport failure (EngineError's domain, SPEC.md
/// §2.4) — it is normal tool output that the *model* should see and reason
/// about ("that file doesn't exist"), not an error the orchestrator
/// necessarily has to abandon the turn for.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolResult {
    /// The tool ran and produced this JSON payload.
    Ok(Value),
    /// The tool could not produce a result; the message is surfaced back
    /// to the model as the tool's output, so it can decide how to
    /// proceed (retry, apologize, try something else) rather than the
    /// turn silently dying.
    Err(String),
}

/// A tool the model may call (SPEC.md §2.5.1).
///
/// Implementors describe themselves (`name`/`description`/`parameters`)
/// rather than the broker hard-coding a catalog, so registering a tool is
/// the only step needed to add it to what the model sees.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name the model references in a tool call. Must be unique across
    /// every tool registered on one [`ToolBroker`] (built-ins and MCP
    /// tools alike, the latter namespaced `serverName.toolName` per
    /// EPIC 6.4).
    fn name(&self) -> &str;
    /// Human-readable description shown to the model, so it can decide
    /// when to call this tool.
    fn description(&self) -> &str;
    /// JSON Schema for this tool's call arguments.
    fn parameters(&self) -> Value;
    /// Runs the tool. `cancel` is the run's cancellation token (§2.5.1),
    /// cloned in by the broker; a read-only tool aborts and discards its
    /// result when it fires, a side-effecting one may ignore it.
    async fn call(&self, args: Value, cancel: CancellationToken) -> ToolResult;
    /// Requests early termination of an in-flight call. Must exist on
    /// every tool (§2.5.1) even when it is a no-op — an atomic tool with
    /// nothing to interrupt still has to satisfy the trait.
    fn cancel(&self);
    /// This tool's safety class, read by the (not-yet-built) enforcement
    /// path in EPIC 6.5/14.3.
    fn safety_class(&self) -> SafetyClass;
}

/// A name collision when registering a tool.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("a tool named {0:?} is already registered")]
pub struct DuplicateToolError(pub String);

/// Owns every callable [`Tool`] and hands the LLM a single merged catalog
/// (SPEC.md §4).
///
/// Built-ins (EPIC 6.2) and MCP tools (EPIC 6.4) register into the same
/// broker — from the THINKING loop's point of view (EPIC 6.3) there is no
/// difference between the two once they are in here.
#[derive(Default)]
pub struct ToolBroker {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolBroker {
    /// Creates an empty broker.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers `tool` under [`Tool::name`].
    ///
    /// Errors on a name collision rather than silently overwriting: two
    /// tools racing for one name is a configuration bug (e.g. an MCP
    /// server's namespaced name colliding with a built-in), and the model
    /// dispatching to whichever registered last would hide it.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), DuplicateToolError> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(DuplicateToolError(name));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// The merged catalog, in the shape
    /// [`ChatRequest::tools`][crate::llm::ChatRequest] expects.
    ///
    /// Built from what is actually registered, not a separately
    /// maintained list — a tool that fails to register (a duplicate name)
    /// cannot appear in the catalog while being uncallable, because there
    /// is only one source of truth.
    pub fn catalog(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|tool| ToolSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
            })
            .collect()
    }

    /// Looks up `name` and runs it with `args`, or returns a structured
    /// error for a name the model hallucinated or that maps to nothing
    /// registered — never a panic, since an LLM inventing a tool name is
    /// an expected failure mode, not a bug.
    pub async fn dispatch(&self, name: &str, args: Value, cancel: CancellationToken) -> ToolResult {
        match self.tools.get(name) {
            Some(tool) => tool.call(args, cancel).await,
            None => ToolResult::Err(format!("unknown tool: {name:?}")),
        }
    }

    /// Requests cancellation of `name`'s in-flight call, if it is
    /// registered. A name that is not registered is a no-op rather than
    /// an error: nothing to cancel is not a fault.
    pub fn cancel(&self, name: &str) {
        if let Some(tool) = self.tools.get(name) {
            tool.cancel();
        }
    }

    /// Number of tools currently registered.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True when nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A stub tool for exercising the broker without a real built-in.
    struct StubTool {
        name: String,
        cancelled: Arc<AtomicBool>,
    }

    #[async_trait]
    impl Tool for StubTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "a stub tool for tests"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn call(&self, args: Value, _cancel: CancellationToken) -> ToolResult {
            ToolResult::Ok(args)
        }
        fn cancel(&self) {
            self.cancelled.store(true, Ordering::SeqCst);
        }
        fn safety_class(&self) -> SafetyClass {
            SafetyClass::ReadOnly
        }
    }

    fn stub(name: &str) -> Arc<StubTool> {
        Arc::new(StubTool {
            name: name.to_string(),
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }

    #[test]
    fn an_empty_broker_has_an_empty_catalog() {
        let broker = ToolBroker::new();
        assert!(broker.catalog().is_empty());
        assert!(broker.is_empty());
    }

    #[test]
    fn catalog_reports_every_registered_tool_with_its_schema() {
        let mut broker = ToolBroker::new();
        broker.register(stub("get_time")).unwrap();

        let catalog = broker.catalog();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].name, "get_time");
        assert_eq!(catalog[0].description, "a stub tool for tests");
        assert_eq!(
            catalog[0].parameters,
            serde_json::json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn registering_a_duplicate_name_errors_rather_than_overwriting() {
        let mut broker = ToolBroker::new();
        broker.register(stub("get_time")).unwrap();

        let err = broker.register(stub("get_time")).unwrap_err();
        assert_eq!(err, DuplicateToolError("get_time".to_string()));
        // The original registration survives the failed second one.
        assert_eq!(broker.len(), 1);
    }

    #[tokio::test]
    async fn dispatch_runs_the_named_tool_and_returns_its_result() {
        let mut broker = ToolBroker::new();
        broker.register(stub("echo")).unwrap();

        let result = broker
            .dispatch("echo", serde_json::json!({"x": 1}), CancellationToken::new())
            .await;

        assert_eq!(result, ToolResult::Ok(serde_json::json!({"x": 1})));
    }

    #[tokio::test]
    async fn dispatching_an_unknown_tool_returns_a_structured_error() {
        let broker = ToolBroker::new();

        let result = broker
            .dispatch("nonexistent", Value::Null, CancellationToken::new())
            .await;

        assert_eq!(result, ToolResult::Err("unknown tool: \"nonexistent\"".to_string()));
    }

    #[test]
    fn cancel_reaches_the_named_tools_cancel_method() {
        let mut broker = ToolBroker::new();
        let tool = stub("slow_thing");
        let cancelled = Arc::clone(&tool.cancelled);
        broker.register(tool).unwrap();

        broker.cancel("slow_thing");

        assert!(cancelled.load(Ordering::SeqCst));
    }

    #[test]
    fn cancelling_an_unregistered_tool_is_a_no_op() {
        let broker = ToolBroker::new();
        // Must not panic.
        broker.cancel("nothing-here");
    }
}
