//! Integration tests for MCP discovery + registration (EPIC 6.4): connect
//! a real fake MCP server (a subprocess, real stdio), discover its tools,
//! register them namespaced, and call one through the broker exactly the
//! way the THINKING loop would (EPIC 6.3).

use marceline_core::config::{McpServerConfig, McpTransportConfig};
use marceline_core::{register_mcp_tools, ToolBroker, ToolResult};
use tokio_util::sync::CancellationToken;

fn fixture_path() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fake_mcp_stdio_server.py").to_string()
}

fn working_server(name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_string(),
        transport: McpTransportConfig::Stdio {
            command: "python3".to_string(),
            args: vec![fixture_path()],
        },
    }
}

fn broken_server(name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_string(),
        transport: McpTransportConfig::Stdio {
            command: "definitely-not-a-real-executable-xyz".to_string(),
            args: vec![],
        },
    }
}

#[tokio::test]
async fn a_discovered_tool_registers_namespaced_and_is_callable() {
    let mut broker = ToolBroker::new();
    let skipped = register_mcp_tools(&mut broker, &[working_server("calc")]).await;

    assert!(skipped.is_empty());
    let catalog = broker.catalog();
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].name, "calc.add");
    assert_eq!(catalog[0].description, "Adds two numbers.");
    assert_eq!(
        catalog[0].parameters["required"],
        serde_json::json!(["a", "b"])
    );

    let result = broker
        .dispatch(
            "calc.add",
            serde_json::json!({"a": 4, "b": 5}),
            CancellationToken::new(),
        )
        .await;

    let ToolResult::Ok(content) = result else {
        panic!("expected Ok, got {result:?}");
    };
    assert_eq!(content[0]["text"], "9");
}

#[tokio::test]
async fn an_unreachable_server_is_skipped_and_others_still_register() {
    let mut broker = ToolBroker::new();
    let skipped = register_mcp_tools(
        &mut broker,
        &[broken_server("dead"), working_server("calc")],
    )
    .await;

    assert_eq!(skipped, vec!["dead".to_string()]);
    let catalog = broker.catalog();
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].name, "calc.add");
}

#[tokio::test]
async fn two_servers_namespace_their_tools_independently() {
    let mut broker = ToolBroker::new();
    let skipped = register_mcp_tools(
        &mut broker,
        &[working_server("first"), working_server("second")],
    )
    .await;

    assert!(skipped.is_empty());
    let mut names: Vec<String> = broker.catalog().into_iter().map(|spec| spec.name).collect();
    names.sort();
    assert_eq!(names, vec!["first.add".to_string(), "second.add".to_string()]);
}
