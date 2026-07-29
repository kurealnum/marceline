//! The JSON-RPC 2.0 response envelope shared by both transports.
//!
//! Requests differ in how they are sent (a line on stdin vs. an HTTP POST
//! body), but a successful/error response looks identical either way, so
//! both [`super::stdio_transport`] and [`super::http_transport`] parse the
//! same shape.

use serde::Deserialize;
use serde_json::Value;

/// One JSON-RPC response: either `result` or `error` is present, never
/// both (per spec) — modeled as two `Option`s rather than an enum because
/// a malformed server sending neither, or both, should be visible as
/// "nothing usable" rather than a deserialization panic.
#[derive(Debug, Deserialize)]
pub struct WireResponse {
    /// Echoes the request's id. `None` on a response this client did not
    /// ask for (or a malformed one) — the caller decides what to do.
    #[serde(default)]
    pub id: Option<u64>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<WireError>,
}

/// A JSON-RPC error object.
#[derive(Debug, Deserialize)]
pub struct WireError {
    /// Human-readable error description.
    #[serde(default)]
    pub message: String,
}
