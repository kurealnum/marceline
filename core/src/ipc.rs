//! Worker IPC transport (SPEC.md §2.3): gRPC over a local unix domain
//! socket.
//!
//! Everything that talks to a Python worker — the supervisor's health
//! polling (EPIC 0.6), the STT client backend (EPIC 3.2), TTS later —
//! dials through here, so the "how do we reach a worker" decision lives
//! in exactly one place.

use std::path::Path;

use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;

/// Connects a gRPC channel to a worker over its unix domain socket.
///
/// The URI is a placeholder that `tonic::transport::Endpoint` requires but
/// the connector ignores: it always dials `socket_path`. Nothing is sent
/// until the channel is used, so a successful return means the socket
/// accepted a connection, not that the worker finished loading its model.
pub async fn connect_uds(socket_path: &Path) -> Result<Channel, tonic::transport::Error> {
    let socket_path = socket_path.to_path_buf();
    Endpoint::try_from("http://[::]:0")?
        .connect_with_connector(service_fn(move |_: Uri| {
            let socket_path = socket_path.clone();
            async move {
                let stream = tokio::net::UnixStream::connect(socket_path).await?;
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
            }
        }))
        .await
}
