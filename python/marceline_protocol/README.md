# marceline_protocol (generated)

Python gRPC stubs generated from `protocol/proto/*.proto` (SPEC.md
§2.4.1, EPIC 0.5) — the same schema `protocol/build.rs` compiles to
Rust. Consumed by Python workers (`workers/*`, EPIC 0.4).

**Do not hand-edit `*_pb2.py` / `*_pb2_grpc.py` / `*_pb2.pyi` files.**
Regenerate them after changing a `.proto`:

```
python3 -m venv .venv-protoc && .venv-protoc/bin/pip install grpcio-tools
PATH="$(pwd)/.venv-protoc/bin:$PATH" ./scripts/gen-python-proto.sh
```
