#!/usr/bin/env bash
# Regenerates Python gRPC stubs from protocol/proto/*.proto into
# python/marceline_protocol/ (SPEC.md §2.4.1, EPIC 0.5). Run from repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${ROOT}/protocol/proto"
OUT_DIR="${ROOT}/python/marceline_protocol"

mkdir -p "${OUT_DIR}"

python3 -m grpc_tools.protoc \
  -I "${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  --pyi_out="${OUT_DIR}" \
  "${PROTO_DIR}"/common.proto "${PROTO_DIR}"/stt.proto "${PROTO_DIR}"/tts.proto

# grpc_tools emits absolute-style `import common_pb2` — rewrite to relative
# imports so the generated package is importable as `marceline_protocol.*`.
for f in "${OUT_DIR}"/*_pb2*.py; do
  sed -i -E 's/^import (common_pb2|stt_pb2|tts_pb2)/from . import \1/' "$f"
done

touch "${OUT_DIR}/__init__.py"
echo "Python stubs generated in ${OUT_DIR}"
