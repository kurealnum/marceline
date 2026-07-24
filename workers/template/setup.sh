#!/usr/bin/env bash
# Creates a venv next to this script and installs the pinned worker deps.
# Real workers run this after copying the template directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT}/.venv"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${ROOT}/requirements.txt"

echo "Worker venv ready at ${VENV_DIR}"
echo "Run: ${VENV_DIR}/bin/python worker.py --socket-path /tmp/worker.sock --model-id template --device cpu"
