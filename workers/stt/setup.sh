#!/usr/bin/env bash
# Creates the STT worker's venv and installs its pinned deps (EPIC 3.1).
#
# Also drops a .pth file into the venv so the generated protobuf package
# at python/marceline_protocol is importable as `marceline_protocol`
# without copying or vendoring the stubs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
VENV_DIR="${ROOT}/.venv"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${ROOT}/requirements.txt"

SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${REPO_ROOT}/python" > "${SITE_PACKAGES}/marceline_protocol.pth"

echo "STT worker venv ready at ${VENV_DIR}"
echo "Run: ${VENV_DIR}/bin/python ${ROOT}/worker.py \\"
echo "       --socket-path /tmp/marceline-stt.sock --model-id large-v3 --device cuda"
