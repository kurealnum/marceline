#!/usr/bin/env bash
# Creates the faster-whisper STT worker's venv and installs its deps (EPIC 3.5).
#
# Also drops a .pth file into the venv pointing at the repository's python/
# directory, which makes both shared packages importable without copying or
# vendoring them: `marceline_protocol` (generated stubs) and
# `marceline_worker` (the gRPC service every STT backend shares).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
VENV_DIR="${ROOT}/.venv"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${ROOT}/requirements.txt"

SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${REPO_ROOT}/python" > "${SITE_PACKAGES}/marceline.pth"

echo "faster-whisper worker venv ready at ${VENV_DIR}"
echo "Run: ${VENV_DIR}/bin/python ${ROOT}/worker.py \\"
echo "       --socket-path /tmp/marceline-stt.sock --model-id large-v3 --device cuda"
