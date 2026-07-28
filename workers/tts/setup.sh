#!/usr/bin/env bash
# Creates the TTS worker's venv and installs its pinned deps (EPIC 5.1).
#
# Also drops a .pth file into the venv pointing at the repository's python/
# directory, which makes both shared packages importable without copying or
# vendoring them: `marceline_protocol` (generated stubs) and
# `marceline_worker` (the gRPC service every TTS backend shares).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
VENV_DIR="${ROOT}/.venv"

# torch==2.4.1 (pinned below) has no wheels past cp312, so the venv must be
# built with a specific interpreter rather than whatever `python3` resolves
# to on the host (e.g. 3.14).
PYTHON_BIN="$(command -v python3.12 || true)"
if [[ -z "${PYTHON_BIN}" ]] && command -v pyenv >/dev/null; then
  PYTHON_BIN="$(pyenv root)/versions/3.12.11/bin/python3.12"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python3.12 not found. Install it (e.g. 'pyenv install 3.12.11') and retry." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${ROOT}/requirements.txt"

SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${REPO_ROOT}/python" > "${SITE_PACKAGES}/marceline.pth"

echo "TTS worker venv ready at ${VENV_DIR}"
echo "Run: ${VENV_DIR}/bin/python ${ROOT}/worker.py \\"
echo "       --socket-path /tmp/marceline-tts.sock --model-id af_sky --device cpu"
