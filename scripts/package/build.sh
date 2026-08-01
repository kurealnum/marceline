#!/usr/bin/env bash
# Builds a release install tree for the Rust daemon + CLI plus the Python
# STT/TTS workers (EPIC 12.1).
#
# What this does NOT do (see this story's scope in issue 12.1):
#   - Download models (EPIC 12.2's job — first run does that).
#   - Cross-compile for another OS. Linux is the only OS this repo builds
#     or tests on today; a macOS/Windows variant of this script is future
#     work once there is a machine to validate it against (this one only
#     has `uname` on Linux to check).
#
# Output layout (relative to --out, default `dist/marceline`):
#   bin/marceline         release CLI/daemon binary
#   workers/<name>/...    each worker directory, venv included, copied
#                         verbatim (so `core::worker_paths::workers_root`'s
#                         "workers/ next to bin/" resolution just works)
#   config.toml           the repo's default config, as a starting point
#   PACKAGING.md           this story's install/CUDA-prerequisite notes
#
# Usage: scripts/package/build.sh [--out <dir>]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/dist/marceline"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: this build script only supports Linux today (see this file's header)." >&2
  exit 1
fi

echo "==> Building the release workspace binary (cli, core, protocol)"
(cd "${ROOT}" && cargo build --release -p cli)

echo "==> Assembling install tree at ${OUT}"
rm -rf "${OUT}"
mkdir -p "${OUT}/bin"
cp "${ROOT}/target/release/marceline" "${OUT}/bin/marceline"

echo "==> Bundling Python workers"
mkdir -p "${OUT}/workers"
for worker_dir in "${ROOT}"/workers/*/; do
  name="$(basename "${worker_dir}")"
  if [[ ! -d "${worker_dir}/.venv" ]]; then
    if [[ -x "${worker_dir}/setup.sh" ]]; then
      echo "    ${name}: no .venv yet, running its setup.sh"
      (cd "${worker_dir}" && ./setup.sh)
    else
      echo "    ${name}: no .venv and no setup.sh — skipping (nothing to bundle)"
      continue
    fi
  fi
  echo "    ${name}: copying (including its .venv)"
  cp -a "${worker_dir}" "${OUT}/workers/${name}"
done

echo "==> Copying shared python/ packages the workers' venvs point at via .pth"
cp -a "${ROOT}/python" "${OUT}/python"
# Each worker venv's marceline.pth points at the *dev checkout's* absolute
# python/ path (see e.g. workers/stt/setup.sh) — rewrite it to the copy
# that ships alongside this install tree instead, so the bundle is
# self-contained rather than silently depending on the checkout it was
# built from still existing at the same path.
for venv in "${OUT}"/workers/*/.venv; do
  pth="$(find "${venv}" -name 'marceline.pth' 2>/dev/null | head -n1 || true)"
  if [[ -n "${pth}" ]]; then
    echo "${OUT}/python" > "${pth}"
  fi
done

cp "${ROOT}/config.toml" "${OUT}/config.toml"
cp "${ROOT}/scripts/package/PACKAGING.md" "${OUT}/PACKAGING.md"
cp "${ROOT}/scripts/package/install.sh" "${OUT}/install.sh"
chmod +x "${OUT}/install.sh"

echo "==> Done. Install tree ready at ${OUT}"
echo "    Run ${OUT}/install.sh to install it, or read ${OUT}/PACKAGING.md."
