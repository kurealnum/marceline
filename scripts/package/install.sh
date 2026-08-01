#!/usr/bin/env bash
# Installs a build tree produced by scripts/package/build.sh onto this
# machine (EPIC 12.1). Copies the whole tree to --prefix (default
# ~/.local/share/marceline) and symlinks bin/marceline onto --bin-dir
# (default ~/.local/bin) so it's runnable as just `marceline` if that
# directory is on $PATH.
#
# This script is copied into the install tree itself by build.sh, so it
# runs correctly whether invoked from there or from a source checkout's
# scripts/package/ directly.
#
# Usage: install.sh [--prefix <dir>] [--bin-dir <dir>]
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${HOME}/.local/share/marceline"
BIN_DIR="${HOME}/.local/bin"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --bin-dir)
      BIN_DIR="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${SELF_DIR}/bin/marceline" ]]; then
  echo "error: ${SELF_DIR}/bin/marceline not found — run this from an install tree produced by scripts/package/build.sh" >&2
  exit 1
fi

echo "==> Installing to ${PREFIX}"
mkdir -p "${PREFIX}"
cp -a "${SELF_DIR}/bin" "${SELF_DIR}/workers" "${SELF_DIR}/python" "${PREFIX}/"
if [[ ! -f "${PREFIX}/config.toml" ]]; then
  # Never overwrite a config an operator may have already edited on a
  # reinstall/upgrade — only seed it the first time.
  cp "${SELF_DIR}/config.toml" "${PREFIX}/config.toml"
fi

echo "==> Linking ${BIN_DIR}/marceline -> ${PREFIX}/bin/marceline"
mkdir -p "${BIN_DIR}"
ln -sf "${PREFIX}/bin/marceline" "${BIN_DIR}/marceline"

echo
echo "Installed. If ${BIN_DIR} is on your PATH, run: marceline --version"
echo
echo "CUDA prerequisite (SPEC.md §9.9): v1's default STT/TTS backends"
echo "expect a CUDA-capable GPU + driver. A working \`nvidia-smi\` on this"
echo "machine is the quick check before \`marceline start\`; see PACKAGING.md"
echo "next to this script for details."
