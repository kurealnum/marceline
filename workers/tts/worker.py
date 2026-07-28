#!/usr/bin/env python3
"""Kokoro TTS worker entrypoint (SPEC.md §2.3, EPIC 5.1).

The default speech engine and the process the Rust core talks to for all
TTS. All the transport, streaming, and cancel plumbing lives in
`marceline_worker.tts_service`, shared with the Piper worker (EPIC 5.5) so
the two cannot drift apart on the contract. What is specific to this
worker is one line: which backend class to load.
"""

import sys

from marceline_worker.tts_service import run_worker


def build_backend(args) -> object:
    """Builds the Kokoro backend for the parsed CLI args.

    Imported here rather than at module scope so `kokoro`/`torch` are only
    loaded by the real worker process — the shared service's own tests
    inject a stub backend and need neither.
    """
    from kokoro_backend import KokoroBackend

    return KokoroBackend(args.device)


if __name__ == "__main__":
    sys.exit(run_worker(build_backend, description="Marceline TTS worker (Kokoro)"))
