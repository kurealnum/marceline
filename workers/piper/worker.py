#!/usr/bin/env python3
"""Piper TTS worker entrypoint (SPEC.md §2.3, EPIC 5.5).

The second speech engine, selected by `[tts].backend = "piper"`. All the
transport, streaming, and cancel plumbing lives in
`marceline_worker.tts_service`, shared with the Kokoro worker (EPIC 5.1)
so the two cannot drift apart on the contract. What is specific to this
worker is one line: which backend class to load.
"""

import sys

from marceline_worker.tts_service import run_worker


def build_backend(args) -> object:
    """Builds the Piper backend for the parsed CLI args.

    Imported here rather than at module scope so `piper`/`onnxruntime` are
    only loaded by the real worker process — the shared service's own
    tests inject a stub backend and need neither.
    """
    from piper_backend import PiperBackend, resolve_model_path

    model_path = resolve_model_path(args.voice)
    return PiperBackend(model_path, args.voice, args.device)


if __name__ == "__main__":
    sys.exit(run_worker(build_backend, description="Marceline TTS worker (Piper)"))
