#!/usr/bin/env python3
"""`faster-whisper` STT worker entrypoint (SPEC.md §2.4, EPIC 3.5).

Selected by `[stt].backend = "faster-whisper"`, which resolves to this
directory (see `SttWorkerPaths::for_backend`). Everything about the gRPC
contract — transport, buffering, cancel plumbing, health service — comes
from `marceline_worker.stt_service`, shared with the default `whisper`
worker. That sharing is the point: the backend swap is a different model
runtime, not a different protocol.
"""

import sys

from marceline_worker.stt_service import run_worker


def build_backend(args) -> object:
    """Builds the faster-whisper backend for the parsed CLI args.

    Imported here rather than at module scope so the ML stack loads only in
    the real worker process, keeping the shared service importable (and
    testable) without it.
    """
    from faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend(args.model_id, args.device, args.lang)


if __name__ == "__main__":
    sys.exit(
        run_worker(build_backend, description="Marceline STT worker (faster-whisper)")
    )
