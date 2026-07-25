#!/usr/bin/env python3
"""HF `whisper` STT worker entrypoint (SPEC.md §2.3, EPIC 3.1).

The default STT backend. All the transport, buffering, and cancel plumbing
lives in `marceline_worker.stt_service`, shared with the `faster-whisper`
worker (EPIC 3.5) so the two cannot drift apart on the contract. What is
specific to this worker is one line: which backend class to load.
"""

import sys

from marceline_worker.stt_service import run_worker


def build_backend(args) -> object:
    """Builds the HF Whisper backend for the parsed CLI args.

    Imported here rather than at module scope so the multi-gigabyte
    torch/transformers stack is only loaded by the real worker process —
    the shared service's own tests inject a stub backend and need neither.
    """
    from whisper_backend import WhisperBackend

    return WhisperBackend(args.model_id, args.device, args.lang)


if __name__ == "__main__":
    sys.exit(run_worker(build_backend, description="Marceline STT worker (HF whisper)"))
