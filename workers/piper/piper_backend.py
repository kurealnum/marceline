#!/usr/bin/env python3
"""Piper TTS backend (SPEC.md §2.4.1, EPIC 5.5).

The second TTS backend, proving the `Tts` gRPC contract was not quietly
shaped around Kokoro's assumptions (EPIC 5.2's Rust client works against
this with no code change — only `[tts].backend`/`voice` differ). Piper is
also the designated low-resource fallback (§1.1, §10): a single ONNX voice
model, light enough to run on CPU where Kokoro is not viable.

Unlike Kokoro's fixed voice set selectable per request, a Piper worker
loads exactly one voice model at startup — swapping voices is a worker
restart with a different `--voice`, the same "hot-swappable via a config
line" pattern the STT workers use for models (§2.4).

`piper`/`onnxruntime` are imported at module import time, so `worker.py`
imports this module lazily — the shared service's own tests inject a stub
backend and need neither.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from piper import PiperVoice

log = logging.getLogger("marceline.tts.piper")

# Directory holding downloaded `.onnx` (+ `.onnx.json`) voice models,
# alongside this file. Keeps voice resolution a filename lookup rather
# than requiring a full path in `[tts].voice`.
_VOICES_DIR = Path(__file__).parent / "voices"


def resolve_model_path(voice: str) -> str:
    """Resolves a configured voice id to an on-disk Piper model path.

    A `voice` that already names an existing file (an absolute path, or a
    relative one resolvable from the current directory) is used as-is, so
    a model living anywhere is still usable. Otherwise `voice` is treated
    as a short name and looked up as `<voice>.onnx` under `voices/`, the
    same shape `piper` voice downloads ship in.
    """
    as_path = Path(voice)
    if as_path.is_file():
        return str(as_path)
    return str(_VOICES_DIR / f"{voice}.onnx")


class PiperBackend:
    """Wraps a loaded Piper `PiperVoice` behind the worker's backend seam.

    `worker.py` only ever calls [`load`][PiperBackend.load],
    [`synthesize`][PiperBackend.synthesize], [`name`][PiperBackend.name]
    and [`voices`][PiperBackend.voices] — the same seam `KokoroBackend`
    implements, which is what lets the shared `tts_service` (and the Rust
    gRPC client) treat both identically.
    """

    def __init__(self, model_path: str, voice_id: str, device: str) -> None:
        """Records what to load; loading itself happens in `load()`.

        Args:
            model_path: Resolved filesystem path to the `.onnx` voice model.
            voice_id: Configured voice id from `[tts].voice`, reported in
                `TtsInfo.name`/`voices` (the model may have been resolved
                from it, but the id itself is what config asked for).
            device: Device name from `[tts].device`; `"cuda"` enables
                Piper's ONNX Runtime CUDA execution provider.
        """
        self._model_path = model_path
        self._voice_id = voice_id
        self._device = device
        self._voice: PiperVoice | None = None

    @property
    def name(self) -> str:
        """Backend-qualified name reported in `TtsInfo.name`."""
        return f"piper:{self._voice_id}"

    @property
    def sample_rate(self) -> int:
        """Sample rate (Hz) this backend actually emits.

        Piper voices are not all the same rate (some ship at 16 kHz, others
        22.05 kHz), so this reads the loaded model's own config rather than
        assuming one — the exact chipmunk-bug trap §2.4.1 warns about.
        """
        if self._voice is None:
            raise RuntimeError("sample_rate accessed before load()")
        return self._voice.config.sample_rate

    @property
    def voices(self) -> tuple[str, ...]:
        """Voice ids this backend can synthesize.

        Exactly one: a Piper worker loads a single voice model at startup,
        unlike Kokoro's fixed set selectable per request.
        """
        return (self._voice_id,)

    def load(self) -> None:
        """Loads the Piper voice model onto the configured device."""
        log.info("loading piper voice %s from %s", self._voice_id, self._model_path)
        self._voice = PiperVoice.load(
            self._model_path, use_cuda=self._device == "cuda"
        )
        log.info(
            "loaded piper voice %s (sample_rate=%d)",
            self._voice_id,
            self._voice.config.sample_rate,
        )

    def synthesize(
        self, text: str, voice: str, cancel: threading.Event
    ) -> Iterator[np.ndarray]:
        """Synthesizes one text span into a sequence of mono f32 PCM chunks.

        Args:
            text: Pre-segmented text to synthesize (sentence-chunking is
                the caller's job, §5.3).
            voice: Voice id the request asked for. This worker only ever
                has the one it loaded; a mismatch is logged and ignored
                rather than failing the turn, since the fixed-voice-per-
                worker model makes a mid-stream swap meaningless.
            cancel: Checked between per-sentence audio chunks; when set,
                generation stops early rather than burning compute on
                audio nobody will hear (§2.5.1).

        Yields:
            Mono f32 PCM arrays at [`sample_rate`][PiperBackend.sample_rate].
        """
        if self._voice is None:
            raise RuntimeError("synthesize() called before load()")
        if voice and voice != self._voice_id:
            log.warning(
                "requested voice %r ignored; this worker loaded the fixed voice %r",
                voice,
                self._voice_id,
            )
        if cancel.is_set():
            return

        # `PiperVoice.synthesize` yields one already-f32, already-
        # normalized `AudioChunk` per sentence it detects in `text` (Piper
        # does its own clause splitting inside a span); each is handed
        # onward as soon as it is ready rather than concatenated.
        for chunk in self._voice.synthesize(text):
            if cancel.is_set():
                log.info("cancel observed between audio chunks, stopping synthesis")
                return
            yield chunk.audio_float_array
