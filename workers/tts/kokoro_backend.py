#!/usr/bin/env python3
"""Kokoro TTS backend (SPEC.md §2.4.1, EPIC 5.1).

This is the default TTS backend: Kokoro-82M, loaded from the `kokoro` hub
package and run on the device the Rust core selected (EPIC 0.7). Kokoro is
light enough to run on CPU while the GPU serves STT/LLM (SPEC.md §10) and
ships a **fixed voice set** rather than arbitrary voice cloning.

`kokoro`/`torch` are imported at module import time, so `worker.py` imports
this module lazily — the shared service's own tests inject a stub backend
and need neither.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

import numpy as np
from kokoro import KPipeline

log = logging.getLogger("marceline.tts.kokoro")

# Sample rate Kokoro actually emits. Declared here rather than assumed
# elsewhere (§2.4.1's chipmunk-bug warning): every `AudioChunk` this
# backend produces carries this rate.
KOKORO_SAMPLE_RATE = 24_000

# Kokoro's fixed voice set (v1: English-only, SPEC.md §55-56) — American
# ("a") and British ("b") English voices the `af_*`/`am_*`/`bf_*`/`bm_*`
# checkpoints ship. Swapping the configured `[tts].voice` among these is a
# config line, not a model change.
KOKORO_VOICES: tuple[str, ...] = (
    "af_heart",
    "af_bella",
    "af_sky",
    "af_nicole",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bm_george",
)

# Kokoro is English-only per voice set in v1; the language code selects
# which grapheme-to-phoneme front end `KPipeline` loads.
_LANG_CODE = "a"


class KokoroBackend:
    """Wraps a loaded Kokoro `KPipeline` behind the worker's backend seam.

    `worker.py` only ever calls [`load`][KokoroBackend.load],
    [`synthesize`][KokoroBackend.synthesize], [`name`][KokoroBackend.name]
    and [`voices`][KokoroBackend.voices], so a second backend (Piper,
    EPIC 5.5) drops in by matching those.
    """

    def __init__(self, device: str) -> None:
        """Records what to load; loading itself happens in `load()`.

        Args:
            device: Device name from `[tts].device` (`"cuda"` / `"cpu"`).
        """
        self._device = device
        self._pipeline: KPipeline | None = None

    @property
    def name(self) -> str:
        """Backend-qualified name reported in `TtsInfo.name`."""
        return "kokoro:82M"

    @property
    def sample_rate(self) -> int:
        """Sample rate (Hz) this backend actually emits."""
        return KOKORO_SAMPLE_RATE

    @property
    def voices(self) -> tuple[str, ...]:
        """Voice ids this backend can synthesize."""
        return KOKORO_VOICES

    def load(self) -> None:
        """Loads the Kokoro pipeline onto the configured device."""
        log.info("loading kokoro onto %s", self._device)
        self._pipeline = KPipeline(lang_code=_LANG_CODE, device=self._device)
        log.info("loaded kokoro")

    def synthesize(
        self, text: str, voice: str, cancel: threading.Event
    ) -> Iterator[np.ndarray]:
        """Synthesizes one text span into a sequence of mono f32 PCM chunks.

        Kokoro splits a span into its own sub-utterances internally
        (roughly clause-by-clause); each is yielded as soon as it is
        ready rather than concatenated, which is what lets playback start
        before the whole span finishes synthesizing.

        Args:
            text: Pre-segmented text to synthesize (sentence-chunking is
                the caller's job, §5.3).
            voice: Voice id to synthesize with, e.g. `"af_sky"`.
            cancel: Checked between sub-utterances; when set, generation
                stops early rather than burning compute on audio nobody
                will hear (§2.5.1).

        Yields:
            Mono f32 PCM arrays at [`sample_rate`][KokoroBackend.sample_rate].
        """
        if self._pipeline is None:
            raise RuntimeError("synthesize() called before load()")
        if cancel.is_set():
            return

        for _graphemes, _phonemes, audio in self._pipeline(text, voice=voice):
            if cancel.is_set():
                log.info("cancel observed between sub-utterances, stopping synthesis")
                return
            pcm = audio.detach().cpu().numpy().astype(np.float32, copy=False)
            yield pcm
