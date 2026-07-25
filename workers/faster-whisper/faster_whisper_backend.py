#!/usr/bin/env python3
"""`faster-whisper` STT backend (SPEC.md §2.4, EPIC 3.5).

The second STT backend, and the proof that swapping engines is a config
line: it satisfies the same three-method contract the shared service
(`marceline_worker.stt_service`) calls, so neither that service nor any
Rust code changes to use it.

Same model weights as the HF backend, different runtime — CTranslate2
instead of PyTorch, which is faster and lighter on VRAM. The differences
that *do* exist are exactly the ones the contract was designed to absorb:

* **Model ids are plain names.** `"large-v3"` is what `faster-whisper`
  wants; the HF backend expands the same value to `openai/whisper-large-v3`.
  Naming stays a backend concern, so `[stt].model` means the same thing to
  a user across both.
* **Cancel is a generator break, not a stopping criterion.** `transcribe()`
  returns a lazy generator that decodes one segment per `next()`, so the
  cooperative cancel check (§2.5.1) is a loop condition here where the HF
  backend needed a `StoppingCriteria`. Both check between decode steps;
  neither relies on socket close.
* **Confidence comes from `avg_logprob`** the library already reports,
  rather than being recomputed from token scores.

If supporting this had required touching the gRPC service or the Rust
client, that would have been a contract leak to fix in the contract — not
to paper over here.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger("marceline.stt.faster_whisper")

# Sample rate every Whisper-family checkpoint expects. Audio at any other
# rate is resampled by the shared service before it reaches this backend.
WHISPER_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Transcription:
    """One committed transcript produced from a single audio segment.

    Structurally identical to the HF backend's, because the shared service
    reads the same three fields off both.
    """

    #: Recognized text, whitespace-trimmed. Empty when decoding was
    #: cancelled before producing anything.
    text: str
    #: Model confidence in `[0, 1]`.
    confidence: float
    #: True when decoding stopped early because of a cooperative cancel.
    cancelled: bool = False
    #: Highest per-segment probability that this audio held no speech, or
    #: `None` when unavailable. The maximum rather than the mean: one
    #: segment the model thinks is silence is enough to be suspicious, and
    #: averaging would let confident neighbours hide it.
    no_speech_prob: float | None = None
    #: Mean per-segment average log probability, or `None` when unavailable.
    avg_logprob: float | None = None


class FasterWhisperBackend:
    """Wraps a CTranslate2 Whisper model behind the worker backend seam."""

    def __init__(self, model: str, device: str, lang: str) -> None:
        """Records what to load; loading itself happens in `load()`.

        Args:
            model: Model name from `[stt].model`, e.g. `"large-v3"`. Passed
                through as-is: `faster-whisper` resolves both its own short
                names and full hub ids.
            device: Device name from `[stt].device` (`"cuda"` / `"cpu"`).
            lang: Recognition language from `[stt].lang`, e.g. `"en"`.
        """
        self._model_id = model
        self._device = device
        self._lang = lang
        self._model: WhisperModel | None = None

    @property
    def name(self) -> str:
        """Backend-qualified model name reported in `SttInfo.name`.

        The backend prefix is what makes a swap visible in logs and in
        `marceline transcribe` output — two backends can load the same
        weights, and it matters which one answered.
        """
        return f"faster-whisper:{self._model_id}"

    @property
    def sample_rate(self) -> int:
        """Sample rate (Hz) this backend requires its input audio in."""
        return WHISPER_SAMPLE_RATE

    def load(self) -> None:
        """Loads the CTranslate2 model onto the configured device.

        int8 on CPU and float16 on GPU are the compute types this runtime is
        built around; picking them here rather than exposing another config
        knob keeps `[stt]` about *what* to run, not how to quantize it.
        """
        compute_type = "float16" if self._device == "cuda" else "int8"
        log.info(
            "loading %s onto %s (%s)", self._model_id, self._device, compute_type
        )
        self._model = WhisperModel(
            self._model_id, device=self._device, compute_type=compute_type
        )
        log.info("loaded %s", self._model_id)

    def transcribe(self, pcm: np.ndarray, cancel: threading.Event) -> Transcription:
        """Transcribes one mono 16 kHz f32 segment into committed text.

        Args:
            pcm: Mono f32 samples at
                [`sample_rate`][FasterWhisperBackend.sample_rate].
            cancel: Checked between decoded segments; when set, decoding
                stops early and the result is marked `cancelled`.

        Returns:
            The committed [`Transcription`][Transcription] for this segment.
        """
        if self._model is None:
            raise RuntimeError("transcribe() called before load()")
        if cancel.is_set():
            return Transcription(text="", confidence=0.0, cancelled=True)

        # `transcribe` itself is cheap: it returns a generator that does the
        # decoding lazily, which is what makes the cancel check below land
        # between decode steps rather than after all of them.
        segments, _info = self._model.transcribe(
            pcm,
            # Language is pinned from config rather than detected: v1 is
            # English-only, and letting Whisper guess is a known source of
            # spurious language switches on short, noisy segments.
            language=self._lang,
            task="transcribe",
        )

        texts: list[str] = []
        logprobs: list[float] = []
        no_speech_probs: list[float] = []
        for segment in segments:
            if cancel.is_set():
                # Text decoded so far is a truncated fragment, not a
                # committed transcript; drop it rather than letting half an
                # utterance reach the LLM.
                log.info("transcription cancelled after %d segments", len(texts))
                return Transcription(text="", confidence=0.0, cancelled=True)
            texts.append(segment.text)
            logprobs.append(segment.avg_logprob)
            no_speech_probs.append(segment.no_speech_prob)

        finite_logprobs = [value for value in logprobs if np.isfinite(value)]
        return Transcription(
            text="".join(texts).strip(),
            confidence=self._confidence(logprobs),
            # Reported straight through for the Rust-side hallucination
            # guard (EPIC 3.6); this runtime gives us both signals per
            # segment, so neither has to be inferred.
            no_speech_prob=max(no_speech_probs) if no_speech_probs else None,
            avg_logprob=(
                float(np.mean(finite_logprobs)) if finite_logprobs else None
            ),
        )

    def _confidence(self, logprobs: list[float]) -> float:
        """Turns per-segment average log-probs into a `[0, 1]` confidence.

        `exp(mean logprob)`, matching what the HF backend derives from token
        scores, so the number means the same thing whichever backend
        produced it. Returns 0.0 when there is nothing to score, so a
        missing signal reads as "no confidence" rather than as certainty.
        """
        finite = [value for value in logprobs if np.isfinite(value)]
        if not finite:
            return 0.0
        confidence = float(np.exp(np.mean(finite)))
        return max(0.0, min(1.0, confidence))
