#!/usr/bin/env python3
"""HuggingFace `whisper` STT backend (SPEC.md §2.4.1, EPIC 3.1).

This is the default STT backend: a Whisper-family model loaded from the
HuggingFace hub, run on the device the Rust core selected (EPIC 0.7).

Whisper is *chunk-based*: it consumes a padded 30-second window and
returns one committed transcript for it. There is no meaningful
token-by-token "live partial" to emit, so this backend advertises
`partials = False` and only ever produces final text (§2.4.1). Faking
partials by re-decoding prefixes would be slower and would leak
half-words into the LLM prompt.

`torch`/`transformers` are imported at module import time, so
`worker.py` imports this module lazily — the gRPC plumbing and its tests
do not need a multi-gigabyte ML stack installed.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

log = logging.getLogger("marceline.stt.whisper")

# Sample rate every Whisper-family checkpoint expects. Audio arriving at
# any other rate is resampled by the caller before it reaches this
# backend.
WHISPER_SAMPLE_RATE = 16_000

# Model ids in config (`[stt].model`) are short names like "large-v3".
# Expand those to the canonical hub repo; a value that already looks like
# a hub id (contains "/") is passed through untouched, so any Whisper
# fine-tune on the hub works without a code change.
_HUB_PREFIX = "openai/whisper-"


@dataclass(frozen=True)
class Transcription:
    """One committed transcript produced from a single audio segment."""

    #: Recognized text, whitespace-trimmed. Empty when decoding was
    #: cancelled before producing anything.
    text: str
    #: Model confidence in `[0, 1]`, derived from mean token log-prob.
    confidence: float
    #: True when decoding stopped early because of a cooperative cancel.
    cancelled: bool = False
    #: Mean per-token log probability, or `None` when scores were
    #: unavailable. Feeds the Rust-side hallucination guard (EPIC 3.6).
    avg_logprob: float | None = None
    #: Always `None` here. `generate()` does not surface a no-speech
    #: probability, and inferring one from the `<|nospeech|>` token logit
    #: would be a guess dressed up as a measurement — the field is
    #: `optional` on the wire so the guard can see it is genuinely
    #: unavailable and lean on duration and log-prob instead. The
    #: `faster-whisper` backend does report it.
    no_speech_prob: float | None = None


class _CancelStoppingCriteria(StoppingCriteria):
    """Stops `generate()` between decode steps when a cancel flag is set.

    This is the teeth behind the explicit in-band `Cancel` message
    (§2.5.1): socket close is not a reliable stop signal mid-GPU-op, so
    the generate loop polls this flag instead of relying on transport
    state.
    """

    def __init__(self, cancel: threading.Event) -> None:
        self._cancel = cancel

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: object
    ) -> torch.BoolTensor:
        stop = self._cancel.is_set()
        return torch.full(
            (input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device
        )


def resolve_model_id(model: str) -> str:
    """Expands a config model name into a HuggingFace hub repo id.

    `"large-v3"` becomes `"openai/whisper-large-v3"`; anything that already
    names a repo (`"distil-whisper/distil-large-v3"`) is left alone.
    """
    return model if "/" in model else f"{_HUB_PREFIX}{model}"


class WhisperBackend:
    """Wraps a loaded HF Whisper model behind the worker's backend seam.

    `worker.py` only ever calls [`load`][WhisperBackend.load],
    [`transcribe`][WhisperBackend.transcribe] and
    [`name`][WhisperBackend.name], so a second backend (`faster-whisper`,
    EPIC 3.5) drops in by matching those three.
    """

    def __init__(self, model: str, device: str, lang: str) -> None:
        """Records what to load; loading itself happens in `load()`.

        Args:
            model: Model name from `[stt].model`, e.g. `"large-v3"`.
            device: Device name from `[stt].device` (`"cuda"` / `"cpu"`).
            lang: Recognition language from `[stt].lang`, e.g. `"en"`.
        """
        self._model_id = resolve_model_id(model)
        self._device = device
        self._lang = lang
        self._processor: AutoProcessor | None = None
        self._model: AutoModelForSpeechSeq2Seq | None = None

    @property
    def name(self) -> str:
        """Backend-qualified model name reported in `SttInfo.name`."""
        return f"whisper:{self._model_id}"

    @property
    def sample_rate(self) -> int:
        """Sample rate (Hz) this backend requires its input audio in."""
        return WHISPER_SAMPLE_RATE

    def load(self) -> None:
        """Loads processor + weights onto the configured device.

        Raises whatever `transformers` raises on failure (bad model id,
        CUDA OOM). The caller leaves the worker's health status at
        NOT_SERVING in that case, so the supervisor sees a worker that
        never came up rather than one that answers with garbage.
        """
        # fp16 on GPU roughly halves both load time and VRAM for
        # large-v3; CPU kernels for fp16 are poor, so stay fp32 there.
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        log.info("loading %s onto %s (%s)", self._model_id, self._device, dtype)

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(self._device)
        model.eval()
        self._model = model
        log.info("loaded %s", self._model_id)

    def transcribe(self, pcm: np.ndarray, cancel: threading.Event) -> Transcription:
        """Transcribes one mono 16 kHz f32 segment into committed text.

        Args:
            pcm: Mono f32 samples at [`sample_rate`][WhisperBackend.sample_rate].
            cancel: Checked between decode steps; when set, generation
                returns early and the result is marked `cancelled`.

        Returns:
            The committed [`Transcription`][Transcription] for this segment.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("transcribe() called before load()")
        if cancel.is_set():
            return Transcription(text="", confidence=0.0, cancelled=True)

        features = self._processor(
            pcm, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to(self._device, dtype=self._model.dtype)

        # English-only checkpoints (model id ending `.en`, e.g. `small.en`)
        # reject `language`/`task` outright -- there's only one language to
        # pin. Multilingual checkpoints still get it pinned from config
        # rather than detected: v1 is English-only, and letting Whisper
        # guess is a known source of spurious language switches on short,
        # noisy segments.
        generate_kwargs = {}
        if not self._model_id.endswith(".en"):
            generate_kwargs["language"] = self._lang
            generate_kwargs["task"] = "transcribe"

        with torch.inference_mode():
            generated = self._model.generate(
                features,
                stopping_criteria=StoppingCriteriaList(
                    [_CancelStoppingCriteria(cancel)]
                ),
                output_scores=True,
                return_dict_in_generate=True,
                **generate_kwargs,
            )

        cancelled = cancel.is_set()
        text = self._processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )[0].strip()
        avg_logprob = self._avg_logprob(generated)

        if cancelled:
            # Text decoded up to the cancel point is a truncated
            # fragment, not a committed transcript; drop it rather than
            # letting half an utterance reach the LLM.
            log.info("transcription cancelled after %d tokens", generated.sequences.shape[-1])
            return Transcription(text="", confidence=0.0, cancelled=True)

        return Transcription(
            text=text,
            # exp(mean logprob) is the standard sequence-level proxy; 0.0
            # when the log-prob is unavailable, so a missing signal reads as
            # "no confidence" rather than as certainty.
            confidence=(
                max(0.0, min(1.0, float(math.exp(avg_logprob))))
                if avg_logprob is not None
                else 0.0
            ),
            avg_logprob=avg_logprob,
        )

    def _avg_logprob(self, generated: object) -> float | None:
        """Mean per-token log probability, or `None` if unavailable.

        Returns `None` rather than a sentinel number so callers — and the
        Rust-side guard — can tell "the model was unsure" from "we could not
        measure it".
        """
        try:
            scores = self._model.compute_transition_scores(
                generated.sequences, generated.scores, normalize_logits=True
            )
        except Exception:  # noqa: BLE001 - best-effort telemetry
            log.debug("token scores unavailable for this segment", exc_info=True)
            return None

        finite = scores[torch.isfinite(scores)]
        if finite.numel() == 0:
            return None
        return float(finite.mean())
