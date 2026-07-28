#!/usr/bin/env python3
"""Tests for the faster-whisper backend's model handling (EPIC 3.5).

Only the *backend* is tested here. The gRPC contract — streaming, cancel
delivery, audio conditioning, error propagation — is covered once for every
backend in `python/marceline_worker/tests`, which is the point of sharing
that service.

`faster_whisper` itself is stubbed into `sys.modules`, so these run without
CTranslate2 or model weights installed. That means they check the glue this
file actually owns (lazy generator consumption, cancel between segments,
confidence math) rather than re-testing the library.

Run from this worker's directory:

    .venv/bin/python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class FakeSegment:
    """Stands in for a `faster_whisper` segment."""

    text: str
    avg_logprob: float
    #: This runtime reports a per-segment no-speech probability, which the
    #: Rust guard gates hallucinations on (EPIC 3.6).
    no_speech_prob: float = 0.01


class FakeWhisperModel:
    """Scriptable stand-in for `faster_whisper.WhisperModel`.

    Yields its segments lazily, like the real thing, so a test can observe
    what happens when cancel fires partway through decoding.
    """

    #: Constructor args of the most recently built model.
    last_init: dict | None = None

    def __init__(self, model, device=None, compute_type=None):
        FakeWhisperModel.last_init = {
            "model": model,
            "device": device,
            "compute_type": compute_type,
        }
        self.segments: list[FakeSegment] = []
        #: Set by a test to fire mid-iteration, emulating a cancel arriving
        #: while the generator is still producing segments.
        self.cancel_after: int | None = None
        self.cancel_event: threading.Event | None = None
        #: How many segments the generator actually produced.
        self.yielded = 0
        self.transcribe_kwargs: dict = {}

    def transcribe(self, pcm, **kwargs):
        self.transcribe_kwargs = kwargs

        def generate():
            for index, segment in enumerate(self.segments):
                if (
                    self.cancel_after is not None
                    and self.cancel_event is not None
                    and index == self.cancel_after
                ):
                    self.cancel_event.set()
                self.yielded += 1
                yield segment

        return generate(), types.SimpleNamespace(language="en")


# Install the stub before importing the backend, which imports the library
# at module scope (as the real worker process does).
fake_module = types.ModuleType("faster_whisper")
fake_module.WhisperModel = FakeWhisperModel
sys.modules["faster_whisper"] = fake_module

from faster_whisper_backend import FasterWhisperBackend  # noqa: E402


def loaded_backend(segments: list[FakeSegment], device: str = "cuda"):
    """Returns a loaded backend and its underlying fake model."""
    backend = FasterWhisperBackend("large-v3", device, "en")
    backend.load()
    model = backend._model  # noqa: SLF001 - the seam under test
    model.segments = segments
    return backend, model


class FasterWhisperBackendTest(unittest.TestCase):
    def test_reports_a_backend_qualified_name(self) -> None:
        """A swap has to be visible: both backends can load the same weights."""
        backend = FasterWhisperBackend("large-v3", "cuda", "en")
        self.assertEqual(backend.name, "faster-whisper:large-v3")
        self.assertEqual(backend.sample_rate, 16_000)

    def test_passes_the_model_id_through_unchanged(self) -> None:
        """Unlike the HF backend, no `openai/` prefix is added.

        Model naming is a backend concern, which is what lets `[stt].model`
        mean the same thing to a user across both backends.
        """
        backend, _ = loaded_backend([])
        self.assertEqual(FakeWhisperModel.last_init["model"], "large-v3")
        self.assertEqual(backend.name, "faster-whisper:large-v3")

    def test_uses_float16_on_cuda_and_int8_on_cpu(self) -> None:
        loaded_backend([], device="cuda")
        self.assertEqual(FakeWhisperModel.last_init["compute_type"], "float16")
        loaded_backend([], device="cpu")
        self.assertEqual(FakeWhisperModel.last_init["compute_type"], "int8")

    def test_joins_segments_and_trims(self) -> None:
        backend, _ = loaded_backend(
            [FakeSegment(" marceline", -0.1), FakeSegment(" what time is it", -0.3)]
        )
        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), threading.Event())

        self.assertEqual(result.text, "marceline what time is it")
        self.assertFalse(result.cancelled)
        # exp(mean(-0.1, -0.3)) = exp(-0.2)
        self.assertAlmostEqual(result.confidence, float(np.exp(-0.2)), places=5)

    def test_pins_the_configured_language(self) -> None:
        """Detection misfires on short noisy segments, and v1 is English."""
        backend, model = loaded_backend([FakeSegment("hi", -0.1)])
        backend.transcribe(np.zeros(1_600, dtype=np.float32), threading.Event())

        self.assertEqual(model.transcribe_kwargs["language"], "en")
        self.assertEqual(model.transcribe_kwargs["task"], "transcribe")

    def test_cancel_before_decoding_returns_immediately(self) -> None:
        backend, model = loaded_backend([FakeSegment("hi", -0.1)])
        cancel = threading.Event()
        cancel.set()

        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), cancel)

        self.assertTrue(result.cancelled)
        self.assertEqual(result.text, "")
        self.assertEqual(model.yielded, 0, "no decoding should have happened")

    def test_cancel_mid_decode_stops_early_and_commits_nothing(self) -> None:
        """The §2.5.1 check, which is a generator break for this backend.

        A truncated fragment is not a committed transcript, so a cancelled
        decode must yield no text at all rather than half an utterance.
        """
        cancel = threading.Event()
        backend, model = loaded_backend(
            [
                FakeSegment("one", -0.1),
                FakeSegment("two", -0.1),
                FakeSegment("three", -0.1),
            ]
        )
        model.cancel_event = cancel
        model.cancel_after = 1  # fires while producing the second segment

        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), cancel)

        self.assertTrue(result.cancelled)
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertLess(
            model.yielded,
            len(model.segments),
            "decoding should have stopped before the last segment",
        )

    def test_reports_guard_signals(self) -> None:
        """Both signals reach the caller for the Rust guard (EPIC 3.6)."""
        backend, _ = loaded_backend(
            [
                FakeSegment("one", -0.2, no_speech_prob=0.05),
                FakeSegment("two", -0.4, no_speech_prob=0.30),
            ]
        )
        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), threading.Event())

        # The *worst* of each: one suspicious segment is enough to be worth
        # flagging, and averaging would let confident neighbours hide it.
        self.assertAlmostEqual(result.no_speech_prob, 0.30, places=5)
        self.assertAlmostEqual(result.avg_logprob, -0.3, places=5)

    def test_signals_are_none_when_nothing_was_decoded(self) -> None:
        """Absent stays absent rather than becoming a confident zero."""
        backend, _ = loaded_backend([])
        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), threading.Event())

        self.assertIsNone(result.no_speech_prob)
        self.assertIsNone(result.avg_logprob)

    def test_confidence_is_zero_when_there_is_no_signal(self) -> None:
        """Missing scores read as no confidence, never as certainty."""
        backend, _ = loaded_backend([])
        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), threading.Event())

        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)

    def test_confidence_ignores_non_finite_logprobs(self) -> None:
        backend, _ = loaded_backend(
            [FakeSegment("a", float("-inf")), FakeSegment("b", -0.2)]
        )
        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), threading.Event())

        self.assertAlmostEqual(result.confidence, float(np.exp(-0.2)), places=5)

    def test_confidence_is_clamped_to_a_probability(self) -> None:
        backend, _ = loaded_backend([FakeSegment("a", 5.0)])
        result = backend.transcribe(np.zeros(1_600, dtype=np.float32), threading.Event())

        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_transcribe_before_load_is_an_error(self) -> None:
        backend = FasterWhisperBackend("large-v3", "cuda", "en")
        with self.assertRaises(RuntimeError):
            backend.transcribe(np.zeros(16, dtype=np.float32), threading.Event())


if __name__ == "__main__":
    unittest.main()
