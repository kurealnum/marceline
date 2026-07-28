#!/usr/bin/env python3
"""Tests for the Kokoro backend's model handling (EPIC 5.1).

Only the *backend* is tested here. The gRPC contract — streaming, cancel
delivery, error propagation — is covered once for every backend in
`python/marceline_worker/tests`, which is the point of sharing that
service.

`kokoro` itself is stubbed into `sys.modules`, so these run without torch
or model weights installed. That means they check the glue this file
actually owns (lazy generator consumption, cancel between sub-utterances)
rather than re-testing the library.

Run from this worker's directory:

    .venv/bin/python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeAudio:
    """Stands in for the torch tensor Kokoro yields per sub-utterance."""

    def __init__(self, value: float) -> None:
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.full(4, self._value, dtype=np.float32)


class FakeKPipeline:
    """Scriptable stand-in for `kokoro.KPipeline`.

    Yields its sub-utterances lazily, like the real thing, so a test can
    observe what happens when cancel fires partway through generation.
    """

    #: Constructor args of the most recently built pipeline.
    last_init: dict | None = None

    def __init__(self, lang_code=None, device=None):
        FakeKPipeline.last_init = {"lang_code": lang_code, "device": device}
        self.sub_utterances: list[float] = []
        #: Set by a test to fire mid-iteration, emulating a cancel arriving
        #: while the generator is still producing sub-utterances.
        self.cancel_after: int | None = None
        self.cancel_event: threading.Event | None = None
        #: How many sub-utterances the generator actually produced.
        self.yielded = 0
        self.call_kwargs: dict = {}

    def __call__(self, text, **kwargs):
        self.call_kwargs = kwargs
        self.last_text = text

        def generate():
            for index, value in enumerate(self.sub_utterances):
                if (
                    self.cancel_after is not None
                    and self.cancel_event is not None
                    and index == self.cancel_after
                ):
                    self.cancel_event.set()
                self.yielded += 1
                yield (None, None, FakeAudio(value))

        return generate()


def _install_fake_kokoro() -> FakeKPipeline:
    """Stubs `kokoro.KPipeline` into `sys.modules` and returns the class."""
    module = types.ModuleType("kokoro")
    module.KPipeline = FakeKPipeline
    sys.modules["kokoro"] = module
    return FakeKPipeline


class KokoroBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        _install_fake_kokoro()
        # Reload so `from kokoro import KPipeline` binds to the fake.
        sys.modules.pop("kokoro_backend", None)
        global kokoro_backend
        import kokoro_backend  # noqa: PLC0415

        self.mod = kokoro_backend

    def test_load_builds_pipeline_on_configured_device(self) -> None:
        backend = self.mod.KokoroBackend("cpu")
        backend.load()

        self.assertEqual(
            FakeKPipeline.last_init, {"lang_code": "a", "device": "cpu"}
        )

    def test_synthesize_yields_pcm_per_sub_utterance(self) -> None:
        backend = self.mod.KokoroBackend("cpu")
        backend.load()
        fake_pipeline = backend._pipeline
        fake_pipeline.sub_utterances = [0.1, 0.2, 0.3]

        chunks = list(backend.synthesize("hi there", "af_sky", threading.Event()))

        self.assertEqual(len(chunks), 3)
        for chunk, value in zip(chunks, [0.1, 0.2, 0.3]):
            self.assertTrue(np.allclose(chunk, value))
            self.assertEqual(chunk.dtype, np.float32)
        self.assertEqual(fake_pipeline.call_kwargs, {"voice": "af_sky"})

    def test_cancel_between_sub_utterances_stops_generation_early(self) -> None:
        backend = self.mod.KokoroBackend("cpu")
        backend.load()
        fake_pipeline = backend._pipeline
        fake_pipeline.sub_utterances = [0.1, 0.2, 0.3]

        cancel = threading.Event()
        fake_pipeline.cancel_after = 1
        fake_pipeline.cancel_event = cancel

        chunks = list(backend.synthesize("a long sentence", "af_sky", cancel))

        # The cancel fires as sub-utterance index 1 starts; only the first
        # (already-yielded) chunk should reach the caller.
        self.assertEqual(len(chunks), 1)

    def test_synthesize_before_load_raises(self) -> None:
        backend = self.mod.KokoroBackend("cpu")
        with self.assertRaises(RuntimeError):
            list(backend.synthesize("hi", "af_sky", threading.Event()))

    def test_reports_fixed_voice_set_and_sample_rate(self) -> None:
        backend = self.mod.KokoroBackend("cpu")
        self.assertIn("af_sky", backend.voices)
        self.assertEqual(backend.sample_rate, 24_000)
        self.assertEqual(backend.name, "kokoro:82M")


if __name__ == "__main__":
    unittest.main()
