#!/usr/bin/env python3
"""Tests for the Piper backend's model handling (EPIC 5.5).

Only the *backend* is tested here. The gRPC contract — streaming, cancel
delivery, error propagation — is covered once for every TTS backend in
`python/marceline_worker/tests`, which is the point of sharing that
service.

`piper` itself is stubbed into `sys.modules`, so these run without
onnxruntime or model weights installed. That means they check the glue
this file actually owns (raw-frame consumption, cancel between frames,
int16-to-f32 normalization, sample-rate passthrough) rather than
re-testing the library.

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


class FakeConfig:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate


class FakePiperVoice:
    """Scriptable stand-in for `piper.PiperVoice`.

    Yields its raw int16 frames lazily, like the real thing, so a test can
    observe what happens when cancel fires partway through generation.
    """

    #: Constructor args of the most recently loaded voice.
    last_load: dict | None = None

    def __init__(self, sample_rate: int = 22_050) -> None:
        self.config = FakeConfig(sample_rate)
        self.frames: list[list[int]] = []
        #: Set by a test to fire mid-iteration, emulating a cancel arriving
        #: while the generator is still producing frames.
        self.cancel_after: int | None = None
        self.cancel_event: threading.Event | None = None
        #: How many frames the generator actually produced.
        self.yielded = 0
        self.last_text: str | None = None

    @classmethod
    def load(cls, model_path, use_cuda=False):
        FakePiperVoice.last_load = {"model_path": model_path, "use_cuda": use_cuda}
        return cls()

    def synthesize_stream_raw(self, text):
        self.last_text = text

        def generate():
            for index, samples in enumerate(self.frames):
                if (
                    self.cancel_after is not None
                    and self.cancel_event is not None
                    and index == self.cancel_after
                ):
                    self.cancel_event.set()
                self.yielded += 1
                yield np.array(samples, dtype=np.int16).tobytes()

        return generate()


def _install_fake_piper() -> type:
    """Stubs `piper.PiperVoice` into `sys.modules` and returns the class."""
    module = types.ModuleType("piper")
    module.PiperVoice = FakePiperVoice
    sys.modules["piper"] = module
    return FakePiperVoice


class PiperBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        _install_fake_piper()
        sys.modules.pop("piper_backend", None)
        global piper_backend
        import piper_backend  # noqa: PLC0415

        self.mod = piper_backend

    def test_resolve_model_path_uses_an_existing_file_as_is(self) -> None:
        this_file = os.path.abspath(__file__)
        self.assertEqual(self.mod.resolve_model_path(this_file), this_file)

    def test_resolve_model_path_falls_back_to_voices_dir(self) -> None:
        resolved = self.mod.resolve_model_path("en_US-lessac-medium")
        self.assertTrue(resolved.endswith("voices/en_US-lessac-medium.onnx"))

    def test_load_builds_voice_from_resolved_path_and_device(self) -> None:
        backend = self.mod.PiperBackend("/models/voice.onnx", "voice-id", "cuda")
        backend.load()

        self.assertEqual(
            FakePiperVoice.last_load,
            {"model_path": "/models/voice.onnx", "use_cuda": True},
        )

    def test_sample_rate_reflects_the_loaded_model(self) -> None:
        backend = self.mod.PiperBackend("/models/voice.onnx", "voice-id", "cpu")
        backend.load()
        backend._voice.config.sample_rate = 16_000

        self.assertEqual(backend.sample_rate, 16_000)

    def test_synthesize_yields_normalized_f32_per_frame(self) -> None:
        backend = self.mod.PiperBackend("/models/voice.onnx", "voice-id", "cpu")
        backend.load()
        fake_voice = backend._voice
        fake_voice.frames = [[16_384, -16_384], [0, 32_767]]

        chunks = list(backend.synthesize("hi there", "voice-id", threading.Event()))

        self.assertEqual(len(chunks), 2)
        self.assertTrue(np.allclose(chunks[0], [0.5, -0.5], atol=1e-4))
        self.assertTrue(np.allclose(chunks[1], [0.0, 0.999969], atol=1e-4))
        for chunk in chunks:
            self.assertEqual(chunk.dtype, np.float32)
        self.assertEqual(fake_voice.last_text, "hi there")

    def test_cancel_between_frames_stops_generation_early(self) -> None:
        backend = self.mod.PiperBackend("/models/voice.onnx", "voice-id", "cpu")
        backend.load()
        fake_voice = backend._voice
        fake_voice.frames = [[100], [200], [300]]

        cancel = threading.Event()
        fake_voice.cancel_after = 1
        fake_voice.cancel_event = cancel

        chunks = list(backend.synthesize("a long sentence", "voice-id", cancel))

        # The cancel fires as frame index 1 starts; only the first
        # (already-yielded) chunk should reach the caller.
        self.assertEqual(len(chunks), 1)

    def test_synthesize_before_load_raises(self) -> None:
        backend = self.mod.PiperBackend("/models/voice.onnx", "voice-id", "cpu")
        with self.assertRaises(RuntimeError):
            list(backend.synthesize("hi", "voice-id", threading.Event()))

    def test_mismatched_voice_is_ignored_rather_than_failing(self) -> None:
        backend = self.mod.PiperBackend("/models/voice.onnx", "voice-id", "cpu")
        backend.load()
        backend._voice.frames = [[100]]

        # Must not raise even though "other-voice" != "voice-id".
        chunks = list(backend.synthesize("hi", "other-voice", threading.Event()))
        self.assertEqual(len(chunks), 1)

    def test_reports_the_single_loaded_voice(self) -> None:
        backend = self.mod.PiperBackend("/models/voice.onnx", "voice-id", "cpu")
        self.assertEqual(backend.voices, ("voice-id",))
        self.assertEqual(backend.name, "piper:voice-id")


if __name__ == "__main__":
    unittest.main()
