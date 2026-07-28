#!/usr/bin/env python3
"""Tests for the shared TTS worker gRPC surface (EPIC 5.1, 5.5).

These exercise the real gRPC stack over a real unix domain socket — the
transport is half the story of this worker, so stubbing it out would test
nothing interesting. What *is* stubbed is the model: a fake backend stands
in for Kokoro so the suite runs in milliseconds, needs no GPU, and needs
neither kokoro nor torch installed.

Because this service is shared by every TTS backend, these tests cover the
contract for all of them; a backend's own tests only need to cover its
model handling.

Run from a worker directory that has a venv:

    .venv/bin/python -m unittest discover -s ../../python/marceline_worker/tests
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from concurrent import futures

import grpc
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from marceline_protocol import common_pb2, tts_pb2, tts_pb2_grpc  # noqa: E402

from marceline_worker.tts_service import TtsServicer  # noqa: E402

# Sample rate the fake backend claims to emit, matching real Kokoro.
SAMPLE_RATE = 24_000


class FakeBackend:
    """Scriptable stand-in for a loaded model backend.

    Records the text/voice it was handed so tests can assert on request
    handling, and can be told to block inside `synthesize` so a cancel
    lands genuinely mid-synthesis.
    """

    name = "fake:test-model"
    sample_rate = SAMPLE_RATE
    voices = ("af_test", "am_test")

    def __init__(self, chunks_per_call: int = 1, block: bool = False) -> None:
        self._chunks_per_call = chunks_per_call
        #: When True, `synthesize` waits for the cancel flag before
        #: yielding its second chunk, emulating a slow generate loop.
        self._block = block
        #: (text, voice) for every call, in call order.
        self.calls: list[tuple[str, str]] = []
        #: Set once `synthesize` has been entered.
        self.entered = threading.Event()
        #: True if a blocking call observed the cancel flag rather than
        #: running to completion — i.e. cancel actually interrupted synthesis.
        self.observed_cancel = False

    def synthesize(self, text: str, voice: str, cancel: threading.Event):
        self.calls.append((text, voice))
        self.entered.set()
        for i in range(self._chunks_per_call):
            if cancel.is_set():
                self.observed_cancel = True
                return
            if self._block and i == 1:
                self.observed_cancel = cancel.wait(timeout=5.0)
                if self.observed_cancel:
                    return
            yield np.full(4, 0.1 * (i + 1), dtype=np.float32)


def text_request(text: str):
    """Builds a `TtsRequest` carrying a text span."""
    return tts_pb2.TtsRequest(text=text)


def voice_request(voice: str):
    """Builds a `TtsRequest` selecting a voice."""
    return tts_pb2.TtsRequest(voice=voice)


class WorkerHarness:
    """Runs a `TtsServicer` on a real UDS gRPC server for one test."""

    def __init__(self, backend: FakeBackend, default_voice: str = "af_test") -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self._dir.name, "tts.sock")
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        tts_pb2_grpc.add_TtsServicer_to_server(
            TtsServicer(backend, default_voice), self._server
        )
        self._server.add_insecure_port(f"unix://{self.socket_path}")
        self._server.start()
        self._channel = grpc.insecure_channel(f"unix://{self.socket_path}")
        self.stub = tts_pb2_grpc.TtsStub(self._channel)

    def close(self) -> None:
        self._channel.close()
        self._server.stop(grace=None).wait()
        self._dir.cleanup()


class TtsWorkerTest(unittest.TestCase):
    def harness(self, backend: FakeBackend, **kwargs: object) -> WorkerHarness:
        harness = WorkerHarness(backend, **kwargs)
        self.addCleanup(harness.close)
        return harness

    def test_synthesizes_a_text_span_to_audio_chunks(self) -> None:
        """The `Done when` case: text in, streamed audio chunks back."""
        backend = FakeBackend(chunks_per_call=3)
        stub = self.harness(backend).stub

        responses = list(stub.Synthesize(iter([text_request("hello there")])))

        self.assertEqual(len(responses), 3)
        self.assertEqual([r.audio.seq for r in responses], [0, 1, 2])
        self.assertEqual(backend.calls, [("hello there", "af_test")])
        for r in responses:
            self.assertEqual(r.audio.sample_rate, SAMPLE_RATE)
            self.assertEqual(r.audio.channels, 1)

    def test_uses_default_voice_when_none_selected(self) -> None:
        backend = FakeBackend()
        stub = self.harness(backend, default_voice="am_test").stub

        list(stub.Synthesize(iter([text_request("hi")])))

        self.assertEqual(backend.calls, [("hi", "am_test")])

    def test_voice_message_selects_the_voice_for_the_stream(self) -> None:
        backend = FakeBackend()
        stub = self.harness(backend, default_voice="af_test").stub

        requests = [voice_request("am_test"), text_request("hi")]
        list(stub.Synthesize(iter(requests)))

        self.assertEqual(backend.calls, [("hi", "am_test")])

    def test_multiple_text_spans_stream_sequenced_chunks(self) -> None:
        backend = FakeBackend(chunks_per_call=2)
        stub = self.harness(backend).stub

        requests = [text_request("first"), text_request("second")]
        responses = list(stub.Synthesize(iter(requests)))

        self.assertEqual(len(responses), 4)
        self.assertEqual([r.audio.seq for r in responses], [0, 1, 2, 3])
        self.assertEqual(backend.calls, [("first", "af_test"), ("second", "af_test")])

    def test_cancel_mid_synthesis_stops_generate_loop_early(self) -> None:
        """A `Cancel` sent while synthesis runs reaches the generate loop.

        The fake backend blocks before its second chunk until it sees the
        cancel flag, so this only passes if the request stream is being
        read concurrently with synthesis.
        """
        backend = FakeBackend(chunks_per_call=3, block=True)
        stub = self.harness(backend).stub

        cancel_sent = threading.Event()

        def requests():
            yield text_request("a long sentence that takes a while")
            self.assertTrue(backend.entered.wait(timeout=5.0))
            cancel_sent.set()
            yield tts_pb2.TtsRequest(cancel=common_pb2.Cancel())

        responses = list(stub.Synthesize(requests()))

        self.assertTrue(cancel_sent.is_set())
        self.assertTrue(
            backend.observed_cancel, "cancel flag never reached the generate loop"
        )
        # Cancel lands before the second chunk, so only the first is seen.
        self.assertEqual(len(responses), 1)

    def test_backend_failure_surfaces_in_band(self) -> None:
        """A worker-side failure reaches the client as a stream error."""

        class ExplodingBackend(FakeBackend):
            def synthesize(self, text, voice, cancel):  # type: ignore[override]
                raise RuntimeError("model exploded")
                yield  # pragma: no cover - makes this a generator

        stub = self.harness(ExplodingBackend()).stub
        with self.assertRaises(grpc.RpcError) as caught:
            list(stub.Synthesize(iter([text_request("hi")])))

        self.assertEqual(caught.exception.code(), grpc.StatusCode.INTERNAL)
        self.assertIn("model exploded", caught.exception.details())

    def test_get_info_advertises_voices_and_sample_rate(self) -> None:
        stub = self.harness(FakeBackend()).stub
        info = stub.GetInfo(tts_pb2.TtsInfoRequest())

        self.assertEqual(info.name, "fake:test-model")
        self.assertEqual(list(info.voices), ["af_test", "am_test"])
        self.assertEqual(info.output_sample_rate, SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
