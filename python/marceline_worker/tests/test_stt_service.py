#!/usr/bin/env python3
"""Tests for the shared STT worker gRPC surface (EPIC 3.1, 3.5).

These exercise the real gRPC stack over a real unix domain socket — the
transport is half the story of this worker, so stubbing it out would test
nothing interesting. What *is* stubbed is the model: a fake backend stands
in for Whisper so the suite runs in milliseconds, needs no GPU, and needs
neither torch nor transformers installed.

Because this service is shared by every STT backend, these tests cover the
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
from dataclasses import dataclass

import grpc
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from marceline_protocol import common_pb2, stt_pb2, stt_pb2_grpc  # noqa: E402

from marceline_worker.stt_service import SttServicer  # noqa: E402

# Sample rate the fake backend claims to want, matching real Whisper.
SAMPLE_RATE = 16_000


@dataclass
class FakeTranscription:
    """Stand-in for `whisper_backend.Transcription`."""

    text: str
    confidence: float
    cancelled: bool = False


class FakeBackend:
    """Scriptable stand-in for a loaded model backend.

    Records the audio it was handed so tests can assert on conditioning
    (downmix, resample, buffering), and can be told to block inside
    `transcribe` so a cancel lands genuinely mid-inference.
    """

    name = "fake:test-model"
    sample_rate = SAMPLE_RATE

    def __init__(self, text: str = "hello there", block: bool = False) -> None:
        self._text = text
        #: When True, `transcribe` waits for the cancel flag instead of
        #: returning immediately, emulating a slow generate loop.
        self._block = block
        #: Every pcm array passed to `transcribe`, in call order.
        self.calls: list[np.ndarray] = []
        #: Set once `transcribe` has been entered.
        self.entered = threading.Event()
        #: True if a blocking call observed the cancel flag rather than
        #: timing out — i.e. cancel actually interrupted inference.
        self.observed_cancel = False

    def transcribe(self, pcm: np.ndarray, cancel: threading.Event) -> FakeTranscription:
        self.calls.append(pcm)
        self.entered.set()
        if self._block:
            # Poll like a generate loop checking between decode steps.
            self.observed_cancel = cancel.wait(timeout=5.0)
            if self.observed_cancel:
                return FakeTranscription(text="", confidence=0.0, cancelled=True)
        if cancel.is_set():
            return FakeTranscription(text="", confidence=0.0, cancelled=True)
        return FakeTranscription(text=self._text, confidence=0.75)


def audio_request(pcm: list[float], seq: int, rate: int = SAMPLE_RATE, channels: int = 1):
    """Builds an `SttRequest` carrying one audio chunk."""
    return stt_pb2.SttRequest(
        audio=common_pb2.AudioChunk(
            seq=seq, pcm=pcm, sample_rate=rate, channels=channels
        )
    )


class WorkerHarness:
    """Runs an `SttServicer` on a real UDS gRPC server for one test."""

    def __init__(self, backend: FakeBackend, lang: str = "en") -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self._dir.name, "stt.sock")
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        stt_pb2_grpc.add_SttServicer_to_server(
            SttServicer(backend, lang), self._server
        )
        self._server.add_insecure_port(f"unix://{self.socket_path}")
        self._server.start()
        self._channel = grpc.insecure_channel(f"unix://{self.socket_path}")
        self.stub = stt_pb2_grpc.SttStub(self._channel)

    def close(self) -> None:
        self._channel.close()
        self._server.stop(grace=None).wait()
        self._dir.cleanup()


class SttWorkerTest(unittest.TestCase):
    def harness(self, backend: FakeBackend, **kwargs: object) -> WorkerHarness:
        harness = WorkerHarness(backend, **kwargs)
        self.addCleanup(harness.close)
        return harness

    def test_transcribes_a_fed_segment_to_one_final(self) -> None:
        """The `Done when` case: audio in, a single `final` back."""
        backend = FakeBackend(text="marceline what time is it")
        stub = self.harness(backend).stub

        requests = [audio_request([0.1] * 1600, seq=i) for i in range(3)]
        responses = list(stub.Transcribe(iter(requests)))

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].WhichOneof("transcript"), "final")
        self.assertEqual(responses[0].final.text, "marceline what time is it")
        self.assertAlmostEqual(responses[0].final.confidence, 0.75, places=5)
        # All three chunks land in one transcribe call, not three.
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0].size, 4800)

    def test_never_emits_partials(self) -> None:
        """v1 is final-only; a `partial` must never appear on the stream."""
        stub = self.harness(FakeBackend()).stub
        responses = list(stub.Transcribe(iter([audio_request([0.0] * 800, seq=0)])))
        self.assertTrue(
            all(r.WhichOneof("transcript") == "final" for r in responses)
        )

    def test_cancel_mid_inference_stops_generate_loop_early(self) -> None:
        """A `Cancel` sent while inference runs reaches the generate loop.

        The fake backend blocks inside `transcribe` until it sees the
        cancel flag, so this only passes if the request stream is being
        read concurrently with inference.
        """
        backend = FakeBackend(block=True)
        stub = self.harness(backend).stub

        cancel_sent = threading.Event()

        def requests():
            yield audio_request([0.2] * SAMPLE_RATE * 30, seq=0)  # fills the window
            # Wait until inference has actually started, so the cancel is
            # mid-inference rather than merely queued ahead of it.
            self.assertTrue(backend.entered.wait(timeout=5.0))
            cancel_sent.set()
            yield stt_pb2.SttRequest(cancel=common_pb2.Cancel())

        responses = list(stub.Transcribe(requests()))

        self.assertTrue(cancel_sent.is_set())
        self.assertTrue(
            backend.observed_cancel, "cancel flag never reached the generate loop"
        )
        # A cancelled decode yields a truncated fragment, which must not
        # be committed as a transcript.
        self.assertEqual(responses, [])

    def test_flushes_at_the_whisper_window_boundary(self) -> None:
        """Audio past 30s is transcribed as a second segment, not dropped."""
        backend = FakeBackend()
        stub = self.harness(backend).stub

        requests = [
            audio_request([0.1] * (SAMPLE_RATE * 30), seq=0),
            audio_request([0.1] * (SAMPLE_RATE * 5), seq=1),
        ]
        responses = list(stub.Transcribe(iter(requests)))

        self.assertEqual(len(responses), 2)
        self.assertEqual(
            [call.size for call in backend.calls], [SAMPLE_RATE * 30, SAMPLE_RATE * 5]
        )

    def test_splits_a_chunk_that_overshoots_the_window(self) -> None:
        """A single over-long chunk is split, not truncated by the model.

        Whisper silently discards anything past its 30-second window, so
        a 31-second chunk must become a 30s segment plus a 1s remainder
        rather than one 31s call that loses the tail.
        """
        backend = FakeBackend()
        stub = self.harness(backend).stub

        requests = [audio_request([0.1] * (SAMPLE_RATE * 31), seq=0)]
        responses = list(stub.Transcribe(iter(requests)))

        self.assertEqual(len(responses), 2)
        self.assertEqual(
            [call.size for call in backend.calls], [SAMPLE_RATE * 30, SAMPLE_RATE]
        )

    def test_transcribes_exactly_one_window_without_an_empty_tail(self) -> None:
        """Audio landing exactly on the boundary yields one segment."""
        backend = FakeBackend()
        stub = self.harness(backend).stub

        requests = [audio_request([0.1] * (SAMPLE_RATE * 30), seq=0)]
        responses = list(stub.Transcribe(iter(requests)))

        self.assertEqual(len(responses), 1)
        self.assertEqual([call.size for call in backend.calls], [SAMPLE_RATE * 30])

    def test_downmixes_and_resamples_to_the_backend_rate(self) -> None:
        """Off-format audio is conditioned, not assumed (invariant 2)."""
        backend = FakeBackend()
        stub = self.harness(backend).stub

        # 1 second of interleaved stereo at 48 kHz -> 16 kHz mono.
        stereo = [0.5, -0.5] * 48_000
        requests = [audio_request(stereo, seq=0, rate=48_000, channels=2)]
        list(stub.Transcribe(iter(requests)))

        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0].size, SAMPLE_RATE)
        # Averaging +0.5/-0.5 channels cancels to silence.
        self.assertTrue(np.allclose(backend.calls[0], 0.0, atol=1e-6))

    def test_rejects_format_change_mid_stream(self) -> None:
        """A rate change mid-stream is an error, not a silent bad transcript."""
        stub = self.harness(FakeBackend()).stub

        requests = [
            audio_request([0.1] * 1600, seq=0, rate=16_000),
            audio_request([0.1] * 1600, seq=1, rate=48_000),
        ]
        with self.assertRaises(grpc.RpcError) as caught:
            list(stub.Transcribe(iter(requests)))

        self.assertEqual(caught.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("changed mid-stream", caught.exception.details())

    def test_backend_failure_surfaces_in_band(self) -> None:
        """A worker-side failure reaches the client as a stream error."""

        class ExplodingBackend(FakeBackend):
            def transcribe(self, pcm, cancel):  # type: ignore[override]
                raise RuntimeError("CUDA out of memory")

        stub = self.harness(ExplodingBackend()).stub
        with self.assertRaises(grpc.RpcError) as caught:
            list(stub.Transcribe(iter([audio_request([0.1] * 1600, seq=0)])))

        self.assertEqual(caught.exception.code(), grpc.StatusCode.INTERNAL)
        self.assertIn("CUDA out of memory", caught.exception.details())

    def test_get_info_advertises_final_only(self) -> None:
        """`SttInfo.partials` is false for chunk-based Whisper (§2.4.1)."""
        stub = self.harness(FakeBackend(), lang="en").stub
        info = stub.GetInfo(stt_pb2.SttInfoRequest())

        self.assertEqual(info.name, "fake:test-model")
        self.assertEqual(list(info.langs), ["en"])
        self.assertEqual(info.input_sample_rate, SAMPLE_RATE)
        self.assertFalse(info.partials)


if __name__ == "__main__":
    unittest.main()
