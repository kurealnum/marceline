#!/usr/bin/env python3
"""Shared STT worker service (SPEC.md §2.2/§2.3/§2.4.1, EPIC 3.1, 3.5).

Every STT worker — the default HF `whisper` one and `faster-whisper` — is
the same gRPC server over the same `marceline.stt.Stt` contract, differing
only in the model behind it. That shared half lives here so the two cannot
drift apart: a backend supplies `name`, `sample_rate` and
`transcribe(pcm, cancel)`, and gets the transport, buffering, audio
conditioning, cancel plumbing, and health service for free.

Sharing it is also what keeps the Rust client unchanged across a backend
swap (EPIC 3.5's actual claim). Two copies of this file would let one
backend quietly start answering differently.

Streaming shape: the client streams `AudioChunk`s for one gate-emitted
segment and half-closes. Audio accumulates until either the 30-second
Whisper window fills or the request stream ends, at which point that buffer
is transcribed and a single `final` is streamed back. A `Cancel` message
interleaved on the request stream is read by a reader thread while
inference is running, and the backend's decode loop notices it between
steps (§2.5.1).
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
from concurrent import futures
from dataclasses import dataclass
from typing import Callable

import grpc
import numpy as np
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from marceline_protocol import stt_pb2, stt_pb2_grpc

from .audio import resample, to_mono

# Name the supervisor's health check asks for.
SERVICE_NAME = "stt"

# Longest audio buffer transcribed in one shot. Whisper's receptive field
# is a padded 30-second window, so a longer buffer would be truncated:
# flushing at the window boundary keeps every sample in some segment.
MAX_SEGMENT_SECONDS = 30.0

# gRPC threads: one Transcribe stream is served at a time in v1 (one mic,
# one conversation), plus headroom for GetInfo/health probes arriving
# while a transcription is in flight.
MAX_WORKERS = 4

log = logging.getLogger("marceline.stt")


@dataclass
class _StreamFormat:
    """Sample rate and channel count claimed by a stream's first chunk."""

    sample_rate: int
    channels: int


class _ProtocolError(Exception):
    """A client sent something inconsistent with the audio it declared.

    Reported to the client as `INVALID_ARGUMENT` mid-stream, so the Rust
    side sees it as a stream `Err` item (invariant 1, §2.4.1) rather than
    as a silently wrong transcript.
    """


class SttServicer(stt_pb2_grpc.SttServicer):
    """Serves `marceline.stt.Stt` against a loaded backend.

    The backend is injected rather than constructed here, which is what
    lets `faster-whisper` (EPIC 3.5) drop in and lets these RPCs be
    tested without an ML stack installed.
    """

    def __init__(self, backend: object, lang: str) -> None:
        """Binds the servicer to a loaded backend.

        Args:
            backend: Loaded backend exposing `name`, `sample_rate` and
                `transcribe(pcm, cancel)` (see `whisper_backend.py`).
            lang: Configured recognition language, reported in `SttInfo`.
        """
        self._backend = backend
        self._lang = lang

    def GetInfo(  # noqa: N802 - gRPC-generated method name
        self, request: stt_pb2.SttInfoRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SttInfo:
        """Reports the loaded backend's capabilities (§2.4).

        `partials` is hard-false: Whisper is chunk-based, so this worker
        never emits a `partial`. Consumers must not assume partials exist.
        """
        return stt_pb2.SttInfo(
            name=self._backend.name,
            langs=[self._lang],
            input_sample_rate=self._backend.sample_rate,
            partials=False,
        )

    def Transcribe(  # noqa: N802 - gRPC-generated method name
        self, request_iterator: object, context: grpc.ServicerContext
    ) -> object:
        """Streams audio in, streams committed transcripts out.

        Yields one `SttResponse.final` per transcribed buffer. Errors are
        surfaced mid-stream via `context.abort`, which the Rust client
        receives as a stream `Err` item.
        """
        cancel = threading.Event()
        # Deliberately unbounded: a bounded queue would let the reader
        # thread block in `put()` mid-stream, and a blocked reader cannot
        # notice the `Cancel` that follows — which is the one message
        # whose latency actually matters (§2.5.1). Growth is bounded in
        # practice by the gate's utterance length upstream.
        inbox: queue.Queue[tuple[str, object]] = queue.Queue()
        reader = threading.Thread(
            target=self._read_requests,
            args=(request_iterator, inbox, cancel),
            name="stt-request-reader",
            daemon=True,
        )
        reader.start()

        buffer: list[np.ndarray] = []
        fmt: _StreamFormat | None = None

        try:
            while True:
                kind, payload = inbox.get()

                if kind == "error":
                    raise payload  # type: ignore[misc]

                if kind == "cancel":
                    log.info("cancel received, ending transcription stream")
                    return

                if kind == "audio":
                    chunk = payload
                    fmt = self._check_format(chunk, fmt)
                    try:
                        mono = to_mono(
                            np.asarray(chunk.pcm, dtype=np.float32), fmt.channels
                        )
                    except ValueError as exc:
                        # The chunk's samples contradict its own declared
                        # channel count — a client bug, not a worker fault.
                        raise _ProtocolError(str(exc)) from exc
                    buffer.append(mono)

                    # Drain whole windows, splitting at an exact sample
                    # boundary rather than handing an over-long buffer to
                    # the model: Whisper truncates anything past 30s, so
                    # an overshooting chunk would silently lose its tail.
                    window = int(MAX_SEGMENT_SECONDS * fmt.sample_rate)
                    while self._buffered_samples(buffer) >= window:
                        pending = np.concatenate(buffer)
                        head, tail = pending[:window], pending[window:]
                        buffer = [tail] if tail.size else []

                        final = self._transcribe(head, fmt, cancel)
                        if final is None:
                            return
                        yield stt_pb2.SttResponse(final=final)

                    if cancel.is_set():
                        log.info("cancel observed after flush, ending stream")
                        return
                    continue

                if kind == "end":
                    # Client half-closed: transcribe the sub-window
                    # remainder, which is the normal case for a
                    # gate-emitted utterance.
                    if buffer and fmt is not None:
                        final = self._transcribe(np.concatenate(buffer), fmt, cancel)
                        if final is not None:
                            yield stt_pb2.SttResponse(final=final)
                    return

        except _ProtocolError as exc:
            log.warning("rejecting stream: %s", exc)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # noqa: BLE001 - must reach the client in-band
            # A worker-side failure (CUDA OOM at chunk 40, a bad model
            # state) has to propagate mid-stream, not just kill the
            # process silently.
            log.exception("transcription failed")
            context.abort(grpc.StatusCode.INTERNAL, f"transcription failed: {exc}")

    def _transcribe(
        self, buffered: np.ndarray, fmt: _StreamFormat, cancel: threading.Event
    ) -> stt_pb2.FinalTranscript | None:
        """Transcribes one buffered segment into a `FinalTranscript`.

        Args:
            buffered: Mono f32 audio at the stream's native rate, at most
                one Whisper window long.
            fmt: The stream's declared format, for the resample source rate.
            cancel: Cooperative cancel flag, polled by the generate loop.

        Returns:
            The committed transcript, or `None` when decoding was
            cancelled — signalling the caller to end the response stream
            without emitting a truncated fragment.
        """
        pcm = resample(buffered, fmt.sample_rate, self._backend.sample_rate)
        log.debug(
            "transcribing %.2fs of audio (%d samples at %d Hz)",
            pcm.size / self._backend.sample_rate,
            pcm.size,
            self._backend.sample_rate,
        )

        result = self._backend.transcribe(pcm, cancel)
        if result.cancelled:
            return None
        return stt_pb2.FinalTranscript(
            text=result.text, confidence=result.confidence
        )

    def _buffered_samples(self, buffer: list[np.ndarray]) -> int:
        """Sample count buffered so far, counted without concatenating."""
        return sum(part.size for part in buffer)

    def _check_format(
        self, chunk: object, fmt: _StreamFormat | None
    ) -> _StreamFormat:
        """Validates a chunk's declared format against the stream's.

        The first chunk fixes the stream's rate and channel count; a later
        chunk that disagrees would silently corrupt the concatenated
        buffer, so it is rejected instead.

        Raises:
            _ProtocolError: On a non-positive or mid-stream-changed format.
        """
        if chunk.sample_rate <= 0 or chunk.channels < 1:
            raise _ProtocolError(
                f"chunk declares invalid format: sample_rate={chunk.sample_rate} "
                f"channels={chunk.channels}"
            )
        if fmt is None:
            log.debug(
                "stream format: %d Hz, %d channel(s)",
                chunk.sample_rate,
                chunk.channels,
            )
            return _StreamFormat(
                sample_rate=chunk.sample_rate, channels=chunk.channels
            )
        if (chunk.sample_rate, chunk.channels) != (fmt.sample_rate, fmt.channels):
            raise _ProtocolError(
                f"audio format changed mid-stream: {fmt.sample_rate} Hz/"
                f"{fmt.channels}ch -> {chunk.sample_rate} Hz/{chunk.channels}ch"
            )
        return fmt

    def _read_requests(
        self,
        request_iterator: object,
        inbox: queue.Queue[tuple[str, object]],
        cancel: threading.Event,
    ) -> None:
        """Drains the request stream onto `inbox` on a separate thread.

        Running the read separately from inference is what makes
        cooperative cancel work: a `Cancel` arriving while the GPU is busy
        sets `cancel` immediately, and the generate loop sees it between
        decode steps (§2.5.1). Reading inline would leave the message
        queued in the transport until inference finished, which defeats
        the point.
        """
        expected_seq = 0
        try:
            for request in request_iterator:
                which = request.WhichOneof("payload")
                if which == "cancel":
                    cancel.set()
                    inbox.put(("cancel", None))
                    return
                if which == "audio":
                    chunk = request.audio
                    if chunk.seq != expected_seq:
                        # Not fatal — a gap means upstream dropped audio,
                        # and a partly-heard utterance still beats none —
                        # but it must be visible when transcripts go odd.
                        log.warning(
                            "audio chunk seq gap: expected %d, got %d",
                            expected_seq,
                            chunk.seq,
                        )
                    expected_seq = chunk.seq + 1
                    inbox.put(("audio", chunk))
                else:
                    inbox.put(
                        (
                            "error",
                            _ProtocolError("request carried no audio or cancel payload"),
                        )
                    )
                    return
        except Exception as exc:  # noqa: BLE001 - hand the failure to the RPC thread
            inbox.put(("error", exc))
            return
        inbox.put(("end", None))


def parse_args(argv: list[str], description: str) -> argparse.Namespace:
    """Parses the standard worker CLI plus the STT-specific `--lang`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--socket-path",
        required=True,
        help="Filesystem path of the unix domain socket to bind and listen on.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Whisper model to load, e.g. 'large-v3' or a full HF repo id.",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Compute device to run the model on (v1: cuda only, see EPIC 0.7).",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Recognition language, fixed rather than auto-detected (v1: en).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Raise log level from INFO to DEBUG.",
    )
    return parser.parse_args(argv)


def build_server(
    socket_path: str, servicer: SttServicer
) -> tuple[grpc.Server, health.HealthServicer]:
    """Builds a UDS gRPC server serving `Stt` plus the health service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    stt_pb2_grpc.add_SttServicer_to_server(servicer, server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Remove a stale socket file from a previous, uncleanly-killed run.
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server.add_insecure_port(f"unix://{socket_path}")
    return server, health_servicer


def run_worker(
    build_backend: Callable[[argparse.Namespace], object],
    description: str = "Marceline STT worker",
    argv: list[str] | None = None,
) -> int:
    """Runs an STT worker end to end for the backend `build_backend` returns.

    `build_backend` is called with the parsed args *after* logging is set up
    and before the server starts serving, so a backend can import its ML
    stack lazily and log while loading. Every worker entrypoint is then just
    a one-line call to this.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv, description)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log.info(
        "starting stt worker model_id=%s device=%s lang=%s socket_path=%s",
        args.model_id,
        args.device,
        args.lang,
        args.socket_path,
    )

    backend = build_backend(args)
    servicer = SttServicer(backend, args.lang)
    server, health_servicer = build_server(args.socket_path, servicer)

    # Load before start()/SERVING: the supervisor's health probe must not
    # report ready while the weights are still landing on the GPU.
    backend.load()

    server.start()
    health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    log.info("stt worker up, listening on unix://%s", args.socket_path)

    stopping = {"stopping": False}

    def handle_signal(signum: int, _frame: object) -> None:
        if stopping["stopping"]:
            return
        stopping["stopping"] = True
        log.info("received signal %s, shutting down", signum)
        health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
        server.stop(grace=5).wait()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    server.wait_for_termination()
    return 0
