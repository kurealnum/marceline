#!/usr/bin/env python3
"""Shared TTS worker service (SPEC.md §2.2/§2.3/§2.4.1, EPIC 5.1, 5.5).

Every TTS worker — the default Kokoro one and Piper (EPIC 5.5) — is the
same gRPC server over the same `marceline.tts.Tts` contract, differing
only in the model behind it. That shared half lives here so the two cannot
drift apart: a backend supplies `name`, `sample_rate`, `voices` and
`synthesize(text, voice, cancel)`, and gets the transport, streaming,
cancel plumbing, and health service for free.

Streaming shape: the client streams already-segmented `text` spans (an
optional leading `voice` selects the voice for the whole stream) and the
worker streams back one or more `AudioChunk`s per span as they are
synthesized. A `Cancel` message interleaved on the request stream is read
by a reader thread while synthesis is running, and the backend's generate
loop notices it between steps (§2.5.1) — mirroring the STT worker's cancel
plumbing in `stt_service.py`.
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
from typing import Callable

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from marceline_protocol import common_pb2, tts_pb2, tts_pb2_grpc

# Name the supervisor's health check asks for.
SERVICE_NAME = "tts"

# gRPC threads: one Synthesize stream is served at a time in v1 (one
# response being spoken at a time), plus headroom for GetInfo/health
# probes arriving while synthesis is in flight.
MAX_WORKERS = 4

log = logging.getLogger("marceline.tts")


class _ProtocolError(Exception):
    """A client sent something inconsistent with the request stream.

    Reported to the client as `INVALID_ARGUMENT` mid-stream, so the Rust
    side sees it as a stream `Err` item (invariant 1, §2.4.1) rather than
    silent, wrong, or missing audio.
    """


class TtsServicer(tts_pb2_grpc.TtsServicer):
    """Serves `marceline.tts.Tts` against a loaded backend.

    The backend is injected rather than constructed here, which is what
    lets Piper (EPIC 5.5) drop in and lets these RPCs be tested without an
    ML stack installed.
    """

    def __init__(self, backend: object, default_voice: str) -> None:
        """Binds the servicer to a loaded backend.

        Args:
            backend: Loaded backend exposing `name`, `sample_rate`,
                `voices` and `synthesize(text, voice, cancel)`.
            default_voice: Configured voice id used when a request stream
                never sends its own `voice` message.
        """
        self._backend = backend
        self._default_voice = default_voice

    def GetInfo(  # noqa: N802 - gRPC-generated method name
        self, request: tts_pb2.TtsInfoRequest, context: grpc.ServicerContext
    ) -> tts_pb2.TtsInfo:
        """Reports the loaded backend's capabilities (§2.4)."""
        return tts_pb2.TtsInfo(
            name=self._backend.name,
            voices=list(self._backend.voices),
            output_sample_rate=self._backend.sample_rate,
        )

    def Synthesize(  # noqa: N802 - gRPC-generated method name
        self, request_iterator: object, context: grpc.ServicerContext
    ) -> object:
        """Streams text in, streams synthesized `AudioChunk`s out.

        Errors are surfaced mid-stream via `context.abort`, which the Rust
        client receives as a stream `Err` item.
        """
        cancel = threading.Event()
        # Deliberately unbounded, matching the STT worker: a bounded queue
        # would let the reader thread block in `put()` mid-stream, and a
        # blocked reader cannot notice the `Cancel` that follows — the one
        # message whose latency actually matters (§2.5.1).
        inbox: queue.Queue[tuple[str, object]] = queue.Queue()
        reader = threading.Thread(
            target=self._read_requests,
            args=(request_iterator, inbox, cancel),
            name="tts-request-reader",
            daemon=True,
        )
        reader.start()

        voice = self._default_voice
        seq = 0

        try:
            while True:
                kind, payload = inbox.get()

                if kind == "error":
                    raise payload  # type: ignore[misc]

                if kind == "cancel":
                    log.info("cancel received, ending synthesis stream")
                    return

                if kind == "voice":
                    voice = payload
                    continue

                if kind == "text":
                    text = payload
                    if not text:
                        continue
                    for pcm in self._backend.synthesize(text, voice, cancel):
                        yield tts_pb2.TtsResponse(
                            audio=common_pb2.AudioChunk(
                                seq=seq,
                                pcm=pcm.tolist() if hasattr(pcm, "tolist") else pcm,
                                sample_rate=self._backend.sample_rate,
                                channels=1,
                            )
                        )
                        seq += 1
                    if cancel.is_set():
                        log.info("cancel observed mid-synthesis, ending stream")
                        return
                    continue

                if kind == "end":
                    return

        except _ProtocolError as exc:
            log.warning("rejecting stream: %s", exc)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # noqa: BLE001 - must reach the client in-band
            log.exception("synthesis failed")
            context.abort(grpc.StatusCode.INTERNAL, f"synthesis failed: {exc}")

    def _read_requests(
        self,
        request_iterator: object,
        inbox: queue.Queue[tuple[str, object]],
        cancel: threading.Event,
    ) -> None:
        """Drains the request stream onto `inbox` on a separate thread.

        Running the read separately from synthesis is what makes
        cooperative cancel work: a `Cancel` arriving while synthesis is
        busy sets `cancel` immediately, and the generate loop sees it
        between steps (§2.5.1). Reading inline would leave the message
        queued in the transport until synthesis finished.
        """
        try:
            for request in request_iterator:
                which = request.WhichOneof("payload")
                if which == "cancel":
                    cancel.set()
                    inbox.put(("cancel", None))
                    return
                if which == "voice":
                    inbox.put(("voice", request.voice))
                elif which == "text":
                    inbox.put(("text", request.text))
                else:
                    inbox.put(
                        (
                            "error",
                            _ProtocolError(
                                "request carried no text, voice or cancel payload"
                            ),
                        )
                    )
                    return
        except Exception as exc:  # noqa: BLE001 - hand the failure to the RPC thread
            inbox.put(("error", exc))
            return
        inbox.put(("end", None))


def parse_args(argv: list[str], description: str) -> argparse.Namespace:
    """Parses the standard worker CLI, per the template's
    `--socket-path`/`--model-id`/`--device` convention (EPIC 0.4) — the
    same convention `WorkerSpec::command` in the Rust supervisor always
    launches a worker with. `--model-id` carries the default voice id
    here; `dest="voice"` keeps the rest of this module reading `args.voice`
    rather than the wire-flag name.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--socket-path",
        required=True,
        help="Filesystem path of the unix domain socket to bind and listen on.",
    )
    parser.add_argument(
        "--model-id",
        dest="voice",
        required=True,
        help="Default voice id to synthesize with, e.g. 'af_sky'.",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Compute device to run the model on (EPIC 0.7); Kokoro runs fine on CPU.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Raise log level from INFO to DEBUG.",
    )
    return parser.parse_args(argv)


def build_server(
    socket_path: str, servicer: TtsServicer
) -> tuple[grpc.Server, health.HealthServicer]:
    """Builds a UDS gRPC server serving `Tts` plus the health service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    tts_pb2_grpc.add_TtsServicer_to_server(servicer, server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Remove a stale socket file from a previous, uncleanly-killed run.
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server.add_insecure_port(f"unix://{socket_path}")
    return server, health_servicer


def run_worker(
    build_backend: Callable[[argparse.Namespace], object],
    description: str = "Marceline TTS worker",
    argv: list[str] | None = None,
) -> int:
    """Runs a TTS worker end to end for the backend `build_backend` returns.

    `build_backend` is called with the parsed args *after* logging is set
    up and before the server starts serving, so a backend can import its ML
    stack lazily and log while loading. Every worker entrypoint is then just
    a one-line call to this.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv, description)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log.info(
        "starting tts worker voice=%s device=%s socket_path=%s",
        args.voice,
        args.device,
        args.socket_path,
    )

    backend = build_backend(args)
    servicer = TtsServicer(backend, args.voice)
    server, health_servicer = build_server(args.socket_path, servicer)

    # Load before start()/SERVING: the supervisor's health probe must not
    # report ready while the weights are still landing on the device.
    backend.load()

    server.start()
    health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    log.info("tts worker up, listening on unix://%s", args.socket_path)

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
