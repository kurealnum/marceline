#!/usr/bin/env python3
"""Reusable Python worker template (SPEC.md §2.2, EPIC 0.4).

Every model worker (STT, TTS, later the embedder) is a long-lived
subprocess managed by the Rust core, talking gRPC over a local unix
domain socket. This template provides that shape: arg parsing, a UDS
gRPC server, and a health/ping RPC the supervisor (EPIC 0.6) uses to
confirm the worker is up.

Real workers copy this file (and requirements.txt), keep the arg
convention and health service, and add model-loading/inference code
where marked below.
"""

import argparse
import logging
import os
import signal
import sys
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

# Name of the health-checked service. Real workers should rename this to
# something specific (e.g. "stt", "tts") when they specialize the template.
SERVICE_NAME = "worker"

log = logging.getLogger("marceline.worker")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parses the standard worker CLI convention.

    Every worker accepts the same three flags so the Rust supervisor
    (EPIC 0.6) can launch any worker the same way; model-specific flags
    are added by real workers alongside these.
    """
    parser = argparse.ArgumentParser(description="Marceline model worker")
    parser.add_argument(
        "--socket-path",
        required=True,
        help="Filesystem path of the unix domain socket to bind and listen on.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Identifier of the model this worker should load.",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Compute device to run the model on (v1: cuda only, see EPIC 0.7).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Raise log level from INFO to DEBUG.",
    )
    return parser.parse_args(argv)


def build_server(socket_path: str) -> tuple[grpc.Server, health.HealthServicer]:
    """Builds a gRPC server bound to `socket_path` with the standard
    `grpc.health.v1.Health` service registered.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Remove a stale socket file from a previous, uncleanly-killed run.
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server.add_insecure_port(f"unix://{socket_path}")
    return server, health_servicer


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log.info(
        "starting worker model_id=%s device=%s socket_path=%s",
        args.model_id,
        args.device,
        args.socket_path,
    )

    server, health_servicer = build_server(args.socket_path)

    # --- real workers: load the model here, using args.model_id/args.device ---

    server.start()
    # Mark the worker serving only once startup (model load, above) succeeds,
    # so the supervisor's health/ping RPC reflects real readiness.
    health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    log.info("worker up, listening on unix://%s", args.socket_path)

    stop_event_holder = {"stopping": False}

    def handle_signal(signum: int, _frame: object) -> None:
        if stop_event_holder["stopping"]:
            return
        stop_event_holder["stopping"] = True
        log.info("received signal %s, shutting down", signum)
        health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
        server.stop(grace=5).wait()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    sys.exit(main())
