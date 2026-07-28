#!/usr/bin/env python3
"""Health/ping and capability client for a TTS worker's UDS server.

Verifies by hand that a worker came up, bound its socket, and finished
loading its model, and prints the voice ids and sample rate it reports.
Works against any TTS backend, since they all serve the same contract.
The Rust supervisor (EPIC 0.6) performs the same health check.

    python -m marceline_worker.tts_health_check --socket-path /tmp/marceline-tts.sock
"""

import argparse
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from marceline_protocol import tts_pb2, tts_pb2_grpc

from marceline_worker.tts_service import SERVICE_NAME


def check(socket_path: str, timeout_s: float = 5.0) -> bool:
    """Returns True if the worker at `socket_path` reports SERVING.

    Also prints the worker's `TtsInfo` when it is serving.
    """
    channel = grpc.insecure_channel(f"unix://{socket_path}")
    try:
        health = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service=SERVICE_NAME), timeout=timeout_s
        )
        if health.status != health_pb2.HealthCheckResponse.SERVING:
            return False
        info = tts_pb2_grpc.TtsStub(channel).GetInfo(
            tts_pb2.TtsInfoRequest(), timeout=timeout_s
        )
    except grpc.RpcError as exc:
        print(f"health check failed: {exc}", file=sys.stderr)
        return False

    print(
        f"name={info.name} voices={list(info.voices)} "
        f"output_sample_rate={info.output_sample_rate}"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ping the TTS worker")
    parser.add_argument("--socket-path", required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if check(args.socket_path):
        print("SERVING")
        return 0
    print("NOT SERVING")
    return 1


if __name__ == "__main__":
    sys.exit(main())
