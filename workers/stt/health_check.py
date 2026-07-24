#!/usr/bin/env python3
"""Health/ping and capability client for the STT worker's UDS server.

Verifies the `Done when` criterion of EPIC 3.1 by hand — that the worker
came up, bound its socket, and finished loading the model — and prints the
capabilities it reports, so `partials = false` is visible without reading
code. The Rust supervisor (EPIC 0.6) performs the same health check.
"""

import argparse
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from marceline_protocol import stt_pb2, stt_pb2_grpc

from worker import SERVICE_NAME


def check(socket_path: str, timeout_s: float = 5.0) -> bool:
    """Returns True if the worker at `socket_path` reports SERVING.

    Also prints the worker's `SttInfo` when it is serving.
    """
    channel = grpc.insecure_channel(f"unix://{socket_path}")
    try:
        health = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service=SERVICE_NAME), timeout=timeout_s
        )
        if health.status != health_pb2.HealthCheckResponse.SERVING:
            return False
        info = stt_pb2_grpc.SttStub(channel).GetInfo(
            stt_pb2.SttInfoRequest(), timeout=timeout_s
        )
    except grpc.RpcError as exc:
        print(f"health check failed: {exc}", file=sys.stderr)
        return False

    print(
        f"name={info.name} langs={list(info.langs)} "
        f"input_sample_rate={info.input_sample_rate} partials={info.partials}"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ping the STT worker")
    parser.add_argument("--socket-path", required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if check(args.socket_path):
        print("SERVING")
        return 0
    print("NOT SERVING")
    return 1


if __name__ == "__main__":
    sys.exit(main())
