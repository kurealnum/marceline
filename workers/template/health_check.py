#!/usr/bin/env python3
"""Standalone health/ping client for a worker's UDS gRPC server.

Used to verify a worker is up (`Done when` criterion of EPIC 0.4), and
by the Rust supervisor (EPIC 0.6) as a reference for the same check.
"""

import argparse
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from worker import SERVICE_NAME


def check(socket_path: str, timeout_s: float = 5.0) -> bool:
    """Returns True if the worker at `socket_path` reports SERVING."""
    channel = grpc.insecure_channel(f"unix://{socket_path}")
    stub = health_pb2_grpc.HealthStub(channel)
    try:
        response = stub.Check(
            health_pb2.HealthCheckRequest(service=SERVICE_NAME), timeout=timeout_s
        )
    except grpc.RpcError as exc:
        print(f"health check failed: {exc}", file=sys.stderr)
        return False
    return response.status == health_pb2.HealthCheckResponse.SERVING


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ping a worker's health RPC")
    parser.add_argument("--socket-path", required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if check(args.socket_path):
        print("SERVING")
        return 0
    print("NOT SERVING")
    return 1


if __name__ == "__main__":
    sys.exit(main())
