"""Shared Python worker machinery (SPEC.md §2.2, EPIC 3.1, 3.5).

Model workers differ only in the model. Everything else — the UDS gRPC
server, the streaming `Stt` service, audio conditioning, cooperative
cancel, the health service — lives here, so a second backend is a backend
class plus a one-line entrypoint rather than a second copy of the
transport.

Importable because the worker venvs put the repository's `python/`
directory on `sys.path` (see any worker's `setup.sh`), the same way
`marceline_protocol` is.
"""
