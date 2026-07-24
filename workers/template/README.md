# Python worker template

Reusable shape for every model worker (STT, TTS, later the embedder):
a venv, a standard entrypoint, and a gRPC health-checked server bound
to a unix domain socket. See SPEC.md §2.2/§2.3 and EPIC 0.4.

## Creating a real worker

1. Copy this directory (e.g. `workers/stt/`).
2. Append model-specific dependencies to `requirements.txt`.
3. Run `./setup.sh` to create the venv and install deps.
4. In `worker.py`, load the model where marked (`--- real workers: load
   the model here ---`) and add the model's RPCs once the gRPC contracts
   land (EPIC 0.5).
5. Rename `SERVICE_NAME` to something specific (e.g. `"stt"`).

## Standard CLI convention

Every worker accepts the same three flags, so the Rust supervisor can
launch any worker identically:

```
worker.py --socket-path <path> --model-id <id> --device <device> [--verbose]
```

## Manual smoke test

```
./setup.sh
.venv/bin/python worker.py --socket-path /tmp/marceline-worker.sock \
  --model-id template --device cpu &
.venv/bin/python health_check.py --socket-path /tmp/marceline-worker.sock
```
