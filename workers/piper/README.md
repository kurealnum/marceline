# TTS worker (Piper)

The second text-to-speech backend (EPIC 5.5), selected by
`[tts].backend = "piper"`. Proves the `TtsEngine` seam is real —
swapping backends is a config line, the Rust gRPC client (EPIC 5.2) is
unchanged — and doubles as the low-resource fallback when Kokoro isn't
viable (§1.1, §10).

Built on the worker template (EPIC 0.4); see SPEC.md §2.2, §2.3, §2.4.1.

## Layout

| File | Role |
| --- | --- |
| `worker.py` | Entrypoint; picks the backend, defers everything else to the shared service |
| `piper_backend.py` | Piper voice model load + inference (imports `piper`) |
| `voices/` | Downloaded `.onnx` + `.onnx.json` voice models (gitignored contents) |

Transport, streaming, cancel plumbing, and the health service live in
`python/marceline_worker/tts_service.py`, shared with the Kokoro worker
(EPIC 5.1) so the two cannot drift apart on the contract. Contract-level
tests live once, next to that service, and cover every backend.

`worker.py` imports `piper_backend` lazily, so the shared service and its
tests need neither `piper` nor `onnxruntime`.

## One voice per worker

Unlike Kokoro's fixed voice set selectable per request, a Piper worker
loads exactly **one** voice model at startup. Switching voices is a
worker restart with a different `--model-id`, mirroring how the STT workers
swap models (§2.4) — not a mid-stream selection. A request that names a
different voice is honored with a logged warning and synthesized in the
loaded voice anyway, rather than failing the turn.

## Setup

```
./setup.sh
```

Creates `.venv`, drops a `.pth` pointing at `python/`, and creates
`voices/`. Download a voice model (e.g. from the `piper-voices` releases)
into `voices/` as `<name>.onnx` + `<name>.onnx.json`.

## Running

```
.venv/bin/python worker.py --socket-path /tmp/marceline-tts.sock \
  --model-id en_US-lessac-medium --device cpu
```

`--model-id` is either a short name resolved against `voices/<name>.onnx`, or
a full filesystem path to a `.onnx` model elsewhere.

Normally you don't run this by hand — `marceline say` launches it from
`[tts]` config.

Smoke test it from another shell — this prints the worker's capabilities
and exits non-zero if it is not serving:

```
.venv/bin/python -m marceline_worker.tts_health_check --socket-path /tmp/marceline-tts.sock
```

## Tests

Backend-only tests, stubbing `piper` so they run without ONNX Runtime or
model weights:

```
.venv/bin/python -m unittest discover -s tests
```

The contract tests, shared by every TTS backend, live once in
`python/marceline_worker/tests` (see `workers/tts/README.md`).

## Streaming behavior

- **Real output rate declared, not assumed.** Piper voices are not all
  the same sample rate; `GetInfo`/every `AudioChunk` carries the loaded
  model's actual rate, not a guess (§2.4.1's chipmunk-bug warning).
- **Cooperative cancel.** The request stream is drained on a reader
  thread; a `Cancel` arriving mid-synthesis sets a flag the generate loop
  checks between per-sentence audio chunks and returns early (§2.5.1).
- **In-band errors.** Worker-side failures abort the stream with
  `INTERNAL`, which the Rust client sees as a stream `Err` item
  (invariant 1).
