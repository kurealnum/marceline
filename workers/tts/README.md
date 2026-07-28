# TTS worker (Kokoro)

The default text-to-speech backend (EPIC 5.1): a long-lived Python
subprocess hosting the Kokoro-82M model, serving the streaming
`marceline.tts.Tts` contract over a unix domain socket. Built on the
worker template (EPIC 0.4). See SPEC.md §2.2, §2.3, §2.4.1.

`workers/piper/` will be the alternative backend, selected by
`[tts].backend = "piper"` (EPIC 5.5).

The model runs out of process on purpose: a crash kills a worker the
supervisor restarts, instead of taking down the daemon (§2.2). Kokoro is
light and CPU-capable, so `[tts].device` can stay `"cpu"` while the GPU
serves STT/LLM.

## Layout

| File | Role |
| --- | --- |
| `worker.py` | Entrypoint; picks the backend, defers everything else to the shared service |
| `kokoro_backend.py` | Kokoro model load + inference (imports kokoro/torch) |

Transport, streaming, cancel plumbing, and the health service live in
`python/marceline_worker/tts_service.py`, shared with the Piper worker
(EPIC 5.5) so the two cannot drift apart on the contract. Contract-level
tests live once, next to that service, and cover every backend.

`worker.py` imports `kokoro_backend` lazily, so the shared service and its
tests need neither kokoro nor torch.

## Setup

```
./setup.sh
```

Creates `.venv` and drops a `.pth` pointing at `python/` so the generated
`marceline_protocol` stubs are importable.

## Running

```
.venv/bin/python worker.py --socket-path /tmp/marceline-tts.sock \
  --voice af_sky --device cpu
```

Normally you don't run this by hand — `marceline say` launches it from
`[tts]` config (EPIC 5.2).

Smoke test it from another shell — this prints the worker's capabilities
and exits non-zero if it is not serving:

```
.venv/bin/python -m marceline_worker.tts_health_check --socket-path /tmp/marceline-tts.sock
```

## Tests

The contract tests are shared by every TTS backend:

```
.venv/bin/python -m unittest discover -s ../../python/marceline_worker/tests
```

## Streaming behavior

- **Voice from the request or config.** A request stream may send a
  `voice` message before any `text`; a stream that never sends one uses
  the worker's configured default (`--voice`).
- **Multiple chunks per span.** Kokoro splits a text span into its own
  sub-utterances internally; each is streamed back as an `AudioChunk` as
  soon as it is ready, rather than concatenated, so playback can start
  before the whole span finishes synthesizing.
- **Cooperative cancel.** The request stream is drained on a reader
  thread, so a `Cancel` arriving mid-synthesis sets a flag the generate
  loop checks between sub-utterances and returns early (§2.5.1). Socket
  close is not used as a stop signal.
- **Self-describing audio.** Every `AudioChunk` carries Kokoro's actual
  output sample rate (24 kHz) rather than an assumed one (§2.4.1's
  chipmunk-bug warning).
- **In-band errors.** Worker-side failures abort the stream with
  `INTERNAL`, which the Rust client sees as a stream `Err` item
  (invariant 1).
- **Fixed voice set.** Kokoro ships specific voice ids (`af_sky`,
  `am_adam`, ...); `GetInfo` reports exactly the ids this worker can
  synthesize.
