# STT worker (HF `whisper`)

The default speech-to-text backend (EPIC 3.1): a long-lived Python
subprocess hosting a Whisper-family model, serving the streaming
`marceline.stt.Stt` contract over a unix domain socket. Built on the
worker template (EPIC 0.4). See SPEC.md §2.2, §2.3, §2.4.1.

The model runs out of process on purpose: a CUDA OOM kills a worker the
supervisor restarts, instead of taking down the daemon (§2.2). Swapping
models is a restart with a different `--model-id` (EPIC 3.4), not a code
change.

## Layout

| File | Role |
| --- | --- |
| `worker.py` | Entrypoint, UDS gRPC server, `Stt` servicer, cancel plumbing |
| `whisper_backend.py` | HF Whisper model load + inference (imports torch/transformers) |
| `audio.py` | Downmix + resample incoming chunks to mono 16 kHz |
| `health_check.py` | Manual health/ping + capability dump |
| `tests/` | `unittest` suite; stubs the model, uses a real gRPC socket |

`worker.py` imports `whisper_backend` lazily, so the gRPC layer and its
tests need neither torch nor transformers.

## Setup

```
./setup.sh
```

Creates `.venv` and drops a `.pth` pointing at `python/` so the generated
`marceline_protocol` stubs are importable.

## Running

```
.venv/bin/python worker.py --socket-path /tmp/marceline-stt.sock \
  --model-id large-v3 --device cuda --lang en
```

`--model-id` takes either a short Whisper name (`large-v3`, expanded to
`openai/whisper-large-v3`) or a full hub repo id
(`distil-whisper/distil-large-v3`).

Smoke test it from another shell — this prints the worker's capabilities
and exits non-zero if it is not serving:

```
.venv/bin/python health_check.py --socket-path /tmp/marceline-stt.sock
```

## Tests

```
.venv/bin/python -m unittest discover -s tests
```

## Streaming behavior

- **Final-only.** Whisper is chunk-based, so the worker emits `final`
  transcripts and never a `partial`; `GetInfo` reports
  `partials = false`. Consumers must not assume partials exist (§2.4.1).
- **One `final` per flush.** Audio accumulates until the client
  half-closes or the buffer reaches Whisper's 30-second window; each
  flush produces one `final`. A gate-emitted utterance is well under the
  window, so it yields exactly one. A buffer that overshoots the window
  is split at an exact sample boundary, because Whisper silently
  discards anything past 30 seconds.
- **Cooperative cancel.** The request stream is drained on a reader
  thread, so a `Cancel` arriving while the GPU is busy sets a flag that
  the generate loop checks between decode steps and returns early
  (§2.5.1). Socket close is not used as a stop signal. A cancelled decode
  emits nothing — a truncated fragment is not a committed transcript.
- **Self-describing audio.** Sample rate and channel count travel with
  every chunk (invariant 2); the worker downmixes and resamples to
  16 kHz mono. A format that *changes* mid-stream is rejected with
  `INVALID_ARGUMENT` rather than silently concatenated.
- **In-band errors.** Worker-side failures (CUDA OOM mid-segment) abort
  the stream with `INTERNAL`, which the Rust client sees as a stream
  `Err` item (invariant 1).
- **Language is pinned** to `--lang`, not auto-detected: v1 is
  English-only and detection misfires on short noisy segments.
