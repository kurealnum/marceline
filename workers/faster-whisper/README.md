# STT worker (`faster-whisper`)

The second STT backend (EPIC 3.5), selected by
`[stt].backend = "faster-whisper"`. Same Whisper weights as the default
worker, different runtime: CTranslate2 instead of PyTorch, which is faster
and lighter on VRAM.

Its reason for existing is as much a test as a feature. If adding a second
engine had required changing the gRPC service or the Rust client, that would
have meant the stream contract (§2.4.1) had baked in assumptions from
backend #1. It didn't: this worker is a backend class plus a one-line
entrypoint.

## Layout

| File | Role |
| --- | --- |
| `worker.py` | Entrypoint; picks the backend, defers everything else to the shared service |
| `faster_whisper_backend.py` | CTranslate2 model load + inference |
| `tests/` | Backend-only tests; stubs `faster_whisper`, so no weights needed |

Transport, buffering, audio conditioning, cancel plumbing, and the health
service all come from `python/marceline_worker/stt_service.py`, shared with
the default worker. Contract-level tests live once, next to that service.

## Setup

```
./setup.sh
```

## Running

```
.venv/bin/python worker.py --socket-path /tmp/marceline-stt.sock \
  --model-id large-v3 --device cuda --lang en
```

Normally you don't run this by hand — set `[stt].backend = "faster-whisper"`
and `marceline transcribe` launches it (EPIC 3.4).

Smoke test from another shell:

```
.venv/bin/python -m marceline_worker.health_check --socket-path /tmp/marceline-stt.sock
```

Expect `partials=False` and a name like `faster-whisper:large-v3`.

## Tests

```
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s ../../python/marceline_worker/tests
```

## How it differs from the `whisper` worker

The user-visible config means the same thing across both — `model`,
`device`, `lang` — and both are final-only. What differs is inside the
backend, which is where backend differences belong:

- **Model ids pass through unchanged.** `faster-whisper` resolves
  `"large-v3"` itself, where the HF backend expands it to
  `openai/whisper-large-v3`.
- **Cancel is a generator break.** `transcribe()` returns a lazy generator
  that decodes one segment per step, so the cooperative cancel check
  (§2.5.1) is a loop condition rather than a `StoppingCriteria`. Both check
  between decode steps; neither relies on socket close.
- **Confidence comes from the `avg_logprob`** the library reports, not from
  recomputed token scores — but it is the same `exp(mean logprob)` quantity,
  so the number means the same thing whichever backend produced it.
- **Compute type** is chosen by device (`float16` on CUDA, `int8` on CPU)
  rather than being another config knob: `[stt]` is about what to run, not
  how to quantize it.
