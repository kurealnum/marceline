# Packaging Marceline (EPIC 12.1)

This directory produces an installable build of the Rust daemon/CLI
bundled with the Python STT/TTS workers, so a machine doesn't need a
hand-assembled Rust+Python toolchain to run Marceline. It does **not**
bundle the STT/TTS models themselves — those are downloaded on first run
(EPIC 12.2).

## Supported OS

Linux only, today. This is the only OS this repository builds or tests
on; `scripts/package/build.sh` checks `uname` and refuses to run anywhere
else rather than producing a build nobody has verified works.

## Prerequisites

- A Rust toolchain able to build this workspace (`cargo build --release
  -p cli` must succeed on its own first — see the repo's top-level
  README for toolchain setup).
- Python 3.12 (STT's worker pins `torch==2.4.1`, which has no wheels past
  `cp312` — see `workers/stt/setup.sh`'s own note). Each worker's
  `setup.sh` builds its own `.venv`, so this is only needed at build time,
  not by whoever runs the packaged install afterward.
- **CUDA.** v1's default STT (Whisper) and TTS (Kokoro) backends are
  CUDA-only (SPEC.md §9.9) — there is no CPU fallback path for them in
  this version. The machine running the packaged install needs a
  CUDA-capable GPU and driver; `nvidia-smi` succeeding is the quick
  sanity check. `faster-whisper`/`piper` are lighter-weight alternative
  backends (`[stt].backend`/`[tts].backend`) if CUDA isn't available, but
  are not v1's default and are not exercised by this packaging story.

## Building

```sh
scripts/package/build.sh [--out dist/marceline]
```

This is the "documented, reproducible command per OS" this story asks
for — rerunning it from a clean checkout produces an equivalent install
tree. It:

1. Runs `cargo build --release -p cli` (builds `core`+`protocol`+`cli`,
   per SPEC.md §0.1's workspace layout).
2. For each `workers/<name>/` directory: builds its `.venv` via its own
   `setup.sh` if missing, then copies the whole directory (venv
   included) into the output tree.
3. Copies `python/` (the shared `marceline_protocol`/`marceline_worker`
   packages every worker's venv points at via a `.pth` file) alongside
   the workers, and rewrites each copied venv's `.pth` to point at the
   copy rather than the original checkout — the output tree does not
   depend on the checkout it was built from still existing afterward.
4. Copies the repo's default `config.toml`, this file, and `install.sh`
   into the output tree.

Output layout (default `dist/marceline/`):

```
bin/marceline          release daemon + CLI binary
workers/<name>/...      each worker's script + venv
python/                 shared packages the workers' venvs import
config.toml             starting-point config
install.sh, PACKAGING.md
```

## Installing

```sh
dist/marceline/install.sh [--prefix ~/.local/share/marceline] [--bin-dir ~/.local/bin]
```

Copies the tree to `--prefix` and symlinks `bin/marceline` into
`--bin-dir`. A pre-existing `config.toml` at the prefix is never
overwritten on a reinstall, so an operator's edits survive an upgrade.

## How the daemon finds its workers once installed

`core::worker_paths::workers_root` (EPIC 12.1) resolves where the
`workers/` tree lives, in order:

1. `MARCELINE_WORKERS_DIR` env var, if set — an explicit override.
2. A `workers/` directory next to the running binary's parent directory
   (i.e. `<prefix>/workers/` next to `<prefix>/bin/marceline`) — the
   layout this install produces.
3. `workers/` relative to the current working directory — unchanged dev
   behavior for `cargo run`/`cargo test` from the repo root.

So a `marceline` installed by `install.sh` and run from anywhere (with
`--bin-dir` on `PATH`) finds its bundled workers without any extra
configuration.

## Verifying the install

On a clean machine, after `install.sh`:

```sh
marceline --version   # binary runs
marceline start        # boots the daemon; workers/ resolves per above
marceline status       # per-stage health, once workers/models are set up
marceline stop
```

Model download/config scaffolding for a truly first-run machine is EPIC
12.2's job — `marceline start` before that story lands still expects
`[stt].model`/`[tts].voice` in `config.toml` to already have their
weights fetched (whatever the worker's own model-loading path expects).
