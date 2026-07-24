# Marceline — Voice Assistant System Spec

> A local-first, hackable voice assistant. Speak to it → it thinks (LLM + tools) →
> it speaks back in a human voice. Every stage (STT, LLM, TTS) is hot-swappable.
> Persona and behavior are user-authored in a `SOUL.md` file.

**Status:** Draft v0.2
**Owner:** oscar
**Last updated:** 2026-07-16

---

## 1. Product summary

Marceline is a headless local daemon that runs on the user's machine and provides a
hands-free, conversational voice interface to an LLM. The interaction loop is:

1. **Listen** — always-on audio in, but gated by a wake word ("Marceline, …").
   Voice-activity detection (VAD) handles utterance start/stop after the wake word fires.
2. **Transcribe** — audio → text via a pluggable STT model (default: a Whisper-family
   model such as HuggingFace `openai/whisper` or `faster-whisper`).
3. **Think** — text → response via a pluggable LLM behind the **OpenAI-compatible API
   standard**, so the user can point at LM Studio, Ollama, Anthropic (via proxy),
   OpenAI, or any compatible endpoint. During thinking the LLM may call **tools**
   (built-in + MCP servers).
4. **Speak** — response text → audio via a pluggable **local** TTS engine
   (Kokoro default; Piper alternate). Streamed to the speakers.
5. **Barge-in** — the user can interrupt playback by speaking; TTS stops and Marceline
   listens again.

Persona, tone, rules, and defaults come from a user-editable `SOUL.md`.

### 1.1 Locked-in decisions (from planning Q&A)

| Area             | Decision                                                                      |
| ---------------- | ----------------------------------------------------------------------------- |
| Language / stack | **Hybrid**: Rust core (orchestrator, audio, IPC) + Python workers (ML models) |
| Form factor      | **Local headless daemon** + thin CLI control surface                          |
| Trigger          | **Wake-word-gated VAD** — wake word opens the mic, VAD does endpointing       |
| TTS              | **Local, pluggable** (Kokoro default; Piper behind one interface)             |
| Tools            | **Both** — a few native built-ins **and** MCP client support                  |
| Barge-in         | **Yes, in v1** — **headphones assumed, no AEC** (open-air AEC deferred)        |
| Memory           | **Persistent history + long-term (retrieval) memory**                         |
| MVP bar          | **Full loop, one provider per stage** — prove the pipeline, then generalize   |

### 1.2 Explicit non-goals (for now)

- No cloud/hosted multi-user service. Single machine, single user.
- No mobile/browser client in v1 (architecture should not preclude it later).
- No GUI in v1 (CLI + config files only).
- No cloud TTS in v1 (interface allows it; no adapter shipped).
- **CUDA-only in v1** (NVIDIA GPU required). CPU / Metal / ROCm are deferred; a device
  seam (story 0.7) keeps them addable without refactoring call sites.
- **Headphones assumed in v1** (no AEC). Open-air speaker+mic use is deferred (§2.6).
- **English-only in v1.** Whisper is multilingual and Kokoro has per-language voices,
  but v1 fixes `lang = "en"`; multi-language autodetect + voice matching is later.

---

## 2. Architecture

### 2.1 High-level topology

```
┌──────────────────────────────────────────────────────────────────────┐
│                        marceline-core  (Rust daemon)                    │
│                                                                        │
│  ┌────────────┐   ┌───────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  Audio I/O  │──▶│ Wake+VAD  │──▶│  Orchestrator │──▶│  Audio out   │  │
│  │  (cpal)     │   │  gate      │   │ (state machine)│  │ + AEC/duck   │  │
│  └────────────┘   └───────────┘   └──────┬───────┘   └─────────────┘  │
│         ▲                                  │                    ▲       │
│         │ mic frames                       │ IPC                │ pcm    │
│         │        ┌─────────────────────────┼────────────────────┘       │
│         │        │                          │                            │
│         │   ┌────▼─────┐   ┌────────────────▼───┐   ┌──────────────┐    │
│         │   │ SOUL.md   │   │  Tool broker       │   │ Memory store  │    │
│         │   │ loader    │   │ (built-ins + MCP)  │   │ (SQLite+vec)  │    │
│         │   └───────────┘   └────────────────────┘   └──────────────┘    │
└────────────────────────────┬───────────────────┬──────────────────────┘
                             │ IPC (gRPC/stdio)  │ HTTP (OpenAI std)
                   ┌──────────▼─────────┐  ┌──────▼───────────────────┐
                   │  Python workers     │  │  LLM endpoint (external)  │
                   │  • STT worker       │  │  LM Studio / Ollama /     │
                   │  • TTS worker       │  │  OpenAI / Anthropic-proxy │
                   │  (model runtime)    │  └──────────────────────────┘
                   └────────────────────┘
```

### 2.2 Why hybrid Rust + Python

- **Rust core** owns everything realtime and long-lived: audio capture/playback,
  the wake/VAD gate, the conversation state machine, IPC, tool brokering, memory,
  config. Rust gives predictable latency, a clean single daemon, easy distribution.
- **Python workers** own the ML models where the ecosystem lives (HuggingFace,
  `faster-whisper`, Kokoro / Piper). Isolating them in subprocesses means a
  model crash or CUDA OOM does not take down the daemon, and models can be
  restarted/hot-swapped independently.

### 2.3 IPC choice (recommendation)

- **STT/TTS workers:** long-lived Python subprocesses managed by the Rust core.
  Transport: **gRPC over a local UDS (unix domain socket)** with **streaming RPCs**
  — critical because we want streaming STT (partial transcripts) and streaming TTS
  (first-audio-chunk latency). Fallback/simplest: length-prefixed msgpack over stdio.
- **LLM:** plain HTTP against an **OpenAI-compatible** endpoint (`/v1/chat/completions`
  with `stream: true` and `tools`). No worker needed — it's just a client.
- **Tools/MCP:** MCP servers are their own processes; the Rust core is an MCP client
  (stdio or HTTP transport per server).

Decision: commit to gRPC from the start (see §10).

### 2.4 The plugin contracts (the heart of "hot-swappable")

Three Rust traits, each with a config-selectable backend:

```rust
// Pseudocode — the real thing is async + streaming.

trait SttEngine {
    // audio frames in, incremental + final transcripts out
    async fn transcribe(&self, audio: AudioStream) -> TranscriptStream;
    fn info(&self) -> SttInfo;   // name, langs, input sample rate, partials?
}

trait LlmEngine {
    // OpenAI-standard under the hood; supports tool calls + streaming
    async fn chat(&self, req: ChatRequest) -> ChatEventStream;
    fn info(&self) -> LlmInfo;   // name, ctx window, tool-call support, streaming?
}

trait TtsEngine {
    // text in, pcm audio chunks out (streamed so playback can start early)
    async fn synthesize(&self, text: TextStream, voice: VoiceId) -> AudioStream;
    fn info(&self) -> TtsInfo;   // name, output sample rate, available voice ids
}
```

- **Per-trait info types** (`SttInfo` / `LlmInfo` / `TtsInfo`), not one shared
  `EngineInfo` — the three stages report different capabilities and a single struct
  would carry wrong/empty fields per stage.

- **STT** and **TTS** backends are usually "talk to the Python worker over gRPC."
  Swapping models = restart the worker with a different model id; the Rust trait impl
  is unchanged. This is what makes HF `whisper` vs. `faster-whisper` a config line.
- **LLM** backend is "OpenAI-standard HTTP client." Swapping to LM Studio vs. OpenAI
  vs. an Anthropic proxy = change `base_url` + `api_key` + `model` in config.

#### 2.4.1 Stream contract types (define before writing any backend)

The trait signatures above lean on four stream types. These **are** the plugin
contract — the "hot-swappable via a config line" promise rests entirely on getting
them right. Leave them vague and backend #2 exposes assumptions baked into backend #1,
forcing a rewrite. Pin them down first. Three invariants govern all four:

1. **Every stream item is a `Result`** — errors propagate in-band, mid-stream
   (worker OOM at chunk 40, LLM 500 mid-token). Never `Stream<Item = T>`; always
   `Stream<Item = Result<T, EngineError>>`.
2. **All internal audio is f32; rate + channels travel with the data.** Sample
   *format* is fixed by the type (`pcm: Vec<f32>`), not carried — every mic/worker
   boundary converts to f32 once on the way in, so no consumer branches on format.
   (If zero-copy i16 passthrough is ever needed for perf, switch `pcm` to `Vec<u8>` +
   a `format: SampleFormat` field then — deliberately deferred; f32-everywhere is
   simpler and Whisper/Kokoro/cpal all handle it.) Sample rate / channels still travel
   with each chunk: no out-of-band rate assumption (that is how you get a chipmunk
   voice when a 22050 Hz worker feeds a 48000 Hz sink). Resampling ownership is
   explicit: the audio-out stage owns it, driven by the chunk's declared rate.
3. **LLM output is a tagged event enum, not a token string** — tool calls must be
   first-class events or the THINKING loop (§4, §6) cannot be built.

```rust
// LLM output — tool calls are first-class, interleaved with text.
enum ChatEvent {
    TextDelta(String),
    ToolCallDelta { id: String, name: Option<String>, args_delta: String },
    ToolCallDone { id: String },
    Done { finish_reason: FinishReason },
}
type ChatEventStream = Pin<Box<dyn Stream<Item = Result<ChatEvent, EngineError>> + Send>>;

// TTS out / mic in — self-describing PCM, sequence numbers to detect drops.
struct AudioChunk { seq: u64, pcm: Vec<f32>, sample_rate: u32, channels: u8 }
type AudioStream = Pin<Box<dyn Stream<Item = Result<AudioChunk, EngineError>> + Send>>;

// STT out — provisional vs committed must be distinguishable.
enum Transcript {
    Partial(String),                          // revisable ("helo" → "hello")
    Final { text: String, confidence: f32 },  // committed; only this goes to the LLM
}
type TranscriptStream = Pin<Box<dyn Stream<Item = Result<Transcript, EngineError>> + Send>>;

// Into TTS — sentence-chunking is the CALLER's job (§5.3); the trait receives
// already-segmented text so Kokoro vs. Piper granularity differences do not leak.
type TextStream = Pin<Box<dyn Stream<Item = Result<String, EngineError>> + Send>>;
```

- **`TranscriptStream`**: downstream sends only `Final` to the LLM; `Partial` is for
  UI/debug and endpointing tuning (§9.3). Modeling STT as a plain `String` stream
  loses this and leaks half-words into the prompt. **Live partials are a backend
  capability, not a guarantee:** the default HF `whisper` is chunk-based and effectively
  final-only, so `SttInfo { partials: bool, .. }` advertises whether a backend emits
  real partials. **v1 ships final-only**; consumers must not assume partials exist.
- **`ChatEventStream`**: sentence-chunking for TTS (§5.3) consumes `TextDelta` only;
  `ToolCall*` events drive the tool broker. This type is the one most likely to force
  a rewrite if gotten wrong — the entire tool-calling epic depends on it.
- **`AudioStream`** is shared by mic-in and TTS-out; `seq` lets the consumer detect
  dropped/reordered chunks.

### 2.5 Conversation state machine (the orchestrator)

```
                    ┌──────────────────── barge-in ─────────────────┐
                    │                                               │
IDLE ──wake word──▶ LISTENING ──VAD end──▶ TRANSCRIBING ──▶ THINKING ──┐
 ▲  ▲                   │                       │             │        │ first
 │  │                   │ no-speech             │ empty       │ (tool  │ TTS
 │  │                   │ timeout               │ transcript  │  loop) │ chunk
 │  │                   ▼                       ▼             ▼        ▼
 │  └── playback done ── SPEAKING ◀──────────────────────────── (final text)
 │                          │
 │                          └── barge-in ──▶ LISTENING
 │
 └── ERROR ── (any stage: worker down / LLM error / tool timeout) ──▶ speak
     graceful message ──▶ IDLE
```

- **THINKING** may loop: LLM emits tool call → tool broker runs it → result fed back
  → LLM continues, until a final text answer. Streamed tokens flow into TTS as they
  arrive (sentence-chunked) so speaking starts before the full answer is generated.
  (There is no separate RESPONDING state — the first TTS chunk simply transitions
  THINKING → SPEAKING; "generating" and "playing" are the same state from the loop's
  point of view, since tokens stream into playback continuously.)
- **Barge-in**: the wake/VAD gate stays armed during **both THINKING and SPEAKING**
  (§10); detected user speech fires the run's cancellation token (§2.5.1) → cancel the
  in-flight stage + flush audio out → jump to LISTENING.
- **Error / timeout edges** (every state has one): `no-speech` timeout in LISTENING,
  empty/failed transcript in TRANSCRIBING, LLM error or tool timeout in THINKING,
  **hallucinated transcript on near-silence** in TRANSCRIBING (see below),
  worker-down in TRANSCRIBING/SPEAKING. All route through **ERROR**, which speaks a
  graceful message (§9.11) and returns to IDLE — never a silent *hang*. Exception: if
  the failed stage **is TTS itself**, no spoken message is possible; ERROR logs and
  returns to IDLE silently (accepted — §9.11). Concrete timeout values are a tuning knob
  (EPIC 8.3).
- **Wake word while already SPEAKING/THINKING** is treated as barge-in, not a new
  session (§9.13). A second wake word mid-LISTENING re-arms the utterance.

#### 2.5.1 Cancellation protocol

Barge-in (and interrupting THINKING, §10) means one "stop" must propagate across
**three processes** — Rust core, Python STT/TTS workers, external LLM HTTP — plus any
running tool. Cancellation is *not* "close the socket." It is a cooperative,
multi-process protocol.

**One run token.** The orchestrator holds a `CancellationToken`
(`tokio_util::sync::CancellationToken`) scoped to the current conversation run. It is
cloned into every stage (STT, LLM, TTS, each tool). Firing it once → all stages
observe cancellation. This is the whole mechanism on the Rust side; no per-stage
bespoke signaling.

**Every tool exposes a `cancel` method.** The tool trait requires a cancel entry point
alongside its invocation — internals differ per tool, but the method must exist:

```rust
trait Tool {
    async fn call(&self, args: Value, cancel: CancellationToken) -> ToolResult;
    fn cancel(&self);          // MUST exist; may no-op for atomic tools
    fn safety_class(&self) -> SafetyClass;
}
```

- **Read-only / cancellable tools** (web_search, read_file): `cancel` aborts the
  in-flight work; result discarded.
- **Side-effecting tools**: cannot un-ring the bell. Once execution has started they
  are **non-cancellable** — `cancel` may be a no-op. Cancellation then means: do not
  feed the result back, do not let the LLM continue the loop. Define this per tool via
  `safety_class`. (v1 tools are read-only anyway, §10, so this stays theoretical for
  now — but the contract exists from day one.)

**Cross-process propagation (workers).** Socket close is an unreliable signal
mid-GPU-op — a Whisper inference kernel does not stop because a stream dropped.
So the gRPC protocol carries an **explicit cancel message**, and the worker's generate
loop checks the flag between decode steps and returns early. Cooperative cancellation,
checked inside the loop — otherwise the worker keeps burning GPU producing audio nobody
hears.

**LLM.** Drop the HTTP stream; most providers stop billing further tokens.

**Buffer flush.** Audio already in the speaker ring buffer keeps playing for tens of ms
after cancel. Barge-in must flush the playback ring buffer too, or Marceline talks over
the user for a beat.

**Partial-state policy.** On cancel, a partial LLM response exists. Log the turn to
history (§10) marked `interrupted`, storing the partial text, **and feed it back to the
LLM as context on the next turn** ("you were saying X when the user interrupted") so
the conversation stays coherent.

**Barge-in intent gate (resume vs. real interrupt).** Not every sound during playback
is a real interruption — breathing, keyboard, background speech. The gate is the
**wake-word model** (already running cheaply on every frame, §2.6), not full STT — this
avoids an STT round-trip on every stray sound and protects the ≤1.5s latency feel.
Policy:
1. During SPEAKING the wake-word model stays armed on the mic frames.
2. **No wake word fires → ignore the audio, keep playing.** Nothing is transcribed,
   nothing reaches the LLM.
3. **Wake word (`marceline` / `marcy`, from `[wake].words`) fires → commit the
   barge-in:** fire the cancel token, flush playback, → LISTENING; the following
   utterance is captured, transcribed, and becomes the next turn.

Consequence (accepted trade-off): interrupting requires saying the wake word, not just
starting to talk. This keeps false audio out of the pipeline entirely and makes "keep
playing" the default. (Alternative — any-speech interrupt via VAD — is more natural but
adds latency + false fires; deferred.)

Repeated interrupts must not balloon context: **only the single most-recent interrupted
partial** is carried forward as "you were saying…" context; older ones are logged to
history but dropped from the live prompt.

**Debounce.** Cancelling is expensive (kills GPU work). Require N ms of confirmed
speech before firing (§EPIC 7.4) so a cough does not waste a turn.

### 2.6 Audio pipeline details

- Capture + playback via `cpal` (cross-platform). Ring buffer between capture and gate.
- **Pre-roll buffer (same-breath capture).** The capture ring retains the last ~1–2s of
  mic frames at all times. When the wake word fires, utterance capture is **seeded from
  the pre-roll**, not started empty — otherwise a same-breath command
  ("Marceline stop that's wrong") loses the words spoken during the ~300ms state flip to
  LISTENING. Applies to wake-from-IDLE and to barge-in (§2.5.1) alike.
- **Wake word:** `openWakeWord` (permissive license; models produced in EPIC 13). Runs
  cheaply on every frame — used both to open the mic from IDLE and as the barge-in
  intent gate during SPEAKING (§2.5.1). (Porcupine considered and dropped — commercial
  licensing; no licensed dependency ships.)
- **VAD:** Silero VAD (ONNX) for endpointing after wake word.
- **AEC — deferred (v1 assumes headphones).** v1 **assumes the user wears headphones**,
  so the mic does not hear the speakers and no acoustic echo cancellation is required.
  This is a deliberate, risky simplification to unblock barge-in: it makes barge-in a
  pure "keep the gate armed + cancel + flush" problem (§2.5.1) with no DSP. Open-air
  speaker+mic use (real AEC via `speexdsp`/WebRTC, plus TTS ducking) is explicitly a
  **later hardening story**, not v1. Revisit before any non-headphone deployment.

---

## 3. Configuration & customization

### 3.1 `config.toml` (machine/runtime config)

`config.toml` holds **how it runs** — all machine/runtime knobs, model + device
selection, endpoints, paths, thresholds. It is schema-versioned (`version`) so upgrades
can validate + migrate old files (see EPIC 0.2). It never holds persona (that's SOUL)
and never holds secrets inline (only `*_env` pointers to environment variables).

```toml
version = 1                      # config schema version; drives migration on upgrade

[stt]
backend = "whisper"             # whisper (HF default) | faster-whisper
model   = "large-v3"
device  = "cuda"                 # v1: cuda only (device seam exists, story 0.7)
lang    = "en"                   # v1 is English-only

[llm]
backend  = "openai-compatible"
base_url = "http://localhost:1234/v1"   # LM Studio, or api.openai.com, etc.
model    = "local-model"
api_key_env = "MARCELINE_LLM_KEY"       # secret lives in env, never in this file
max_tokens_per_turn    = 2048           # cost guardrail (EPIC 4.5)
max_requests_per_session = 200          # cost guardrail (EPIC 4.5)
max_tool_iterations_per_turn = 8        # TEMP: caps the THINKING tool-call loop (EPIC 6.3)
                                        # so the LLM can't spin tool calls forever. v1 reads the
                                        # override from env MARCELINE_MAX_TOOL_ITERS; promote to a
                                        # first-class tuned knob later. On breach: stop the loop,
                                        # feed a "tool budget exhausted" note back, force a final answer.

[tts]
backend = "kokoro"              # kokoro | piper
voice   = "af_sky"              # backend-specific voice id (Kokoro fixed voice set)
device  = "cuda"                 # kokoro is light; runs fine on CPU too

[wake]
words       = ["marceline", "marcy"]   # wake words / barge-in intent words
sensitivity = 0.6

[vad]
silence_ms       = 700          # endpointing: silence that ends an utterance (2.5)
min_utterance_ms = 300
max_utterance_ms = 15000

[memory]
db_path         = "~/.marceline/history.db"
longterm        = true
embed_model     = "sentence-transformers/all-MiniLM-L6-v2"  # local HF, CPU-fine
embed_device    = "cpu"

[egress]
log = true                       # audit everything leaving the machine (EPIC 14.4)
```

### 3.2 `SOUL.md` (persona/behavior — user-authored)

A markdown file that shapes who Marceline _is_. Compiled into the system prompt +
runtime policy. Suggested structure:

```markdown
# Identity

Name, personality, speaking style, verbosity, humor.

# Voice

Preferred TTS voice, pacing, when to be terse vs. expansive.

# Values / rules

Hard rules ("never do X"), tone boundaries, safety preferences.

# Knowledge about me (the user)

Standing facts: name, timezone, projects, preferences.

# Tools policy

Which tools are allowed, which need confirmation, which are off.

# Examples

Few-shot exchanges that demonstrate desired behavior.
```

- SOUL.md is hot-reloaded on change (file watch).
- Separation of concerns: `config.toml` = _how it runs_; `SOUL.md` = _who it is_.
- **SOUL.md is never written by the system.** Learned facts live only in the memory DB
  (§5). At prompt-compile time the system prompt is assembled as:
  `SOUL.md (user-authored, read-only) + retrieved memories (from DB)`. Memory is
  **injected at compile time, never persisted back into the SOUL file**. This is what
  keeps hot-reload (§9.2) from ever fighting the background summarizer (§9.15) — they
  touch different files, one read-only to the system.

---

## 4. Tooling (function calling)

- LLM tool calls use the OpenAI `tools` schema. The **tool broker** in Rust exposes a
  merged catalog to the model:
  - **Built-in tools** (native Rust): `web_search`, `read_file`, `list_dir`,
    `get_time`. Small, fast, no external process. **v1 built-ins are read-only**
    (§10) — no `run_shell` and no filesystem-write tool ships until the dedicated
    security pass lands (§9.5). Every tool still implements the `cancel` contract
    (§2.5.1) regardless of safety class.
  - **MCP tools:** any configured MCP server's tools, discovered at startup and
    namespaced (`serverName.toolName`).
- **Safety gating:** tools are classified (read-only / side-effecting / dangerous).
  `SOUL.md` tool policy decides auto-run vs. spoken confirmation ("You want me to
  delete that file — confirm?"). Confirmation happens _in voice_.
- Tool results are fed back into the THINKING loop; the model decides when done.

---

## 5. Memory

Three layers, increasing scope:

1. **Working context** — current conversation turns, in RAM, trimmed to context window.
2. **Persistent history** — every turn logged to **SQLite** (`~/.marceline/history.db`).
   Survives restarts, searchable, replayable.
3. **Long-term memory** — durable facts/summaries retrieved by relevance. Use
   **SQLite + a vector extension** (`sqlite-vec`) so it's one file, no extra service.
   A background summarizer distills conversations into memory entries; retrieval
   injects top-k relevant memories into the prompt.

> Design note: keep memory _auditable and editable_ — plain rows the user can inspect
> and delete. Privacy is a feature for a local assistant.

### 5.1 Untrusted-content provenance (persists through memory)

Prompt-injection defense (EPIC 14.1) treats tool-returned content (web pages, MCP
results) as untrusted. That taint **must survive into long-term memory**, or the
summarizer (§9.15 / 10.4) launders untrusted web/MCP text into a memory entry that is
later injected into the system prompt as *trusted* — a persistent, cross-session
injection vector.

- Every memory row (and every conversation turn) carries a `provenance` /
  `trust` tag: `user` | `assistant` | `tool_untrusted`.
- The summarizer preserves the tag on derived entries: anything distilled from
  `tool_untrusted` source stays `tool_untrusted`.
- At prompt-compile time, `tool_untrusted` memories are injected inside a clearly
  fenced, non-authoritative block — never as instructions, never able to escalate
  tool permissions or trigger side-effecting tools (same rule as EPIC 14.1).

### 5.2 Embedding-model swaps must re-embed

Vectors are only comparable within the space of the model that produced them.
Changing `[memory].embed_model` changes dimensionality / geometry, so old vectors
become garbage against new query embeddings — silently (no error, just nonsense
retrieval).

- **Store the source text of every embedded entry**, plus the `embed_model` id and
  vector dimension used, alongside the vector.
- On startup, if the configured `embed_model` differs from the stored one, run a
  **re-embed migration**: recompute all vectors from the saved source text with the
  new model, then swap. Never mix vectors from two models in one index.
- CLI surfaces this (`marceline memory reembed`) so a swap is explicit, not silent
  corruption. (EPIC 10.3.)

---

## 6. Tech stack summary

| Concern       | Choice                           | Notes                                            |
| ------------- | -------------------------------- | ------------------------------------------------ |
| Core language | Rust                             | daemon, audio, orchestration, IPC, tools, memory |
| Model runtime | Python workers                   | STT + TTS model hosts                            |
| Audio I/O     | `cpal`                           | cross-platform capture/playback                  |
| Wake word     | openWakeWord                     | permissive; custom models in EPIC 13             |
| VAD           | Silero VAD (ONNX)                | endpointing                                      |
| AEC           | none in v1 (headphones assumed)  | open-air AEC (WebRTC/speex) deferred to later    |
| STT           | Whisper (HF) / faster-whisper    | pluggable via worker                             |
| LLM           | OpenAI-compatible HTTP           | LM Studio/Ollama/OpenAI/etc.                     |
| TTS           | Kokoro (default), Piper          | local, pluggable; Kokoro is light (CPU-capable)  |
| IPC           | gRPC over UDS (streaming)        | stdio+msgpack fallback                           |
| Tools         | native + MCP client              | merged catalog                                   |
| Memory        | SQLite + sqlite-vec              | history + long-term                              |
| Config        | TOML + SOUL.md                   | runtime vs. persona                              |

---

## 7. Epics & stories (Agile)

Format: **Epic → Stories**. Each story has a rough dependency note. Estimation left to
you; the dependency graph is in §8.

> **Every epic must ship something demoable.** No epic is "done" until it produces a
> visible/audible artifact — a script that prints service startup, a CLI stage test
> (§11.4), a canned-audio run, a printed transcript, an audio file written to disk.
> The **Demoable** bullet on each epic states its minimum demo. This keeps progress
> observable in a realtime multi-process system where regressions otherwise hide.

### EPIC 0 — Project foundation & scaffolding

Goal: a runnable Rust daemon skeleton with config, logging, and a Python worker harness.

- **0.1** As a dev, I want a Cargo workspace (`core`, `protocol`, `cli`) so the codebase
  is organized. _(blocks everything)_
- **0.2** As a dev, I want TOML config loading + validation, keyed on a `version`
  field, with a migration path that upgrades older config files on load (warn + rewrite,
  never silently drop keys). SOUL.md is free-form and un-versioned; only config.toml is
  versioned.
- **0.3** As a dev, I want structured logging + a `--verbose` flag for debugging.
- **0.4** As a dev, I want a Python worker template (venv, entrypoint, gRPC stub) so
  model workers have a consistent shape. _(depends 0.1)_
- **0.5** As a dev, I want the gRPC/protobuf contracts defined (`Stt`, `Tts` streaming
  services) so Rust and Python share one schema. _(depends 0.1)_
- **0.6** As a dev, I want a supervisor that launches/monitors/restarts Python workers.
  _(depends 0.4, 0.5)_
- **0.7** As a dev, I want a `device` abstraction in worker config (`cuda` only in v1)
  behind a small enum, so CPU / Metal / ROCm can be added later **without touching call
  sites**. v1 hard-requires CUDA (§9.8/§9.9) but the seam must exist from day one.
  _(depends 0.2)_
- **Demoable:** `marceline --version` + a script that boots the daemon, launches a
  stub worker, and logs "core up / worker up / worker restarted" on kill.

### EPIC 1 — Audio capture & playback

Goal: reliable realtime audio in/out.

- **1.1** Capture mic frames via `cpal` into a ring buffer. _(depends 0.1)_
- **1.2** Play PCM audio to speakers via `cpal`, streamed. _(depends 0.1)_
- **1.3** Device selection + sample-rate/resampling handling in config.
- **1.4** Audio-level metering / debug tap (write-to-wav) for troubleshooting.
- **Demoable:** record 3s from the mic to a wav and play it back; print live level meter.

### EPIC 2 — Wake word + VAD gate

Goal: only listen when addressed; detect utterance boundaries.

- **2.1** Integrate wake-word model; fire event on "{name}". _(depends 1.1, 13.2)_
- **2.2** Integrate Silero VAD for endpointing after wake. _(depends 1.1)_
- **2.3** Gate state machine: IDLE→LISTENING→(collect utterance)→emit audio segment.
  _(depends 2.1, 2.2)_
- **2.4** Sensitivity/false-trigger tuning + config knobs.
- **2.5** Endpointing tuning: silence-duration / min-utterance / max-utterance
  thresholds exposed as `config.toml` knobs (`[wake]`/`[vad]`), with a measured tuning
  pass on real speech (pauses vs. done). Bad endpointing feels worse than bad STT
  (§9.3) — budget real time here. _(depends 2.3)_
- **Demoable:** say "Marceline …"; console prints WAKE, then the captured utterance
  segment is written to a wav at VAD-detected end.

### EPIC 3 — STT worker & streaming transcription

Goal: audio segment → text, hot-swappable model.

- **3.1** Python STT worker hosting HF `whisper` (default); gRPC streaming impl. _(depends 0.5, 0.6)_
- **3.2** Rust `SttEngine` trait + gRPC client backend. _(depends 0.5)_
- **3.3** Wire gate output → STT → transcript. _(depends 2.3, 3.1, 3.2)_
- **3.4** Model hot-swap: change model via config/CLI, worker restarts. _(depends 3.1)_
- **3.5** Support `faster-whisper` as a second backend (vs. the HF `whisper` default)
  to prove pluggability via a config line. _(depends 3.1)_
- **3.6** Silence/hallucination guard. Whisper invents plausible text on near-silence
  or non-speech (famous failure mode) — VAD endpointing reduces but does not eliminate
  it. Gate the transcript before it reaches the LLM: drop segments below a min speech
  duration, use the backend's no-speech / avg-logprob signals where available, and route
  a rejected transcript through the empty-transcript ERROR edge (§2.5) rather than
  speaking a hallucination back. _(depends 3.3)_
- **Demoable:** `marceline transcribe sample.wav` prints the transcript; swap model
  in config, rerun, still works.

### EPIC 4 — LLM engine (OpenAI standard) + streaming

Goal: text → streamed response, provider-agnostic.

- **4.1** Rust `LlmEngine` trait + OpenAI-compatible HTTP client with streaming.
  _(depends 0.1)_
- **4.2** Compile SOUL.md + memory into the system prompt. _(depends 4.1, 9.x, 6.x)_
- **4.3** Conversation turn management + context-window trimming. **Basic trimming is
  part of the MVP loop (not optional)** — a long conversation blows the window before
  the EPIC 10.4 summarizer lands (§9.10). Ship simple oldest-turn drop now; summarizer
  layers on later. _(depends 4.1)_
- **4.4** Verify against LM Studio _and_ a hosted provider (config swap only).
  _(depends 4.1)_
- **4.5** Cost/rate guardrail for cloud endpoints: per-turn + per-session token/request
  cap, so a retry storm or barge-in loop can't run up unbounded cost. Refuse + speak a
  graceful message on breach. _(depends 4.1)_
- **Demoable:** `marceline say-to-llm "hello"` streams tokens to stdout against LM
  Studio, then against a hosted provider with only a config change.

### EPIC 5 — TTS worker & streaming speech

Goal: text → streamed human voice, local + pluggable.

- **5.1** Python TTS worker hosting Kokoro; gRPC streaming audio out. _(depends 0.5, 0.6)_
- **5.2** Rust `TtsEngine` trait + gRPC client backend. _(depends 0.5)_
- **5.3** Sentence-chunk streamed LLM tokens into TTS for low first-audio latency.
  _(depends 4.1, 5.1)_
- **5.4** Playback wiring: TTS chunks → audio out. _(depends 1.2, 5.2)_
- **5.5** Second TTS backend (Piper) to prove pluggability. Piper also serves as the
  low-resource fallback. _(depends 5.1)_
- **Demoable:** `marceline say "hello there"` speaks it and writes a wav; swap
  kokoro→piper in config, rerun.

### EPIC 6 — Tooling (built-ins + MCP)

Goal: the LLM can act, not just talk.

- **6.1** Tool broker + catalog exposure in OpenAI `tools` format. _(depends 4.1)_
- **6.2** Built-in tools: `web_search`, `read_file`, `list_dir`, `get_time`. _(depends 6.1)_
- **6.3** THINKING tool-call loop: run tool → feed result → continue. _(depends 4.1, 6.1)_
- **6.4** MCP client: discover + call MCP server tools, namespaced. _(depends 6.1)_
- **6.5** Tool safety classes + voice confirmation for side-effecting tools.
  _(depends 6.3, SOUL policy 9.x)_ **v1 ships read-only tools only** (§4, §10); the
  side-effecting/confirmation path is built but no dangerous tool is registered until
  the security pass (§9.5).
- **6.6** Tool `cancel` contract: every tool (built-in + MCP) implements `cancel`
  wired to the run cancel token (§2.5.1). _(depends 6.1)_
- **Demoable:** ask the LLM "what time is it" → it calls `get_time` → spoken answer;
  logs show the tool call + result fed back.

### EPIC 7 — Barge-in (headphones assumption, no AEC)

Goal: interrupt the agent by speaking. **v1 assumes headphones — no AEC** (§2.6). This
removes the DSP risk; barge-in becomes gate + cancel + flush.

- **7.1** Keep wake/VAD armed during THINKING **and** SPEAKING. _(depends 2.3, 5.4)_
- **7.2** Barge-in intent gate (§2.5.1): the **wake-word model** runs on mic frames
  during playback; only when it fires do we fire the run cancel token, flush, →
  LISTENING and capture the next utterance. Audio without a wake word never stops
  playback or reaches the LLM. _(depends 7.1, 8.4)_
- **7.3** Debounce on the wake-word fire to avoid a spurious detection cutting off
  playback. _(depends 7.2)_
- **7.4** _(Deferred — later hardening)_ Open-air support: WebRTC/speex AEC on the
  capture path + TTS loudness ducking, for speaker+mic use without headphones.
  Not v1. _(depends 7.2)_
- **Demoable:** speak over Marceline mid-sentence (wearing headphones); playback stops
  within debounce window and it re-listens.

### EPIC 8 — Orchestrator / conversation state machine

Goal: tie all stages into one loop.

- **8.1** State machine IDLE→LISTENING→TRANSCRIBING→THINKING→SPEAKING (no separate
  RESPONDING state — see §2.5). _(depends 2.3, 3.3, 4.1, 5.4)_
- **8.2** End-to-end happy path (the MVP loop). _(depends 8.1)_
- **8.3** Error/timeout handling per stage (worker down, LLM error, no speech).
- **8.4** Cancellation plumbing: one run `CancellationToken` cloned into every stage
  and every tool (§2.5.1); barge-in / ctrl-c / restart all fire it; workers observe an
  explicit gRPC cancel; playback ring buffer flushed. _(depends 7.2)_
- **Demoable:** end-to-end — wake, speak a question, hear a spoken answer (the MVP
  loop, 8.2); then trigger an error (kill a worker) and hear the graceful spoken failure.

### EPIC 9 — SOUL.md persona system

Goal: user authors who Marceline is.

- **9.1** Parse SOUL.md → structured persona + system-prompt compiler. _(depends 0.1)_
- **9.2** Hot-reload on file change. _(depends 9.1)_
- **9.3** Tool policy section drives 6.5 gating. _(depends 9.1, 6.5)_
- **9.4** Voice/pacing preferences flow to TTS. _(depends 9.1, 5.x)_
- **Demoable:** edit SOUL.md persona line, save; next spoken reply reflects the new
  tone without restart (hot-reload).

### EPIC 10 — Memory (history + long-term)

Goal: remembers within and across sessions.

- **10.1** SQLite schema + per-turn history logging. Enable **WAL mode**; history
  logging, the background summarizer (10.4), and vector search (10.5) all share one
  file and SQLite is single-writer, so route all writes through **one owning task/
  connection** (a write actor) with readers on separate connections. _(depends 8.1)_
- **10.2** Context assembly from recent history. _(depends 10.1, 4.3)_
- **10.3** sqlite-vec long-term store + embedding pipeline. Embeddings come from a
  local HF model — **`sentence-transformers/all-MiniLM-L6-v2`** (small, CPU-fine, no
  egress). This is a stealth fourth model stage but runs on CPU, so it does not add GPU
  pressure (§9.8). _(depends 10.1)_
- **10.4** Background summarizer → memory entries. _(depends 10.3)_
- **10.5** Relevance retrieval injected into prompt. _(depends 10.3, 4.2)_
- **10.6** User inspect/edit/delete memory via CLI. _(depends 10.1)_
- **Demoable:** tell Marceline a fact in one session, restart, ask for it in the next —
  it recalls; `marceline memory list` shows the stored row.

### EPIC 11 — CLI control surface

Goal: operate the daemon without a GUI.

- **11.1** `marceline start/stop/status`. Graceful shutdown ordering on SIGTERM/stop:
  (1) fire the run cancel token, (2) let each side-effecting tool run its own kill logic
  (per-tool, §2.5.1 — a tool decides whether to finish or abort), (3) flush + stop audio
  out, (4) checkpoint memory/history to SQLite, (5) signal workers to exit, (6) wait
  with a timeout then hard-kill stragglers. `status` reports **per-stage health** —
  STT worker up?, TTS worker up?, LLM endpoint reachable?, current state-machine state —
  using the supervisor's (0.6) view, so a wedged stage is visible without log-diving.
  _(depends 8.2)_
- **11.2** `marceline config` get/set + model swap commands. _(depends 3.4, 5.5)_
- **11.3** `marceline memory` list/search/forget. _(depends 10.6)_
- **11.4** `marceline say "…"` / `transcribe <file>` for testing stages. _(depends 3.x, 5.x)_
- **11.5** Live logs / `--follow` for debugging. _(depends 0.3)_
- **Demoable:** each subcommand runs and prints expected output against a live daemon.

### EPIC 12 — Packaging, ops, quality

Goal: installable, observable, testable.

- **12.1** Build + package Rust daemon + Python workers (per-OS). _(depends 8.2)_
- **12.2** First-run setup: download default models, create config/SOUL templates.
- **12.3** Metrics: per-stage latency (wake→first-audio), logged/exposed.
- **12.4** Integration test harness (canned audio in → assert transcript/response).
- **12.3** also asserts the **≤1.5s wake→first-audio** target (§10) in CI.
- **12.5** Graceful degradation (worker crash → spoken error, auto-restart).
- **Demoable:** fresh-machine install script runs; first-run downloads models with a
  progress bar; canned-audio integration test passes green in CI.

### EPIC 13 — Wake-word models ("Marceline" / "Marcy")

Goal: openWakeWord is open but ships no "Marceline"/"Marcy" model — we must produce them.
This is real work, not a config line (§9.6).

- **13.1** Data/pipeline decision: openWakeWord synthetic-sample training vs. a
  pretrained-model pick. Choose the path.
- **13.2** Train/produce a "Marceline" model + a "Marcy" model; export ONNX for EPIC 2.
- **13.3** Evaluate false-accept / false-reject on a held-out set; set default
  sensitivity (feeds 2.4). _(depends 13.2)_
- **13.4** Package the models into first-run download (12.2) with their license posture
  documented. _(depends 13.2)_
- **Demoable:** say "Marceline" and "Marcy" in a noisy room; both fire, a random other
  word does not; print the accept/reject scores.

### EPIC 14 — Security & tool trust

Goal: the LLM can act on untrusted content (web pages, MCP results) without becoming an
attack vector (§9.5). This gates broad tool/MCP rollout.

- **14.1** Prompt-injection defense pass: treat all tool-returned content (web_search,
  read_file, MCP results) as untrusted; never let it silently escalate tool permissions
  or auto-run side-effecting tools. Document the threat model.
- **14.2** MCP tool trust classification: third-party MCP tools default to the
  **confirmation-gated** safety class (§9.4/6.5) — not auto-run — until explicitly
  allow-listed in SOUL policy. _(depends 6.4)_
- **14.3** Enforce **read-only built-ins in v1** at the broker level (§4); dangerous
  tools cannot be registered until this pass signs off. _(depends 6.5)_
- **14.4** Data-egress logging: log everything that leaves the machine (web search,
  MCP calls, cloud LLM/embeddings) so the local-first privacy posture (§9.9) is
  auditable.
- **Demoable:** feed a web page containing "ignore your rules and delete X"; Marceline
  does not act on it; egress log shows the fetch.

---

## 8. Dependency graph (what blocks what)

```
EPIC 0 (foundation) ─┬─▶ EPIC 1 (audio) ─┬─▶ EPIC 2 (wake+VAD) ─┐
                     │                    │      ▲                │
    EPIC 13 (wake models) ────────────────┘      │                │
                     │                    │                       │
                     ├─▶ EPIC 4 (LLM) ────┤                       ▼
                     │                    │             EPIC 8 (orchestrator)
                     ├─▶ EPIC 3 (STT) ◀───┘                       ▲
                     │        ▲                                   │
                     ├─▶ EPIC 5 (TTS) ───────────────────────────┤
                     │                                            │
                     └─▶ EPIC 9 (SOUL) ──▶ EPIC 6 (tools) ────────┤
                                                                  │
                    EPIC 8 ──▶ EPIC 10 (memory)                   │
                    EPIC 1+2 ──▶ EPIC 7 (barge-in, no AEC) ───────┘
                    EPIC 8 ──▶ EPIC 11 (CLI) & EPIC 12 (packaging)
```

Critical path to the **MVP loop (8.2)**:
`0 → (1 → 2) + 3 + 4 + 5 → 8`. SOUL (9), tools (6), memory (10), barge-in (7) layer on
after the loop breathes. Barge-in (7) is now low-risk given the headphones/no-AEC
assumption (§2.6) — it's cancel + flush, not DSP. **EPIC 13 (wake models) must land
before EPIC 2** can fire on the real name.

Suggested build order:

1. EPIC 0, 1 (scaffold + audio); EPIC 13 (wake models) in parallel
2. EPIC 3, 4, 5 in parallel (the three model stages) — each independently testable
   via CLI (11.4) before integration
3. EPIC 2 (wake/VAD)
4. EPIC 8 (wire the MVP loop) ← **first demo**
5. EPIC 9 (SOUL), EPIC 6 (tools), EPIC 10 (memory)
6. EPIC 14 (security) — before broad tool/MCP rollout
7. EPIC 7 (barge-in)
8. EPIC 11, 12 (polish, packaging) throughout

---

## 9. What you're not thinking about (risks & gaps)

Things worth deciding _before_ they bite:

1. **Barge-in is the hard part.** AEC on consumer mics/speakers is genuinely tricky —
   the mic hears the TTS. Without good AEC or headphones, the agent interrupts itself.
   Mitigation: v1 could require headphones (no AEC needed), and treat speaker+mic AEC
   as its own hardening story. Decide the v1 acoustic assumption.
   **→ RESOLVED: v1 assumes headphones, no AEC (§2.6, §10). Open-air AEC deferred
   (EPIC 7.4). Risk accepted; revisit before non-headphone deployment.**
2. **Latency budget.** Perceived responsiveness = wake→first-audio-out. Every stage
   adds up (VAD tail + STT + LLM TTFT + TTS first chunk). Set a target (e.g. <1.5s to
   first spoken word) and instrument it (12.3). Streaming everywhere is non-negotiable.
   **→ RESOLVED: target ≤1.5s wake→first-audio, asserted in CI (§10, 12.3).**
3. **Turn-taking / endpointing quality.** Knowing when the user _stopped_ talking is
   subtle (pauses vs. done). Bad endpointing feels worse than bad STT. Budget tuning time.
4. **Interruptions vs. thinking.** Can the user interrupt while Marceline is _thinking_
   (mid tool call), not just speaking? Define cancellation semantics for THINKING too.
   **→ RESOLVED: both THINKING and SPEAKING interruptible via one run cancel token;
   every tool implements `cancel`; side-effecting tools are non-cancellable once
   started (§2.5.1, §10).**
5. **Tool safety is a real attack surface.** An LLM with `run_shell` / filesystem write
   - web content in context = prompt-injection risk (a web page tells the model to
     delete files). Need: allow-lists, confirmation gates, sandboxing, and never
     auto-running dangerous tools from untrusted content. This deserves its own security
     pass.
   **→ RESOLVED: EPIC 14 (security & tool trust) — injection defense, MCP tools
   confirmation-gated by default, read-only enforced at broker, egress logging.**
6. **Wake-word licensing.** Porcupine is high quality but commercially licensed;
   openWakeWord is open but you may need to train/pick a "Marceline" model. Decide early.
   **→ RESOLVED: openWakeWord (§10); "Marceline"/"Marcy" models produced in EPIC 13.**
7. **Model download & first-run UX.** STT/TTS models are big. First run needs a sane
   downloader + progress + disk-space checks (12.2).
8. **GPU/CPU resource contention.** STT and TTS may both want the GPU; a local LLM too.
   Decide device allocation and whether STT/TTS run on CPU while the GPU serves the LLM.
   **→ PARTLY RESOLVED: Kokoro is light and CPU-capable, so TTS need not fight for
   VRAM; Whisper STT + (optional local) LLM remain the GPU pressure. v1 is CUDA-only
   (§9.9) with a device seam (0.7) for later CPU/Metal/ROCm. Still: pick a default
   allocation for a single-GPU box.**
9. **Privacy posture.** Local-first is a selling point — but web search, MCP servers,
   and cloud LLM endpoints leak data. Be explicit in SOUL/config about what leaves the
   machine, and log it.
   **→ RESOLVED: egress logging is EPIC 14.4.**
10. **Multi-turn context growth.** Long conversations blow the context window; you need
    trimming + summarization (10.4) early, not as an afterthought.
    **→ RESOLVED: basic trimming is in the MVP loop (4.3); summarizer layers on later.**
11. **Error voice UX.** When STT fails / LLM errors / a tool times out, Marceline should
    _say something graceful_, not hang silently. Design the spoken failure modes.
12. **Observability.** Because it's realtime + multi-process, you'll be blind without
    per-stage tracing and a debug tap on audio (1.4, 11.5, 12.3). Build it early.
13. **State when addressed twice / overlapping speech.** What if the wake word fires
    while already SPEAKING or THINKING? Define the state machine's behavior for every
    interrupt at every state (partly EPIC 7/8).
    **→ RESOLVED: wake word during SPEAKING/THINKING = barge-in, not a new session;
    error/timeout edges on every state (§2.5).**
14. **Testing realtime audio is hard.** Invest in a canned-audio → assert harness (12.4)
    so you can regression-test the pipeline without a microphone.
15. **Config vs. SOUL boundary drift.** Keep "how it runs" (config.toml) strictly apart
    from "who it is" (SOUL.md), and keep _learned_ memory out of the user-authored SOUL
    file so hot-reload never fights the summarizer.

---

## 10. Open questions (need your call)

- **Acoustic assumption for v1:** headphones-only (skip AEC) vs. speaker+mic AEC now?
  > **Headphones-only, no AEC.** Deliberate, risky simplification to unblock barge-in
  (§2.6). Open-air AEC is deferred to a later hardening story (EPIC 7.4). Revisit
  before any non-headphone deployment.
- **Wake word engine:** openWakeWord (open, DIY model) vs. Porcupine (licensed)? > openWakeWord
- **IPC:** commit to gRPC now, or start stdio+msgpack and migrate? > gRPC
- **Latency target** for wake→first-spoken-word? > **≤ 1.5s** wake→first-audio-out.
  Hard target instrumented in §12.3; treat as the regression-test threshold. Understood
  that hitting it consistently across STT + LLM TTFT + TTS may be challenging — streaming
  at every stage is non-negotiable to get there.
- **Can THINKING be interrupted**, or only SPEAKING? > Both
- **Default models** to ship (STT / TTS) and their license posture? > Whisper
  (https://huggingface.co/docs/transformers/en/model_doc/whisper) for STT and
  **Kokoro** (local, Apache-2.0, ~82M params) for TTS. Kokoro keeps the local-first
  promise intact (no cloud TTS), is light enough to run on CPU while the GPU serves
  STT/LLM, and its permissive license is commercial-safe. Piper is the fallback.
  Kokoro ships a **fixed voice set** (no custom/cloned voices) — acceptable for v1;
  SOUL voice prefs (9.4) pick from that set. A voice-cloning backend can be hot-swapped
  in later behind the same `TtsEngine` trait (§2.4) without touching the loop.
- **Agent name** = "Marceline" (repo name) as the wake word — confirm? > Marceline and Marcy
- **Shell/filesystem-write tools in v1**, or read-only until the security pass lands? > Read-only

```

```
