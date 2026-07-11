# Marceline — Voice Assistant System Spec

> A local-first, hackable voice assistant. Speak to it → it thinks (LLM + tools) →
> it speaks back in a human voice. Every stage (STT, LLM, TTS) is hot-swappable.
> Persona and behavior are user-authored in a `SOUL.md` file.

**Status:** Draft v0.1
**Owner:** oscar
**Last updated:** 2026-07-10

---

## 1. Product summary

Marceline is a headless local daemon that runs on the user's machine and provides a
hands-free, conversational voice interface to an LLM. The interaction loop is:

1. **Listen** — always-on audio in, but gated by a wake word ("Marceline, …").
   Voice-activity detection (VAD) handles utterance start/stop after the wake word fires.
2. **Transcribe** — audio → text via a pluggable STT model (default: a Whisper-family
   model such as `whisper` / `faster-whisper` / a Wispr-Flow model from HuggingFace).
3. **Think** — text → response via a pluggable LLM behind the **OpenAI-compatible API
   standard**, so the user can point at LM Studio, Ollama, Anthropic (via proxy),
   OpenAI, or any compatible endpoint. During thinking the LLM may call **tools**
   (built-in + MCP servers).
4. **Speak** — response text → audio via a pluggable **local** TTS engine
   (Piper / Kokoro / Coqui). Streamed to the speakers.
5. **Barge-in** — the user can interrupt playback by speaking; TTS stops and Marceline
   listens again.

Persona, tone, rules, and defaults come from a user-editable `SOUL.md`.

### 1.1 Locked-in decisions (from planning Q&A)

| Area             | Decision                                                                      |
| ---------------- | ----------------------------------------------------------------------------- |
| Language / stack | **Hybrid**: Rust core (orchestrator, audio, IPC) + Python workers (ML models) |
| Form factor      | **Local headless daemon** + thin CLI control surface                          |
| Trigger          | **Wake-word-gated VAD** — wake word opens the mic, VAD does endpointing       |
| TTS              | **Local, pluggable** (Piper / Kokoro / Coqui behind one interface)            |
| Tools            | **Both** — a few native built-ins **and** MCP client support                  |
| Barge-in         | **Yes, in v1** — requires acoustic echo cancellation (AEC)                    |
| Memory           | **Persistent history + long-term (retrieval) memory**                         |
| MVP bar          | **Full loop, one provider per stage** — prove the pipeline, then generalize   |

### 1.2 Explicit non-goals (for now)

- No cloud/hosted multi-user service. Single machine, single user.
- No mobile/browser client in v1 (architecture should not preclude it later).
- No GUI in v1 (CLI + config files only).
- No cloud TTS in v1 (interface allows it; no adapter shipped).

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
  `faster-whisper`, Piper/Kokoro/Coqui). Isolating them in subprocesses means a
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

> Decision to confirm: gRPC adds a build dependency (protoc) but pays off with
> streaming + typed contracts. If you want zero-friction v1, start with stdio+msgpack
> and swap to gRPC behind the same trait. **Recommend gRPC from the start** because
> retrofitting streaming later is painful.

### 2.4 The plugin contracts (the heart of "hot-swappable")

Three Rust traits, each with a config-selectable backend:

```rust
// Pseudocode — the real thing is async + streaming.

trait SttEngine {
    // audio frames in, incremental + final transcripts out
    async fn transcribe(&self, audio: AudioStream) -> TranscriptStream;
    fn info(&self) -> EngineInfo; // name, lang, sample rate, capabilities
}

trait LlmEngine {
    // OpenAI-standard under the hood; supports tool calls + streaming
    async fn chat(&self, req: ChatRequest) -> ChatEventStream;
    fn info(&self) -> EngineInfo;
}

trait TtsEngine {
    // text in, pcm audio chunks out (streamed so playback can start early)
    async fn synthesize(&self, text: TextStream, voice: VoiceId) -> AudioStream;
    fn info(&self) -> EngineInfo;
}
```

- **STT** and **TTS** backends are usually "talk to the Python worker over gRPC."
  Swapping models = restart the worker with a different model id; the Rust trait impl
  is unchanged. This is what makes Wispr-Flow-from-HF vs. faster-whisper a config line.
- **LLM** backend is "OpenAI-standard HTTP client." Swapping to LM Studio vs. OpenAI
  vs. an Anthropic proxy = change `base_url` + `api_key` + `model` in config.

### 2.5 Conversation state machine (the orchestrator)

```
IDLE ──wake word──▶ LISTENING ──VAD end──▶ TRANSCRIBING ──▶ THINKING
                        ▲                                       │
                        │                              (tool calls loop)
                     barge-in                                   │
                        │                                       ▼
   IDLE ◀── playback done ── SPEAKING ◀── first TTS chunk ── RESPONDING
```

- **THINKING** may loop: LLM emits tool call → tool broker runs it → result fed back
  → LLM continues, until a final text answer. Streamed tokens flow into TTS as they
  arrive (sentence-chunked) so speaking starts before the full answer is generated.
- **Barge-in**: while SPEAKING, the wake/VAD gate stays armed; detected user speech
  → cancel TTS + flush audio out → jump to LISTENING.

### 2.6 Audio pipeline details

- Capture + playback via `cpal` (cross-platform). Ring buffer between capture and gate.
- **Wake word:** `openWakeWord` (permissive license) or Porcupine (commercial-friendly
  but licensing). Runs cheaply on every frame; only after it fires do we open the mic.
- **VAD:** Silero VAD (ONNX) for endpointing after wake word.
- **AEC (for barge-in):** the mic hears the speakers. Need acoustic echo cancellation
  (e.g. `speexdsp`/WebRTC AEC via a Rust binding, or run capture through the OS AEC).
  Plus loudness ducking of TTS when user speech is detected. **This is the single
  hardest realtime piece — call it out early.**

---

## 3. Configuration & customization

### 3.1 `config.toml` (machine/runtime config)

```toml
[stt]
backend = "faster-whisper"      # hot-swappable
model   = "large-v3"
device  = "cuda"                 # or "cpu"

[llm]
backend  = "openai-compatible"
base_url = "http://localhost:1234/v1"   # LM Studio, or api.openai.com, etc.
model    = "local-model"
api_key_env = "MARCELINE_LLM_KEY"

[tts]
backend = "piper"               # piper | kokoro | coqui
voice   = "en_US-amy-medium"

[wake]
word        = "marceline"
sensitivity = 0.6

[memory]
db_path      = "~/.marceline/history.db"
longterm     = true
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
- Long-term memory can _append_ to a managed section, but the user-authored parts are
  never overwritten (keep learned facts in a separate `MEMORY.md` or DB, referenced
  from SOUL).

---

## 4. Tooling (function calling)

- LLM tool calls use the OpenAI `tools` schema. The **tool broker** in Rust exposes a
  merged catalog to the model:
  - **Built-in tools** (native Rust): `web_search`, `read_file`, `list_dir`,
    `get_time`, `run_shell` (gated). Small, fast, no external process.
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

---

## 6. Tech stack summary

| Concern       | Choice                           | Notes                                            |
| ------------- | -------------------------------- | ------------------------------------------------ |
| Core language | Rust                             | daemon, audio, orchestration, IPC, tools, memory |
| Model runtime | Python workers                   | STT + TTS model hosts                            |
| Audio I/O     | `cpal`                           | cross-platform capture/playback                  |
| Wake word     | openWakeWord                     | check license vs. Porcupine                      |
| VAD           | Silero VAD (ONNX)                | endpointing                                      |
| AEC           | WebRTC/speex AEC                 | needed for barge-in                              |
| STT           | faster-whisper / Wispr-Flow (HF) | pluggable via worker                             |
| LLM           | OpenAI-compatible HTTP           | LM Studio/Ollama/OpenAI/etc.                     |
| TTS           | Piper (default), Kokoro, Coqui   | local, pluggable                                 |
| IPC           | gRPC over UDS (streaming)        | stdio+msgpack fallback                           |
| Tools         | native + MCP client              | merged catalog                                   |
| Memory        | SQLite + sqlite-vec              | history + long-term                              |
| Config        | TOML + SOUL.md                   | runtime vs. persona                              |

---

## 7. Epics & stories (Agile)

Format: **Epic → Stories**. Each story has a rough dependency note. Estimation left to
you; the dependency graph is in §8.

### EPIC 0 — Project foundation & scaffolding

Goal: a runnable Rust daemon skeleton with config, logging, and a Python worker harness.

- **0.1** As a dev, I want a Cargo workspace (`core`, `protocol`, `cli`) so the codebase
  is organized. _(blocks everything)_
- **0.2** As a dev, I want TOML config loading + validation so runtime settings load.
- **0.3** As a dev, I want structured logging + a `--verbose` flag for debugging.
- **0.4** As a dev, I want a Python worker template (venv, entrypoint, gRPC stub) so
  model workers have a consistent shape. _(depends 0.1)_
- **0.5** As a dev, I want the gRPC/protobuf contracts defined (`Stt`, `Tts` streaming
  services) so Rust and Python share one schema. _(depends 0.1)_
- **0.6** As a dev, I want a supervisor that launches/monitors/restarts Python workers.
  _(depends 0.4, 0.5)_

### EPIC 1 — Audio capture & playback

Goal: reliable realtime audio in/out.

- **1.1** Capture mic frames via `cpal` into a ring buffer. _(depends 0.1)_
- **1.2** Play PCM audio to speakers via `cpal`, streamed. _(depends 0.1)_
- **1.3** Device selection + sample-rate/resampling handling in config.
- **1.4** Audio-level metering / debug tap (write-to-wav) for troubleshooting.

### EPIC 2 — Wake word + VAD gate

Goal: only listen when addressed; detect utterance boundaries.

- **2.1** Integrate wake-word model; fire event on "{name}". _(depends 1.1)_
- **2.2** Integrate Silero VAD for endpointing after wake. _(depends 1.1)_
- **2.3** Gate state machine: IDLE→LISTENING→(collect utterance)→emit audio segment.
  _(depends 2.1, 2.2)_
- **2.4** Sensitivity/false-trigger tuning + config knobs.

### EPIC 3 — STT worker & streaming transcription

Goal: audio segment → text, hot-swappable model.

- **3.1** Python STT worker hosting faster-whisper; gRPC streaming impl. _(depends 0.5, 0.6)_
- **3.2** Rust `SttEngine` trait + gRPC client backend. _(depends 0.5)_
- **3.3** Wire gate output → STT → transcript. _(depends 2.3, 3.1, 3.2)_
- **3.4** Model hot-swap: change model via config/CLI, worker restarts. _(depends 3.1)_
- **3.5** Support a Wispr-Flow HF model as a second backend to prove pluggability.
  _(depends 3.1)_

### EPIC 4 — LLM engine (OpenAI standard) + streaming

Goal: text → streamed response, provider-agnostic.

- **4.1** Rust `LlmEngine` trait + OpenAI-compatible HTTP client with streaming.
  _(depends 0.1)_
- **4.2** Compile SOUL.md + memory into the system prompt. _(depends 4.1, 9.x, 6.x)_
- **4.3** Conversation turn management + context-window trimming. _(depends 4.1)_
- **4.4** Verify against LM Studio _and_ a hosted provider (config swap only).
  _(depends 4.1)_

### EPIC 5 — TTS worker & streaming speech

Goal: text → streamed human voice, local + pluggable.

- **5.1** Python TTS worker hosting Piper; gRPC streaming audio out. _(depends 0.5, 0.6)_
- **5.2** Rust `TtsEngine` trait + gRPC client backend. _(depends 0.5)_
- **5.3** Sentence-chunk streamed LLM tokens into TTS for low first-audio latency.
  _(depends 4.1, 5.1)_
- **5.4** Playback wiring: TTS chunks → audio out. _(depends 1.2, 5.2)_
- **5.5** Second TTS backend (Kokoro or Coqui) to prove pluggability. _(depends 5.1)_

### EPIC 6 — Tooling (built-ins + MCP)

Goal: the LLM can act, not just talk.

- **6.1** Tool broker + catalog exposure in OpenAI `tools` format. _(depends 4.1)_
- **6.2** Built-in tools: `web_search`, `read_file`, `list_dir`, `get_time`. _(depends 6.1)_
- **6.3** THINKING tool-call loop: run tool → feed result → continue. _(depends 4.1, 6.1)_
- **6.4** MCP client: discover + call MCP server tools, namespaced. _(depends 6.1)_
- **6.5** Tool safety classes + voice confirmation for side-effecting tools.
  _(depends 6.3, SOUL policy 9.x)_

### EPIC 7 — Barge-in & AEC

Goal: interrupt the agent by speaking. **Highest technical risk.**

- **7.1** Acoustic echo cancellation on capture path. _(depends 1.1, 1.2)_
- **7.2** Keep wake/VAD armed during SPEAKING. _(depends 2.3, 5.4)_
- **7.3** On detected speech during playback: cancel TTS, flush audio, → LISTENING.
  _(depends 7.1, 7.2)_
- **7.4** TTS ducking + debounce to avoid self-interruption. _(depends 7.3)_

### EPIC 8 — Orchestrator / conversation state machine

Goal: tie all stages into one loop.

- **8.1** State machine IDLE→LISTENING→TRANSCRIBING→THINKING→RESPONDING→SPEAKING.
  _(depends 2.3, 3.3, 4.1, 5.4)_
- **8.2** End-to-end happy path (the MVP loop). _(depends 8.1)_
- **8.3** Error/timeout handling per stage (worker down, LLM error, no speech).
- **8.4** Cancellation plumbing (barge-in, ctrl-c, restart). _(depends 7.3)_

### EPIC 9 — SOUL.md persona system

Goal: user authors who Marceline is.

- **9.1** Parse SOUL.md → structured persona + system-prompt compiler. _(depends 0.1)_
- **9.2** Hot-reload on file change. _(depends 9.1)_
- **9.3** Tool policy section drives 6.5 gating. _(depends 9.1, 6.5)_
- **9.4** Voice/pacing preferences flow to TTS. _(depends 9.1, 5.x)_

### EPIC 10 — Memory (history + long-term)

Goal: remembers within and across sessions.

- **10.1** SQLite schema + per-turn history logging. _(depends 8.1)_
- **10.2** Context assembly from recent history. _(depends 10.1, 4.3)_
- **10.3** sqlite-vec long-term store + embedding pipeline. _(depends 10.1)_
- **10.4** Background summarizer → memory entries. _(depends 10.3)_
- **10.5** Relevance retrieval injected into prompt. _(depends 10.3, 4.2)_
- **10.6** User inspect/edit/delete memory via CLI. _(depends 10.1)_

### EPIC 11 — CLI control surface

Goal: operate the daemon without a GUI.

- **11.1** `marceline start/stop/status`. _(depends 8.2)_
- **11.2** `marceline config` get/set + model swap commands. _(depends 3.4, 5.5)_
- **11.3** `marceline memory` list/search/forget. _(depends 10.6)_
- **11.4** `marceline say "…"` / `transcribe <file>` for testing stages. _(depends 3.x, 5.x)_
- **11.5** Live logs / `--follow` for debugging. _(depends 0.3)_

### EPIC 12 — Packaging, ops, quality

Goal: installable, observable, testable.

- **12.1** Build + package Rust daemon + Python workers (per-OS). _(depends 8.2)_
- **12.2** First-run setup: download default models, create config/SOUL templates.
- **12.3** Metrics: per-stage latency (wake→first-audio), logged/exposed.
- **12.4** Integration test harness (canned audio in → assert transcript/response).
- **12.5** Graceful degradation (worker crash → spoken error, auto-restart).

---

## 8. Dependency graph (what blocks what)

```
EPIC 0 (foundation) ─┬─▶ EPIC 1 (audio) ─┬─▶ EPIC 2 (wake+VAD) ─┐
                     │                    │                       │
                     ├─▶ EPIC 4 (LLM) ────┤                       │
                     │                    │                       ▼
                     ├─▶ EPIC 3 (STT) ◀───┘             EPIC 8 (orchestrator)
                     │        ▲                                   ▲
                     ├─▶ EPIC 5 (TTS) ───────────────────────────┤
                     │                                            │
                     └─▶ EPIC 9 (SOUL) ──▶ EPIC 6 (tools) ────────┤
                                                                  │
                    EPIC 8 ──▶ EPIC 10 (memory)                   │
                    EPIC 1+2+5 ──▶ EPIC 7 (barge-in/AEC) ─────────┘
                    EPIC 8 ──▶ EPIC 11 (CLI) & EPIC 12 (packaging)
```

Critical path to the **MVP loop (8.2)**:
`0 → (1 → 2) + 3 + 4 + 5 → 8`. SOUL (9), tools (6), memory (10), barge-in (7) layer on
after the loop breathes. **Do barge-in (7) last of the core features** — it's the
riskiest and depends on stable audio in+out.

Suggested build order:

1. EPIC 0, 1 (scaffold + audio)
2. EPIC 3, 4, 5 in parallel (the three model stages) — each independently testable
   via CLI (11.4) before integration
3. EPIC 2 (wake/VAD)
4. EPIC 8 (wire the MVP loop) ← **first demo**
5. EPIC 9 (SOUL), EPIC 6 (tools), EPIC 10 (memory)
6. EPIC 7 (barge-in/AEC)
7. EPIC 11, 12 (polish, packaging) throughout

---

## 9. What you're not thinking about (risks & gaps)

Things worth deciding _before_ they bite:

1. **Barge-in is the hard part.** AEC on consumer mics/speakers is genuinely tricky —
   the mic hears the TTS. Without good AEC or headphones, the agent interrupts itself.
   Mitigation: v1 could require headphones (no AEC needed), and treat speaker+mic AEC
   as its own hardening story. Decide the v1 acoustic assumption.
2. **Latency budget.** Perceived responsiveness = wake→first-audio-out. Every stage
   adds up (VAD tail + STT + LLM TTFT + TTS first chunk). Set a target (e.g. <1.5s to
   first spoken word) and instrument it (12.3). Streaming everywhere is non-negotiable.
3. **Turn-taking / endpointing quality.** Knowing when the user _stopped_ talking is
   subtle (pauses vs. done). Bad endpointing feels worse than bad STT. Budget tuning time.
4. **Interruptions vs. thinking.** Can the user interrupt while Marceline is _thinking_
   (mid tool call), not just speaking? Define cancellation semantics for THINKING too.
5. **Tool safety is a real attack surface.** An LLM with `run_shell` / filesystem write
   - web content in context = prompt-injection risk (a web page tells the model to
     delete files). Need: allow-lists, confirmation gates, sandboxing, and never
     auto-running dangerous tools from untrusted content. This deserves its own security
     pass.
6. **Wake-word licensing.** Porcupine is high quality but commercially licensed;
   openWakeWord is open but you may need to train/pick a "Marceline" model. Decide early.
7. **Model download & first-run UX.** STT/TTS models are big. First run needs a sane
   downloader + progress + disk-space checks (12.2).
8. **GPU/CPU resource contention.** STT and TTS may both want the GPU; a local LLM too.
   Decide device allocation and whether STT/TTS run on CPU while the GPU serves the LLM.
9. **Privacy posture.** Local-first is a selling point — but web search, MCP servers,
   and cloud LLM endpoints leak data. Be explicit in SOUL/config about what leaves the
   machine, and log it.
10. **Multi-turn context growth.** Long conversations blow the context window; you need
    trimming + summarization (10.4) early, not as an afterthought.
11. **Error voice UX.** When STT fails / LLM errors / a tool times out, Marceline should
    _say something graceful_, not hang silently. Design the spoken failure modes.
12. **Observability.** Because it's realtime + multi-process, you'll be blind without
    per-stage tracing and a debug tap on audio (1.4, 11.5, 12.3). Build it early.
13. **State when addressed twice / overlapping speech.** What if the wake word fires
    while already SPEAKING or THINKING? Define the state machine's behavior for every
    interrupt at every state (partly EPIC 7/8).
14. **Testing realtime audio is hard.** Invest in a canned-audio → assert harness (12.4)
    so you can regression-test the pipeline without a microphone.
15. **Config vs. SOUL boundary drift.** Keep "how it runs" (config.toml) strictly apart
    from "who it is" (SOUL.md), and keep _learned_ memory out of the user-authored SOUL
    file so hot-reload never fights the summarizer.

---

## 10. Open questions (need your call)

- **Acoustic assumption for v1:** headphones-only (skip AEC) vs. speaker+mic AEC now?
- **Wake word engine:** openWakeWord (open, DIY model) vs. Porcupine (licensed)?
- **IPC:** commit to gRPC now, or start stdio+msgpack and migrate?
- **Latency target** for wake→first-spoken-word?
- **Can THINKING be interrupted**, or only SPEAKING?
- **Default models** to ship (STT / TTS) and their license posture?
- **Agent name** = "Marceline" (repo name) as the wake word — confirm?
- **Shell/filesystem-write tools in v1**, or read-only until the security pass lands?

```

```
