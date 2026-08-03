//! The `marceline memory list|search|edit|forget` commands (EPIC 10.6).
//!
//! Exposes `crate::history::HistoryStore`'s `memories` table over the CLI so
//! long-term memory stays auditable and editable, per SPEC.md §5's design
//! note: "plain rows the user can inspect, edit, and delete". `list`
//! (and `search`) only ever open a reader connection — `edit` and `forget`
//! are mutations, so they go through the same write actor every other write
//! in the process uses (see `core::history`'s module doc), never a second
//! connection.
//!
//! `edit` and `search` both need a real embedding pipeline: editing a row's
//! text must re-embed it (the old vector no longer matches new text), and
//! searching embeds the query the same way retrieval does. Both build the
//! real [`MiniLmEmbedder`] from `[memory]` config rather than a fake — there
//! is no "fake but still correct" option outside tests. If the configured
//! model files aren't on disk (this sandbox has none vendored), loading
//! fails with a clear message instead of panicking; `list` and `forget`
//! never need the pipeline at all, so they still work with no model
//! present.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use marceline_core::{
    Config, EmbeddingPipeline, HistoryStore, MemoryRecord, MiniLmEmbedder, Trust, TurnRecord,
};

/// How many recent turns `memory list` shows by default, overridable with
/// `--turns <n>`. Long-term memories (the summarizer's distilled facts,
/// EPIC 10.4) are the more durable half of what's "remembered" and print
/// in full; raw turn history is comparatively high-volume, so it's capped
/// unless the operator asks for more.
const DEFAULT_TURN_LIMIT: usize = 20;

/// Default directory `MiniLmEmbedder::load` reads `model.onnx` +
/// `tokenizer.json` from, relative to this crate — mirrors `converse.rs`'s
/// `models/silero_vad.onnx` convention. Not vendored in this repo (the
/// weights are a network download), so this is the path that is expected to
/// exist once an operator fetches the model, not something this sandbox can
/// exercise end to end.
fn default_model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/all-MiniLM-L6-v2")
}

/// Renders a [`Trust`] the way a human reading the CLI output wants to see
/// it — the same three strings `history.rs` persists in the `provenance`
/// column, but that mapping is private to `core::history`, so the CLI (pure
/// presentation, no policy) re-derives it here rather than reaching into
/// core internals for something this small.
fn provenance_str(trust: Trust) -> &'static str {
    match trust {
        Trust::User => "user",
        Trust::Assistant => "assistant",
        Trust::ToolUntrusted => "tool_untrusted",
    }
}

fn print_memory(m: &MemoryRecord) {
    println!(
        "#{id}  [{provenance}]  ({embed_model}, dim={dim})\n    {text}",
        id = m.id,
        provenance = provenance_str(m.provenance),
        embed_model = m.embed_model,
        dim = m.dim,
        text = m.text,
    );
}

fn print_turn(t: &TurnRecord) {
    let interrupted = if t.interrupted { " (interrupted)" } else { "" };
    println!(
        "#{id}  [{provenance}]  {session_id}/{role}{interrupted}\n    {text}",
        id = t.id,
        provenance = provenance_str(t.provenance),
        session_id = t.session_id,
        role = t.role,
        text = t.text,
    );
}

/// Opens the history store at `[memory].db_path` from `config_path`.
fn open_store(config_path: &Path) -> Result<HistoryStore, String> {
    let config = Config::load(config_path).map_err(|err| format!("failed to load config: {err}"))?;
    HistoryStore::open(config.memory.expanded_db_path())
        .map_err(|err| format!("failed to open history database: {err}"))
}

/// Builds the real embedding pipeline from `[memory]` config, for the
/// commands (`edit`, `search`) that need one. Reports a missing model
/// directory as a plain error, per this story's requirement that a missing
/// model fails gracefully rather than panicking.
fn open_pipeline(config_path: &Path, model_dir: Option<&Path>) -> Result<MiniLmEmbedder, String> {
    let config = Config::load(config_path).map_err(|err| format!("failed to load config: {err}"))?;
    let model_dir = model_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_model_dir);
    MiniLmEmbedder::load(&model_dir, config.memory.embed_model.clone()).map_err(|err| {
        format!(
            "failed to load embedding model from {}: {err}\n\
             (edit/search need the real MiniLM model + tokenizer on disk; \
             see core/src/embedding.rs's module doc — this is expected in \
             an environment without the model vendored)",
            model_dir.display()
        )
    })
}

/// Runs `marceline memory <list|search|edit|forget>` (EPIC 10.6).
///
/// Mirrors `run_config`'s dispatch shape (`main.rs`'s closest precedent for
/// a multi-level subcommand). Blocking `HistoryStore`/embedding calls run
/// inside `spawn_blocking`, consistent with `history.rs`'s own doc comment
/// recommending it for async callers, even though today's call sites happen
/// to run them from a one-shot `main` with nothing else on the runtime.
pub async fn run_memory(args: &[String]) -> ExitCode {
    let config_path =
        PathBuf::from(flag_value(args, "--config").unwrap_or_else(|| "config.toml".to_string()));
    let model_dir = flag_value(args, "--model-dir").map(PathBuf::from);

    match args.get(2).map(String::as_str) {
        Some("list") => {
            let turn_limit = flag_value(args, "--turns")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TURN_LIMIT);
            run_list(&config_path, turn_limit).await
        }
        Some("search") => {
            let Some(query) = args.get(3).cloned() else {
                eprintln!("memory search requires a query");
                return ExitCode::FAILURE;
            };
            let k = flag_value(args, "--k")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(5);
            run_search(&config_path, model_dir, query, k).await
        }
        Some("edit") => {
            let (Some(id), Some(text)) = (args.get(3), args.get(4)) else {
                eprintln!("memory edit requires an id and new text");
                return ExitCode::FAILURE;
            };
            let Ok(id) = id.parse::<i64>() else {
                eprintln!("memory edit: {id} is not a valid row id");
                return ExitCode::FAILURE;
            };
            run_edit(&config_path, model_dir, id, text.clone()).await
        }
        Some("forget") => {
            let Some(id) = args.get(3) else {
                eprintln!("memory forget requires an id");
                return ExitCode::FAILURE;
            };
            let Ok(id) = id.parse::<i64>() else {
                eprintln!("memory forget: {id} is not a valid row id");
                return ExitCode::FAILURE;
            };
            run_forget(&config_path, id).await
        }
        _ => {
            eprintln!("memory takes `list`, `search <query>`, `edit <id> <new-text>`, or `forget <id>`");
            ExitCode::FAILURE
        }
    }
}

/// Runs `marceline memory list` (EPIC 11.3): prints both halves of what's
/// stored — recent turn history (capped at `turn_limit`, oldest first) and
/// every long-term memory (the summarizer's distilled facts, EPIC 10.4,
/// printed in full since it's the durable, comparatively low-volume half).
async fn run_list(config_path: &Path, turn_limit: usize) -> ExitCode {
    let config_path = config_path.to_path_buf();
    let result = tokio::task::spawn_blocking(
        move || -> Result<(Vec<TurnRecord>, Vec<MemoryRecord>), String> {
            let store = open_store(&config_path)?;
            let turns = store
                .recent_turns_all_sessions(turn_limit)
                .map_err(|err| format!("failed to list turn history: {err}"))?;
            let memories = store
                .all_memories()
                .map_err(|err| format!("failed to list memories: {err}"))?;
            Ok((turns, memories))
        },
    )
    .await
    .expect("blocking task panicked");

    match result {
        Ok((turns, memories)) => {
            println!("== turn history (most recent {turn_limit}) ==");
            if turns.is_empty() {
                println!("no turns logged yet");
            } else {
                for t in &turns {
                    print_turn(t);
                }
            }

            println!("== long-term memories ==");
            if memories.is_empty() {
                println!("no memories stored yet");
            } else {
                for m in &memories {
                    print_memory(m);
                }
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("memory list failed: {err}");
            ExitCode::FAILURE
        }
    }
}

async fn run_search(
    config_path: &Path,
    model_dir: Option<PathBuf>,
    query: String,
    k: usize,
) -> ExitCode {
    let config_path = config_path.to_path_buf();
    let result = tokio::task::spawn_blocking(
        move || -> Result<Vec<(MemoryRecord, f64)>, String> {
            let store = open_store(&config_path)?;
            let mut pipeline = open_pipeline(&config_path, model_dir.as_deref())?;
            let vector = pipeline
                .embed(&query)
                .map_err(|err| format!("failed to embed query: {err}"))?;
            store
                .search_similar(&vector, k)
                .map_err(|err| format!("memory search failed: {err}"))
        },
    )
    .await
    .expect("blocking task panicked");

    match result {
        Ok(results) if results.is_empty() => {
            println!("no matching memories");
            ExitCode::SUCCESS
        }
        Ok(results) => {
            for (m, distance) in &results {
                print_memory(m);
                println!("    distance={distance:.4}");
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("memory search failed: {err}");
            ExitCode::FAILURE
        }
    }
}

async fn run_edit(config_path: &Path, model_dir: Option<PathBuf>, id: i64, text: String) -> ExitCode {
    let config_path = config_path.to_path_buf();
    let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let store = open_store(&config_path)?;
        let mut pipeline = open_pipeline(&config_path, model_dir.as_deref())?;
        let vector = pipeline
            .embed(&text)
            .map_err(|err| format!("failed to embed new text: {err}"))?;
        store
            .update_memory_text(id, text, vector, pipeline.model_id().to_string())
            .map_err(|err| format!("memory edit failed: {err}"))
    })
    .await
    .expect("blocking task panicked");

    match result {
        Ok(()) => {
            println!("memory #{id} updated");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

async fn run_forget(config_path: &Path, id: i64) -> ExitCode {
    let config_path = config_path.to_path_buf();
    let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let store = open_store(&config_path)?;
        store
            .delete_memory(id)
            .map_err(|err| format!("memory forget failed: {err}"))
    })
    .await
    .expect("blocking task panicked");

    match result {
        Ok(()) => {
            println!("memory #{id} forgotten");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

/// Reads the value following `flag` in `args`, if present. Duplicated from
/// `main.rs`'s private helper of the same name/shape — each subcommand
/// module owns its own tiny copy rather than a shared util module, matching
/// how `converse.rs`/`say.rs`/`say_to_llm.rs` already do it.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.get(index + 1).cloned()
}
