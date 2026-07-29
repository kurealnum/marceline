//! Long-term memory: embed-then-store and retrieve-by-relevance on top of
//! [`crate::history::HistoryStore`]'s `memories` table (SPEC.md §5, layer
//! 3, EPIC 10.3).
//!
//! This module owns none of the storage or inference itself — it is the
//! thin seam between [`crate::embedding::EmbeddingPipeline`] (turns text
//! into a vector) and `HistoryStore`'s low-level `*_memory` methods (writes
//! it through the single write actor, per SPEC.md §5.2's single-writer
//! discipline; see `history.rs`'s module doc). Keeping the two separate is
//! what lets tests exercise the storage/retrieval logic with
//! [`crate::embedding::fake::FakeEmbedder`] instead of the real ONNX model,
//! which this sandbox and CI don't have on disk.
//!
//! ## Why `sqlite-vec`, not a brute-force fallback
//!
//! The issue called for trying a real `sqlite-vec` crate first and only
//! falling back to a brute-force cosine scan if none built offline. The
//! `sqlite-vec` crate (by the extension's own author, asg017) links
//! `sqlite-vec.c` straight into the same process as `rusqlite`'s bundled
//! SQLite via `SQLITE_CORE` + `sqlite3_auto_extension` — no loadable
//! `.so`/`.dylib`, no runtime `load_extension` call, and it compiles fully
//! offline once fetched from crates.io (`cc` compiles a vendored C file,
//! nothing more). The only wrinkle: the latest `-alpha` releases
//! (0.1.10-alpha.*) ship a C file that `#include`s a sibling
//! `sqlite-vec-diskann.c` the crate forgot to package, so they fail to
//! build; pinning to `0.1.9`, the latest non-alpha release, builds cleanly.
//! So the real dependency works, and the brute-force fallback was not
//! needed.
//!
//! ## Re-embed migration
//!
//! Vectors from two different embedding models are not comparable (SPEC.md
//! §5.2) — a config change to `[memory].embed_model`, or swapping in a
//! different pipeline, must never leave the index serving mixed-model
//! vectors. [`ensure_current_embed_model`] checks the stored model id(s)
//! against the pipeline in hand and, on a mismatch, calls [`reembed_all`],
//! which recomputes every vector and swaps the whole index in one write-actor
//! transaction ([`crate::history::HistoryStore::apply_reembed`]) — so a
//! reader either sees all-old or all-new vectors, never a mix.

use crate::embedding::{EmbedError, EmbeddingPipeline};
use crate::history::{HistoryError, HistoryStore, MemoryRecord, NewMemory};
use crate::llm::Trust;

/// Errors from the embed-then-store/retrieve pipeline.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Computing the embedding failed.
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// The underlying store operation failed.
    #[error(transparent)]
    Store(#[from] HistoryError),
}

/// Embeds `text` with `pipeline` and stores it as a new long-term memory,
/// returning its row id.
///
/// `provenance` should be carried over unchanged from whatever produced
/// `text` (e.g. the summarizer distilling a run of `turns`, EPIC 10.4) —
/// per SPEC.md §5.1, taint must survive into memory so untrusted tool/web
/// content can never be laundered into a trusted, injectable memory.
pub fn store_memory(
    store: &HistoryStore,
    pipeline: &mut dyn EmbeddingPipeline,
    text: impl Into<String>,
    provenance: Trust,
    created_at_ms: i64,
) -> Result<i64, MemoryError> {
    let text = text.into();
    let vector = pipeline.embed(&text)?;
    let id = store.insert_memory(NewMemory {
        text,
        embed_model: pipeline.model_id().to_string(),
        vector,
        provenance,
        created_at_ms,
    })?;
    Ok(id)
}

/// Embeds `query` with `pipeline` and returns the `k` most similar stored
/// memories, most similar first, alongside their distance.
///
/// Returns an empty result (not an error) if nothing has been stored yet.
pub fn retrieve_similar(
    store: &HistoryStore,
    pipeline: &mut dyn EmbeddingPipeline,
    query: &str,
    k: usize,
) -> Result<Vec<(MemoryRecord, f64)>, MemoryError> {
    let vector = pipeline.embed(query)?;
    Ok(store.search_similar(&vector, k)?)
}

/// Recomputes every stored memory's vector under `pipeline` and swaps the
/// whole index atomically, returning the number of memories re-embedded.
///
/// Reads every memory's saved source text (never the old vector — vectors
/// from the old model are meaningless once the model changes), embeds it
/// fresh, then commits every new vector plus the new `embed_model`/`dim` in
/// one write-actor transaction ([`HistoryStore::apply_reembed`]). A later
/// CLI story (EPIC 10.6) exposes this as `marceline memory reembed`.
pub fn reembed_all(
    store: &HistoryStore,
    pipeline: &mut dyn EmbeddingPipeline,
) -> Result<usize, MemoryError> {
    let existing = store.all_memories()?;
    let mut vectors = Vec::with_capacity(existing.len());
    for memory in &existing {
        vectors.push((memory.id, pipeline.embed(&memory.text)?));
    }
    let count = store.apply_reembed(pipeline.model_id(), pipeline.dim(), vectors)?;
    Ok(count)
}

/// Checks whether any stored memory was embedded under a different model
/// than `pipeline`, and if so runs [`reembed_all`] to bring the whole index
/// back to a single model's vector space.
///
/// Intended to be called once at startup, after opening the
/// [`HistoryStore`] and constructing the configured embedding pipeline —
/// wiring that into the daemon's startup sequence is EPIC 10.6/CLI-side
/// work, out of scope here; this function is the piece a later story calls.
/// A store with no memories yet is a no-op (`Ok(0)`): there is nothing to
/// disagree with.
pub fn ensure_current_embed_model(
    store: &HistoryStore,
    pipeline: &mut dyn EmbeddingPipeline,
) -> Result<usize, MemoryError> {
    let existing_models = store.distinct_embed_models()?;
    let mismatched = existing_models
        .iter()
        .any(|model| model != pipeline.model_id());
    if !mismatched {
        return Ok(0);
    }
    reembed_all(store, pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::fake::FakeEmbedder;

    fn store() -> HistoryStore {
        let dir = tempfile::tempdir().unwrap();
        // Leak the tempdir so the DB file outlives this helper — each test
        // only needs one store for its own duration and the OS reclaims
        // the directory at process exit, same tradeoff the history.rs
        // tests make implicitly by holding `dir` in scope instead.
        let path = dir.path().join("memory.db");
        std::mem::forget(dir);
        HistoryStore::open(path).unwrap()
    }

    #[test]
    fn stores_and_lists_a_memory() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);

        let id = store_memory(
            &store,
            &mut pipeline,
            "the user's timezone is US/Eastern",
            Trust::User,
            1_000,
        )
        .unwrap();

        let all = store.all_memories().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, id);
        assert_eq!(all[0].text, "the user's timezone is US/Eastern");
        assert_eq!(all[0].embed_model, "fake-v1");
        assert_eq!(all[0].dim, 16);
        assert_eq!(all[0].provenance, Trust::User);
    }

    #[test]
    fn retrieves_the_most_similar_memory_first() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);

        store_memory(&store, &mut pipeline, "the sky is blue", Trust::User, 1_000).unwrap();
        store_memory(
            &store,
            &mut pipeline,
            "the sky is blue today too",
            Trust::User,
            1_000,
        )
        .unwrap();
        store_memory(
            &store,
            &mut pipeline,
            "bananas are yellow",
            Trust::User,
            1_000,
        )
        .unwrap();

        let results = retrieve_similar(&store, &mut pipeline, "the sky is blue", 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.text, "the sky is blue");
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn retrieval_on_an_empty_store_is_empty_not_an_error() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);
        let results = retrieve_similar(&store, &mut pipeline, "anything", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn provenance_survives_storage() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);

        for provenance in [Trust::User, Trust::Assistant, Trust::ToolUntrusted] {
            let id = store_memory(&store, &mut pipeline, "text", provenance, 1_000).unwrap();
            assert_eq!(
                store.get_memory(id).unwrap().unwrap().provenance,
                provenance
            );
        }
    }

    #[test]
    fn ensure_current_embed_model_is_a_noop_when_models_already_match() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);
        store_memory(&store, &mut pipeline, "hello", Trust::User, 1_000).unwrap();

        let count = ensure_current_embed_model(&store, &mut pipeline).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn ensure_current_embed_model_is_a_noop_on_an_empty_store() {
        let store = store();
        let mut pipeline = FakeEmbedder::new("fake-v1", 16);
        assert_eq!(
            ensure_current_embed_model(&store, &mut pipeline).unwrap(),
            0
        );
    }

    #[test]
    fn changing_the_embed_model_triggers_a_full_reembed() {
        let store = store();
        let mut old_pipeline = FakeEmbedder::new("fake-v1", 16);
        store_memory(&store, &mut old_pipeline, "first", Trust::User, 1_000).unwrap();
        store_memory(&store, &mut old_pipeline, "second", Trust::User, 1_000).unwrap();

        let mut new_pipeline = FakeEmbedder::new("fake-v2", 16);
        let count = ensure_current_embed_model(&store, &mut new_pipeline).unwrap();
        assert_eq!(count, 2);

        for memory in store.all_memories().unwrap() {
            assert_eq!(memory.embed_model, "fake-v2");
        }

        // The re-embedded index is queryable under the new model's vectors.
        let results = retrieve_similar(&store, &mut new_pipeline, "first", 1).unwrap();
        assert_eq!(results[0].0.text, "first");
    }

    #[test]
    fn reembed_to_a_different_dimension_rebuilds_the_index() {
        let store = store();
        let mut small = FakeEmbedder::new("fake-small", 8);
        store_memory(&store, &mut small, "one", Trust::User, 1_000).unwrap();

        let mut large = FakeEmbedder::new("fake-large", 32);
        let count = reembed_all(&store, &mut large).unwrap();
        assert_eq!(count, 1);
        assert_eq!(store.all_memories().unwrap()[0].dim, 32);

        // Now queryable at the new dimension.
        let results = retrieve_similar(&store, &mut large, "one", 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn inserting_a_mismatched_dimension_without_reembedding_errors() {
        let store = store();
        let mut small = FakeEmbedder::new("fake-small", 8);
        store_memory(&store, &mut small, "one", Trust::User, 1_000).unwrap();

        let mut large = FakeEmbedder::new("fake-large", 32);
        let err = store_memory(&store, &mut large, "two", Trust::User, 1_000).unwrap_err();
        assert!(matches!(
            err,
            MemoryError::Store(HistoryError::DimMismatch { .. })
        ));
    }
}
