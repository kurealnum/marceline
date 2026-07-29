//! CPU embedding pipeline for long-term memory (SPEC.md §5, §9.8, EPIC
//! 10.3): turns memory source text into a fixed-length vector so it can be
//! indexed and retrieved by semantic relevance.
//!
//! Mirrors the pattern [`crate::vad::SileroVad`] established for Silero VAD
//! (EPIC 2.2): a real, off-the-shelf ONNX model is loaded from a local
//! `.onnx` file and run through `ort`, no placeholder inference. The model
//! here is `sentence-transformers/all-MiniLM-L6-v2` — small enough to run
//! on CPU with no GPU pressure (§9.8), which is why `[memory].embed_device`
//! only ever needs to say `cpu` in v1.
//!
//! [`EmbeddingPipeline`] is the seam: [`MiniLmEmbedder`] is the real
//! ONNX-backed implementation, but nothing in [`crate::memory`] talks to
//! `ort` or a tokenizer directly, so storage/retrieval tests substitute a
//! deterministic fake instead of requiring the ~90MB model + tokenizer
//! files this sandbox (and CI) doesn't have on disk.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// `all-MiniLM-L6-v2`'s sentence-embedding output width. Fixed for this
/// model family; a different embedding model would need its own constant
/// and its own `vec0` table dimension (see `crate::memory`'s module doc for
/// why v1 hardcodes one dimension rather than making the vector index
/// dimension-generic).
pub const MINILM_DIM: usize = 384;

/// Errors from loading or running an embedding pipeline.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// Loading the ONNX model failed (missing file, bad format, etc.).
    #[error("failed to load embedding model: {0}")]
    LoadModel(#[source] ort::Error),
    /// Loading the tokenizer (`tokenizer.json`) failed.
    #[error("failed to load tokenizer at {path}: {source}")]
    LoadTokenizer {
        /// Path the tokenizer was loaded from.
        path: std::path::PathBuf,
        /// Underlying tokenizers-crate error.
        #[source]
        source: tokenizers::Error,
    },
    /// Tokenizing the input text failed.
    #[error("failed to tokenize text: {0}")]
    Tokenize(#[source] tokenizers::Error),
    /// Running inference failed.
    #[error("embedding inference failed: {0}")]
    Inference(#[source] ort::Error),
}

/// Something that turns text into a fixed-length embedding vector.
///
/// The one seam between `crate::memory`'s storage/retrieval logic and
/// whatever actually computes vectors — [`MiniLmEmbedder`] in production,
/// a deterministic fake in tests. Every implementation must always return
/// vectors of the same length for a given instance so callers can rely on
/// [`EmbeddingPipeline::dim`] to size the `vec0` column and validate rows.
pub trait EmbeddingPipeline {
    /// Embeds `text`, returning a vector of exactly [`EmbeddingPipeline::dim`] floats.
    ///
    /// Takes `&mut self`: `ort::Session::run` needs a mutable borrow for
    /// its internal scratch buffers, same as [`crate::vad::SileroVad::infer`].
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// The fixed length every [`EmbeddingPipeline::embed`] call returns.
    fn dim(&self) -> usize;

    /// Identifies the model/config that produced these vectors (stored per
    /// row as `embed_model`, SPEC.md §5.2) — vectors from two different ids
    /// are never comparable and must never share an index (EPIC 10.3).
    fn model_id(&self) -> &str;
}

/// Real CPU embedding pipeline: `sentence-transformers/all-MiniLM-L6-v2`
/// run locally through `ort`, following the same
/// download-or-vendor-the-.onnx-file convention as [`crate::vad::SileroVad`].
///
/// Expects the model directory to contain `model.onnx` and `tokenizer.json`
/// (the two files `optimum-cli export onnx` / the HF `tokenizers` export
/// produce for this model) — not vendored in this repo yet, since the
/// actual weights are a network download this sandbox doesn't have. Once
/// they land at the configured path, [`MiniLmEmbedder::load`] is the only
/// thing that needs to run.
pub struct MiniLmEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    model_id: String,
}

impl MiniLmEmbedder {
    /// Loads `model.onnx` and `tokenizer.json` from `model_dir`.
    ///
    /// `model_id` is the string persisted as each memory row's
    /// `embed_model` (SPEC.md §5.2) — normally `[memory].embed_model` from
    /// config, e.g. `"sentence-transformers/all-MiniLM-L6-v2"`.
    pub fn load(
        model_dir: impl AsRef<Path>,
        model_id: impl Into<String>,
    ) -> Result<Self, EmbedError> {
        let model_dir = model_dir.as_ref();
        let session = Session::builder()
            .map_err(EmbedError::LoadModel)?
            .commit_from_file(model_dir.join("model.onnx"))
            .map_err(EmbedError::LoadModel)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|source| EmbedError::LoadTokenizer {
                path: tokenizer_path,
                source,
            })?;
        Ok(Self {
            session,
            tokenizer,
            model_id: model_id.into(),
        })
    }
}

impl EmbeddingPipeline for MiniLmEmbedder {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(EmbedError::Tokenize)?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let seq_len = ids.len() as i64;

        let input_ids =
            Tensor::from_array(([1i64, seq_len], ids)).map_err(EmbedError::Inference)?;
        let attention_mask =
            Tensor::from_array(([1i64, seq_len], mask.clone())).map_err(EmbedError::Inference)?;
        let token_type_ids =
            Tensor::from_array(([1i64, seq_len], type_ids)).map_err(EmbedError::Inference)?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
            .map_err(EmbedError::Inference)?;

        // `last_hidden_state`: [1, seq_len, MINILM_DIM]. Mean-pool over
        // real (non-padding) tokens per sentence-transformers' own pooling
        // config for this model, then L2-normalize so cosine similarity
        // and Euclidean distance rank identically (both are what `vec0`'s
        // default L2 metric computes over).
        let (_, hidden) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(EmbedError::Inference)?;

        let mut pooled = vec![0.0f32; MINILM_DIM];
        let mut valid_tokens = 0.0f32;
        for (t, &m) in mask.iter().enumerate() {
            if m == 0 {
                continue;
            }
            valid_tokens += 1.0;
            for d in 0..MINILM_DIM {
                pooled[d] += hidden[t * MINILM_DIM + d];
            }
        }
        if valid_tokens > 0.0 {
            for v in pooled.iter_mut() {
                *v /= valid_tokens;
            }
        }
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in pooled.iter_mut() {
                *v /= norm;
            }
        }

        Ok(pooled)
    }

    fn dim(&self) -> usize {
        MINILM_DIM
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
pub mod fake {
    //! A deterministic fake [`super::EmbeddingPipeline`] for storage/
    //! retrieval tests that must not depend on the real ~90MB model +
    //! tokenizer files (unavailable offline, and not vendored in this
    //! repo).

    use super::{EmbedError, EmbeddingPipeline};

    /// Hashes character frequency into a small fixed-dimension vector.
    ///
    /// Not a real embedding — it has no semantic properties — but it is
    /// deterministic (same text always maps to the same vector) and
    /// distinct texts usually land at different points, which is all
    /// [`crate::memory`]'s tests need to exercise store/retrieve and
    /// similarity ranking end to end.
    pub struct FakeEmbedder {
        dim: usize,
        model_id: String,
    }

    impl FakeEmbedder {
        /// Builds a fake embedder that reports `model_id` and always
        /// returns `dim`-length vectors.
        pub fn new(model_id: impl Into<String>, dim: usize) -> Self {
            Self {
                dim,
                model_id: model_id.into(),
            }
        }
    }

    impl EmbeddingPipeline for FakeEmbedder {
        fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
            let mut v = vec![0.0f32; self.dim];
            for (i, byte) in text.bytes().enumerate() {
                v[i % self.dim] += byte as f32;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
            Ok(v)
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }
    }

    #[test]
    fn same_text_embeds_identically() {
        let mut e = FakeEmbedder::new("fake-v1", 16);
        assert_eq!(e.embed("hello").unwrap(), e.embed("hello").unwrap());
    }

    #[test]
    fn different_text_usually_embeds_differently() {
        let mut e = FakeEmbedder::new("fake-v1", 16);
        assert_ne!(e.embed("hello").unwrap(), e.embed("goodbye").unwrap());
    }

    #[test]
    fn vectors_have_the_configured_dimension() {
        let mut e = FakeEmbedder::new("fake-v1", 16);
        assert_eq!(e.embed("hello").unwrap().len(), 16);
        assert_eq!(e.dim(), 16);
    }
}
