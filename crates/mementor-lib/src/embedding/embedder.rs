use anyhow::Context;
use fastembed::{InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};

use crate::config::EMBEDDING_DIMENSION;

/// Wrapper around fastembed's `TextEmbedding` model.
/// Uses the bundled BGE-small-en-v1.5 fp32 ONNX model.
pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Create a new embedder with the bundled BGE-small-en-v1.5 model.
    pub fn new() -> anyhow::Result<Self> {
        let user_model = UserDefinedEmbeddingModel::new(
            include_bytes!("../../../../models/bge-small-en-v1.5/model.onnx").to_vec(),
            TokenizerFiles {
                tokenizer_file: include_bytes!(
                    "../../../../models/bge-small-en-v1.5/tokenizer.json"
                )
                .to_vec(),
                config_file: include_bytes!("../../../../models/bge-small-en-v1.5/config.json")
                    .to_vec(),
                special_tokens_map_file: include_bytes!(
                    "../../../../models/bge-small-en-v1.5/special_tokens_map.json"
                )
                .to_vec(),
                tokenizer_config_file: include_bytes!(
                    "../../../../models/bge-small-en-v1.5/tokenizer_config.json"
                )
                .to_vec(),
            },
        );

        let model =
            TextEmbedding::try_new_from_user_defined(user_model, InitOptionsUserDefined::default())
                .context("Failed to initialize embedding model")?;

        Ok(Self { model })
    }

    /// Embed a batch of text strings and return their vector representations.
    pub fn embed_batch(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        let embeddings = self
            .model
            .embed(owned, None)
            .context("Failed to embed texts")?;
        Ok(embeddings)
    }

    /// Return the embedding dimension (384 for BGE-small-en-v1.5).
    #[must_use]
    pub const fn dimension() -> usize {
        EMBEDDING_DIMENSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_dimension_is_384() {
        assert_eq!(Embedder::dimension(), 384);
    }

    #[test]
    fn embed_batch_returns_correct_count() {
        let mut embedder = Embedder::new().unwrap();
        let texts = &["hello world", "how are you"];
        let embeddings = embedder.embed_batch(texts).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }

    #[test]
    fn embed_single_text() {
        let mut embedder = Embedder::new().unwrap();
        let embeddings = embedder.embed_batch(&["test"]).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);
    }
}
