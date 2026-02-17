/// Embedding dimension for BGE-small-en-v1.5.
pub const EMBEDDING_DIMENSION: usize = 384;

/// Target chunk size in tokens for markdown-aware sub-chunking.
pub const CHUNK_TARGET_TOKENS: usize = 256;

/// Number of overlap tokens between adjacent sub-chunks within a turn.
pub const CHUNK_OVERLAP_TOKENS: usize = 40;

/// Default number of top-k results for vector similarity search.
pub const DEFAULT_TOP_K: usize = 5;
