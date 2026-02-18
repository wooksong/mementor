/// Embedding dimension for BGE-small-en-v1.5.
pub const EMBEDDING_DIMENSION: usize = 384;

/// Target chunk size in tokens for markdown-aware sub-chunking.
pub const CHUNK_TARGET_TOKENS: usize = 256;

/// Number of overlap tokens between adjacent sub-chunks within a turn.
pub const CHUNK_OVERLAP_TOKENS: usize = 40;

/// Default number of top-k results for vector similarity search.
pub const DEFAULT_TOP_K: usize = 5;

/// Multiplier for over-fetching candidates from vector search.
///
/// The search fetches `k * OVER_FETCH_MULTIPLIER` candidates, then applies
/// post-filters (in-context removal, distance threshold, turn dedup) to
/// produce the final `k` results.
pub const OVER_FETCH_MULTIPLIER: usize = 4;

/// Maximum cosine distance for a search result to be considered relevant.
///
/// Results with distance above this threshold are discarded.
/// BGE-small-en-v1.5 cosine distance range: \[0, 2\].
/// Semantically related content typically falls below 0.40.
pub const MAX_COSINE_DISTANCE: f64 = 0.45;
