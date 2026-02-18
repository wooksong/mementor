# Improve Recall Quality: Post-Search Filtering Pipeline

## Background

The `UserPromptSubmit` hook embeds the user's prompt and searches for similar
memories via `vector_full_scan`. Currently, the raw top-k results are returned
without post-processing. This causes self-referential results — meta-questions
like "recall한 컨텍스트가 뭐였어?" match other instances of similar
meta-questions rather than substantive project context.

Root causes:
1. **In-context results waste slots** — same-session memories still in Claude's
   context window are returned but provide no new information.
2. **Duplicate turns dominate** — multiple sub-chunks of the same turn occupy
   all k slots with redundant content.
3. **No distance threshold** — high-distance (irrelevant) results are included.

## Goals

- Add a multi-phase post-search filtering pipeline to `search_context`.
- Filter out in-context results at the SQL level.
- Apply distance threshold to remove irrelevant results.
- Deduplicate by turn, keeping only the best-scoring chunk per turn.
- Reconstruct full turn text from sibling chunks after dedup.
- Maintain backward compatibility (no API signature changes for callers).

## Design Decisions

1. **In-context filtering in SQL** — extend `search_memories` with
   `exclude_session_id` and `compact_boundary` parameters. WHERE clause filters
   before results reach Rust code.
2. **Over-fetch** — `vector_full_scan` returns top-k before WHERE, so fetch
   `k * 4` candidates to compensate for filtering.
3. **Distance threshold** — `MAX_COSINE_DISTANCE = 0.45`. BGE-small-en-v1.5
   semantically related content clusters below 0.40.
4. **Turn reconstruction** — `get_turns_chunks` batch query retrieves all
   sibling chunks in one SQL round-trip. Chunks concatenated with `"\n\n"`.
   Overlap tokens (~40 per boundary) are acceptable duplication.
5. **Single batch query for reconstruction** — avoids repeated connection
   overhead in mementor's short-lived hook process.

## TODO

- [x] Create feature worktree
- [x] Create history document
- [x] Add config constants (`OVER_FETCH_MULTIPLIER`, `MAX_COSINE_DISTANCE`)
- [x] Extend `search_memories` with in-context filter params
- [x] Add `get_turns_chunks` batch query
- [x] Add tests for `get_turns_chunks`
- [x] Rewrite `search_context` with 5-phase filter pipeline
- [x] Update existing `search_returns_relevant_results` test
- [x] Add 7 new unit tests for filter pipeline
- [x] Verify: clippy + all tests pass
- [x] Update history document with results
- [x] Commit (`cf7883d`)

## Results

- **Clippy**: zero warnings
- **Tests**: 105 total (38 cli + 67 lib), all passing
- Previous test count: 96 (from PR #10) → now 105 (+9 new tests)
- **Deviation**: `ensure_session` test helper needed to set
  `last_compact_line_index` via raw SQL because `upsert_session` intentionally
  does not update that column.

### Files Changed

| File | Change |
|------|--------|
| `crates/mementor-lib/src/config.rs` | +2 constants |
| `crates/mementor-lib/src/db/queries.rs` | Extended `search_memories`, added `get_turns_chunks`, +2 tests |
| `crates/mementor-lib/src/pipeline/ingest.rs` | Rewrote `search_context` (5-phase pipeline), fixed existing test, +7 tests |

## Future Work

- MMR (Maximal Marginal Relevance) diversity re-ranking if results still cluster
  too tightly after these filters.
- Session-level diversity limits (max N results per session).
- Consider making `MAX_COSINE_DISTANCE` tunable via CLI flag or config file.
