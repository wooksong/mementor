# Recall Quality v3: Query-Side Improvement

## Motivation

Recall quality v2 improves the **indexing/storage side** — better embeddings
via thinking blocks (R1) and tool context (R2), structured file-path lookup
(R3), and metadata storage (R4/R5). But the **query side** is untouched:
the raw user prompt is passed directly to the embedding model with zero
preprocessing.

This produces two failure modes:

1. **Trivial prompts waste compute and inject noise.** Short operational
   commands ("push", "/commit", "ok", "check ci") produce vague embeddings
   that match past instances of similar short commands (distance 0.27–0.35)
   rather than substantive project context. These results consume context
   tokens with no value.

2. **Non-trivial prompts lack context.** Even reasonable queries like "why
   was this file changed?" lack the specificity that session context could
   provide. After v2 Task 3, the `file_mentions` table records which files
   were touched in the current session. Appending these to the query before
   embedding biases similarity toward relevant turns.

## Problem Analysis

### Current query flow

```
UserPromptSubmit hook stdin:
  { session_id, prompt (nullable), cwd }

prompt.rs handle_prompt():
  1. Early return if prompt is empty
  2. Early return if DB not ready
  3. search_context(conn, embedder, &input.prompt, k=5, session_id)
  4. Write results to stdout if non-empty
```

No query classification, no enrichment, no context augmentation.

### Observed failure examples

| Prompt | Distance | Matched content | Useful? |
|--------|----------|-----------------|---------|
| "push" | 0.28 | Past "push" conversation | No |
| "/commit" | 0.31 | Past commit conversation | No |
| "check ci" | 0.33 | Past CI check conversation | No |
| "ok" | 0.35 | Past acknowledgment | No |

All produce low-quality matches against superficially similar past prompts
rather than substantive project context.

## Requirements

### R1: Query Classification

Classify the incoming prompt and skip recall entirely for trivial inputs.
These produce poor embeddings that match noise rather than substance.

Trivial prompt categories:
- Slash commands (`/commit`, `/worktree`, `/review-pr`)
- Short phrases with fewer than 3 words ("push", "ok", "check ci")
- Acknowledgment patterns ("sounds good", "lgtm", "thanks", "got it")

### R2: Query Enrichment

Augment non-trivial prompts with session context before embedding. After
v2 Task 3, the `file_mentions` table records which files were touched in the
current session. Appending recently-touched filenames transforms vague queries
into contextually grounded ones.

Example:
```
Original: "why was this changed?"
Enriched: "why was this changed?\n\n[Context: recently touched files: enable.rs, git.rs, query.rs]"
```

## Architecture Overview

All query-side logic lives in a new pipeline module. The search pipeline
(`search_context`) remains unchanged — it receives a (possibly enriched)
query string and searches as before.

```
UserPromptSubmit hook
  |
  +-- [existing] Early return if empty prompt or DB not ready
  |
  +-- [R1] classify_query(prompt) -> QueryClass
  |     Trivial? -> debug log + return (no recall)
  |
  +-- [R2] enrich_query(conn, prompt, session_id) -> enriched_query
  |     Append session file context from file_mentions
  |
  +-- [existing] search_context(conn, embedder, enriched_query, k, session_id)
  +-- [existing] Write results to stdout
```

### New module: `pipeline/query.rs`

```rust
#[derive(Debug, PartialEq)]
pub enum QueryClass {
    Searchable,
    Trivial { reason: &'static str },
}

pub fn classify_query(prompt: &str) -> QueryClass;
pub fn enrich_query(conn: &Connection, query: &str, session_id: Option<&str>) -> Result<String>;
```

### Classification rules (applied in order)

1. **Slash commands:** `prompt.trim().starts_with('/')` → `Trivial("slash command")`
2. **Word count:** Split on whitespace, count < `MIN_QUERY_WORDS` (3)
   → `Trivial("too short")`
3. **Acknowledgments:** Case-insensitive exact match against static set
   ("ok", "okay", "y", "yes", "no", "sure", "thanks", "lgtm", "done",
   "next", "continue", "go ahead", "proceed", "sounds good", "makes sense",
   "got it") → `Trivial("acknowledgment")`

### Enrichment strategy

1. Skip if `session_id` is `None` (handles `mementor query` CLI).
2. Query `file_mentions` for the current session's recently-touched files:
   ```sql
   SELECT DISTINCT file_path FROM file_mentions
   WHERE session_id = ?1 ORDER BY line_index DESC LIMIT ?2
   ```
3. Extract filename component only (not full path) to avoid machine-specific
   noise.
4. Append: `{query}\n\n[Context: recently touched files: foo.rs, bar.rs]`
5. If no files found, return query unchanged.

### Integration points

| Location | Change |
|----------|--------|
| `crates/mementor-cli/src/hooks/prompt.rs` | Add classification + enrichment before `search_context` |
| `crates/mementor-cli/src/commands/query.rs` | Add classification only (no session context) |

## Design Constraints

- **No new hooks**: Classification and enrichment happen inside the existing
  `UserPromptSubmit` hook handler.
- **No API changes to `search_context`**: It still takes a query string.
  Preprocessing happens at the call site.
- **Zero-cost for trivial prompts**: Classification is O(1) pattern matching.
  No embedding or DB access needed for skipped prompts.
- **Graceful degradation**: If `file_mentions` table doesn't exist yet
  (v2 Task 3 not deployed), enrichment returns the original query unchanged.

## Implementation Plan

| Task | Document | Scope | Depends On |
|------|----------|-------|------------|
| 1 | (this document, Part A) | query.rs + config + prompt.rs + query cmd | -- |
| 2 | (this document, Part B) | query.rs + queries.rs + prompt.rs | v2 Task 3 |

**Dependency:** Task 1 (classification) is independent. Task 2 (enrichment)
depends on v2 Task 3 (file_mentions table).

## Key Files

| File | Change |
|------|--------|
| `crates/mementor-lib/src/pipeline/query.rs` | **NEW**: `classify_query()`, `enrich_query()`, `QueryClass` |
| `crates/mementor-lib/src/pipeline/mod.rs` | Add `pub mod query;` |
| `crates/mementor-lib/src/config.rs` | `MIN_QUERY_WORDS`, `MAX_ENRICHMENT_FILES` |
| `crates/mementor-lib/src/db/queries.rs` | `get_session_file_paths()` |
| `crates/mementor-cli/src/hooks/prompt.rs` | Classification + enrichment integration |
| `crates/mementor-cli/src/commands/query.rs` | Classification integration |

## TODO

### Part A: Query Classification

- [ ] Add `MIN_QUERY_WORDS` constant to `config.rs`
- [ ] Create `crates/mementor-lib/src/pipeline/query.rs`
- [ ] Add `pub mod query;` to `pipeline/mod.rs`
- [ ] Implement `QueryClass` enum
- [ ] Implement `classify_query()` with slash, word-count, acknowledgment rules
- [ ] Integrate into `handle_prompt` in `prompt.rs` (early return on Trivial)
- [ ] Integrate into `run_query` in `query.rs` (user-facing message on Trivial)
- [ ] Add test: `classify_slash_commands`
- [ ] Add test: `classify_short_prompts`
- [ ] Add test: `classify_acknowledgments`
- [ ] Add test: `classify_searchable_prompts`
- [ ] Add test: `classify_whitespace_handling`
- [ ] Add test (CLI): `try_run_hook_prompt_trivial_skipped`
- [ ] Add test (CLI): `try_run_query_trivial_message`

### Part B: Query Enrichment

- [ ] Add `MAX_ENRICHMENT_FILES` constant to `config.rs`
- [ ] Implement `get_session_file_paths()` in `queries.rs`
- [ ] Implement `enrich_query()` in `query.rs`
- [ ] Integrate into `handle_prompt` in `prompt.rs`
- [ ] Add test: `enrich_query_with_files`
- [ ] Add test: `enrich_query_no_files`
- [ ] Add test: `enrich_query_no_session`
- [ ] Add test: `enrich_query_extracts_filenames`
- [ ] Add test: `enrich_query_caps_at_max_files`
- [ ] Add test: `get_session_file_paths_returns_distinct`
- [ ] Add test: `get_session_file_paths_ordered_by_recency`
- [ ] Add test (CLI): `try_run_hook_prompt_enriched_search`

### Finalization

- [ ] Verify: clippy + all tests pass

## Previous Work

- [recall-quality-v2](2026-02-18_recall-quality-v2.md) — the indexing/storage
  side improvements that this work complements
- [improve-recall-quality](2026-02-18_improve-recall-quality.md) — the
  5-phase post-search filter pipeline

## Estimated Scope

~120 lines of code + ~150 lines of test. No migration needed.
