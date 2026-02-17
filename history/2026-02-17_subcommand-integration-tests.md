# Subcommand-Level Integration Tests

**Date**: 2026-02-17
**Status**: Complete

## Background

Mementor's CLI tests at the subcommand level were incomplete. Some subcommands
had no tests at all, and existing tests called `run_xxx()` functions directly
rather than going through the `try_run()` CLI dispatcher.

### Test Coverage (before → after)

| Subcommand | Before | After | Pattern |
|------------|--------|-------|---------|
| `enable` | 4 (direct `run_enable()`) | 3 (via `try_run()`) | Replaced |
| `ingest` | 0 | 3 | New |
| `query` | 0 | 3 | New |
| `hook stop` | 0 | 3 | New |
| `hook user-prompt-submit` | 0 | 3 | New |

**Net change**: 4 existing tests replaced, 15 new tests added → +11 tests.

## The 5 Rules

1. **Location**: Tests live in the same file as the `run_xxx()` function.
2. **Invocation**: Call `try_run()` with simulated CLI args, not `run_xxx()`
   directly. This tests argument parsing + dispatch + execution as a unit.
3. **DB isolation**: Each test creates its own SQLite instance (in-memory) to
   ensure test isolation.
4. **Seeding**: When seed data is needed, insert it into the in-memory SQLite
   instance, then call `try_run(["subcommand", "arg1", ...], ...)` and verify
   DB state, stdout, and stderr.
5. **Full output matching**: Assert the entire stdout buffer and entire stderr
   buffer with `assert_eq!`, not partial `.contains()` checks.

## Implementation

### Prerequisites

PR #7 completed the DI refactoring that made in-memory DB injection possible:

- `DatabaseDriver` enum (`File` / `InMemory`) as a connection factory
- `Runtime` struct bundling `MementorContext` + `DatabaseDriver`
- `DatabaseDriver::in_memory(name)` uses SQLite shared-cache mode so that
  seeded data is visible to subsequent `open()` calls

### Test Helpers (`crates/mementor-cli/src/test_util.rs`)

| Helper | Purpose |
|--------|---------|
| `runtime_in_memory(name)` | In-memory DB + tempdir context |
| `runtime_not_enabled()` | File DB at nonexistent path (`is_ready()` = false) |
| `seed_memory(driver, embedder, ...)` | Upsert session + embed + insert memory |
| `make_entry(role, text)` | JSONL transcript line |
| `write_transcript(dir, lines)` | Create transcript file |

### Key Design Decisions

- **Real embeddings for seeding**: `seed_memory` uses the real `Embedder` to
  generate vectors. When the test queries with the same text, cosine distance
  is exactly `0.0000` (deterministic), enabling full output matching.
- **`NO_COLOR=1`**: Added `[tasks.test]` to `mise.toml` to prevent ANSI escape
  codes from breaking `assert_eq!` comparisons.
- **Deterministic UUIDs**: `make_entry` uses an `AtomicUsize` counter instead
  of time-based randomness.

### Tests Implemented

**enable.rs** (3 tests, replacing 4 existing):
- `try_run_enable_creates_db_and_configures_project`
- `try_run_enable_is_idempotent`
- `try_run_enable_preserves_existing_settings`

**ingest.rs** (3 new):
- `try_run_ingest_success`
- `try_run_ingest_not_enabled`
- `try_run_ingest_transcript_not_found`

**query.rs** (3 new):
- `try_run_query_with_results`
- `try_run_query_no_results`
- `try_run_query_not_enabled`

**stop.rs** (3 new):
- `try_run_hook_stop_success`
- `try_run_hook_stop_not_enabled`
- `try_run_hook_stop_invalid_json`

**prompt.rs** (3 new):
- `try_run_hook_prompt_with_results`
- `try_run_hook_prompt_not_enabled`
- `try_run_hook_prompt_invalid_json`

## Completed Work

- [x] Implement the 5-rule integration test pattern after DI refactoring
- [x] Create test pattern guide at `docs/testing-patterns.md`
- [x] Add a link from `CLAUDE.md` (Testing section) to `docs/testing-patterns.md`
- [x] Add `[tasks.test]` to `mise.toml` with `NO_COLOR=1`
