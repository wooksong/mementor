# Subcommand-Level Integration Tests

**Date**: 2026-02-17
**Status**: Deferred (prerequisite DI refactoring needed first)

## Background

Mementor's CLI tests at the subcommand level were incomplete. Some subcommands
had no tests at all, and existing tests called `run_xxx()` functions directly
rather than going through the `try_run()` CLI dispatcher.

### Current Test Coverage (before this task)

| Subcommand | Tests | Pattern |
|------------|-------|---------|
| `enable` | 4 | Direct `run_enable()` call |
| `ingest` | 0 | Only lib-level pipeline tests |
| `query` | 0 | None |
| `hook stop` | 0 | Only input parsing tests |
| `hook user-prompt-submit` | 0 | Only input parsing tests |

## Goals

Establish a consistent integration test pattern for all subcommands with these
5 rules:

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

## Design Plan

### Test Helpers (`test_util.rs`)

- `write_transcript(dir, lines)` — create JSONL transcript file
- `make_entry(role, text)` — build a transcript JSONL line
- `seed_memory(db_path, session_id, content)` — insert seed data with real
  embeddings

### Planned Tests Per Subcommand

**enable.rs** (replace 4 existing tests):
- `enable_creates_db_and_configures_project`
- `enable_is_idempotent`
- `enable_preserves_existing_settings`

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

## Technical Bottleneck

Rule 3 (in-memory SQLite isolation) conflicts with the current architecture:

- Subcommands open DB connections internally via
  `open_db(context.db_path())` — a file-path-based function.
- `try_run()` dispatches to subcommands which call `open_db()` themselves.
- There is no way to inject an in-memory `:memory:` connection because the
  subcommand owns the DB lifecycle.
- For seeding, the test would insert data into an in-memory DB, but the
  subcommand would open a **different** in-memory DB (each
  `Connection::open(":memory:")` creates a new database).

### Failed Approach: tempdir

Using `tempfile::tempdir()` with file-based SQLite achieves test isolation but
violates Rule 3 (each test should use an in-memory instance). The user
explicitly requested in-memory DB injection via DI refactoring.

## Decision

Pivot to a prerequisite DI refactoring session:

1. Add `DatabaseDriver` enum (File / InMemory) as a connection factory.
2. Convert `MementorContext` from trait to concrete struct (eliminate
   unnecessary `C` type parameter).
3. Bundle dependencies into a `Runtime` struct for future extensibility.

Once the DI refactoring is complete, the integration tests can be written
using `DatabaseDriver::in_memory()` with SQLite shared-cache mode.

## Future Work

- [ ] Implement the 5-rule integration test pattern after DI refactoring
- [ ] Create test pattern guide at `docs/testing-patterns.md` documenting the
  5-rule integration test pattern, helper utilities, and examples so that future
  test authors follow a consistent approach
- [ ] Add a link from `CLAUDE.md` (Testing section) to `docs/testing-patterns.md`
