# DI Refactoring — Runtime, DatabaseDriver, MementorContext Concrete

**Date**: 2026-02-17
**Status**: Complete

## Background

While designing subcommand-level integration tests, we discovered that the
current architecture prevents injecting in-memory SQLite instances for test
isolation. Subcommands open DB connections internally via
`open_db(context.db_path())`, and there is no DI mechanism to substitute an
in-memory connection.

Additionally, `MementorContext` is a trait with only one implementation
(`RealMementorContext`), adding unnecessary generic `C` type parameters to
every function signature. With 3 external dependencies now being injected
(context, DB, IO), individual parameters don't scale well.

## Goals

1. **`MementorContext`**: Convert from trait to concrete struct. Eliminate the
   `C` type parameter from all function signatures.
2. **`DatabaseDriver`**: New enum that acts as a connection factory. Supports
   file-based SQLite (production) and shared-cache in-memory SQLite (tests).
3. **`Runtime`**: New struct that bundles `MementorContext` + `DatabaseDriver`
   into a single immutable dependency. Future external dependencies (e.g.,
   network) can be added as fields without changing function signatures.

## Design Decisions

### MementorContext: trait → struct

`RealMementorContext` is the only implementation. Tests use it with
`tempfile::tempdir()` paths. No mock implementation exists or is needed.
Removing the trait simplifies all generic bounds from `<C: MementorContext>`
to nothing.

### DatabaseDriver: enum, not trait

Only 2 variants needed (File, InMemory). An enum is simpler than a trait —
no dynamic dispatch, no trait objects, and the variants are exhaustively
matched.

### Runtime: struct, not trait

Bundles `context` and `db` as public fields. `io` stays as a separate `&mut`
parameter because it requires mutable access. Adding future dependencies
means adding a field to `Runtime`, not changing function signatures.

### In-memory SQLite: shared-cache mode

SQLite URI `file:{name}?mode=memory&cache=shared` allows multiple connections
to share the same named in-memory database within a process. An "anchor"
connection kept alive in `DatabaseDriver::InMemory` prevents the database from
being destroyed. Subsequent `open()` calls create new connections to the same
shared database.

## TODO

- [x] Create history documents
- [x] Create feature branch (`feat/di-refactoring`, no worktree)
- [x] `MementorContext` trait → concrete struct
- [x] Add `open_db_in_memory()` to connection.rs
- [x] Create `DatabaseDriver` enum (db/driver.rs)
- [x] Create `Runtime` struct (runtime.rs)
- [x] Update `try_run()` and all subcommands to use `Runtime`
- [x] Update `main.rs` and fix existing tests
- [x] Verify: 68 tests pass, clippy 0 warnings

## Results

- **Tests**: 62 → 68 (6 new: 2 in connection.rs, 4 in driver.rs)
- **New files**: `runtime.rs`, `db/driver.rs`
- **Modified files**: `context.rs`, `lib.rs` (both crates), `main.rs`,
  `enable.rs`, `ingest.rs`, `query.rs`, `stop.rs`, `prompt.rs`, `logging.rs`,
  `test_util.rs` (both crates), `db/mod.rs`
- **Removed**: `MementorContext` trait, `RealMementorContext` name,
  `C: MementorContext` generic bound from all functions

## Future Work

- Write subcommand-level integration tests using `DatabaseDriver::in_memory()`
  (see `history/2026-02-17_subcommand-integration-tests.md`)
