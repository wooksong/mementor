# Worktree DB Sharing

## Background

Mementor determines the SQLite database path from `std::env::current_dir()` →
`<cwd>/.mementor/mementor.db`. In a git worktree, cwd is the worktree
directory, so each worktree creates an isolated empty DB. Cross-session memory
recall from the main repo is lost. Since `.mementor/` is gitignored, it never
propagates to worktrees.

## Goals

1. All git worktrees share the primary worktree's `.mementor/mementor.db`
2. `mementor enable` is restricted to the primary worktree only
3. Submodules are skipped during root resolution (treated as part of the
   parent project)
4. Concurrent DB access from multiple worktrees is safe (WAL mode)

## Design Decisions

### Pure Rust file-based worktree detection

Instead of shelling out to `git rev-parse --git-common-dir`, we read `.git`
files and the `commondir` file directly. This avoids subprocess overhead on
every hook invocation (hooks fire on every prompt).

### Walk-up algorithm for `.git` discovery

Like git itself, we walk up from cwd to the filesystem root looking for `.git`.
This supports running mementor from subdirectories, not just the repo root.

### Submodule handling

A submodule's `.git` is a file pointing to `<parent>/.git/modules/<name>`.
Unlike linked worktrees, there is no `commondir` file in the gitdir. When we
encounter a `.git` file without `commondir`, we skip it and continue walking up
to find the actual project root. This ensures submodules are treated as part of
the parent project.

### `ResolvedWorktree` enum for symlink-safe classification

Initial implementation used `is_primary_worktree(cwd)` in the `enable` guard,
which checks `cwd.join(".git").is_dir()`. This has two problems:

1. **Subdirectory bug**: Running `mementor enable` from `project/src/` would be
   rejected because `project/src/.git` doesn't exist.
2. **Path comparison is fragile**: Alternatives like `cwd.starts_with(project_root)`
   break when symlinks are involved (`resolve_primary_root` canonicalizes linked
   worktree paths but `std::env::current_dir()` may not match).

The fix: `resolve_worktree(cwd)` returns a `ResolvedWorktree` enum
(`Primary(PathBuf)` / `Linked(PathBuf)` / `NotGitRepo`) determined at startup
by `.git` entry type inspection. The `is_linked` flag is stored in
`MementorContext` and checked by the `enable` guard — no path comparison needed.

### `cwd` field in `MementorContext`

We store both the original working directory (`cwd`) and the resolved primary
root (`project_root`) in `MementorContext`. The `is_linked_worktree` flag
records whether we are in a linked worktree, determined at startup from the
`.git` entry type.

### WAL mode for concurrent access

With multiple worktrees sharing a DB, concurrent writes become possible. SQLite
WAL mode + `busy_timeout` ensures safe concurrent access.

## TODO

- [x] Create history document
- [x] Create `crates/mementor-lib/src/git.rs` with `resolve_primary_root` and
  `is_primary_worktree`
- [x] Add `cwd` field to `MementorContext` in `context.rs`
- [x] Wire `resolve_primary_root` in `main.rs`
- [x] Add primary worktree guard to `enable` command
- [x] Enable WAL mode + `busy_timeout` in `connection.rs`
- [x] Pass `cargo clippy -- -D warnings` and `cargo test` (90 tests, 0 failures)
- [x] Fix enable guard subdirectory bug: replace `is_primary_worktree(cwd)`
  with `ResolvedWorktree` enum + `is_linked_worktree` context flag
- [x] Remove dead code: `resolve_primary_root` and `is_primary_worktree`
  (subsumed by `resolve_worktree` + `ResolvedWorktree` enum)
- [x] Pass `cargo clippy -- -D warnings` and `cargo test` (134 tests, 0 failures)
- [x] Fix CI: add `protocol.file.allow=always` to `run_git` helper for
  submodule tests (Git 2.38.1+ blocks local file transport by default)
- [x] Fix enable guard tests to follow testing guidelines: full output
  matching (`assert_eq!`), stdout/stderr verification, real git repo setup
  kept for linked worktree test
- [x] Enforce testing guidelines reference in CLAUDE.md
- [x] Extract shared `mementor-test-util` crate with `init_git_repo`,
  `run_git`, `assert_paths_eq` — used by both `mementor-lib` and `mementor-cli`
- [x] Shorten fully-qualified `mementor_lib::` paths in enable guard tests
  with `use` imports

## Results

- 10+ files modified/created
- 137 tests passing (was 77 before; +9 git tests, +2 connection tests,
  +3 context tests, +3 enable guard tests, +3 test-util self-tests)
- Clippy clean with `-D warnings`

## Future Work

- None identified yet
