# Mementor -- Agent Instructions

## Project Overview

Mementor is a local RAG memory agent for Claude Code. It vectorizes
conversation transcripts into a local SQLite vector database and automatically
recalls relevant past context via Claude Code lifecycle hooks. The goal is
cross-session context persistence without any external API dependencies.

## Tech Stack

- **Language**: Rust, edition 2024 (resolver 3)
- **Toolchain**: Rust 1.93.1 (managed via mise, see `mise.toml`)
- **Database**: SQLite via `rusqlite` with the `bundled` feature (statically
  linked SQLite)
- **Vector search**: sqlite-vector -- uses **BLOB columns and custom SQL
  functions** (`vector_distance_cos`, etc.), **NOT virtual tables**. The C
  sources live in `vendor/sqlite-vector/` and are statically compiled via the
  `cc` crate in `build.rs`.
- **Embedding**: `fastembed-rs` with the bundled BGE-small-en-v1.5 fp32 ONNX
  model (384 dimensions). Model files are in `models/bge-small-en-v1.5/`.
- **Text splitting**: `text-splitter` crate with `MarkdownSplitter`
- **Schema migration**: `rusqlite_migration`
- **Error handling**: `anyhow`
- **Logging**: `tracing`
- **CLI**: `clap`

## Constraints

- **Static linking required**: All native dependencies (SQLite, sqlite-vector)
  must be statically linked. No runtime shared library dependencies.
- **No external API dependencies**: Everything runs locally. No network calls
  for embedding, search, or any other operation.
- **macOS only (Milestone 1)**: Target Apple Silicon (ARM64) and Intel (x86_64)
  Macs. Do not add Linux or Windows-specific code yet.

## Directory Structure

```
mementor/
  Cargo.toml              Workspace root (members: crates/*)
  mise.toml               Rust toolchain version
  AGENTS.md               This file (symlinked as CLAUDE.md)
  README.md               Project README

  crates/
    mementor-lib/         Core library
      src/lib.rs          Library root
      build.rs            Compiles sqlite-vector C sources via cc
    mementor-cli/         CLI layer with DI
      src/lib.rs          CLI command dispatch
    mementor-main/        Thin binary entry point
      src/main.rs         main() -- wires DI, calls CLI

  vendor/
    sqlite-vector/
      src/                C source files
      libs/fp16/          FP16 support headers

  models/
    bge-small-en-v1.5/    ONNX model files

  history/                Task documents (one per session/milestone)
  scripts/                Build and utility scripts
```

### Crate Responsibilities

- **mementor-lib**: All business logic. Database schema and operations,
  embedding pipeline, transcript parsing, turn grouping, text chunking, vector
  search. Defines DI traits (`MementorContext`, `ConsoleIO<IN, OUT, ERR>`).
  No direct CLI or I/O concerns.

- **mementor-cli**: CLI command implementations using `clap`. Each command
  accepts trait objects from mementor-lib for testability. Handles argument
  parsing, output formatting, and command dispatch.

- **mementor-main**: The `[[bin]]` crate (binary name: `mementor`). Constructs
  real implementations of all traits and passes them to mementor-cli. Should
  contain minimal logic -- just wiring.

## Git Worktree

**Always use the `/worktree` skill when managing git worktrees.** Do not run
`git worktree add` or `git worktree remove` directly. The `/worktree` skill
handles mise environment setup (copying `mise.local.toml`, trusting config
files, installing the toolchain) and branch cleanup automatically.

Background: mise does not auto-trust config files outside the original
repository path. Without proper setup, `cargo` and other mise-managed tools
will not be available inside the worktree. The `/worktree` skill ensures this
is handled correctly.

## Build

```bash
cargo build
```

The `build.rs` in mementor-lib automatically compiles the sqlite-vector C
sources from `vendor/sqlite-vector/src/` using the `cc` crate. No manual steps
are needed.

For a release build:

```bash
cargo build --release
```

### ONNX Model

The BGE-small-en-v1.5 ONNX model is embedded into the binary at compile time
via `include_bytes!`. The model file must exist before building:

```bash
mise run model:download
```

This downloads `models/bge-small-en-v1.5/model.onnx` from Hugging Face Hub if
it is not already present. If you cloned with Git LFS, the file is already
available and the script is a no-op.

### ONNX Runtime on x86_64 macOS (Intel Mac)

`ort-sys` (the ONNX Runtime binding used by `fastembed`) does **not** provide
prebuilt static binaries for `x86_64-apple-darwin`. Microsoft dropped x86_64
macOS binaries starting with ONNX Runtime 1.24.1 (only `aarch64-apple-darwin`
is supported).

**Platform-specific handling**: The fastembed dependency uses target-specific
features in `crates/mementor-lib/Cargo.toml`:

- **Apple Silicon** (aarch64): `ort-download-binaries` — statically links a
  prebuilt ONNX Runtime binary. No runtime dependency needed.
- **Intel Mac** (x86_64): `ort-load-dynamic` — loads `libonnxruntime.dylib` at
  runtime via `dlopen`. Requires Homebrew installation.

**Setup** (required on Intel Macs only):

```bash
brew install onnxruntime
```

The `ORT_DYLIB_PATH` environment variable is configured in `mise.local.toml`
(gitignored, machine-local). If you're on an Intel Mac and the file doesn't
exist, create it:

```toml
# mise.local.toml
[env]
ORT_DYLIB_PATH = "/usr/local/lib/libonnxruntime.dylib"
```

**Do NOT** add `ort-download-binaries` to the x86_64 target — prebuilt
binaries do not exist for `x86_64-apple-darwin` and the build will fail.

## sqlite-vector Integration

sqlite-vector (v0.9.90) provides vector operations via **custom SQL functions**
and a **`vector_full_scan` virtual table** for similarity search.

Key points:
- Vector data is stored as BLOB in regular SQLite columns using
  `vector_as_f32(json_text)` for conversion.
- After schema creation, call `vector_init('table', 'column', 'type=f32,
  dimension=384, distance=cosine')` to register the table with the extension.
  This must be done on every new connection.
- Similarity search uses the `vector_full_scan` virtual table:
  ```sql
  SELECT vs.id, vs.distance
  FROM vector_full_scan('memories', 'embedding', ?query_json, ?k) vs
  JOIN memories m ON m.rowid = vs.id
  ```
  Arguments: `(table_name TEXT, column_name TEXT, query_vector TEXT, k INTEGER)`.
- The extension (`sqlite3_vector_init`) must be loaded via FFI into each
  `rusqlite` connection after opening.
- The C sources are compiled to a static library and linked at build time.
- On x86_64 macOS, AVX2/AVX512 distance functions are stubbed to fall back to
  SSE2 due to Apple Clang compile-time constant initializer limitations.

## Incremental Update Pattern

Mementor tracks ingestion progress per transcript file using two values:

- **`last_line_index`**: The line number (0-based) of the last successfully
  processed line in the transcript JSONL file. On the next ingestion, reading
  starts from `last_line_index + 1`.
- **`provisional_turn_start`**: The line index where the current incomplete
  (provisional) turn begins. If a transcript ends mid-conversation, the
  incomplete turn is stored provisionally. On the next ingestion, the
  provisional turn is deleted and re-processed with the now-complete data.

This ensures:
- No duplicate processing of already-ingested content.
- Incomplete turns are properly completed on subsequent ingestion passes.
- The system is resilient to interruptions.

## Turn-Based Chunking

A **Turn** groups consecutive messages for semantic coherence:

```
Turn[n] = User[n] + Assistant[n] + User[n+1]
```

- `User[n]`: The user's prompt.
- `Assistant[n]`: The assistant's response.
- `User[n+1]`: The next user prompt, included as **forward context**. This
  provides a hint about what the assistant's response was really addressing,
  improving embedding quality.

When a turn exceeds the model's token limit, it is split into sub-chunks using
`MarkdownSplitter`. Adjacent sub-chunks within the same turn share **~40 tokens
(~15%) overlap** via post-processing to preserve semantic continuity at split
boundaries.

## Coding Conventions

### Rust Edition and Style

- **Edition 2024** -- use all edition 2024 features and idioms.
- Follow standard Rust formatting (`cargo fmt`).
- Use `anyhow::Result` for fallible functions. Use `anyhow::Context` for adding
  context to errors.
- Use `tracing` for logging (`tracing::info!`, `tracing::debug!`, etc.).

### Linting

```bash
cargo clippy -- -D warnings
```

This command **must pass with zero warnings**. All clippy lints at `warn` level
for `all` and `pedantic` groups are enabled in the workspace `Cargo.toml`. The
following lints are explicitly allowed:
- `module_name_repetitions`
- `must_use_candidate`
- `missing_errors_doc`
- `missing_panics_doc`

### Deno Scripts

Deno TypeScript scripts live under `.claude/`. Follow the conventions in
[`docs/deno-script-conventions.md`](docs/deno-script-conventions.md).

### Dependency Management

Use `cargo add` to add dependencies. **Do not edit `Cargo.toml` dependency
sections directly.** This ensures proper version resolution and formatting.

Example:
```bash
cargo add -p mementor-lib anyhow
cargo add -p mementor-lib --build cc
```

### Git Commits

**Always use the `/commit` skill when creating git commits.** Do not run
`git commit` directly. The `/commit` skill enforces the project's commit
conventions and must be used for every commit without exception.

### Language Rule

**All documents, code comments, commit messages, and user-facing strings must
be written in English only.**

### Task Documents

Task documents record what was done in each working session. They live in the
`history/` directory with the naming convention:

```
history/YYYY-MM-DD_what-we-do.md
```

Each document includes background, goals, design decisions, a TODO checklist,
and future work items.

## Workflow

Every implementation task **must** follow this workflow. **When creating
implementation plans (e.g., in plan mode), explicitly include every step
below.** Do not omit or assume any step is implicit.

1. **Create a feature branch**: Use the `AskUserQuestion` tool to ask the user
   whether to use a separate worktree or the current directory. If worktree,
   run `/worktree add <branch>`. If current directory, use `git checkout -b`.

2. **Create a history document**: Before writing any code, create a task
   document at `history/YYYY-MM-DD_task-name.md` with background, goals,
   design decisions, and a TODO checklist. This document is the implementation
   plan — do not start coding until it exists.

3. **Implement and track progress**: Use todo tracking throughout the session.
   Mark items as in-progress when starting and completed when done.

4. **Update the history document**: Before every commit, update the history
   document with current results, any deviations from the original plan, and
   future work items. Keep it up to date as work progresses.

5. **Commit via `/commit`**: Use the `/commit` skill for every commit. Do not
   run `git commit` directly. Always update the history document (step 4)
   before committing.

6. **Complete all TODOs before creating a PR**: Every TODO item in the history
   document must be done before opening a pull request. If any item is found
   to be infeasible during implementation, move it to a "Future work" section
   with an explanation -- do not leave unfinished TODOs.

## Testing

Run all tests (unit + integration):

```bash
mise run test
```

This runs `NO_COLOR=1 cargo test` to prevent ANSI escape codes from interfering
with output assertions.

Run unit tests only:

```bash
mise run test:unit
```

Tests should be colocated with the code they test (in `#[cfg(test)]` modules)
for unit tests. Integration tests go in `tests/` directories within each crate.

For the standard subcommand-level integration test pattern (5 rules, helpers,
and annotated examples), see [`docs/testing-patterns.md`](docs/testing-patterns.md).
