# Init Mementor

**Date**: 2026-02-17
**Status**: Complete

## Background

Claude Code cannot persist context across sessions. Each new conversation starts
from scratch, losing valuable project knowledge, design decisions, and
implementation history accumulated in previous sessions.

Mementor is a local RAG (Retrieval-Augmented Generation) memory agent that
solves this problem. It vectorizes conversation content from Claude Code
transcripts into a local SQLite Vector DB and automatically recalls relevant
past context via Claude Code lifecycle hooks. When a new session begins,
Mementor searches for semantically similar past conversations and injects them
as context, giving Claude Code a persistent memory without relying on any
external APIs.

## Goals

- **Project structure setup**: Establish a 3-crate Cargo workspace with clear
  separation of concerns (core library, CLI with DI, thin binary entry point).
- **sqlite-vector C source static compilation**: Integrate sqlite-vector C
  sources via the `cc` crate in `build.rs`, producing a statically linked
  artifact with no runtime shared library dependencies.
- **Incremental-update SQLite schema**: Design a schema that supports
  incremental ingestion of transcripts, tracking `last_line_index` and
  `provisional_turn_start` to avoid re-processing already ingested content.
- **Turn-based chunking with forward context and sub-chunk overlap**: Implement
  a chunking strategy where a Turn = [User[n] + Assistant[n] + User[n+1]], with
  sub-chunk overlap of ~40 tokens (~15%) between sub-chunks within a turn.
- **CLI commands**: Implement `enable`, `ingest`, `query`, `hook stop`, and
  `hook user-prompt-submit` subcommands.
- **Hook integration**: Wire up Claude Code lifecycle hooks (Stop and
  UserPromptSubmit) so that ingestion and recall happen automatically.
- **Comprehensive tests**: Unit tests for each module and integration tests for
  end-to-end ingestion and query pipelines.

## Tech Stack

| Component         | Choice                                                         |
| ----------------- | -------------------------------------------------------------- |
| Language          | Rust (edition 2024, resolver 3)                                |
| Rust toolchain    | 1.93.1 (via mise)                                              |
| Database          | SQLite via `rusqlite` (bundled feature)                        |
| Vector extension  | sqlite-vector (Elastic License 2.0), statically compiled via cc|
| Embedding model   | BGE-small-en-v1.5, fp32 ONNX, bundled via `fastembed-rs`      |
| Text splitting    | `text-splitter` crate with `MarkdownSplitter`                  |
| Schema migration  | `rusqlite_migration`                                           |
| Error handling    | `anyhow`                                                       |
| Logging           | `tracing`                                                      |
| CLI parsing       | `clap`                                                         |

## Design Decisions

### 3-Crate Cargo Workspace

The project is organized as a Cargo workspace with three crates:

```
crates/
  mementor-lib/    # Core library: DB, embedding, chunking, schema
  mementor-cli/    # CLI layer with DI: command dispatch, argument parsing
  mementor-main/   # Thin binary entry point: wires DI and calls CLI
```

- **mementor-lib**: Contains all business logic. No I/O assumptions beyond file
  system and SQLite. Defines traits for dependency injection.
- **mementor-cli**: Implements CLI commands using `clap`. Depends on
  mementor-lib. Uses trait-based DI for testability.
- **mementor-main**: The `[[bin]]` crate named `mementor`. Constructs real
  implementations and passes them to mementor-cli. As thin as possible.

### Trait-Based DI Pattern

Inspired by the fenv project. Core traits:

- **`MementorContext`**: Provides access to configuration, database path,
  model path, and other environment-dependent values.
- **`ConsoleOutput<OUT, ERR>`**: Abstracts stdout/stderr for testable CLI
  output. Generic over output stream types.

All CLI commands accept trait objects, enabling unit tests with mock
implementations.

### Turn-Based Chunking

A **Turn** is defined as:

```
Turn[n] = User[n] + Assistant[n] + User[n+1]
```

The `User[n+1]` message serves as **forward context**, providing semantic
continuity by hinting at what the assistant's response was addressing. This
improves embedding quality for retrieval.

**Provisional turn mechanism**: During incremental ingestion, if a transcript
ends mid-conversation (e.g., the assistant has not yet responded), the last
incomplete turn is stored as a "provisional turn". On the next ingestion pass,
the provisional turn is replaced with the completed turn data.

### Sub-Chunk Overlap

Turns may exceed the embedding model's token limit. When a turn must be split
into sub-chunks:

- `text-splitter` with `MarkdownSplitter` handles the initial split.
- Post-processing adds **~40 tokens (~15%) overlap** between adjacent
  sub-chunks. This overlap is prepended from the tail of the previous sub-chunk
  to the head of the next sub-chunk.
- Overlap ensures that semantic boundaries are not lost at split points.

### Schema Versioning

`rusqlite_migration` manages schema versions. Each migration is an idempotent
SQL script. The schema includes:

- `sessions` table: Tracks ingested transcript files and their state
  (`last_line_index`, `provisional_turn_start`).
- `memories` table: Stores embedded chunks with a vector BLOB `embedding`
  column, linked to sessions via `session_id` foreign key.

### sqlite-vector Integration

sqlite-vector (v0.9.90) provides vector operations via custom SQL functions and
the `vector_full_scan` virtual table for similarity search. The C source files
in `vendor/sqlite-vector/` are compiled statically via `build.rs` using the `cc`
crate. At runtime, `sqlite3_vector_init` is loaded via FFI, `vector_init()`
registers tables for search, and `vector_full_scan()` performs top-k similarity
search. On x86_64 macOS, AVX2/AVX512 stubs redirect to SSE2 due to Apple Clang
limitations.

### Target Platform

Milestone 1 targets **macOS only** (ARM64 Apple Silicon + x86_64 Intel).

## TODO

### Milestone 1: Core Pipeline

- [x] Set up workspace Cargo.toml with shared lint configuration
- [x] Vendor sqlite-vector C sources and write build.rs for static compilation
- [x] Bundle BGE-small-en-v1.5 ONNX model in models/ directory
- [x] Define MementorContext and ConsoleOutput traits in mementor-lib
- [x] Design and implement SQLite schema with rusqlite_migration
- [x] Implement transcript JSONL parser
- [x] Implement turn grouping logic (User[n] + Assistant[n] + User[n+1])
- [x] Implement provisional turn mechanism for incremental ingestion
- [x] Implement text splitting with MarkdownSplitter and sub-chunk overlap
- [x] Implement fastembed-rs embedding pipeline
- [x] Implement vector storage and cosine similarity search
- [x] Implement `mementor enable` command (install hooks into Claude Code config)
- [x] Implement `mementor ingest` command
- [x] Implement `mementor query` command
- [x] Implement `mementor hook stop` command (hook entry point for Stop event)
- [x] Implement `mementor hook user-prompt-submit` command (hook entry point for UserPromptSubmit event)
- [x] Wire up DI in mementor-main
- [x] Write unit tests for chunking, embedding, and DB modules
- [x] Write integration tests for full ingest-query pipeline
- [x] Ensure `cargo clippy -- -D warnings` passes with zero warnings
- [x] Ensure `cargo test` passes

## Known Issues

### ONNX Runtime Dynamic Linking on x86_64 macOS

The `ort-sys` crate (used by `fastembed` via `ort`) does not provide prebuilt
static binaries for `x86_64-apple-darwin`. Microsoft stopped distributing
x86_64 macOS binaries starting with ONNX Runtime 1.24.1 — only
`aarch64-apple-darwin` is supported.

**Root cause**: When `fastembed` enables `ort-download-binaries`, the build
script in `ort-sys` looks up the target triple in its `dist.txt` manifest.
The entry `(x86_64-apple-darwin, none)` does not exist, causing a build error:

```
error: ort-sys@2.0.0-rc.11: ort does not provide prebuilt binaries for the
target `x86_64-apple-darwin` with feature set (no features).
```

Note: "no features" refers to the execution provider feature set (CUDA, TensorRT,
etc.), not Cargo features. The default "none" feature set simply means CPU-only
inference.

**Resolution**: Switched fastembed from default features to:

```toml
fastembed = { version = "5.9.0", default-features = false, features = [
    "ort-load-dynamic",
    "hf-hub-native-tls",
    "image-models",
] }
```

This uses `ort-load-dynamic` which loads `libonnxruntime.dylib` at runtime via
`dlopen` instead of statically linking the ONNX Runtime. The runtime library is
installed via Homebrew (`brew install onnxruntime`) and located at
`/usr/local/lib/libonnxruntime.dylib`.

**Impact**: The `ORT_DYLIB_PATH` environment variable must be set when building
and running mementor on x86_64 macOS. This is configured via `mise.local.toml`
(gitignored) which mise automatically loads:

```toml
# mise.local.toml
[env]
ORT_DYLIB_PATH = "/usr/local/lib/libonnxruntime.dylib"
```

This is a development machine constraint, not an end-user issue on Apple Silicon
where prebuilt binaries exist.

**Future improvement**: A future milestone should implement conditional
feature selection — use `ort-download-binaries` for aarch64-apple-darwin (where
prebuilt binaries exist) and `ort-load-dynamic` for x86_64-apple-darwin. Or
build ONNX Runtime from source for full static linking on all platforms.

## Future Work

### CLI Commands

- **`mementor disable`**: Remove Claude Code lifecycle hooks from the project
  configuration without deleting the vector database. This allows users to
  temporarily or permanently stop Mementor from running while preserving all
  previously ingested context for potential future re-enablement.

- **`mementor doctor`**: A diagnostic health-check command that reports:
  - SIMD backend status (NEON on ARM64, SSE/AVX on x86_64) and whether the
    runtime CPU supports the compiled backend.
  - Embedded model metadata (name, dimension, quantization, file size).
  - Claude Code hook configuration status (whether hooks are correctly installed
    and pointing to the mementor binary).
  - Database statistics (number of sessions, turns, chunks, total vectors,
    database file size).

### Chunking Enhancements

- **Action Chunks**: A tool-centric chunking strategy that groups `tool_use` and
  `tool_result` message pairs into coherent action units. This would capture the
  semantics of "Claude used tool X to accomplish Y" as a single retrievable
  chunk, improving recall for tool-heavy conversations (e.g., file edits, shell
  commands, search operations).

- **Code-specific embeddings**: Use `tree-sitter` to parse code blocks within
  conversation content and generate structure-aware embeddings. This would
  improve retrieval quality for code-related queries by understanding syntax
  trees rather than treating code as plain text.

- **Wider cross-turn context window**: Extend the forward context beyond a
  single `User[n+1]` message. Experiment with including `User[n+1]` +
  `Assistant[n+1]` or even larger windows to capture longer-range semantic
  dependencies between turns.

### Platform Support

- **Linux**: Support x86_64 and aarch64 Linux targets. This requires testing
  sqlite-vector compilation on Linux, verifying ONNX runtime compatibility, and
  potentially adjusting build.rs for different linker behaviors.

- **Windows**: Support x86_64 Windows. This requires MSVC-compatible
  sqlite-vector compilation, Windows path handling throughout the codebase, and
  Windows-specific Claude Code configuration paths.

### Binary Size Optimization

- **Runtime model loading**: The current approach of bundling the ONNX model at
  compile time via `fastembed-rs` results in a ~130MB binary. A future
  optimization would download or locate the model at runtime (e.g., in
  `~/.mementor/models/`), reducing the binary to ~5-10MB. This requires a model
  management subsystem with download, verification, and caching logic.

- **Custom/alternative model support**: Allow users to configure alternative
  embedding models beyond BGE-small-en-v1.5. This would support different
  dimension sizes, quantization levels (fp16, int8), and specialized models for
  different use cases.

### Licensing

- **Elastic License 2.0 clarification for sqlite-vector**: The sqlite-vector
  library is distributed under the Elastic License 2.0, which has specific
  restrictions on providing the software as a managed service. Before any public
  distribution of Mementor, the implications of bundling sqlite-vector under
  this license need to be formally reviewed. This includes determining whether
  Mementor's use case (local-only CLI tool) is fully compliant, and whether
  alternative vector search implementations should be considered for broader
  distribution.
