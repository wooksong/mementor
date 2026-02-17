# Mementor

**Incremental Local RAG Memory Agent for Claude Code**

## Vision

Claude Code loses all context when a session ends. Mementor gives it persistent
memory by vectorizing conversation transcripts into a local SQLite database and
automatically recalling relevant past context at the start of each new session.

No external APIs. No cloud services. Everything runs locally on your machine.

## Features

- **Incremental transcript ingestion** -- Only processes new lines from
  transcript files, tracking progress with `last_line_index` and provisional
  turn state.
- **Turn-based chunking with forward context** -- Groups messages into turns
  (User[n] + Assistant[n] + User[n+1]) for semantically coherent chunks. The
  next user message provides forward context that improves embedding quality.
- **Local ONNX embedding** -- Uses BGE-small-en-v1.5 (384-dim, fp32) via
  fastembed-rs. No API calls, no network dependency.
- **SQLite + sqlite-vector** -- Vector storage and cosine similarity search in a
  single SQLite database file. No separate vector DB service required.
- **Claude Code lifecycle hooks** -- Automatic ingestion on session stop and
  automatic context recall on user prompt submission. Zero manual intervention
  after setup.

## Tech Stack

| Component        | Choice                                              |
| ---------------- | --------------------------------------------------- |
| Language         | Rust (edition 2024)                                 |
| Database         | SQLite via rusqlite (bundled)                        |
| Vector search    | sqlite-vector (statically compiled)                 |
| Embedding        | fastembed-rs + BGE-small-en-v1.5 ONNX               |
| Text splitting   | text-splitter with MarkdownSplitter                 |
| CLI              | clap                                                |

## Install

```bash
cargo build --release
```

The binary is produced at `target/release/mementor`.

## Quick Start

1. **Enable Mementor** in your project:

   ```bash
   mementor enable
   ```

   This installs Claude Code lifecycle hooks in your project's
   `.claude/settings.json`.

2. **Use Claude Code normally.** Mementor works in the background:
   - When a session ends, the **Stop hook** ingests the conversation transcript.
   - When a new prompt is submitted, the **UserPromptSubmit hook** searches for
     relevant past context and injects it.

3. That is it. Claude Code now remembers across sessions.

## CLI Reference

### `mementor enable`

Install Claude Code lifecycle hooks into the current project. Writes hook
configuration to `.claude/settings.json` so that Mementor is called
automatically on session stop and user prompt submission.

### `mementor ingest`

Manually ingest a transcript JSONL file into the vector database. Supports
incremental ingestion -- only new lines since the last ingestion are processed.

### `mementor query`

Search the vector database for chunks semantically similar to a query string.
Returns the top-k most relevant past conversation fragments.

### `mementor hook stop`

Entry point called by the Claude Code Stop lifecycle hook. Reads the session
transcript from stdin, parses it, chunks it into turns, embeds the chunks, and
stores them in the database.

### `mementor hook user-prompt-submit`

Entry point called by the Claude Code UserPromptSubmit lifecycle hook. Takes the
current user prompt, searches the vector database for relevant past context, and
outputs the context for injection into the conversation.

## Architecture

```
Session End (Stop Hook)
  |
  v
Transcript JSONL (stdin)
  |
  v
Turn Grouping: [User[n] + Assistant[n] + User[n+1]]
  |
  v
Text Splitting (MarkdownSplitter) + Sub-chunk Overlap (~40 tokens)
  |
  v
Embedding (BGE-small-en-v1.5 via fastembed-rs)
  |
  v
Storage (SQLite + sqlite-vector BLOB column)

---

New Prompt (UserPromptSubmit Hook)
  |
  v
Query Embedding
  |
  v
Cosine Similarity Search (vector_distance_cos)
  |
  v
Top-k Context Retrieval
  |
  v
Context Injection into Claude Code
```

### Workspace Structure

```
crates/
  mementor-lib/    Core library: DB, embedding, chunking, schema, traits
  mementor-cli/    CLI layer: command dispatch, argument parsing, DI
  mementor-main/   Thin binary entry point: wires real implementations
vendor/
  sqlite-vector/   C source files for static compilation
models/
  bge-small-en-v1.5/   ONNX model files
```

## License

TBD

> **Note**: This project statically links sqlite-vector, which is distributed
> under the [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license).
> The Elastic License 2.0 permits most use cases but restricts providing the
> software as a managed service. Review the license terms before redistribution.
