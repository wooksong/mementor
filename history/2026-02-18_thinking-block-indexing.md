# Task 1: Thinking Block Indexing

- **Parent:** [recall-quality-v2](2026-02-18_recall-quality-v2.md) — R1
- **Depends on:** none
- **Required by:** [Task 2: tool-context-enrichment](2026-02-18_tool-context-enrichment.md)
  — shares `ContentBlock` enum changes
- **Previous work:** [improve-recall-quality](2026-02-18_improve-recall-quality.md)
  — the 5-phase filter pipeline this builds upon

## Background

`ContentBlock` uses `#[serde(tag = "type")]` for deserialization. When a
transcript contains a `thinking` block (or any unknown block type like
`server_tool_use`), `serde_json::from_str` on the **entire JSONL line** fails
because `ContentBlock` has no matching variant. The `parser.rs` `warn!` macro
logs the error and skips the line.

This means an assistant message containing thinking blocks AND text blocks
loses ALL its content — the text blocks are dropped along with the thinking
blocks. This is worse than "thinking blocks are ignored" — entire turns are
lost silently.

Thinking blocks contain 13-117 entries per session with valuable reasoning
like "I chose X because Y" that directly answers "why" questions in recall.

## Goals

- Add `Thinking` variant to `ContentBlock` enum to extract thinking text.
- Add `#[serde(other)] Unknown` fallback for future unknown block types.
- Include thinking text in turn embeddings alongside regular text.
- No prefix (e.g., `[Thinking]`) — thinking text joins naturally with `\n\n`.

## Design Decisions

### ContentBlock changes

```rust
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: Option<String>,
        #[allow(dead_code)]
        signature: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse { ... },  // existing
    #[serde(rename = "tool_result")]
    ToolResult { ... },  // existing
    #[serde(other)]
    Unknown,
}
```

**Why `thinking: Option<String>`:** Transcript entries may contain partial
thinking blocks during streaming or when the block was interrupted. Using
`Option` prevents deserialization failures for edge cases.

**Why `#[serde(other)]` on `Unknown`:** Protects against future Claude Code
block types causing deserialization failure. With `#[serde(tag = "type")]`,
`#[serde(other)]` requires a unit variant.

### extract_text() changes

```rust
ContentBlock::Text { text } => Some(text.as_str()),
ContentBlock::Thinking { thinking, .. } => thinking.as_deref().filter(|s| !s.is_empty()),
// ToolUse, ToolResult, Unknown -> None
```

### Behavioral impact

After this change, re-ingesting the same transcript will produce different
results — turns that previously failed to parse (and were skipped) will now
be processed. This is desirable and correct. The incremental ingestion pattern
handles this naturally via provisional turn reprocessing.

## Key Files

| File | Change |
|------|--------|
| `crates/mementor-lib/src/transcript/types.rs` | Add `Thinking` + `Unknown` variants, update `extract_text()`, add `has_unknown_blocks()` |
| `crates/mementor-lib/src/transcript/parser.rs` | Add `debug!` log with raw JSONL line for unknown block types |

## TODO

- [x] Add `Thinking` variant with `thinking: Option<String>` and `signature: Option<String>`
- [x] Add `#[serde(other)] Unknown` variant
- [x] Update `extract_text()` match arms
- [x] Add `has_unknown_blocks()` method to `Content`
- [x] Add `debug!` logging in `parser.rs` for unknown blocks (includes raw line)
- [x] Add test: `deserialize_thinking_block`
- [x] Add test: `deserialize_thinking_block_none` (`thinking: null`)
- [x] Add test: `deserialize_thinking_block_empty` (`thinking: ""`)
- [x] Add test: `unknown_block_type_skipped`
- [x] Add test: `thinking_and_text_interleaved`
- [x] Add test: `only_thinking_block_produces_text`
- [x] Update existing `deserialize_blocks_content` to include a thinking block
- [x] Add test: `parse_message_with_thinking_blocks` (parser integration)
- [x] Add test: `parse_message_with_unknown_blocks` (parser integration)
- [x] Verify: clippy + all tests pass

## Results

- All 114 tests pass (75 lib + 39 cli), including 9 new tests
- Clippy passes with zero warnings
- Scope: types.rs (~40 lines code + ~75 lines test), parser.rs (~5 lines code + ~20 lines test)
