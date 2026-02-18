# Fix null string fields in hook stdin input

## Background

GitHub issue #20: Mementor crashes when Claude Code sends `null` for the
`prompt` field in the `UserPromptSubmit` hook input JSON. This happens when
users reference files via `@`-mentions without typing a text prompt.

`serde_json` cannot deserialize `null` into `String`, causing a deserialization
error that propagates up and fails the entire hook invocation. No memory recall
occurs for that prompt.

## Goals

1. Handle `null` prompt gracefully — deserialize as empty string
2. Skip recall when prompt is empty (no useful query to search with)
3. Add tests for null and missing prompt scenarios

## Design Decisions

- Use a custom serde deserializer (`nullable_string`) rather than
  `Option<String>`, keeping the field as `String` and avoiding changes to all
  downstream consumers.
- `#[serde(default)]` alone does NOT handle null — it only handles missing
  fields. Must combine with `#[serde(deserialize_with = "nullable_string")]`.
- Apply only to `PromptHookInput::prompt` (the confirmed crash site). Other
  fields (`session_id`, `cwd`) being null would indicate a Claude Code bug.

## TODO

- [x] Create worktree and feature branch
- [x] Create this history document
- [x] Add `nullable_string` deserializer in `input.rs`
- [x] Annotate `PromptHookInput::prompt` with serde attributes
- [x] Add unit tests: null prompt, missing prompt
- [x] Add early return in `handle_prompt` for empty prompt
- [x] Add integration test: null prompt via `try_run`
- [x] Run tests and clippy (109 tests pass, 0 clippy warnings)
- [ ] Commit and create PR

## Files Modified

- `crates/mementor-cli/src/hooks/input.rs`
- `crates/mementor-cli/src/hooks/prompt.rs`
