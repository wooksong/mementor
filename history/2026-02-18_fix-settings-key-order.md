# Fix settings.json key order preservation

## Background

Running `mementor enable` on a project that already has mementor hooks
configured causes a cosmetic diff in `.claude/settings.json`. The JSON key
order within hook entries changes from `"command"` → `"type"` to `"type"` →
`"command"`.

This happens because `upsert_hook_entry` unconditionally removes and re-adds
mementor hooks. The `serde_json::json!()` macro produces keys in insertion
order (`type` before `command`), which differs from the order that Claude Code
writes (`command` before `type`).

## Goals

- Make `mementor enable` idempotent with respect to key ordering — running it
  on an already-configured project should produce zero diff.
- Add a regression test using the actual `settings.json` content.

## Design Decision

Modify `upsert_hook_entry` to check whether an existing entry already has the
exact same command string. If so, skip the remove+add cycle entirely. This
preserves whatever key order the original file had.

## TODO

- [x] Create history document
- [x] Add failing test `try_run_enable_preserves_existing_key_order`
- [x] Fix `upsert_hook_entry` to skip unchanged hooks
- [x] Verify: clippy (0 warnings), all 67 tests pass, manual `mementor enable` produces no diff
- [x] Commit
- [x] Update history document

## Results

- 1 file changed: `crates/mementor-cli/src/commands/enable.rs`
- 1 new test added (67 total)
- `mementor enable` is now fully idempotent — no cosmetic diffs on re-run

## Future Work

None expected.
