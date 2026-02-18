# Workflow and Permissions Improvements

## Background

Two problems addressed in this task:

1. **Workflow step 6 missing**: The workflow section (revised in #15) still
   lacks a step requiring all TODO items to be completed before creating a PR.

2. **Worktree permission annoyance**: `.claude/settings.local.json` is
   gitignored and resolved per directory path. When a new worktree is created,
   it lacks this file, forcing the user to re-approve ~42 permissions.
   Additionally, Claude launched from the main worktree uses
   `git -C <new-worktree>` to operate on the separate worktree — these need
   permission patterns with the exact worktree path.

## Goals

- Add workflow step 6: all TODOs must be done before creating a PR
- Promote safe (read-only/local) permissions from `settings.local.json` to
  committed `settings.json`
- Create a Deno TypeScript script that:
  - Copies `settings.local.json` to new worktrees (with path rewriting)
  - Generates per-subcommand `git -C` permission patterns in main worktree
  - Cleans up `-C` entries when a worktree is removed
  - Merges newly-approved permissions back to main during worktree removal
- Add unit + integration tests for the Deno script
- Add a separate CI workflow for Deno scripts with conditional success pattern
- Update the `/worktree` skill to invoke the script
- Document Deno script conventions

## Design Decisions

- **Safe vs risky permissions**: Safe (read-only/local) commands like
  `git add`, `git fetch`, `gh pr view` are promoted to committed settings.
  Risky commands (`git commit`, `git push`, `gh pr merge`, `gh api`) stay in
  `settings.local.json`.
- **`git -C` pattern generation**: Instead of one broad `git -C /parent/*`
  pattern, generate per-subcommand patterns (e.g.,
  `Bash(git -C /full/path add *)`) for each worktree. This preserves
  fine-grained control.
- **Deno for scripting**: Using Deno TypeScript with cliffy for CLI parsing
  and dax for subprocess execution in tests. Added to `mise.toml` (pinned to
  2.6.10) for reproducible builds. Dependencies managed via `deno add` with
  `deno.json` import map.
- **`import.meta.main` guard**: Scripts define a `main()` function and guard
  execution with `if (import.meta.main)`, enabling direct imports in tests
  instead of fragile subprocess-only testing.
- **Underscore file naming**: Following the Deno style guide
  (`worktree_settings.ts`, not `worktree-settings.ts`).
- **CI conditional success**: Uses `dorny/paths-filter` + `if: always()`
  status job pattern so the check succeeds when skipped (no `.ts` changes).
- **Constraint**: Do not touch user-scope config files.

## TODO

- [x] Create worktree
- [x] Create history document
- [x] Add deno 2.6.10 to mise.toml
- [x] Add workflow step 6 to AGENTS.md
- [x] Promote safe permissions to `.claude/settings.json`
- [x] Clean up `.claude/settings.local.json`
- [x] Create `worktree_settings.ts` Deno script with cliffy
- [x] Write unit + integration tests (10 tests: 3 unit, 2 direct, 5 integration)
- [x] Add mise tasks (deno:test, deno:fmt, deno:check) with auto-discovery
- [x] Create `.github/workflows/deno.yml` CI workflow
- [x] Update `/worktree` skill SKILL.md
- [x] Set up `deno.json` with `deno add` for dependencies
- [x] Add `docs/deno-script-conventions.md`
- [x] Add Deno Scripts reference link in AGENTS.md Coding Conventions
- [x] Verify: deno tests pass (10/10), rust tests pass (106/106)

## Results

All changes implemented:

- **AGENTS.md**: Added workflow step 6 (complete TODOs before PR), Deno Scripts
  reference in Coding Conventions
- **`.claude/settings.json`**: Promoted 33 safe permissions, migrated `:*` to ` *`
- **`.claude/settings.local.json`** (main worktree): Stripped to 5 risky
  permissions + enabledPlugins
- **`worktree_settings.ts`**: Deno CLI with `setup`/`cleanup` subcommands,
  `import.meta.main` guard, exported functions for testability
- **`worktree_settings_test.ts`**: 10 tests (unit + integration via dax)
- **`deno.json`**: Import map for `@cliffy/command`, `@std/assert`, `@david/dax`
- **`.github/workflows/deno.yml`**: CI with conditional success pattern
- **`SKILL.md`**: Added settings script steps to add/remove flows
- **`docs/deno-script-conventions.md`**: Deno script conventions doc

## Future Work

(none)
