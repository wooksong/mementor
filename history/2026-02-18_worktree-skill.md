# Worktree Management Skill

## Background

Git worktree management in the Mementor project requires multiple manual steps:
creating the worktree, copying `mise.local.toml`, running `mise trust` twice,
and `mise install`. On removal, the associated branch must also be cleaned up.
These steps are documented in CLAUDE.md but are easy to forget or execute
incorrectly.

## Goals

Create a custom Claude Code skill (`/worktree`) that formalizes the worktree
workflow into a single slash command with three subcommands:

1. **list** -- display all worktrees
2. **add** -- create a worktree with automatic mise environment setup
3. **remove** -- remove a worktree with branch cleanup

Update CLAUDE.md to enforce use of the skill (matching the `/commit` pattern).

## Design Decisions

- **`disable-model-invocation: false`**: Claude should auto-invoke this during
  the Workflow step 1 (feature branch creation), matching the `/commit` pattern.
- **`allowed-tools: Bash, Read`**: Bash for git/mise commands, Read for file
  verification. `AskUserQuestion` is always available.
- **Subcommand routing via `$0`**: First argument determines which handler runs.
- **Auto-derived worktree path**: `/worktree add <branch>` creates a sibling
  directory named `<repo>-<sanitized-branch>`.
- **Safe removal**: On `remove`, if cwd is inside the target worktree, `cd` to
  the primary worktree first. Ask user confirmation before `--force` or `-D`.
- **Settings consolidation**: Replace specific `git -C .../mementor*` entries
  in `settings.local.json` with a single wildcard for all fenv-org directories.

## Additional Changes

- Consolidated `settings.local.json` permissions: replaced 7 specific
  `git -C .../mementor*` entries with a single
  `Bash(git -C /Users/.../fenv-org/:*)` wildcard pattern. Also added
  `Bash(cp:*)` for worktree setup file copying.

## TODO

- [x] Create feature branch (worktree)
- [x] Create history document
- [x] Create `.claude/skills/worktree/SKILL.md`
- [x] Update `CLAUDE.md` with skill enforcement
- [x] Update `.claude/settings.json` permissions
- [ ] Commit
