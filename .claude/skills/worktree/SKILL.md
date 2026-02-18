---
name: worktree
description: Manage git worktrees with automatic mise environment setup and branch cleanup
disable-model-invocation: false
argument-hint: "<list|add|remove> [args...]"
allowed-tools: Bash, Read, Bash(deno *)
---

# Worktree Skill

Manage git worktrees for this project. Route to the correct handler based on
the subcommand in `$0`.

## Subcommand routing

- If `$0` is `list` or empty/omitted: go to **List worktrees**
- If `$0` is `add`: go to **Add a worktree**
- If `$0` is `remove` or `rm`: go to **Remove a worktree**
- Otherwise: show usage help listing the three subcommands

---

## List worktrees

Run:

```bash
git worktree list
```

Display the output to the user.

---

## Add a worktree

### Determine arguments

- **One argument** (`$1` only): treat it as the **branch name**. Derive the
  worktree path automatically:
  1. Get the main worktree path from the first entry of
     `git worktree list --porcelain` (the `worktree` line).
  2. Extract the parent directory and repo name from that path.
  3. Build the worktree path as a sibling directory:
     `<parent>/<repo>-<sanitized-branch>` where `<sanitized-branch>` replaces
     `/` with `-` in the branch name.
- **Two arguments** (`$1` and `$2`): treat `$1` as the **path** and `$2` as
  the **branch name**.

### Execute

1. **Create the worktree and branch**:
   - If the branch does not exist yet:
     ```bash
     git worktree add -b <branch> <path>
     ```
   - If the branch already exists:
     ```bash
     git worktree add <path> <branch>
     ```

2. **Set up the mise environment** in the new worktree:
   a. Check whether `mise.local.toml` exists in the main worktree.
   b. If it exists, copy it to the new worktree:
      ```bash
      cp <main-worktree>/mise.local.toml <new-worktree>/
      ```
   c. If it does not exist, skip the copy. This is normal on Apple Silicon
      where `ORT_DYLIB_PATH` is not needed.
   d. Trust the mise config files and install the toolchain:
      ```bash
      cd <new-worktree> && mise trust && mise trust mise.local.toml && mise install
      ```
      Skip `mise trust mise.local.toml` if the file was not copied.

3. **Set up Claude Code settings** for the new worktree:
   ```bash
   deno run --allow-read --allow-write \
     <main-worktree>/.claude/skills/worktree/scripts/worktree_settings.ts \
     setup <main-worktree> <new-worktree>
   ```
   This copies `settings.local.json` to the new worktree and generates
   `git -C` permission entries in the main worktree's `settings.local.json`.

4. **Verify** by running `git worktree list` and displaying the result.

5. **Report** the new worktree path and branch name to the user.

---

## Remove a worktree

### Resolve the target

`$1` identifies the worktree to remove. It can be:
- A full absolute path (e.g., `/Users/.../mementor-feat-foo`)
- A directory name (e.g., `mementor-feat-foo`) -- resolve relative to the main
  worktree's parent directory
- A branch name (e.g., `feat/foo`) -- find the matching worktree from
  `git worktree list --porcelain`

Run `git worktree list --porcelain` to resolve the target to an absolute path
and identify the associated branch.

### Execute

1. **Handle cwd inside target**: If the current working directory is inside the
   worktree being removed, `cd` to the primary worktree first (the first entry
   from `git worktree list --porcelain`).

2. **Clean up Claude Code settings** before removing:
   ```bash
   deno run --allow-read --allow-write \
     <main-worktree>/.claude/skills/worktree/scripts/worktree_settings.ts \
     cleanup <main-worktree> <removed-worktree-path>
   ```
   This merges any new permissions from the removed worktree back to the main
   worktree's `settings.local.json` and removes `git -C` entries.

3. **Remove the worktree**:
   ```bash
   git worktree remove <path>
   ```
   If this fails because the worktree has uncommitted changes, use
   `AskUserQuestion` to ask the user whether to force-remove:
   > The worktree at `<path>` has uncommitted changes. Force remove?

   If the user agrees:
   ```bash
   git worktree remove --force <path>
   ```
   If the user declines, abort.

4. **Delete the associated branch**:
   ```bash
   git branch -d <branch>
   ```
   If this fails because the branch has unmerged changes, use
   `AskUserQuestion` to ask:
   > Branch `<branch>` has unmerged changes. Force delete?

   If the user agrees:
   ```bash
   git branch -D <branch>
   ```
   If the user declines, keep the branch and inform them.

5. **Prune stale references**:
   ```bash
   git worktree prune
   ```

6. **Verify** by running `git worktree list` and displaying the result.
