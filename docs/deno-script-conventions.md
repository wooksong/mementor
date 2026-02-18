# Deno Script Conventions

This document describes the conventions for Deno TypeScript scripts in this
project. All scripts live under `.claude/` and are managed via `deno.json` at
the repository root.

Follow the [Deno Style Guide](https://docs.deno.com/runtime/contributing/style_guide/)
as the baseline. The rules below are project-specific additions and emphasis.

## Entry point pattern

Every executable script must define a `main` function and guard its execution
with `import.meta.main`. This prevents side effects when the module is imported
by tests or other scripts.

```typescript
#!/usr/bin/env -S deno run --allow-read --allow-write

export function main(): void {
  // CLI parsing and orchestration here
}

if (import.meta.main) {
  main();
}
```

For async entry points:

```typescript
export async function main(): Promise<void> {
  await new Command()
    .name("my-tool")
    .command("sub <arg:string>")
    .action(handleSub)
    .parse(Deno.args);
}

if (import.meta.main) {
  await main();
}
```

**Why**: Top-level `await` and side effects at module scope make the module
impossible to import without executing. The `import.meta.main` guard allows
tests to import and call individual functions directly, avoiding fragile
subprocess-based testing.

## File naming

Use **underscores**, not dashes: `worktree_settings.ts`, not
`worktree-settings.ts`.

Test files are named with a `_test` suffix: `worktree_settings_test.ts`.

## Exports

Export all functions and interfaces that tests or other modules need. Keep
internal helpers unexported.

```typescript
// Exported — used by tests
export function extractGitSubcmds(rules: string[]): string[] { ... }

// Not exported — internal helper
function dirname(path: string): string { ... }
```

Export parameter/return interfaces alongside the functions that use them:

```typescript
export interface Settings {
  permissions?: { allow?: string[]; deny?: string[] };
  [key: string]: unknown;
}

export function readSettings(path: string): Settings { ... }
```

## Function signatures

- Use the `function` keyword for top-level functions. Reserve arrow functions
  for closures and callbacks.
- Maximum 2 required parameters. Use an options object for additional
  configuration.

```typescript
// Good
function setup(mainWt: string, newWt: string): void { ... }

// Good — options object for optional params
function setup(mainWt: string, newWt: string, options?: SetupOptions): void { ... }

// Bad — too many positional args
function setup(mainWt: string, newWt: string, dryRun: boolean, verbose: boolean): void { ... }
```

## Dependencies

Manage all dependencies via `deno add` and import using bare specifiers from
the `deno.json` import map. Do not use `jsr:` or `npm:` URLs in source files.

```typescript
// Good — bare specifier from deno.json
import { Command } from "@cliffy/command";
import { assertEquals } from "@std/assert";

// Bad — inline URL
import { Command } from "jsr:@cliffy/command@1.0.0-rc.7";
```

## Error messages

Follow the Deno style guide conventions:

- Start with uppercase, no trailing period
- Use active voice: `"Cannot parse input"`, not `"Input cannot be parsed"`
- Quote string values: `"Cannot find file \"foo.json\""`
- No contractions: `"Cannot"`, not `"Can't"`

## Testing

### Companion test files

Every script `foo.ts` must have a companion `foo_test.ts`. Tests use
`Deno.test` with descriptive names.

### Direct imports over subprocesses

Because scripts use the `import.meta.main` guard, tests import functions
directly instead of spawning subprocesses:

```typescript
import { assertEquals } from "@std/assert";
import { extractGitSubcmds, setup } from "./worktree_settings.ts";

Deno.test("extractGitSubcmds: parses space-style rules", () => {
  const result = extractGitSubcmds(["Bash(git add *)", "Bash(git fetch *)"]);
  assertEquals(result, ["add", "fetch"]);
});
```

Subprocess-based integration tests are acceptable when testing the full CLI
entry point, but unit-level tests should prefer direct imports.

### Temp directories for filesystem tests

Use `Deno.makeTempDirSync()` for tests that need filesystem state. Deno cleans
up temp directories automatically.

```typescript
Deno.test("setup: copies settings to new worktree", () => {
  const parent = Deno.makeTempDirSync();
  const mainWt = `${parent}/mementor`;
  const newWt = `${parent}/mementor-feature`;
  Deno.mkdirSync(`${mainWt}/.claude`, { recursive: true });
  Deno.mkdirSync(newWt, { recursive: true });
  // ... seed files, call function, assert results
});
```

## Running checks

Three mise tasks validate all Deno scripts. They auto-discover `.ts` files
under `.claude/` — no paths need to be specified.

```bash
mise run deno:fmt       # Format (or --check)
mise run deno:check     # Type-check
mise run deno:test      # Run tests
```

All three must pass before committing.
