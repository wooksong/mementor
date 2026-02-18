#!/usr/bin/env -S deno run --allow-read --allow-write

import { Command } from "@cliffy/command";

interface Settings {
  permissions?: { allow?: string[]; ask?: string[]; deny?: string[] };
  [key: string]: unknown;
}

function readJson(path: string): Settings {
  return JSON.parse(Deno.readTextFileSync(path));
}

function writeJson(path: string, data: Settings): void {
  Deno.writeTextFileSync(path, JSON.stringify(data, null, 2) + "\n");
}

function tryReadJson(path: string): Settings | null {
  try {
    return readJson(path);
  } catch {
    return null;
  }
}

function dirname(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx > 0 ? path.substring(0, idx) : path;
}

/** Extract git subcommands from permission rules, skipping -C entries. */
export function extractGitSubcmds(rules: string[]): string[] {
  const subcmds: Set<string> = new Set();
  for (const rule of rules) {
    // Match Bash(git <subcmd>:*) or Bash(git <subcmd> *)
    const colonMatch = rule.match(/^Bash\(git ([^-][^:]*?):\*\)$/);
    if (colonMatch) {
      subcmds.add(colonMatch[1].trim());
      continue;
    }
    const spaceMatch = rule.match(/^Bash\(git ([^-]\S+) \*\)$/);
    if (spaceMatch) {
      subcmds.add(spaceMatch[1].trim());
    }
  }
  return [...subcmds];
}

export function setup(mainWt: string, newWt: string): void {
  const mainClaudeDir = `${mainWt}/.claude`;
  const newClaudeDir = `${newWt}/.claude`;
  const mainLocalPath = `${mainClaudeDir}/settings.local.json`;
  const mainSettingsPath = `${mainClaudeDir}/settings.json`;

  // 1. Copy settings.local.json to new worktree
  const mainLocal = tryReadJson(mainLocalPath);
  if (mainLocal) {
    const mainParent = dirname(mainWt);
    const newParent = dirname(newWt);

    Deno.mkdirSync(newClaudeDir, { recursive: true });

    if (mainParent === newParent) {
      // Same parent — copy as-is
      writeJson(`${newClaudeDir}/settings.local.json`, mainLocal);
    } else {
      // Different parent — rewrite path-specific patterns
      const oldPrefix = mainParent + "/";
      const newPrefix = newParent + "/";
      const rewritten = structuredClone(mainLocal);
      for (const key of ["allow", "ask", "deny"] as const) {
        const rules = rewritten.permissions?.[key];
        if (rules) {
          rewritten.permissions![key] = rules.map((r) =>
            r.includes(oldPrefix) ? r.replaceAll(oldPrefix, newPrefix) : r
          );
        }
      }
      writeJson(`${newClaudeDir}/settings.local.json`, rewritten);
    }
  }

  // 2. Generate git -C entries in main's settings.local.json
  const mainSettings = tryReadJson(mainSettingsPath);
  const allGitRules: string[] = [];
  if (mainSettings?.permissions?.allow) {
    allGitRules.push(...mainSettings.permissions.allow);
  }
  if (mainLocal?.permissions?.allow) {
    allGitRules.push(...mainLocal.permissions.allow);
  }

  const subcmds = extractGitSubcmds(allGitRules);
  if (subcmds.length === 0) return;

  // Re-read main local (it may have been the source for copy)
  const current = tryReadJson(mainLocalPath) ?? { permissions: { allow: [] } };
  const allow = current.permissions?.allow ?? [];

  const newEntries: string[] = [];
  for (const subcmd of subcmds) {
    const entry = `Bash(git -C ${newWt} ${subcmd} *)`;
    if (!allow.includes(entry)) {
      newEntries.push(entry);
    }
  }

  if (newEntries.length > 0) {
    if (!current.permissions) current.permissions = {};
    if (!current.permissions.allow) current.permissions.allow = [];
    current.permissions.allow.push(...newEntries);
    writeJson(mainLocalPath, current);
  }
}

export function cleanup(mainWt: string, removedWt: string): void {
  const mainLocalPath = `${mainWt}/.claude/settings.local.json`;
  const removedLocalPath = `${removedWt}/.claude/settings.local.json`;

  const mainLocal = tryReadJson(mainLocalPath);
  if (!mainLocal) return;

  // 1. Merge new permissions from the removed worktree
  const removedLocal = tryReadJson(removedLocalPath);
  if (removedLocal?.permissions?.allow) {
    const mainAllow = new Set(mainLocal.permissions?.allow ?? []);
    const newPerms = removedLocal.permissions.allow.filter(
      (r) => !mainAllow.has(r) && !r.includes("git -C "),
    );
    if (newPerms.length > 0) {
      if (!mainLocal.permissions) mainLocal.permissions = {};
      if (!mainLocal.permissions.allow) mainLocal.permissions.allow = [];
      mainLocal.permissions.allow.push(...newPerms);
    }
  }

  // 2. Remove -C entries for the removed worktree
  if (mainLocal.permissions?.allow) {
    mainLocal.permissions.allow = mainLocal.permissions.allow.filter(
      (r) => !r.includes(`git -C ${removedWt} `),
    );
  }

  writeJson(mainLocalPath, mainLocal);
}

async function main(): Promise<void> {
  await new Command()
    .name("worktree-settings")
    .description("Manage Claude Code settings for git worktrees")
    .command("setup <main-wt:string> <new-wt:string>")
    .description("Copy settings to new worktree and generate git -C entries")
    .action((_options: void, mainWt: string, newWt: string) =>
      setup(mainWt, newWt)
    )
    .command("cleanup <main-wt:string> <removed-wt:string>")
    .description("Merge permissions and remove git -C entries")
    .action((_options: void, mainWt: string, removedWt: string) =>
      cleanup(mainWt, removedWt)
    )
    .parse(Deno.args);
}

if (import.meta.main) {
  await main();
}
