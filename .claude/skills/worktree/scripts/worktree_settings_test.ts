import { assertArrayIncludes, assertEquals } from "@std/assert";
import $ from "@david/dax";
import { cleanup, extractGitSubcmds, setup } from "./worktree_settings.ts";

const SCRIPT = new URL("./worktree_settings.ts", import.meta.url).pathname;

function readJson(path: string): Record<string, unknown> {
  return JSON.parse(Deno.readTextFileSync(path));
}

function writeSettings(dir: string, name: string, data: unknown): void {
  Deno.mkdirSync(`${dir}/.claude`, { recursive: true });
  Deno.writeTextFileSync(
    `${dir}/.claude/${name}`,
    JSON.stringify(data, null, 2) + "\n",
  );
}

// --- Unit tests (direct import) ---

Deno.test("extractGitSubcmds: parses space-style rules", () => {
  const result = extractGitSubcmds(["Bash(git add *)", "Bash(git fetch *)"]);
  assertEquals(result, ["add", "fetch"]);
});

Deno.test("extractGitSubcmds: parses colon-style rules", () => {
  const result = extractGitSubcmds(["Bash(git commit:*)", "Bash(git push:*)"]);
  assertEquals(result, ["commit", "push"]);
});

Deno.test("extractGitSubcmds: skips -C entries", () => {
  const result = extractGitSubcmds([
    "Bash(git add *)",
    "Bash(git -C /tmp/wt add *)",
  ]);
  assertEquals(result, ["add"]);
});

Deno.test("setup: copies settings.local.json (direct)", () => {
  const parent = Deno.makeTempDirSync();
  const mainWt = `${parent}/mementor`;
  const newWt = `${parent}/mementor-feature`;
  Deno.mkdirSync(newWt, { recursive: true });

  writeSettings(mainWt, "settings.local.json", {
    permissions: { allow: ["Bash(git commit:*)"] },
    enabledPlugins: { "feature-dev@claude-plugins-official": true },
  });
  writeSettings(mainWt, "settings.json", { permissions: { allow: [] } });

  setup(mainWt, newWt);

  const copied = readJson(`${newWt}/.claude/settings.local.json`);
  assertEquals(
    (copied.permissions as { allow: string[] }).allow,
    ["Bash(git commit:*)"],
  );
  assertEquals(copied.enabledPlugins, {
    "feature-dev@claude-plugins-official": true,
  });
});

Deno.test("cleanup: removes -C entries (direct)", () => {
  const parent = Deno.makeTempDirSync();
  const mainWt = `${parent}/mementor`;
  const removedWt = `${parent}/mementor-feature`;
  const otherWt = `${parent}/mementor-other`;
  Deno.mkdirSync(`${removedWt}/.claude`, { recursive: true });

  writeSettings(mainWt, "settings.local.json", {
    permissions: {
      allow: [
        "Bash(git commit:*)",
        `Bash(git -C ${removedWt} add *)`,
        `Bash(git -C ${removedWt} commit *)`,
        `Bash(git -C ${otherWt} add *)`,
      ],
    },
  });

  cleanup(mainWt, removedWt);

  const mainLocal = readJson(`${mainWt}/.claude/settings.local.json`);
  assertEquals((mainLocal.permissions as { allow: string[] }).allow, [
    "Bash(git commit:*)",
    `Bash(git -C ${otherWt} add *)`,
  ]);
});

// --- Integration tests (subprocess via dax) ---

Deno.test("setup: rewrites paths when parents differ (integration)", async () => {
  const oldParent = Deno.makeTempDirSync();
  const newParent = Deno.makeTempDirSync();
  const mainWt = `${oldParent}/mementor`;
  const newWt = `${newParent}/mementor-feature`;
  Deno.mkdirSync(newWt, { recursive: true });

  writeSettings(mainWt, "settings.local.json", {
    permissions: {
      allow: [
        `Bash(git -C ${oldParent}/mementor commit *)`,
        "Bash(git push:*)",
      ],
    },
  });
  writeSettings(mainWt, "settings.json", { permissions: { allow: [] } });

  await $`deno run -A ${SCRIPT} setup ${mainWt} ${newWt}`;

  const copied = readJson(`${newWt}/.claude/settings.local.json`);
  const allow = (copied.permissions as { allow: string[] }).allow;
  assertEquals(allow[0], `Bash(git -C ${newParent}/mementor commit *)`);
  assertEquals(allow[1], "Bash(git push:*)");
});

Deno.test("setup: generates -C entries in main (integration)", async () => {
  const parent = Deno.makeTempDirSync();
  const mainWt = `${parent}/mementor`;
  const newWt = `${parent}/mementor-feature`;
  Deno.mkdirSync(newWt, { recursive: true });

  writeSettings(mainWt, "settings.json", {
    permissions: { allow: ["Bash(git add *)", "Bash(git fetch *)"] },
  });
  writeSettings(mainWt, "settings.local.json", {
    permissions: { allow: ["Bash(git commit:*)"] },
  });

  await $`deno run -A ${SCRIPT} setup ${mainWt} ${newWt}`;

  const mainLocal = readJson(`${mainWt}/.claude/settings.local.json`);
  const allow = (mainLocal.permissions as { allow: string[] }).allow;
  assertArrayIncludes(allow, [
    `Bash(git -C ${newWt} add *)`,
    `Bash(git -C ${newWt} fetch *)`,
    `Bash(git -C ${newWt} commit *)`,
  ]);
});

Deno.test("setup: skips existing -C entries (integration)", async () => {
  const parent = Deno.makeTempDirSync();
  const mainWt = `${parent}/mementor`;
  const newWt = `${parent}/mementor-feature`;
  Deno.mkdirSync(newWt, { recursive: true });

  writeSettings(mainWt, "settings.json", {
    permissions: { allow: ["Bash(git add *)"] },
  });
  writeSettings(mainWt, "settings.local.json", {
    permissions: { allow: [] },
  });

  // Run twice
  await $`deno run -A ${SCRIPT} setup ${mainWt} ${newWt}`;
  await $`deno run -A ${SCRIPT} setup ${mainWt} ${newWt}`;

  const mainLocal = readJson(`${mainWt}/.claude/settings.local.json`);
  const allow = (mainLocal.permissions as { allow: string[] }).allow;
  const cEntries = allow.filter((r: string) =>
    r.includes(`git -C ${newWt} add`)
  );
  assertEquals(cEntries.length, 1, "Should have exactly one -C entry");
});

Deno.test("setup: handles missing settings.local.json (integration)", async () => {
  const parent = Deno.makeTempDirSync();
  const mainWt = `${parent}/mementor`;
  const newWt = `${parent}/mementor-feature`;
  Deno.mkdirSync(newWt, { recursive: true });

  writeSettings(mainWt, "settings.json", {
    permissions: { allow: ["Bash(git add *)"] },
  });

  await $`deno run -A ${SCRIPT} setup ${mainWt} ${newWt}`;

  let exists = true;
  try {
    Deno.statSync(`${newWt}/.claude/settings.local.json`);
  } catch {
    exists = false;
  }
  assertEquals(exists, false);
});

Deno.test("cleanup: merges permissions from worktree (integration)", async () => {
  const parent = Deno.makeTempDirSync();
  const mainWt = `${parent}/mementor`;
  const removedWt = `${parent}/mementor-feature`;

  writeSettings(mainWt, "settings.local.json", {
    permissions: {
      allow: [
        "Bash(git commit:*)",
        `Bash(git -C ${removedWt} add *)`,
      ],
    },
  });
  writeSettings(removedWt, "settings.local.json", {
    permissions: {
      allow: [
        "Bash(git commit:*)",
        "Bash(npm:*)",
      ],
    },
  });

  await $`deno run -A ${SCRIPT} cleanup ${mainWt} ${removedWt}`;

  const mainLocal = readJson(`${mainWt}/.claude/settings.local.json`);
  const allow = (mainLocal.permissions as { allow: string[] }).allow;
  assertArrayIncludes(allow, ["Bash(git commit:*)", "Bash(npm:*)"]);
  assertEquals(
    allow.filter((r: string) => r.includes("git -C")).length,
    0,
    "All -C entries should be removed",
  );
});
