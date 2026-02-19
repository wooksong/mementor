use std::io::{Read, Write};

use mementor_lib::context::MementorContext;
use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleIO;
use mementor_lib::runtime::Runtime;

/// Run the `mementor enable` command.
///
/// 1. Create `.mementor/` directory and initialize DB with schema.
/// 2. Verify bundled embedding model loads.
/// 3. Append `.mementor/` to `.gitignore` if not present.
/// 4. Add mementor hooks to `.claude/settings.json`.
pub fn run_enable<IN, OUT, ERR>(
    runtime: &Runtime,
    io: &mut dyn ConsoleIO<IN, OUT, ERR>,
) -> anyhow::Result<()>
where
    IN: Read,
    OUT: Write,
    ERR: Write,
{
    // Guard: only allow enable from primary worktree
    if runtime.context.is_linked_worktree() {
        anyhow::bail!(
            "mementor enable must be run from the primary worktree.\n\
             Primary worktree: {}\n\
             Run `mementor enable` from that directory instead.",
            runtime.context.project_root().display(),
        );
    }

    // Step 1: Create DB (open creates parent dirs + schema + vector_init)
    writeln!(io.stderr(), "Initializing database...")?;
    let _conn = runtime.db.open()?;
    let db_path = runtime.context.db_path();
    writeln!(io.stderr(), "  Database created at {}", db_path.display())?;

    // Step 2: Verify embedding model loads
    writeln!(io.stderr(), "Verifying embedding model...")?;
    let _embedder = Embedder::new()?;
    writeln!(io.stderr(), "  Embedding model OK")?;

    // Step 3: Update .gitignore
    update_gitignore(&runtime.context)?;
    writeln!(io.stderr(), "  .gitignore updated")?;

    // Step 4: Configure Claude Code hooks
    configure_hooks(&runtime.context)?;
    writeln!(io.stderr(), "  Claude Code hooks configured")?;

    writeln!(io.stdout(), "mementor enabled successfully.")?;
    Ok(())
}

/// Append `.mementor/` to `.gitignore` if not already present.
fn update_gitignore(ctx: &MementorContext) -> anyhow::Result<()> {
    let gitignore_path = ctx.gitignore_path();
    let entry = ".mementor/";

    if gitignore_path.exists() {
        let existing = std::fs::read_to_string(&gitignore_path)?;
        if existing.lines().any(|line| line.trim() == entry) {
            return Ok(());
        }
        // Append with newline separator
        let separator = if existing.ends_with('\n') { "" } else { "\n" };
        std::fs::write(&gitignore_path, format!("{existing}{separator}{entry}\n"))?;
    } else {
        std::fs::write(&gitignore_path, format!("{entry}\n"))?;
    }

    Ok(())
}

/// Add mementor hooks to `.claude/settings.json`.
fn configure_hooks(ctx: &MementorContext) -> anyhow::Result<()> {
    let settings_path = ctx.claude_settings_path();

    // Ensure .claude/ directory exists
    if let Some(parent) = settings_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (mut settings, has_trailing_newline): (serde_json::Value, bool) = if settings_path.exists()
    {
        let raw = std::fs::read_to_string(&settings_path)?;
        let trailing = raw.ends_with('\n');
        (serde_json::from_str(&raw)?, trailing)
    } else {
        (serde_json::json!({}), true)
    };

    // Build hook config
    let stop_hook = serde_json::json!({
        "hooks": [{
            "type": "command",
            "command": "mementor hook stop"
        }]
    });

    let prompt_hook = serde_json::json!({
        "hooks": [{
            "type": "command",
            "command": "mementor hook user-prompt-submit"
        }]
    });

    let pre_compact_hook = serde_json::json!({
        "hooks": [{
            "type": "command",
            "command": "mementor hook pre-compact"
        }]
    });

    // Merge hooks into settings
    let hooks = settings
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("settings.json root is not an object"))?
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));

    let hooks_obj = hooks
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("hooks field is not an object"))?;

    // Upsert mementor hooks (replace existing mementor entries, preserve others)
    upsert_hook_entry(hooks_obj, "Stop", stop_hook);
    upsert_hook_entry(hooks_obj, "UserPromptSubmit", prompt_hook);
    upsert_hook_entry(hooks_obj, "PreCompact", pre_compact_hook);

    // Write settings back, preserving original EOF newline behavior
    let mut formatted = serde_json::to_string_pretty(&settings)?;
    if has_trailing_newline {
        formatted.push('\n');
    }
    std::fs::write(&settings_path, formatted)?;

    Ok(())
}

/// Upsert a mementor hook entry into the event array.
///
/// If an entry with the exact same command already exists, it is left
/// untouched (preserving the original key order). Otherwise, any existing
/// mementor entries (commands starting with "mementor") are removed and the
/// new entry is appended. Non-mementor entries are always preserved.
fn upsert_hook_entry(
    hooks_obj: &mut serde_json::Map<String, serde_json::Value>,
    event_name: &str,
    hook_entry: serde_json::Value,
) {
    let arr = hooks_obj
        .entry(event_name)
        .or_insert_with(|| serde_json::json!([]));

    if let Some(arr) = arr.as_array_mut() {
        let target_cmd = hook_entry["hooks"]
            .as_array()
            .and_then(|h| h.first())
            .and_then(|h| h["command"].as_str());

        let already_exists = target_cmd.is_some_and(|cmd| {
            arr.iter().any(|entry| {
                entry["hooks"]
                    .as_array()
                    .is_some_and(|hooks| hooks.iter().any(|h| h["command"].as_str() == Some(cmd)))
            })
        });

        if already_exists {
            return;
        }

        arr.retain(|entry| {
            !entry["hooks"].as_array().is_some_and(|hooks| {
                hooks.iter().any(|h| {
                    h["command"]
                        .as_str()
                        .is_some_and(|cmd| cmd.starts_with("mementor"))
                })
            })
        });

        arr.push(hook_entry);
    }
}

#[cfg(test)]
mod tests {
    use mementor_lib::context::MementorContext;
    use mementor_lib::db::driver::DatabaseDriver;
    use mementor_lib::output::BufferedIO;
    use mementor_lib::runtime::Runtime;

    use crate::test_util::runtime_in_memory;
    use crate::test_util::trim_margin;

    #[test]
    fn try_run_enable_creates_db_and_configures_project() {
        let (_tmp, runtime) = runtime_in_memory("enable_creates");
        let mut io = BufferedIO::new();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        // stdout
        assert_eq!(io.stdout_to_string(), "mementor enabled successfully.\n");

        // stderr (db_path is dynamic, construct expected string)
        let db_path = runtime.context.db_path();
        let db_path = db_path.display();
        let expected_stderr = trim_margin!(
            "|Initializing database...
             |  Database created at {db_path}
             |Verifying embedding model...
             |  Embedding model OK
             |  .gitignore updated
             |  Claude Code hooks configured
             |"
        );
        assert_eq!(io.stderr_to_string(), expected_stderr);

        // .gitignore
        let gitignore = std::fs::read_to_string(runtime.context.gitignore_path()).unwrap();
        assert_eq!(gitignore, ".mementor/\n");

        // .claude/settings.json
        let raw = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        let settings: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(settings["hooks"]["Stop"].is_array());
        assert!(settings["hooks"]["UserPromptSubmit"].is_array());
    }

    #[test]
    fn try_run_enable_is_idempotent() {
        let (_tmp, runtime) = runtime_in_memory("enable_idempotent");
        let mut io = BufferedIO::new();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let mut io2 = BufferedIO::new();
        crate::try_run(&["mementor", "enable"], &runtime, &mut io2).unwrap();

        // Both runs produce identical stderr
        assert_eq!(io.stderr_to_string(), io2.stderr_to_string());

        // No duplicate .gitignore entries
        let gitignore = std::fs::read_to_string(runtime.context.gitignore_path()).unwrap();
        assert_eq!(
            gitignore.matches(".mementor/").count(),
            1,
            "gitignore should not have duplicate entries",
        );

        // No duplicate hook entries
        let raw = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        let settings: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            settings["hooks"]["Stop"].as_array().unwrap().len(),
            1,
            "Stop hooks should not have duplicates",
        );
    }

    #[test]
    fn try_run_enable_preserves_existing_settings() {
        let (_tmp, runtime) = runtime_in_memory("enable_preserves");
        let mut io = BufferedIO::new();

        // Pre-create settings.json with a custom key
        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            runtime.context.claude_settings_path(),
            r#"{"customKey": "customValue"}"#,
        )
        .unwrap();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let raw = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        let settings: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(settings["customKey"], "customValue");
        assert!(settings["hooks"]["Stop"].is_array());
    }

    /// Returns the expected JSON for only-mementor-hooks settings.
    fn mementor_hooks_only() -> String {
        crate::test_util::_trim_margin(
            r#"|{
               |  "hooks": {
               |    "Stop": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook stop"
               |          }
               |        ]
               |      }
               |    ],
               |    "UserPromptSubmit": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook user-prompt-submit"
               |          }
               |        ]
               |      }
               |    ],
               |    "PreCompact": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook pre-compact"
               |          }
               |        ]
               |      }
               |    ]
               |  }
               |}"#,
        )
    }

    #[test]
    fn try_run_enable_preserves_key_order() {
        let (_tmp, runtime) = runtime_in_memory("enable_key_order");
        let mut io = BufferedIO::new();

        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            runtime.context.claude_settings_path(),
            r#"{"permissions": {"allow": ["Bash(cargo:*)"]}, "hooks": {}}"#,
        )
        .unwrap();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        let expected = crate::test_util::_trim_margin(
            r#"|{
               |  "permissions": {
               |    "allow": [
               |      "Bash(cargo:*)"
               |    ]
               |  },
               |  "hooks": {
               |    "Stop": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook stop"
               |          }
               |        ]
               |      }
               |    ],
               |    "UserPromptSubmit": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook user-prompt-submit"
               |          }
               |        ]
               |      }
               |    ],
               |    "PreCompact": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook pre-compact"
               |          }
               |        ]
               |      }
               |    ]
               |  }
               |}"#,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_run_enable_preserves_eof_newline() {
        let (_tmp, runtime) = runtime_in_memory("enable_eof_nl");
        let mut io = BufferedIO::new();

        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();

        let hooks_only = mementor_hooks_only();

        // With trailing newline
        std::fs::write(runtime.context.claude_settings_path(), "{}\n").unwrap();
        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();
        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        assert_eq!(actual, format!("{hooks_only}\n"));

        // Without trailing newline
        std::fs::write(runtime.context.claude_settings_path(), "{}").unwrap();
        let mut io2 = BufferedIO::new();
        crate::try_run(&["mementor", "enable"], &runtime, &mut io2).unwrap();
        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        assert_eq!(actual, hooks_only);
    }

    #[test]
    fn try_run_enable_new_file_gets_trailing_newline() {
        let (_tmp, runtime) = runtime_in_memory("enable_new_file_nl");
        let mut io = BufferedIO::new();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        assert_eq!(actual, format!("{}\n", mementor_hooks_only()));
    }

    #[test]
    fn try_run_enable_upserts_mementor_hooks() {
        let (_tmp, runtime) = runtime_in_memory("enable_upsert");
        let mut io = BufferedIO::new();

        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            runtime.context.claude_settings_path(),
            crate::test_util::_trim_margin(
                r#"|{
                   |  "hooks": {
                   |    "Stop": [
                   |      {
                   |        "hooks": [
                   |          {
                   |            "type": "command",
                   |            "command": "mementor hook old-stop-command"
                   |          }
                   |        ]
                   |      }
                   |    ]
                   |  }
                   |}
                   |"#,
            ),
        )
        .unwrap();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        assert_eq!(actual, format!("{}\n", mementor_hooks_only()));
    }

    #[test]
    fn try_run_enable_preserves_non_mementor_hooks_in_same_event() {
        let (_tmp, runtime) = runtime_in_memory("enable_non_mementor");
        let mut io = BufferedIO::new();

        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            runtime.context.claude_settings_path(),
            crate::test_util::_trim_margin(
                r#"|{
                   |  "hooks": {
                   |    "Stop": [
                   |      {
                   |        "hooks": [
                   |          {
                   |            "type": "command",
                   |            "command": "other-tool hook stop"
                   |          }
                   |        ]
                   |      }
                   |    ]
                   |  }
                   |}
                   |"#,
            ),
        )
        .unwrap();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        let expected = crate::test_util::_trim_margin(
            r#"|{
               |  "hooks": {
               |    "Stop": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "other-tool hook stop"
               |          }
               |        ]
               |      },
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook stop"
               |          }
               |        ]
               |      }
               |    ],
               |    "UserPromptSubmit": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook user-prompt-submit"
               |          }
               |        ]
               |      }
               |    ],
               |    "PreCompact": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook pre-compact"
               |          }
               |        ]
               |      }
               |    ]
               |  }
               |}
               |"#,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_run_enable_preserves_unrelated_hook_events() {
        let (_tmp, runtime) = runtime_in_memory("enable_unrelated_events");
        let mut io = BufferedIO::new();

        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            runtime.context.claude_settings_path(),
            crate::test_util::_trim_margin(
                r#"|{
                   |  "hooks": {
                   |    "PreToolUse": [
                   |      {
                   |        "hooks": [
                   |          {
                   |            "type": "command",
                   |            "command": "my-linter check"
                   |          }
                   |        ]
                   |      }
                   |    ]
                   |  }
                   |}
                   |"#,
            ),
        )
        .unwrap();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        let expected = crate::test_util::_trim_margin(
            r#"|{
               |  "hooks": {
               |    "PreToolUse": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "my-linter check"
               |          }
               |        ]
               |      }
               |    ],
               |    "Stop": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook stop"
               |          }
               |        ]
               |      }
               |    ],
               |    "UserPromptSubmit": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook user-prompt-submit"
               |          }
               |        ]
               |      }
               |    ],
               |    "PreCompact": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook pre-compact"
               |          }
               |        ]
               |      }
               |    ]
               |  }
               |}
               |"#,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_run_enable_preserves_non_hooks_keys() {
        let (_tmp, runtime) = runtime_in_memory("enable_non_hooks_keys");
        let mut io = BufferedIO::new();

        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            runtime.context.claude_settings_path(),
            crate::test_util::_trim_margin(
                r#"|{
                   |  "attribution": {"commit": "", "pr": ""},
                   |  "permissions": {"allow": ["Bash(cargo:*)", "Skill(commit)"]},
                   |  "env": {"MY_VAR": "value"}
                   |}
                   |"#,
            ),
        )
        .unwrap();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        let expected = crate::test_util::_trim_margin(
            r#"|{
               |  "attribution": {
               |    "commit": "",
               |    "pr": ""
               |  },
               |  "permissions": {
               |    "allow": [
               |      "Bash(cargo:*)",
               |      "Skill(commit)"
               |    ]
               |  },
               |  "env": {
               |    "MY_VAR": "value"
               |  },
               |  "hooks": {
               |    "Stop": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook stop"
               |          }
               |        ]
               |      }
               |    ],
               |    "UserPromptSubmit": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook user-prompt-submit"
               |          }
               |        ]
               |      }
               |    ],
               |    "PreCompact": [
               |      {
               |        "hooks": [
               |          {
               |            "type": "command",
               |            "command": "mementor hook pre-compact"
               |          }
               |        ]
               |      }
               |    ]
               |  }
               |}
               |"#,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_run_enable_rejects_linked_worktree() {
        use mementor_test_util::git::{init_git_repo, run_git};

        let tmp = tempfile::tempdir().unwrap();
        let main_dir = tmp.path().join("main");
        std::fs::create_dir_all(&main_dir).unwrap();

        // Create a real git repo + linked worktree.
        init_git_repo(&main_dir);
        run_git(
            &main_dir,
            &[
                "worktree",
                "add",
                tmp.path().join("wt").to_str().unwrap(),
                "-b",
                "test-wt",
            ],
        );

        let wt_dir = tmp.path().join("wt");
        let ctx = MementorContext::with_cwd_and_log_dir(wt_dir, main_dir.clone(), true, None);
        let db = DatabaseDriver::in_memory("enable_reject_wt").unwrap();
        let runtime = Runtime { context: ctx, db };
        let mut io = BufferedIO::new();

        let result = crate::try_run(&["mementor", "enable"], &runtime, &mut io);
        let expected = format!(
            "mementor enable must be run from the primary worktree.\n\
             Primary worktree: {}\n\
             Run `mementor enable` from that directory instead.",
            main_dir.display(),
        );
        assert_eq!(result.unwrap_err().to_string(), expected);
        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");
    }

    /// Running `mementor enable` from a **subdirectory** of the primary
    /// worktree should succeed. The worktree kind is determined at startup
    /// by `.git` entry type, not by path comparison — so subdirectories
    /// inherit the "primary" classification from the root.
    #[test]
    fn try_run_enable_from_primary_subdirectory_should_succeed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("project");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir(root.join(".git")).unwrap();

        let subdir = root.join("src").join("deep");
        std::fs::create_dir_all(&subdir).unwrap();

        // is_linked_worktree = false: cwd is inside the primary worktree.
        let ctx = MementorContext::with_cwd_and_log_dir(subdir, root, false, None);
        let db = DatabaseDriver::in_memory("enable_primary_subdir").unwrap();
        let runtime = Runtime { context: ctx, db };
        let mut io = BufferedIO::new();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        assert_eq!(io.stdout_to_string(), "mementor enabled successfully.\n");

        let db_path = runtime.context.db_path();
        let db_path = db_path.display();
        let expected_stderr = trim_margin!(
            "|Initializing database...
             |  Database created at {db_path}
             |Verifying embedding model...
             |  Embedding model OK
             |  .gitignore updated
             |  Claude Code hooks configured
             |"
        );
        assert_eq!(io.stderr_to_string(), expected_stderr);
    }

    /// Running `mementor enable` from a **subdirectory** of a linked worktree
    /// should be rejected, just like running from the linked worktree root.
    #[test]
    fn try_run_enable_rejects_linked_worktree_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("main");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir(root.join(".git")).unwrap();

        let wt_subdir = tmp.path().join("wt").join("src").join("lib");
        std::fs::create_dir_all(&wt_subdir).unwrap();

        // is_linked_worktree = true: cwd is inside a linked worktree.
        let ctx = MementorContext::with_cwd_and_log_dir(wt_subdir, root.clone(), true, None);
        let db = DatabaseDriver::in_memory("enable_reject_wt_subdir").unwrap();
        let runtime = Runtime { context: ctx, db };
        let mut io = BufferedIO::new();

        let result = crate::try_run(&["mementor", "enable"], &runtime, &mut io);
        let expected = format!(
            "mementor enable must be run from the primary worktree.\n\
             Primary worktree: {}\n\
             Run `mementor enable` from that directory instead.",
            root.display(),
        );
        assert_eq!(result.unwrap_err().to_string(), expected);
        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");
    }

    #[test]
    fn try_run_enable_preserves_existing_key_order() {
        let (_tmp, runtime) = runtime_in_memory("enable_existing_order");
        let mut io = BufferedIO::new();

        let claude_dir = runtime
            .context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();

        // Real-world settings.json: "command" before "type" (Claude Code's ordering)
        let input = crate::test_util::_trim_margin(
            r#"|{
               |  "attribution": {
               |    "commit": "",
               |    "pr": ""
               |  },
               |  "hooks": {
               |    "PreCompact": [
               |      {
               |        "hooks": [
               |          {
               |            "command": "mementor hook pre-compact",
               |            "type": "command"
               |          }
               |        ]
               |      }
               |    ],
               |    "Stop": [
               |      {
               |        "hooks": [
               |          {
               |            "command": "mementor hook stop",
               |            "type": "command"
               |          }
               |        ]
               |      }
               |    ],
               |    "UserPromptSubmit": [
               |      {
               |        "hooks": [
               |          {
               |            "command": "mementor hook user-prompt-submit",
               |            "type": "command"
               |          }
               |        ]
               |      }
               |    ]
               |  },
               |  "permissions": {
               |    "allow": [
               |      "Bash(cargo:*)",
               |      "Bash(rustc --print:*)",
               |      "Bash(./scripts/update-sqlite-vector.sh:*)",
               |      "Bash(ls:*)",
               |      "Bash(find:*)",
               |      "Bash(grep:*)",
               |      "Skill(commit)",
               |      "Skill(worktree)"
               |    ]
               |  }
               |}"#,
        );
        std::fs::write(runtime.context.claude_settings_path(), format!("{input}\n")).unwrap();

        crate::try_run(&["mementor", "enable"], &runtime, &mut io).unwrap();

        let actual = std::fs::read_to_string(runtime.context.claude_settings_path()).unwrap();
        assert_eq!(actual, format!("{input}\n"));
    }
}
