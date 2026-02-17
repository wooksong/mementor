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

    let mut settings: serde_json::Value = if settings_path.exists() {
        let raw = std::fs::read_to_string(&settings_path)?;
        serde_json::from_str(&raw)?
    } else {
        serde_json::json!({})
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

    // Merge hooks into settings
    let hooks = settings
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("settings.json root is not an object"))?
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));

    let hooks_obj = hooks
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("hooks field is not an object"))?;

    // Add Stop hook (as array, append if exists)
    merge_hook_array(hooks_obj, "Stop", stop_hook);

    // Add UserPromptSubmit hook
    merge_hook_array(hooks_obj, "UserPromptSubmit", prompt_hook);

    // Write settings back
    let formatted = serde_json::to_string_pretty(&settings)?;
    std::fs::write(&settings_path, formatted)?;

    Ok(())
}

/// Merge a hook entry into the hook array, avoiding duplicates.
fn merge_hook_array(
    hooks_obj: &mut serde_json::Map<String, serde_json::Value>,
    event_name: &str,
    hook_entry: serde_json::Value,
) {
    let arr = hooks_obj
        .entry(event_name)
        .or_insert_with(|| serde_json::json!([]));

    if let Some(arr) = arr.as_array_mut() {
        // Check if mementor hook already exists
        let has_mementor = arr.iter().any(|entry| {
            entry["hooks"].as_array().is_some_and(|hooks| {
                hooks.iter().any(|h| {
                    h["command"]
                        .as_str()
                        .is_some_and(|cmd| cmd.starts_with("mementor"))
                })
            })
        });

        if !has_mementor {
            arr.push(hook_entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use mementor_lib::output::BufferedIO;

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
}
