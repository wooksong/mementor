use std::io::Write;

use mementor_lib::context::MementorContext;
use mementor_lib::db::connection::open_db;
use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleOutput;

/// Run the `mementor enable` command.
///
/// 1. Create `.mementor/` directory and initialize DB with schema.
/// 2. Verify bundled embedding model loads.
/// 3. Append `.mementor/` to `.gitignore` if not present.
/// 4. Add mementor hooks to `.claude/settings.json`.
pub fn run_enable<C, OUT, ERR>(
    context: &C,
    output: &mut dyn ConsoleOutput<OUT, ERR>,
) -> anyhow::Result<()>
where
    C: MementorContext,
    OUT: Write,
    ERR: Write,
{
    let db_path = context.db_path();

    // Step 1: Create DB (open_db creates parent dirs + schema + vector_init)
    writeln!(output.stderr(), "Initializing database...")?;
    let _conn = open_db(&db_path)?;
    writeln!(
        output.stderr(),
        "  Database created at {}",
        db_path.display()
    )?;

    // Step 2: Verify embedding model loads
    writeln!(output.stderr(), "Verifying embedding model...")?;
    let _embedder = Embedder::new()?;
    writeln!(output.stderr(), "  Embedding model OK")?;

    // Step 3: Update .gitignore
    update_gitignore(context)?;
    writeln!(output.stderr(), "  .gitignore updated")?;

    // Step 4: Configure Claude Code hooks
    configure_hooks(context)?;
    writeln!(output.stderr(), "  Claude Code hooks configured")?;

    writeln!(output.stdout(), "mementor enabled successfully.")?;
    Ok(())
}

/// Append `.mementor/` to `.gitignore` if not already present.
fn update_gitignore<C: MementorContext>(ctx: &C) -> anyhow::Result<()> {
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
fn configure_hooks<C: MementorContext>(ctx: &C) -> anyhow::Result<()> {
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
    use super::*;
    use mementor_lib::context::RealMementorContext;
    use mementor_lib::output::BufferedOutput;

    #[test]
    fn enable_creates_db_and_gitignore() {
        let tmp = tempfile::tempdir().unwrap();
        let context = RealMementorContext::new(tmp.path().to_path_buf());
        let mut output = BufferedOutput::new();

        run_enable(&context, &mut output).unwrap();

        assert!(context.db_path().exists());
        assert!(context.gitignore_path().exists());
        let gitignore = std::fs::read_to_string(context.gitignore_path()).unwrap();
        assert!(gitignore.contains(".mementor/"));
    }

    #[test]
    fn enable_creates_hooks_config() {
        let tmp = tempfile::tempdir().unwrap();
        let context = RealMementorContext::new(tmp.path().to_path_buf());
        let mut output = BufferedOutput::new();

        run_enable(&context, &mut output).unwrap();

        let settings_path = context.claude_settings_path();
        assert!(settings_path.exists());
        let content = std::fs::read_to_string(&settings_path).unwrap();
        let settings: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(settings["hooks"]["Stop"].is_array());
        assert!(settings["hooks"]["UserPromptSubmit"].is_array());
    }

    #[test]
    fn enable_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let context = RealMementorContext::new(tmp.path().to_path_buf());
        let mut output = BufferedOutput::new();

        run_enable(&context, &mut output).unwrap();
        run_enable(&context, &mut output).unwrap();

        let gitignore = std::fs::read_to_string(context.gitignore_path()).unwrap();
        // Should only contain one .mementor/ entry
        assert_eq!(
            gitignore.matches(".mementor/").count(),
            1,
            "gitignore should not have duplicate entries"
        );

        let settings_content = std::fs::read_to_string(context.claude_settings_path()).unwrap();
        let settings: serde_json::Value = serde_json::from_str(&settings_content).unwrap();
        assert_eq!(
            settings["hooks"]["Stop"].as_array().unwrap().len(),
            1,
            "Stop hooks should not have duplicates"
        );
    }

    #[test]
    fn enable_preserves_existing_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let context = RealMementorContext::new(tmp.path().to_path_buf());
        let mut output = BufferedOutput::new();

        // Create existing settings with custom key
        let claude_dir = context
            .claude_settings_path()
            .parent()
            .unwrap()
            .to_path_buf();
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            context.claude_settings_path(),
            r#"{"customKey": "customValue"}"#,
        )
        .unwrap();

        run_enable(&context, &mut output).unwrap();

        let content = std::fs::read_to_string(context.claude_settings_path()).unwrap();
        let settings: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(settings["customKey"], "customValue");
        assert!(settings["hooks"]["Stop"].is_array());
    }
}
