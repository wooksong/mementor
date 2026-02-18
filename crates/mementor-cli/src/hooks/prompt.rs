use std::io::{Read, Write};

use mementor_lib::config::DEFAULT_TOP_K;
use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleIO;
use mementor_lib::pipeline::ingest::search_context;
use mementor_lib::runtime::Runtime;
use tracing::debug;

use super::input::PromptHookInput;

/// Handle the `UserPromptSubmit` hook: embed prompt, search memories,
/// output RAG context to stdout.
pub fn handle_prompt<IN, OUT, ERR>(
    input: &PromptHookInput,
    runtime: &Runtime,
    io: &mut dyn ConsoleIO<IN, OUT, ERR>,
) -> anyhow::Result<()>
where
    IN: Read,
    OUT: Write,
    ERR: Write,
{
    debug!(
        hook = "UserPromptSubmit",
        session_id = %input.session_id,
        prompt_len = input.prompt.len(),
        prompt = %input.prompt,
        "Hook received"
    );

    if input.prompt.is_empty() {
        debug!(
            hook = "UserPromptSubmit",
            "prompt is empty, skipping recall"
        );
        return Ok(());
    }

    if !runtime.db.is_ready() {
        // Silently skip — mementor not enabled
        return Ok(());
    }

    let conn = runtime.db.open()?;
    let mut embedder = Embedder::new()?;

    let rag_context = search_context(
        &conn,
        &mut embedder,
        &input.prompt,
        DEFAULT_TOP_K,
        Some(&input.session_id),
    )?;

    debug!(
        hook = "UserPromptSubmit",
        context_len = rag_context.len(),
        has_results = !rag_context.is_empty(),
        context = %rag_context,
        "Search completed"
    );

    if !rag_context.is_empty() {
        write!(io.stdout(), "{rag_context}")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use mementor_lib::embedding::embedder::Embedder;
    use mementor_lib::output::BufferedIO;

    use crate::test_util::{runtime_in_memory, runtime_not_enabled, seed_memory};

    #[test]
    fn try_run_hook_prompt_with_results() {
        let (_tmp, runtime) = runtime_in_memory("hook_prompt_results");
        let mut embedder = Embedder::new().unwrap();

        let seed_text = "Implementing authentication in Rust";
        seed_memory(&runtime.db, &mut embedder, "s1", 0, 0, seed_text);

        let stdin_json = serde_json::json!({
            "session_id": "s2",
            "prompt": seed_text,
            "cwd": "/tmp/project"
        })
        .to_string();
        let mut io = BufferedIO::with_stdin(stdin_json.as_bytes());

        crate::try_run(
            &["mementor", "hook", "user-prompt-submit"],
            &runtime,
            &mut io,
        )
        .unwrap();

        let expected_stdout = format!(
            "## Relevant past context\n\n\
             ### Memory 1 (distance: 0.0000)\n\
             {seed_text}\n\n",
        );
        assert_eq!(io.stdout_to_string(), expected_stdout);
        assert_eq!(io.stderr_to_string(), "");
    }

    #[test]
    fn try_run_hook_prompt_not_enabled() {
        let (_tmp, runtime) = runtime_not_enabled();
        let stdin_json = serde_json::json!({
            "session_id": "s1",
            "prompt": "How do I do X?",
            "cwd": "/tmp/project"
        })
        .to_string();
        let mut io = BufferedIO::with_stdin(stdin_json.as_bytes());

        crate::try_run(
            &["mementor", "hook", "user-prompt-submit"],
            &runtime,
            &mut io,
        )
        .unwrap();

        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");
    }

    #[test]
    fn try_run_hook_prompt_null_prompt() {
        let (_tmp, runtime) = runtime_in_memory("hook_prompt_null");

        let stdin_json = serde_json::json!({
            "session_id": "s1",
            "prompt": null,
            "cwd": "/tmp/project"
        })
        .to_string();
        let mut io = BufferedIO::with_stdin(stdin_json.as_bytes());

        crate::try_run(
            &["mementor", "hook", "user-prompt-submit"],
            &runtime,
            &mut io,
        )
        .unwrap();

        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");
    }

    #[test]
    fn try_run_hook_prompt_invalid_json() {
        let (_tmp, runtime) = runtime_in_memory("hook_prompt_invalid");
        let mut io = BufferedIO::with_stdin(b"not valid json");

        let result = crate::try_run(
            &["mementor", "hook", "user-prompt-submit"],
            &runtime,
            &mut io,
        );

        assert!(result.is_err());
    }
}
