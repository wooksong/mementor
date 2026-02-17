use std::io::{Read, Write};

use mementor_lib::config::DEFAULT_TOP_K;
use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleIO;
use mementor_lib::pipeline::ingest::search_context;
use mementor_lib::runtime::Runtime;

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
    if !runtime.db.is_ready() {
        // Silently skip — mementor not enabled
        return Ok(());
    }

    let conn = runtime.db.open()?;
    let mut embedder = Embedder::new()?;

    let rag_context = search_context(&conn, &mut embedder, &input.prompt, DEFAULT_TOP_K)?;

    if !rag_context.is_empty() {
        write!(io.stdout(), "{rag_context}")?;
    }

    Ok(())
}
