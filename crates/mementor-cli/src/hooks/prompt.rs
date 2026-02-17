use std::io::Write;

use mementor_lib::config::DEFAULT_TOP_K;
use mementor_lib::context::MementorContext;
use mementor_lib::db::connection::open_db;
use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleOutput;
use mementor_lib::pipeline::ingest::search_context;

use super::input::PromptHookInput;

/// Handle the `UserPromptSubmit` hook: embed prompt, search memories,
/// output RAG context to stdout.
pub fn handle_prompt<C, OUT, ERR>(
    input: &PromptHookInput,
    context: &C,
    output: &mut dyn ConsoleOutput<OUT, ERR>,
) -> anyhow::Result<()>
where
    C: MementorContext,
    OUT: Write,
    ERR: Write,
{
    let db_path = context.db_path();
    if !db_path.exists() {
        // Silently skip — mementor not enabled
        return Ok(());
    }

    let conn = open_db(&db_path)?;
    let mut embedder = Embedder::new()?;

    let rag_context = search_context(&conn, &mut embedder, &input.prompt, DEFAULT_TOP_K)?;

    if !rag_context.is_empty() {
        write!(output.stdout(), "{rag_context}")?;
    }

    Ok(())
}
