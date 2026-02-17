use std::io::Write;
use std::path::Path;

use mementor_lib::context::MementorContext;
use mementor_lib::db::connection::open_db;
use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleOutput;
use mementor_lib::pipeline::chunker::load_tokenizer;
use mementor_lib::pipeline::ingest::run_ingest;

/// Run the `mementor ingest` command.
pub fn run_ingest_cmd<C, OUT, ERR>(
    transcript: &str,
    session_id: &str,
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
        anyhow::bail!("mementor is not enabled. Run `mementor enable` first.");
    }

    let conn = open_db(&db_path)?;
    let mut embedder = Embedder::new()?;
    let tokenizer = load_tokenizer()?;

    let transcript_path = Path::new(transcript);
    if !transcript_path.exists() {
        anyhow::bail!("Transcript file not found: {transcript}");
    }

    let project_dir = context.project_root().to_string_lossy().to_string();
    run_ingest(
        &conn,
        &mut embedder,
        &tokenizer,
        session_id,
        transcript_path,
        &project_dir,
    )?;

    writeln!(
        output.stdout(),
        "Ingestion complete for session {session_id}."
    )?;
    Ok(())
}
