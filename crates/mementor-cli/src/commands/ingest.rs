use std::io::{Read, Write};
use std::path::Path;

use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleIO;
use mementor_lib::pipeline::chunker::load_tokenizer;
use mementor_lib::pipeline::ingest::run_ingest;
use mementor_lib::runtime::Runtime;

/// Run the `mementor ingest` command.
pub fn run_ingest_cmd<IN, OUT, ERR>(
    transcript: &str,
    session_id: &str,
    runtime: &Runtime,
    io: &mut dyn ConsoleIO<IN, OUT, ERR>,
) -> anyhow::Result<()>
where
    IN: Read,
    OUT: Write,
    ERR: Write,
{
    if !runtime.db.is_ready() {
        anyhow::bail!("mementor is not enabled. Run `mementor enable` first.");
    }

    let conn = runtime.db.open()?;
    let mut embedder = Embedder::new()?;
    let tokenizer = load_tokenizer()?;

    let transcript_path = Path::new(transcript);
    if !transcript_path.exists() {
        anyhow::bail!("Transcript file not found: {transcript}");
    }

    let project_dir = runtime.context.project_root().to_string_lossy().to_string();
    run_ingest(
        &conn,
        &mut embedder,
        &tokenizer,
        session_id,
        transcript_path,
        &project_dir,
    )?;

    writeln!(io.stdout(), "Ingestion complete for session {session_id}.")?;
    Ok(())
}
