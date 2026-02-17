use std::io::Write;
use std::path::Path;

use mementor_lib::context::MementorContext;
use mementor_lib::db::connection::open_db;
use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleOutput;
use mementor_lib::pipeline::chunker::load_tokenizer;
use mementor_lib::pipeline::ingest::run_ingest;

use super::input::StopHookInput;

/// Handle the Stop hook: parse stdin, run incremental ingest.
pub fn handle_stop<C, OUT, ERR>(
    input: &StopHookInput,
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
        writeln!(
            output.stderr(),
            "mementor is not enabled for this project. Run `mementor enable` first."
        )?;
        return Ok(());
    }

    let conn = open_db(&db_path)?;
    let mut embedder = Embedder::new()?;
    let tokenizer = load_tokenizer()?;

    run_ingest(
        &conn,
        &mut embedder,
        &tokenizer,
        &input.session_id,
        Path::new(&input.transcript_path),
        &input.cwd,
    )?;

    Ok(())
}
