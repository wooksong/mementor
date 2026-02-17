use std::io::{Read, Write};
use std::path::Path;

use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleIO;
use mementor_lib::pipeline::chunker::load_tokenizer;
use mementor_lib::pipeline::ingest::run_ingest;
use mementor_lib::runtime::Runtime;

use super::input::StopHookInput;

/// Handle the Stop hook: parse stdin, run incremental ingest.
pub fn handle_stop<IN, OUT, ERR>(
    input: &StopHookInput,
    runtime: &Runtime,
    io: &mut dyn ConsoleIO<IN, OUT, ERR>,
) -> anyhow::Result<()>
where
    IN: Read,
    OUT: Write,
    ERR: Write,
{
    if !runtime.db.is_ready() {
        writeln!(
            io.stderr(),
            "mementor is not enabled for this project. Run `mementor enable` first."
        )?;
        return Ok(());
    }

    let conn = runtime.db.open()?;
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
