use std::io::{Read, Write};

use mementor_lib::embedding::embedder::Embedder;
use mementor_lib::output::ConsoleIO;
use mementor_lib::pipeline::ingest::search_context;
use mementor_lib::runtime::Runtime;

/// Run the `mementor query` command.
pub fn run_query<IN, OUT, ERR>(
    text: &str,
    k: usize,
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

    let result = search_context(&conn, &mut embedder, text, k)?;

    if result.is_empty() {
        writeln!(io.stdout(), "No matching memories found.")?;
    } else {
        write!(io.stdout(), "{result}")?;
    }

    Ok(())
}
