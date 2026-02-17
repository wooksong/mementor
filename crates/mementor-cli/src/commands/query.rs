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

#[cfg(test)]
mod tests {
    use mementor_lib::embedding::embedder::Embedder;
    use mementor_lib::output::BufferedIO;

    use crate::test_util::{runtime_in_memory, runtime_not_enabled, seed_memory};

    #[test]
    fn try_run_query_with_results() {
        let (_tmp, runtime) = runtime_in_memory("query_with_results");
        let mut embedder = Embedder::new().unwrap();

        let seed_text = "Hello world";
        seed_memory(&runtime.db, &mut embedder, "s1", 0, 0, seed_text);

        let mut io = BufferedIO::new();
        crate::try_run(&["mementor", "query", seed_text], &runtime, &mut io).unwrap();

        let expected_stdout = format!(
            "## Relevant past context\n\n\
             ### Memory 1 (distance: 0.0000)\n\
             {seed_text}\n\n",
        );
        assert_eq!(io.stdout_to_string(), expected_stdout);
        assert_eq!(io.stderr_to_string(), "");
    }

    #[test]
    fn try_run_query_no_results() {
        let (_tmp, runtime) = runtime_in_memory("query_no_results");
        let mut io = BufferedIO::new();

        crate::try_run(
            &["mementor", "query", "some search text"],
            &runtime,
            &mut io,
        )
        .unwrap();

        assert_eq!(io.stdout_to_string(), "No matching memories found.\n");
        assert_eq!(io.stderr_to_string(), "");
    }

    #[test]
    fn try_run_query_not_enabled() {
        let (_tmp, runtime) = runtime_not_enabled();
        let mut io = BufferedIO::new();

        let result = crate::try_run(&["mementor", "query", "test query"], &runtime, &mut io);

        assert_eq!(
            result.unwrap_err().to_string(),
            "mementor is not enabled. Run `mementor enable` first.",
        );
        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");
    }
}
