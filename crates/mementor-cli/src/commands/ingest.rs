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

#[cfg(test)]
mod tests {
    use mementor_lib::output::BufferedIO;

    use crate::test_util::{make_entry, runtime_in_memory, runtime_not_enabled, write_transcript};

    #[test]
    fn try_run_ingest_success() {
        let (tmp, runtime) = runtime_in_memory("ingest_success");
        let mut io = BufferedIO::new();

        let lines = vec![
            make_entry("user", "Hello, how are you?"),
            make_entry("assistant", "I am doing great, thank you!"),
        ];
        let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let transcript = write_transcript(tmp.path(), &line_refs);

        crate::try_run(
            &["mementor", "ingest", transcript.to_str().unwrap(), "s1"],
            &runtime,
            &mut io,
        )
        .unwrap();

        assert_eq!(
            io.stdout_to_string(),
            "Ingestion complete for session s1.\n"
        );
        assert_eq!(io.stderr_to_string(), "");

        // Verify session was stored in DB
        let conn = runtime.db.open().unwrap();
        let session = mementor_lib::db::queries::get_session(&conn, "s1")
            .unwrap()
            .expect("session should exist");
        assert_eq!(session.session_id, "s1");
    }

    #[test]
    fn try_run_ingest_not_enabled() {
        let (_tmp, runtime) = runtime_not_enabled();
        let mut io = BufferedIO::new();

        let result = crate::try_run(
            &["mementor", "ingest", "/tmp/fake.jsonl", "s1"],
            &runtime,
            &mut io,
        );

        assert_eq!(
            result.unwrap_err().to_string(),
            "mementor is not enabled. Run `mementor enable` first.",
        );
        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");
    }

    #[test]
    fn try_run_ingest_transcript_not_found() {
        let (_tmp, runtime) = runtime_in_memory("ingest_not_found");
        let mut io = BufferedIO::new();

        let result = crate::try_run(
            &["mementor", "ingest", "/nonexistent/transcript.jsonl", "s1"],
            &runtime,
            &mut io,
        );

        assert_eq!(
            result.unwrap_err().to_string(),
            "Transcript file not found: /nonexistent/transcript.jsonl",
        );
        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");
    }
}
