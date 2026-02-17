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

#[cfg(test)]
mod tests {
    use mementor_lib::output::BufferedIO;

    use crate::test_util::{make_entry, runtime_in_memory, runtime_not_enabled, write_transcript};

    #[test]
    fn try_run_hook_stop_success() {
        let (tmp, runtime) = runtime_in_memory("hook_stop_success");

        let lines = vec![
            make_entry("user", "What is Rust?"),
            make_entry("assistant", "Rust is a systems programming language."),
        ];
        let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let transcript = write_transcript(tmp.path(), &line_refs);

        let stdin_json = serde_json::json!({
            "session_id": "s1",
            "transcript_path": transcript.to_str().unwrap(),
            "cwd": tmp.path().to_str().unwrap()
        })
        .to_string();
        let mut io = BufferedIO::with_stdin(stdin_json.as_bytes());

        crate::try_run(&["mementor", "hook", "stop"], &runtime, &mut io).unwrap();

        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(io.stderr_to_string(), "");

        // Verify session was stored in DB
        let conn = runtime.db.open().unwrap();
        let session = mementor_lib::db::queries::get_session(&conn, "s1")
            .unwrap()
            .expect("session should exist");
        assert_eq!(session.session_id, "s1");
    }

    #[test]
    fn try_run_hook_stop_not_enabled() {
        let (_tmp, runtime) = runtime_not_enabled();
        let stdin_json = serde_json::json!({
            "session_id": "s1",
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/tmp/project"
        })
        .to_string();
        let mut io = BufferedIO::with_stdin(stdin_json.as_bytes());

        crate::try_run(&["mementor", "hook", "stop"], &runtime, &mut io).unwrap();

        assert_eq!(io.stdout_to_string(), "");
        assert_eq!(
            io.stderr_to_string(),
            "mementor is not enabled for this project. Run `mementor enable` first.\n",
        );
    }

    #[test]
    fn try_run_hook_stop_invalid_json() {
        let (_tmp, runtime) = runtime_in_memory("hook_stop_invalid");
        let mut io = BufferedIO::with_stdin(b"not valid json");

        let result = crate::try_run(&["mementor", "hook", "stop"], &runtime, &mut io);

        assert!(result.is_err());
    }
}
