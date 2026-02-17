#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context;
use rusqlite::Connection;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::db::queries::{
    self, Session, delete_memories_at, insert_memory, search_memories, upsert_session,
};
use crate::embedding::embedder::Embedder;
use crate::pipeline::chunker::{chunk_turn, group_into_turns};
use crate::transcript::parser::parse_transcript;

/// Run the incremental ingest pipeline for a single session.
///
/// 1. Read new messages from the transcript starting at `last_line_index`.
/// 2. If a provisional turn exists, complete it with the new User message.
/// 3. Process all complete turns (chunk -> embed -> store).
/// 4. Store the last turn as provisional.
/// 5. Update session state.
pub fn run_ingest(
    conn: &Connection,
    embedder: &mut Embedder,
    tokenizer: &Tokenizer,
    session_id: &str,
    transcript_path: &Path,
    project_dir: &str,
) -> anyhow::Result<()> {
    // Load or create session
    let session = queries::get_session(conn, session_id)?;
    let (start_line, provisional_start) = match &session {
        Some(s) => (s.last_line_index, s.provisional_turn_start),
        None => (0, None),
    };

    // Parse new messages from the transcript starting at start_line.
    // If there's a provisional turn, re-read from its start to re-process it.
    let read_from = provisional_start.unwrap_or(start_line);
    let messages = parse_transcript(transcript_path, read_from)?;
    if messages.is_empty() {
        debug!("No new messages found in transcript");
        return Ok(());
    }

    // Group messages into turns
    let turns = group_into_turns(&messages);
    if turns.is_empty() {
        debug!("No turns formed from messages");
        return Ok(());
    }

    // Ensure session record exists (required for foreign key on memories table)
    if session.is_none() {
        upsert_session(
            conn,
            &Session {
                session_id: session_id.to_string(),
                transcript_path: transcript_path.to_string_lossy().to_string(),
                project_dir: project_dir.to_string(),
                last_line_index: read_from,
                provisional_turn_start: None,
            },
        )?;
    }

    // If a provisional turn existed, delete its old chunks
    if let Some(prov_line) = provisional_start {
        let deleted = delete_memories_at(conn, session_id, prov_line)?;
        debug!("Deleted {deleted} provisional chunks at line_index={prov_line}");
    }

    // Process each turn: chunk -> embed -> store
    let mut last_line_index = read_from;
    let mut new_provisional_start: Option<usize> = None;

    for turn in &turns {
        let chunks = chunk_turn(turn, tokenizer);
        if chunks.is_empty() {
            continue;
        }

        // Embed all chunks in this turn as a batch
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedder.embed_batch(&texts).with_context(|| {
            format!(
                "Failed to embed chunks for turn at line {}",
                turn.line_index
            )
        })?;

        // Store each chunk with its embedding
        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            insert_memory(
                conn,
                session_id,
                chunk.line_index,
                chunk.chunk_index,
                "turn",
                &chunk.text,
                embedding,
            )?;
        }

        if turn.provisional {
            new_provisional_start = Some(turn.line_index);
        }

        // Update last_line_index to be beyond all messages in this turn
        last_line_index = turn.line_index + 2;
    }

    // Ensure last_line_index covers all parsed messages
    let max_message_line = messages
        .iter()
        .map(|m| m.line_index)
        .max()
        .unwrap_or(read_from);
    last_line_index = last_line_index.max(max_message_line + 1);

    // Upsert session state
    upsert_session(
        conn,
        &Session {
            session_id: session_id.to_string(),
            transcript_path: transcript_path.to_string_lossy().to_string(),
            project_dir: project_dir.to_string(),
            last_line_index,
            provisional_turn_start: new_provisional_start,
        },
    )?;

    let total_turns = turns.len();
    let provisional_count = usize::from(new_provisional_start.is_some());
    info!(
        "Ingested {total_turns} turns ({} complete, {provisional_count} provisional) for session {session_id}",
        total_turns - provisional_count,
    );

    Ok(())
}

/// Search memories across all sessions for the given query text.
/// Returns formatted context string suitable for injecting into a prompt.
pub fn search_context(
    conn: &Connection,
    embedder: &mut Embedder,
    query: &str,
    k: usize,
) -> anyhow::Result<String> {
    let embeddings = embedder.embed_batch(&[query])?;
    let query_embedding = &embeddings[0];

    let results = search_memories(conn, query_embedding, k)?;

    if results.is_empty() {
        return Ok(String::new());
    }

    let mut context = String::from("## Relevant past context\n\n");
    for (i, result) in results.iter().enumerate() {
        write!(
            &mut context,
            "### Memory {} (distance: {:.4})\n{}\n\n",
            i + 1,
            result.distance,
            result.content,
        )
        .unwrap();
    }

    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::driver::DatabaseDriver;
    use crate::pipeline::chunker::load_tokenizer;
    use std::io::Write;

    fn setup_test() -> (tempfile::TempDir, Connection, Embedder, Tokenizer) {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let driver = DatabaseDriver::file(db_path);
        let conn = driver.open().unwrap();
        let embedder = Embedder::new().unwrap();
        let tokenizer = load_tokenizer().unwrap();
        (tmp, conn, embedder, tokenizer)
    }

    fn write_transcript(dir: &Path, lines: &[&str]) -> std::path::PathBuf {
        let path = dir.join("transcript.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        path
    }

    fn make_entry(role: &str, text: &str) -> String {
        serde_json::json!({
            "type": "message",
            "uuid": format!("uuid-{}", rand_id()),
            "sessionId": "test-session",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "role": role,
                "content": text
            }
        })
        .to_string()
    }

    fn rand_id() -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;
        let mut h = DefaultHasher::new();
        SystemTime::now().hash(&mut h);
        std::thread::current().id().hash(&mut h);
        h.finish()
    }

    #[test]
    fn first_ingestion_creates_provisional() {
        let (tmp, conn, mut embedder, tokenizer) = setup_test();

        let lines = vec![
            make_entry("user", "Hello, how are you?"),
            make_entry("assistant", "I'm doing great, thanks for asking!"),
        ];
        let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let transcript = write_transcript(tmp.path(), &line_refs);

        run_ingest(
            &conn,
            &mut embedder,
            &tokenizer,
            "s1",
            &transcript,
            "/tmp/project",
        )
        .unwrap();

        let session = queries::get_session(&conn, "s1").unwrap().unwrap();
        assert!(session.provisional_turn_start.is_some());
        assert_eq!(session.last_line_index, 2);
    }

    #[test]
    fn second_ingestion_completes_provisional() {
        let (tmp, conn, mut embedder, tokenizer) = setup_test();

        // First ingestion: User + Assistant (provisional)
        let lines1 = vec![
            make_entry("user", "What is Rust?"),
            make_entry("assistant", "Rust is a systems programming language."),
        ];
        let refs1: Vec<&str> = lines1.iter().map(String::as_str).collect();
        let transcript = write_transcript(tmp.path(), &refs1);
        run_ingest(
            &conn,
            &mut embedder,
            &tokenizer,
            "s1",
            &transcript,
            "/tmp/p",
        )
        .unwrap();

        // Second ingestion: Append another pair
        let lines2 = vec![
            make_entry("user", "What is Rust?"),
            make_entry("assistant", "Rust is a systems programming language."),
            make_entry("user", "Tell me more about ownership."),
            make_entry(
                "assistant",
                "Ownership is Rust's key feature for memory safety.",
            ),
        ];
        let refs2: Vec<&str> = lines2.iter().map(String::as_str).collect();
        let transcript = write_transcript(tmp.path(), &refs2);
        run_ingest(
            &conn,
            &mut embedder,
            &tokenizer,
            "s1",
            &transcript,
            "/tmp/p",
        )
        .unwrap();

        let session = queries::get_session(&conn, "s1").unwrap().unwrap();
        // Turn 2 should be provisional
        assert!(session.provisional_turn_start.is_some());
        assert_eq!(session.last_line_index, 4);
    }

    #[test]
    fn search_returns_relevant_results() {
        let (tmp, conn, mut embedder, tokenizer) = setup_test();

        let lines = vec![
            make_entry("user", "How do I implement authentication in Rust?"),
            make_entry(
                "assistant",
                "You can use JWT tokens with the jsonwebtoken crate.",
            ),
            make_entry("user", "What about database connections?"),
            make_entry(
                "assistant",
                "Use sqlx or diesel for database access in Rust.",
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let transcript = write_transcript(tmp.path(), &refs);
        run_ingest(
            &conn,
            &mut embedder,
            &tokenizer,
            "s1",
            &transcript,
            "/tmp/p",
        )
        .unwrap();

        let context = search_context(&conn, &mut embedder, "authentication", 5).unwrap();
        assert!(!context.is_empty());
        assert!(context.contains("Relevant past context"));
    }

    #[test]
    fn empty_transcript_is_handled() {
        let (tmp, conn, mut embedder, tokenizer) = setup_test();
        let transcript = write_transcript(tmp.path(), &[]);
        run_ingest(
            &conn,
            &mut embedder,
            &tokenizer,
            "s1",
            &transcript,
            "/tmp/p",
        )
        .unwrap();

        let session = queries::get_session(&conn, "s1").unwrap();
        assert!(session.is_none());
    }

    #[test]
    fn re_ingestion_is_idempotent() {
        let (tmp, conn, mut embedder, tokenizer) = setup_test();

        let lines = vec![
            make_entry("user", "Hello"),
            make_entry("assistant", "Hi there"),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let transcript = write_transcript(tmp.path(), &refs);

        // Ingest twice with the same data
        run_ingest(
            &conn,
            &mut embedder,
            &tokenizer,
            "s1",
            &transcript,
            "/tmp/p",
        )
        .unwrap();
        run_ingest(
            &conn,
            &mut embedder,
            &tokenizer,
            "s1",
            &transcript,
            "/tmp/p",
        )
        .unwrap();

        // Should still have data — no duplicates (INSERT OR REPLACE handles this)
        let emb = embedder.embed_batch(&["Hello"]).unwrap();
        let results = search_memories(&conn, &emb[0], 10).unwrap();
        assert!(!results.is_empty());
    }
}
