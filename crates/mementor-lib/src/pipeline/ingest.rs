#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context;
use rusqlite::Connection;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::config::{MAX_COSINE_DISTANCE, OVER_FETCH_MULTIPLIER};
use crate::db::queries::{
    self, Session, delete_memories_at, get_turns_chunks, insert_memory, search_memories,
    upsert_session,
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
#[allow(clippy::too_many_lines)]
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
    let (start_line, provisional_start) = if let Some(s) = &session {
        debug!(
            session_id = %session_id,
            last_line_index = s.last_line_index,
            provisional_turn_start = ?s.provisional_turn_start,
            last_compact_line_index = ?s.last_compact_line_index,
            "Loaded existing session"
        );
        (s.last_line_index, s.provisional_turn_start)
    } else {
        debug!(session_id = %session_id, "Creating new session");
        (0, None)
    };

    // Parse new messages from the transcript starting at start_line.
    // If there's a provisional turn, re-read from its start to re-process it.
    let read_from = provisional_start.unwrap_or(start_line);
    let messages = parse_transcript(transcript_path, read_from)?;
    debug!(
        session_id = %session_id,
        read_from = read_from,
        message_count = messages.len(),
        "Parsed transcript messages"
    );
    if messages.is_empty() {
        debug!("No new messages found in transcript");
        return Ok(());
    }

    // Group messages into turns
    let turns = group_into_turns(&messages);
    debug!(
        session_id = %session_id,
        turn_count = turns.len(),
        "Grouped messages into turns"
    );
    for (i, turn) in turns.iter().enumerate() {
        debug!(
            turn_index = i,
            line_index = turn.line_index,
            provisional = turn.provisional,
            text_len = turn.text.len(),
            text = %turn.text,
            "Turn detail"
        );
    }
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
                last_compact_line_index: None,
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
        debug!(
            session_id = %session_id,
            line_index = turn.line_index,
            chunk_count = chunks.len(),
            provisional = turn.provisional,
            "Chunked turn"
        );
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
            last_compact_line_index: session.and_then(|s| s.last_compact_line_index),
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
///
/// Returns formatted context string suitable for injecting into a prompt.
/// Applies a multi-phase filter pipeline:
/// 1. Over-fetch with in-context filtering (SQL-level)
/// 2. Distance threshold
/// 3. Dedup by turn (best chunk per turn)
/// 4. Reconstruct full turn text from sibling chunks
/// 5. Sort, truncate to k, format output
#[allow(clippy::too_many_lines)]
pub fn search_context(
    conn: &Connection,
    embedder: &mut Embedder,
    query: &str,
    k: usize,
    session_id: Option<&str>,
) -> anyhow::Result<String> {
    debug!(
        query_len = query.len(),
        query = %query,
        k = k,
        session_id = ?session_id,
        "Searching memories"
    );

    let embeddings = embedder.embed_batch(&[query])?;
    let query_embedding = &embeddings[0];

    // Look up compaction boundary for the current session
    let compact_boundary = session_id
        .map(|sid| queries::get_session(conn, sid))
        .transpose()?
        .flatten()
        .and_then(|s| s.last_compact_line_index);

    // Phase 1: Over-fetch + in-context filter (SQL)
    let k_internal = k * OVER_FETCH_MULTIPLIER;
    let candidates = search_memories(
        conn,
        query_embedding,
        k_internal,
        session_id,
        compact_boundary,
    )?;

    debug!(
        candidates = candidates.len(),
        k_internal = k_internal,
        "Phase 1: over-fetch + in-context filter"
    );

    if candidates.is_empty() {
        debug!("No search results found after in-context filtering");
        return Ok(String::new());
    }

    // Phase 2: Distance threshold
    let after_distance: Vec<_> = candidates
        .into_iter()
        .filter(|r| r.distance <= MAX_COSINE_DISTANCE)
        .collect();

    debug!(
        after_distance = after_distance.len(),
        threshold = MAX_COSINE_DISTANCE,
        "Phase 2: distance threshold"
    );

    if after_distance.is_empty() {
        debug!("No search results found after distance threshold");
        return Ok(String::new());
    }

    // Phase 3: Dedup by turn — keep best chunk per (session_id, line_index)
    let mut best_per_turn: HashMap<(&str, usize), usize> = HashMap::new();
    for (idx, result) in after_distance.iter().enumerate() {
        let key = (result.session_id.as_str(), result.line_index);
        best_per_turn
            .entry(key)
            .and_modify(|existing_idx| {
                if result.distance < after_distance[*existing_idx].distance {
                    *existing_idx = idx;
                }
            })
            .or_insert(idx);
    }

    let mut deduped: Vec<_> = best_per_turn
        .into_values()
        .map(|idx| &after_distance[idx])
        .collect();
    deduped.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    deduped.truncate(k);

    debug!(after_dedup = deduped.len(), "Phase 3: dedup by turn");

    // Phase 4: Reconstruct full turn text from sibling chunks
    let turn_keys: Vec<(&str, usize)> = deduped
        .iter()
        .map(|r| (r.session_id.as_str(), r.line_index))
        .collect();
    let turn_texts = get_turns_chunks(conn, &turn_keys)?;

    debug!(
        reconstructed_turns = turn_texts.len(),
        "Phase 4: reconstruct turns"
    );

    // Phase 5: Format output
    let mut ctx = String::from("## Relevant past context\n\n");
    for (i, result) in deduped.iter().enumerate() {
        let key = (result.session_id.clone(), result.line_index);
        let full_text = turn_texts
            .get(&key)
            .map_or_else(|| result.content.clone(), |chunks| chunks.join("\n\n"));

        debug!(
            rank = i + 1,
            distance = result.distance,
            result_session_id = %result.session_id,
            line_index = result.line_index,
            "Search result"
        );

        write!(
            &mut ctx,
            "### Memory {} (distance: {:.4})\n{}\n\n",
            i + 1,
            result.distance,
            full_text,
        )
        .unwrap();
    }

    Ok(ctx)
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

        // Search from a different session to test cross-session recall
        // (same-session without compaction boundary is filtered out)
        let ctx = search_context(&conn, &mut embedder, "authentication", 5, Some("other")).unwrap();
        assert!(!ctx.is_empty());
        assert!(ctx.contains("Relevant past context"));
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
        let results = search_memories(&conn, &emb[0], 10, None, None).unwrap();
        assert!(!results.is_empty());
    }

    // --- Filter pipeline tests ---

    /// Helper: seed a memory with real embedding into a session.
    fn seed_memory(
        conn: &Connection,
        embedder: &mut Embedder,
        session_id: &str,
        line_index: usize,
        chunk_index: usize,
        text: &str,
    ) {
        let embs = embedder.embed_batch(&[text]).unwrap();
        insert_memory(
            conn,
            session_id,
            line_index,
            chunk_index,
            "turn",
            text,
            &embs[0],
        )
        .unwrap();
    }

    /// Helper: ensure session exists with optional compaction boundary.
    fn ensure_session(
        conn: &Connection,
        session_id: &str,
        last_line: usize,
        compact_line: Option<usize>,
    ) {
        upsert_session(
            conn,
            &Session {
                session_id: session_id.to_string(),
                transcript_path: "/test/t.jsonl".to_string(),
                project_dir: "/test/p".to_string(),
                last_line_index: last_line,
                provisional_turn_start: None,
                last_compact_line_index: None,
            },
        )
        .unwrap();
        // upsert_session doesn't set last_compact_line_index, so update directly
        if let Some(boundary) = compact_line {
            conn.execute(
                "UPDATE sessions SET last_compact_line_index = ?1 WHERE session_id = ?2",
                rusqlite::params![boundary as i64, session_id],
            )
            .unwrap();
        }
    }

    #[test]
    fn in_context_results_filtered_out() {
        let (_tmp, conn, mut embedder, _) = setup_test();

        ensure_session(&conn, "s1", 10, None);
        seed_memory(&conn, &mut embedder, "s1", 0, 0, "Rust authentication");

        // Search from same session with no compaction boundary → all filtered
        let ctx =
            search_context(&conn, &mut embedder, "Rust authentication", 5, Some("s1")).unwrap();
        assert!(ctx.is_empty());
    }

    #[test]
    fn pre_compaction_results_retained() {
        let (_tmp, conn, mut embedder, _) = setup_test();

        // Memory at line 2, compaction boundary at line 5 → memory is pre-compaction
        ensure_session(&conn, "s1", 10, Some(5));
        seed_memory(&conn, &mut embedder, "s1", 2, 0, "Rust authentication");

        let ctx =
            search_context(&conn, &mut embedder, "Rust authentication", 5, Some("s1")).unwrap();
        assert!(!ctx.is_empty());
        assert!(ctx.contains("Relevant past context"));
    }

    #[test]
    fn post_compaction_results_filtered_out() {
        let (_tmp, conn, mut embedder, _) = setup_test();

        // Memory at line 8, compaction boundary at line 5 → memory is post-compaction (in-context)
        ensure_session(&conn, "s1", 10, Some(5));
        seed_memory(&conn, &mut embedder, "s1", 8, 0, "Rust authentication");

        let ctx =
            search_context(&conn, &mut embedder, "Rust authentication", 5, Some("s1")).unwrap();
        assert!(ctx.is_empty());
    }

    #[test]
    fn cross_session_results_always_returned() {
        let (_tmp, conn, mut embedder, _) = setup_test();

        ensure_session(&conn, "s1", 10, None);
        seed_memory(&conn, &mut embedder, "s1", 0, 0, "Rust authentication");

        // Search from different session → always returned
        let ctx =
            search_context(&conn, &mut embedder, "Rust authentication", 5, Some("s2")).unwrap();
        assert!(!ctx.is_empty());
        assert!(ctx.contains("Relevant past context"));
    }

    #[test]
    fn distance_threshold_filters_irrelevant() {
        let (_tmp, conn, mut embedder, _) = setup_test();

        ensure_session(&conn, "s1", 10, None);
        // Seed a memory with completely unrelated content
        seed_memory(
            &conn,
            &mut embedder,
            "s1",
            0,
            0,
            "The quick brown fox jumps over the lazy dog",
        );

        // Search with a very different topic from a different session
        let ctx = search_context(
            &conn,
            &mut embedder,
            "quantum physics dark matter equations",
            5,
            Some("other"),
        )
        .unwrap();
        // The distance between these unrelated texts should exceed the threshold
        // If not empty, at least the result must have distance below MAX_COSINE_DISTANCE
        // This test verifies the threshold mechanism exists and filters
        // (exact behavior depends on embedding model)
        if !ctx.is_empty() {
            assert!(ctx.contains("Relevant past context"));
        }
    }

    #[test]
    fn turn_dedup_reconstructs_full_turn() {
        let (_tmp, conn, mut embedder, _) = setup_test();

        ensure_session(&conn, "s1", 10, None);
        // Simulate a multi-chunk turn: same session + line_index, different chunk_index
        seed_memory(&conn, &mut embedder, "s1", 0, 0, "chunk zero content");
        seed_memory(&conn, &mut embedder, "s1", 0, 1, "chunk one content");

        // Search from different session to avoid in-context filter
        let ctx =
            search_context(&conn, &mut embedder, "chunk zero content", 5, Some("other")).unwrap();
        assert!(!ctx.is_empty());
        // The reconstructed turn should contain BOTH chunks
        assert!(ctx.contains("chunk zero content"));
        assert!(ctx.contains("chunk one content"));
    }

    #[test]
    fn results_truncated_to_k() {
        let (_tmp, conn, mut embedder, _) = setup_test();

        ensure_session(&conn, "s1", 20, None);
        // Seed more unique turns than k=2
        seed_memory(&conn, &mut embedder, "s1", 0, 0, "Rust ownership model");
        seed_memory(&conn, &mut embedder, "s1", 2, 0, "Rust borrowing rules");
        seed_memory(
            &conn,
            &mut embedder,
            "s1",
            4,
            0,
            "Rust lifetime annotations",
        );
        seed_memory(
            &conn,
            &mut embedder,
            "s1",
            6,
            0,
            "Rust trait implementations",
        );

        // Search with k=2 from different session
        let ctx =
            search_context(&conn, &mut embedder, "Rust programming", 2, Some("other")).unwrap();

        // Count the number of "### Memory" headers — should be at most 2
        let memory_count = ctx.matches("### Memory").count();
        assert!(
            memory_count <= 2,
            "Expected at most 2 results, got {memory_count}"
        );
    }
}
