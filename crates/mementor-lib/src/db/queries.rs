#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;
use std::fmt::Write as _;

use anyhow::Context;
use rusqlite::{Connection, params};
use tracing::debug;

/// Session data stored in the `sessions` table.
#[derive(Debug)]
pub struct Session {
    pub session_id: String,
    pub transcript_path: String,
    pub project_dir: String,
    pub last_line_index: usize,
    pub provisional_turn_start: Option<usize>,
    pub last_compact_line_index: Option<usize>,
}

/// Insert or update a session record.
pub fn upsert_session(conn: &Connection, session: &Session) -> anyhow::Result<()> {
    debug!(
        session_id = %session.session_id,
        last_line_index = session.last_line_index,
        provisional_turn_start = ?session.provisional_turn_start,
        last_compact_line_index = ?session.last_compact_line_index,
        "Upserting session"
    );
    conn.execute(
        "INSERT INTO sessions (session_id, transcript_path, project_dir, last_line_index, provisional_turn_start, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))
         ON CONFLICT(session_id) DO UPDATE SET
           transcript_path = excluded.transcript_path,
           last_line_index = excluded.last_line_index,
           provisional_turn_start = excluded.provisional_turn_start,
           updated_at = datetime('now')",
        params![
            session.session_id,
            session.transcript_path,
            session.project_dir,
            session.last_line_index as i64,
            session.provisional_turn_start.map(|v| v as i64),
        ],
    )
    .context("Failed to upsert session")?;
    Ok(())
}

/// Get a session by ID. Returns `None` if not found.
pub fn get_session(conn: &Connection, session_id: &str) -> anyhow::Result<Option<Session>> {
    let mut stmt = conn
        .prepare(
            "SELECT session_id, transcript_path, project_dir, last_line_index,
                    provisional_turn_start, last_compact_line_index
             FROM sessions WHERE session_id = ?1",
        )
        .context("Failed to prepare get_session query")?;

    let result = stmt
        .query_row(params![session_id], |row| {
            Ok(Session {
                session_id: row.get(0)?,
                transcript_path: row.get(1)?,
                project_dir: row.get(2)?,
                last_line_index: row.get::<_, i64>(3)? as usize,
                provisional_turn_start: row.get::<_, Option<i64>>(4)?.map(|v| v as usize),
                last_compact_line_index: row.get::<_, Option<i64>>(5)?.map(|v| v as usize),
            })
        })
        .optional()
        .context("Failed to query session")?;

    Ok(result)
}

/// Insert a memory chunk with its embedding vector.
pub fn insert_memory(
    conn: &Connection,
    session_id: &str,
    line_index: usize,
    chunk_index: usize,
    role: &str,
    content: &str,
    embedding: &[f32],
) -> anyhow::Result<()> {
    debug!(
        session_id = %session_id,
        line_index = line_index,
        chunk_index = chunk_index,
        content_len = content.len(),
        content = %content,
        "Inserting memory chunk"
    );
    let embedding_json = serde_json::to_string(embedding)?;

    conn.execute(
        "INSERT OR REPLACE INTO memories (session_id, line_index, chunk_index, role, content, embedding)
         VALUES (?1, ?2, ?3, ?4, ?5, vector_as_f32(?6))",
        params![
            session_id,
            line_index as i64,
            chunk_index as i64,
            role,
            content,
            embedding_json,
        ],
    )
    .context("Failed to insert memory")?;
    Ok(())
}

/// Delete all memories for a session at a specific line index.
/// Used when re-processing a provisional turn.
pub fn delete_memories_at(
    conn: &Connection,
    session_id: &str,
    line_index: usize,
) -> anyhow::Result<usize> {
    let deleted = conn
        .execute(
            "DELETE FROM memories WHERE session_id = ?1 AND line_index = ?2",
            params![session_id, line_index as i64],
        )
        .context("Failed to delete memories")?;
    Ok(deleted)
}

/// A search result from vector similarity search.
#[derive(Debug)]
pub struct MemorySearchResult {
    pub session_id: String,
    pub line_index: usize,
    pub chunk_index: usize,
    pub role: String,
    pub content: String,
    pub distance: f64,
}

/// Search for the top-k most similar memories using cosine distance.
///
/// Uses the `vector_full_scan` virtual table provided by sqlite-vector.
/// When `exclude_session_id` is provided, results from that session are
/// filtered out unless they fall at or before the `compact_boundary`.
pub fn search_memories(
    conn: &Connection,
    query_embedding: &[f32],
    k: usize,
    exclude_session_id: Option<&str>,
    compact_boundary: Option<usize>,
) -> anyhow::Result<Vec<MemorySearchResult>> {
    let query_json = serde_json::to_string(query_embedding)?;

    let mut stmt = conn.prepare(
        "SELECT m.session_id, m.line_index, m.chunk_index, m.role, m.content, vs.distance
         FROM vector_full_scan('memories', 'embedding', ?1, ?2) vs
         JOIN memories m ON m.rowid = vs.id
         WHERE ?3 IS NULL
            OR m.session_id != ?3
            OR (?4 IS NOT NULL AND m.line_index <= ?4)
         ORDER BY vs.distance ASC",
    )?;

    let results = stmt
        .query_map(
            params![
                query_json,
                k as i64,
                exclude_session_id,
                compact_boundary.map(|v| v as i64),
            ],
            |row| {
                Ok(MemorySearchResult {
                    session_id: row.get(0)?,
                    line_index: row.get::<_, i64>(1)? as usize,
                    chunk_index: row.get::<_, i64>(2)? as usize,
                    role: row.get(3)?,
                    content: row.get(4)?,
                    distance: row.get(5)?,
                })
            },
        )?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to search memories")?;

    debug!(
        k = k,
        exclude_session_id = ?exclude_session_id,
        compact_boundary = ?compact_boundary,
        result_count = results.len(),
        "Vector search completed"
    );

    Ok(results)
}

/// Batch-retrieve all chunks for multiple turns in a single query.
///
/// Returns a map from `(session_id, line_index)` to the ordered list of chunk
/// contents for that turn. Chunks are ordered by `chunk_index` ascending.
pub fn get_turns_chunks(
    conn: &Connection,
    turn_keys: &[(&str, usize)],
) -> anyhow::Result<HashMap<(String, usize), Vec<String>>> {
    if turn_keys.is_empty() {
        return Ok(HashMap::new());
    }

    // Build dynamic WHERE clause with OR conditions
    let mut sql = String::from(
        "SELECT session_id, line_index, content
         FROM memories WHERE ",
    );
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for (i, (sid, line_idx)) in turn_keys.iter().enumerate() {
        if i > 0 {
            sql.push_str(" OR ");
        }
        let p1 = i * 2 + 1;
        let p2 = i * 2 + 2;
        write!(&mut sql, "(session_id = ?{p1} AND line_index = ?{p2})").unwrap();
        param_values.push(Box::new((*sid).to_string()));
        param_values.push(Box::new(*line_idx as i64));
    }
    sql.push_str(" ORDER BY session_id, line_index, chunk_index");

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| &**p).collect();
    let mut stmt = conn
        .prepare(&sql)
        .context("Failed to prepare get_turns_chunks query")?;
    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)? as usize,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to query turn chunks")?;

    let mut result: HashMap<(String, usize), Vec<String>> = HashMap::new();
    for (sid, line_idx, raw) in rows {
        result.entry((sid, line_idx)).or_default().push(raw);
    }

    debug!(
        turn_count = turn_keys.len(),
        total_chunks = result.values().map(Vec::len).sum::<usize>(),
        "Batch-retrieved turn chunks"
    );

    Ok(result)
}

/// Update the compaction boundary for a session.
///
/// Sets `last_compact_line_index` to the current `last_line_index`,
/// marking all memories up to that point as pre-compaction.
pub fn update_compact_line(conn: &Connection, session_id: &str) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE sessions SET last_compact_line_index = last_line_index, updated_at = datetime('now')
         WHERE session_id = ?1",
        params![session_id],
    )
    .context("Failed to update compact line")?;
    debug!(session_id = %session_id, "Updated compaction boundary");
    Ok(())
}

/// Trait extension for `rusqlite::OptionalExtension`.
trait OptionalExt<T> {
    fn optional(self) -> rusqlite::Result<Option<T>>;
}

impl<T> OptionalExt<T> for rusqlite::Result<T> {
    fn optional(self) -> rusqlite::Result<Option<T>> {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::connection::open_db;
    use super::*;

    fn test_db() -> (tempfile::TempDir, Connection) {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let conn = open_db(&db_path).unwrap();
        (tmp, conn)
    }

    #[test]
    fn upsert_and_get_session() {
        let (_tmp, conn) = test_db();

        let session = Session {
            session_id: "test-session".to_string(),
            transcript_path: "/tmp/transcript.jsonl".to_string(),
            project_dir: "/tmp/project".to_string(),
            last_line_index: 0,
            provisional_turn_start: None,
            last_compact_line_index: None,
        };
        upsert_session(&conn, &session).unwrap();

        let result = get_session(&conn, "test-session").unwrap().unwrap();
        assert_eq!(result.session_id, "test-session");
        assert_eq!(result.last_line_index, 0);
        assert!(result.provisional_turn_start.is_none());
        assert!(result.last_compact_line_index.is_none());
    }

    #[test]
    fn upsert_session_updates_existing() {
        let (_tmp, conn) = test_db();

        let session = Session {
            session_id: "s1".to_string(),
            transcript_path: "/tmp/t.jsonl".to_string(),
            project_dir: "/tmp/p".to_string(),
            last_line_index: 0,
            provisional_turn_start: None,
            last_compact_line_index: None,
        };
        upsert_session(&conn, &session).unwrap();

        let updated = Session {
            last_line_index: 10,
            provisional_turn_start: Some(8),
            ..session
        };
        upsert_session(&conn, &updated).unwrap();

        let result = get_session(&conn, "s1").unwrap().unwrap();
        assert_eq!(result.last_line_index, 10);
        assert_eq!(result.provisional_turn_start, Some(8));
    }

    #[test]
    fn get_nonexistent_session() {
        let (_tmp, conn) = test_db();
        let result = get_session(&conn, "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn insert_and_search_memories() {
        let (_tmp, conn) = test_db();

        let session = Session {
            session_id: "s1".to_string(),
            transcript_path: "/tmp/t.jsonl".to_string(),
            project_dir: "/tmp/p".to_string(),
            last_line_index: 2,
            provisional_turn_start: None,
            last_compact_line_index: None,
        };
        upsert_session(&conn, &session).unwrap();

        // Insert two memories with different embeddings
        let emb1 = vec![1.0_f32; 384];
        let emb2 = vec![0.5_f32; 384];
        insert_memory(&conn, "s1", 0, 0, "user", "Hello world", &emb1).unwrap();
        insert_memory(&conn, "s1", 0, 1, "assistant", "Hi there", &emb2).unwrap();

        // Search should return results ordered by distance
        let query = vec![1.0_f32; 384]; // Same as emb1
        let results = search_memories(&conn, &query, 5, None, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "Hello world"); // Closest match
    }

    #[test]
    fn delete_memories_at_line_index() {
        let (_tmp, conn) = test_db();

        let session = Session {
            session_id: "s1".to_string(),
            transcript_path: "/tmp/t.jsonl".to_string(),
            project_dir: "/tmp/p".to_string(),
            last_line_index: 4,
            provisional_turn_start: None,
            last_compact_line_index: None,
        };
        upsert_session(&conn, &session).unwrap();

        let emb = vec![1.0_f32; 384];
        insert_memory(&conn, "s1", 0, 0, "user", "chunk 0-0", &emb).unwrap();
        insert_memory(&conn, "s1", 0, 1, "user", "chunk 0-1", &emb).unwrap();
        insert_memory(&conn, "s1", 2, 0, "user", "chunk 2-0", &emb).unwrap();

        let deleted = delete_memories_at(&conn, "s1", 0).unwrap();
        assert_eq!(deleted, 2);

        // Only chunk at line_index=2 should remain
        let results = search_memories(&conn, &emb, 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].line_index, 2);
    }

    #[test]
    fn get_turns_chunks_returns_grouped_results() {
        let (_tmp, conn) = test_db();

        let session = Session {
            session_id: "s1".to_string(),
            transcript_path: "/tmp/t.jsonl".to_string(),
            project_dir: "/tmp/p".to_string(),
            last_line_index: 10,
            provisional_turn_start: None,
            last_compact_line_index: None,
        };
        upsert_session(&conn, &session).unwrap();

        let emb = vec![1.0_f32; 384];
        // Turn at line 0: 2 chunks
        insert_memory(&conn, "s1", 0, 0, "turn", "chunk-0-0", &emb).unwrap();
        insert_memory(&conn, "s1", 0, 1, "turn", "chunk-0-1", &emb).unwrap();
        // Turn at line 4: 3 chunks
        insert_memory(&conn, "s1", 4, 0, "turn", "chunk-4-0", &emb).unwrap();
        insert_memory(&conn, "s1", 4, 1, "turn", "chunk-4-1", &emb).unwrap();
        insert_memory(&conn, "s1", 4, 2, "turn", "chunk-4-2", &emb).unwrap();
        // Turn at line 8: 1 chunk (not queried)
        insert_memory(&conn, "s1", 8, 0, "turn", "chunk-8-0", &emb).unwrap();

        let keys = vec![("s1", 0), ("s1", 4)];
        let result = get_turns_chunks(&conn, &keys).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(
            result[&("s1".to_string(), 0)],
            vec!["chunk-0-0", "chunk-0-1"]
        );
        assert_eq!(
            result[&("s1".to_string(), 4)],
            vec!["chunk-4-0", "chunk-4-1", "chunk-4-2"]
        );
        // line 8 should NOT be in results
        assert!(!result.contains_key(&("s1".to_string(), 8)));
    }

    #[test]
    fn get_turns_chunks_empty_keys() {
        let (_tmp, conn) = test_db();
        let result = get_turns_chunks(&conn, &[]).unwrap();
        assert!(result.is_empty());
    }
}
