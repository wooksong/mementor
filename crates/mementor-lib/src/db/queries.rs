#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use anyhow::Context;
use rusqlite::{Connection, params};

/// Session data stored in the `sessions` table.
#[derive(Debug)]
pub struct Session {
    pub session_id: String,
    pub transcript_path: String,
    pub project_dir: String,
    pub last_line_index: usize,
    pub provisional_turn_start: Option<usize>,
}

/// Insert or update a session record.
pub fn upsert_session(conn: &Connection, session: &Session) -> anyhow::Result<()> {
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
            "SELECT session_id, transcript_path, project_dir, last_line_index, provisional_turn_start
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
/// Uses the `vector_full_scan` virtual table provided by sqlite-vector.
pub fn search_memories(
    conn: &Connection,
    query_embedding: &[f32],
    k: usize,
) -> anyhow::Result<Vec<MemorySearchResult>> {
    let query_json = serde_json::to_string(query_embedding)?;

    let mut stmt = conn.prepare(
        "SELECT m.session_id, m.line_index, m.chunk_index, m.role, m.content, vs.distance
         FROM vector_full_scan('memories', 'embedding', ?1, ?2) vs
         JOIN memories m ON m.rowid = vs.id
         ORDER BY vs.distance ASC",
    )?;

    let results = stmt
        .query_map(params![query_json, k as i64], |row| {
            Ok(MemorySearchResult {
                session_id: row.get(0)?,
                line_index: row.get::<_, i64>(1)? as usize,
                chunk_index: row.get::<_, i64>(2)? as usize,
                role: row.get(3)?,
                content: row.get(4)?,
                distance: row.get(5)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to search memories")?;

    Ok(results)
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
        };
        upsert_session(&conn, &session).unwrap();

        let result = get_session(&conn, "test-session").unwrap().unwrap();
        assert_eq!(result.session_id, "test-session");
        assert_eq!(result.last_line_index, 0);
        assert!(result.provisional_turn_start.is_none());
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
        };
        upsert_session(&conn, &session).unwrap();

        // Insert two memories with different embeddings
        let emb1 = vec![1.0_f32; 384];
        let emb2 = vec![0.5_f32; 384];
        insert_memory(&conn, "s1", 0, 0, "user", "Hello world", &emb1).unwrap();
        insert_memory(&conn, "s1", 0, 1, "assistant", "Hi there", &emb2).unwrap();

        // Search should return results ordered by distance
        let query = vec![1.0_f32; 384]; // Same as emb1
        let results = search_memories(&conn, &query, 5).unwrap();
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
        };
        upsert_session(&conn, &session).unwrap();

        let emb = vec![1.0_f32; 384];
        insert_memory(&conn, "s1", 0, 0, "user", "chunk 0-0", &emb).unwrap();
        insert_memory(&conn, "s1", 0, 1, "user", "chunk 0-1", &emb).unwrap();
        insert_memory(&conn, "s1", 2, 0, "user", "chunk 2-0", &emb).unwrap();

        let deleted = delete_memories_at(&conn, "s1", 0).unwrap();
        assert_eq!(deleted, 2);

        // Only chunk at line_index=2 should remain
        let results = search_memories(&conn, &emb, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].line_index, 2);
    }
}
