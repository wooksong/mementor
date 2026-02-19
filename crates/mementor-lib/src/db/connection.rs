use std::path::Path;

use anyhow::Context;
use rusqlite::{Connection, OpenFlags, params};

use crate::config::EMBEDDING_DIMENSION;

use super::schema::apply_migrations;

#[allow(unsafe_code)]
unsafe extern "C" {
    fn sqlite3_vector_init(
        db: *mut rusqlite::ffi::sqlite3,
        pz_err_msg: *mut *mut std::ffi::c_char,
        p_api: *const rusqlite::ffi::sqlite3_api_routines,
    ) -> std::ffi::c_int;
}

/// Open a file-backed `SQLite` connection with sqlite-vector loaded and schema applied.
pub fn open_db(path: &Path) -> anyhow::Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    let mut conn = Connection::open(path)
        .with_context(|| format!("Failed to open database: {}", path.display()))?;
    init_connection(&mut conn)?;
    Ok(conn)
}

/// Open an in-memory `SQLite` connection with sqlite-vector loaded and schema applied.
///
/// Uses the `SQLite` URI format with shared-cache mode so multiple connections
/// can share the same named in-memory database within the same process.
/// The in-memory database persists as long as at least one connection to it remains open.
pub fn open_db_in_memory(name: &str) -> anyhow::Result<Connection> {
    let uri = format!("file:{name}?mode=memory&cache=shared");
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_URI
        | OpenFlags::SQLITE_OPEN_SHARED_CACHE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;

    let mut conn = Connection::open_with_flags(uri, flags)
        .with_context(|| format!("Failed to open in-memory database: {name}"))?;
    init_connection(&mut conn)?;
    Ok(conn)
}

/// Load sqlite-vector extension, apply schema migrations, register vector table,
/// and configure WAL mode for safe concurrent access from multiple worktrees.
fn init_connection(conn: &mut Connection) -> anyhow::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")
        .context("Failed to enable WAL mode")?;
    conn.pragma_update(None, "busy_timeout", 5000)
        .context("Failed to set busy_timeout")?;
    register_vector_extension(conn)?;
    apply_migrations(conn)?;
    register_vector_table(conn)?;
    Ok(())
}

/// Register the sqlite-vector extension into the connection.
#[allow(unsafe_code)]
fn register_vector_extension(conn: &Connection) -> anyhow::Result<()> {
    let rc = unsafe {
        let db_ptr = conn.handle();
        sqlite3_vector_init(db_ptr, std::ptr::null_mut(), std::ptr::null())
    };

    if rc != rusqlite::ffi::SQLITE_OK {
        anyhow::bail!("sqlite3_vector_init failed with code {rc}");
    }

    Ok(())
}

/// Register the `memories.embedding` column with sqlite-vector for search.
/// Must be called after migrations have created the `memories` table.
fn register_vector_table(conn: &Connection) -> anyhow::Result<()> {
    let options = format!("type=f32, dimension={EMBEDDING_DIMENSION}, distance=cosine");
    conn.query_row(
        "SELECT vector_init('memories', 'embedding', ?1)",
        params![options],
        |_row| Ok(()),
    )
    .context("Failed to call vector_init for memories table")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_db_creates_file() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let conn = open_db(&db_path).unwrap();
        assert!(db_path.exists());
        drop(conn);
    }

    #[test]
    fn vector_extension_loads() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let conn = open_db(&db_path).unwrap();

        // Verify vector_version() is available
        let version: String = conn
            .query_row("SELECT vector_version()", [], |row| row.get(0))
            .unwrap();
        assert!(!version.is_empty());
    }

    #[test]
    fn open_in_memory_works() {
        let conn = open_db_in_memory("test_open").unwrap();
        let version: String = conn
            .query_row("SELECT vector_version()", [], |row| row.get(0))
            .unwrap();
        assert!(!version.is_empty());
    }

    #[test]
    fn in_memory_shared_cache_persists() {
        // First connection creates data
        let conn1 = open_db_in_memory("test_shared").unwrap();
        conn1
            .execute(
                "INSERT INTO sessions (session_id, transcript_path, project_dir, last_line_index)
                 VALUES (?1, ?2, ?3, ?4)",
                params!["sess-1", "/tmp/t.jsonl", "/tmp/p", 42],
            )
            .unwrap();

        // Second connection sees the same data
        let conn2 = open_db_in_memory("test_shared").unwrap();
        let val: i64 = conn2
            .query_row(
                "SELECT last_line_index FROM sessions WHERE session_id = ?1",
                params!["sess-1"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn wal_mode_is_active() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test_wal.db");
        let conn = open_db(&db_path).unwrap();

        let mode: String = conn
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
    }

    #[test]
    fn busy_timeout_is_set() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test_timeout.db");
        let conn = open_db(&db_path).unwrap();

        let timeout: i64 = conn
            .pragma_query_value(None, "busy_timeout", |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    #[test]
    fn vector_backend_available() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let conn = open_db(&db_path).unwrap();

        // Check which SIMD backend is active
        let backend: String = conn
            .query_row("SELECT vector_backend()", [], |row| row.get(0))
            .unwrap();
        // On x86_64: "SSE2" (AVX2/512 stubs fallback to SSE2); on aarch64: "NEON"
        let backend_lower = backend.to_lowercase();
        assert!(
            backend_lower.contains("sse")
                || backend_lower.contains("avx")
                || backend_lower.contains("neon"),
            "Unexpected backend: {backend}"
        );
    }
}
