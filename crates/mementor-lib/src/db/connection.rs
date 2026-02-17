use std::path::Path;

use anyhow::Context;
use rusqlite::{Connection, params};

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

/// Open a `SQLite` connection with sqlite-vector loaded and schema applied.
pub fn open_db(path: &Path) -> anyhow::Result<Connection> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    let mut conn = Connection::open(path)
        .with_context(|| format!("Failed to open database: {}", path.display()))?;

    register_vector_extension(&conn)?;
    apply_migrations(&mut conn)?;
    register_vector_table(&conn)?;

    Ok(conn)
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
