use std::path::PathBuf;

use rusqlite::Connection;

use super::connection::{open_db, open_db_in_memory};

/// Connection factory for `SQLite` databases.
///
/// `File` opens a file-backed database at the given path.
/// `InMemory` uses a named shared-cache in-memory database; the anchor
/// connection keeps the database alive for the lifetime of the driver.
pub enum DatabaseDriver {
    /// File-backed `SQLite` database.
    File(PathBuf),
    /// Named in-memory `SQLite` database with shared cache.
    /// The `_anchor` connection keeps the database alive.
    InMemory { name: String, _anchor: Connection },
}

impl DatabaseDriver {
    /// Create a driver for a file-backed database.
    #[must_use]
    pub fn file(path: PathBuf) -> Self {
        Self::File(path)
    }

    /// Create a driver for a named in-memory database.
    ///
    /// Opens an anchor connection that keeps the shared-cache database alive.
    /// Subsequent calls to [`open`](Self::open) return new connections to the
    /// same in-memory database.
    pub fn in_memory(name: &str) -> anyhow::Result<Self> {
        let anchor = open_db_in_memory(name)?;
        Ok(Self::InMemory {
            name: name.to_string(),
            _anchor: anchor,
        })
    }

    /// Open a new connection to the underlying database.
    pub fn open(&self) -> anyhow::Result<Connection> {
        match self {
            Self::File(path) => open_db(path),
            Self::InMemory { name, .. } => open_db_in_memory(name),
        }
    }

    /// Check whether the database is ready to use.
    ///
    /// For file-backed databases, this checks if the database file exists.
    /// For in-memory databases, this always returns `true` (the anchor
    /// connection guarantees the database is alive).
    #[must_use]
    pub fn is_ready(&self) -> bool {
        match self {
            Self::File(path) => path.exists(),
            Self::InMemory { .. } => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_driver_is_not_ready_before_open() {
        let driver = DatabaseDriver::file(PathBuf::from("/tmp/nonexistent/test.db"));
        assert!(!driver.is_ready());
    }

    #[test]
    fn file_driver_is_ready_after_open() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let driver = DatabaseDriver::file(db_path.clone());
        let _conn = driver.open().unwrap();
        assert!(driver.is_ready());
    }

    #[test]
    fn in_memory_driver_is_always_ready() {
        let driver = DatabaseDriver::in_memory("driver_ready").unwrap();
        assert!(driver.is_ready());
    }

    #[test]
    fn in_memory_driver_shares_data() {
        let driver = DatabaseDriver::in_memory("driver_share").unwrap();

        // Write via one connection
        let conn1 = driver.open().unwrap();
        conn1
            .execute(
                "INSERT INTO sessions (session_id, transcript_path, project_dir, last_line_index)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params!["s1", "/t.jsonl", "/p", 10],
            )
            .unwrap();
        drop(conn1);

        // Read via another connection
        let conn2 = driver.open().unwrap();
        let val: i64 = conn2
            .query_row(
                "SELECT last_line_index FROM sessions WHERE session_id = ?1",
                rusqlite::params!["s1"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(val, 10);
    }
}
