use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

/// Define all schema migrations.
fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(
        "CREATE TABLE sessions (
            session_id                TEXT PRIMARY KEY,
            transcript_path           TEXT NOT NULL,
            project_dir               TEXT NOT NULL,
            last_line_index           INTEGER NOT NULL DEFAULT 0,
            provisional_turn_start    INTEGER,
            created_at                TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at                TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE memories (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT NOT NULL REFERENCES sessions(session_id),
            line_index       INTEGER NOT NULL,
            chunk_index      INTEGER NOT NULL,
            role             TEXT NOT NULL,
            content          TEXT NOT NULL,
            embedding        BLOB,
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(session_id, line_index, chunk_index)
        );

        CREATE INDEX idx_memories_session
            ON memories(session_id, line_index);",
    )])
}

/// Apply all pending migrations to the database.
pub fn apply_migrations(conn: &mut Connection) -> anyhow::Result<()> {
    migrations()
        .to_latest(conn)
        .map_err(|e| anyhow::anyhow!("Failed to apply migrations: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_are_valid() {
        migrations().validate().unwrap();
    }

    #[test]
    fn apply_migrations_creates_tables() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_migrations(&mut conn).unwrap();

        // Verify sessions table exists
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='sessions'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Verify memories table exists
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memories'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn apply_migrations_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_migrations(&mut conn).unwrap();
        apply_migrations(&mut conn).unwrap(); // Should not fail
    }
}
