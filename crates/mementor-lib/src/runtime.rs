use crate::context::MementorContext;
use crate::db::driver::DatabaseDriver;

/// Immutable dependency bundle for mementor commands.
///
/// Bundles the project context and database driver so that all command
/// handlers receive a single `&Runtime` parameter instead of individual
/// dependencies.
pub struct Runtime {
    pub context: MementorContext,
    pub db: DatabaseDriver,
}
