use mementor_lib::context::MementorContext;
use mementor_lib::db::driver::DatabaseDriver;
use mementor_lib::output::BufferedIO;
use mementor_lib::runtime::Runtime;

/// Run a CLI test with a temporary project directory, file-backed DB, and buffered I/O.
pub fn test_with_runtime<F>(f: F)
where
    F: FnOnce(&Runtime, &mut BufferedIO),
{
    let tmp = tempfile::tempdir().unwrap();
    let context = MementorContext::new(tmp.path().to_path_buf());
    let db = DatabaseDriver::file(context.db_path());
    let runtime = Runtime { context, db };
    let mut io = BufferedIO::new();
    f(&runtime, &mut io);
}
