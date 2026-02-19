use std::path::{Path, PathBuf};

use mementor_lib::context::MementorContext;
use mementor_lib::db::driver::DatabaseDriver;
use mementor_lib::git::resolve_worktree;
use mementor_lib::output::StdIO;
use mementor_lib::runtime::Runtime;

fn main() -> anyhow::Result<()> {
    // 1. Read env + classify worktree type + resolve project root
    let cwd = std::env::current_dir()?;
    let resolved = resolve_worktree(&cwd);
    let is_linked = resolved.is_linked();
    let project_root = resolved
        .primary_root()
        .map_or_else(|| cwd.clone(), Path::to_path_buf);
    let log_dir = std::env::var("MEMENTOR_LOG_DIR").ok().map(PathBuf::from);
    let context = MementorContext::with_cwd_and_log_dir(cwd, project_root, is_linked, log_dir);

    // 2. Init file logging (no-op if log_dir is None)
    mementor_cli::logging::init_file_logging(&context);

    // 3. Set panic hook (logs to file if available, always prints to stderr)
    std::panic::set_hook(Box::new(|info| {
        tracing::error!("{info}");
        eprintln!("{info}");
    }));

    // 4. Build runtime
    let db = DatabaseDriver::file(context.db_path());
    let runtime = Runtime { context, db };

    // 5. Run CLI
    let args: Vec<String> = std::env::args().collect();
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let mut io = StdIO::new();

    let result = mementor_cli::try_run(&args_refs, &runtime, &mut io);
    if let Err(ref e) = result {
        tracing::error!(error = format!("{e:?}"), "command failed");
    }
    result
}
