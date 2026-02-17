use std::path::PathBuf;

use crate::context::RealMementorContext;
use crate::output::BufferedOutput;

/// Run a test with a temporary project directory and buffered output.
pub fn test_with_context<F>(f: F)
where
    F: FnOnce(&RealMementorContext, &mut BufferedOutput),
{
    let tmp = tempfile::tempdir().unwrap();
    let context = RealMementorContext::new(tmp.path().to_path_buf());
    let mut output = BufferedOutput::new();
    f(&context, &mut output);
}

/// Run a test and return the temporary directory alongside results
/// (useful when the test needs the temp dir to persist for assertions).
pub fn test_with_context_and_dir<F>(f: F) -> (tempfile::TempDir, String, String)
where
    F: FnOnce(&RealMementorContext, &mut BufferedOutput),
{
    let tmp = tempfile::tempdir().unwrap();
    let context = RealMementorContext::new(tmp.path().to_path_buf());
    let mut output = BufferedOutput::new();
    f(&context, &mut output);
    let stdout = output.stdout_to_string();
    let stderr = output.stderr_to_string();
    (tmp, stdout, stderr)
}

/// Create a `RealMementorContext` from an explicit path.
pub fn context_at(path: PathBuf) -> RealMementorContext {
    RealMementorContext::new(path)
}
