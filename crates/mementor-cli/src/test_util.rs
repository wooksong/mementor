use mementor_lib::context::RealMementorContext;
use mementor_lib::output::BufferedOutput;

/// Run a CLI test with a temporary project directory and buffered output.
pub fn test_with_context<F>(f: F)
where
    F: FnOnce(&RealMementorContext, &mut BufferedOutput),
{
    let tmp = tempfile::tempdir().unwrap();
    let context = RealMementorContext::new(tmp.path().to_path_buf());
    let mut output = BufferedOutput::new();
    f(&context, &mut output);
}
