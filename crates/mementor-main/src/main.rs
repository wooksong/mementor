use mementor_lib::context::RealMementorContext;
use mementor_lib::output::StdOutput;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();

    let context = RealMementorContext::from_cwd()?;
    let mut output = StdOutput::new();
    let mut stdin = std::io::stdin();

    mementor_cli::try_run(&args_refs, &context, &mut output, &mut stdin)
}
