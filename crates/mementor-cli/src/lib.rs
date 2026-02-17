pub mod cli;
pub mod commands;
pub mod hooks;

#[cfg(test)]
pub mod test_util;

use std::io::{Read, Write};

use clap::Parser;

use mementor_lib::context::MementorContext;
use mementor_lib::output::ConsoleOutput;

use cli::{Cli, Command, HookCommand};

/// Main CLI entry point. Parses args and dispatches to the appropriate command.
pub fn try_run<C, OUT, ERR>(
    args: &[&str],
    context: &C,
    output: &mut dyn ConsoleOutput<OUT, ERR>,
    stdin: &mut dyn Read,
) -> anyhow::Result<()>
where
    C: MementorContext,
    OUT: Write,
    ERR: Write,
{
    let cli = Cli::try_parse_from(args)?;

    match cli.command {
        Command::Enable => commands::enable::run_enable(context, output),
        Command::Ingest {
            transcript,
            session_id,
        } => commands::ingest::run_ingest_cmd(&transcript, &session_id, context, output),
        Command::Query { text, k } => commands::query::run_query(&text, k, context, output),
        Command::Hook { hook_command } => match hook_command {
            HookCommand::Stop => {
                let input = hooks::input::read_stop_input(stdin)?;
                hooks::stop::handle_stop(&input, context, output)
            }
            HookCommand::UserPromptSubmit => {
                let input = hooks::input::read_prompt_input(stdin)?;
                hooks::prompt::handle_prompt(&input, context, output)
            }
        },
    }
}
