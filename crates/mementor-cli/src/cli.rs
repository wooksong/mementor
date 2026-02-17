use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "mementor", about = "Local RAG memory agent for Claude Code")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Set up mementor for the current project (create DB, configure hooks).
    Enable,

    /// Manually ingest a transcript file into the memory database.
    Ingest {
        /// Path to the JSONL transcript file.
        transcript: String,
        /// Session ID for this transcript.
        session_id: String,
    },

    /// Search stored memories by semantic similarity.
    Query {
        /// The text to search for.
        text: String,
        /// Number of results to return.
        #[arg(short, long, default_value_t = 5)]
        k: usize,
    },

    /// Hook subcommands (called by Claude Code lifecycle hooks).
    Hook {
        #[command(subcommand)]
        hook_command: HookCommand,
    },
}

#[derive(Subcommand, Debug)]
pub enum HookCommand {
    /// Stop hook handler: reads stdin JSON and runs incremental ingestion.
    Stop,
    /// `UserPromptSubmit` hook handler: reads stdin, performs RAG search,
    /// outputs context to stdout.
    #[command(name = "user-prompt-submit")]
    UserPromptSubmit,
}
