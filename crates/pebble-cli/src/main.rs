//! `pebble`: a coding agent at the command line.
//!
//! `pebble exec` runs one non-interactive coding-agent session: one prompt in,
//! the tools the model asks for, the final answer out. Events are rendered to
//! standard error as they happen, the answer goes to standard output, and the
//! exit status says how the prompt ended.

use std::process::ExitCode;

mod exec;
mod render;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "pebble",
    version,
    about = "Run a coding agent from the command line",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    /// Run one prompt to completion and print the answer.
    Exec(exec::ExecArgs),
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Exec(args) => exec::run(args).await,
    }
}
