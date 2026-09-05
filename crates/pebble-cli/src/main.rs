//! `pebble`: a coding agent at the command line.
//!
//! Bare `pebble` opens an interactive session in normal terminal scrollback.
//! `pebble exec` runs one non-interactive coding-agent session: one prompt in,
//! the tools the model asks for, the final answer out. Events are rendered to
//! standard error as they happen, the answer goes to standard output, and the
//! exit status says how the prompt ended.

use std::process::ExitCode;

mod application;
mod auth;
mod credentials;
mod exec;
mod interactive;
mod render;
mod secret_input;
mod settings;
mod storage;

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
    command: Option<Command>,

    #[command(flatten)]
    interactive: interactive::InteractiveArgs,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    /// Run one prompt to completion and print the answer.
    Exec(exec::ExecArgs),
    /// Manage provider credentials.
    Auth(auth::AuthArgs),
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Exec(args)) => exec::run(args).await,
        Some(Command::Auth(args)) => auth::run(args).await,
        None => interactive::run(cli.interactive).await,
    }
}
