//! Provider credential commands, using the same store as interactive sessions.

use std::fmt::Write as _;
use std::io::{self, Write as _};
use std::process::ExitCode;

use anyhow::{Context as _, Result, ensure};
use clap::{Args, Subcommand};
use lithos_llm::middleware::RetryPolicy;

use crate::application::Application;
use crate::credentials::accepts_api_key;
use crate::secret_input;
use crate::terminal::print_err;

#[derive(Debug, Args)]
pub struct AuthArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Save an API key for a provider. Subscription login is not supported yet.
    Login {
        provider: String,
        /// Read the key from standard input instead of masked terminal input.
        #[arg(long)]
        stdin:    bool,
    },
    /// Remove a provider's saved credential or explicit credential source.
    Logout { provider: String },
    /// Show credential sources without secret values. Does not contact
    /// providers.
    Status { provider: Option<String> },
}

pub async fn run(args: AuthArgs) -> ExitCode {
    match execute(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            print_err(&format!("error: {error:#}"));
            ExitCode::FAILURE
        }
    }
}

async fn execute(args: AuthArgs) -> Result<()> {
    let app = Application::load(RetryPolicy::exponential().max_attempts(4)).await?;
    match args.command {
        Command::Login { provider, stdin } => {
            let provider = app
                .client
                .catalog()
                .provider(&provider)
                .context("unknown provider; use `pebble auth status` to list providers")?;
            ensure!(
                accepts_api_key(provider),
                "this provider needs explicit credential headers or a different authentication scheme; see docs/models-and-credentials.md"
            );
            let key = if stdin {
                secret_input::from_stdin().await?
            } else {
                print_err(&format!(
                    "Save an API key for {}. Esc cancels.",
                    provider.display_name()
                ));
                secret_input::prompt().await?
            };
            app.auth.save_key(provider, key).await?;
            print_err(&format!(
                "Saved credentials for {} in {}.",
                provider.id(),
                app.auth.path().display()
            ));
            print_err(&format!(
                "Active source: {}",
                app.auth.resolve(provider).await?.source
            ));
        }
        Command::Logout { provider } => {
            // Exact saved ids can also be removed after a custom provider was
            // deleted from models.toml.
            let configured = app.client.catalog().provider(&provider).ok();
            let id = configured.map_or(provider.as_str(), |provider| provider.id().as_str());
            let removed = app.auth.remove(id).await?;
            print_err(if removed {
                "Removed saved credentials."
            } else {
                "No saved credentials to remove."
            });
            if let Some(provider) = configured
                && let Ok(resolved) = app.auth.resolve(provider).await
            {
                print_err(&format!("Still configured through {}.", resolved.source));
            }
        }
        Command::Status { provider } => {
            let providers: Vec<_> = if let Some(provider) = provider {
                vec![
                    app.client
                        .catalog()
                        .provider(&provider)
                        .context("unknown provider")?,
                ]
            } else {
                app.client.catalog().providers().collect()
            };
            let snapshot = app.auth.snapshot().await?;
            let mut output = format!("Credentials: {}\n", app.auth.path().display());
            for provider in providers {
                let status = match snapshot.resolve(provider).await {
                    Ok(resolved) => resolved.source,
                    Err(error) => error.to_string(),
                };
                let adapter = if app.client.available_providers().contains(provider.id()) {
                    ""
                } else {
                    " · adapter unavailable"
                };
                writeln!(output, "{}: {status}{adapter}", provider.id())?;
            }
            output.push_str("Configured sources do not verify account access.\n");
            io::stdout()
                .lock()
                .write_all(output.as_bytes())
                .context("writing credential status")?;
        }
    }
    Ok(())
}
