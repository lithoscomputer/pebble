//! The `exec` command: one prompt, run to completion.

use std::io::{self, IsTerminal as _, Read as _, Write as _};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use std::{env, fmt};

use anyhow::{Context as _, Result, bail};
use clap::{Args, ValueEnum};
use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::client::{ClientBuild, ProviderBuildCause};
use lithos_llm::credentials::EnvironmentCredentials;
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::events::RetryEventObserver;
use pebble_coding_agent::subagents::SubagentOptions;
use pebble_coding_agent::tools::PermissionLevel;
use pebble_coding_agent::{
    CodingAgent, CodingAgentOptions, Error as AgentError, InterruptReason, PromptOutcome,
    ShutdownReason,
};
use tokio::signal::ctrl_c;

use crate::render::{Renderer, Style};

/// The model used when the command line names none.
const DEFAULT_MODEL: &str = "claude-sonnet-5";

/// The prefix of the variables that point a built-in provider at a
/// compatible endpoint: `PEBBLE_OPENAI_BASE_URL`, `PEBBLE_MOONSHOT_BASE_URL`,
/// and so on. A proxy, a self-hosted model, or a test double.
const BASE_URL_ENV_PREFIX: &str = "PEBBLE_";
const BASE_URL_ENV_SUFFIX: &str = "_BASE_URL";

/// The exit status when the prompt was interrupted or timed out.
const INTERRUPTED: u8 = 130;

#[derive(Debug, Args)]
pub(crate) struct ExecArgs {
    /// The prompt. Read from standard input when omitted.
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,

    /// The model selector, as the catalog knows it.
    #[arg(short = 'm', long, default_value = DEFAULT_MODEL, value_name = "MODEL")]
    model: String,

    /// The directory the agent works in. Created when missing.
    #[arg(short = 'C', long, default_value = ".", value_name = "DIR")]
    cwd: PathBuf,

    /// What the agent may do without asking.
    #[arg(long, value_enum, default_value_t = PermissionArg::ReadWrite)]
    permission: PermissionArg,

    /// End the prompt after this long, for example 30s, 10m, or 1h.
    #[arg(long, value_parser = parse_duration, value_name = "DURATION")]
    timeout: Option<Duration>,

    /// Extra instructions added to the system prompt.
    #[arg(long, value_name = "TEXT")]
    instructions: Option<String>,

    /// Write every event to standard error as one JSON object per line.
    #[arg(long)]
    json: bool,

    /// Write nothing to standard error but errors.
    #[arg(short = 'q', long, conflicts_with = "json")]
    quiet: bool,

    /// Let the agent spawn subagents for independent work.
    #[arg(long)]
    subagents: bool,
}

/// The permission ladder, as a flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PermissionArg {
    /// Reads only.
    ReadOnly,
    /// Reads and writes, but no commands.
    ReadWrite,
    /// Everything, including commands.
    Full,
}

impl From<PermissionArg> for PermissionLevel {
    fn from(level: PermissionArg) -> Self {
        match level {
            PermissionArg::ReadOnly => Self::ReadOnly,
            PermissionArg::ReadWrite => Self::ReadWrite,
            PermissionArg::Full => Self::Full,
        }
    }
}

/// Runs the command and answers with the process exit code.
pub(crate) async fn run(args: ExecArgs) -> ExitCode {
    let style = if args.json {
        Style::Json
    } else if args.quiet {
        Style::Quiet
    } else {
        Style::Text
    };
    match exec(args, style).await {
        Ok(Ending::Answered) => ExitCode::SUCCESS,
        Ok(Ending::Interrupted) => ExitCode::from(INTERRUPTED),
        Err(error) => {
            report_error(&error);
            ExitCode::FAILURE
        }
    }
}

/// How a prompt that produced no error ended.
enum Ending {
    Answered,
    Interrupted,
}

async fn exec(args: ExecArgs, style: Style) -> Result<Ending> {
    let prompt = read_prompt(args.prompt.as_deref())?;
    let level = PermissionLevel::from(args.permission);

    let replay = RetryPolicy::exponential().max_attempts(4);
    let client = build_client(replay)?;

    let environment = LocalEnvironment::new(&args.cwd);
    environment
        .prepare()
        .await
        .with_context(|| format!("preparing the working directory {}", args.cwd.display()))?;

    let mut options = CodingAgentOptions::default().with_turn_replay(replay);
    if let Some(timeout) = args.timeout {
        options = options.with_wall_clock_timeout(timeout);
    }
    if let Some(instructions) = args.instructions {
        options = options.with_user_instructions(instructions);
    }

    let mut builder = CodingAgent::builder(client, Arc::new(environment))
        .model(&args.model)
        .options(options)
        .permission_level(level);
    if args.subagents {
        builder = builder.subagents(SubagentOptions::enabled());
    }
    let mut agent = builder.build().await.context("building the coding agent")?;

    let renderer = tokio::spawn(Renderer::new(style).run(agent.subscribe()));
    let control = agent.control_handle();
    let interrupt = tokio::spawn(async move {
        if ctrl_c().await.is_ok() {
            control.abort();
        }
    });

    let result = agent.prompt(prompt).await;
    interrupt.abort();
    let reason = match &result {
        Ok(_) => ShutdownReason::Completed,
        Err(AgentError::Interrupted(_)) => ShutdownReason::Cancelled,
        Err(_) => ShutdownReason::Error,
    };
    // Closing publishes what is queued and ends the stream, which is what
    // lets the renderer be joined here and its summary come after the answer.
    let shutdown = agent.shutdown(reason).await;
    let summary = renderer.await.context("joining the event renderer")?;

    match result {
        Ok(outcome) => {
            write_answer(&outcome)?;
            summary.report(&outcome, style);
            shutdown.context("shutting the agent down")?;
            Ok(Ending::Answered)
        }
        Err(AgentError::Interrupted(reason)) => {
            report_interrupt(reason);
            Ok(Ending::Interrupted)
        }
        Err(error) => Err(error).context("running the prompt"),
    }
}

/// The prompt from the command line, or from standard input when absent.
fn read_prompt(argument: Option<&str>) -> Result<String> {
    let prompt = if let Some(prompt) = argument {
        prompt.to_owned()
    } else {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            bail!("no prompt was given: pass one as an argument or on standard input");
        }
        let mut prompt = String::new();
        stdin
            .lock()
            .read_to_string(&mut prompt)
            .context("reading the prompt from standard input")?;
        prompt
    };
    if prompt.trim().is_empty() {
        bail!("the prompt is empty");
    }
    Ok(prompt)
}

/// A client that talks to whatever provider the model resolves to, with the
/// retry middleware pebble asks an application to install.
fn build_client(policy: RetryPolicy) -> Result<Client> {
    let mut catalog = Catalog::builder().with_builtin();
    for (variable, provider, base_url) in base_url_overrides() {
        catalog = catalog
            .toml_layer(&variable, &base_url_layer(&provider, &base_url))
            .with_context(|| format!("applying {variable}"))?;
    }
    let catalog = catalog
        .build()
        .context("loading the built-in model catalog")?;
    let ClientBuild { client, issues, .. } = Client::builder()
        .catalog(catalog)
        .credentials(EnvironmentCredentials::conventional())
        .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
        .build()
        .context("building the model client")?;
    // A provider whose adapter feature this build does not enable degrades
    // the client rather than failing it. That is expected for the providers
    // pebble leaves out on purpose, so only a provider that should have been
    // built is worth a line.
    for issue in &issues {
        if !matches!(
            issue.cause,
            ProviderBuildCause::AdapterFeatureDisabled { .. }
        ) {
            print_err(&format!(
                "warning: provider {} is unavailable: {}",
                issue.provider, issue.cause
            ));
        }
    }
    Ok(client)
}

/// Writes the final answer to standard output, ending it with one newline.
fn write_answer(outcome: &PromptOutcome) -> Result<()> {
    let Some(text) = outcome.text() else {
        return Ok(());
    };
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(text.as_bytes())
        .and_then(|()| {
            if text.ends_with('\n') {
                Ok(())
            } else {
                stdout.write_all(b"\n")
            }
        })
        .context("writing the answer")
}

fn report_interrupt(reason: InterruptReason) {
    let what = match reason {
        InterruptReason::WallClockTimeout => "the prompt ran out of time",
        InterruptReason::Cancelled => "the prompt was interrupted",
        _ => "the prompt was stopped",
    };
    print_err(&format!("error: {what}"));
}

/// Renders an error and every cause under it.
fn report_error(error: &anyhow::Error) {
    let mut rendered = format!("error: {error}");
    for cause in error.chain().skip(1) {
        rendered.push_str("\n  caused by: ");
        rendered.push_str(&cause.to_string());
    }
    print_err(&rendered);
}

/// Every `PEBBLE_<PROVIDER>_BASE_URL` set in the environment, as the variable,
/// the provider id it names, and its value. Sorted, so layers apply in one
/// order however the environment is enumerated.
fn base_url_overrides() -> Vec<(String, String, String)> {
    let mut overrides: Vec<_> = env::vars_os()
        .filter_map(|(name, value)| {
            let name = name.into_string().ok()?;
            let provider = name
                .strip_prefix(BASE_URL_ENV_PREFIX)?
                .strip_suffix(BASE_URL_ENV_SUFFIX)?;
            if provider.is_empty() {
                return None;
            }
            let value = value.into_string().ok()?;
            Some((name.clone(), provider.to_ascii_lowercase(), value))
        })
        .collect();
    overrides.sort();
    overrides
}

/// A catalog layer that moves one built-in provider to `base_url`.
///
/// Layers merge over the built-in catalog, so every model row keeps its
/// limits, pricing, and harness; only where requests go changes.
fn base_url_layer(provider: &str, base_url: &str) -> String {
    format!(
        "schema_version = 1\n[providers.{}]\nbase_url = {}\n",
        TomlKey(provider),
        TomlString(base_url)
    )
}

/// A table key, bare when TOML allows it and quoted otherwise.
struct TomlKey<'a>(&'a str);

impl fmt::Display for TomlKey<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bare = !self.0.is_empty()
            && self.0.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            });
        if bare {
            formatter.write_str(self.0)
        } else {
            TomlString(self.0).fmt(formatter)
        }
    }
}

/// A string as a TOML basic string literal.
struct TomlString<'a>(&'a str);

impl fmt::Display for TomlString<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("\"")?;
        for character in self.0.chars() {
            match character {
                '"' => formatter.write_str("\\\"")?,
                '\\' => formatter.write_str("\\\\")?,
                other => write!(formatter, "{other}")?,
            }
        }
        formatter.write_str("\"")
    }
}

fn parse_duration(raw: &str) -> Result<Duration, String> {
    if raw.trim().parse::<u64>().is_ok() {
        return Err("give the duration a unit, for example 30s, 10m, or 1h".to_owned());
    }
    humantime::parse_duration(raw).map_err(|error| error.to_string())
}

#[expect(clippy::print_stderr, reason = "the command's stderr boundary")]
pub(crate) fn print_err(text: &str) {
    eprintln!("{text}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_duration_needs_a_unit() {
        assert!(parse_duration("30").is_err());
        assert_eq!(parse_duration("30s"), Ok(Duration::from_secs(30)));
        assert_eq!(parse_duration("2m"), Ok(Duration::from_secs(120)));
    }

    #[test]
    fn the_base_url_layer_quotes_its_values() {
        let layer = base_url_layer("moonshot", "http://127.0.0.1:1234/v1");
        assert!(layer.contains("[providers.moonshot]"));
        assert!(base_url_layer("odd name", "x").contains("[providers.\"odd name\"]"));
        assert!(layer.contains("base_url = \"http://127.0.0.1:1234/v1\""));
        assert_eq!(TomlString("a\"b\\c").to_string(), "\"a\\\"b\\\\c\"");
    }

    #[test]
    fn a_permission_flag_maps_onto_the_ladder() {
        assert_eq!(
            PermissionLevel::from(PermissionArg::ReadOnly),
            PermissionLevel::ReadOnly
        );
        assert_eq!(
            PermissionLevel::from(PermissionArg::Full),
            PermissionLevel::Full
        );
    }
}
