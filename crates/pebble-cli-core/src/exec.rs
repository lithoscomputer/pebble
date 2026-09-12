//! The `exec` command: one prompt, run to completion.

use std::io::{self, IsTerminal as _, Read as _};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use clap::Args;
use lithos_llm::middleware::RetryPolicy;
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::subagents::SubagentOptions;
use pebble_coding_agent::tools::PermissionLevel;
use pebble_coding_agent::{CodingAgent, CodingAgentOptions, Error as AgentError, InterruptReason};
use tokio::signal::ctrl_c;
use tokio_util::sync::CancellationToken;

use crate::application::{Application, DEFAULT_MODEL, PermissionArg, model_route};
use crate::render::{JsonStream, Style};
use crate::resources;
use crate::session::{SessionOptions, run_prompt};
use crate::settings::project;
use crate::terminal::print_err;

/// The exit status when the prompt was interrupted or timed out.
const INTERRUPTED: u8 = 130;

#[derive(Debug, Args)]
pub struct ExecArgs {
    /// The prompt. Read from standard input when omitted.
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,

    /// The model selector (default: saved preference, then claude-sonnet-5).
    #[arg(short = 'm', long, value_name = "MODEL")]
    model: Option<String>,

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

/// Runs the command and answers with the process exit code.
pub async fn run(args: ExecArgs) -> ExitCode {
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
    let application = Application::load(replay).await?;
    let defaults = project::load(&args.cwd, &application.settings).await?;
    let model = args
        .model
        .or(defaults.model)
        .unwrap_or_else(|| DEFAULT_MODEL.into());
    let route = model_route(&application.client, &model)?;
    application.auth.resolve(route.provider()).await?;
    let client = application.client;

    let environment = LocalEnvironment::new(&args.cwd);
    environment
        .prepare()
        .await
        .with_context(|| format!("preparing the working directory {}", args.cwd.display()))?;

    let mut options = CodingAgentOptions::default()
        .with_turn_replay(replay)
        .with_reasoning_effort(defaults.reasoning);
    if let Some(timeout) = args.timeout {
        options = options.with_wall_clock_timeout(timeout);
    }
    if let Some(instructions) = args.instructions {
        options = options.with_user_instructions(instructions);
    }
    let options = resources::options(&args.cwd, options).await?;

    let mut builder = CodingAgent::builder(client, Arc::new(environment))
        .model(&model)
        .options(options)
        .permission_level(level);
    if args.subagents {
        builder = builder.subagents(SubagentOptions::enabled());
    }
    let agent = builder.build().await.context("building the coding agent")?;

    // Ctrl-C ends the prompt; the session shuts down as cancelled.
    let cancel = CancellationToken::new();
    let interrupt = tokio::spawn({
        let cancel = cancel.clone();
        async move {
            if ctrl_c().await.is_ok() {
                cancel.cancel();
            }
        }
    });
    let report = run_prompt(agent, prompt, &cancel, SessionOptions {
        style,
        json_to: JsonStream::Stderr,
        write_answer: true,
    })
    .await;
    interrupt.abort();
    match report?.result {
        Ok(_) => Ok(Ending::Answered),
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

fn report_interrupt(reason: InterruptReason) {
    let what = match reason {
        InterruptReason::WallClockTimeout => "the prompt ran out of time",
        InterruptReason::Cancelled => "the prompt was interrupted",
        InterruptReason::TurnLimit => "the prompt used its turn budget",
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

fn parse_duration(raw: &str) -> Result<Duration, String> {
    if raw.trim().parse::<u64>().is_ok() {
        return Err("give the duration a unit, for example 30s, 10m, or 1h".to_owned());
    }
    humantime::parse_duration(raw).map_err(|error| error.to_string())
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
}
