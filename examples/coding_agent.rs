//! A coding agent, end to end.
//!
//! The example builds a lithos-llm client, runs a pebble session over a fresh
//! directory, renders the session's events as they arrive, steers one run,
//! interrupts another, and reports what the whole thing did and cost.
//!
//! ```sh
//! cargo run --example coding_agent              # the default model
//! cargo run --example coding_agent -- gpt-5.6   # any selector the catalog knows
//! ```
//!
//! It talks to a real provider. Credentials come from the conventional
//! environment variables lithos-llm resolves for each provider —
//! `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and so on — and nothing here reads
//! one itself.
//!
//! The session works in a new directory under the system temporary directory.
//! The example prints the path and leaves the directory behind, so the files
//! the model wrote can be read afterwards.

#![expect(
    clippy::print_stderr,
    reason = "an example reports to the person who ran it, and the terminal is where they are"
)]

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{env, process};

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::client::ClientBuild;
use lithos_llm::credentials::EnvironmentCredentials;
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use pebble::{
    AgentEvent, LocalEnvironment, RetryEventObserver, Session, SessionControlHandle, SessionEvent,
    SessionOptions, ShutdownReason, TokenUsage,
};
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;
use tokio::task::JoinHandle;
use tokio::time::timeout;

/// The model the example asks for when the command line names none.
const DEFAULT_MODEL: &str = "claude-sonnet-5";

/// The work the first run is given: a file, an edit, and a command.
const FIRST_PROMPT: &str = "\
Work only inside your working directory, and use your tools rather than telling me what to do.

1. Write a file `greeting.txt` with exactly three lines of text.
2. Edit `greeting.txt` so its second line reads exactly `edited by pebble`.
3. Run one shell command that prints the file with line numbers.

Then tell me, in one sentence, what the command printed.";

/// What the steer adds to the first run, once the model is already working.
const STEER: &str = "One more thing: also write `notes.md` listing what you have done so far.";

/// The work the second run is given, which is long enough to interrupt.
const SECOND_PROMPT: &str = "\
Write a long, detailed description of every file in your working directory: what it contains, \
line by line, and how you would extend it. Take your time and be thorough.";

/// What is said to the session the interrupt parked.
const REDIRECT: &str = "Never mind the description. Reply with the single word DONE.";

/// How long a control task waits for the moment it acts on.
///
/// The end of the run is the usual answer, and this is only the backstop for a
/// stream that stops without ending.
const CONTROL_PATIENCE: Duration = Duration::from_secs(300);

/// How long one run may take before the session ends it.
const RUN_BUDGET: Duration = Duration::from_secs(600);

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("\nthe example failed: {error}");
            let mut cause = error.source();
            while let Some(source) = cause {
                eprintln!("  caused by: {source}");
                cause = source.source();
            }
            ExitCode::FAILURE
        }
    }
}

/// Everything the example does, with one failure type on the way out.
async fn run() -> Result<(), Box<dyn StdError>> {
    let model = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_owned());
    let workspace = workspace_path();
    eprintln!("model:     {model}");
    eprintln!("workspace: {}", workspace.display());

    // The application builds the client, so the application installs the retry
    // middleware. `RetryEventObserver` is what puts the client's own retries on
    // the session's event stream; without it a session still runs correctly and
    // simply never reports one.
    let policy = RetryPolicy::exponential().max_attempts(4);
    let client = build_client(policy)?;

    // Tools act through the environment, and this one is a directory on this
    // machine. `prepare` creates it and proves commands will run, so a missing
    // interpreter is one clear failure rather than a puzzling first tool call.
    let environment = LocalEnvironment::new(&workspace);
    environment.prepare().await?;

    let mut session = Session::builder(client)
        .model(&model)
        .environment(Arc::new(environment))
        .options(SessionOptions {
            wall_clock_timeout: Some(RUN_BUDGET),
            // The same policy the client's middleware was built with, so one
            // failure is spaced the same way whichever layer handles it.
            retry_policy: policy,
            ..SessionOptions::default()
        })
        .build()?;

    // Subscribed before anything runs, so the renderer sees the session open.
    let renderer = tokio::spawn(render_events(session.subscribe()));

    // Loads what the session was told — nothing here, because no memory files
    // and no skill directories were named — and captures where it is working.
    session.initialize().await?;

    let mut totals = Totals::default();

    // --- One run, steered while it works ---
    //
    // `run` borrows the session until it returns, so everything said to a live
    // run goes through a control handle.
    let steering = steer_once(session.subscribe(), session.control_handle());

    eprintln!("\n--- run 1: write, edit, run a command ---");
    let answer = session.run(FIRST_PROMPT).await?;
    steering.await?;
    totals.add(&session);
    eprintln!("\nanswer: {}", answer.as_deref().unwrap_or("(no text)"));

    // --- One run, interrupted while it works ---
    let interrupting = interrupt_once(session.subscribe(), session.control_handle());

    eprintln!("\n--- run 2: interrupted mid-answer ---");
    let answer = session.run(SECOND_PROMPT).await?;
    interrupting.await?;
    totals.add(&session);
    eprintln!("\nanswer: {}", answer.as_deref().unwrap_or("(no text)"));

    // Closing publishes what is queued and joins everything the session owns.
    // Joining the event pump is what ends the renderer: the stream closes when
    // the session is shut down, not when the session value is dropped, so the
    // renderer can be awaited here — while `session` is still alive — and its
    // summary is the last thing printed.
    let turns = session.history().turns().len();
    session.shutdown(ShutdownReason::Completed).await?;
    let observed = renderer.await?;

    report(&model, &workspace, turns, &totals, &observed);
    Ok(())
}

/// A client that talks to whatever provider the model resolves to.
///
/// Two things here matter to a session: the catalog, which is where it reads
/// the model's limits and the harness that model expects, and the retry
/// middleware, which is the client's half of repeating a failed call.
fn build_client(policy: RetryPolicy) -> Result<Client, Box<dyn StdError>> {
    let catalog = Catalog::builder().with_builtin().build()?;
    let ClientBuild { client, issues, .. } = Client::builder()
        .catalog(catalog)
        // Every provider's conventional environment variables, resolved per
        // call by lithos-llm. Nothing here reads a key.
        .credentials(EnvironmentCredentials::conventional())
        .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
        .build()?;
    // A provider whose adapter could not be built degrades the client rather
    // than failing it, so name the ones that are missing before a run needs
    // them.
    for issue in &issues {
        eprintln!("provider unavailable: {} ({})", issue.provider, issue.cause);
    }
    Ok(client)
}

/// A new directory for this run, named so two runs never share one.
fn workspace_path() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis());
    env::temp_dir().join(format!("pebble-coding-agent-{}-{unique}", process::id()))
}

/// Adds to the job of a run that is already working.
///
/// The waiting happens in its own task, because `Session::run` borrows the
/// session for as long as the run takes. The first finished tool call is the
/// sign that the model is working and that a round boundary is coming, and a
/// steer sent then lands as its own turn at the next boundary of a run still
/// in progress.
///
/// Nothing here can make the run wait for it. A steer that loses the race to a
/// natural completion is not lost and is not injected either: it stays queued
/// and opens the next run. `SteeringInjected` is what says it landed, so this
/// task waits for that event and says so when it never comes. An application
/// that needs the race closed rather than reported gives the session a
/// `CompletionCoordinator`.
fn steer_once(
    mut events: broadcast::Receiver<SessionEvent>,
    handle: SessionControlHandle,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        wait_for_start(&mut events).await;
        if !wait_for(&mut events, |event| {
            matches!(event, AgentEvent::ToolCallCompleted { .. })
        })
        .await
        {
            eprintln!("[control] the run passed the point this was waiting for");
            return;
        }
        handle.steer(STEER, None);
        if !wait_for(&mut events, |event| {
            matches!(event, AgentEvent::SteeringInjected { .. })
        })
        .await
        {
            eprintln!("[control] the run finished first: the steer waits for the next run");
        }
    })
}

/// Abandons the round the model is answering in, then says what to do instead.
///
/// `SessionControlHandle::interrupt` ends the round and nothing else: with
/// nothing queued the session parks at the round boundary and waits, so an
/// operator can stop a run mid-thought and decide afterwards what to say. That
/// is why the steer follows. `Session::interrupt` is the other gesture — that
/// one ends the run for good.
fn interrupt_once(
    mut events: broadcast::Receiver<SessionEvent>,
    handle: SessionControlHandle,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        wait_for_start(&mut events).await;
        if !wait_for(&mut events, |event| {
            matches!(event, AgentEvent::TextDelta { .. })
        })
        .await
        {
            eprintln!("[control] the run answered before it could be interrupted");
            return;
        }
        handle.interrupt();
        // Exactly one `RoundInterrupted` is published per gesture, so waiting
        // for it is waiting for this interrupt to have settled.
        if wait_for(&mut events, |event| {
            matches!(event, AgentEvent::RoundInterrupted { .. })
        })
        .await
        {
            handle.steer(REDIRECT, None);
        }
    })
}

/// Waits until the run this task was made for has begun.
///
/// A subscriber joins the stream where it is, and the run before this one may
/// still have events in flight. Waiting for the input that starts this run is
/// what keeps the last run's ending from being read as this one's.
async fn wait_for_start(events: &mut broadcast::Receiver<SessionEvent>) {
    wait_until(events, |event| {
        matches!(event, AgentEvent::UserInput { .. })
    })
    .await;
}

/// Waits for the first event `wanted` matches, and reports whether one came.
///
/// The end of the run's processing cycle is the answer "no": a task waiting
/// for a moment the run went past must not go on waiting after the run that
/// would have produced it is over.
async fn wait_for(
    events: &mut broadcast::Receiver<SessionEvent>,
    wanted: impl Fn(&AgentEvent) -> bool + Sync,
) -> bool {
    wait_until(events, |event| {
        wanted(event) || matches!(event, AgentEvent::ProcessingEnd)
    })
    .await
    .is_some_and(|event| wanted(&event))
}

/// Reads the stream until `wanted` matches, and answers with what matched.
async fn wait_until(
    events: &mut broadcast::Receiver<SessionEvent>,
    wanted: impl Fn(&AgentEvent) -> bool + Sync,
) -> Option<AgentEvent> {
    let found = timeout(CONTROL_PATIENCE, async {
        loop {
            match events.recv().await {
                Ok(event) if wanted(&event.event) => return Some(event.event),
                Ok(_) | Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => return None,
            }
        }
    })
    .await;
    found.unwrap_or_default()
}

/// What one session accumulated across its runs.
///
/// `last_run_usage` and `last_run_cost_usd_micros` describe the run that just
/// finished, so an application that wants a session total keeps its own.
#[derive(Debug, Default)]
struct Totals {
    usage:           TokenUsage,
    cost_usd_micros: u64,
    priced:          bool,
    runs:            usize,
}

impl Totals {
    /// Adds what the run that just finished reported.
    fn add(&mut self, session: &Session) {
        let usage = session.last_run_usage();
        self.usage = TokenUsage {
            input:       self.usage.input + usage.input,
            output:      self.usage.output + usage.output,
            reasoning:   self.usage.reasoning + usage.reasoning,
            cache_read:  self.usage.cache_read + usage.cache_read,
            cache_write: self.usage.cache_write + usage.cache_write,
        };
        if let Some(cost) = session.last_run_cost_usd_micros() {
            self.cost_usd_micros += cost;
            self.priced = true;
        }
        self.runs += 1;
    }
}

/// What the event stream said, kept for the closing summary.
#[derive(Debug, Default)]
struct Observed {
    tools:      BTreeMap<String, usize>,
    failures:   usize,
    retries:    usize,
    interrupts: usize,
    steers:     usize,
    dropped:    u64,
}

/// Renders the session's events until the session closes.
///
/// The live stream is lossy for a subscriber that falls behind, which is what
/// makes it right for a terminal and wrong for a ledger: an application that
/// must see every event installs an `EventSink` instead.
///
/// Two things end this loop, and either one would do. `SessionEnded` is the
/// session saying so on its own stream, and a closed stream is the same news
/// from the pipeline: `Session::shutdown` joins the pump, and that is what
/// closes every receiver `subscribe` handed out.
async fn render_events(mut events: broadcast::Receiver<SessionEvent>) -> Observed {
    let mut observed = Observed::default();
    let mut streaming = false;
    loop {
        match events.recv().await {
            Ok(event) => {
                let ended = matches!(event.event, AgentEvent::SessionEnded);
                render(&event.event, &mut observed, &mut streaming);
                if ended {
                    break;
                }
            }
            Err(RecvError::Lagged(dropped)) => {
                observed.dropped += dropped;
                eprintln!("[{dropped} events dropped: this reader fell behind]");
            }
            Err(RecvError::Closed) => break,
        }
    }
    if streaming {
        eprintln!();
    }
    observed
}

/// Prints one event, and remembers the ones the summary counts.
fn render(event: &AgentEvent, observed: &mut Observed, streaming: &mut bool) {
    // The model's text is the one thing that gets no line of its own: it
    // arrives in pieces and is printed as it arrives.
    if let AgentEvent::TextDelta { delta } = event {
        eprint!("{delta}");
        *streaming = true;
        return;
    }
    if *streaming {
        eprintln!();
        *streaming = false;
    }

    match event {
        AgentEvent::SessionStarted { provider, model } => eprintln!(
            "[open] {}/{}",
            provider.as_deref().unwrap_or("?"),
            model.as_deref().unwrap_or("?")
        ),
        AgentEvent::SessionEnded => eprintln!("[close]"),
        AgentEvent::MemoryLoaded { files, .. } => eprintln!("[memory] {} file(s)", files.len()),
        AgentEvent::SkillsDiscovered { skills, .. } => {
            eprintln!("[skills] {} discovered", skills.len());
        }
        AgentEvent::LlmRequestStarted { requested_model } => eprintln!("[ask] {requested_model}"),
        AgentEvent::AssistantMessage {
            usage,
            cost_usd_micros,
            tool_call_count,
            ..
        } => eprintln!(
            "[turn] {} tokens, {tool_call_count} tool call(s){}",
            usage.total(),
            cost_usd_micros.map_or_else(String::new, |cost| format!(", {}", dollars(cost)))
        ),
        AgentEvent::AssistantOutputReplace { .. } => {
            eprintln!("[replay] the last output was withdrawn and the turn is being asked again");
        }
        AgentEvent::ToolCallStarted {
            tool_name,
            arguments,
            ..
        } => eprintln!("[tool] {tool_name} {}", abbreviate(&arguments.to_string())),
        AgentEvent::ToolCallCompleted {
            tool_name,
            is_error,
            ..
        } => {
            *observed.tools.entry(tool_name.clone()).or_default() += 1;
            if *is_error {
                observed.failures += 1;
                eprintln!("[tool] {tool_name} failed");
            }
        }
        AgentEvent::ToolProcessCompleted {
            exit_code,
            duration_ms,
            ..
        } => eprintln!(
            "[exec] exit {} in {duration_ms} ms",
            exit_code.map_or_else(|| "?".to_owned(), |code| code.to_string())
        ),
        AgentEvent::SteeringInjected { text, .. } => {
            observed.steers += 1;
            eprintln!("[steer] {}", first_line(text));
        }
        AgentEvent::RoundInterrupted { generation } => {
            observed.interrupts += 1;
            eprintln!("[interrupt] round abandoned (gesture {generation})");
        }
        AgentEvent::LlmRetry {
            attempt,
            delay_secs,
            error,
            ..
        } => {
            observed.retries += 1;
            eprintln!(
                "[retry] attempt {attempt} in {delay_secs:.1}s: {}",
                error.message
            );
        }
        AgentEvent::LoopDetected => eprintln!("[loop] the session is repeating itself"),
        AgentEvent::CompactionCompleted {
            original_turn_count,
            preserved_turn_count,
            ..
        } => eprintln!("[compact] {original_turn_count} turns down to {preserved_turn_count}"),
        AgentEvent::Error { error } => eprintln!("[error] {}", error.message),
        AgentEvent::Warning { kind, message, .. } => eprintln!("[warning] {kind}: {message}"),
        AgentEvent::TodoCreated(props) => eprintln!("[plan] {}", props.subject),
        AgentEvent::TodoUpdated(props) => eprintln!(
            "[plan] {} is now {}",
            props.todo_id,
            props
                .status
                .map_or_else(|| "unchanged".to_owned(), |status| format!("{status:?}"))
        ),
        // Everything else — the pieces of a command's output, the model's
        // reasoning, the end of a processing cycle — is left off a terminal
        // that is busy enough. `AgentEvent` is `#[non_exhaustive]`, so a
        // reader needs this arm however much it renders.
        _ => {}
    }
}

/// What the whole example did.
fn report(model: &str, workspace: &Path, turns: usize, totals: &Totals, observed: &Observed) {
    let calls: usize = observed.tools.values().sum();
    let named = observed
        .tools
        .iter()
        .map(|(name, count)| format!("{name} x{count}"))
        .collect::<Vec<_>>()
        .join(", ");

    eprintln!("\n--- summary ---");
    eprintln!("model:      {model}");
    eprintln!("runs:       {}", totals.runs);
    eprintln!("turns:      {turns}");
    eprintln!("tools:      {calls} call(s), {} failed", observed.failures);
    if !named.is_empty() {
        eprintln!("            {named}");
    }
    eprintln!(
        "tokens:     {} in, {} out, {} reasoning, {} cached ({} total)",
        totals.usage.input,
        totals.usage.output,
        totals.usage.reasoning,
        totals.usage.cache_read + totals.usage.cache_write,
        totals.usage.total()
    );
    if totals.priced {
        eprintln!("cost:       {}", dollars(totals.cost_usd_micros));
    } else {
        eprintln!("cost:       not reported for this model");
    }
    eprintln!("steers:     {}", observed.steers);
    eprintln!("interrupts: {}", observed.interrupts);
    eprintln!("retries:    {}", observed.retries);
    if observed.dropped > 0 {
        eprintln!(
            "dropped:    {} event(s) this reader missed",
            observed.dropped
        );
    }
    eprintln!("workspace:  {}", workspace.display());
}

/// Millionths of a dollar, as dollars.
fn dollars(usd_micros: u64) -> String {
    format!("${:.4}", usd_micros as f64 / 1_000_000.0)
}

/// The first line of a block of text, for a one-line report of it.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

/// One line of a tool call's arguments, short enough to read.
fn abbreviate(text: &str) -> String {
    const LIMIT: usize = 120;
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= LIMIT {
        return flat;
    }
    flat.chars().take(LIMIT).collect::<String>() + "…"
}
