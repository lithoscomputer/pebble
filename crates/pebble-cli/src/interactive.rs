//! Interactive coding sessions in normal terminal scrollback.

use std::env;
use std::io::{self, IsTerminal as _};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use clap::Args;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use lithos_llm::Client;
use lithos_llm::middleware::RetryPolicy;
use pebble_coding_agent::events::CodingEvent;
use pebble_coding_agent::{CodingInput, SteeringMessage, SteeringOutcome};
use tokio::fs;
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{MissedTickBehavior, interval};
use tokio_util::sync::CancellationToken;

use crate::application::{Application, DEFAULT_MODEL, PermissionArg, model_route};
use crate::credentials::AuthStore;
use crate::exec::print_err;

mod attachments;
mod branch;
mod commands;
mod dialog;
mod editor;
mod highlight;
mod input;
mod menu;
mod services;
mod setup;
mod shell;
use crate::settings;
mod store;
mod terminal;
mod text;
mod tool_render;
mod transcript;
mod worker;

use dialog::Dialog;
use editor::Editor;
use input::Input;
use services::{Request, Services};
use store::{Metadata, Store};
use terminal::Terminal;
use transcript::{Output, Transcript};
use worker::{Command, Worker};

#[derive(Debug, Default, Args)]
pub(crate) struct InteractiveArgs {
    /// An optional first prompt for the interactive session.
    #[arg(value_name = "PROMPT")]
    prompt:           Option<String>,
    /// The model selector, as the catalog knows it.
    #[arg(short = 'm', long, value_name = "MODEL")]
    model:            Option<String>,
    /// The directory the agent works in.
    #[arg(short = 'C', long, value_name = "DIR")]
    cwd:              Option<PathBuf>,
    /// What the agent may do without asking.
    #[arg(long, value_enum)]
    permission:       Option<PermissionArg>,
    /// Resume a session id, or the latest session when no id is given.
    #[arg(short = 'r', long, num_args = 0..=1, default_missing_value = "latest", value_name = "ID")]
    resume:           Option<String>,
    /// Continue the latest session for this directory.
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    continue_session: bool,
    /// Display name for this session.
    #[arg(long)]
    name:             Option<String>,
    /// Directory for saved sessions (default: ~/.pebble/sessions).
    #[arg(long, value_name = "DIR")]
    sessions_dir:     Option<PathBuf>,
    /// Use temporary session storage, removed on exit.
    #[arg(long, conflicts_with_all = ["resume", "continue_session"])]
    no_session:       bool,
    /// Hide disallowed tools instead of asking for approval.
    #[arg(long)]
    no_approvals:     bool,
    /// Let the agent spawn subagents for independent work.
    #[arg(long)]
    subagents:        bool,
    /// Extra instructions added to the system prompt.
    #[arg(long, value_name = "TEXT")]
    instructions:     Option<String>,
}

pub(crate) async fn run(args: InteractiveArgs) -> ExitCode {
    match Box::pin(start(args)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            print_err(&format!("error: {error:#}"));
            ExitCode::FAILURE
        }
    }
}

struct App {
    client:           Client,
    auth:             AuthStore,
    login:            Option<setup::Login>,
    root:             PathBuf,
    store:            Arc<Store>,
    metadata:         Metadata,
    worker:           Option<Worker>,
    services:         Arc<Services>,
    requests:         mpsc::Receiver<Request>,
    terminal:         Terminal,
    input:            Option<Input>,
    editor:           Editor,
    transcript:       Transcript,
    dialog:           Option<Dialog>,
    menu:             Option<menu::Menu>,
    completion_files: Option<Vec<String>>,
    shell:            Option<shell::Job>,
    busy:             bool,
    dirty:            bool,
    quit:             bool,
    last_interrupt:   Option<Instant>,
    operation_cancel: CancellationToken,
    settings:         settings::Settings,
    settings_path:    PathBuf,
    attachments:      attachments::Attachments,
}

async fn start(args: InteractiveArgs) -> Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        bail!("interactive mode needs a terminal; use `pebble exec <PROMPT>` for scripts");
    }
    let Application {
        paths,
        mut settings,
        client,
        auth,
    } = Application::load(RetryPolicy::exponential().max_attempts(4)).await?;
    let settings_path = paths.settings();
    let temporary = if args.no_session {
        Some(tempfile::tempdir()?)
    } else {
        None
    };
    let root = if let Some(directory) = &temporary {
        directory.path().join("sessions")
    } else if let Some(root) = args.sessions_dir {
        root
    } else {
        paths.sessions()
    };
    let cwd = args.cwd.unwrap_or(env::current_dir()?);
    fs::create_dir_all(&cwd)
        .await
        .context("creating the working directory")?;
    let cwd = fs::canonicalize(cwd).await?;
    let requested = if args.continue_session {
        Some("latest".to_owned())
    } else {
        args.resume
    };
    let requested = if requested.as_deref() == Some("latest") {
        Some(
            Store::list(&root)
                .await?
                .into_iter()
                .find(|session| session.cwd == cwd)
                .context("no saved session for this directory")?
                .id,
        )
    } else {
        requested
    };
    let store = Store::open(&root, requested.as_deref()).await?;
    let checkpoint = if requested.is_some() {
        Some(store.load().await?)
    } else {
        None
    };
    let needs_model_choice =
        checkpoint.is_none() && args.model.is_none() && settings.model.is_none();
    let mut metadata = checkpoint.as_ref().map_or_else(
        || Metadata {
            id: store.id(),
            name: args.name.clone().unwrap_or_else(|| "New session".into()),
            cwd,
            model: settings
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL.into()),
            permission: PermissionArg::ReadWrite,
            reasoning: settings.reasoning,
            subagents: args.subagents,
            instructions: args.instructions.clone(),
            approvals: !args.no_approvals,
            updated_at: store::timestamp(),
            forked_from: None,
            forked_at: None,
        },
        |checkpoint| checkpoint.metadata.clone(),
    );
    if let Some(model) = args.model {
        metadata.model = model;
    }
    if let Some(permission) = args.permission {
        metadata.permission = permission;
    }
    if let Some(name) = args.name {
        metadata.name = name;
    }
    metadata.subagents |= args.subagents;
    if let Some(instructions) = args.instructions {
        metadata.instructions = Some(instructions);
    }
    if args.no_approvals {
        metadata.approvals = false;
    }
    if checkpoint.is_some() {
        shell::reconcile(&store).await?;
    }
    let last_sequence = store.last_sequence().await?;
    let mut recovered = store.repaired_tail;
    let record = checkpoint.map(|checkpoint| {
        let mut record = checkpoint.record;
        recovered |= last_sequence > record.last_event_seq;
        record.advance_event_cursor(last_sequence);
        record
    });

    let (services, requests) = Services::channel();
    let services = Arc::new(services);
    let attachments = attachments::Attachments::open(store.directory()).await?;
    Terminal::install_panic_hook();
    let mut terminal =
        Terminal::open(env::var_os("NO_COLOR").is_none()).context("opening the terminal")?;
    let mut input = Input::start();
    let selected = setup::choose(
        &client,
        &auth,
        &mut settings,
        &settings_path,
        if needs_model_choice {
            None
        } else {
            Some(&metadata.model)
        },
        &mut terminal,
        &mut input,
    )
    .await;
    match selected {
        Ok(Some(model)) => metadata.model = model,
        other => {
            input.shutdown().await?;
            terminal.pause()?;
            return other.map(|_| ());
        }
    }
    let worker = Worker::start(
        client.clone(),
        metadata.clone(),
        store.clone(),
        record,
        services.clone(),
    )
    .await;
    let worker = match worker {
        Ok(worker) => worker,
        Err(error) => {
            input.shutdown().await?;
            terminal.pause()?;
            return Err(error);
        }
    };
    metadata.model = format!("{}/{}", worker.snapshot.provider(), worker.snapshot.model());
    let mut transcript = Transcript::new(worker.snapshot.session_id());
    transcript.show_reasoning = settings.show_reasoning;
    transcript.expand_tools = settings.expand_tools;
    let mut app = App {
        client,
        auth,
        login: None,
        root,
        store,
        metadata,
        worker: Some(worker),
        services,
        requests,
        terminal,
        input: Some(input),
        editor: Editor::default(),
        transcript,
        dialog: None,
        menu: None,
        completion_files: None,
        shell: None,
        busy: false,
        dirty: true,
        quit: false,
        last_interrupt: None,
        operation_cancel: CancellationToken::new(),
        settings,
        settings_path,
        attachments,
    };
    let result = async {
        app.header()?;
        app.replay(true).await?;
        if recovered { let output = app.transcript.interrupted(); app.output(output)?; app.terminal.message("Recovered the last saved checkpoint. Later work is shown in the transcript but will not be rerun.")?; }
        if let Some(prompt) = args.prompt { app.submit(prompt, false).await?; }
        app.run().await
    }.await;
    let shell_shutdown = if let Some(shell) = app.shell.take() {
        shell.shutdown().await
    } else {
        Ok(())
    };
    let shutdown = if let Some(worker) = app.worker.take() {
        worker.shutdown().await
    } else {
        Ok(())
    };
    if let Some(input) = app.input.take() {
        input.shutdown().await?;
    }
    if !args.no_session {
        app.terminal
            .message(&format!("Resume: pebble --resume {}", app.metadata.id))?;
    }
    app.terminal.pause()?;
    drop(app);
    drop(temporary);
    shutdown?;
    shell_shutdown?;
    result
}

impl App {
    fn header(&mut self) -> Result<()> {
        self.terminal.message(&format!("Pebble · {}\n{}\n{} · {}{}\nEnter submits · Shift+Enter adds a line · Esc cancels · /help", self.metadata.name, self.metadata.cwd.display(), self.metadata.model, self.metadata.permission, if self.metadata.approvals { " · asks for other tools" } else { " · other tools disabled" }))?;
        Ok(())
    }

    async fn run(&mut self) -> Result<()> {
        let mut tick = interval(Duration::from_millis(33));
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        #[cfg(unix)]
        let mut terminate = signal(SignalKind::terminate())?;
        #[cfg(unix)]
        let mut hangup = signal(SignalKind::hangup())?;
        let shutdown_signal = async move {
            #[cfg(unix)]
            tokio::select! { _ = terminate.recv() => {}, _ = hangup.recv() => {} }
            #[cfg(not(unix))]
            std::future::pending::<()>().await;
        };
        tokio::pin!(shutdown_signal);
        loop {
            let worker = self
                .worker
                .as_mut()
                .context("the interactive agent is unavailable")?;
            tokio::select! {
                () = &mut shutdown_signal => { self.operation_cancel.cancel(); self.quit = true; }
                input = async { self.input.as_mut().expect("the terminal input reader is installed during the main loop").recv().await } => {
                    match input {
                        Some(Ok(Event::Key(key))) if key.kind != KeyEventKind::Release => self.key(key).await?,
                        Some(Ok(Event::Paste(value))) => {
                            if let Some(login) = &mut self.login {
                                if !login.input.paste(&value) { self.terminal.message("API key must be printable ASCII, at most 8192 bytes.")?; }
                            } else if let Some(menu) = self.menu.as_mut().filter(|menu| !menu.is_completion()) {
                                menu.paste(&value);
                            } else {
                                self.menu = None;
                                let editor = self.dialog.as_mut().map_or(&mut self.editor, |dialog| &mut dialog.editor);
                                if !editor.insert(&value) { self.terminal.message("Paste is too large; attach a file or use a smaller prompt.")?; }
                            }
                            self.dirty = true;
                        }
                        Some(Ok(Event::Resize(width, height))) => { self.terminal.resize(width, height); self.dirty = true; }
                        Some(Err(error)) => return Err(error).context("reading terminal input"),
                        None => break,
                        _ => {}
                    }
                }
                event = worker.events.recv(), if !worker.events_closed => {
                    match event {
                        Ok(event) => { let output = self.transcript.apply(&event, false); self.output(output)?; }
                        Err(broadcast::error::RecvError::Lagged(_)) => self.replay(false).await?,
                        Err(broadcast::error::RecvError::Closed) => {
                            if self.busy { self.terminal.message("The agent closed. Use /resume or /new to continue.")?; }
                            self.busy = false;
                            if let Some(worker) = self.worker.as_mut() { worker.events_closed = true; }
                            self.dirty = true;
                        }
                    }
                }
                output = async { self.shell.as_mut().expect("shell branch is enabled only while running").output.recv().await }, if self.shell.is_some() => {
                    if let Some(output) = output { self.terminal.message(&output)?; self.dirty = true; }
                    else { self.finish_shell().await?; }
                }
                notice = worker.finished.recv() => {
                    let notice = notice.context("the coding agent task stopped")?;
                    self.replay(false).await?;
                    shell::reconcile(&self.store).await?;
                    self.busy = false;
                    if let Some(error) = notice.error { self.terminal.message(&error)?; }
                    for input in notice.restored { let restored = self.attachments.render_parts(input.content().parts()).await?; self.editor.restore(&restored); }
                    if let Some(worker) = self.worker.as_ref() {
                        let (steering, follow_ups) = worker.control.take_pending_input().into_parts();
                        for input in steering.into_iter().chain(follow_ups) { let restored = self.attachments.render_parts(input.content().parts()).await?; self.editor.restore(&restored); }
                    }
                    self.completion_files = None;
                    self.metadata = notice.metadata;
                    if let Some(worker) = self.worker.as_mut() { worker.snapshot = notice.snapshot; }
                    self.dirty = true;
                }
                Some(request) = self.requests.recv(), if self.dialog.is_none() => {
                    let dialog = Dialog::new(request);
                    self.terminal.message(&dialog.description())?;
                    self.dialog = Some(dialog); self.dirty = true;
                }
                _ = tick.tick() => {
                    if self.dialog.as_ref().is_some_and(Dialog::cancelled) { self.dialog.take(); self.dirty = true; }
                    if self.dirty { self.draw()?; self.dirty = false; }
                }
            }
            if self.quit {
                break;
            }
        }
        Ok(())
    }

    fn output(&mut self, output: Vec<Output>) -> Result<()> {
        for item in output {
            match item {
                Output::Text(text) => self.terminal.message(&text)?,
                Output::Markdown(text) => self.terminal.markdown(&text)?,
                Output::Code { source, language } => self.terminal.code(&source, &language)?,
            }
        }
        self.dirty = true;
        Ok(())
    }

    async fn replay(&mut self, historical: bool) -> Result<()> {
        let mut reader = self.store.reader().await?;
        while let Some(event) = reader.next().await? {
            if historical
                && event.seq > self.transcript.cursor
                && event.session_id == self.transcript.root
            {
                match &event.event {
                    CodingEvent::UserInput { text, content, .. }
                    | CodingEvent::SteeringInjected { text, content, .. } => {
                        let text = if let Some(content) = content {
                            self.attachments.render_parts(content.parts()).await?
                        } else {
                            text.clone()
                        };
                        self.editor.add_history(text);
                    }
                    _ => {}
                }
            }
            let output = self.transcript.apply(&event, historical);
            self.output(output)?;
        }
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        let mut activity = self.transcript.activity(self.busy, self.terminal.width());
        if self.shell.is_some() {
            activity.push("Shell running · Esc cancels".into());
        } else if !self.busy {
            activity.push("Ready".into());
        }
        if let Some(worker) = &self.worker {
            let pending = worker.control.pending_input();
            if !pending.is_empty() {
                activity.push(format!(
                    "Queued: {} steering · {} follow-up · Alt+Up to edit",
                    pending.steering().len(),
                    pending.follow_ups().len()
                ));
            }
        }
        let context = self.transcript.context_percent.map_or_else(
            || "context ?".into(),
            |value| format!("context {value:.0}%"),
        );
        let cost = if self.transcript.unknown_cost {
            format!("${:.4}+", self.transcript.cost as f64 / 1_000_000.0)
        } else {
            format!("${:.4}", self.transcript.cost as f64 / 1_000_000.0)
        };
        let reasoning = self.metadata.reasoning.map_or_else(String::new, |effort| {
            format!(
                " · {}",
                serde_json::to_value(effort)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_default()
            )
        });
        let status = format!(
            "{}{reasoning} · {} in / {} out · {cost} · {context}",
            self.metadata.model, self.transcript.input_tokens, self.transcript.output_tokens
        );
        if let Some(login) = &self.login {
            login.draw(&mut self.terminal, &status)?;
        } else if let Some(dialog) = &self.dialog {
            self.terminal
                .draw(&dialog.editor, &status, &[], &dialog.lines())?;
        } else if let Some(menu) = self.menu.as_ref().filter(|menu| !menu.is_completion()) {
            menu.draw(&mut self.terminal, &status)?;
        } else {
            let menu = self
                .menu
                .as_ref()
                .map_or_else(Vec::new, |menu| menu.lines(5));
            self.terminal
                .draw(&self.editor, &status, &activity, &menu)?;
        }
        Ok(())
    }

    async fn key(&mut self, key: KeyEvent) -> Result<()> {
        self.dirty = true;
        if self.login.is_some() {
            return self.login_key(key).await;
        }
        if let Some(dialog) = self.dialog.as_mut() {
            if dialog.key(key) {
                self.dialog.take();
            }
            return Ok(());
        }
        if self.menu.is_some() && self.menu_key(key).await? {
            return Ok(());
        }
        let key = self.settings.remap(key);
        let previous_draft = self.editor.text().to_owned();
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) || control => {
                self.editor.insert("\n");
            }
            KeyCode::Char('j') if control => {
                self.editor.insert("\n");
            }
            KeyCode::Enter => {
                let text = self.editor.take();
                if text.trim().is_empty() {
                    return Ok(());
                }
                if !self.command(&text).await? {
                    self.submit(text, alt).await?;
                }
            }
            KeyCode::Esc => {
                self.operation_cancel.cancel();
                if let Some(shell) = &self.shell {
                    shell.cancel.cancel();
                }
                if let Some(worker) = &self.worker {
                    if worker.control.is_compacting() {
                        worker.control.cancel_compaction();
                    } else {
                        worker.control.abort();
                    }
                }
            }
            KeyCode::Char('c') if control => {
                let now = Instant::now();
                if self
                    .last_interrupt
                    .is_some_and(|previous| now.duration_since(previous) < Duration::from_secs(1))
                {
                    self.quit = true;
                } else if !self.editor.text().is_empty() {
                    self.editor.clear();
                } else {
                    self.terminal.message("Press Ctrl+C again to exit.")?;
                }
                self.last_interrupt = Some(now);
            }
            KeyCode::Char('d') if control && self.editor.text().is_empty() => self.quit = true,
            KeyCode::Char('q') if control => self.quit = true,
            KeyCode::Char('z') if control => self.suspend().await?,
            KeyCode::Char('g') if control => {
                if let Err(error) = self.external_editor().await {
                    self.terminal.message(&format!("{error:#}"))?;
                }
            }
            KeyCode::Char('l') if control => {
                self.command("/model").await?;
            }
            KeyCode::Char('p' | 'P') if control => {
                if let Err(error) = self
                    .cycle_model(
                        key.modifiers.contains(KeyModifiers::SHIFT)
                            || key.code == KeyCode::Char('P'),
                    )
                    .await
                {
                    self.terminal.message(&format!("{error:#}"))?;
                }
            }
            KeyCode::BackTab => {
                if let Err(error) = self.cycle_thinking() {
                    self.terminal.message(&format!("{error:#}"))?;
                }
            }
            KeyCode::Char('o') if control => {
                self.transcript.expand_tools = !self.transcript.expand_tools;
            }
            KeyCode::Char('t') if control => {
                self.transcript.show_reasoning = !self.transcript.show_reasoning;
            }
            KeyCode::Up if alt => {
                if let Some(worker) = &self.worker {
                    let (steering, follow_ups) = worker.control.take_pending_input().into_parts();
                    for input in steering.into_iter().chain(follow_ups) {
                        let restored = self
                            .attachments
                            .render_parts(input.content().parts())
                            .await?;
                        self.editor.restore(&restored);
                    }
                }
            }
            KeyCode::Tab => {
                if let Err(error) = self.complete().await {
                    self.terminal.message(&format!("{error:#}"))?;
                }
            }
            _ => {
                self.last_interrupt = None;
                self.editor.handle(key);
            }
        }
        if self.editor.text() != previous_draft {
            self.refresh_completion().await;
        }
        Ok(())
    }

    async fn finish_shell(&mut self) -> Result<()> {
        if let Some(shell) = self.shell.take() {
            let record = shell.task.await.context("joining the shell command")??;
            self.terminal.message(&format!(
                "Shell: {} · {}",
                record.status,
                if record.include_context {
                    "included in next prompt"
                } else {
                    "excluded from model context"
                }
            ))?;
        }
        self.completion_files = None;
        self.dirty = true;
        Ok(())
    }

    async fn submit(&mut self, text: String, follow_up: bool) -> Result<()> {
        if self.shell.is_some() {
            self.editor.restore(&text);
            self.terminal
                .message("Wait for the shell command or press Esc to cancel it.")?;
            return Ok(());
        }
        if let Some(command) = text.strip_prefix('!') {
            if self.busy {
                self.editor.restore(&text);
                self.terminal
                    .message("Wait for the agent before running a shell command.")?;
                return Ok(());
            }
            let include_context = !command.starts_with('!');
            let command = command.strip_prefix('!').unwrap_or(command).trim();
            if !command.is_empty() {
                match shell::Job::start(
                    command,
                    include_context,
                    self.metadata.clone(),
                    self.store.clone(),
                    self.services.clone(),
                )
                .await
                {
                    Ok(job) => {
                        self.terminal.message(&format!("$ {command}"))?;
                        self.editor.add_history(text);
                        self.shell = Some(job);
                    }
                    Err(error) => {
                        self.editor.restore(&text);
                        self.terminal.message(&format!("{error:#}"))?;
                    }
                }
            }
            self.dirty = true;
            return Ok(());
        }
        if !self.busy {
            let checked = async {
                let route = model_route(&self.client, &self.metadata.model)?;
                self.auth.resolve(route.provider()).await
            }
            .await;
            if let Err(error) = checked {
                self.editor.restore(&text);
                self.terminal.message(&format!("{error:#}"))?;
                self.dirty = true;
                return Ok(());
            }
        }
        let mut parts = match self.attachments.expand(&text).await {
            Ok(parts) => parts,
            Err(error) => {
                self.editor.restore(&text);
                self.terminal.message(&format!("{error:#}"))?;
                return Ok(());
            }
        };
        if !self.busy {
            match shell::context(&self.store).await {
                Ok(mut context) => {
                    context.append(&mut parts);
                    parts = context;
                }
                Err(error) => {
                    self.editor.restore(&text);
                    self.terminal.message(&format!("{error:#}"))?;
                    return Ok(());
                }
            }
        }
        let Some(worker) = &self.worker else {
            self.editor.restore(&text);
            return Ok(());
        };
        if worker.control.is_closed() {
            self.editor.restore(&text);
            self.terminal.message(
                "This agent is closed. Use /resume or /new before sending another prompt.",
            )?;
        } else if self.busy {
            let message = SteeringMessage::from_content(parts);
            let outcome = if follow_up {
                worker.control.queue_follow_up(message)
            } else {
                worker.control.queue_steering(message)
            };
            match outcome {
                SteeringOutcome::Closed => self.editor.restore(&text),
                SteeringOutcome::Evicted(evicted) => {
                    let restored = self
                        .attachments
                        .render_parts(evicted.content().parts())
                        .await?;
                    self.editor.restore(&restored);
                    self.terminal.message(
                        "The input queue is full. The oldest message was restored to the editor.",
                    )?;
                }
                _ => {}
            }
        } else {
            self.operation_cancel = CancellationToken::new();
            if let Err(error) = worker.send(Command::Prompt(
                CodingInput::new(parts),
                self.operation_cancel.clone(),
            )) {
                self.editor.restore(&text);
                return Err(error);
            }
            self.busy = true;
        }
        self.dirty = true;
        Ok(())
    }
}
