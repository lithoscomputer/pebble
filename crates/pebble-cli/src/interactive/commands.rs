//! Slash commands and small searchable pickers inside the live area.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context as _, Result, bail};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lithos_llm::Client;
use lithos_llm::types::ReasoningEffort;
#[cfg(unix)]
use rustix::process::{Signal, getpid, kill_process};
use tokio::fs;
use tokio::io::AsyncWriteExt as _;
use tokio::process::Command as ProcessCommand;
use tokio_util::sync::CancellationToken;

use super::input::Input;
use super::setup::Login;
use super::{App, Command, Store, Transcript, Worker, text};
use crate::application::{model_choices, model_route};
use crate::credentials::{AuthStore, accepts_api_key};
use crate::secret_input::Action;
use crate::storage;

const COMMANDS: &[(&str, &str)] = &[
    ("/help", "Commands and keyboard shortcuts"),
    ("/new", "Start a new session"),
    ("/resume", "Choose a saved session"),
    ("/name", "Name this session"),
    ("/session", "Session details"),
    ("/model", "Choose a model"),
    ("/login", "Save a provider API key"),
    ("/logout", "Remove saved provider credentials"),
    ("/thinking", "Choose reasoning effort"),
    ("/compact", "Compact model context"),
    ("/tools", "Inspect saved tool output"),
    ("/agents", "Subagent activity"),
    ("/skills", "Available skills"),
    ("/copy", "Copy the last answer"),
    ("/export", "Export the transcript"),
    ("/editor", "Edit the prompt externally"),
    ("/attach", "Attach an image to the prompt"),
    ("/settings", "Saved preferences and keybindings"),
    ("/suspend", "Suspend and return to the shell"),
    ("/quit", "Save and exit"),
];

pub(super) enum Purpose {
    Command,
    Completion(usize),
    Sessions,
    Models,
    Login,
    Logout,
    Tools,
    Agents,
    Thinking,
    Settings,
}

pub(super) struct Menu {
    pub purpose: Purpose,
    items:       Vec<(String, String)>,
    query:       String,
    selected:    usize,
}

impl Menu {
    pub(super) fn new(purpose: Purpose, items: Vec<(String, String)>) -> Self {
        Self {
            purpose,
            items,
            query: String::new(),
            selected: 0,
        }
    }

    fn matches(&self) -> Vec<&(String, String)> {
        self.items
            .iter()
            .filter(|(label, _)| fuzzy(&self.query, label))
            .collect()
    }

    pub(super) fn lines(&self) -> Vec<String> {
        let mut lines = vec![format!("Search: {}", self.query)];
        let matches = self.matches();
        let selected = self.selected.min(matches.len().saturating_sub(1));
        for (index, (label, _)) in matches
            .iter()
            .enumerate()
            .skip(selected.saturating_sub(3))
            .take(5)
        {
            lines.push(format!(
                "{} {label}",
                if index == selected { "›" } else { " " }
            ));
        }
        if matches.is_empty() {
            lines.push("No matches".into());
        }
        lines.push("↑/↓ choose · Enter selects · Esc closes".into());
        lines
    }

    pub(super) fn key(&mut self, key: KeyEvent) -> MenuAction {
        match key.code {
            KeyCode::Esc => MenuAction::Close,
            KeyCode::Char('c' | 'q' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                MenuAction::Close
            }
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                MenuAction::Editing
            }
            KeyCode::Down | KeyCode::Tab => {
                self.selected = (self.selected + 1).min(self.matches().len().saturating_sub(1));
                MenuAction::Editing
            }
            KeyCode::Backspace => {
                self.query.pop();
                self.selected = 0;
                MenuAction::Editing
            }
            KeyCode::Char(value)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.query.push(value);
                self.selected = 0;
                MenuAction::Editing
            }
            KeyCode::Enter => self
                .matches()
                .get(self.selected)
                .map_or(MenuAction::Editing, |(_, value)| {
                    MenuAction::Select(value.clone())
                }),
            _ => MenuAction::Unhandled,
        }
    }
}

pub(super) enum MenuAction {
    Editing,
    Unhandled,
    Close,
    Select(String),
}

pub(super) async fn model_menu(client: &Client, auth: &AuthStore, all: bool) -> Result<Menu> {
    let choices = model_choices(client, auth).await?;
    let mut items: Vec<_> = choices
        .into_iter()
        .filter(|choice| all || choice.unavailable.is_none())
        .map(|choice| (choice.label, choice.selector))
        .collect();
    items.push(("Set up a provider API key".into(), "@login".into()));
    items.push(if all {
        ("Show configured models".into(), "@configured".into())
    } else {
        (
            "Show all models and setup requirements".into(),
            "@all".into(),
        )
    });
    Ok(Menu::new(Purpose::Models, items))
}

pub(super) fn provider_menu(client: &Client, purpose: Purpose) -> Menu {
    let login = matches!(purpose, Purpose::Login);
    let items = client
        .catalog()
        .providers()
        .filter(|provider| {
            !login
                || (accepts_api_key(provider)
                    && client.available_providers().contains(provider.id()))
        })
        .map(|provider| {
            (
                format!("{} · {}", provider.display_name(), provider.id()),
                provider.id().as_str().to_owned(),
            )
        })
        .collect();
    Menu::new(purpose, items)
}

impl App {
    pub(super) async fn command(&mut self, input: &str) -> Result<bool> {
        let input = input.trim();
        if !input.starts_with('/') {
            return Ok(false);
        }
        let (command, argument) = input
            .split_once(char::is_whitespace)
            .map_or((input, ""), |(command, value)| (command, value.trim()));
        if let Some(name) = command.strip_prefix("/skill:") {
            self.submit(format!("/{name} {argument}"), false).await?;
            return Ok(true);
        }
        if !COMMANDS.iter().any(|(name, _)| *name == command) {
            // The coding layer owns slash-skill expansion and ordinary path inputs.
            return Ok(false);
        }
        let result = self.dispatch(command, argument).await;
        if let Err(error) = result {
            self.terminal.message(&format!("{error:#}"))?;
        }
        self.dirty = true;
        Ok(true)
    }

    async fn dispatch(&mut self, command: &str, argument: &str) -> Result<()> {
        match command {
            "/help" => {
                self.terminal.message(
                    &COMMANDS
                        .iter()
                        .map(|(name, description)| format!("{name:<12} {description}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                )?;
                self.terminal.message("\nEnter: send/steer · Alt+Enter: follow-up · Shift+Enter or Ctrl+J: newline\nEsc: cancel · Ctrl+G: editor · Ctrl+O: live tools · Ctrl+T: reasoning\nTab: completion · Alt+Up: recover queued input · Ctrl+_: undo\nCtrl+Z: suspend · Ctrl+C: clear; press twice to quit · Ctrl+D: quit with an empty editor")?;
            }
            "/quit" => {
                if !argument.is_empty() {
                    bail!("/quit takes no arguments");
                }
                self.quit = true;
            }
            "/session" => {
                self.terminal.message(&format!(
                    "{}\nSession: {}\nDirectory: {}\nModel: {}\nSaved in: {}\nLast event: {}",
                    self.metadata.name,
                    self.metadata.id,
                    self.metadata.cwd.display(),
                    self.metadata.model,
                    self.store.directory().display(),
                    self.transcript.cursor
                ))?;
            }
            "/name" => {
                self.require_idle()?;
                if argument.is_empty() {
                    bail!("usage: /name <name>");
                }
                self.worker()?.send(Command::Name(argument.into()))?;
                self.busy = true;
            }
            "/compact" => {
                self.require_idle()?;
                self.operation_cancel = CancellationToken::new();
                self.worker()?.send(Command::Compact(
                    argument.into(),
                    self.operation_cancel.clone(),
                ))?;
                self.busy = true;
            }
            "/new" => {
                self.require_idle()?;
                self.change_session(None, None).await?;
            }
            "/resume" if !argument.is_empty() => {
                self.require_idle()?;
                self.change_session(Some(argument), None).await?;
            }
            "/resume" => {
                self.require_idle()?;
                let items = Store::list(&self.root)
                    .await?
                    .into_iter()
                    .map(|metadata| {
                        (
                            format!(
                                "{} · {} · {}",
                                metadata.name,
                                metadata.cwd.display(),
                                metadata.id
                            ),
                            metadata.id,
                        )
                    })
                    .collect();
                self.menu = Some(Menu::new(Purpose::Sessions, items));
            }
            "/login" => {
                self.require_idle()?;
                if argument.is_empty() {
                    self.menu = Some(provider_menu(&self.client, Purpose::Login));
                } else {
                    let provider = self
                        .client
                        .catalog()
                        .provider(argument)
                        .context("unknown provider; use /login to choose a provider")?;
                    anyhow::ensure!(
                        accepts_api_key(provider),
                        "this provider needs explicit credential headers or a different authentication scheme"
                    );
                    self.login = Some(Login::new(provider.id().as_str(), provider.display_name()));
                }
            }
            "/logout" => {
                self.require_idle()?;
                if argument.is_empty() {
                    self.menu = Some(provider_menu(&self.client, Purpose::Logout));
                } else {
                    let provider = self.client.catalog().provider(argument).ok();
                    let id = provider.map_or(argument, |provider| provider.id().as_str());
                    let removed = self.auth.remove(id).await?;
                    self.terminal.message(if removed {
                        "Removed saved credentials."
                    } else {
                        "No saved credentials to remove."
                    })?;
                    if let Some(provider) = provider
                        && let Ok(resolved) = self.auth.resolve(provider).await
                    {
                        self.terminal
                            .message(&format!("Still configured through {}.", resolved.source))?;
                    }
                }
            }
            "/model" if argument == "all" => {
                self.require_idle()?;
                self.menu = Some(model_menu(&self.client, &self.auth, true).await?);
            }
            "/model" if !argument.is_empty() => {
                self.require_idle()?;
                let route = model_route(&self.client, argument)?;
                self.auth.resolve(route.provider()).await?;
                self.change_session(Some(&self.metadata.id.clone()), Some(argument))
                    .await?;
            }
            "/model" => {
                self.require_idle()?;
                self.menu = Some(model_menu(&self.client, &self.auth, false).await?);
            }
            "/thinking" if !argument.is_empty() => {
                self.require_idle()?;
                let effort = if argument == "default" {
                    None
                } else {
                    Some(
                        serde_json::from_value::<ReasoningEffort>(serde_json::Value::String(
                            argument.into(),
                        ))
                        .context("choose default, minimal, low, medium, high, xhigh, or max")?,
                    )
                };
                self.worker()?.send(Command::Reasoning(effort))?;
                self.busy = true;
            }
            "/thinking" => {
                self.menu = Some(Menu::new(
                    Purpose::Thinking,
                    [
                        "default", "minimal", "low", "medium", "high", "xhigh", "max",
                    ]
                    .into_iter()
                    .map(|value| (value.into(), value.into()))
                    .collect(),
                ));
            }
            "/tools" if !argument.is_empty() => self.tool_details(argument).await?,
            "/tools" => {
                self.menu = Some(Menu::new(
                    Purpose::Tools,
                    self.transcript
                        .tools
                        .iter()
                        .rev()
                        .map(|tool| {
                            (
                                format!(
                                    "{} · {} · {}",
                                    tool.name,
                                    if tool.failed {
                                        "failed"
                                    } else if tool.complete {
                                        "done"
                                    } else {
                                        "running"
                                    },
                                    text::truncate(&tool.arguments, 90)
                                ),
                                format!("{}:{}", tool.session, tool.id),
                            )
                        })
                        .collect(),
                ));
            }
            "/agents" if !argument.is_empty() => self.agent_details(argument).await?,
            "/agents" => {
                if self.transcript.agents.is_empty() {
                    self.terminal.message("No subagents in this session.")?;
                }
                self.menu = Some(Menu::new(
                    Purpose::Agents,
                    self.transcript
                        .agents
                        .iter()
                        .map(|(id, state)| (format!("{id} · {state}"), id.clone()))
                        .collect(),
                ));
            }
            "/skills" => {
                let skills = self.worker()?.snapshot.skills().to_vec();
                if skills.is_empty() {
                    self.terminal.message("No skills found for this session.")?;
                }
                for skill in skills {
                    self.terminal
                        .message(&format!("/skill:{} — {}", skill.name, skill.description))?;
                }
            }
            "/attach" => {
                if argument.is_empty() {
                    bail!("usage: /attach <image path>");
                }
                let marker = self
                    .attachments
                    .attach_image(&self.metadata.cwd.join(argument))
                    .await?;
                self.editor.insert(&marker);
                self.terminal.message(
                    "Image attached. Add a prompt, or delete the placeholder to remove it.",
                )?;
            }
            "/copy" => self.copy_answer().await?,
            "/editor" => self.external_editor().await?,
            "/suspend" => self.suspend().await?,
            "/settings" if argument.is_empty() => {
                self.terminal
                    .message(&format!("Preferences: {}", self.settings_path.display()))?;
                self.menu = Some(Menu::new(Purpose::Settings, vec![
                    (
                        format!("Use {} at startup", self.metadata.model),
                        "model".into(),
                    ),
                    (
                        "Use the current reasoning level at startup".into(),
                        "reasoning".into(),
                    ),
                    (
                        format!(
                            "{} reasoning by default",
                            if self.transcript.show_reasoning {
                                "Hide"
                            } else {
                                "Show"
                            }
                        ),
                        "show-reasoning".into(),
                    ),
                    (
                        format!(
                            "{} live tool details by default",
                            if self.transcript.expand_tools {
                                "Hide"
                            } else {
                                "Show"
                            }
                        ),
                        "tools".into(),
                    ),
                ]));
            }
            "/settings" => {
                match argument {
                    "model" => self.settings.model = Some(self.metadata.model.clone()),
                    "reasoning" => self.settings.reasoning = self.metadata.reasoning,
                    "show-reasoning" => {
                        self.transcript.show_reasoning = !self.transcript.show_reasoning;
                        self.settings.show_reasoning = self.transcript.show_reasoning;
                    }
                    "tools" => {
                        self.transcript.expand_tools = !self.transcript.expand_tools;
                        self.settings.expand_tools = self.transcript.expand_tools;
                    }
                    _ => bail!("Use /settings to choose a preference."),
                }
                self.settings.save(&self.settings_path).await?;
                self.terminal.message("Saved preferences.")?;
            }
            "/export" => self.export(argument).await?,
            _ => {}
        }
        Ok(())
    }

    fn worker(&self) -> Result<&Worker> {
        self.worker
            .as_ref()
            .context("the coding agent is unavailable")
    }
    fn require_idle(&self) -> Result<()> {
        if self.busy {
            bail!("Wait for the current work to finish, or press Esc to cancel it.");
        }
        Ok(())
    }

    pub(super) async fn menu_key(&mut self, key: KeyEvent) -> Result<bool> {
        let Some(menu) = self.menu.as_mut() else {
            return Ok(false);
        };
        match menu.key(key) {
            MenuAction::Close => {
                self.menu = None;
            }
            MenuAction::Editing => {}
            MenuAction::Unhandled => return Ok(false),
            MenuAction::Select(value) => {
                let purpose = self
                    .menu
                    .take()
                    .expect("the open menu is still present")
                    .purpose;
                match purpose {
                    Purpose::Completion(start) => {
                        self.editor.replace_token(start, &format!("{value} "));
                    }
                    Purpose::Command => self.editor.set(format!("{value} ")),
                    Purpose::Sessions => {
                        self.command(&format!("/resume {value}")).await?;
                    }
                    Purpose::Models => match value.as_str() {
                        "@login" => {
                            self.command("/login").await?;
                        }
                        "@all" => {
                            self.menu = Some(model_menu(&self.client, &self.auth, true).await?);
                        }
                        "@configured" => {
                            self.menu = Some(model_menu(&self.client, &self.auth, false).await?);
                        }
                        _ => {
                            self.command(&format!("/model {value}")).await?;
                        }
                    },
                    Purpose::Login => {
                        self.command(&format!("/login {value}")).await?;
                    }
                    Purpose::Logout => {
                        self.command(&format!("/logout {value}")).await?;
                    }
                    Purpose::Thinking => {
                        self.command(&format!("/thinking {value}")).await?;
                    }
                    Purpose::Settings => {
                        self.command(&format!("/settings {value}")).await?;
                    }
                    Purpose::Tools => {
                        self.command(&format!("/tools {value}")).await?;
                    }
                    Purpose::Agents => {
                        self.command(&format!("/agents {value}")).await?;
                    }
                }
            }
        }
        Ok(true)
    }

    pub(super) async fn login_key(&mut self, key: KeyEvent) -> Result<()> {
        let Some(login) = self.login.as_mut() else {
            return Ok(());
        };
        match login.input.key(key) {
            Action::Editing => {}
            Action::Cancel => {
                self.login = None;
                self.terminal.message("Login cancelled.")?;
            }
            Action::Submit(key) => {
                let provider = self
                    .client
                    .catalog()
                    .provider(&login.provider)
                    .context("provider no longer exists")?;
                match self.auth.save_key(provider, key).await {
                    Ok(()) => {
                        self.login = None;
                        self.terminal.message("Saved provider credentials.")?;
                        match self.auth.resolve(provider).await {
                            Ok(resolved) => self
                                .terminal
                                .message(&format!("Active source: {}", resolved.source))?,
                            Err(error) => self.terminal.message(&format!("{error:#}"))?,
                        }
                    }
                    Err(error) => self.terminal.message(&format!("{error:#}"))?,
                }
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub(super) async fn complete(&mut self) -> Result<()> {
        let (start, token) = self.editor.token();
        let token = token.to_owned();
        if start == 0 && token.starts_with('/') {
            let mut items: Vec<_> = COMMANDS
                .iter()
                .filter(|(name, _)| name.starts_with(&token))
                .map(|(name, description)| (format!("{name} — {description}"), (*name).into()))
                .collect();
            for skill in self.worker()?.snapshot.skills() {
                let command = format!("/skill:{}", skill.name);
                if command.starts_with(&token) {
                    items.push((format!("{command} — {}", skill.description), command));
                }
            }
            self.menu = Some(Menu::new(Purpose::Command, items));
        } else if let Some(query) = token.strip_prefix('@') {
            let output = ProcessCommand::new("git")
                .args([
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ])
                .current_dir(&self.metadata.cwd)
                .kill_on_drop(true)
                .output()
                .await
                .context("listing project files")?;
            if !output.status.success() {
                bail!("File mentions need a Git repository; use path completion here.");
            }
            let mut items: Vec<_> = String::from_utf8_lossy(&output.stdout)
                .split('\0')
                .filter(|path| !path.is_empty() && fuzzy(query, path))
                .take(1000)
                .map(|path| (path.to_owned(), format!("@{path}")))
                .collect();
            items.sort();
            items.dedup();
            self.menu = Some(Menu::new(Purpose::Completion(start), items));
        } else {
            let path = PathBuf::from(&token);
            let parent = if token.ends_with('/') {
                path.clone()
            } else {
                path.parent().unwrap_or_else(|| Path::new("")).to_path_buf()
            };
            let prefix = if token.ends_with('/') {
                ""
            } else {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
            };
            let mut directory = fs::read_dir(self.metadata.cwd.join(&parent))
                .await
                .context("listing path completions")?;
            let mut items = Vec::new();
            while let Some(entry) = directory.next_entry().await? {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !name.starts_with(prefix) {
                    continue;
                }
                let mut value = parent.join(name).to_string_lossy().into_owned();
                if entry.file_type().await?.is_dir() {
                    value.push('/');
                }
                items.push((value.clone(), value));
                if items.len() == 1000 {
                    break;
                }
            }
            items.sort();
            self.menu = Some(Menu::new(Purpose::Completion(start), items));
        }
        Ok(())
    }

    pub(super) async fn external_editor(&mut self) -> Result<()> {
        let file = tempfile::Builder::new().suffix(".md").tempfile()?;
        fs::write(file.path(), self.editor.text()).await?;
        let editor = self
            .settings
            .external_editor
            .clone()
            .or_else(|| env::var("VISUAL").ok())
            .or_else(|| env::var("EDITOR").ok())
            .unwrap_or_else(|| "vi".into());
        if let Some(input) = self.input.take() {
            input.shutdown().await?;
        }
        self.terminal.pause()?;
        // The configured editor is shell syntax. The path is a positional argument,
        // never interpolated.
        let status = ProcessCommand::new("sh")
            .arg("-c")
            .arg(format!("exec {editor} \"$1\""))
            .arg("pebble-editor")
            .arg(file.path())
            .status()
            .await;
        let resumed = self.terminal.resume();
        self.input = Some(Input::start());
        resumed?;
        if status.context("starting the external editor")?.success() {
            if fs::metadata(file.path()).await?.len() > 1024 * 1024 {
                bail!("Edited prompt exceeds 1 MiB. The original draft is still available.");
            }
            self.editor.set(fs::read_to_string(file.path()).await?);
        } else {
            self.terminal.message(
                "The editor exited unsuccessfully; the original draft is still available.",
            )?;
        }
        self.dirty = true;
        Ok(())
    }

    pub(super) async fn suspend(&mut self) -> Result<()> {
        #[cfg(unix)]
        {
            if let Some(input) = self.input.take() {
                input.shutdown().await?;
            }
            self.terminal.pause()?;
            let stopped = kill_process(getpid(), Signal::STOP);
            let resumed = self.terminal.resume();
            self.input = Some(Input::start());
            stopped.context("suspending Pebble")?;
            resumed?;
            self.dirty = true;
        }
        #[cfg(not(unix))]
        self.terminal
            .message("Suspend is supported on Unix terminals.")?;
        Ok(())
    }

    async fn copy_answer(&mut self) -> Result<()> {
        if self.transcript.last_answer.is_empty() {
            bail!("There is no assistant answer to copy yet.");
        }
        let commands: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
            &[("pbcopy", &[])]
        } else {
            &[("wl-copy", &[]), ("xclip", &["-selection", "clipboard"])]
        };
        for (program, arguments) in commands {
            let child = ProcessCommand::new(program)
                .args(*arguments)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .kill_on_drop(true)
                .spawn();
            let Ok(mut child) = child else {
                continue;
            };
            if let Some(mut input) = child.stdin.take() {
                input
                    .write_all(self.transcript.last_answer.as_bytes())
                    .await?;
            }
            if child.wait().await?.success() {
                self.terminal.message("Copied the last answer.")?;
                return Ok(());
            }
        }
        bail!("No clipboard command is available. Use terminal selection or /export.")
    }

    async fn tool_details(&mut self, selected: &str) -> Result<()> {
        use pebble_coding_agent::events::CodingEvent;
        let mut reader = self.store.reader().await?;
        let mut found = false;
        while let Some(event) = reader.next().await? {
            let id = match &event.event {
                CodingEvent::ToolCallStarted { tool_call_id, .. }
                | CodingEvent::ToolCallCompleted { tool_call_id, .. } => {
                    Some(tool_call_id.as_str())
                }
                _ => event.tool_call_id.as_deref(),
            };
            let Some(id) = id else {
                continue;
            };
            if selected != id && selected != format!("{}:{id}", event.session_id) {
                continue;
            }
            match &event.event {
                CodingEvent::ToolCallStarted {
                    tool_name,
                    arguments,
                    ..
                } => {
                    self.terminal.message(&format!(
                        "\n{tool_name} · {id}\n{}",
                        serde_json::to_string_pretty(&arguments)?
                    ))?;
                    found = true;
                }
                CodingEvent::ToolCallCompleted {
                    output,
                    output_bytes_omitted,
                    ..
                } => {
                    self.terminal
                        .message(output.as_str().unwrap_or(&output.to_string()))?;
                    if *output_bytes_omitted > 0 {
                        self.terminal.message(&format!(
                            "[{output_bytes_omitted} bytes were not retained by the tool]"
                        ))?;
                    }
                }
                CodingEvent::ToolCallOutputDelta { delta } => self.terminal.message(delta)?,
                _ => {}
            }
        }
        if !found {
            bail!("No saved tool call matches {selected}.");
        }
        Ok(())
    }

    async fn agent_details(&mut self, id: &str) -> Result<()> {
        let mut reader = self.store.reader().await?;
        let mut projection = Transcript::new(id);
        projection.show_reasoning = self.transcript.show_reasoning;
        while let Some(event) = reader.next().await? {
            let output = projection.apply(&event, true);
            self.output(output)?;
        }
        Ok(())
    }

    async fn export(&mut self, argument: &str) -> Result<()> {
        let path = if argument.is_empty() {
            self.store.directory().join("transcript.md")
        } else {
            self.metadata.cwd.join(argument)
        };
        let bytes = if path
            .extension()
            .is_some_and(|extension| extension == "jsonl")
        {
            fs::read(self.store.directory().join("events.jsonl")).await?
        } else {
            let mut reader = self.store.reader().await?;
            let mut projection = Transcript::new(&self.transcript.root);
            let mut markdown = format!("# {}\n\n", self.metadata.name);
            while let Some(event) = reader.next().await? {
                for output in projection.apply(&event, true) {
                    match output {
                        super::Output::Text(text) | super::Output::Markdown(text) => {
                            markdown.push_str(&text::plain(&text));
                            markdown.push_str("\n\n");
                        }
                    }
                }
            }
            markdown.into_bytes()
        };
        storage::atomic_write(path.clone(), bytes).await?;
        self.terminal
            .message(&format!("Exported {}", path.display()))?;
        Ok(())
    }

    async fn change_session(&mut self, id: Option<&str>, model: Option<&str>) -> Result<()> {
        let same = id == Some(self.metadata.id.as_str());
        let store = if same {
            self.store.clone()
        } else {
            Store::open(&self.root, id).await?
        };
        let mut metadata = self.metadata.clone();
        let mut record = if id.is_some() {
            if same {
                Some(self.worker()?.record().await?)
            } else {
                let checkpoint = store.load().await?;
                metadata = checkpoint.metadata;
                Some(checkpoint.record)
            }
        } else {
            metadata.id = store.id();
            metadata.name = "New session".into();
            metadata.forked_from = None;
            None
        };
        let mut attachments = super::attachments::Attachments::open(store.directory()).await?;
        let draft = if same {
            None
        } else {
            Some(
                attachments
                    .render_parts(&self.attachments.expand(self.editor.text()).await?)
                    .await?,
            )
        };
        let previous_model = metadata.model.clone();
        if let Some(model) = model {
            metadata.model = model.into();
        }
        if let Some(worker) = self.worker.take() {
            worker.shutdown().await?;
        }
        if let Some(record) = record.as_mut() {
            record.advance_event_cursor(store.last_sequence().await?);
        }
        let worker = Worker::start(
            self.client.clone(),
            metadata.clone(),
            store.clone(),
            record.clone(),
            self.services.clone(),
        )
        .await;
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                self.terminal
                    .message(&format!("Could not switch sessions: {error:#}"))?;
                // Restore the prior checkpoint and model so a failed selection leaves a usable
                // UI.
                metadata = self.metadata.clone();
                metadata.model = if same { previous_model } else { metadata.model };
                let mut checkpoint = self.store.load().await?;
                checkpoint
                    .record
                    .advance_event_cursor(self.store.last_sequence().await?);
                let worker = Worker::start(
                    self.client.clone(),
                    metadata,
                    self.store.clone(),
                    Some(checkpoint.record),
                    self.services.clone(),
                )
                .await?;
                self.worker = Some(worker);
                return Ok(());
            }
        };
        metadata.model = format!("{}/{}", worker.snapshot.provider(), worker.snapshot.model());
        self.metadata = metadata;
        self.store = store;
        self.attachments = attachments;
        if !same {
            self.transcript = Transcript::new(worker.snapshot.session_id());
            self.transcript.show_reasoning = self.settings.show_reasoning;
            self.transcript.expand_tools = self.settings.expand_tools;
            self.editor = super::Editor::default();
            if let Some(draft) = draft {
                self.editor.set(draft);
            }
        }
        self.worker = Some(worker);
        self.busy = false;
        self.header()?;
        self.replay(!same).await?;
        Ok(())
    }
}

fn fuzzy(query: &str, candidate: &str) -> bool {
    let candidate = candidate.to_lowercase();
    let mut chars = candidate.chars();
    query
        .to_lowercase()
        .chars()
        .all(|needle| chars.by_ref().any(|character| character == needle))
}
