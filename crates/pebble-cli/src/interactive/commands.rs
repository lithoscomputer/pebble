//! Slash commands and small searchable pickers inside the live area.

use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use crossterm::event::KeyEvent;
use lithos_llm::Client;
use lithos_llm::types::{ContentPart, ReasoningEffort};
use pebble_coding_agent::events::CodingEvent;
#[cfg(unix)]
use rustix::process::{Signal, getpid, kill_process};
use tokio::fs;
use tokio::io::AsyncWriteExt as _;
use tokio::process::Command as ProcessCommand;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use super::input::Input;
use super::menu::{Menu, MenuAction, Purpose, score};
use super::setup::Login;
use super::transcript::ToolRecord;
use super::{App, Command, Store, Transcript, Worker, text, tool_render};
use crate::application::{model_choices, model_route};
use crate::credentials::{AuthStore, accepts_api_key};
use crate::secret_input::Action;
use crate::storage;

const COMMANDS: &[(&str, &str)] = &[
    ("/help", "Commands and keyboard shortcuts"),
    ("/new", "Start a new session"),
    ("/fork", "Branch before an earlier prompt, or at @bookmark"),
    ("/clone", "Copy the current session into a new branch"),
    (
        "/tree",
        "Navigate branches and saved conversation boundaries",
    ),
    (
        "/bookmark",
        "Name the current boundary, or choose a bookmark",
    ),
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
    (
        "/shells",
        "Show shell results; /shells clear drops pending context",
    ),
    ("/export", "Export the transcript"),
    ("/editor", "Edit the prompt externally"),
    ("/paste", "Paste a clipboard image (Ctrl+V)"),
    ("/attach", "Attach an image to the prompt"),
    ("/settings", "Saved preferences and keybindings"),
    ("/suspend", "Suspend and return to the shell"),
    ("/quit", "Save and exit"),
];

pub(super) async fn model_menu(
    client: &Client,
    auth: &AuthStore,
    all: bool,
    current: Option<&str>,
    default: Option<&str>,
) -> Result<Menu> {
    let default = default
        .and_then(|value| model_route(client, value).ok())
        .map(|route| route.handle().to_string());
    let mut choices = model_choices(client, auth).await?;
    choices.sort_by_key(|choice| {
        (
            current != Some(choice.selector.as_str()),
            default.as_deref() != Some(choice.selector.as_str()),
        )
    });
    let mut items: Vec<_> = choices
        .into_iter()
        .filter(|choice| all || choice.unavailable.is_none())
        .map(|choice| {
            let current_mark = if current == Some(choice.selector.as_str()) {
                " · current"
            } else {
                ""
            };
            let default_mark = if default.as_deref() == Some(choice.selector.as_str()) {
                " · default"
            } else {
                ""
            };
            (
                format!(
                    "{}{current_mark}{default_mark} · {}{}",
                    choice.selector,
                    choice.display_name,
                    choice
                        .unavailable
                        .map_or_else(String::new, |reason| format!(" · {reason}"))
                ),
                choice.selector,
            )
        })
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
            "/shells" => {
                if argument == "clear" {
                    self.require_idle()?;
                    super::shell::clear(&self.store).await?;
                    self.terminal.message("Pending shell context cleared.")?;
                } else if argument.is_empty() {
                    for record in super::shell::records(&self.store).await? {
                        self.terminal.message(&record.display())?;
                    }
                } else {
                    bail!("usage: /shells [clear]");
                }
            }
            "/help" => {
                self.terminal.message(
                    &COMMANDS
                        .iter()
                        .map(|(name, description)| format!("{name:<12} {description}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                )?;
                self.terminal.message("\n!command: shell result in next prompt · !!command: shell without model context\nEnter: send/steer · Alt+Enter: follow-up · Shift+Enter or Ctrl+J: newline\nEsc: cancel · Ctrl+G: editor · Ctrl+V: paste image · Ctrl+O: live tools · Ctrl+T: reasoning\nTab: completion · Ctrl+L: models · Ctrl+P: next model · Shift+Tab: thinking\nAlt+Up: recover queued input · Ctrl+_: undo\nCtrl+Z: suspend · Ctrl+C: clear; press twice to quit · Ctrl+D: quit with an empty editor")?;
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
            "/fork" => {
                self.require_idle()?;
                self.fork_session(argument).await?;
            }
            "/clone" => {
                self.require_idle()?;
                if !argument.is_empty() {
                    bail!("/clone takes no arguments");
                }
                self.clone_session().await?;
            }
            "/tree" => {
                self.require_idle()?;
                self.tree().await?;
            }
            "/bookmark" => {
                self.require_idle()?;
                self.bookmark(argument).await?;
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
                self.menu = Some(
                    model_menu(
                        &self.client,
                        &self.auth,
                        true,
                        Some(&self.metadata.model),
                        self.settings.model.as_deref(),
                    )
                    .await?,
                );
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
                self.menu = Some(
                    model_menu(
                        &self.client,
                        &self.auth,
                        false,
                        Some(&self.metadata.model),
                        self.settings.model.as_deref(),
                    )
                    .await?,
                );
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
                self.start_image(super::images::Source::File(
                    self.metadata.cwd.join(argument),
                ))?;
            }
            "/paste" => self.start_image(super::images::Source::Clipboard)?,
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

    pub(super) fn worker(&self) -> Result<&Worker> {
        self.worker
            .as_ref()
            .context("the coding agent is unavailable")
    }
    fn require_idle(&self) -> Result<()> {
        if self.busy || self.shell.is_some() || self.image_job.is_some() {
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
            MenuAction::SaveDefault(value) => {
                self.menu = None;
                if !value.starts_with('@') {
                    self.command(&format!("/model {value}")).await?;
                    if self.metadata.model == value {
                        self.command("/settings model").await?;
                    }
                }
            }
            MenuAction::Select(value) => {
                let purpose = self
                    .menu
                    .take()
                    .expect("the open menu is still present")
                    .purpose;
                match purpose {
                    Purpose::Completion(start) => {
                        let suffix = if value.ends_with('/') { "" } else { " " };
                        self.editor
                            .replace_token(start, &format!("{value}{suffix}"));
                    }
                    Purpose::Command => self.editor.replace_token(0, &format!("{value} ")),
                    Purpose::Navigate => {
                        Box::pin(self.command(&value)).await?;
                    }
                    Purpose::Sessions => {
                        self.command(&format!("/resume {value}")).await?;
                    }
                    Purpose::Models => match value.as_str() {
                        "@login" => {
                            self.command("/login").await?;
                        }
                        "@all" => {
                            self.menu = Some(
                                model_menu(
                                    &self.client,
                                    &self.auth,
                                    true,
                                    Some(&self.metadata.model),
                                    self.settings.model.as_deref(),
                                )
                                .await?,
                            );
                        }
                        "@configured" => {
                            self.menu = Some(
                                model_menu(
                                    &self.client,
                                    &self.auth,
                                    false,
                                    Some(&self.metadata.model),
                                    self.settings.model.as_deref(),
                                )
                                .await?,
                            );
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

    pub(super) async fn refresh_completion(&mut self) {
        if self.menu.as_ref().is_some_and(|menu| !menu.is_completion()) {
            return;
        }
        let (start, token) = self.editor.token();
        if (start == 0 && token.starts_with('/')) || token.starts_with('@') {
            // Automatic suggestions are optional. Explicit Tab reports failures.
            if self.complete().await.is_err() {
                self.menu = None;
            }
        } else {
            self.menu = None;
        }
    }

    pub(super) async fn complete(&mut self) -> Result<()> {
        let (start, token) = self.editor.token();
        let token = token.to_owned();
        let mut menu = if start == 0 && token.starts_with('/') {
            let mut items: Vec<_> = COMMANDS
                .iter()
                .map(|(name, description)| (format!("{name} — {description}"), (*name).into()))
                .collect();
            for skill in self.worker()?.snapshot.skills() {
                let command = format!("/skill:{}", skill.name);
                items.push((format!("{command} — {}", skill.description), command));
            }
            let mut menu = Menu::new(Purpose::Command, items);
            menu.filter(&token);
            menu
        } else if let Some(query) = token.strip_prefix('@') {
            if self.completion_files.is_none() {
                let output = timeout(
                    Duration::from_secs(2),
                    ProcessCommand::new("git")
                        .args([
                            "ls-files",
                            "--cached",
                            "--others",
                            "--exclude-standard",
                            "-z",
                        ])
                        .current_dir(&self.metadata.cwd)
                        .kill_on_drop(true)
                        .output(),
                )
                .await
                .context("project file listing timed out")?
                .context("listing project files")?;
                if !output.status.success() {
                    bail!("File mentions need a Git repository; use path completion here.");
                }
                let mut files: Vec<_> = String::from_utf8_lossy(&output.stdout)
                    .split('\0')
                    .filter(|path| !path.is_empty())
                    .map(str::to_owned)
                    .collect();
                files.sort();
                files.dedup();
                self.completion_files = Some(files);
            }
            let mut matches: Vec<_> = self
                .completion_files
                .iter()
                .flatten()
                .filter_map(|path| score(query, path).map(|rank| (rank, path)))
                .collect();
            matches.sort_by_key(|(rank, _)| *rank);
            let items = matches
                .into_iter()
                .take(1000)
                .map(|(_, path)| (path.clone(), format!("@{path}")))
                .collect();
            Menu::new(Purpose::Completion(start), items)
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
            Menu::new(Purpose::Completion(start), items)
        };
        // Keep ordinary typing in the prompt; Tab inserts the selected item.
        if matches!(menu.purpose, Purpose::Command) {
            menu.filter(&token);
        }
        self.menu = Some(menu);
        Ok(())
    }

    pub(super) async fn cycle_model(&mut self, reverse: bool) -> Result<()> {
        self.require_idle()?;
        let choices: Vec<_> = model_choices(&self.client, &self.auth)
            .await?
            .into_iter()
            .filter(|choice| choice.unavailable.is_none())
            .collect();
        if choices.is_empty() {
            bail!("No configured models. Use /login to configure a provider.");
        }
        let index = choices
            .iter()
            .position(|choice| choice.selector == self.metadata.model)
            .unwrap_or(0);
        let next = if reverse {
            (index + choices.len() - 1) % choices.len()
        } else {
            (index + 1) % choices.len()
        };
        self.command(&format!("/model {}", choices[next].selector))
            .await?;
        Ok(())
    }

    pub(super) fn cycle_thinking(&mut self) -> Result<()> {
        self.require_idle()?;
        let route = model_route(&self.client, &self.metadata.model)?;
        let capabilities = route.model().capabilities();
        let mut levels = vec![None];
        for level in [
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max,
        ] {
            if !capabilities.reasoning_effort(level).is_unsupported() {
                levels.push(Some(level));
            }
        }
        if levels.len() == 1 {
            bail!("This model does not support reasoning effort.");
        }
        let index = levels
            .iter()
            .position(|level| *level == self.metadata.reasoning)
            .unwrap_or(0);
        self.worker()?
            .send(Command::Reasoning(levels[(index + 1) % levels.len()]))?;
        self.busy = true;
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
        let mut reader = self.store.reader().await?;
        let mut found = false;
        let mut call = None;
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
                    self.terminal.message(&format!("\nTool call {id}"))?;
                    call = Some(ToolRecord {
                        id:        id.into(),
                        session:   event.session_id.clone(),
                        name:      tool_name.clone(),
                        arguments: arguments
                            .as_str()
                            .map_or_else(|| arguments.to_string(), str::to_owned),
                        output:    String::new(),
                        complete:  false,
                        failed:    false,
                    });
                    found = true;
                }
                CodingEvent::ToolCallCompleted {
                    output,
                    output_bytes_omitted,
                    is_error,
                    ..
                } => {
                    if let Some(mut tool) = call.take() {
                        tool.complete = true;
                        tool.failed = *is_error;
                        tool.output = output
                            .as_str()
                            .map_or_else(|| output.to_string(), str::to_owned);
                        self.output(tool_render::result(&tool, None, *output_bytes_omitted))?;
                    }
                }
                // Delta chunks are retained independently of the final tool result.
                CodingEvent::ToolCallOutputDelta { delta } => self.terminal.message(delta)?,
                _ => {}
            }
        }
        if let Some(tool) = call {
            self.terminal
                .message(&tool_render::heading(&tool.name, &tool.arguments))?;
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
                        super::Output::Code { source, language } => {
                            let longest = source
                                .lines()
                                .map(|line| {
                                    line.trim_start().chars().take_while(|c| *c == '`').count()
                                })
                                .max()
                                .unwrap_or(0);
                            let fence = "`".repeat((longest + 1).max(3));
                            let language: String = language
                                .chars()
                                .filter(char::is_ascii_alphanumeric)
                                .collect();
                            write!(
                                markdown,
                                "{fence}{language}\n{}\n{fence}\n\n",
                                text::plain(&source)
                            )?;
                        }
                        super::Output::Text(text) | super::Output::Markdown(text) => {
                            markdown.push_str(&text::plain(&text));
                            markdown.push_str("\n\n");
                        }
                    }
                }
                if let CodingEvent::UserInput {
                    content: Some(content),
                    ..
                }
                | CodingEvent::SteeringInjected {
                    content: Some(content),
                    ..
                } = &event.event
                {
                    for part in content.parts() {
                        if let ContentPart::Image(image) = part
                            && let Some(image) = super::images::markdown(image)?
                        {
                            markdown.push_str(&image);
                        }
                    }
                }
            }
            for record in super::shell::records(&self.store).await? {
                // Indented code remains literal even when shell output contains fences.
                markdown.push('\n');
                for line in text::plain(&record.display()).lines() {
                    writeln!(markdown, "    {line}")?;
                }
            }
            markdown.into_bytes()
        };
        storage::atomic_write(path.clone(), bytes).await?;
        self.terminal
            .message(&format!("Exported {}", path.display()))?;
        Ok(())
    }

    pub(super) async fn change_session(
        &mut self,
        id: Option<&str>,
        model: Option<&str>,
    ) -> Result<()> {
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
                super::shell::reconcile(&store).await?;
                let checkpoint = store.load().await?;
                metadata = checkpoint.metadata;
                Some(checkpoint.record)
            }
        } else {
            metadata.id = store.id();
            metadata.name = "New session".into();
            metadata.forked_from = None;
            metadata.forked_at = None;
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
        self.completion_files = None;
        self.header()?;
        self.replay(!same).await?;
        Ok(())
    }
}
