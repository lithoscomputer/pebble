//! First-run model selection and credential setup before an agent starts.

use std::path::Path;

use anyhow::{Context as _, Result};
use crossterm::event::{Event, KeyEventKind};
use lithos_llm::Client;
use tokio::signal::ctrl_c;
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};

use super::commands::{model_menu, provider_menu};
use super::menu::{MenuAction, Purpose};
use super::{Editor, Input, Terminal};
use crate::application::{model_choices, model_route};
use crate::credentials::{AuthStore, accepts_api_key};
use crate::secret_input::{Action, SecretInput};
use crate::settings::Settings;

pub(super) struct Login {
    pub provider: String,
    name:         String,
    pub input:    SecretInput,
}

impl Login {
    pub(super) fn new(provider: &str, name: &str) -> Self {
        Self {
            provider: provider.into(),
            name:     name.into(),
            input:    SecretInput::default(),
        }
    }

    pub(super) fn draw(&self, terminal: &mut Terminal, status: &str) -> Result<()> {
        // The prompt editor sees only mask characters, never the credential.
        let mut masked = Editor::default();
        masked.set(self.input.masked());
        terminal.draw(&masked, status, &[], &[
            format!("API key for {} (masked)", self.name),
            "Enter saves · Esc cancels · Ctrl+U clears".into(),
        ])?;
        Ok(())
    }
}

pub(super) async fn choose(
    client: &Client,
    auth: &AuthStore,
    settings: &mut Settings,
    settings_path: &Path,
    requested: Option<&str>,
    terminal: &mut Terminal,
    input: &mut Input,
) -> Result<Option<String>> {
    let required = requested
        .map(|selector| model_route(client, selector))
        .transpose()?;
    let mut login = None;
    if let Some(route) = &required {
        match auth.resolve(route.provider()).await {
            Ok(_) => return Ok(Some(route.handle().to_string())),
            Err(error) => {
                terminal.message(&format!("Cannot use {} yet: {error:#}", route.handle()))?;
                anyhow::ensure!(
                    accepts_api_key(route.provider()),
                    "set this provider's environment variables or explicit credential headers before resuming"
                );
                login = Some(Login::new(
                    route.provider().id().as_str(),
                    route.provider().display_name(),
                ));
            }
        }
    } else {
        terminal.message("Choose a default model. Pebble will save your choice.")?;
    }
    let mut menu = if model_choices(client, auth)
        .await?
        .iter()
        .any(|choice| choice.unavailable.is_none())
    {
        model_menu(client, auth, false, requested, settings.model.as_deref()).await?
    } else {
        terminal.message("No configured model is available. Choose a provider to save an API key, or set its environment variable and restart Pebble.")?;
        provider_menu(client, Purpose::Login)
    };
    #[cfg(unix)]
    let mut terminate = signal(SignalKind::terminate())?;
    #[cfg(unix)]
    let mut hangup = signal(SignalKind::hangup())?;
    let shutdown = async {
        #[cfg(unix)]
        tokio::select! { _ = terminate.recv() => {}, _ = hangup.recv() => {}, _ = ctrl_c() => {} }
        #[cfg(not(unix))]
        {
            let _ = ctrl_c().await;
        }
    };
    tokio::pin!(shutdown);
    loop {
        if let Some(login) = &login {
            login.draw(terminal, "Provider setup")?;
        } else {
            menu.draw(terminal, "Model setup · Type to search")?;
        }
        let event = tokio::select! {
            () = &mut shutdown => return Ok(None),
            event = input.recv() => event,
        };
        let Some(event) = event else {
            return Ok(None);
        };
        match event.context("reading setup input")? {
            Event::Resize(width, height) => terminal.resize(width, height),
            Event::Paste(text) => {
                if let Some(login) = &mut login {
                    if !login.input.paste(&text) {
                        terminal.message("API key must be printable ASCII, at most 8192 bytes.")?;
                    }
                } else {
                    menu.paste(&text);
                }
            }
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                if let Some(form) = &mut login {
                    match form.input.key(key) {
                        Action::Editing => {}
                        Action::Cancel => {
                            if required.is_some() {
                                return Ok(None);
                            }
                            login = None;
                            menu = provider_menu(client, Purpose::Login);
                        }
                        Action::Submit(key) => {
                            let provider = client.catalog().provider(&form.provider)?;
                            let result = async {
                                auth.save_key(provider, key).await?;
                                auth.resolve(provider).await
                            }
                            .await;
                            match result {
                                Ok(resolved) => {
                                    terminal.message(&format!(
                                        "Saved provider credentials. Active source: {}.",
                                        resolved.source
                                    ))?;
                                    if let Some(route) = &required {
                                        return Ok(Some(route.handle().to_string()));
                                    }
                                    login = None;
                                    menu = model_menu(
                                        client,
                                        auth,
                                        false,
                                        requested,
                                        settings.model.as_deref(),
                                    )
                                    .await?;
                                }
                                Err(error) => terminal.message(&format!("{error:#}"))?,
                            }
                        }
                    }
                    continue;
                }
                match menu.key(key) {
                    MenuAction::Close => {
                        if matches!(menu.purpose, Purpose::Login) {
                            menu = model_menu(
                                client,
                                auth,
                                false,
                                requested,
                                settings.model.as_deref(),
                            )
                            .await?;
                        } else {
                            return Ok(None);
                        }
                    }
                    MenuAction::Select(value) | MenuAction::SaveDefault(value) => {
                        if matches!(menu.purpose, Purpose::Login) {
                            let provider = client.catalog().provider(&value)?;
                            login =
                                Some(Login::new(provider.id().as_str(), provider.display_name()));
                            continue;
                        }
                        match value.as_str() {
                            "@login" => menu = provider_menu(client, Purpose::Login),
                            "@all" => {
                                menu = model_menu(
                                    client,
                                    auth,
                                    true,
                                    requested,
                                    settings.model.as_deref(),
                                )
                                .await?;
                            }
                            "@configured" => {
                                menu = model_menu(
                                    client,
                                    auth,
                                    false,
                                    requested,
                                    settings.model.as_deref(),
                                )
                                .await?;
                            }
                            _ => {
                                let result = async {
                                    let route = model_route(client, &value)?;
                                    auth.resolve(route.provider()).await?;
                                    Ok::<_, anyhow::Error>(route.handle().to_string())
                                }
                                .await;
                                match result {
                                    Ok(selector) => {
                                        settings.model = Some(selector.clone());
                                        settings.save(settings_path).await?;
                                        return Ok(Some(selector));
                                    }
                                    Err(error) => terminal.message(&format!("{error:#}"))?,
                                }
                            }
                        }
                    }
                    MenuAction::Editing | MenuAction::Unhandled | MenuAction::ToggleFavorite(_) => {
                    }
                }
            }
            _ => {}
        }
    }
}
