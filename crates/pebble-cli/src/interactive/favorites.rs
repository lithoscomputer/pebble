//! A global model shortlist that leaves the full model picker available.

use anyhow::{Result, bail};
use lithos_llm::Client;

use super::App;
use super::menu::{Menu, Purpose};
use crate::application::{ModelChoice, model_choices, model_route};

fn selectors(client: &Client, saved: &[String]) -> Vec<String> {
    let mut selectors = Vec::new();
    for selector in saved {
        let selector = model_route(client, selector)
            .map_or_else(|_| selector.clone(), |route| route.handle().to_string());
        if !selectors.contains(&selector) {
            selectors.push(selector);
        }
    }
    selectors
}

pub(super) fn cycling_choices(
    client: &Client,
    favorites: &[String],
    choices: Vec<ModelChoice>,
) -> Vec<ModelChoice> {
    if favorites.is_empty() {
        return choices
            .into_iter()
            .filter(|choice| choice.unavailable.is_none())
            .collect();
    }
    let favorites = selectors(client, favorites);
    let mut choices: Vec<_> = choices
        .into_iter()
        .filter_map(|choice| {
            if choice.unavailable.is_some() {
                return None;
            }
            favorites
                .iter()
                .position(|selector| selector == &choice.selector)
                .map(|index| (index, choice))
        })
        .collect();
    choices.sort_by_key(|(index, _)| *index);
    choices.into_iter().map(|(_, choice)| choice).collect()
}

impl App {
    pub(super) async fn favorites(&mut self, argument: &str) -> Result<()> {
        if argument.is_empty() {
            self.menu = Some(Menu::new(Purpose::Favorites, self.favorite_items().await?));
        } else {
            self.toggle_favorite(argument).await?;
            self.terminal.message(&format!("Saved {} favorite models. Ctrl+P uses this shortlist; an empty list uses all configured models.", self.settings.favorite_models.len()))?;
        }
        Ok(())
    }

    pub(super) async fn favorite_items(&self) -> Result<Vec<(String, String)>> {
        let favorites = selectors(&self.client, &self.settings.favorite_models);
        let choices = model_choices(&self.client, &self.auth).await?;
        let mut items = vec![(
            "Clear favorites and cycle all configured models".into(),
            "clear".into(),
        )];
        for selector in &favorites {
            if !choices.iter().any(|choice| &choice.selector == selector) {
                items.push((
                    format!("[x] {selector} · unavailable in this catalog"),
                    selector.clone(),
                ));
            }
        }
        items.extend(choices.into_iter().map(|choice| {
            let mark = if favorites.contains(&choice.selector) {
                "x"
            } else {
                " "
            };
            let status = choice
                .unavailable
                .map_or_else(String::new, |reason| format!(" · {reason}"));
            (
                format!(
                    "[{mark}] {} · {}{status}",
                    choice.selector, choice.display_name
                ),
                choice.selector,
            )
        }));
        Ok(items)
    }

    pub(super) async fn toggle_favorite(&mut self, argument: &str) -> Result<()> {
        let mut settings = self.settings.clone();
        let mut favorites = selectors(&self.client, &settings.favorite_models);
        if argument == "clear" {
            favorites.clear();
        } else {
            let selector = model_route(&self.client, argument)
                .map_or_else(|_| argument.to_owned(), |route| route.handle().to_string());
            if favorites.contains(&selector) {
                favorites.retain(|value| value != &selector);
            } else {
                // Unknown saved entries can be removed; new entries must resolve.
                let route = model_route(&self.client, argument)?;
                if favorites.len() >= 256 {
                    bail!("Keep at most 256 favorite models.");
                }
                favorites.push(route.handle().to_string());
            }
        }
        settings.favorite_models = favorites;
        settings.save(&self.settings_path).await?;
        self.settings = settings;
        Ok(())
    }
}
