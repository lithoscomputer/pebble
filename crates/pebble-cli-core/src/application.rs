//! Application-owned client configuration shared by both command modes.

use std::path::{Path, PathBuf};
use std::{env, fmt, io};

use anyhow::{Context as _, Result};
use clap::ValueEnum;
use lithos_llm::Client;
use lithos_llm::catalog::{Catalog, CatalogModel, CatalogProvider};
use lithos_llm::client::{ClientBuild, ProviderBuildCause};
use lithos_llm::credentials::{self, ConventionalCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::resolver::ResolvedRoute;
use lithos_llm::types::Request;
use pebble_coding_agent::events::{AgentProfileKind, RetryEventObserver};
use pebble_coding_agent::tools::PermissionLevel;
use tokio::fs;

use crate::credentials::AuthStore;
use crate::settings::Settings;
use crate::terminal::print_err;

pub(crate) const DEFAULT_MODEL: &str = "claude-sonnet-5";
const BASE_URL_ENV_PREFIX: &str = "PEBBLE_";
const BASE_URL_ENV_SUFFIX: &str = "_BASE_URL";

/// The permission ladder, as a flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, serde::Serialize, serde::Deserialize)]
pub(crate) enum PermissionArg {
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

impl fmt::Display for PermissionArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ReadOnly => "read-only",
            Self::ReadWrite => "read-write",
            Self::Full => "full",
        })
    }
}

pub(crate) struct Paths {
    pub home: PathBuf,
}

impl Paths {
    pub(crate) fn from_env() -> Result<Self> {
        let home = env::var_os("PEBBLE_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("HOME")
                    .filter(|value| !value.is_empty())
                    .map(|home| PathBuf::from(home).join(".pebble"))
            })
            .context("set HOME or PEBBLE_HOME for Pebble configuration")?;
        Ok(Self { home })
    }

    pub(crate) fn settings(&self) -> PathBuf {
        self.home.join("settings.json")
    }
    pub(crate) fn models(&self) -> PathBuf {
        self.home.join("models.toml")
    }
    pub(crate) fn auth(&self) -> PathBuf {
        self.home.join("auth.json")
    }
    pub(crate) fn sessions(&self) -> PathBuf {
        self.home.join("sessions")
    }
}

pub(crate) struct Application {
    pub paths:    Paths,
    pub settings: Settings,
    pub client:   Client,
    pub auth:     AuthStore,
}

impl Application {
    pub(crate) async fn load(policy: RetryPolicy) -> Result<Self> {
        let paths = Paths::from_env()?;
        let settings = Settings::load(&paths.settings()).await?;
        let auth = AuthStore::new(paths.auth());
        auth.snapshot().await?;
        let layer = optional_catalog(&paths.models()).await?;
        let client = build_client(policy, &paths.models(), layer.as_deref(), auth.clone())?;
        Ok(Self {
            paths,
            settings,
            client,
            auth,
        })
    }
}

async fn optional_catalog(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path).await {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

fn build_client(
    policy: RetryPolicy,
    path: &Path,
    layer: Option<&str>,
    auth: AuthStore,
) -> Result<Client> {
    let mut catalog = Catalog::builder().with_builtin();
    if let Some(layer) = layer {
        catalog = catalog
            .toml_layer(path.display().to_string(), layer)
            .with_context(|| format!("loading {}", path.display()))?;
    }
    for (variable, provider, base_url) in base_url_overrides() {
        catalog = catalog
            .toml_layer(&variable, &base_url_layer(&provider, &base_url))
            .with_context(|| format!("applying {variable}"))?;
    }
    let catalog = catalog.build().context("loading the model catalog")?;
    let ClientBuild { client, issues, .. } = Client::builder()
        .catalog(catalog)
        .application("pebble")
        .credentials(auth)
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

pub(crate) fn model_route(client: &Client, selector: &str) -> Result<ResolvedRoute> {
    let request = Request::builder().model(selector).user("probe").build()?;
    let route = client
        .resolve_route(&request)
        .context("selecting the model")?;
    check_profile(route.provider(), route.model())?;
    Ok(route)
}

pub(crate) struct ModelChoice {
    pub display_name: String,
    pub selector:     String,
    pub unavailable:  Option<String>,
}

pub(crate) async fn model_choices(client: &Client, auth: &AuthStore) -> Result<Vec<ModelChoice>> {
    let snapshot = auth.snapshot().await?;
    let readiness = credentials::readiness(client.catalog().providers(), &snapshot).await;
    let mut choices = Vec::new();
    for provider in client.catalog().providers() {
        let status = if readiness.is_ready(provider.id()) {
            None
        } else if let Some(issue) = readiness.issue(provider.id()) {
            Some(issue.to_string())
        } else {
            Some(not_configured_hint(provider))
        };
        for model in provider.models() {
            let unavailable = if !client.available_providers().contains(provider.id()) {
                Some("provider adapter unavailable in this build".into())
            } else if let Err(error) = check_profile(provider, model) {
                Some(error.to_string())
            } else {
                status.clone()
            };
            let selector = format!("{}/{}", provider.id(), model.id());
            choices.push(ModelChoice {
                display_name: model.display_name().into(),
                selector,
                unavailable,
            });
        }
    }
    Ok(choices)
}

/// The operator-facing line for a provider the credential store holds
/// nothing for. `readiness` reports such a provider as silence, so the hint
/// names what would configure it.
fn not_configured_hint(provider: &CatalogProvider) -> String {
    let names = ConventionalCredentials::new().secret_names(provider);
    if names.is_empty() {
        format!("run `pebble auth login {}`", provider.id())
    } else {
        format!(
            "set {} or run `pebble auth login {}`",
            names.join(" or "),
            provider.id()
        )
    }
}

fn check_profile(provider: &CatalogProvider, model: &CatalogModel) -> Result<()> {
    // Match the coding layer's per-field metadata precedence and parse the
    // public profile vocabulary. The worker still performs final validation.
    #[derive(Default, serde::Deserialize)]
    struct ProfileMetadata {
        profile:               Option<String>,
        #[serde(rename = "reasoning_by_default")]
        _reasoning_by_default: Option<bool>,
    }
    let model = model
        .metadata()
        .namespace::<ProfileMetadata>("agent")
        .context("invalid model metadata.agent")?
        .unwrap_or_default();
    let provider = provider
        .metadata()
        .namespace::<ProfileMetadata>("agent")
        .context("invalid provider metadata.agent")?
        .unwrap_or_default();
    let profile = model
        .profile
        .or(provider.profile)
        .context("missing metadata.agent.profile")?;
    anyhow::ensure!(
        AgentProfileKind::ALL
            .iter()
            .any(|kind| kind.as_str() == profile),
        "unsupported coding profile: {profile}"
    );
    Ok(())
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
/// limits, pricing, and harness; only where requests go changes. Pointing a
/// provider somewhere is opting into it, so the layer also enables providers
/// that ship disabled.
fn base_url_layer(provider: &str, base_url: &str) -> String {
    format!(
        "schema_version = 1\n[providers.{}]\nbase_url = {}\nenabled = true\n",
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn model_choices_explain_credentials_adapters_and_profiles_without_requests() -> Result<()>
    {
        let home = tempfile::tempdir()?;
        let auth = AuthStore::new(home.path().join("auth.json"));
        let catalog = Catalog::builder()
            .toml_layer(
                "fixture",
                r#"
schema_version = 1
[providers.local]
display_name = "Local"
adapter = "openai"
codec = "openai-responses"
base_url = "http://127.0.0.1:1/v1"
auth = { type = "none" }
metadata.agent.profile = "openai"
[providers.local.models.ready]
display_name = "Ready"
api_model = "ready"
capabilities = { text = true, tools = true }
knowledge_cutoff = "Fixture date"
[providers.local.models.unsupported]
display_name = "Unsupported"
api_model = "unsupported"
metadata.agent.profile = "future-profile"
[providers.local.models.malformed]
display_name = "Malformed"
api_model = "malformed"
metadata.agent.profile = 123
[providers.private]
display_name = "Private"
adapter = "openai"
codec = "openai-responses"
base_url = "http://127.0.0.1:1/v1"
auth = { type = "bearer" }
metadata.agent.profile = "openai"
[providers.private.models.locked]
display_name = "Locked"
api_model = "locked"
[providers.unavailable]
display_name = "Unavailable"
adapter = "not-installed"
codec = "openai-responses"
base_url = "http://127.0.0.1:1/v1"
auth = { type = "none" }
metadata.agent.profile = "openai"
[providers.unavailable.models.disabled]
display_name = "Disabled"
api_model = "disabled"
"#,
            )?
            .build()?;
        let ClientBuild { client, .. } = Client::builder()
            .catalog(catalog)
            .credentials(auth.clone())
            .build()?;
        let choices = model_choices(&client, &auth).await?;
        let ready: Vec<_> = choices
            .iter()
            .filter(|choice| choice.unavailable.is_none())
            .map(|choice| choice.selector.as_str())
            .collect();
        assert_eq!(ready, ["local/ready"]);
        for (selector, reason) in [
            ("local/unsupported", "unsupported coding profile"),
            ("local/malformed", "invalid model metadata"),
            ("private/locked", "PRIVATE_API_KEY"),
            ("unavailable/disabled", "adapter unavailable"),
        ] {
            let choice = choices
                .iter()
                .find(|choice| choice.selector == selector)
                .unwrap();
            assert!(choice.unavailable.as_ref().unwrap().contains(reason));
        }
        assert!(model_route(&client, "local/ready").is_ok());
        assert!(model_route(&client, "local/unsupported").is_err());
        Ok(())
    }
}
