//! Application-owned credentials. Files and secret input never enter agent
//! history.

use std::collections::BTreeMap;
use std::io::{self, Read as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{env, fs as sync_fs, thread};

use anyhow::{Context as _, Result, bail};
use async_trait::async_trait;
use lithos_llm::catalog::{AuthScheme, CatalogProvider};
use lithos_llm::credentials::{
    ConventionalCredentials, CredentialError, CredentialHeader, CredentialProvider, Credentials,
    SecretValue,
};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::AsyncReadExt as _;
use tokio::task::spawn_blocking;

use crate::storage;

const MAX_AUTH_BYTES: u64 = 1024 * 1024;
pub(crate) const MAX_KEY_BYTES: usize = 8192;

#[derive(Clone)]
pub(crate) struct AuthStore {
    path: PathBuf,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AuthFile {
    version:   u32,
    providers: BTreeMap<String, Entry>,
}

impl Default for AuthFile {
    fn default() -> Self {
        Self {
            version:   1,
            providers: BTreeMap::new(),
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Entry {
    ApiKey {
        #[serde(with = "secret")]
        key: SecretValue,
    },
    Env {
        variable: String,
    },
    Headers {
        headers: BTreeMap<String, SecretSource>,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum SecretSource {
    Literal {
        #[serde(with = "secret")]
        value: SecretValue,
    },
    Env {
        variable: String,
    },
}

pub(crate) struct ResolvedCredentials {
    pub credentials: Credentials,
    pub source:      String,
}

impl AuthStore {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) async fn snapshot(&self) -> Result<AuthFile> {
        let file = match fs::File::open(&self.path).await {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(AuthFile::default()),
            Err(error) => {
                return Err(error).with_context(|| format!("reading {}", self.path.display()));
            }
        };
        let mut bytes = Vec::new();
        file.take(MAX_AUTH_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .with_context(|| format!("reading {}", self.path.display()))?;
        decode(&bytes, &self.path)
    }

    pub(crate) async fn resolve(&self, provider: &CatalogProvider) -> Result<ResolvedCredentials> {
        self.snapshot().await?.resolve(provider).await.with_context(|| {
            format!("configure credentials with `pebble auth login {}` or the provider's environment variables", provider.id())
        })
    }

    pub(crate) async fn save_key(
        &self,
        provider: &CatalogProvider,
        key: SecretValue,
    ) -> Result<()> {
        key_credentials(provider, key.clone())?;
        self.update(
            provider.id().as_str().to_owned(),
            Some(Entry::ApiKey { key }),
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn remove(&self, provider: &str) -> Result<bool> {
        self.update(provider.to_owned(), None).await
    }

    async fn update(&self, provider: String, entry: Option<Entry>) -> Result<bool> {
        let path = self.path.clone();
        spawn_blocking(move || -> Result<bool> {
            let parent = path
                .parent()
                .context("auth.json needs a parent directory")?;
            storage::private_directory(parent)?;
            let mut options = sync_fs::OpenOptions::new();
            options.create(true).read(true).write(true).truncate(false);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                options.mode(0o600);
            }
            let lock = options.open(parent.join("auth.lock"))?;
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                match fs2::FileExt::try_lock_exclusive(&lock) {
                    Ok(()) => break,
                    Err(error)
                        if error.kind() == io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        thread::sleep(Duration::from_millis(25));
                    }
                    Err(error) => {
                        return Err(error).context(
                            "locking auth.json; another Pebble process may be updating credentials",
                        );
                    }
                }
            }
            // Re-read under the OS lock. A stale in-memory snapshot must not
            // erase another process's credentials. Closing `lock` releases it.
            let mut current = match sync_fs::File::open(&path) {
                Ok(file) => {
                    let mut bytes = Vec::new();
                    file.take(MAX_AUTH_BYTES + 1).read_to_end(&mut bytes)?;
                    decode(&bytes, &path)?
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => AuthFile::default(),
                Err(error) => {
                    return Err(error).with_context(|| format!("reading {}", path.display()));
                }
            };
            let existed = current.providers.remove(&provider).is_some();
            if let Some(entry) = entry {
                current.providers.insert(provider, entry);
            }
            let bytes = serde_json::to_vec_pretty(&current)?;
            anyhow::ensure!(
                bytes.len() as u64 <= MAX_AUTH_BYTES,
                "credentials would exceed the 1 MiB file limit"
            );
            storage::write_atomic(&path, &bytes)?;
            Ok(existed)
        })
        .await
        .context("joining the credential writer")?
    }
}

#[async_trait]
impl CredentialProvider for AuthStore {
    async fn credentials(
        &self,
        provider: &CatalogProvider,
    ) -> Result<Credentials, CredentialError> {
        // The upstream credential boundary has a closed error vocabulary.
        // Detailed file errors are reported by startup, auth status, and the
        // prompt preflight. A file changed mid-turn still fails closed here.
        let file = self
            .snapshot()
            .await
            .map_err(|_| CredentialError::NotConfigured {
                provider: provider.id().clone(),
            })?;
        file.resolve(provider)
            .await
            .map(|resolved| resolved.credentials)
    }
}

impl AuthFile {
    pub(crate) async fn resolve(
        &self,
        provider: &CatalogProvider,
    ) -> Result<ResolvedCredentials, CredentialError> {
        let entry = self.providers.get(provider.id().as_str());
        match entry {
            Some(Entry::Env { variable }) => {
                return Ok(ResolvedCredentials {
                    credentials: key_credentials(
                        provider,
                        environment_secret(provider, variable)?,
                    )?,
                    source:      format!("environment: {variable} (auth.json)"),
                });
            }
            Some(Entry::Headers { headers }) => {
                if !matches!(provider.auth(), AuthScheme::None | AuthScheme::Headers) {
                    return Err(mismatch(provider));
                }
                let mut resolved = Vec::new();
                for (name, source) in headers {
                    let value = match source {
                        SecretSource::Literal { value } => value.clone(),
                        SecretSource::Env { variable } => environment_secret(provider, variable)?,
                    };
                    if !valid_secret(value.expose_secret()) {
                        return Err(mismatch(provider));
                    }
                    resolved.push(CredentialHeader::new(name, value));
                }
                return Ok(ResolvedCredentials {
                    credentials: Credentials::headers(resolved),
                    source:      "explicit headers (auth.json)".into(),
                });
            }
            _ => {}
        }
        if matches!(provider.auth(), AuthScheme::None) {
            return Ok(ResolvedCredentials {
                credentials: Credentials::none(),
                source:      "no authentication required".into(),
            });
        }
        let environment = ConventionalCredentials::new().credentials(provider).await;
        match environment {
            Ok(credentials) => {
                let credentials = adapt_environment_credentials(provider, credentials)?;
                // Validate secrets without formatting either the value or a
                // VarError::NotUnicode, which could disclose an environment value.
                validate_credentials(provider, &credentials)?;
                Ok(ResolvedCredentials {
                    credentials,
                    source: "environment".into(),
                })
            }
            Err(error) => {
                if let Some(Entry::ApiKey { key }) = entry {
                    return Ok(ResolvedCredentials {
                        credentials: key_credentials(provider, key.clone())?,
                        source:      "saved key (auth.json)".into(),
                    });
                }
                Err(error)
            }
        }
    }
}

pub(crate) fn accepts_api_key(provider: &CatalogProvider) -> bool {
    matches!(
        provider.auth(),
        AuthScheme::Bearer { .. } | AuthScheme::Header { .. } | AuthScheme::BedrockBearer
    )
}

fn key_credentials(
    provider: &CatalogProvider,
    key: SecretValue,
) -> Result<Credentials, CredentialError> {
    if !valid_secret(key.expose_secret()) {
        return Err(mismatch(provider));
    }
    match provider.auth() {
        AuthScheme::Bearer { .. } => Ok(Credentials::bearer(key)),
        AuthScheme::Header { name } => Ok(Credentials::header(CredentialHeader::new(name, key))),
        AuthScheme::BedrockBearer => Ok(Credentials::BedrockBearer(key)),
        _ => Err(mismatch(provider)),
    }
}

fn environment_secret(
    provider: &CatalogProvider,
    variable: &str,
) -> Result<SecretValue, CredentialError> {
    env::var(variable)
        .map(SecretValue::new)
        .map_err(|_| CredentialError::MissingSecret {
            provider: provider.id().clone(),
            name:     variable.into(),
        })
}

fn mismatch(provider: &CatalogProvider) -> CredentialError {
    CredentialError::SchemeMismatch {
        provider: provider.id().clone(),
    }
}

fn adapt_environment_credentials(
    provider: &CatalogProvider,
    credentials: Credentials,
) -> Result<Credentials, CredentialError> {
    use lithos_llm::credentials::HttpAuthentication;
    if !accepts_api_key(provider) {
        return Ok(credentials);
    }
    match credentials {
        Credentials::Http(http) => {
            let key = match http.auth {
                HttpAuthentication::Bearer(value) => value,
                HttpAuthentication::Header(header) => header.value,
                _ => return Err(mismatch(provider)),
            };
            let mut resolved = key_credentials(provider, key)?;
            if let Credentials::Http(target) = &mut resolved {
                target.extra_headers = http.extra_headers;
            }
            Ok(resolved)
        }
        other => Ok(other),
    }
}

fn validate_credentials(
    provider: &CatalogProvider,
    credentials: &Credentials,
) -> Result<(), CredentialError> {
    use lithos_llm::credentials::HttpAuthentication;
    let valid = match credentials {
        Credentials::Http(http) => {
            let primary = match &http.auth {
                HttpAuthentication::None => true,
                HttpAuthentication::Bearer(value) => valid_secret(value.expose_secret()),
                HttpAuthentication::Header(header) => valid_secret(header.value.expose_secret()),
                _ => false,
            };
            primary
                && http
                    .extra_headers
                    .iter()
                    .all(|header| valid_secret(header.value.expose_secret()))
        }
        Credentials::BedrockBearer(value) => valid_secret(value.expose_secret()),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(mismatch(provider))
    }
}

pub(crate) fn valid_secret(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_KEY_BYTES
        && value.bytes().all(|byte| (32..127).contains(&byte))
}

fn decode(bytes: &[u8], path: &Path) -> Result<AuthFile> {
    if bytes.len() as u64 > MAX_AUTH_BYTES {
        bail!("{} exceeds 1 MiB", path.display());
    }
    let file: AuthFile = serde_json::from_slice(bytes).map_err(|error| {
        // Serde diagnostics can quote an invalid field's secret value. Keep
        // location and category, but never retain that error as a source.
        anyhow::anyhow!("invalid credentials in {} at line {}, column {} ({:?}); expected version 1 and a providers object", path.display(), error.line(), error.column(), error.classify())
    })?;
    if file.version != 1 {
        bail!(
            "unsupported credential format in {}; expected version 1",
            path.display()
        );
    }
    for entry in file.providers.values() {
        match entry {
            Entry::ApiKey { key } if !valid_secret(key.expose_secret()) => bail!(
                "invalid API key in {}; keys must be nonempty printable ASCII, at most 8192 bytes",
                path.display()
            ),
            Entry::Env { variable } => validate_variable(variable, path)?,
            Entry::Headers { headers } => {
                if headers.is_empty() {
                    bail!("empty credential headers in {}", path.display());
                }
                for (name, source) in headers {
                    if name.is_empty()
                        || !name
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
                    {
                        bail!("invalid credential header name in {}", path.display());
                    }
                    match source {
                        SecretSource::Literal { value } if !valid_secret(value.expose_secret()) => {
                            bail!("invalid credential header value in {}", path.display())
                        }
                        SecretSource::Env { variable } => validate_variable(variable, path)?,
                        SecretSource::Literal { .. } => {}
                    }
                }
            }
            Entry::ApiKey { .. } => {}
        }
    }
    Ok(file)
}

fn validate_variable(variable: &str, path: &Path) -> Result<()> {
    if variable.is_empty()
        || !variable
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        bail!("invalid environment variable name in {}", path.display());
    }
    Ok(())
}

mod secret {
    use lithos_llm::credentials::SecretValue;
    use serde::{Deserialize as _, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        value: &SecretValue,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(value.expose_secret())
    }
    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<SecretValue, D::Error> {
        String::deserialize(deserializer).map(SecretValue::new)
    }
}

#[cfg(test)]
mod tests {
    use lithos_llm::catalog::Catalog;
    use lithos_llm::credentials::HttpAuthentication;
    use tokio::task::JoinSet;

    use super::*;

    fn provider(id: &str, scheme: &str) -> CatalogProvider {
        Catalog::builder()
            .toml_layer(
                "fixture",
                &format!(
                    r#"
schema_version = 1
[providers.{id}]
display_name = "Fixture"
adapter = "openai"
codec = "openai-responses"
base_url = "http://127.0.0.1:1/v1"
auth = {{ {scheme} }}
"#
                ),
            )
            .unwrap()
            .build()
            .unwrap()
            .provider(id)
            .unwrap()
            .clone()
    }

    #[tokio::test]
    async fn saved_keys_follow_the_catalog_authentication_scheme_and_live_file_changes()
    -> Result<()> {
        let root = tempfile::tempdir()?;
        let first = AuthStore::new(root.path().join("auth.json"));
        let second = first.clone();
        let provider = provider("private", "type = 'header', name = 'x-fixture-key'");
        first
            .save_key(&provider, SecretValue::new("first-key"))
            .await?;
        let resolved = second.resolve(&provider).await?;
        let Credentials::Http(http) = resolved.credentials else {
            panic!("HTTP credentials");
        };
        let HttpAuthentication::Header(header) = http.auth else {
            panic!("named header");
        };
        assert_eq!(header.name, "x-fixture-key");
        assert_eq!(header.value.expose_secret(), "first-key");
        first
            .save_key(&provider, SecretValue::new("second-key"))
            .await?;
        assert_eq!(
            second.resolve(&provider).await?.source,
            "saved key (auth.json)"
        );
        assert!(second.remove("private").await?);
        assert!(first.resolve(&provider).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn independent_writers_preserve_each_others_provider_entries() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("home/auth.json");
        let mut writers = JoinSet::new();
        for index in 0..8 {
            let store = AuthStore::new(path.clone());
            let provider = provider(&format!("private_{index}"), "type = 'bearer'");
            writers.spawn(async move {
                store
                    .save_key(&provider, SecretValue::new("fixture-key"))
                    .await
            });
        }
        while let Some(result) = writers.join_next().await {
            result??;
        }
        assert_eq!(
            AuthStore::new(path.clone())
                .snapshot()
                .await?
                .providers
                .len(),
            8
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                sync_fs::metadata(&path)?.permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                sync_fs::metadata(path.parent().unwrap())?
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn invalid_auth_file_is_redacted_and_never_overwritten() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("auth.json");
        let malformed =
            br#"{"version":1,"providers":{"private":{"type":"API-KEY-MUST-NOT-LEAK"}}}"#;
        fs::write(&path, malformed).await?;
        let store = AuthStore::new(path.clone());
        let Err(error) = store.snapshot().await else {
            panic!("invalid entry");
        };
        assert!(format!("{error:#}").contains("line 1"));
        assert!(!format!("{error:?}").contains("API-KEY-MUST-NOT-LEAK"));
        assert!(
            store
                .save_key(
                    &provider("private", "type = 'bearer'"),
                    SecretValue::new("replacement")
                )
                .await
                .is_err()
        );
        assert_eq!(fs::read(path).await?, malformed);
        Ok(())
    }

    #[tokio::test]
    async fn local_providers_need_no_key_and_header_sources_require_a_compatible_scheme()
    -> Result<()> {
        let file = AuthFile::default();
        let local = provider("local", "type = 'none'");
        assert_eq!(file.resolve(&local).await?.credentials, Credentials::none());
        let file = decode(br#"{"version":1,"providers":{"private":{"type":"headers","headers":{"x-proxy-key":{"type":"literal","value":"proxy-secret"}}}}}"#, Path::new("fixture.json"))?;
        let headers = provider("private", "type = 'headers'");
        let resolved = file.resolve(&headers).await?;
        assert_eq!(
            resolved.credentials,
            Credentials::headers([CredentialHeader::new(
                "x-proxy-key",
                SecretValue::new("proxy-secret")
            )])
        );
        assert!(!format!("{:?}", resolved.credentials).contains("proxy-secret"));
        assert!(
            file.resolve(&provider("private", "type = 'bearer'"))
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_missing_explicit_environment_source_is_an_error() -> Result<()> {
        let variable = format!("PEBBLE_TEST_{}", uuid::Uuid::new_v4().simple());
        let bytes = serde_json::to_vec(
            &serde_json::json!({"version":1,"providers":{"private":{"type":"env","variable":variable}}}),
        )?;
        let file = decode(&bytes, Path::new("fixture.json"))?;
        let error = file
            .resolve(&provider("private", "type = 'bearer'"))
            .await
            .err()
            .context("missing source")?;
        assert!(error.to_string().contains(&variable));
        Ok(())
    }

    #[test]
    fn control_characters_and_oversized_keys_are_rejected_without_echoing_them() {
        for key in ["", "   ", "key\nheader", "key\rheader", "🔑"] {
            assert!(!valid_secret(key));
        }
        assert!(!valid_secret(&"a".repeat(MAX_KEY_BYTES + 1)));
        assert!(valid_secret("fixture-key_123"));
    }
}
