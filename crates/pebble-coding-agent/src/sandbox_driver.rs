//! A sandbox-driver handle as the [`Environment`] a session runs in.
//!
//! Pebble's tools speak the [`Environment`] contract; sandbox-driver's
//! `Sandbox` speaks files, ports, and exec. This module is the mapping
//! between the two, for an application whose sandboxes come from the
//! driver's `Sandboxes` catalog: [`SandboxEnvironment`] resolves paths
//! against the working directory the application names (which may sit below
//! the provider's own) and runs commands under [`SandboxExec`]'s policy;
//! [`SandboxPortRoutes`] answers pebble's port routing with the driver's
//! port exposures; [`display_for_log`] renders a failure with its cause chain
//! and the redacted tail of the output a failed command left behind.
//!
//! The sandbox itself is the application's: whoever brought it to `Running`
//! owns starting and disposing of it, as with every environment. Pebble
//! ships no secret detector, so the places here that render process output
//! for a log or an event take the application's [`Redactor`].
//!
//! # Parity gaps with the old driver API
//!
//! sandbox-driver's new API is smaller than the one this module was written
//! against. What Pebble needs and the new API lacks is done here, over exec,
//! until the library carries it; each place says so in its own docs:
//!
//! - the `TERM` then `KILL` stop grace ([`SandboxExec`], which captures through
//!   the driver's `run_captured_with`, runs scripts with `ExecSpec::bash`, and
//!   applies the driver's [`OutputSanitizer`] to the output it shows);
//! - stat, exists, rename, remove, mkdir (in [`SandboxEnvironment`]);
//! - grep and file walks (the derived search, run as commands);
//! - platform information (`uname -sr` at [`SandboxEnvironment::attach`]);
//! - command output on errors ([`CommandFailure`]);
//! - whether a handle can route ports (the descriptor's `ports`, passed to
//!   [`port_routes`]).
//!
//! [`Environment`]: crate::environment::Environment
//! [`Redactor`]: crate::extensions::Redactor

mod environment;
mod exec;
mod log;
mod path;
#[cfg(feature = "mcp")]
mod ports;
mod search;

#[cfg(any(test, feature = "sandbox-driver-test-util"))]
pub mod test_support;

pub use sandbox_driver::{OutputSanitization, OutputSanitizer};

pub use self::environment::SandboxEnvironment;
pub use self::exec::{
    BASH_ENV_VAR, CommandFailure, DEFAULT_RETAINED_OUTPUT_BYTES, DEFAULT_STOP_GRACE, ExecResult,
    ExecStreamingResult, SandboxExec, bash_source, command_termination, program_exit_code,
    redacted_output_tail, stdin_bytes,
};
pub use self::log::{default_redacted_output_tail, display_for_log};
pub use self::path::{join_sandbox_path, resolve_path};
#[cfg(feature = "mcp")]
pub use self::ports::{SandboxPortRoutes, port_routes};

/// Shared fixtures for this module's tests.
#[cfg(test)]
mod testing {
    use std::borrow::Cow;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use sandbox_driver::{
        CatalogOptions, Environment, EnvironmentName, Error, ProviderEntry, ProviderKind,
        ProviderName, Registry, Sandbox, SandboxSpec, Sandboxes, SecretResolver, Source,
        StaticEnvironments,
    };
    use sandbox_driver_host::HostFactory;

    use crate::redact::Redactor;

    /// A redactor that masks one known token, standing in for the secret
    /// scanner an application installs.
    pub(super) struct MaskToken(pub(super) &'static str);

    impl Redactor for MaskToken {
        fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
            if text.contains(self.0) {
                Cow::Owned(text.replace(self.0, "REDACTED"))
            } else {
                Cow::Borrowed(text)
            }
        }
    }

    struct NoSecrets;

    #[async_trait]
    impl SecretResolver for NoSecrets {
        async fn resolve(&self, _: &str) -> Result<Option<String>, Error> {
            Ok(None)
        }
    }

    /// A sandbox from the driver's built-in Host provider, through the
    /// `Sandboxes` catalog, in a temporary directory the provider manages.
    /// Keep the catalog alive as long as the sandbox.
    pub(super) async fn host_sandbox() -> (Sandboxes, Arc<dyn Sandbox>) {
        let entry: ProviderEntry =
            serde_json::from_value(serde_json::json!({"plugin": "host"})).expect("entry");
        let provider = ProviderName::try_new("local").expect("name");
        let environments = BTreeMap::from([(
            EnvironmentName::try_new("scratch").expect("environment"),
            Environment {
                provider: provider.clone(),
                spec:     SandboxSpec::new(Source::HostDirectory),
            },
        )]);
        let registry = Registry::new()
            .builtin(
                ProviderKind::try_new("host").expect("kind"),
                Arc::new(HostFactory),
            )
            .expect("registry");
        let catalog = Sandboxes::try_new(
            [(provider, entry)],
            Arc::new(StaticEnvironments(environments)),
            &registry,
            Arc::new(NoSecrets),
            CatalogOptions::default(),
        )
        .expect("catalog");
        let sandbox = catalog
            .builder("scratch")
            .await
            .expect("builder")
            .create()
            .await
            .expect("a host sandbox");
        (catalog, Arc::from(sandbox))
    }
}
