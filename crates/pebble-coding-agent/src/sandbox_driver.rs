//! A sandbox-driver handle as the [`Environment`] a session runs in.
//!
//! Pebble's tools speak the [`Environment`] contract; the sandbox driver
//! speaks facets: exec, filesystem, search, preview URLs. This module is the
//! mapping between the two, for an application whose sandboxes come from the
//! driver: [`SandboxEnvironment`] resolves paths against the working
//! directory the application names (which may sit below the provider's own)
//! and runs commands under [`SandboxExec`]'s policy; [`SandboxPortRoutes`]
//! answers pebble's port routing with the driver's preview URLs;
//! [`display_for_log`] renders a driver failure with its cause chain and the
//! redacted tail of the output a failed command left behind.
//!
//! The sandbox itself is the application's: whoever brought it to `Running`
//! owns starting and disposing of it, as with every environment. Pebble
//! ships no secret detector, so the places here that render process output
//! for a log or an event take the application's [`Redactor`].
//!
//! [`Environment`]: crate::environment::Environment
//! [`Redactor`]: crate::extensions::Redactor

mod environment;
mod exec;
mod log;
mod path;
#[cfg(feature = "mcp")]
mod ports;

#[cfg(any(test, feature = "sandbox-driver-test-util"))]
pub mod test_support;

pub use self::environment::SandboxEnvironment;
pub use self::exec::{
    DEFAULT_RETAINED_OUTPUT_BYTES, DEFAULT_STOP_GRACE, ExecResultExt, SandboxExec,
    command_termination, program_exit_code, redacted_output_tail,
};
pub use self::log::{default_redacted_output_tail, display_for_log};
pub use self::path::{join_sandbox_path, resolve_path};
#[cfg(feature = "mcp")]
pub use self::ports::{SandboxPortRoutes, port_routes};

/// A redactor that masks one known token, standing in for the secret
/// scanner an application installs.
#[cfg(test)]
mod testing {
    use std::borrow::Cow;

    use crate::redact::Redactor;

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
}
