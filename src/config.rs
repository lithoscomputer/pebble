//! How an application configures one session.
//!
//! [`SessionOptions`] is a plain record with a [`Default`]: build one with
//! `..Default::default()` and set only what differs. Everything that needs a
//! decision from the application rather than a value — which tools may run,
//! what happens around each call — arrives as a trait object on the same
//! record.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::{ReasoningEffort, Speed};
use serde_json::Value;

use crate::truncation::{
    DEFAULT_TOOL_OUTPUT_RETENTION_BYTES, DEFAULT_TOOL_OUTPUT_SERIALIZED_BYTES,
};
use crate::types::{AgentProfileKind, PermissionLevel, ToolErrorKind};

/// Approves or refuses one tool call, synchronously.
///
/// A convenience over [`ToolHookCallback`] for applications whose decision is
/// a plain function of the name and the arguments: return `Ok(())` to allow,
/// or `Err(reason)` to block with a reason the model reads. Install it with
/// [`ToolApprovalAdapter`].
pub type ToolApprovalFn = Arc<dyn Fn(&str, &Value) -> Result<(), String> + Send + Sync>;

/// What a session may do with one tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolAccess {
    /// The tool can be advertised and run with no approval step.
    Allowed,
    /// The tool can be advertised only where the session has an approval path.
    RequiresApproval,
    /// The tool must not be advertised or run.
    Denied,
}

impl ToolAccess {
    /// Whether a tool with this access is advertised to the model under
    /// `mode`.
    #[must_use]
    pub const fn is_exposed(self, mode: ToolExposureMode) -> bool {
        match self {
            Self::Allowed => true,
            Self::RequiresApproval => matches!(mode, ToolExposureMode::IncludeRequiresApproval),
            Self::Denied => false,
        }
    }
}

/// Whether approval-required tools appear in the tool schemas a session sends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolExposureMode {
    /// Advertise only tools that run without an approval path.
    #[default]
    AutoApprovedOnly,
    /// Advertise tools classified [`ToolAccess::RequiresApproval`] as well.
    IncludeRequiresApproval,
}

/// Which tools a session may advertise and run.
///
/// Deliberately name-only and synchronous: it answers the same way every time
/// it is asked, so a session can filter the tool schemas it sends with it.
/// Argument-sensitive, asynchronous, or logged decisions belong in
/// [`ToolHookCallback`].
///
/// Pebble ships the [`permission table`](crate::PermissionLevel::auto_approves)
/// an application can build a policy out of, but installs no policy of its
/// own; without one, every registered tool is exposed.
pub trait ToolAccessPolicy: Send + Sync {
    /// What the session may do with the tool the model would call
    /// `tool_name`.
    ///
    /// The name is the exposed one, after any vocabulary rename. Resolve it
    /// with [`canonical_tool_name`](crate::canonical_tool_name) to decide by
    /// identity.
    fn access_for_tool(&self, tool_name: &str) -> ToolAccess;
}

/// What a [`ToolHookCallback`] decided about a call that is about to run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolHookDecision {
    /// Run the tool.
    #[default]
    Proceed,
    /// Do not run the tool. The reason is given to the model as the call's
    /// output, so it is written for the model.
    Block { reason: String },
}

/// What an application does around each tool call.
///
/// Called once before a call and once after it, so an application can audit,
/// meter, mirror output somewhere, or block a call the static policy allowed.
/// Hooks run inside the call: a slow hook slows the session down.
#[async_trait::async_trait]
pub trait ToolHookCallback: Send + Sync {
    /// Runs before the tool, with the arguments the model produced.
    async fn pre_tool_use(&self, tool_name: &str, tool_input: &Value) -> ToolHookDecision;

    /// Runs after a call that succeeded, with the output the model will read.
    ///
    /// The output is the retained form — bounded by the session's output
    /// budgets — not the further-truncated copy history keeps.
    async fn post_tool_use(&self, tool_name: &str, tool_call_id: &str, tool_output: &str);

    /// Runs after a call that failed, with the message the model will read and
    /// why the call failed.
    ///
    /// The kind is the same one the call's
    /// [`ToolCallCompleted`](crate::AgentEvent::ToolCallCompleted) event
    /// carries, so a hook can branch on a refusal, a bad argument, or an
    /// interrupt without parsing the message.
    async fn post_tool_use_failure(
        &self,
        tool_name: &str,
        tool_call_id: &str,
        error: &str,
        error_kind: ToolErrorKind,
    );
}

/// A [`ToolHookCallback`] that only approves, built from a [`ToolApprovalFn`].
pub struct ToolApprovalAdapter(pub ToolApprovalFn);

#[async_trait::async_trait]
impl ToolHookCallback for ToolApprovalAdapter {
    async fn pre_tool_use(&self, tool_name: &str, tool_input: &Value) -> ToolHookDecision {
        match (self.0)(tool_name, tool_input) {
            Ok(()) => ToolHookDecision::Proceed,
            Err(reason) => ToolHookDecision::Block { reason },
        }
    }

    async fn post_tool_use(&self, _tool_name: &str, _tool_call_id: &str, _tool_output: &str) {}

    async fn post_tool_use_failure(
        &self,
        _tool_name: &str,
        _tool_call_id: &str,
        _error: &str,
        _error_kind: ToolErrorKind,
    ) {
    }
}

/// What the built-in tools are built with.
///
/// Captured when a profile builds its tools, because the numbers a model was
/// trained to budget against differ by harness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeToolOptions {
    /// How long a command runs when the model names no timeout.
    pub default_command_timeout_ms: u64,
    /// The longest timeout a model may ask for.
    pub max_command_timeout_ms:     u64,
}

impl NativeToolOptions {
    /// The options a profile starts from.
    ///
    /// Only the default timeout varies, and it matches what each harness
    /// documents: Anthropic's 120s, Kimi Code's 60s, and Codex's 10s, which is
    /// also pebble's own default.
    #[must_use]
    pub fn for_profile(profile_kind: AgentProfileKind) -> Self {
        let defaults = Self::default();
        // Matched exhaustively so a new profile has to state its answer rather
        // than silently inheriting the default timeout.
        let default_command_timeout_ms = match profile_kind {
            AgentProfileKind::Anthropic | AgentProfileKind::Claude5 => 120_000,
            AgentProfileKind::Kimi => 60_000,
            AgentProfileKind::OpenAi | AgentProfileKind::Gemini | AgentProfileKind::Gpt56 => {
                defaults.default_command_timeout_ms
            }
        };
        Self {
            default_command_timeout_ms,
            ..defaults
        }
    }
}

impl Default for NativeToolOptions {
    fn default() -> Self {
        Self {
            default_command_timeout_ms: 10_000,
            max_command_timeout_ms:     600_000,
        }
    }
}

/// Everything one session's behavior is tuned by.
///
/// Every member has a default that runs a session, so set the ones that
/// matter:
///
/// ```
/// use pebble::SessionOptions;
///
/// let options = SessionOptions {
///     enable_context_compaction: false,
///     ..SessionOptions::default()
/// };
/// assert_eq!(options.compaction_preserve_turns, 6);
/// ```
#[derive(Clone)]
pub struct SessionOptions {
    /// How hard the model should think, where the provider offers a choice.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Which latency or cost tier to ask for, where the provider offers one.
    pub speed: Option<Speed>,
    /// The most tokens the model may produce per turn. Absent takes the
    /// catalog's default for the model.
    pub max_tokens: Option<i64>,
    /// Per-tool character budgets for what history keeps, keyed by the name
    /// the model calls or by pebble's canonical name. Overrides the built-in
    /// limits.
    pub tool_output_limits: HashMap<String, usize>,
    /// Per-tool line budgets for what history keeps, keyed the same way.
    pub tool_line_limits: HashMap<String, usize>,
    /// Bytes of one tool's output the session retains: the form events, hooks,
    /// and the model all see.
    pub tool_output_retention_bytes: usize,
    /// Bytes that retained output may occupy once serialized as JSON, which
    /// escaping can inflate well past the text's own length.
    pub tool_output_serialized_bytes: usize,
    /// Whether to stop a session that is repeating itself.
    pub enable_loop_detection: bool,
    /// How many recent turns loop detection compares.
    pub loop_detection_window: usize,
    /// The repository root, when the work has one. Prompt assembly and tools
    /// read it; it is not a sandbox boundary.
    pub git_root: Option<String>,
    /// Extra instructions to put in the system prompt.
    pub user_instructions: Option<String>,
    /// Files whose contents are loaded into the system prompt as memory.
    ///
    /// Paths are explicit and resolved through the session's environment.
    /// Empty loads nothing: pebble looks in no conventional location and
    /// guesses no filename.
    pub memory_files: Vec<String>,
    /// Directories searched for skills.
    ///
    /// Explicit, like [`memory_files`](Self::memory_files); empty discovers no
    /// skills.
    pub skill_dirs: Vec<String>,
    /// What runs around each tool call.
    pub tool_hooks: Option<Arc<dyn ToolHookCallback>>,
    /// Which tools may be advertised and run. Absent exposes every registered
    /// tool.
    pub tool_access_policy: Option<Arc<dyn ToolAccessPolicy>>,
    /// The permission level the session started under, for an application that
    /// builds its policy from pebble's table and wants the level recorded
    /// beside it.
    pub permission_level: Option<PermissionLevel>,
    /// Whether approval-required tools are advertised, when a policy is
    /// installed.
    pub tool_exposure_mode: ToolExposureMode,
    /// Whether history is summarized as it approaches the context window.
    pub enable_context_compaction: bool,
    /// The share of the context window that triggers compaction.
    pub compaction_threshold_percent: usize,
    /// How many recent turns compaction leaves untouched.
    pub compaction_preserve_turns: usize,
    /// How long one run may take before the session cancels itself.
    pub wall_clock_timeout: Option<Duration>,
    /// How the session spaces the turn replays it owns.
    ///
    /// A stream that fails **after** the model produced visible output is
    /// replayed by the session rather than by the client's retry middleware,
    /// because a middleware reconnect would duplicate what the reader already
    /// saw. This policy decides the wait before each of those replays.
    ///
    /// Set it to the same policy the client's
    /// [`RetryMiddleware`](lithos_llm::middleware::RetryMiddleware) was built
    /// with, so one failure is spaced the same way wherever it is handled. Its
    /// `max_attempts` bounds the session's replays as well: the default of
    /// four allows the three replays pebble is willing to spend on one turn.
    ///
    /// The type is re-exported as [`pebble::RetryPolicy`](crate::RetryPolicy).
    pub retry_policy: RetryPolicy,
}

impl fmt::Debug for SessionOptions {
    /// Reports the callbacks as placeholders: they are application code, and
    /// what they close over is not pebble's to print.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionOptions")
            .field("reasoning_effort", &self.reasoning_effort)
            .field("speed", &self.speed)
            .field("max_tokens", &self.max_tokens)
            .field("tool_output_limits", &self.tool_output_limits)
            .field("tool_line_limits", &self.tool_line_limits)
            .field(
                "tool_output_retention_bytes",
                &self.tool_output_retention_bytes,
            )
            .field(
                "tool_output_serialized_bytes",
                &self.tool_output_serialized_bytes,
            )
            .field("enable_loop_detection", &self.enable_loop_detection)
            .field("loop_detection_window", &self.loop_detection_window)
            .field("git_root", &self.git_root)
            .field("user_instructions", &self.user_instructions)
            .field("memory_files", &self.memory_files)
            .field("skill_dirs", &self.skill_dirs)
            .field(
                "tool_hooks",
                &self.tool_hooks.as_ref().map(|_| "<callback>"),
            )
            .field(
                "tool_access_policy",
                &self.tool_access_policy.as_ref().map(|_| "<policy>"),
            )
            .field("permission_level", &self.permission_level)
            .field("tool_exposure_mode", &self.tool_exposure_mode)
            .field("enable_context_compaction", &self.enable_context_compaction)
            .field(
                "compaction_threshold_percent",
                &self.compaction_threshold_percent,
            )
            .field("compaction_preserve_turns", &self.compaction_preserve_turns)
            .field("wall_clock_timeout", &self.wall_clock_timeout)
            .field("retry_policy", &self.retry_policy)
            .finish()
    }
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            reasoning_effort: None,
            speed: None,
            max_tokens: None,
            tool_output_limits: HashMap::new(),
            tool_line_limits: HashMap::new(),
            tool_output_retention_bytes: DEFAULT_TOOL_OUTPUT_RETENTION_BYTES,
            tool_output_serialized_bytes: DEFAULT_TOOL_OUTPUT_SERIALIZED_BYTES,
            enable_loop_detection: true,
            loop_detection_window: 10,
            git_root: None,
            user_instructions: None,
            memory_files: Vec::new(),
            skill_dirs: Vec::new(),
            tool_hooks: None,
            tool_access_policy: None,
            permission_level: None,
            tool_exposure_mode: ToolExposureMode::AutoApprovedOnly,
            enable_context_compaction: true,
            compaction_threshold_percent: 80,
            compaction_preserve_turns: 6,
            wall_clock_timeout: None,
            retry_policy: RetryPolicy::exponential().max_attempts(DEFAULT_RETRY_ATTEMPTS),
        }
    }
}

/// How many attempts the default [`SessionOptions::retry_policy`] allows.
///
/// One opening attempt plus the three replays a session will spend on a turn
/// whose stream broke after it had already shown output.
const DEFAULT_RETRY_ATTEMPTS: u32 = 4;

impl SessionOptions {
    /// What the installed policy says about one tool, or
    /// [`ToolAccess::Allowed`] when there is no policy.
    #[must_use]
    pub fn tool_access_for(&self, tool_name: &str) -> ToolAccess {
        self.tool_access_policy
            .as_ref()
            .map_or(ToolAccess::Allowed, |policy| {
                policy.access_for_tool(tool_name)
            })
    }

    /// Whether one tool is advertised to the model.
    #[must_use]
    pub fn exposes_tool(&self, tool_name: &str) -> bool {
        self.tool_access_policy.as_ref().is_none_or(|policy| {
            policy
                .access_for_tool(tool_name)
                .is_exposed(self.tool_exposure_mode)
        })
    }

    /// Why the policy refuses one tool, or `None` when it does not.
    ///
    /// The message is what the model reads in place of the call's output, so a
    /// model that calls a tool it was never shown learns why.
    #[must_use]
    pub fn tool_access_denial_reason(&self, tool_name: &str) -> Option<String> {
        self.tool_access_policy.as_ref()?;
        match self.tool_access_for(tool_name) {
            ToolAccess::Allowed => None,
            ToolAccess::RequiresApproval
                if matches!(
                    self.tool_exposure_mode,
                    ToolExposureMode::IncludeRequiresApproval
                ) =>
            {
                None
            }
            ToolAccess::RequiresApproval => Some(format!(
                "{tool_name} tool requires approval, but this session does not expose \
                 approval-required tools"
            )),
            ToolAccess::Denied => Some(format!("{tool_name} tool denied by tool access policy")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, PoisonError};

    use lithos_llm::types::{Error as LlmError, ErrorKind as LlmErrorKind, RetryClassification};
    use serde_json::json;

    use super::*;

    struct StaticToolPolicy(ToolAccess);

    impl ToolAccessPolicy for StaticToolPolicy {
        fn access_for_tool(&self, _tool_name: &str) -> ToolAccess {
            self.0
        }
    }

    #[test]
    fn default_config_values() {
        let config = SessionOptions::default();

        assert!(config.reasoning_effort.is_none());
        assert!(config.speed.is_none());
        assert!(config.tool_output_limits.is_empty());
        assert!(config.tool_line_limits.is_empty());
        assert!(config.enable_loop_detection);
        assert_eq!(config.loop_detection_window, 10);
        assert!(config.user_instructions.is_none());
        assert!(config.tool_access_policy.is_none());
        assert!(config.permission_level.is_none());
        assert_eq!(
            config.tool_exposure_mode,
            ToolExposureMode::AutoApprovedOnly
        );
        assert!(config.wall_clock_timeout.is_none());
    }

    #[test]
    fn memory_and_skill_paths_start_empty() {
        let config = SessionOptions::default();
        assert!(config.memory_files.is_empty());
        assert!(config.skill_dirs.is_empty());
    }

    #[test]
    fn default_output_budgets_are_the_documented_ones() {
        let config = SessionOptions::default();
        assert_eq!(config.tool_output_retention_bytes, 1024 * 1024);
        assert_eq!(config.tool_output_serialized_bytes, 1_572_864);
    }

    #[test]
    fn native_tool_options_have_expected_profile_defaults() {
        let openai = NativeToolOptions::for_profile(AgentProfileKind::OpenAi);
        let gemini = NativeToolOptions::for_profile(AgentProfileKind::Gemini);
        let gpt56 = NativeToolOptions::for_profile(AgentProfileKind::Gpt56);
        let anthropic = NativeToolOptions::for_profile(AgentProfileKind::Anthropic);
        let claude5 = NativeToolOptions::for_profile(AgentProfileKind::Claude5);
        let kimi = NativeToolOptions::for_profile(AgentProfileKind::Kimi);

        assert_eq!(openai.default_command_timeout_ms, 10_000);
        assert_eq!(gemini.default_command_timeout_ms, 10_000);
        assert_eq!(gpt56.default_command_timeout_ms, 10_000);
        assert_eq!(anthropic.default_command_timeout_ms, 120_000);
        assert_eq!(claude5.default_command_timeout_ms, 120_000);
        assert_eq!(kimi.default_command_timeout_ms, 60_000);
        for options in [openai, gemini, gpt56, anthropic, claude5, kimi] {
            assert_eq!(options.max_command_timeout_ms, 600_000);
        }
    }

    #[test]
    fn the_default_retry_policy_allows_the_replays_a_turn_is_worth() {
        let config = SessionOptions::default();
        let error = LlmError::new(LlmErrorKind::Network, "connection reset")
            .with_retry(RetryClassification::Safe);

        for attempt in 1..DEFAULT_RETRY_ATTEMPTS {
            assert!(
                config.retry_policy.next_delay(attempt, &error).is_some(),
                "attempt {attempt} should still be replayed"
            );
        }
        assert!(
            config
                .retry_policy
                .next_delay(DEFAULT_RETRY_ATTEMPTS, &error)
                .is_none(),
            "the fourth failure spends the budget"
        );
    }

    #[test]
    fn default_config_has_compaction_enabled() {
        let config = SessionOptions::default();
        assert!(config.enable_context_compaction);
        assert_eq!(config.compaction_threshold_percent, 80);
        assert_eq!(config.compaction_preserve_turns, 6);
    }

    #[test]
    fn config_with_custom_values() {
        let config = SessionOptions {
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        };
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn debug_reports_callbacks_as_placeholders() {
        let approval: ToolApprovalFn = Arc::new(|_name, _args| Ok(()));
        let config = SessionOptions {
            tool_hooks: Some(Arc::new(ToolApprovalAdapter(approval))),
            tool_access_policy: Some(Arc::new(StaticToolPolicy(ToolAccess::Allowed))),
            ..SessionOptions::default()
        };

        let debug = format!("{config:?}");

        assert!(
            debug.contains(r#"tool_hooks: Some("<callback>")"#),
            "{debug}"
        );
        assert!(
            debug.contains(r#"tool_access_policy: Some("<policy>")"#),
            "{debug}"
        );
    }

    #[test]
    fn tool_hook_decision_default_is_proceed() {
        assert_eq!(ToolHookDecision::default(), ToolHookDecision::Proceed);
    }

    #[test]
    fn no_tool_access_policy_exposes_tools_by_default() {
        let config = SessionOptions::default();

        assert_eq!(config.tool_access_for("shell"), ToolAccess::Allowed);
        assert!(config.exposes_tool("shell"));
        assert!(config.tool_access_denial_reason("shell").is_none());
    }

    #[test]
    fn denied_tool_access_has_denial_reason() {
        let config = SessionOptions {
            tool_access_policy: Some(Arc::new(StaticToolPolicy(ToolAccess::Denied))),
            ..SessionOptions::default()
        };

        let reason = config
            .tool_access_denial_reason("shell")
            .expect("a denied tool has a reason");

        assert_eq!(reason, "shell tool denied by tool access policy");
        assert!(!config.exposes_tool("shell"));
    }

    #[test]
    fn approval_required_tools_follow_exposure_mode() {
        let config = SessionOptions {
            tool_access_policy: Some(Arc::new(StaticToolPolicy(ToolAccess::RequiresApproval))),
            tool_exposure_mode: ToolExposureMode::AutoApprovedOnly,
            ..SessionOptions::default()
        };

        assert!(!config.exposes_tool("shell"));
        assert_eq!(
            config
                .tool_access_denial_reason("shell")
                .expect("a hidden approval tool has a reason"),
            "shell tool requires approval, but this session does not expose approval-required tools"
        );

        let config = SessionOptions {
            tool_exposure_mode: ToolExposureMode::IncludeRequiresApproval,
            ..config
        };

        assert!(config.exposes_tool("shell"));
        assert!(config.tool_access_denial_reason("shell").is_none());
    }

    #[test]
    fn tool_access_decides_exposure_per_mode() {
        assert!(ToolAccess::Allowed.is_exposed(ToolExposureMode::AutoApprovedOnly));
        assert!(!ToolAccess::RequiresApproval.is_exposed(ToolExposureMode::AutoApprovedOnly));
        assert!(ToolAccess::RequiresApproval.is_exposed(ToolExposureMode::IncludeRequiresApproval));
        assert!(!ToolAccess::Denied.is_exposed(ToolExposureMode::IncludeRequiresApproval));
    }

    #[tokio::test]
    async fn tool_approval_adapter_allows() {
        let approval: ToolApprovalFn = Arc::new(|_name, _args| Ok(()));
        let adapter = ToolApprovalAdapter(approval);

        let decision = adapter.pre_tool_use("shell", &json!({})).await;

        assert_eq!(decision, ToolHookDecision::Proceed);
    }

    #[tokio::test]
    async fn tool_approval_adapter_blocks() {
        let approval: ToolApprovalFn = Arc::new(|_name, _args| Err("denied".to_owned()));
        let adapter = ToolApprovalAdapter(approval);

        let decision = adapter.pre_tool_use("shell", &json!({})).await;

        assert_eq!(decision, ToolHookDecision::Block {
            reason: "denied".to_owned(),
        });
    }

    #[tokio::test]
    async fn tool_approval_adapter_sees_the_arguments() {
        let approval: ToolApprovalFn = Arc::new(|name, args| {
            if name == "shell" && args["command"] == json!("rm -rf /") {
                Err("that command is not allowed".to_owned())
            } else {
                Ok(())
            }
        });
        let adapter = ToolApprovalAdapter(approval);

        assert_eq!(
            adapter
                .pre_tool_use("shell", &json!({"command": "rm -rf /"}))
                .await,
            ToolHookDecision::Block {
                reason: "that command is not allowed".to_owned(),
            }
        );
        assert_eq!(
            adapter
                .pre_tool_use("shell", &json!({"command": "ls"}))
                .await,
            ToolHookDecision::Proceed
        );
    }

    #[tokio::test]
    async fn tool_approval_adapter_post_is_noop() {
        let approval: ToolApprovalFn = Arc::new(|_name, _args| Ok(()));
        let adapter = ToolApprovalAdapter(approval);

        adapter.post_tool_use("shell", "call_1", "output").await;
        adapter
            .post_tool_use_failure("shell", "call_1", "error", ToolErrorKind::Execution)
            .await;
    }

    #[tokio::test]
    async fn a_failure_hook_is_told_why_the_call_failed() {
        struct RecordingHook(Mutex<Vec<(String, ToolErrorKind)>>);

        #[async_trait::async_trait]
        impl ToolHookCallback for RecordingHook {
            async fn pre_tool_use(
                &self,
                _tool_name: &str,
                _tool_input: &Value,
            ) -> ToolHookDecision {
                ToolHookDecision::Proceed
            }

            async fn post_tool_use(
                &self,
                _tool_name: &str,
                _tool_call_id: &str,
                _tool_output: &str,
            ) {
            }

            async fn post_tool_use_failure(
                &self,
                tool_name: &str,
                _tool_call_id: &str,
                _error: &str,
                error_kind: ToolErrorKind,
            ) {
                self.0
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((tool_name.to_owned(), error_kind));
            }
        }

        let hook = RecordingHook(Mutex::new(Vec::new()));

        hook.post_tool_use_failure("shell", "call_1", "denied", ToolErrorKind::Denied)
            .await;

        assert_eq!(
            hook.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_slice(),
            [("shell".to_owned(), ToolErrorKind::Denied)]
        );
    }
}
