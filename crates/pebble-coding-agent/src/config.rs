//! How an application configures one session.
//!
//! [`CodingAgentOptions`] starts from its [`Default`] and is adjusted through
//! focused `with_*` methods, so an option cannot be left half-set. Active
//! behavior around tools belongs on the coding-agent builder as middleware.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::{ReasoningEffort, Speed};
use pebble_agent::AgentConfig;

use crate::truncation::{
    DEFAULT_TOOL_OUTPUT_RETENTION_BYTES, DEFAULT_TOOL_OUTPUT_SERIALIZED_BYTES,
};
use crate::types::{AgentProfileKind, PermissionLevel};

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
/// Every setting has a default that runs a session, so start from
/// [`default`](Self::default) and set the ones that matter through the
/// `with_*` methods:
///
/// ```
/// use std::time::Duration;
///
/// use pebble_coding_agent::CodingAgentOptions;
///
/// let options = CodingAgentOptions::default()
///     .with_context_compaction(false)
///     .with_wall_clock_timeout(Duration::from_secs(600))
///     .with_user_instructions("Always run the tests.");
/// # let _ = options;
/// ```
#[derive(Clone)]
pub struct CodingAgentOptions {
    /// How hard the model should think, where the provider offers a choice.
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    /// Which latency or cost tier to ask for, where the provider offers one.
    pub(crate) speed: Option<Speed>,
    /// The most tokens the model may produce per turn. Absent takes the
    /// catalog's default for the model.
    pub(crate) max_tokens: Option<i64>,
    /// Per-tool character budgets for what history keeps, keyed by the name
    /// the model calls or by pebble's canonical name. Overrides the built-in
    /// limits.
    pub(crate) tool_output_limits: HashMap<String, usize>,
    /// Per-tool line budgets for what history keeps, keyed the same way.
    pub(crate) tool_line_limits: HashMap<String, usize>,
    /// Bytes of one tool's output the session retains: the form events and the
    /// model see.
    pub(crate) tool_output_retention_bytes: usize,
    /// Bytes that retained output may occupy once serialized as JSON, which
    /// escaping can inflate well past the text's own length.
    pub(crate) tool_output_serialized_bytes: usize,
    /// Whether to stop a session that is repeating itself.
    pub(crate) enable_loop_detection: bool,
    /// How many recent turns loop detection compares.
    pub(crate) loop_detection_window: usize,
    /// The repository root, when the work has one. Prompt assembly and tools
    /// read it; it is not a sandbox boundary.
    pub(crate) git_root: Option<String>,
    /// Extra instructions to put in the system prompt.
    pub(crate) user_instructions: Option<String>,
    /// Files whose contents are loaded into the system prompt as memory.
    ///
    /// Paths are explicit and resolved through the session's environment.
    /// Empty loads nothing: pebble looks in no conventional location and
    /// guesses no filename.
    pub(crate) memory_files: Vec<String>,
    /// Directories searched for skills.
    ///
    /// Explicit, like [`memory_files`](Self::memory_files); empty discovers no
    /// skills.
    pub(crate) skill_dirs: Vec<String>,
    /// The permission level the session started under, for an application that
    /// installs matching middleware and wants the level recorded beside it.
    pub(crate) permission_level: Option<PermissionLevel>,
    /// Whether history is summarized as it approaches the context window.
    pub(crate) enable_context_compaction: bool,
    /// The share of the context window that triggers compaction.
    pub(crate) compaction_threshold_percent: usize,
    /// How many recent turns compaction leaves untouched. Compaction leaves
    /// the newest turn whatever this says, because it may hold tool calls that
    /// have not been answered yet.
    pub(crate) compaction_preserve_turns: usize,
    /// How long one prompt may take before the session cancels itself.
    pub(crate) wall_clock_timeout: Option<Duration>,
    /// How the session spaces the turn replays it owns.
    ///
    /// A stream that fails **after** the model produced visible output is
    /// replayed by the session rather than by the client's retry middleware,
    /// because a middleware reconnect would duplicate what the reader already
    /// saw. This policy decides the wait before each of those replays.
    ///
    /// This is separate from the client's
    /// [`RetryMiddleware`](lithos_llm::middleware::RetryMiddleware) policy.
    /// Reuse that policy when identical spacing is useful, or configure the
    /// two independently. `max_attempts` bounds the session's replays: the
    /// default of four allows the three replays Pebble is willing to spend on
    /// one turn.
    ///
    /// The default schedule is [`AgentConfig`]'s: one second before the first
    /// replay, doubling for each replay after that, and capped at sixty
    /// seconds. Each wait is jittered: it lands between half of the computed
    /// wait and the whole of it, so the three replays together take between
    /// about 3.5 and 7 seconds. That outlasts a provider blip of a few seconds,
    /// which the 100 millisecond schedule of [`RetryPolicy::exponential`] does
    /// not. A `Retry-After` from the provider replaces the computed wait; one
    /// longer than sixty seconds ends the replays instead.
    ///
    /// The type is reexported as
    /// [`lithos_llm::middleware::RetryPolicy`].
    pub(crate) turn_replay: RetryPolicy,
}

impl fmt::Debug for CodingAgentOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodingAgentOptions")
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
            .field("permission_level", &self.permission_level)
            .field("enable_context_compaction", &self.enable_context_compaction)
            .field(
                "compaction_threshold_percent",
                &self.compaction_threshold_percent,
            )
            .field("compaction_preserve_turns", &self.compaction_preserve_turns)
            .field("wall_clock_timeout", &self.wall_clock_timeout)
            .field("turn_replay", &self.turn_replay)
            .finish()
    }
}

impl Default for CodingAgentOptions {
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
            permission_level: None,
            enable_context_compaction: true,
            compaction_threshold_percent: 80,
            compaction_preserve_turns: 6,
            wall_clock_timeout: None,
            turn_replay: AgentConfig::default().turn_replay,
        }
    }
}

impl CodingAgentOptions {
    /// Sets how hard the model should think, where the provider offers a
    /// choice.
    #[must_use]
    pub const fn with_reasoning_effort(mut self, effort: Option<ReasoningEffort>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Sets which latency or cost tier to ask for, where the provider offers
    /// one.
    #[must_use]
    pub const fn with_speed(mut self, speed: Option<Speed>) -> Self {
        self.speed = speed;
        self
    }

    /// Sets the most tokens the model may produce per turn. `None` takes the
    /// catalog's default for the model.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: Option<i64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the character budget history keeps of one tool's output, keyed by
    /// the name the model calls or by pebble's canonical name. Overrides the
    /// built-in limit for that tool.
    #[must_use]
    pub fn with_tool_output_limit(mut self, tool_name: impl Into<String>, chars: usize) -> Self {
        self.tool_output_limits.insert(tool_name.into(), chars);
        self
    }

    /// Sets the line budget history keeps of one tool's output, keyed the same
    /// way as [`with_tool_output_limit`](Self::with_tool_output_limit).
    #[must_use]
    pub fn with_tool_line_limit(mut self, tool_name: impl Into<String>, lines: usize) -> Self {
        self.tool_line_limits.insert(tool_name.into(), lines);
        self
    }

    /// Sets how many bytes of one tool's output the session retains: the form
    /// events, middleware, and the model all see.
    #[must_use]
    pub const fn with_tool_output_retention_bytes(mut self, bytes: usize) -> Self {
        self.tool_output_retention_bytes = bytes;
        self
    }

    /// Sets how many bytes retained output may occupy once serialized as JSON,
    /// which escaping can inflate well past the text's own length.
    #[must_use]
    pub const fn with_tool_output_serialized_bytes(mut self, bytes: usize) -> Self {
        self.tool_output_serialized_bytes = bytes;
        self
    }

    /// Sets whether to stop a session that is repeating itself.
    #[must_use]
    pub const fn with_loop_detection(mut self, enabled: bool) -> Self {
        self.enable_loop_detection = enabled;
        self
    }

    /// Sets how many recent turns loop detection compares.
    #[must_use]
    pub const fn with_loop_detection_window(mut self, turns: usize) -> Self {
        self.loop_detection_window = turns;
        self
    }

    /// Names the repository root, when the work has one. Prompt assembly and
    /// tools read it; it is not a sandbox boundary.
    #[must_use]
    pub fn with_git_root(mut self, git_root: impl Into<String>) -> Self {
        self.git_root = Some(git_root.into());
        self
    }

    /// Adds instructions to the system prompt.
    #[must_use]
    pub fn with_user_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.user_instructions = Some(instructions.into());
        self
    }

    /// Names the files whose contents are loaded into the system prompt as
    /// memory. Paths are explicit and resolved through the session's
    /// environment; pebble looks in no conventional location.
    #[must_use]
    pub fn with_memory_files(mut self, files: impl IntoIterator<Item = String>) -> Self {
        self.memory_files = files.into_iter().collect();
        self
    }

    /// Names the directories searched for skills. Explicit, like
    /// [`with_memory_files`](Self::with_memory_files).
    #[must_use]
    pub fn with_skill_dirs(mut self, dirs: impl IntoIterator<Item = String>) -> Self {
        self.skill_dirs = dirs.into_iter().collect();
        self
    }

    /// Records the permission level the session started under, for an
    /// application that builds its policy from pebble's table.
    #[must_use]
    pub const fn with_permission_level(mut self, level: PermissionLevel) -> Self {
        self.permission_level = Some(level);
        self
    }

    /// Sets whether history is summarized as it approaches the context window.
    #[must_use]
    pub const fn with_context_compaction(mut self, enabled: bool) -> Self {
        self.enable_context_compaction = enabled;
        self
    }

    /// Sets the share of the context window that triggers compaction.
    #[must_use]
    pub const fn with_compaction_threshold_percent(mut self, percent: usize) -> Self {
        self.compaction_threshold_percent = percent;
        self
    }

    /// Sets how many recent turns compaction leaves untouched.
    #[must_use]
    pub const fn with_compaction_preserve_turns(mut self, turns: usize) -> Self {
        self.compaction_preserve_turns = turns;
        self
    }

    /// Sets how long one prompt may take before the agent cancels it. The
    /// agent stays open; the next prompt gets a fresh budget.
    #[must_use]
    pub const fn with_wall_clock_timeout(mut self, timeout: Duration) -> Self {
        self.wall_clock_timeout = Some(timeout);
        self
    }

    /// Sets how the session spaces the turn replays it owns.
    ///
    /// A stream that fails after the model produced visible output is replayed
    /// by the session rather than by the client's retry middleware. This policy
    /// decides the wait before each replay; its `max_attempts` bounds them.
    #[must_use]
    pub const fn with_turn_replay(mut self, policy: RetryPolicy) -> Self {
        self.turn_replay = policy;
        self
    }
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::{Error as LlmError, ErrorKind as LlmErrorKind, RetryClassification};

    use super::*;

    /// A dropped connection, which the client may repeat.
    fn dropped_stream() -> LlmError {
        LlmError::new(LlmErrorKind::Network, "connection reset")
            .with_retry(RetryClassification::Safe)
    }

    #[test]
    fn default_config_values() {
        let config = CodingAgentOptions::default();

        assert!(config.reasoning_effort.is_none());
        assert!(config.speed.is_none());
        assert!(config.tool_output_limits.is_empty());
        assert!(config.tool_line_limits.is_empty());
        assert!(config.enable_loop_detection);
        assert_eq!(config.loop_detection_window, 10);
        assert!(config.user_instructions.is_none());
        assert!(config.permission_level.is_none());
        assert!(config.wall_clock_timeout.is_none());
    }

    #[test]
    fn memory_and_skill_paths_start_empty() {
        let config = CodingAgentOptions::default();
        assert!(config.memory_files.is_empty());
        assert!(config.skill_dirs.is_empty());
    }

    #[test]
    fn default_output_budgets_are_the_documented_ones() {
        let config = CodingAgentOptions::default();
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
    fn the_default_turn_replay_allows_the_replays_a_turn_is_worth() {
        // One opening attempt plus the three replays a session spends on a
        // turn whose stream broke after it had already shown output.
        let config = CodingAgentOptions::default();
        let error = dropped_stream();

        for attempt in 1..4 {
            assert!(
                config.turn_replay.next_delay(attempt, &error).is_some(),
                "attempt {attempt} should still be replayed"
            );
        }
        assert!(
            config.turn_replay.next_delay(4, &error).is_none(),
            "the fourth failure spends the budget"
        );
    }

    #[test]
    fn the_default_turn_replay_waits_seconds_not_milliseconds() {
        // The schedule pebble-agent defines and this layer reuses: one second,
        // doubling, jittered. Jitter lands each wait between half of the
        // computed delay and the whole of it, so the three replays together
        // outlast a provider blip of a few seconds.
        let config = CodingAgentOptions::default();
        let error = dropped_stream();
        let expected = [
            (1, Duration::from_millis(500), Duration::from_secs(1)),
            (2, Duration::from_secs(1), Duration::from_secs(2)),
            (3, Duration::from_secs(2), Duration::from_secs(4)),
        ];

        for (attempt, floor, ceiling) in expected {
            let delay = config
                .turn_replay
                .next_delay(attempt, &error)
                .unwrap_or_else(|| panic!("attempt {attempt} is replayed"));
            assert!(
                (floor..=ceiling).contains(&delay),
                "attempt {attempt} waited {delay:?}, outside {floor:?}..={ceiling:?}"
            );
        }
    }

    #[test]
    fn the_default_turn_replay_caps_its_wait_at_a_minute() {
        // The default budget is spent long before the doubling reaches the
        // cap, so a wider budget on the same schedule shows it.
        let policy = CodingAgentOptions::default()
            .turn_replay
            .max_attempts(u32::MAX);

        let delay = policy
            .next_delay(20, &dropped_stream())
            .expect("a replay is still allowed");
        assert!(
            (Duration::from_secs(30)..=Duration::from_secs(60)).contains(&delay),
            "attempt 20 waited {delay:?}, outside the jittered sixty second cap"
        );
    }

    #[test]
    fn default_config_has_compaction_enabled() {
        let config = CodingAgentOptions::default();
        assert!(config.enable_context_compaction);
        assert_eq!(config.compaction_threshold_percent, 80);
        assert_eq!(config.compaction_preserve_turns, 6);
    }

    #[test]
    fn config_with_custom_values() {
        let config = CodingAgentOptions {
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        };
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::High));
    }
}
