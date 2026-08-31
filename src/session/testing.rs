//! What the session's own tests are built from.
//!
//! A session cannot be built without a profile, and pebble ships none yet, so
//! every test here injects one through
//! [`SessionBuilder::with_profile`](super::SessionBuilder::with_profile). The
//! rest is assembly: a session on a scripted client, over a mock environment,
//! with whatever tools and options the test needs.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use lithos_llm::Client;
use lithos_llm::client::ClientBuild;
use lithos_llm::middleware::{ConcurrencyLimitMiddleware, RetryMiddleware, RetryPolicy};
use lithos_llm::types::ToolDefinition;
use serde_json::{Value, json};
use tokio::sync::broadcast;
use tokio::task::yield_now;

use super::{RetryEventObserver, Session, SessionBuilder, ShutdownReason};
use crate::config::SessionOptions;
use crate::environment::Environment;
use crate::profile::{AgentProfile, EnvContext, SubagentSupport};
use crate::skills::{Skill, format_skills_prompt_section};
use crate::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    scripted_client_builder,
};
use crate::tool::{RegisteredTool, ToolError, ToolVocabulary};
use crate::types::{AgentEvent, AgentProfileKind, SessionEvent, ToolSource};

/// A profile that names a harness and contributes only what it is given.
pub(super) struct TestProfile {
    tools: Vec<RegisteredTool>,
}

impl TestProfile {
    /// A profile with no tools of its own.
    pub(super) fn shared() -> Arc<dyn AgentProfile> {
        Arc::new(Self { tools: Vec::new() })
    }

    /// A profile that contributes `tools`.
    pub(super) fn with_tools(tools: Vec<RegisteredTool>) -> Arc<dyn AgentProfile> {
        Arc::new(Self { tools })
    }
}

impl AgentProfile for TestProfile {
    fn profile_kind(&self) -> AgentProfileKind {
        AgentProfileKind::Anthropic
    }

    fn tool_vocabulary(&self) -> ToolVocabulary {
        ToolVocabulary::Canonical
    }

    fn base_tools(&self) -> Vec<RegisteredTool> {
        self.tools.clone()
    }

    fn build_system_prompt(
        &self,
        env_context: &EnvContext,
        memory: &[String],
        user_instructions: Option<&str>,
        skills: &[Skill],
    ) -> String {
        let _ = memory;
        let skills_section = format_skills_prompt_section(skills, ToolVocabulary::Canonical);
        let skills_part = if skills_section.is_empty() {
            String::new()
        } else {
            format!("\n\n{skills_section}")
        };
        let instructions = user_instructions
            .map(|text| format!("\n\n# User Instructions\n{text}"))
            .unwrap_or_default();
        format!(
            "You are a test assistant working in {}.{skills_part}{instructions}",
            env_context.working_directory
        )
    }

    fn subagent_tools(&self, _subagents: &SubagentSupport) -> Vec<RegisteredTool> {
        Vec::new()
    }
}

/// Assembles one session over a scripted provider.
///
/// Every part has an answer that suits most tests, so a test names only what it
/// is about: the script, and whichever of the tools, the options, the model, or
/// the environment it depends on.
pub(super) struct TestSession {
    calls:       Vec<ScriptedCall>,
    delay:       Duration,
    completions: Vec<ScriptedCompletion>,
    tools:       Vec<RegisteredTool>,
    options:     SessionOptions,
    model:       String,
    environment: Option<Arc<dyn Environment>>,
    retries:     Option<RetryPolicy>,
    concurrency: Option<NonZeroUsize>,
}

impl TestSession {
    /// A session whose provider answers `calls`, one per round.
    pub(super) fn new(calls: Vec<ScriptedCall>) -> Self {
        Self {
            calls,
            delay: Duration::ZERO,
            completions: Vec::new(),
            tools: Vec::new(),
            options: SessionOptions::default(),
            model: "test/model".to_owned(),
            environment: None,
            retries: None,
            concurrency: None,
        }
    }

    /// A session whose provider answers every round with one response.
    pub(super) fn answering(calls: Vec<ScriptedCall>) -> (Session, Arc<ScriptedProvider>) {
        Self::new(calls).build()
    }

    /// Scripts what compaction's summarizing call answers with.
    pub(super) fn completing(mut self, completions: Vec<ScriptedCompletion>) -> Self {
        self.completions = completions;
        self
    }

    /// Registers tools on top of the profile's own.
    pub(super) fn tools(mut self, tools: impl IntoIterator<Item = RegisteredTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Sets how the session behaves.
    pub(super) fn options(mut self, options: SessionOptions) -> Self {
        self.options = options;
        self
    }

    /// Names a different model in the test catalog, such as `test/small`.
    pub(super) fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets where the session's tools act.
    pub(super) fn environment(mut self, environment: Arc<dyn Environment>) -> Self {
        self.environment = Some(environment);
        self
    }

    /// Makes every model call take `delay` to answer.
    pub(super) fn delayed(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Installs the retry middleware an application is asked to install, with
    /// pebble's observer on it.
    pub(super) fn retrying(mut self, policy: RetryPolicy) -> Self {
        self.retries = Some(policy);
        self
    }

    /// Installs the client's concurrency limiter, which holds a permit for as
    /// long as the stream it handed out lives.
    ///
    /// # Panics
    ///
    /// Panics when `limit` is zero, which no test asks for.
    pub(super) fn limited(mut self, limit: usize) -> Self {
        self.concurrency = Some(NonZeroUsize::new(limit).expect("a limit of at least one"));
        self
    }

    /// Builds the session, and the provider handle its script is read from.
    pub(super) fn build(self) -> (Session, Arc<ScriptedProvider>) {
        let provider = ScriptedProvider::new(self.calls)
            .completing(self.completions)
            .delayed(self.delay);
        let (client, provider) = match (self.retries, self.concurrency) {
            (None, None) => client_from(provider),
            (retries, concurrency) => {
                let (mut builder, provider) = scripted_client_builder(provider);
                if let Some(policy) = retries {
                    builder = builder
                        .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver));
                }
                if let Some(limit) = concurrency {
                    builder = builder.middleware(ConcurrencyLimitMiddleware::new(limit));
                }
                let ClientBuild { client, .. } =
                    builder.build().expect("the scripted client builds");
                (client, provider)
            }
        };
        let environment = self
            .environment
            .unwrap_or_else(|| Arc::new(MockEnvironment::linux()));
        let session = Session::builder(client)
            .model(self.model)
            .environment(environment)
            .with_profile(TestProfile::with_tools(self.tools))
            .options(self.options)
            .build()
            .expect("the test session builds");
        (session, provider)
    }
}

/// A builder for a session on `client`, with nothing scripted.
pub(super) fn builder(client: Client) -> SessionBuilder {
    Session::builder(client)
        .model("test/model")
        .environment(Arc::new(MockEnvironment::linux()))
        .with_profile(TestProfile::shared())
}

/// Closes the session, which publishes everything queued and joins the pump,
/// then reports every event the receiver holds.
///
/// Shutting down first is what makes the list complete: nothing is published
/// until the pump runs, and the pump stops only when the session tells it to.
pub(super) async fn settled(
    session: &mut Session,
    receiver: &mut broadcast::Receiver<SessionEvent>,
) -> Vec<AgentEvent> {
    session
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the shutdown succeeds");
    drain(receiver)
}

/// Everything published so far, once the pump has had its turn.
///
/// The session queues events for a task to publish, so a test that reads the
/// stream while the session is still open has to let that task run first.
pub(super) async fn drained(receiver: &mut broadcast::Receiver<SessionEvent>) -> Vec<AgentEvent> {
    yield_now().await;
    yield_now().await;
    drain(receiver)
}

/// Everything the receiver already holds.
pub(super) fn drain(receiver: &mut broadcast::Receiver<SessionEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event.event);
    }
    events
}

/// Waits for the first event that satisfies `predicate`.
pub(super) async fn wait_for_event(
    receiver: &mut broadcast::Receiver<SessionEvent>,
    predicate: impl Fn(&AgentEvent) -> bool,
) {
    loop {
        let event = receiver
            .recv()
            .await
            .expect("the session event stream stays open");
        if predicate(&event.event) {
            return;
        }
    }
}

/// Where `matcher` first matched, which a test uses to assert on ordering.
pub(super) fn position(
    events: &[AgentEvent],
    matcher: impl Fn(&AgentEvent) -> bool,
) -> Option<usize> {
    events.iter().position(matcher)
}

/// A short name for each event, for asserting on a whole stream at once.
pub(super) fn event_names(events: &[AgentEvent]) -> Vec<&'static str> {
    events.iter().map(event_name).collect()
}

/// A short name for one event.
pub(super) fn event_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::SessionStarted { .. } => "started",
        AgentEvent::SessionEnded => "ended",
        AgentEvent::ProcessingEnd => "processing_end",
        AgentEvent::MemoryLoaded { .. } => "memory",
        AgentEvent::SkillsDiscovered { .. } => "skills",
        AgentEvent::UserInput { .. } => "input",
        AgentEvent::LlmRequestStarted { .. } => "request",
        AgentEvent::LlmFirstOutput { .. } => "first_output",
        AgentEvent::TextDelta { .. } => "delta",
        AgentEvent::AssistantMessage { .. } => "message",
        _ => "other",
    }
}

/// How many events `matcher` matched.
pub(super) fn count(events: &[AgentEvent], matcher: impl Fn(&AgentEvent) -> bool) -> usize {
    events.iter().filter(|event| matcher(event)).count()
}

/// A tool that answers with `echo: <text>`.
pub(super) fn echo_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "echo",
            "Echoes the input",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        ),
        executor:   Arc::new(|arguments: Value, _context| {
            Box::pin(async move {
                let text = arguments
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("no text");
                Ok(format!("echo: {text}"))
            })
        }),
        source:     ToolSource::Native,
    }
}

/// A tool that always fails.
pub(super) fn failing_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "fail_tool",
            "Always fails",
            json!({"type": "object"}),
        ),
        executor:   Arc::new(|_arguments, _context| {
            Box::pin(async { Err(ToolError::execution("tool execution failed")) })
        }),
        source:     ToolSource::Native,
    }
}

/// A tool named `name` that answers `ok`.
pub(super) fn noop_tool(name: &str) -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            name,
            format!("Tool {name}"),
            json!({"type": "object"}),
        ),
        executor:   Arc::new(|_arguments, _context| Box::pin(async { Ok("ok".to_owned()) })),
        source:     ToolSource::Native,
    }
}

/// A tool that waits until the round or the run is cancelled.
pub(super) fn blocking_tool(name: &'static str) -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            name,
            "Waits until cancelled",
            json!({"type": "object"}),
        ),
        executor:   Arc::new(|_arguments, context| {
            Box::pin(async move {
                context.cancel.cancelled().await;
                Err(ToolError::cancelled("Cancelled"))
            })
        }),
        source:     ToolSource::Native,
    }
}
