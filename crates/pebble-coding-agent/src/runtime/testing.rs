//! What the session's own tests are built from.
//!
//! Every test here injects a profile through
//! [`CodingRuntimeBuilder::with_profile`](super::CodingRuntimeBuilder::with_profile),
//! rather than taking the built-in one its model resolves to: what these tests
//! are about is the loop, and a harness that contributes nothing keeps a tool
//! list or a prompt from a shipped profile out of every assertion. The tests
//! that *are* about a harness live in
//! [`loop_tests::profiles`](super::loop_tests::profiles) and let the catalog
//! choose. The rest is assembly: a session on a scripted client, over a mock
//! environment, with whatever tools and options the test needs.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use lithos_llm::Client;
use lithos_llm::client::ClientBuild;
use lithos_llm::middleware::{ConcurrencyLimitMiddleware, RetryMiddleware, RetryPolicy};
use lithos_llm::types::ToolDefinition;
use pebble_agent::ToolMiddleware;
use serde_json::{Value, json};
use tokio::sync::broadcast;
use tokio::task::yield_now;

use super::{CodingRuntime, CodingRuntimeBuilder, RetryEventObserver, ShutdownReason};
use crate::config::CodingAgentOptions;
use crate::environment::Environment;
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::profile::{AgentProfile, EnvContext};
use crate::redact::Redactor;
use crate::search::SearchProvider;
use crate::skills::{Skill, format_skills_prompt_section};
use crate::subagent::{ChildObserver, SubagentLimits, SubagentOptions};
use crate::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    scripted_client_builder,
};
use crate::tool::{RegisteredTool, ToolEnvProvider, ToolError, ToolRegistry, ToolVocabulary};
use crate::types::{AgentProfileKind, CodingAgentEvent, CodingEvent, Message, ToolSource};

/// A profile that names a harness and contributes only what it is given.
pub(crate) struct TestProfile {
    tools: Vec<RegisteredTool>,
}

impl TestProfile {
    /// A profile with no tools of its own.
    pub(crate) fn shared() -> Arc<dyn AgentProfile> {
        Arc::new(Self { tools: Vec::new() })
    }

    /// A profile that contributes `tools`.
    pub(crate) fn with_tools(tools: Vec<RegisteredTool>) -> Arc<dyn AgentProfile> {
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
        _registry: &ToolRegistry,
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
}

/// Assembles one session over a scripted provider.
///
/// Every part has an answer that suits most tests, so a test names only what it
/// is about: the script, and whichever of the tools, the options, the model, or
/// the environment it depends on.
pub(crate) struct TestSession {
    calls:           Vec<ScriptedCall>,
    delay:           Duration,
    completions:     Vec<ScriptedCompletion>,
    tools:           Vec<RegisteredTool>,
    tool_middleware: Vec<Arc<dyn ToolMiddleware>>,
    options:         CodingAgentOptions,
    model:           String,
    environment:     Option<Arc<dyn Environment>>,
    retries:         Option<RetryPolicy>,
    concurrency:     Option<NonZeroUsize>,
    subagents:       bool,
    observer:        Option<ChildObserver>,
    limits:          SubagentLimits,
    human_input:     Option<Arc<dyn HumanInputProvider>>,
    tool_env:        Option<Arc<dyn ToolEnvProvider>>,
    redactor:        Option<Arc<dyn Redactor>>,
    search_provider: Option<Arc<dyn SearchProvider>>,
}

impl TestSession {
    /// A session whose provider answers `calls`, one per round.
    pub(crate) fn new(calls: Vec<ScriptedCall>) -> Self {
        Self {
            calls,
            delay: Duration::ZERO,
            completions: Vec::new(),
            tools: Vec::new(),
            tool_middleware: Vec::new(),
            options: CodingAgentOptions::default(),
            model: "test/model".to_owned(),
            environment: None,
            retries: None,
            concurrency: None,
            subagents: false,
            observer: None,
            limits: SubagentLimits::default(),
            human_input: None,
            tool_env: None,
            redactor: None,
            search_provider: None,
        }
    }

    /// Strips secrets out of what the session publishes.
    pub(crate) fn redacting(mut self, redactor: Arc<dyn Redactor>) -> Self {
        self.redactor = Some(redactor);
        self
    }

    /// Gives the session somewhere to send a web search.
    pub(crate) fn searching_with(mut self, provider: Arc<dyn SearchProvider>) -> Self {
        self.search_provider = Some(provider);
        self
    }

    /// Lets the session spawn children, built the plain way.
    pub(crate) fn with_subagents(mut self) -> Self {
        self.subagents = true;
        self
    }

    /// Lets the session spawn children, built by `factory`.
    pub(crate) fn observe_children(mut self, observer: ChildObserver) -> Self {
        self.subagents = true;
        self.observer = Some(observer);
        self
    }

    /// Gives the session someone to ask, which only a root ever has.
    pub(crate) fn human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Sets where a tool call's extra environment variables come from.
    pub(crate) fn tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env = Some(provider);
        self
    }

    /// Sets how many sessions the tree may hold open at once.
    pub(crate) const fn subagent_limits(mut self, limits: SubagentLimits) -> Self {
        self.limits = limits;
        self
    }

    /// A session whose provider answers every round with one response.
    pub(crate) fn answering(calls: Vec<ScriptedCall>) -> (CodingRuntime, Arc<ScriptedProvider>) {
        Self::new(calls).build()
    }

    /// Scripts what compaction's summarizing call answers with.
    pub(crate) fn completing(mut self, completions: Vec<ScriptedCompletion>) -> Self {
        self.completions = completions;
        self
    }

    /// Registers tools on top of the profile's own.
    pub(crate) fn tools(mut self, tools: impl IntoIterator<Item = RegisteredTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Adds one layer around tool discovery and calls.
    pub(crate) fn tool_middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.tool_middleware.push(middleware);
        self
    }

    /// Sets how the session behaves.
    pub(crate) fn options(mut self, options: CodingAgentOptions) -> Self {
        self.options = options;
        self
    }

    /// Names a different model in the test catalog, such as `test/small`.
    pub(crate) fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets where the session's tools act.
    pub(crate) fn environment(mut self, environment: Arc<dyn Environment>) -> Self {
        self.environment = Some(environment);
        self
    }

    /// Makes every model call take `delay` to answer.
    pub(crate) fn delayed(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Installs the retry middleware an application is asked to install, with
    /// pebble's observer on it.
    pub(crate) fn retrying(mut self, policy: RetryPolicy) -> Self {
        self.retries = Some(policy);
        self
    }

    /// Installs the client's concurrency limiter, which holds a permit for as
    /// long as the stream it handed out lives.
    ///
    /// # Panics
    ///
    /// Panics when `limit` is zero, which no test asks for.
    pub(crate) fn limited(mut self, limit: usize) -> Self {
        self.concurrency = Some(NonZeroUsize::new(limit).expect("a limit of at least one"));
        self
    }

    /// Builds the session, and the provider handle its script is read from.
    pub(crate) fn build(self) -> (CodingRuntime, Arc<ScriptedProvider>) {
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
        let mut builder = CodingRuntime::builder(client)
            .model(self.model)
            .environment(environment)
            .with_profile(TestProfile::with_tools(self.tools))
            .options(self.options);
        for middleware in self.tool_middleware {
            builder = builder.tool_middleware(middleware);
        }
        if self.subagents {
            builder = builder.subagents(SubagentOptions::enabled().with_limits(self.limits));
        }
        if let Some(observer) = self.observer {
            builder = builder.observe_children(observer);
        }
        if let Some(provider) = self.human_input {
            builder = builder.human_input(provider);
        }
        if let Some(provider) = self.tool_env {
            builder = builder.tool_env_provider(provider);
        }
        if let Some(redactor) = self.redactor {
            builder = builder.redactor(redactor);
        }
        if let Some(provider) = self.search_provider {
            builder = builder.search_provider(provider);
        }
        let session = builder.build().expect("the test session builds");
        (session, provider)
    }
}

/// A builder for a session on `client`, with nothing scripted.
pub(crate) fn builder(client: Client) -> CodingRuntimeBuilder {
    CodingRuntime::builder(client)
        .model("test/model")
        .environment(Arc::new(MockEnvironment::linux()))
        .with_profile(TestProfile::shared())
}

/// A history holding `turns`, oldest first.
pub(crate) fn history_from(turns: Vec<Message>) -> History {
    let mut history = History::default();
    for turn in turns {
        history.push(turn);
    }
    history
}

/// Closes the session, which publishes everything queued and joins the pump,
/// then reports every event the receiver holds.
///
/// Shutting down first is what makes the list complete: nothing is published
/// until the pump runs, and the pump stops only when the session tells it to.
pub(crate) async fn settled(
    session: &mut CodingRuntime,
    receiver: &mut broadcast::Receiver<CodingAgentEvent>,
) -> Vec<CodingEvent> {
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
pub(crate) async fn drained(
    receiver: &mut broadcast::Receiver<CodingAgentEvent>,
) -> Vec<CodingEvent> {
    yield_now().await;
    yield_now().await;
    drain(receiver)
}

/// Everything the receiver already holds.
pub(crate) fn drain(receiver: &mut broadcast::Receiver<CodingAgentEvent>) -> Vec<CodingEvent> {
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event.event);
    }
    events
}

/// Waits for the first event that satisfies `predicate`.
pub(crate) async fn wait_for_event(
    receiver: &mut broadcast::Receiver<CodingAgentEvent>,
    predicate: impl Fn(&CodingEvent) -> bool,
) {
    loop {
        let event = receiver
            .recv()
            .await
            .expect("the coding event stream stays open");
        if predicate(&event.event) {
            return;
        }
    }
}

/// Where `matcher` first matched, which a test uses to assert on ordering.
pub(crate) fn position(
    events: &[CodingEvent],
    matcher: impl Fn(&CodingEvent) -> bool,
) -> Option<usize> {
    events.iter().position(matcher)
}

/// A short name for each event, for asserting on a whole stream at once.
pub(crate) fn event_names(events: &[CodingEvent]) -> Vec<&'static str> {
    events.iter().map(event_name).collect()
}

/// A short name for one event.
pub(crate) fn event_name(event: &CodingEvent) -> &'static str {
    match event {
        CodingEvent::SessionStarted { .. } => "started",
        CodingEvent::SessionEnded => "ended",
        CodingEvent::ProcessingEnd => "processing_end",
        CodingEvent::MemoryLoaded { .. } => "memory",
        CodingEvent::SkillsDiscovered { .. } => "skills",
        CodingEvent::UserInput { .. } => "input",
        CodingEvent::LlmRequestStarted { .. } => "request",
        CodingEvent::LlmFirstOutput { .. } => "first_output",
        CodingEvent::TextDelta { .. } => "delta",
        CodingEvent::AssistantMessage { .. } => "message",
        _ => "other",
    }
}

/// How many events `matcher` matched.
pub(crate) fn count(events: &[CodingEvent], matcher: impl Fn(&CodingEvent) -> bool) -> usize {
    events.iter().filter(|event| matcher(event)).count()
}

/// A tool that answers with `echo: <text>`.
pub(crate) fn echo_tool() -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(
            "echo",
            "Echoes the input",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        ),
        Arc::new(|arguments: Value, _context| {
            Box::pin(async move {
                let text = arguments
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("no text");
                Ok(format!("echo: {text}"))
            })
        }),
    )
    .with_source(ToolSource::Native)
}

/// A tool that always fails.
pub(crate) fn failing_tool() -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function("fail_tool", "Always fails", json!({"type": "object"})),
        Arc::new(|_arguments, _context| {
            Box::pin(async { Err(ToolError::execution("tool execution failed")) })
        }),
    )
    .with_source(ToolSource::Native)
}

/// A tool named `name` that answers `ok`.
pub(crate) fn noop_tool(name: &str) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(name, format!("Tool {name}"), json!({"type": "object"})),
        Arc::new(|_arguments, _context| Box::pin(async { Ok("ok".to_owned()) })),
    )
    .with_source(ToolSource::Native)
}

/// A tool that waits until the round or the prompt is cancelled.
pub(crate) fn blocking_tool(name: &'static str) -> RegisteredTool {
    RegisteredTool::new(
        ToolDefinition::function(name, "Waits until cancelled", json!({"type": "object"})),
        Arc::new(|_arguments, context| {
            Box::pin(async move {
                context.cancel.cancelled().await;
                Err(ToolError::cancelled("Cancelled"))
            })
        }),
    )
    .with_source(ToolSource::Native)
}
