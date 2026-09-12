//! Running pebble's coding tools outside a coding agent.
//!
//! An application sometimes needs one of pebble's tools without a model in the
//! loop — a workflow step that reads a file the way the agent would, or one
//! that runs a command with the agent's output budgets. [`CodingToolSet`]
//! selects and describes the built-in tools, and [`ToolRunner`] executes one
//! call against an [`Environment`] through the same tool service and middleware
//! stack a session uses. The registry and dispatch engine underneath are not
//! the supported route; this facade is.

use std::fmt;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::{Message, ToolCall, ToolDefinition, ToolResult};
use pebble_agent::{ToolMiddleware, ToolOutcome, ToolSystem, TurnContext};
use tokio_util::sync::CancellationToken;

use super::execution::CodingToolService;
use super::registry::{
    RegisteredTool, ToolDefinitionWithSource, ToolEnvProvider, ToolRegistrationError, ToolRegistry,
};
use crate::SessionScope;
use crate::config::{CodingAgentOptions, NativeToolOptions};
use crate::environment::Environment;
use crate::event::{EventOptions, EventPump, EventSink, EventSinkError};
use crate::redact::{NoRedaction, Redactor};
use crate::search::seam::SearchProvider;
use crate::tools::{
    WebFetchSummarizer, make_edit_file_tool, make_glob_tool, make_grep_tool, make_read_file_tool,
    make_shell_tool_with_options, make_web_fetch_tool, make_web_search_tool, make_write_file_tool,
};
use crate::types::{CodingAgentEvent, ToolSummary};

/// A selection of tools, described the way a session would describe them.
///
/// Built-in tools are registered under their canonical names — `read_file`,
/// `shell` — which is what a call must use. An application tool joins the set
/// as it would join a session.
#[derive(Clone)]
pub struct CodingToolSet {
    registry: ToolRegistry,
}

impl CodingToolSet {
    /// No tools at all, to add to.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            registry: ToolRegistry::new(),
        }
    }

    /// The tools every built-in harness gives its model for files and
    /// commands: `read_file`, `write_file`, `edit_file`, `shell`, `grep`, and
    /// `glob`.
    #[must_use]
    pub fn core() -> Self {
        Self::core_with_options(&NativeToolOptions::default())
    }

    /// The core tools, with the command timeouts `options` names.
    #[must_use]
    pub fn core_with_options(options: &NativeToolOptions) -> Self {
        let mut set = Self::empty();
        for tool in [
            make_read_file_tool(),
            make_write_file_tool(),
            make_edit_file_tool(),
            make_shell_tool_with_options(options),
            make_grep_tool(),
            make_glob_tool(),
        ] {
            set.registry
                .register(tool)
                .expect("core tools have distinct names and identities");
        }
        set
    }

    /// Adds `web_fetch`, summarizing fetched pages through `summarizer` when
    /// one is given.
    pub fn with_web_fetch(
        self,
        summarizer: Option<Arc<WebFetchSummarizer>>,
    ) -> StdResult<Self, ToolRegistrationError> {
        self.with_tool(make_web_fetch_tool(summarizer))
    }

    /// Adds `web_search`, answering through `provider`.
    pub fn with_web_search(
        self,
        provider: Arc<dyn SearchProvider>,
    ) -> StdResult<Self, ToolRegistrationError> {
        self.with_tool(make_web_search_tool(provider))
    }

    /// Adds one tool, built-in or the application's own.
    ///
    /// Duplicate names and identities are errors; use `replace_tool` for a
    /// deliberate replacement.
    pub fn with_tool(mut self, tool: RegisteredTool) -> StdResult<Self, ToolRegistrationError> {
        self.registry.register(tool)?;
        Ok(self)
    }

    /// Replaces a tool by its stable identity, preserving its visible name.
    pub fn replace_tool(
        mut self,
        id: &str,
        tool: RegisteredTool,
    ) -> StdResult<Self, ToolRegistrationError> {
        self.registry.replace(id, tool)?;
        Ok(self)
    }

    /// The tools in the set, described as a session's event stream describes
    /// them.
    #[must_use]
    pub fn summaries(&self) -> Vec<ToolSummary> {
        self.registry
            .definitions_with_source()
            .iter()
            .map(ToolDefinitionWithSource::to_tool_summary)
            .collect()
    }

    /// The definitions a model would be sent.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.registry.definitions()
    }

    /// The names in the set.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.registry.names()
    }
}

impl fmt::Debug for CodingToolSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodingToolSet")
            .field("tools", &self.names())
            .finish()
    }
}

/// What a runner does with each event a call publishes.
pub type ToolEventCallback = Arc<dyn Fn(CodingAgentEvent) + Send + Sync>;

/// Executes one coding tool call at a time, the way a session would.
///
/// A call goes through the pipeline a session's rounds use: configured tool
/// middleware inside the same fixed output envelope. The runner projects the
/// call lifecycle onto the event pipeline. Those events reach the callback
/// given to [`on_event`](Self::on_event), in order, before
/// [`run`](Self::run) returns.
#[derive(Clone)]
pub struct ToolRunner {
    registry:          Arc<ToolRegistry>,
    environment:       Arc<dyn Environment>,
    options:           Arc<CodingAgentOptions>,
    tool_middleware:   Vec<Arc<dyn ToolMiddleware>>,
    redactor:          Option<Arc<dyn Redactor>>,
    tool_env_provider: Option<Arc<dyn ToolEnvProvider>>,
    on_event:          Option<ToolEventCallback>,
    session:           SessionScope,
}

impl ToolRunner {
    /// A runner over `tools`, acting through `environment`.
    #[must_use]
    pub fn new(tools: CodingToolSet, environment: Arc<dyn Environment>) -> Self {
        Self {
            registry: Arc::new(tools.registry),
            environment,
            options: Arc::new(CodingAgentOptions::default()),
            tool_middleware: Vec::new(),
            redactor: None,
            tool_env_provider: None,
            on_event: None,
            session: SessionScope::default(),
        }
    }

    /// Sets the output and execution options calls run under.
    #[must_use]
    pub fn options(mut self, options: CodingAgentOptions) -> Self {
        self.options = Arc::new(options);
        self
    }

    /// Adds one middleware inside the runner's fixed output envelope.
    #[must_use]
    pub fn tool_middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.tool_middleware.push(middleware);
        self
    }

    /// Sets what strips secrets out of the process output the runner
    /// publishes and out of the model-facing message of every failed call.
    #[must_use]
    pub fn redactor(mut self, redactor: Arc<dyn Redactor>) -> Self {
        self.redactor = Some(redactor);
        self
    }

    /// Sets where a call's extra environment variables come from.
    #[must_use]
    pub fn tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env_provider = Some(provider);
        self
    }

    /// Sets the identity shared by discovery, calls, and events.
    #[must_use]
    pub fn session(mut self, session: SessionScope) -> Self {
        self.session = session;
        self
    }

    /// Receives every event a call publishes, in order, before the call
    /// returns.
    #[must_use]
    pub fn on_event(mut self, callback: impl Fn(CodingAgentEvent) + Send + Sync + 'static) -> Self {
        self.on_event = Some(Arc::new(callback));
        self
    }

    /// The tools this runner can execute, described as a session would.
    #[must_use]
    pub fn summaries(&self) -> Vec<ToolSummary> {
        self.registry
            .definitions_with_source()
            .iter()
            .map(ToolDefinitionWithSource::to_tool_summary)
            .collect()
    }

    /// Executes one call and answers it.
    ///
    /// Every call is answered, including one middleware refuses, one naming a
    /// tool the runner does not have, and one cancelled through `cancel` before
    /// it finished. The result carries the same message the model would read.
    ///
    /// # Errors
    ///
    /// Returns a tool-system error when discovery or middleware fails. A tool
    /// refusal or execution failure is an ordinary `ToolResult` with
    /// `is_error` set.
    pub async fn run(
        &self,
        call: &ToolCall,
        cancel: CancellationToken,
    ) -> StdResult<ToolResult, pebble_agent::ToolSystemError> {
        let sink = self
            .on_event
            .clone()
            .map(|callback| Arc::new(CallbackSink(callback)) as Arc<dyn EventSink>);
        let (emitter, pump) = EventPump::new(EventOptions {
            sink,
            ..EventOptions::default()
        });
        let mut emitter = emitter.in_stream(self.session.root_session_id().to_string());
        if let Some(parent) = self.session.parent_session_id() {
            emitter = emitter.for_child(parent.as_str());
        }
        let pump = tokio::spawn(pump.run());

        let redactor = self
            .redactor
            .clone()
            .unwrap_or_else(|| Arc::new(NoRedaction));
        let mut service = CodingToolService::new(
            Arc::clone(&self.registry),
            Arc::clone(&self.environment),
            Arc::clone(&self.options),
            emitter.clone(),
            self.session.clone(),
            redactor,
        );
        if let Some(provider) = self.tool_env_provider.as_ref() {
            service = service.with_tool_env_provider(Arc::clone(provider));
        }
        let service = Arc::new(service);
        let mut system = ToolSystem::new(service.clone()).middleware(service.clone());
        for middleware in &self.tool_middleware {
            system = system.middleware(Arc::clone(middleware));
        }

        let messages: [Message; 0] = [];
        let result = match system
            .discover(TurnContext::new(&self.session, "standalone", 0, &messages))
            .await
        {
            Ok(catalog) => {
                service.begin_standalone(call);
                match system
                    .execute(
                        &catalog,
                        TurnContext::new(&self.session, "standalone", 0, &messages),
                        call.clone(),
                        cancel,
                    )
                    .await
                {
                    Ok(outcome) => Ok(service.complete_standalone(call, outcome)),
                    // The started event is out, so the failure is answered
                    // before it ends the run.
                    Err(error) => {
                        let _ = service.complete_standalone(
                            call,
                            ToolOutcome::failure(
                                pebble_agent::ToolErrorKind::Execution,
                                error.message(),
                            ),
                        );
                        Err(error)
                    }
                }
            }
            Err(error) => Err(error),
        };

        // Closing the pipeline publishes everything queued, so the callback has
        // seen every event by the time the caller has the result.
        let close = emitter.close().await.map_err(|source| {
            pebble_agent::ToolSystemError::with_source(
                "tool event pipeline failed",
                source.into_runtime_error(),
            )
        });
        let joined = pump
            .await
            .map_err(|source| {
                pebble_agent::ToolSystemError::with_source(
                    "tool event pipeline task failed",
                    source,
                )
            })?
            .map_err(|source| {
                pebble_agent::ToolSystemError::with_source("tool event pipeline failed", source)
            });
        close?;
        joined?;
        result
    }
}

impl fmt::Debug for ToolRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRunner")
            .field("tools", &self.registry.names())
            .field("session", &self.session)
            .field("tool_middleware", &self.tool_middleware.len())
            .field("has_event_callback", &self.on_event.is_some())
            .finish_non_exhaustive()
    }
}

/// A sink that hands each event to the runner's callback.
struct CallbackSink(ToolEventCallback);

#[async_trait]
impl EventSink for CallbackSink {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        (self.0)(event.clone());
        Ok(())
    }
}
