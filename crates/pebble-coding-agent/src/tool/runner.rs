//! Running pebble's coding tools outside a coding agent.
//!
//! An application sometimes needs one of pebble's tools without a model in the
//! loop — a hook that reads a file the way the agent would, a workflow step
//! that runs a command with the agent's output budgets. [`CodingToolSet`]
//! selects and describes the built-in tools, and [`ToolRunner`] executes one
//! call against an [`Environment`] through the same pipeline a session uses:
//! the same access policy, the same hooks, the same error rendering, and the
//! same output limits. The registry and the dispatch engine underneath are not
//! the supported route; this facade is.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use lithos_llm::types::{ToolCall, ToolDefinition, ToolResult};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use super::execution::ToolDispatch;
use super::registry::{RegisteredTool, ToolDefinitionWithSource, ToolEnvProvider, ToolRegistry};
use crate::config::{CodingAgentOptions, NativeToolOptions};
use crate::environment::Environment;
use crate::event::{EventOptions, EventPump, EventSink, EventSinkError};
use crate::redact::Redactor;
use crate::search::SearchProvider;
use crate::tools::{
    WebFetchSummarizer, make_edit_file_tool, make_glob_tool, make_grep_tool, make_read_file_tool,
    make_shell_tool_with_options, make_web_fetch_tool, make_web_search_tool, make_write_file_tool,
};
use crate::types::{CodingAgentEvent, ToolSummary};

/// The session identifier a runner stamps on its events when the application
/// names none.
const DEFAULT_SESSION_ID: &str = "standalone";

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
        Self::empty()
            .with_tool(make_read_file_tool())
            .with_tool(make_write_file_tool())
            .with_tool(make_edit_file_tool())
            .with_tool(make_shell_tool_with_options(options))
            .with_tool(make_grep_tool())
            .with_tool(make_glob_tool())
    }

    /// Adds `web_fetch`, summarizing fetched pages through `summarizer` when
    /// one is given.
    #[must_use]
    pub fn with_web_fetch(self, summarizer: Option<Arc<WebFetchSummarizer>>) -> Self {
        self.with_tool(make_web_fetch_tool(summarizer))
    }

    /// Adds `web_search`, answering through `provider`.
    #[must_use]
    pub fn with_web_search(self, provider: Arc<dyn SearchProvider>) -> Self {
        self.with_tool(make_web_search_tool(provider))
    }

    /// Adds one tool, built-in or the application's own.
    ///
    /// A tool with a name already in the set replaces it.
    #[must_use]
    pub fn with_tool(mut self, tool: RegisteredTool) -> Self {
        self.registry.register(tool);
        self
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
/// A call goes through the pipeline a session's rounds use: the access policy
/// and hooks on the [`CodingAgentOptions`], the same rendering of a refusal or
/// a failure, and the same output budgets. The events the call publishes reach
/// the callback given to [`on_event`](Self::on_event), in order, before
/// [`run`](Self::run) returns.
#[derive(Clone)]
pub struct ToolRunner {
    registry:          ToolRegistry,
    environment:       Arc<dyn Environment>,
    options:           CodingAgentOptions,
    redactor:          Option<Arc<dyn Redactor>>,
    tool_env_provider: Option<Arc<dyn ToolEnvProvider>>,
    on_event:          Option<ToolEventCallback>,
    session_id:        String,
}

impl ToolRunner {
    /// A runner over `tools`, acting through `environment`.
    #[must_use]
    pub fn new(tools: CodingToolSet, environment: Arc<dyn Environment>) -> Self {
        Self {
            registry: tools.registry,
            environment,
            options: CodingAgentOptions::default(),
            redactor: None,
            tool_env_provider: None,
            on_event: None,
            session_id: DEFAULT_SESSION_ID.to_owned(),
        }
    }

    /// Sets the policy, hooks, and output budgets calls run under.
    ///
    /// The same record a session takes, read the same way: the access policy
    /// and exposure mode decide whether a call runs, the hooks bracket it, and
    /// the output budgets bound what it answers with.
    #[must_use]
    pub fn options(mut self, options: CodingAgentOptions) -> Self {
        self.options = options;
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

    /// Names the session the runner's events are stamped with.
    #[must_use]
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
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
    /// Every call is answered, including one the policy refuses, one the hooks
    /// block, one naming a tool the runner does not have, and one cancelled
    /// through `cancel` before it finished: the result carries the same message
    /// the model would have read.
    pub async fn run(&self, call: &ToolCall, cancel: CancellationToken) -> ToolResult {
        let sink = self
            .on_event
            .clone()
            .map(|callback| Arc::new(CallbackSink(callback)) as Arc<dyn EventSink>);
        let (emitter, pump) = EventPump::new(EventOptions {
            sink,
            ..EventOptions::default()
        });
        let pump = tokio::spawn(pump.run());

        let result = {
            let mut dispatch = ToolDispatch::new(
                &self.registry,
                &self.environment,
                &self.options,
                &emitter,
                &self.session_id,
                &self.session_id,
            );
            if let Some(provider) = &self.tool_env_provider {
                dispatch = dispatch.with_tool_env_provider(provider);
            }
            if let Some(redactor) = &self.redactor {
                dispatch = dispatch.with_redactor(redactor);
            }
            dispatch.execute_one(call, cancel).await
        };

        // Closing the pipeline publishes everything queued, so the callback has
        // seen every event by the time the caller has the result.
        drop(emitter);
        match pump.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => warn!(%error, "a tool runner's event callback failed"),
            Err(error) => warn!(%error, "a tool runner's event pump task failed"),
        }
        result
    }
}

impl fmt::Debug for ToolRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRunner")
            .field("tools", &self.registry.names())
            .field("session_id", &self.session_id)
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
