//! Owns the agent while the terminal accepts input independently.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use lithos_llm::Client;
use lithos_llm::middleware::RetryPolicy;
use lithos_llm::types::ReasoningEffort;
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::events::CodingAgentEvent;
use pebble_coding_agent::state::SessionRecord;
use pebble_coding_agent::subagents::SubagentOptions;
use pebble_coding_agent::tools::{PermissionLevelPolicy, PermissionMiddleware};
use pebble_coding_agent::{
    CodingAgent, CodingAgentControlHandle, CodingAgentOptions, CodingAgentSnapshot, CodingInput,
    CompactionOptions, ResumeMode, ShutdownReason, SteeringMessage,
};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::services::Services;
use super::store::{Metadata, Store};

pub(super) enum Command {
    Prompt(CodingInput, CancellationToken),
    Compact(String, CancellationToken),
    Reasoning(Option<ReasoningEffort>),
    Name(String),
    Checkpoint(oneshot::Sender<SessionRecord>),
    Shutdown,
}

use tokio::sync::oneshot;

pub(super) struct Finished {
    pub snapshot: CodingAgentSnapshot,
    pub error:    Option<String>,
    pub restored: Vec<SteeringMessage>,
    pub metadata: Metadata,
}

pub(super) struct Worker {
    pub control:       CodingAgentControlHandle,
    pub events:        broadcast::Receiver<CodingAgentEvent>,
    pub finished:      mpsc::Receiver<Finished>,
    pub snapshot:      CodingAgentSnapshot,
    pub events_closed: bool,
    sender:            mpsc::Sender<Command>,
    task:              JoinHandle<Result<()>>,
}

impl Worker {
    pub(super) async fn start(
        client: Client,
        mut metadata: Metadata,
        store: Arc<Store>,
        record: Option<SessionRecord>,
        services: Arc<Services>,
    ) -> Result<Self> {
        let environment = Arc::new(LocalEnvironment::new(&metadata.cwd));
        environment
            .prepare()
            .await
            .context("preparing the working directory")?;
        let options = CodingAgentOptions::default()
            .with_turn_replay(RetryPolicy::exponential().max_attempts(4))
            .with_reasoning_effort(metadata.reasoning)
            .with_recorded_permission_level(metadata.permission.into());
        let options = if let Some(instructions) = metadata.instructions.clone() {
            options.with_user_instructions(instructions)
        } else {
            options
        };
        let mut builder = if let Some(record) = record {
            CodingAgent::resume(
                client,
                environment,
                record,
                ResumeMode::UseModel(metadata.model.clone()),
            )
        } else {
            CodingAgent::builder(client, environment).model(&metadata.model)
        };
        builder = builder
            .options(options)
            .event_sink(store.clone())
            .human_input(services.clone());
        if metadata.approvals {
            builder = builder.tool_middleware(Arc::new(
                PermissionMiddleware::new(Arc::new(PermissionLevelPolicy::new(
                    metadata.permission.into(),
                )))
                .with_approval(services),
            ));
        } else {
            builder = builder.permission_level(metadata.permission.into());
        }
        if metadata.subagents {
            builder = builder.subagents(SubagentOptions::enabled());
        }
        let mut agent = builder
            .build()
            .await
            .context("building the interactive coding agent")?;
        metadata.model = format!("{}/{}", agent.provider(), agent.model());
        let (snapshot, events) = agent.observe().await?.into_parts();
        store.checkpoint(&metadata, agent.to_record()).await?;
        let control = agent.control_handle();
        let (sender, commands) = mpsc::channel(8);
        let (finished_tx, finished) = mpsc::channel(8);
        let task = tokio::spawn(run(agent, metadata, store, commands, finished_tx));
        Ok(Self {
            control,
            events,
            finished,
            snapshot,
            sender,
            task,
            events_closed: false,
        })
    }

    pub(super) fn send(&self, command: Command) -> Result<()> {
        self.sender
            .try_send(command)
            .context("sending an action to the coding agent")
    }

    pub(super) async fn record(&self) -> Result<SessionRecord> {
        let (sender, receiver) = oneshot::channel();
        self.send(Command::Checkpoint(sender))?;
        receiver.await.context("receiving the session checkpoint")
    }

    pub(super) async fn shutdown(self) -> Result<()> {
        self.control.close();
        let _ = self.sender.send(Command::Shutdown).await;
        // Closing the receiver keeps a final notice from blocking teardown.
        drop(self.finished);
        self.task.await.context("joining the coding agent")?
    }
}

async fn run(
    mut agent: CodingAgent,
    mut metadata: Metadata,
    store: Arc<Store>,
    mut commands: mpsc::Receiver<Command>,
    finished: mpsc::Sender<Finished>,
) -> Result<()> {
    let result = async {
        while let Some(command) = commands.recv().await {
            let operation = match command {
                Command::Prompt(input, cancel) => agent
                    .prompt_with_cancellation(input, &cancel)
                    .await
                    .map(|_| ()),
                Command::Compact(instructions, cancel) => {
                    let options = if instructions.is_empty() {
                        CompactionOptions::new()
                    } else {
                        CompactionOptions::new().instructions(instructions)
                    };
                    agent
                        .compact_with_cancellation(options, &cancel)
                        .await
                        .map(|_| ())
                }
                Command::Reasoning(effort) => {
                    agent.set_reasoning_effort(effort);
                    metadata.reasoning = effort;
                    Ok(())
                }
                Command::Name(name) => {
                    metadata.name = name;
                    Ok(())
                }
                Command::Checkpoint(reply) => {
                    let _ = reply.send(agent.to_record());
                    continue;
                }
                Command::Shutdown => break,
            };
            let (steering, follow_ups) = agent.control_handle().take_pending_input().into_parts();
            let restored = steering.into_iter().chain(follow_ups).collect();
            store.checkpoint(&metadata, agent.to_record()).await?;
            let notice = Finished {
                snapshot: agent.snapshot(),
                error: operation.err().map(|error| format!("{error:#}")),
                restored,
                metadata: metadata.clone(),
            };
            if finished.send(notice).await.is_err() {
                break;
            }
        }
        Result::<()>::Ok(())
    }
    .await;
    let shutdown = agent.shutdown(ShutdownReason::Completed).await;
    result?;
    shutdown.context("shutting down the interactive agent")?;
    store.checkpoint(&metadata, agent.to_record()).await
}
