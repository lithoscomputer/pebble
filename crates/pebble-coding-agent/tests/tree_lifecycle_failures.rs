//! Lifecycle failures while real children and grandchildren execute tools.

use std::collections::BTreeMap;
use std::error::Error as _;
use std::future::pending;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{poll, stream};
use lithos_llm::Client;
use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::AdapterId;
use lithos_llm::types::{
    Error as LlmError, ErrorKind as LlmErrorKind, Response, ResponseStream, Role,
};
use pebble_agent::{ToolCallNext, ToolCallRequest, ToolMiddleware, ToolOutcome, ToolSystemError};
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent, EventSink, EventSinkError};
use pebble_coding_agent::subagents::{SubagentLimits, SubagentOptions};
use pebble_coding_agent::test_support::{
    MockEnvironment, events_for, message_text, test_catalog, tool_call_response,
};
use pebble_coding_agent::tools::{RegisteredTool, ToolContext, ToolError};
use pebble_coding_agent::{CodingAgent, Error, SessionScope, ShutdownReason};
use serde_json::json;
use tokio::runtime::Handle;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{Notify, mpsc};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const PATIENCE: Duration = Duration::from_secs(15);
const FAILURE_MESSAGE: &str = "injected descendant output failure";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Actor {
    Child,
    Grandchild,
}

impl Actor {
    const fn name(self) -> &'static str {
        match self {
            Self::Child => "child",
            Self::Grandchild => "grandchild",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Topology {
    Child,
    Grandchild,
}

impl Topology {
    const fn actors(self) -> &'static [Actor] {
        match self {
            Self::Child => &[Actor::Child],
            Self::Grandchild => &[Actor::Child, Actor::Grandchild],
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RootBoundary {
    Model,
    Wait,
}

#[derive(Clone, Copy, Debug)]
enum Failure {
    Sink(Actor),
    Drop,
}

#[derive(Debug)]
struct Started {
    scope:  SessionScope,
    cancel: CancellationToken,
}

#[derive(Debug)]
enum Ready {
    Root,
    Tool(Actor, Started),
    Unexpected(String),
}

/// Routes by this session's task and history, independent of request order
/// across sessions. Only model responses and the pending model are simulated.
struct TreeModel {
    id:       AdapterId,
    topology: Topology,
    boundary: RootBoundary,
    ready:    mpsc::UnboundedSender<Ready>,
}

impl TreeModel {
    fn unexpected(&self, message: String) -> LlmError {
        self.ready
            .send(Ready::Unexpected(message.clone()))
            .expect("readiness receiver lives");
        LlmError::new(LlmErrorKind::Middleware, message)
    }
}

#[async_trait]
impl ProviderAdapter for TreeModel {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, _call: &ResolvedCall) -> Result<Response, LlmError> {
        Err(self.unexpected("tree fixture does not expect a completion".to_owned()))
    }

    async fn stream(&self, call: &ResolvedCall) -> Result<ResponseStream, LlmError> {
        let messages = call.request().messages();
        let task = messages
            .iter()
            .find(|message| message.role() == Role::User)
            .map(message_text)
            .unwrap_or_default();
        let rounds = messages
            .iter()
            .filter(|message| message.role() == Role::Assistant)
            .count();
        let response = match (task.as_str(), rounds, self.topology) {
            ("root", 0, _) => {
                tool_call_response("spawn_agent", "spawn-child", json!({"task":"child"}))
            }
            ("root", 1, _) => match self.boundary {
                RootBoundary::Model => {
                    self.ready
                        .send(Ready::Root)
                        .expect("readiness receiver lives");
                    return pending().await;
                }
                RootBoundary::Wait => tool_call_response("wait", "root-wait", json!({})),
            },
            ("child", 0, Topology::Grandchild) => tool_call_response(
                "spawn_agent",
                "spawn-grandchild",
                json!({"task":"grandchild"}),
            ),
            ("child", 0, Topology::Child) | ("child", 1, Topology::Grandchild) => {
                tool_call_response("probe", "child-probe", json!({"actor":"child"}))
            }
            ("grandchild", 0, Topology::Grandchild) => {
                tool_call_response("probe", "grandchild-probe", json!({"actor":"grandchild"}))
            }
            _ => {
                return Err(self.unexpected(
                    format!(
                        "unexpected tree request: task={task:?}, rounds={rounds}, topology={:?}, messages={messages:?}",
                        self.topology
                    ),
                ));
            }
        };
        Ok(ResponseStream::new(stream::iter(
            events_for(&response)
                .into_iter()
                .map(|item| Ok(item.expect("response contains no scripted errors"))),
        )))
    }
}

/// Observe the real wait future after it has been polled to Pending. A
/// queued ToolCallStarted event alone does not prove the wait is executing.
struct WaitObserver(mpsc::UnboundedSender<Ready>);

#[async_trait]
impl ToolMiddleware for WaitObserver {
    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> Result<ToolOutcome, ToolSystemError> {
        if request.call().id != "root-wait" {
            return next.run(request).await;
        }
        let mut execution = Box::pin(next.run(request));
        assert!(
            poll!(execution.as_mut()).is_pending(),
            "wait must block on a running child"
        );
        self.0.send(Ready::Root).expect("readiness receiver lives");
        execution.await
    }
}

/// Owns the executor's release and observes token handling separately from
/// future destruction. No observer task can outlive the fixture.
struct Probe {
    actor:      Actor,
    ready:      mpsc::UnboundedSender<Ready>,
    release:    Notify,
    active:     AtomicBool,
    cooperated: AtomicBool,
    dropped:    AtomicBool,
}

impl Probe {
    fn new(actor: Actor, ready: mpsc::UnboundedSender<Ready>) -> Self {
        Self {
            actor,
            ready,
            release: Notify::new(),
            active: AtomicBool::new(false),
            cooperated: AtomicBool::new(false),
            dropped: AtomicBool::new(false),
        }
    }

    async fn run(&self, context: ToolContext) -> Result<String, ToolError> {
        assert!(
            !self.active.swap(true, Ordering::SeqCst),
            "duplicate {:?} executor",
            self.actor
        );
        let _execution = Execution(self);
        self.ready
            .send(Ready::Tool(self.actor, Started {
                scope:  context.session().clone(),
                cancel: context.cancel().clone(),
            }))
            .expect("readiness receiver lives");
        tokio::select! {
            () = context.cancel().cancelled() => {}
            () = self.release.notified() => {
                context.emit_coding_event(CodingEvent::ToolCallOutputDelta { delta: self.actor.name().to_owned() });
                context.cancel().cancelled().await;
            }
        }
        self.cooperated.store(true, Ordering::SeqCst);
        Err(ToolError::cancelled("probe stopped"))
    }
}

struct Execution<'a>(&'a Probe);

impl Drop for Execution<'_> {
    fn drop(&mut self) {
        self.0.active.store(false, Ordering::SeqCst);
        self.0.dropped.store(true, Ordering::SeqCst);
    }
}

#[derive(Debug, thiserror::Error)]
#[error("storage refused descendant output")]
struct InjectedFailure;

struct TreeSink {
    failure:  Failure,
    accepted: Mutex<Vec<CodingAgentEvent>>,
    rejected: Mutex<Option<CodingAgentEvent>>,
}

impl TreeSink {
    fn events(&self) -> Vec<CodingAgentEvent> {
        self.accepted.lock().expect("accepted events lock").clone()
    }
}

#[async_trait]
impl EventSink for TreeSink {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        if let Failure::Sink(actor) = self.failure
            && matches!(&event.event, CodingEvent::ToolCallOutputDelta { delta } if delta == actor.name())
        {
            let previous = self
                .rejected
                .lock()
                .expect("rejected event lock")
                .replace(event.clone());
            assert!(previous.is_none(), "sink called again after rejection");
            return Err(EventSinkError::new(FAILURE_MESSAGE).with_source(InjectedFailure));
        }
        self.accepted
            .lock()
            .expect("accepted events lock")
            .push(event.clone());
        Ok(())
    }
}

async fn await_readiness(
    receiver: &mut mpsc::UnboundedReceiver<Ready>,
    topology: Topology,
) -> BTreeMap<Actor, Started> {
    let mut root_ready = false;
    let mut started = BTreeMap::new();
    timeout(PATIENCE, async {
        while !root_ready || started.len() != topology.actors().len() {
            match receiver.recv().await.expect("readiness senders live") {
                Ready::Unexpected(message) => panic!("{message}"),
                Ready::Root => {
                    assert!(!root_ready, "root reached its boundary twice");
                    root_ready = true;
                }
                Ready::Tool(actor, execution) => {
                    assert!(topology.actors().contains(&actor), "unexpected {actor:?}");
                    assert!(started.insert(actor, execution).is_none(), "duplicate {actor:?}");
                }
            }
        }
    }).await.unwrap_or_else(|_| panic!("tree readiness timed out: topology={topology:?}, root_ready={root_ready}, started={started:?}"));
    started
}

fn assert_original_failure(error: &Error) {
    let Error::EventSink(failure) = error else {
        panic!("original sink failure was replaced: {error:?}");
    };
    assert_eq!(failure.message(), FAILURE_MESSAGE);
    assert!(
        failure
            .source()
            .expect("sink source preserved")
            .is::<InjectedFailure>(),
        "wrong source: {failure:?}"
    );
}

fn assert_tree(events: &[CodingAgentEvent], root: &str, started: &BTreeMap<Actor, Started>) {
    let sessions: Vec<_> = events
        .iter()
        .filter(|event| matches!(event.event, CodingEvent::SessionStarted { .. }))
        .collect();
    assert_eq!(sessions.len(), started.len() + 1);
    for (actor, execution) in started {
        assert_eq!(execution.scope.root_session_id().as_str(), root);
        assert!(!execution.scope.is_root());
        let parent = match actor {
            Actor::Child => root,
            Actor::Grandchild => started[&Actor::Child].scope.session_id().as_str(),
        };
        let child: Vec<_> = sessions
            .iter()
            .filter(|event| event.session_id == execution.scope.session_id().as_str())
            .collect();
        assert_eq!(child.len(), 1, "one start for {actor:?}");
        assert_eq!(child[0].parent_session_id.as_deref(), Some(parent));
    }
    assert_eq!(
        sessions
            .iter()
            .filter(|event| event.session_id == root && event.parent_session_id.is_none())
            .count(),
        1
    );
}

fn assert_terminal_events(
    events: &[CodingAgentEvent],
    root: &str,
    started: &BTreeMap<Actor, Started>,
) {
    let ended: Vec<_> = events
        .iter()
        .filter(|event| matches!(event.event, CodingEvent::SessionEnded))
        .collect();
    assert_eq!(ended.len(), started.len() + 1);
    let ended_at = |id: &str| {
        let matching: Vec<_> = ended
            .iter()
            .filter(|event| event.session_id == id)
            .collect();
        assert_eq!(matching.len(), 1, "one terminal event for {id}");
        matching[0].seq
    };
    let root_end = ended_at(root);
    let child_end = ended_at(started[&Actor::Child].scope.session_id().as_str());
    assert!(child_end < root_end, "child ends before root");
    if let Some(grandchild) = started.get(&Actor::Grandchild) {
        assert!(
            ended_at(grandchild.scope.session_id().as_str()) < child_end,
            "grandchild ends before child"
        );
    }
}

async fn check_case(topology: Topology, boundary: RootBoundary, failure: Failure) {
    let baseline_tasks = Handle::current().metrics().num_alive_tasks();
    let (ready, mut receiver) = mpsc::unbounded_channel();
    let probes: Arc<Vec<_>> = Arc::new(
        topology
            .actors()
            .iter()
            .map(|actor| Probe::new(*actor, ready.clone()))
            .collect(),
    );
    let executors = Arc::clone(&probes);
    let tool = RegisteredTool::function(
        "probe",
        "Report readiness and wait for cancellation",
        json!({"type":"object", "properties":{"actor":{"type":"string"}}, "required":["actor"]}),
        move |context, arguments| {
            let executors = Arc::clone(&executors);
            async move {
                let probe = executors
                    .iter()
                    .find(|probe| Some(probe.actor.name()) == arguments["actor"].as_str())
                    .expect("known probe actor");
                probe.run(context).await
            }
        },
    )
    .allow_in_subagents();
    let sink = Arc::new(TreeSink {
        failure,
        accepted: Mutex::new(Vec::new()),
        rejected: Mutex::new(None),
    });
    let client = Client::builder()
        .catalog(test_catalog())
        .adapter_arc(
            "test",
            Arc::new(TreeModel {
                id: AdapterId::new("test-adapter"),
                topology,
                boundary,
                ready: ready.clone(),
            }),
        )
        .build()
        .expect("fixture client builds")
        .client;
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .subagents(SubagentOptions::enabled().with_limits(SubagentLimits::new(3)))
        .tools([tool])
        .tool_middleware(Arc::new(WaitObserver(ready)))
        .event_sink(sink.clone())
        .build()
        .await
        .expect("tree builds");
    let root = agent.id().to_owned();
    let control = agent.control_handle();
    let cursor = agent.committed_event_seq();
    let mut subscriber = agent.subscribe();
    let mut operation = Box::pin(agent.prompt("root"));
    let started = tokio::select! {
        result = &mut operation => panic!("root returned before readiness: {result:?}; events={:?}", sink.events()),
        started = await_readiness(&mut receiver, topology) => started,
    };
    assert!(control.is_running());
    for probe in probes.iter() {
        assert!(
            probe.active.load(Ordering::SeqCst),
            "{:?} is executing",
            probe.actor
        );
        assert!(!started[&probe.actor].cancel.is_cancelled());
    }
    assert_tree(&sink.events(), &root, &started);
    match failure {
        Failure::Sink(actor) => {
            probes
                .iter()
                .find(|probe| probe.actor == actor)
                .expect("target exists")
                .release
                .notify_one();
            let error = timeout(PATIENCE, operation)
                .await
                .expect("root stops on descendant sink failure")
                .result
                .expect_err("sink refused output");
            assert_original_failure(&error);
        }
        Failure::Drop => drop(operation),
    }
    timeout(PATIENCE, control.wait_for_idle())
        .await
        .expect("idle waiters finish");
    assert!(control.is_closed());
    assert!(!control.is_running());
    assert!(matches!(
        agent.prompt("again").await.result,
        Err(Error::SessionClosed)
    ));
    // Explicit shutdown must join tasks. It must not be what first delivers
    // the cancellation this failure was already required to propagate.
    timeout(PATIENCE, async {
        for execution in started.values() {
            execution.cancel.cancelled().await;
        }
    })
    .await
    .expect("failure cancels every descendant before explicit shutdown");
    let shutdown = timeout(PATIENCE, agent.shutdown(ShutdownReason::Cancelled))
        .await
        .expect("shutdown joins descendants");
    match failure {
        Failure::Drop => {
            shutdown.expect("healthy sink allows shutdown");
        }
        Failure::Sink(_) => {
            if let Err(error) = shutdown {
                assert_original_failure(&error);
            }
        }
    }
    for (actor, execution) in &started {
        assert!(execution.cancel.is_cancelled(), "{actor:?} token cancelled");
    }
    for probe in probes.iter() {
        assert!(
            !probe.active.load(Ordering::SeqCst),
            "{:?} executor stopped",
            probe.actor
        );
        assert!(
            probe.dropped.load(Ordering::SeqCst),
            "{:?} executor left scope",
            probe.actor
        );
        assert!(
            probe.cooperated.load(Ordering::SeqCst),
            "{:?} tool observed cancellation",
            probe.actor
        );
    }
    let delivered = timeout(PATIENCE, async {
        let mut delivered = Vec::new();
        loop {
            match subscriber.recv().await {
                Ok(event) => delivered.push(event),
                Err(RecvError::Closed) => return delivered,
                Err(error) => panic!("subscriber lost events: {error}"),
            }
        }
    })
    .await
    .expect("subscriber drains and closes");
    let accepted = sink.events();
    assert!(accepted.iter().all(|event| event.stream_id == root));
    assert_eq!(accepted.first().expect("events exist").seq, 1);
    assert!(
        accepted
            .windows(2)
            .all(|pair| pair[1].seq == pair[0].seq + 1),
        "accepted events form one ordered prefix"
    );
    assert_eq!(
        delivered,
        accepted
            .iter()
            .filter(|event| event.seq > cursor)
            .cloned()
            .collect::<Vec<_>>()
    );
    match failure {
        Failure::Sink(actor) => {
            let rejected = sink
                .rejected
                .lock()
                .expect("rejected event lock")
                .clone()
                .expect("failure reached the sink");
            assert_eq!(
                rejected.session_id,
                started[&actor].scope.session_id().as_str()
            );
            assert_eq!(
                rejected.seq,
                accepted.last().expect("prefix exists").seq + 1
            );
            assert!(delivered.iter().all(|event| event.seq < rejected.seq));
        }
        Failure::Drop => assert_terminal_events(&accepted, &root, &started),
    }
    let again = timeout(PATIENCE, agent.shutdown(ShutdownReason::Cancelled))
        .await
        .expect("repeated shutdown completes");
    match failure {
        Failure::Drop => {
            again.expect("repeated shutdown succeeds");
        }
        Failure::Sink(_) => {
            if let Err(error) = again {
                assert_original_failure(&error);
            }
        }
    }
    assert_eq!(
        sink.events(),
        accepted,
        "shutdown does not duplicate events"
    );
    drop(agent);
    drop(control);
    drop(subscriber);
    drop(receiver);
    drop(probes);
    drop(sink);
    assert_eq!(
        Handle::current().metrics().num_alive_tasks(),
        baseline_tasks,
        "no runner, monitor, cleanup task, or event pump survives shutdown"
    );
}

#[tokio::test]
async fn child_sink_failure_while_root_awaits_model() {
    check_case(
        Topology::Child,
        RootBoundary::Model,
        Failure::Sink(Actor::Child),
    )
    .await;
}
#[tokio::test]
async fn child_sink_failure_while_root_waits_for_children() {
    check_case(
        Topology::Child,
        RootBoundary::Wait,
        Failure::Sink(Actor::Child),
    )
    .await;
}
#[tokio::test]
async fn dropped_root_model_future_cancels_child() {
    check_case(Topology::Child, RootBoundary::Model, Failure::Drop).await;
}
#[tokio::test]
async fn dropped_root_wait_future_cancels_child() {
    check_case(Topology::Child, RootBoundary::Wait, Failure::Drop).await;
}
#[tokio::test]
async fn child_sink_failure_cancels_grandchild_while_root_awaits_model() {
    check_case(
        Topology::Grandchild,
        RootBoundary::Model,
        Failure::Sink(Actor::Child),
    )
    .await;
}
#[tokio::test]
async fn child_sink_failure_cancels_grandchild_while_root_waits_for_children() {
    check_case(
        Topology::Grandchild,
        RootBoundary::Wait,
        Failure::Sink(Actor::Child),
    )
    .await;
}
#[tokio::test]
async fn grandchild_sink_failure_while_root_awaits_model() {
    check_case(
        Topology::Grandchild,
        RootBoundary::Model,
        Failure::Sink(Actor::Grandchild),
    )
    .await;
}
#[tokio::test]
async fn grandchild_sink_failure_while_root_waits_for_children() {
    check_case(
        Topology::Grandchild,
        RootBoundary::Wait,
        Failure::Sink(Actor::Grandchild),
    )
    .await;
}
#[tokio::test]
async fn dropped_root_model_future_cancels_child_and_grandchild() {
    check_case(Topology::Grandchild, RootBoundary::Model, Failure::Drop).await;
}
#[tokio::test]
async fn dropped_root_wait_future_cancels_child_and_grandchild() {
    check_case(Topology::Grandchild, RootBoundary::Wait, Failure::Drop).await;
}
