//! Session identity reaches application hooks and survives restoration.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lithos_llm::types::ToolCall;
use pebble_agent::{
    Agent, SessionId, SessionScope, Tool, ToolCallNext, ToolCallRequest, ToolCatalog,
    ToolDescriptor, ToolDiscoveryNext, ToolMiddleware, ToolOutcome, ToolOutput, ToolService,
    ToolSystem, ToolSystemError, TurnContext,
};
use pebble_coding_agent::extensions::{Answer, HumanInputError, HumanInputProvider, Question};
use pebble_coding_agent::state::SessionRecord;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, scripted_client, text_response, tool_call_response,
};
use pebble_coding_agent::tools::{
    CodingToolSet, PermissionMiddleware, RegisteredTool, ToolPermission, ToolPermissionPolicy,
    ToolRunner,
};
use pebble_coding_agent::{CodingAgent, ResumeMode, ShutdownReason};
use serde_json::json;
use tokio::sync::Barrier;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct Audit(Mutex<Vec<(&'static str, SessionScope)>>);

impl Audit {
    fn record(&self, phase: &'static str, session: &SessionScope) {
        self.0
            .lock()
            .expect("audit lock")
            .push((phase, session.clone()));
    }

    fn saw(&self, phase: &'static str, session: &SessionScope) -> bool {
        self.0
            .lock()
            .expect("audit lock")
            .contains(&(phase, session.clone()))
    }
}

#[async_trait]
impl ToolMiddleware for Audit {
    async fn discover(
        &self,
        context: TurnContext<'_>,
        next: ToolDiscoveryNext<'_>,
    ) -> Result<ToolCatalog, ToolSystemError> {
        self.record("discovery", context.session());
        next.run(context).await
    }

    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> Result<ToolOutcome, ToolSystemError> {
        self.record("call", request.session());
        next.run(request).await
    }
}

#[tokio::test]
async fn generic_tools_and_middleware_share_the_agent_scope() {
    let scope = SessionScope::default().child(SessionId::new("child"));
    let audit = Arc::new(Audit::default());
    let tool_audit = audit.clone();
    let tool = Tool::function(
        "inspect",
        "Inspect",
        json!({"type":"object"}),
        move |context, _| {
            tool_audit.record("tool", context.session());
            async { Ok(ToolOutput::from("ok")) }
        },
    )
    .expect("tool builds");
    let (client, _) = scripted_client(vec![
        ScriptedCall::response(tool_call_response("inspect", "call", json!({}))),
        ScriptedCall::response(text_response("done")),
    ]);
    let mut agent = Agent::builder(client, "test/model")
        .session(scope.clone())
        .tools([tool])
        .tool_middleware(audit.clone())
        .build()
        .expect("builds");
    agent.prompt("inspect").await.expect("answers");
    assert_eq!(agent.snapshot().session(), &scope);
    for phase in ["discovery", "call", "tool"] {
        assert!(audit.saw(phase, &scope));
    }
    agent.shutdown();
}

async fn run_coding_scope(scope: SessionScope, audit: Arc<Audit>, barrier: Arc<Barrier>) {
    let tool_audit = audit.clone();
    let expected = scope.clone();
    let tool = RegisteredTool::function(
        "inspect",
        "Inspect",
        json!({"type":"object"}),
        move |context, _| {
            let barrier = barrier.clone();
            assert_eq!(context.session(), &expected);
            tool_audit.record("tool", context.session());
            async move {
                barrier.wait().await;
                Ok("ok".to_owned())
            }
        },
    );
    let (client, _) = scripted_client(vec![
        ScriptedCall::response(tool_call_response("inspect", "call", json!({}))),
        ScriptedCall::response(text_response("done")),
    ]);
    let mut record = SessionRecord::new(scope.clone());
    record.provider = Some("test".to_owned());
    record.model = Some("model".to_owned());
    let mut agent = CodingAgent::resume(
        client,
        Arc::new(MockEnvironment::linux()),
        record,
        ResumeMode::RecordedModel,
    )
    .tools([tool])
    .tool_middleware(audit.clone())
    .build()
    .await
    .expect("restores");
    let mut events = agent.subscribe();
    agent.prompt("inspect").await.result.expect("answers");
    assert_eq!(agent.session(), &scope);
    assert_eq!(agent.snapshot().session(), &scope);
    assert_eq!(agent.to_record().scope, scope);
    for phase in ["discovery", "call", "tool"] {
        assert!(audit.saw(phase, &scope));
    }
    let mut count = 0;
    while let Ok(event) = events.try_recv() {
        assert_eq!(event.session_id, scope.session_id().as_str());
        assert_eq!(event.stream_id, scope.root_session_id().as_str());
        assert_eq!(
            event.parent_session_id.as_deref(),
            scope.parent_session_id().map(SessionId::as_str)
        );
        count += 1;
    }
    assert!(count > 0);
    let mut export = agent.export();
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shuts down");
    export.advance_event_cursor(agent.committed_event_seq());
    let (client, _) = scripted_client(vec![ScriptedCall::response(text_response("resumed"))]);
    let mut successor =
        CodingAgent::resume_from_export(client, Arc::new(MockEnvironment::linux()), export)
            .build()
            .await
            .expect("warm restoration succeeds");
    assert_eq!(successor.session(), &scope);
    assert_eq!(successor.to_record().scope, scope);
    successor
        .prompt("continue")
        .await
        .result
        .expect("continues");
    successor
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shuts down");
}

#[tokio::test]
async fn concurrent_trees_and_restored_grandchildren_keep_their_own_identity() {
    let audit = Arc::new(Audit::default());
    let barrier = Arc::new(Barrier::new(2));
    let root = SessionScope::default();
    let grandchild = SessionScope::default()
        .child(SessionId::new("parent"))
        .child(SessionId::new("grandchild"));
    assert_eq!(grandchild.depth(), 2);
    tokio::join!(
        run_coding_scope(root, audit.clone(), barrier.clone()),
        run_coding_scope(grandchild, audit, barrier)
    );
}

struct RootMaySpawn;

impl ToolPermissionPolicy for RootMaySpawn {
    fn permission(&self, session: &SessionScope, tool: &ToolDescriptor) -> ToolPermission {
        if tool.id().as_str() == "spawn_agent" && !session.is_root() {
            ToolPermission::Deny {
                reason: "children may not spawn".to_owned(),
            }
        } else {
            ToolPermission::Allow
        }
    }
}

struct SpawnTool(ToolDescriptor);

#[async_trait]
impl ToolService for SpawnTool {
    async fn discover(&self, _context: TurnContext<'_>) -> Result<ToolCatalog, ToolSystemError> {
        Ok(ToolCatalog::new([self.0.clone()]))
    }
    async fn call(&self, _request: ToolCallRequest) -> Result<ToolOutcome, ToolSystemError> {
        panic!("a denied spawn must not reach the executor")
    }
}

#[tokio::test]
async fn child_spawn_is_hidden_and_a_call_with_an_existing_descriptor_is_denied() {
    let tool = Tool::function(
        "spawn_agent",
        "Spawn",
        json!({"type":"object"}),
        |_, _| async { Ok(ToolOutput::from("spawned")) },
    )
    .expect("tool builds");
    let system = ToolSystem::new(Arc::new(SpawnTool(tool.descriptor().clone())))
        .middleware(Arc::new(PermissionMiddleware::new(Arc::new(RootMaySpawn))));
    let root = SessionScope::default();
    let child = root.child(SessionId::new("child"));
    let root_context = TurnContext::new(&root, "test/model", 0, &[]);
    let child_context = TurnContext::new(&child, "test/model", 0, &[]);
    let root_catalog = system.discover(root_context).await.expect("discovers");
    assert_eq!(root_catalog.visible_tools().count(), 1);
    assert_eq!(
        system
            .discover(child_context)
            .await
            .expect("discovers")
            .visible_tools()
            .count(),
        0
    );
    let result = system
        .execute(
            &root_catalog,
            child_context,
            ToolCall::function("call", "spawn_agent", json!({})),
            CancellationToken::new(),
        )
        .await
        .expect("policy answers");
    assert_eq!(
        result.error_kind(),
        Some(pebble_agent::ToolErrorKind::Denied)
    );
}

#[tokio::test]
async fn standalone_runners_have_distinct_stable_scopes() {
    let audit = Arc::new(Audit::default());
    let tool_audit = audit.clone();
    let tool = RegisteredTool::function(
        "inspect",
        "Inspect",
        json!({"type":"object"}),
        move |context, _| {
            tool_audit.record("tool", context.session());
            async { Ok("ok".to_owned()) }
        },
    );
    let tools = CodingToolSet::empty()
        .with_tool(tool)
        .expect("tool registers");
    let first = ToolRunner::new(tools.clone(), Arc::new(MockEnvironment::linux()))
        .tool_middleware(audit.clone());
    let second =
        ToolRunner::new(tools, Arc::new(MockEnvironment::linux())).tool_middleware(audit.clone());
    let call = ToolCall::function("call", "inspect", json!({}));
    for runner in [&first, &first, &second] {
        runner
            .run(&call, CancellationToken::new())
            .await
            .expect("runs");
    }
    let calls: Vec<_> = audit
        .0
        .lock()
        .expect("audit lock")
        .iter()
        .filter(|(phase, _)| *phase == "tool")
        .map(|(_, scope)| scope.clone())
        .collect();
    assert_eq!(calls[0], calls[1]);
    assert_ne!(calls[0], calls[2]);
    for scope in calls {
        assert!(scope.is_root());
        assert!(audit.saw("discovery", &scope));
        assert!(audit.saw("call", &scope));
    }
}

struct UnusedHumanInput;

#[async_trait]
impl HumanInputProvider for UnusedHumanInput {
    async fn ask_questions(
        &self,
        _call_id: &str,
        _questions: Vec<Question>,
        _cancel: CancellationToken,
    ) -> Result<Vec<Answer>, HumanInputError> {
        panic!("discovery must not ask a person")
    }
}

#[tokio::test]
async fn restored_children_do_not_advertise_root_only_human_input() {
    let root = SessionScope::default();
    for scope in [root.clone(), root.child(SessionId::new("child"))] {
        let (client, provider) = scripted_client(vec![]);
        let mut record = SessionRecord::new(scope.clone());
        record.provider = Some("test".to_owned());
        record.model = Some("model".to_owned());
        let mut agent = CodingAgent::resume(
            client,
            Arc::new(MockEnvironment::linux()),
            record,
            ResumeMode::RecordedModel,
        )
        .human_input(Arc::new(UnusedHumanInput))
        .build()
        .await
        .expect("restores");
        assert_eq!(
            agent
                .snapshot()
                .tools()
                .iter()
                .any(|tool| tool.name == "AskUserQuestion"),
            scope.is_root()
        );
        assert_eq!(provider.call_count(), 0);
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shuts down");
    }
}
