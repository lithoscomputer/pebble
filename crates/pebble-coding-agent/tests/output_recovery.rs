//! Output stays recoverable when the environment and the store share no paths.

use std::collections::HashMap;
use std::env::temp_dir;
use std::future::pending;
use std::mem::take;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lithos_llm::types::{ContentPart, ToolCall, ToolResult};
use pebble_coding_agent::environment::{Environment, ExecRequest, LocalEnvironment};
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, scripted_client, text_response,
};
use pebble_coding_agent::tools::{
    CodingToolSet, OutputPage, OutputStream, RegisteredTool, ToolArtifact, ToolError,
    ToolOutputStore, ToolOutputWriter, ToolRunner,
};
use pebble_coding_agent::{
    CodingAgent, CodingAgentOptions, SessionId, SessionScope, ShutdownReason,
};
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

type Objects = HashMap<String, (String, Vec<u8>)>;

// This test application stores objects outside the Environment. No artifact
// reference is interpreted as a workspace path.
#[derive(Default)]
struct MemoryStore {
    next:        AtomicUsize,
    fail_finish: bool,
    objects:     Arc<Mutex<Objects>>,
}

struct Writer {
    id:          usize,
    fail_finish: bool,
    root:        String,
    chunks:      Mutex<Vec<(OutputStream, Vec<u8>)>>,
    objects:     Arc<Mutex<Objects>>,
}

#[async_trait]
impl ToolOutputWriter for Writer {
    async fn append(&self, stream: OutputStream, bytes: &[u8]) -> Result<(), ToolError> {
        self.chunks
            .lock()
            .expect("chunks")
            .push((stream, bytes.to_vec()));
        Ok(())
    }

    async fn finish(&self) -> Result<Vec<ToolArtifact>, ToolError> {
        if self.fail_finish {
            return Err(ToolError::execution("Commit failed"));
        }
        let chunks = take(&mut *self.chunks.lock().expect("chunks"));
        let mut artifacts = Vec::new();
        for stream in [
            OutputStream::Stdout,
            OutputStream::Stderr,
            OutputStream::Result,
        ] {
            let bytes: Vec<u8> = chunks
                .iter()
                .filter(|(s, _)| *s == stream)
                .flat_map(|(_, bytes)| bytes.iter().copied())
                .collect();
            if bytes.is_empty() {
                continue;
            }
            let reference = format!("opaque:{}:{stream:?}", self.id);
            let artifact = ToolArtifact {
                reference:   reference.clone(),
                label:       format!("{stream:?}"),
                media_type:  "text/plain".into(),
                byte_length: bytes.len() as u64,
            };
            self.objects
                .lock()
                .expect("objects")
                .insert(reference, (self.root.clone(), bytes));
            artifacts.push(artifact);
        }
        Ok(artifacts)
    }
}

#[async_trait]
impl ToolOutputStore for MemoryStore {
    async fn start(
        &self,
        session: &SessionScope,
        _: &str,
    ) -> Result<Arc<dyn ToolOutputWriter>, ToolError> {
        Ok(Arc::new(Writer {
            id:          self.next.fetch_add(1, Ordering::SeqCst),
            fail_finish: self.fail_finish,
            root:        session.root_session_id().as_str().into(),
            chunks:      Mutex::default(),
            objects:     Arc::clone(&self.objects),
        }))
    }

    async fn read(
        &self,
        session: &SessionScope,
        reference: &str,
        offset: u64,
        limit: usize,
    ) -> Result<OutputPage, ToolError> {
        let objects = self.objects.lock().expect("objects");
        let (owner, bytes) = objects
            .get(reference)
            .ok_or_else(|| ToolError::unavailable("Unknown output"))?;
        if owner != session.root_session_id().as_str() {
            return Err(ToolError::denied("Output belongs to another session"));
        }
        let offset = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        Ok(OutputPage {
            bytes:       bytes[offset..offset.saturating_add(limit).min(bytes.len())].to_vec(),
            total_bytes: bytes.len() as u64,
        })
    }
}

fn text(result: &ToolResult) -> &str {
    let [ContentPart::Text { text }] = result.content.as_slice() else {
        panic!("expected text");
    };
    text
}

fn completion(events: &Mutex<Vec<CodingAgentEvent>>) -> CodingEvent {
    events
        .lock()
        .expect("events")
        .iter()
        .find_map(|event| {
            matches!(event.event, CodingEvent::ToolCallCompleted { .. })
                .then(|| event.event.clone())
        })
        .expect("completion")
}

fn runner(
    env: Arc<dyn Environment>,
    store: Arc<MemoryStore>,
) -> (ToolRunner, Arc<Mutex<Vec<CodingAgentEvent>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = Arc::clone(&events);
    let runner = ToolRunner::new(CodingToolSet::core(), env)
        .session(SessionScope::root(SessionId::new("root")))
        .output_store(store)
        .expect("store installed")
        .options(CodingAgentOptions::default().with_tool_output_limit("shell", 4096))
        .on_event(move |event| capture.lock().expect("events").push(event));
    (runner, events)
}

#[tokio::test]
async fn an_opaque_reference_recovers_the_middle_of_truncated_remote_output() {
    let output = format!("{}MIDDLE{}", "a".repeat(1_100_000), "z".repeat(1_100_000));
    let mut environment = MockEnvironment::linux();
    environment.exec_result.stdout = output.clone();
    environment.exec_result.stderr = "diagnostic".into();
    environment.exec_result.exit_code = Some(9);
    let store = Arc::new(MemoryStore::default());
    let (runner, events) = runner(Arc::new(environment), Arc::clone(&store));
    let result = runner
        .run(
            &ToolCall::function("shell-1", "shell", json!({"command":"remote-command"})),
            CancellationToken::new(),
        )
        .await
        .expect("tool answers");
    assert!(result.is_error, "failed commands retain artifacts too");
    assert!(!text(&result).contains("MIDDLE"));
    let CodingEvent::ToolCallCompleted { metadata, .. } = completion(&events) else {
        unreachable!()
    };
    let reference = &metadata
        .artifacts
        .iter()
        .find(|a| a.label == "Stdout")
        .expect("stdout")
        .reference;
    assert!(text(&result).contains(reference));
    assert_eq!(
        store.objects.lock().expect("objects")[reference].1,
        output.as_bytes()
    );
    // A fresh runner represents resume: only the same store and session ID are
    // needed, with a new environment that knows nothing about the old command.
    let resumed = ToolRunner::new(CodingToolSet::empty(), Arc::new(MockEnvironment::linux()))
        .session(SessionScope::root(SessionId::new("root")))
        .output_store(store.clone())
        .expect("installed");
    let page = resumed
        .run(
            &ToolCall::function(
                "read-1",
                "read_tool_output",
                json!({"reference":reference,"offset":1_100_000,"limit":6}),
            ),
            CancellationToken::new(),
        )
        .await
        .expect("read");
    let page: Value = serde_json::from_str(text(&page)).expect("page");
    assert_eq!(page["text"], "MIDDLE");
    assert_eq!(page["next_offset"], 1_100_006);
    assert_eq!(page["eof"], false);
    let denied = resumed
        .session(SessionScope::root(SessionId::new("unrelated")))
        .run(
            &ToolCall::function("read-2", "read_tool_output", json!({"reference":reference})),
            CancellationToken::new(),
        )
        .await
        .expect("denial");
    assert!(denied.is_error);
    assert!(text(&denied).contains("another session"));
}

#[tokio::test]
async fn local_capture_saves_bytes_before_the_pipe_retention_cap() {
    let store = Arc::new(MemoryStore::default());
    let (runner, events) = runner(
        Arc::new(LocalEnvironment::new(temp_dir())),
        Arc::clone(&store),
    );
    let result = runner.run(&ToolCall::function("local-1", "shell", json!({"command":"head -c 1100000 /dev/zero | tr '\\0' a; printf MIDDLE; head -c 1100000 /dev/zero | tr '\\0' z"})), CancellationToken::new()).await.expect("runs");
    assert!(!result.is_error, "{}", text(&result));
    let CodingEvent::ToolCallCompleted { metadata, .. } = completion(&events) else {
        unreachable!()
    };
    let reference = &metadata.artifacts[0].reference;
    let objects = store.objects.lock().expect("objects");
    assert_eq!(&objects[reference].1[1_100_000..1_100_006], b"MIDDLE");
    assert_eq!(objects[reference].1.len(), 2_200_006);
    assert!(!text(&result).contains("MIDDLE"));
}

#[tokio::test]
async fn oversized_application_tool_output_is_saved_before_truncation() {
    let store = Arc::new(MemoryStore::default());
    let tool =
        RegisteredTool::function("report", "Report", json!({"type":"object"}), |_, _| async {
            Ok(format!("{}MIDDLE{}", "a".repeat(3000), "z".repeat(3000)))
        });
    let tools = CodingToolSet::empty().with_tool(tool).expect("tool");
    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = Arc::clone(&events);
    let runner = ToolRunner::new(tools, Arc::new(MockEnvironment::linux()))
        .output_store(store.clone())
        .expect("store")
        .options(CodingAgentOptions::default().with_tool_output_limit("report", 1024))
        .on_event(move |event| capture.lock().expect("events").push(event));
    let result = runner
        .run(
            &ToolCall::function("report-1", "report", json!({})),
            CancellationToken::new(),
        )
        .await
        .expect("runs");
    assert!(!text(&result).contains("MIDDLE"));
    let CodingEvent::ToolCallCompleted { metadata, .. } = completion(&events) else {
        unreachable!()
    };
    assert!(text(&result).contains(&metadata.artifacts[0].reference));
    assert_eq!(
        &store.objects.lock().expect("objects")[&metadata.artifacts[0].reference].1[3000..3006],
        b"MIDDLE"
    );
}

#[tokio::test]
async fn retrieval_is_only_advertised_with_a_store() {
    for installed in [false, true] {
        let (client, provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut builder =
            CodingAgent::builder(client, Arc::new(MockEnvironment::linux())).model("test/model");
        if installed {
            builder = builder.output_store(Arc::new(MemoryStore::default()));
        }
        let mut agent = builder.build().await.expect("builds");
        agent.prompt("hello").await.result.expect("answers");
        assert_eq!(
            provider.requests()[0]
                .tools()
                .iter()
                .any(|tool| tool.name == "read_tool_output"),
            installed
        );
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shutdown");
    }
}

struct FailingWriter;

#[async_trait]
impl ToolOutputWriter for FailingWriter {
    async fn append(&self, _: OutputStream, _: &[u8]) -> Result<(), ToolError> {
        Err(ToolError::execution("Store unavailable"))
    }
    async fn finish(&self) -> Result<Vec<ToolArtifact>, ToolError> {
        panic!("an incomplete capture must not be finalized")
    }
}

#[tokio::test]
async fn storage_failure_stops_a_local_command_without_waiting_for_its_timeout() {
    let env = LocalEnvironment::new(temp_dir());
    let result = timeout(
        Duration::from_secs(4),
        env.exec(ExecRequest {
            output_writer: Some(Arc::new(FailingWriter)),
            ..ExecRequest::new("printf ready; exec sleep 30")
        }),
    )
    .await
    .expect("storage failure stops the process promptly");
    assert!(result.is_err());
}

struct PendingWriter {
    entered: Notify,
    bytes:   Mutex<Vec<u8>>,
}

#[async_trait]
impl ToolOutputWriter for PendingWriter {
    async fn append(&self, _: OutputStream, bytes: &[u8]) -> Result<(), ToolError> {
        self.bytes.lock().expect("bytes").extend_from_slice(bytes);
        self.entered.notify_one();
        pending().await
    }
    async fn finish(&self) -> Result<Vec<ToolArtifact>, ToolError> {
        panic!("incomplete capture")
    }
}

#[tokio::test]
async fn cancellation_reaps_the_command_even_when_storage_is_stuck() {
    let writer = Arc::new(PendingWriter {
        entered: Notify::new(),
        bytes:   Mutex::default(),
    });
    let environment = LocalEnvironment::new(temp_dir());
    let cancel = CancellationToken::new();
    let execute = environment.exec(ExecRequest {
        cancel_token: Some(cancel.clone()),
        output_writer: Some(writer.clone()),
        ..ExecRequest::new("printf '%s\n' \"$$\"; exec sleep 30")
    });
    let (result, ()) = tokio::join!(timeout(Duration::from_secs(4), execute), async {
        writer.entered.notified().await;
        cancel.cancel();
    });
    assert!(result.expect("cancellation stops the call").is_err());
    let pid = String::from_utf8(writer.bytes.lock().expect("bytes").clone()).expect("PID text");
    assert!(pid.trim().parse::<u32>().is_ok());
    let status = Command::new("kill")
        .args(["-0", pid.trim()])
        .output()
        .expect("check process")
        .status;
    assert!(!status.success(), "the process was reaped");
}

#[tokio::test]
async fn failed_finalization_still_reports_that_the_process_ran() {
    let store = Arc::new(MemoryStore {
        fail_finish: true,
        ..MemoryStore::default()
    });
    let (runner, events) = runner(Arc::new(MockEnvironment::linux()), store);
    let result = runner
        .run(
            &ToolCall::function("finished-1", "shell", json!({"command":"echo done"})),
            CancellationToken::new(),
        )
        .await
        .expect("answers");
    assert!(result.is_error);
    assert!(text(&result).contains("Command finished"));
    assert!(
        events
            .lock()
            .expect("events")
            .iter()
            .any(|event| matches!(event.event, CodingEvent::ToolProcessCompleted { .. }))
    );
    let CodingEvent::ToolCallCompleted { metadata, .. } = completion(&events) else {
        unreachable!()
    };
    assert!(metadata.artifacts.is_empty());
}
