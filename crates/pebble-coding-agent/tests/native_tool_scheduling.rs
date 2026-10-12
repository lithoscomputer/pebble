//! Native mutations preserve updates through profile registration and
//! middleware.
//!
//! Inspired by @denkhaus's lost-edit report and delayed-write reproduction:
//! https://github.com/fabro-sh/fabro/pull/787

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lithos_llm::catalog::Catalog;
use lithos_llm::types::{ContentPart, ToolCall};
use pebble_agent::{
    ToolCallNext, ToolCallRequest, ToolCatalog, ToolDiscoveryNext, ToolMiddleware, ToolOutcome,
    ToolScheduling, ToolSystemError, TurnContext,
};
use pebble_coding_agent::environment::{
    DirEntry, EnvResult, Environment, ExecOutcome, ExecRequest, GrepOptions,
};
use pebble_coding_agent::state::Message;
use pebble_coding_agent::test_support::{
    MutableMockEnvironment, ScriptedCall, ScriptedProvider, TEST_CATALOG, multi_tool_call_response,
    scripted_client_builder, text_response,
};
use pebble_coding_agent::tools::{PermissionLevel, PermissionLevelPolicy, PermissionMiddleware};
use pebble_coding_agent::{CodingAgent, ShutdownReason};
use serde_json::json;
use tokio::time::{sleep, timeout};

/// The delay occurs after the native editor has read and computed its update.
/// Paused time lets concurrent dispatch read the same base deterministically,
/// without a two-reader barrier that would deadlock sequential dispatch.
struct DelayedWrites {
    inner:  MutableMockEnvironment,
    reads:  Mutex<Vec<(String, String)>>,
    active: AtomicUsize,
    peak:   AtomicUsize,
}

impl DelayedWrites {
    fn new() -> Self {
        Self {
            inner:  MutableMockEnvironment::new(
                [
                    ("note.txt".to_owned(), "alpha\nbeta\n".to_owned()),
                    ("other.txt".to_owned(), "alpha\nbeta\n".to_owned()),
                ]
                .into(),
            ),
            reads:  Mutex::new(Vec::new()),
            active: AtomicUsize::new(0),
            peak:   AtomicUsize::new(0),
        }
    }

    fn contents(&self, path: &str) -> String {
        self.inner.files.lock().expect("files lock")[path].clone()
    }

    fn observed_reads(&self) -> Vec<(String, String)> {
        self.reads.lock().expect("reads lock").clone()
    }
}

#[async_trait]
impl Environment for DelayedWrites {
    fn working_directory(&self) -> &str {
        self.inner.working_directory()
    }
    fn platform(&self) -> &str {
        self.inner.platform()
    }
    fn os_version(&self) -> String {
        self.inner.os_version()
    }

    async fn read_file_bytes(&self, path: &str) -> EnvResult<Vec<u8>> {
        let bytes = self.inner.read_file_bytes(path).await?;
        self.reads.lock().expect("reads lock").push((
            path.to_owned(),
            String::from_utf8(bytes.clone()).expect("UTF-8 fixture"),
        ));
        Ok(bytes)
    }

    async fn write_file(&self, path: &str, content: &str) -> EnvResult<()> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        sleep(Duration::from_millis(10)).await;
        let result = self.inner.write_file(path, content).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        result
    }

    async fn rename_file(&self, source: &str, destination: &str) -> EnvResult<()> {
        self.inner.rename_file(source, destination).await
    }
    async fn delete_file(&self, path: &str) -> EnvResult<()> {
        self.inner.delete_file(path).await
    }
    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        self.inner.file_exists(path).await
    }
    async fn list_directory(&self, path: &str, depth: Option<usize>) -> EnvResult<Vec<DirEntry>> {
        self.inner.list_directory(path, depth).await
    }
    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        options: &GrepOptions,
    ) -> EnvResult<Vec<String>> {
        self.inner.grep(pattern, path, options).await
    }
    async fn glob(&self, pattern: &str, path: Option<&str>) -> EnvResult<Vec<String>> {
        self.inner.glob(pattern, path).await
    }
    async fn exec(&self, request: ExecRequest<'_>) -> EnvResult<ExecOutcome> {
        self.inner.exec(request).await
    }
}

/// The negative control changes only discovered scheduling, keeping the real
/// native executors, schemas, aliases, permissions, and dispatch path intact.
#[derive(Default)]
struct ObserveScheduling {
    disable_mutation_scheduling: bool,
    calls: Mutex<Vec<(String, ToolScheduling)>>,
}

#[async_trait]
impl ToolMiddleware for ObserveScheduling {
    async fn discover(
        &self,
        context: TurnContext<'_>,
        next: ToolDiscoveryNext<'_>,
    ) -> Result<ToolCatalog, ToolSystemError> {
        let catalog = next.run(context).await?;
        if self.disable_mutation_scheduling {
            Ok(ToolCatalog::new(catalog.visible_tools().cloned().map(
                |tool| {
                    if matches!(
                        tool.id().as_str(),
                        "edit_file" | "write_file" | "apply_patch"
                    ) {
                        tool.with_scheduling(ToolScheduling::Concurrent)
                    } else {
                        tool
                    }
                },
            )))
        } else {
            Ok(catalog)
        }
    }

    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> Result<ToolOutcome, ToolSystemError> {
        self.calls.lock().expect("calls lock").push((
            request.call().name.clone(),
            request.descriptor().scheduling(),
        ));
        next.run(request).await
    }
}

fn edit(name: &str, path_key: &str, path: &str, id: &str, old: &str, new: &str) -> ToolCall {
    ToolCall::function(
        id,
        name,
        json!({path_key: path, "old_string": old, "new_string": new}),
    )
}

fn patch(id: &str, old: &str, new: &str) -> ToolCall {
    ToolCall::custom(
        id,
        "apply_patch",
        format!("*** Begin Patch\n*** Update File: note.txt\n@@\n-{old}\n+{new}\n*** End Patch"),
    )
}

async fn run_batch(profile: &str, calls: Vec<ToolCall>, concurrent: bool) -> Arc<DelayedWrites> {
    let environment = Arc::new(DelayedWrites::new());
    let mut response = multi_tool_call_response(vec![]);
    response
        .content
        .extend(calls.iter().cloned().map(ContentPart::ToolCall));
    let (builder, provider) = scripted_client_builder(ScriptedProvider::new(vec![
        ScriptedCall::response(response),
        ScriptedCall::response(text_response("done")),
    ]));
    let mut catalog_text = TEST_CATALOG.replace(
        "profile = \"anthropic\"",
        &format!("profile = \"{profile}\""),
    );
    if profile == "openai" {
        catalog_text = catalog_text.replace("test-codec", "openai-responses");
    }
    let catalog = Catalog::builder()
        .overlay_toml(&catalog_text)
        .expect("valid catalog")
        .build()
        .expect("catalog builds");
    let client = builder
        .catalog(catalog)
        .build()
        .expect("client builds")
        .client;
    let audit = Arc::new(ObserveScheduling {
        disable_mutation_scheduling: concurrent,
        ..ObserveScheduling::default()
    });
    let mut agent = CodingAgent::builder(client, environment.clone())
        .model("test/model")
        .tool_middleware(Arc::new(PermissionMiddleware::new(Arc::new(
            PermissionLevelPolicy::new(PermissionLevel::Full),
        ))))
        .tool_middleware(audit.clone())
        .build()
        .await
        .expect("agent builds");
    timeout(Duration::from_secs(5), agent.prompt("perform the batch"))
        .await
        .expect("prompt completes")
        .result
        .expect("prompt succeeds");
    assert_eq!(provider.call_count(), 2);
    let history = agent.history();
    let results = history
        .turns()
        .iter()
        .find_map(|message| match message {
            Message::ToolResults { results, .. } => Some(results),
            _ => None,
        })
        .expect("results committed");
    assert_eq!(results.len(), calls.len());
    for (result, call) in results.iter().zip(&calls) {
        assert_eq!(result.tool_call_id, call.id);
        assert!(!result.is_error, "{result:?}");
        assert!(
            matches!(result.content.as_slice(), [ContentPart::Text { text }] if !text.is_empty())
        );
    }
    let observed = audit.calls.lock().expect("calls lock").clone();
    assert_eq!(observed.len(), calls.len());
    for (name, scheduling) in observed {
        let mutation = matches!(
            name.as_str(),
            "edit_file" | "Edit" | "write_file" | "Write" | "apply_patch"
        );
        assert_eq!(
            scheduling,
            if mutation && !concurrent {
                ToolScheduling::Sequential
            } else {
                ToolScheduling::Concurrent
            },
            "{profile}: {name}"
        );
    }
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
    environment
}

#[tokio::test(start_paused = true)]
async fn same_file_native_edits_survive_profile_aliases_and_permissions() {
    for (profile, name, key) in [
        ("anthropic", "edit_file", "file_path"),
        ("claude-5", "Edit", "file_path"),
        ("kimi", "Edit", "path"),
    ] {
        let environment = run_batch(
            profile,
            vec![
                edit(name, key, "note.txt", "alpha", "alpha", "ALPHA"),
                edit(name, key, "note.txt", "beta", "beta", "BETA"),
            ],
            false,
        )
        .await;
        assert_eq!(
            environment.contents("note.txt"),
            "ALPHA\nBETA\n",
            "{profile}"
        );
        assert_eq!(environment.peak.load(Ordering::SeqCst), 1);
        assert_eq!(environment.observed_reads(), [
            ("note.txt".to_owned(), "alpha\nbeta\n".to_owned()),
            ("note.txt".to_owned(), "ALPHA\nbeta\n".to_owned()),
        ]);
    }
}

#[tokio::test(start_paused = true)]
async fn concurrent_native_edit_negative_control_loses_the_first_successful_edit() {
    let environment = run_batch(
        "anthropic",
        vec![
            edit(
                "edit_file",
                "file_path",
                "note.txt",
                "alpha",
                "alpha",
                "ALPHA",
            ),
            edit("edit_file", "file_path", "note.txt", "beta", "beta", "BETA"),
        ],
        true,
    )
    .await;
    assert_eq!(environment.contents("note.txt"), "alpha\nBETA\n");
    assert_eq!(environment.peak.load(Ordering::SeqCst), 2);
    assert_eq!(environment.observed_reads(), [
        ("note.txt".to_owned(), "alpha\nbeta\n".to_owned()),
        ("note.txt".to_owned(), "alpha\nbeta\n".to_owned()),
    ]);
}

#[tokio::test(start_paused = true)]
async fn different_file_native_edits_are_also_serialized() {
    for concurrent in [false, true] {
        let environment = run_batch(
            "anthropic",
            vec![
                edit(
                    "edit_file",
                    "file_path",
                    "note.txt",
                    "alpha",
                    "alpha",
                    "ALPHA",
                ),
                edit(
                    "edit_file",
                    "file_path",
                    "other.txt",
                    "beta",
                    "beta",
                    "BETA",
                ),
            ],
            concurrent,
        )
        .await;
        assert_eq!(environment.contents("note.txt"), "ALPHA\nbeta\n");
        assert_eq!(environment.contents("other.txt"), "alpha\nBETA\n");
        assert_eq!(
            environment.peak.load(Ordering::SeqCst),
            if concurrent { 2 } else { 1 }
        );
    }
}

#[tokio::test(start_paused = true)]
async fn overlapping_patches_preserve_changes_in_one_response() {
    let environment = run_batch(
        "openai",
        vec![
            patch("alpha", "alpha", "ALPHA"),
            patch("beta", "beta", "BETA"),
        ],
        false,
    )
    .await;
    assert_eq!(environment.contents("note.txt"), "ALPHA\nBETA\n");
    assert_eq!(environment.peak.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn sequential_replacements_keep_model_order_without_merging() {
    let environment = run_batch(
        "anthropic",
        vec![
            ToolCall::function(
                "first",
                "write_file",
                json!({"file_path":"note.txt", "content":"ALPHA beta"}),
            ),
            ToolCall::function(
                "second",
                "write_file",
                json!({"file_path":"note.txt", "content":"alpha BETA"}),
            ),
        ],
        false,
    )
    .await;
    assert_eq!(environment.contents("note.txt"), "alpha BETA");
    assert_eq!(environment.peak.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn kimi_append_mutations_preserve_both_updates() {
    let environment = run_batch(
        "kimi",
        vec![
            ToolCall::function(
                "first",
                "Write",
                json!({"path":"note.txt", "content":"one\n", "mode":"append"}),
            ),
            ToolCall::function(
                "second",
                "Write",
                json!({"path":"note.txt", "content":"two\n", "mode":"append"}),
            ),
        ],
        false,
    )
    .await;
    assert_eq!(environment.contents("note.txt"), "alpha\nbeta\none\ntwo\n");
    assert_eq!(environment.peak.load(Ordering::SeqCst), 1);
}
