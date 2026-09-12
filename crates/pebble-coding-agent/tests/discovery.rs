//! Memory and skill discovery, through the public API.
//!
//! An application names a convention (the profile's own files from the git
//! root down, a few skill directories, one of them required) and pebble does
//! the walking and the checking. These tests see the results where an
//! application does: the `MemoryLoaded` and `SkillsDiscovered` events and the
//! system prompt the model is given.

use std::collections::HashMap;
use std::sync::Arc;

use lithos_llm::types::Role;
use pebble_coding_agent::events::{CodingEvent, SkippedSkillReason};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedProvider, client_from, message_text, text_response,
};
use pebble_coding_agent::{
    CodingAgent, CodingAgentOptions, MemoryDiscovery, ShutdownReason, SkillDiscovery,
};

/// A repository at `/home` whose working directory is `/home/test`, with the
/// repository's instructions at the root and the project's beside the work.
fn repository() -> MockEnvironment {
    MockEnvironment {
        files: HashMap::from([
            ("/home/AGENTS.md".to_owned(), "Repository rules.".to_owned()),
            (
                "/home/test/CLAUDE.md".to_owned(),
                "Project rules.".to_owned(),
            ),
            ("/home/test/skills/.keep".to_owned(), String::new()),
        ]),
        ..MockEnvironment::linux()
    }
}

async fn agent_in(
    env: MockEnvironment,
    options: CodingAgentOptions,
) -> (CodingAgent, Vec<CodingEvent>) {
    let (client, _provider) = client_from(ScriptedProvider::new(vec![ScriptedCall::response(
        text_response("ok"),
    )]));
    let mut agent = CodingAgent::builder(client, Arc::new(env))
        .model("test/model")
        .options(options)
        .build()
        .await
        .expect("the agent builds");
    // The build's events may have passed before a subscriber exists; what
    // the build found is read from the snapshot, and the receiver catches
    // whatever is still in flight.
    let mut events = agent.subscribe();
    agent.flush_events().await.expect("events flush");
    let mut published = Vec::new();
    while let Ok(event) = events.try_recv() {
        published.push(event.event);
    }
    (agent, published)
}

#[tokio::test]
async fn memory_discovery_walks_from_the_root_down_to_the_working_directory() {
    let (mut agent, _) = agent_in(
        repository(),
        CodingAgentOptions::default().with_memory_discovery(MemoryDiscovery::from_root("/home")),
    )
    .await;

    let loaded: Vec<String> = agent
        .snapshot()
        .memory()
        .iter()
        .map(|file| file.path.clone())
        .collect();
    assert_eq!(
        loaded,
        ["/home/AGENTS.md", "/home/test/CLAUDE.md"],
        "the repository's file first, the project's after; missing candidates skipped"
    );

    let report = agent.prompt("go").await;
    assert!(report.result.is_ok(), "{report:?}");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn the_system_prompt_carries_discovered_memory_root_first() {
    let (client, provider) = client_from(ScriptedProvider::new(vec![ScriptedCall::response(
        text_response("ok"),
    )]));
    let mut agent = CodingAgent::builder(client, Arc::new(repository()))
        .model("test/model")
        .options(
            CodingAgentOptions::default()
                .with_memory_discovery(MemoryDiscovery::from_root("/home")),
        )
        .build()
        .await
        .expect("the agent builds");
    agent
        .prompt("go")
        .await
        .result
        .expect("the prompt succeeds");
    let request = provider.requests().remove(0);
    let system = request
        .messages()
        .iter()
        .find(|message| message.role() == Role::System)
        .map(message_text)
        .expect("a system prompt");
    let repository_at = system.find("Repository rules.").expect("the root's rules");
    let project_at = system.find("Project rules.").expect("the project's rules");
    assert!(repository_at < project_at, "root first: {system}");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn working_directory_discovery_reads_only_the_working_directory() {
    let (mut agent, _) = agent_in(
        repository(),
        CodingAgentOptions::default().with_memory_discovery(MemoryDiscovery::working_directory()),
    )
    .await;
    let loaded: Vec<String> = agent
        .snapshot()
        .memory()
        .iter()
        .map(|file| file.path.clone())
        .collect();
    assert_eq!(loaded, ["/home/test/CLAUDE.md"]);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn skill_discovery_resolves_its_directories_and_reports_a_missing_required_one() {
    let (client, _provider) = client_from(ScriptedProvider::new(vec![ScriptedCall::response(
        text_response("ok"),
    )]));
    let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
    struct Sink(Arc<std::sync::Mutex<Vec<CodingEvent>>>);
    #[async_trait::async_trait]
    impl pebble_coding_agent::events::EventSink for Sink {
        async fn record(
            &self,
            event: &pebble_coding_agent::events::CodingAgentEvent,
        ) -> Result<(), pebble_coding_agent::events::EventSinkError> {
            self.0.lock().expect("sink lock").push(event.event.clone());
            Ok(())
        }
    }
    let mut agent = CodingAgent::builder(client, Arc::new(repository()))
        .model("test/model")
        .options(
            CodingAgentOptions::default()
                .with_skill_dirs(["/opt/skills".to_owned()])
                .with_skill_discovery(
                    SkillDiscovery::new()
                        .search_under_git_root(".fabro/skills")
                        .search("skills")
                        .require("named/skills"),
                ),
        )
        .event_sink(Arc::new(Sink(Arc::clone(&recorded))))
        .build()
        .await
        .expect("the agent builds");
    agent.flush_events().await.expect("events flush");

    let events = recorded.lock().expect("sink lock").clone();
    let discovered = events
        .iter()
        .find_map(|event| match event {
            CodingEvent::SkillsDiscovered {
                source_dirs,
                skipped,
                ..
            } => Some((source_dirs.clone(), skipped.clone())),
            _ => None,
        })
        .expect("skills were discovered");
    // The mock has no git, so the working directory stands in for the root.
    assert_eq!(discovered.0, [
        "/opt/skills",
        "/home/test/.fabro/skills",
        "/home/test/skills",
    ]);
    assert_eq!(discovered.1.len(), 1, "{:?}", discovered.1);
    assert_eq!(discovered.1[0].path, "/home/test/named/skills");
    assert_eq!(discovered.1[0].reason, SkippedSkillReason::MissingDirectory);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}
