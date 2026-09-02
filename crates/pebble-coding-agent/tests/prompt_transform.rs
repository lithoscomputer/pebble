//! Adjusting the system prompt from outside the crate.
//!
//! An application cannot install a profile, but it can see the prompt the
//! selected profile wrote and use it, add to it, or replace it. The replacement
//! case is what an "Ask Fabro" persona needs: its own words first, and the
//! built-in tool vocabulary and runtime behavior untouched underneath.

use std::sync::{Arc, Mutex};

use lithos_llm::types::{Request, Role};
use pebble_coding_agent::environment::Environment;
use pebble_coding_agent::extensions::{
    SystemPromptContext, SystemPromptDecision, SystemPromptTransform,
};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, message_text, scripted_client, text_response,
};
use pebble_coding_agent::{CodingAgent, CodingAgentOptions, ShutdownReason};

/// What an Ask Fabro persona says first.
const ASK_FABRO: &str = "You are Ask Fabro. Answer questions about this repository; do not change \
                         it. Use the tools you are given to read what you need.";

/// What the transform was shown, kept for the assertions.
#[derive(Clone, Debug, Default)]
struct Seen {
    default_prompt:    String,
    working_directory: String,
    tools:             Vec<String>,
    memory:            Vec<String>,
    skills:            Vec<String>,
}

/// A transform that answers with a fixed decision and remembers its context.
struct Recording {
    decision: SystemPromptDecision,
    seen:     Mutex<Option<Seen>>,
}

impl Recording {
    fn new(decision: SystemPromptDecision) -> Arc<Self> {
        Arc::new(Self {
            decision,
            seen: Mutex::new(None),
        })
    }

    fn seen(&self) -> Seen {
        self.seen
            .lock()
            .expect("the seen lock is held")
            .clone()
            .expect("the transform was called")
    }
}

impl SystemPromptTransform for Recording {
    fn transform(&self, context: SystemPromptContext<'_>) -> SystemPromptDecision {
        let mut tools: Vec<String> = context
            .tools()
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        tools.sort();
        *self.seen.lock().expect("the seen lock is held") = Some(Seen {
            default_prompt: context.default_prompt().to_owned(),
            working_directory: context.environment().working_directory.clone(),
            tools,
            memory: context
                .memory()
                .iter()
                .map(|file| file.path.clone())
                .collect(),
            skills: context
                .skills()
                .iter()
                .map(|skill| skill.name.clone())
                .collect(),
        });
        self.decision.clone()
    }
}

fn environment() -> Arc<dyn Environment> {
    Arc::new(MockEnvironment::linux())
}

fn system_prompt(request: &Request) -> String {
    let first = request
        .messages()
        .first()
        .expect("the request is not empty");
    assert_eq!(first.role(), Role::System);
    message_text(first)
}

fn tool_names(request: &Request) -> Vec<String> {
    let mut names: Vec<String> = request
        .tools()
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    names.sort();
    names
}

/// Runs one prompt through an agent built with `transform`, answering with the
/// request the model was sent.
async fn round_with(
    transform: Option<Arc<dyn SystemPromptTransform>>,
    environment: Arc<dyn Environment>,
    options: CodingAgentOptions,
) -> Request {
    let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("ok"))]);
    let mut builder = CodingAgent::builder(client, environment)
        .model("test/model")
        .options(options);
    if let Some(transform) = transform {
        builder = builder.system_prompt_transform(transform);
    }
    let mut agent = builder.build().await.expect("the agent builds");
    agent.prompt("hello").await.expect("the prompt succeeds");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
    let requests = provider.requests();
    assert_eq!(requests.len(), 1, "exactly one round ran");
    requests[0].clone()
}

#[tokio::test]
async fn ask_fabro_replaces_the_prompt_and_keeps_the_tool_vocabulary() {
    let transform = Recording::new(SystemPromptDecision::Replace(ASK_FABRO.to_owned()));
    let untouched = round_with(None, environment(), CodingAgentOptions::default()).await;

    let request = round_with(
        Some(Arc::clone(&transform) as Arc<dyn SystemPromptTransform>),
        environment(),
        CodingAgentOptions::default(),
    )
    .await;

    assert_eq!(system_prompt(&request), ASK_FABRO);
    assert_eq!(
        tool_names(&request),
        tool_names(&untouched),
        "the persona changes the words, not the tools or their names"
    );
    let tools = tool_names(&request);
    assert!(tools.contains(&"read_file".to_owned()), "{tools:?}");
    assert!(tools.contains(&"shell".to_owned()), "{tools:?}");

    let seen = transform.seen();
    assert_eq!(seen.default_prompt, system_prompt(&untouched));
    assert_eq!(seen.working_directory, "/home/test");
    assert_eq!(
        seen.tools, tools,
        "the transform is shown what the model is shown"
    );
}

#[tokio::test]
async fn an_appended_prompt_follows_the_default() {
    let transform = Recording::new(SystemPromptDecision::Append(
        "Always answer in French.".to_owned(),
    ));
    let untouched = round_with(None, environment(), CodingAgentOptions::default()).await;

    let request = round_with(
        Some(transform as Arc<dyn SystemPromptTransform>),
        environment(),
        CodingAgentOptions::default(),
    )
    .await;

    assert_eq!(
        system_prompt(&request),
        format!("{}\n\nAlways answer in French.", system_prompt(&untouched))
    );
}

#[tokio::test]
async fn using_the_default_leaves_the_prompt_as_written() {
    let transform = Recording::new(SystemPromptDecision::UseDefault);
    let untouched = round_with(None, environment(), CodingAgentOptions::default()).await;

    let request = round_with(
        Some(transform as Arc<dyn SystemPromptTransform>),
        environment(),
        CodingAgentOptions::default(),
    )
    .await;

    assert_eq!(system_prompt(&request), system_prompt(&untouched));
}

#[tokio::test]
async fn the_transform_is_shown_what_was_loaded_as_memory_and_skills() {
    let environment: Arc<dyn Environment> = Arc::new(MockEnvironment {
        files: [
            (
                "/work/AGENTS.md".to_owned(),
                "Always run the tests.".to_owned(),
            ),
            (
                "/skills/commit/SKILL.md".to_owned(),
                "---\nname: commit\ndescription: Make a commit\n---\nRun commit.".to_owned(),
            ),
        ]
        .into_iter()
        .collect(),
        glob_results: vec!["/skills/commit/SKILL.md".to_owned()],
        ..MockEnvironment::linux()
    });
    let transform = Recording::new(SystemPromptDecision::UseDefault);

    round_with(
        Some(Arc::clone(&transform) as Arc<dyn SystemPromptTransform>),
        environment,
        CodingAgentOptions::default()
            .with_memory_files(["/work/AGENTS.md".to_owned()])
            .with_skill_dirs(["/skills".to_owned()]),
    )
    .await;

    let seen = transform.seen();
    assert_eq!(seen.memory, ["/work/AGENTS.md"]);
    assert_eq!(seen.skills, ["commit"]);
    assert!(
        seen.default_prompt.contains("Always run the tests."),
        "the default prompt carries the loaded memory"
    );
    assert!(
        seen.tools.contains(&"use_skill".to_owned()),
        "a discovered skill registers the tool that loads one: {:?}",
        seen.tools
    );
}
