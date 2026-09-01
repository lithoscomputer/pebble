//! Filling the window, and what the session loads before it starts.
//!
//! A long prompt outgrows the model's context window: the session warns about
//! it whatever it is configured to do, and summarizes the older turns where it
//! is allowed to. What it summarizes with — the prompt, the file list — is part
//! of the contract, because a summary that loses the files loses the prompt's
//! progress.
//!
//! Initialization is here too, since it is what fills the prompt the window is
//! measured against.

use std::collections::HashMap;
use std::time::Duration;

use lithos_llm::types::{ErrorKind as LlmErrorKind, ToolDefinition};
use serde_json::json;
use tokio::time::timeout;

use super::*;
use crate::error::ErrorKind;
use crate::test_support::{
    MockEnvironment, ScriptedCompletion, ScriptedFailure, message_text, with_input_tokens,
};
use crate::types::SkillActivationSource;

/// An input long enough to fill the 100-token window of `test/small`.
fn large_input() -> String {
    "x".repeat(400)
}

/// The summary the scripted provider answers a compaction call with.
fn summary() -> Vec<ScriptedCompletion> {
    vec![ScriptedCompletion::response(text_response(
        "Here is the summary of the conversation so far.",
    ))]
}

/// The `context_window` warnings a prompt published.
fn warnings(events: &[CodingEvent]) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|event| match event {
            CodingEvent::Warning { kind, details, .. } if kind == "context_window" => {
                Some(details.clone())
            }
            _ => None,
        })
        .collect()
}

// --- The context window ---

#[tokio::test]
async fn crossing_the_threshold_warns_the_application() {
    let (mut session, _provider) = TestSession::new(answers("OK")).model("test/small").build();
    let mut events = session.subscribe();

    session
        .prompt(&large_input())
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let warned = warnings(&published);
    assert_eq!(warned.len(), 1, "the pressure is reported once");
    assert_eq!(warned[0]["context_window_size"], 100);
}

#[tokio::test]
async fn a_prompt_well_inside_the_window_warns_about_nothing() {
    let (mut session, _provider) = TestSession::answering(answers("OK"));
    let mut events = session.subscribe();

    session.prompt("Hi").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(warnings(&published).is_empty());
}

// --- Compaction ---

#[tokio::test]
async fn crossing_the_threshold_summarizes_the_older_turns() {
    // A response that reports no usage of its own, so the second checkpoint
    // measures the conversation the session actually holds.
    let (mut session, _provider) = TestSession::new(vec![ScriptedCall::response(with_usage(
        text_response("OK"),
        TokenCounts::default(),
    ))])
    .model("test/small")
    .completing(summary())
    .options(CodingSessionOptions {
        enable_context_compaction: true,
        compaction_preserve_turns: 1,
        ..CodingSessionOptions::default()
    })
    .build();
    let mut events = session.subscribe();

    session
        .prompt(&large_input())
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::CompactionStarted { .. })),
        "compaction announces itself"
    );
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::CompactionCompleted { .. })),
    );
    assert!(
        session.history().turns().iter().any(|turn| {
            matches!(turn, Message::System { content, .. }
                if content.contains("A different assistant began this task"))
        }),
        "the summarized turns are replaced by the summary"
    );
}

#[tokio::test]
async fn the_reported_usage_of_the_last_turn_is_the_baseline() {
    let (mut session, _provider) = TestSession::new(vec![ScriptedCall::response(
        with_input_tokens(text_response("OK"), 90),
    )])
    .model("test/small")
    .completing(summary())
    .options(CodingSessionOptions {
        enable_context_compaction: true,
        compaction_preserve_turns: 1,
        ..CodingSessionOptions::default()
    })
    .build();
    let mut events = session.subscribe();

    session.prompt("hi").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    let started = published.iter().find_map(|event| match event {
        CodingEvent::CompactionStarted {
            estimated_tokens,
            context_window_size,
        } => Some((*estimated_tokens, *context_window_size)),
        _ => None,
    });
    assert_eq!(
        started,
        Some((90, 100)),
        "what the provider counted beats what the session estimated"
    );
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::CompactionCompleted { .. }))
    );
}

#[tokio::test]
async fn a_summary_that_never_arrives_outlasts_an_interrupt() {
    // Summarization has no cancellation of its own — fabro's had none either —
    // so a call that never answers holds the prompt open however the session is
    // ended. This pins the contract rather than endorsing it: an application
    // that must stop while a summary is in flight needs the model call itself
    // to time out.
    let (mut session, _provider) = TestSession::new(vec![ScriptedCall::response(with_usage(
        text_response("OK"),
        TokenCounts::default(),
    ))])
    .model("test/small")
    .completing(vec![ScriptedCompletion::Pending])
    .options(CodingSessionOptions {
        enable_context_compaction: true,
        compaction_preserve_turns: 1,
        ..CodingSessionOptions::default()
    })
    .build();
    let cancel = session.cancel_token();
    let mut events = session.subscribe();
    let controller = tokio::spawn(async move {
        timeout(
            Duration::from_secs(5),
            wait_for_event(&mut events, |event| {
                matches!(event, CodingEvent::CompactionStarted { .. })
            }),
        )
        .await
        .expect("the summarizing call starts");
        cancel.cancel();
    });

    let outcome = timeout(Duration::from_millis(200), session.prompt(&large_input())).await;
    controller.await.expect("the controller finishes");

    assert!(
        outcome.is_err(),
        "a cancelled session still waits on the summarizing call it started"
    );
}

#[tokio::test]
async fn a_compaction_with_nothing_to_summarize_announces_nothing() {
    let (mut session, provider) = TestSession::new(answers("OK"))
        .model("test/small")
        .completing(summary())
        .options(CodingSessionOptions {
            enable_context_compaction: true,
            compaction_preserve_turns: 10,
            ..CodingSessionOptions::default()
        })
        .build();
    let mut events = session.subscribe();

    session
        .prompt(&large_input())
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        warnings(&published).len(),
        1,
        "the threshold was crossed all the same"
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::CompactionStarted { .. } | CodingEvent::CompactionCompleted { .. }
        )),
        0,
        "a compaction that would preserve everything is not a compaction"
    );
    assert_eq!(provider.completion_count(), 0, "nothing was summarized");
}

#[tokio::test]
async fn compaction_that_is_turned_off_never_runs() {
    let (mut session, provider) = TestSession::new(answers("OK"))
        .model("test/small")
        .options(CodingSessionOptions {
            enable_context_compaction: false,
            ..CodingSessionOptions::default()
        })
        .build();
    let mut events = session.subscribe();

    session
        .prompt(&large_input())
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::CompactionStarted { .. } | CodingEvent::CompactionCompleted { .. }
        )),
        0
    );
    assert_eq!(provider.completion_count(), 0);
}

#[tokio::test]
async fn a_session_that_never_compacts_still_reports_what_the_provider_counted() {
    let (mut session, _provider) = TestSession::new(vec![ScriptedCall::response(
        with_input_tokens(text_response("OK"), 90),
    )])
    .model("test/small")
    .options(CodingSessionOptions {
        enable_context_compaction: false,
        compaction_preserve_turns: 1,
        ..CodingSessionOptions::default()
    })
    .build();
    let mut events = session.subscribe();

    session.prompt("hi").await.expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        warnings(&published).iter().any(|details| {
            details["estimated_tokens"] == 90
                && details["estimate_method"] == "api_usage_plus_local_delta"
        }),
        "the warning still comes from the provider's own count: {:?}",
        warnings(&published)
    );
    assert_eq!(
        count(&published, |event| matches!(
            event,
            CodingEvent::CompactionStarted { .. } | CodingEvent::CompactionCompleted { .. }
        )),
        0
    );
}

#[tokio::test]
async fn a_failed_compaction_neither_stops_the_prompt_nor_repeats() {
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::response(with_input_tokens(
            tool_call_response("nonexistent_tool", "call_1", json!({})),
            90,
        )),
        ScriptedCall::response(text_response("OK")),
    ])
    .model("test/small")
    .completing(vec![ScriptedCompletion::Failure(
        ScriptedFailure::terminal(LlmErrorKind::Server, "summarization failed"),
    )])
    .options(CodingSessionOptions {
        enable_context_compaction: true,
        compaction_preserve_turns: 1,
        ..CodingSessionOptions::default()
    })
    .build();
    let mut events = session.subscribe();

    session
        .prompt(&large_input())
        .await
        .expect("a failed summary does not fail the prompt");

    assert_eq!(
        provider.completion_count(),
        1,
        "one failure suppresses the rest of this input's attempts"
    );
    let published = settled(&mut session, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::Error { error } if error.kind == ErrorKind::Compaction
        )),
        "the failure is reported as compaction's, not as the prompt's"
    );
}

#[tokio::test]
async fn the_summarizing_call_carries_the_prompt_and_the_files() {
    let read_file = RegisteredTool {
        definition: ToolDefinition::function(
            "read_file",
            "Read a file",
            json!({"type": "object", "properties": {"file_path": {"type": "string"}}}),
        ),
        executor:   Arc::new(|_arguments, _context| {
            Box::pin(async { Ok("file contents".to_owned()) })
        }),
        source:     ToolSource::Native,
    };
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "read_file",
            "tc1",
            json!({"file_path": "/src/main.rs"}),
        )),
        ScriptedCall::response(text_response("OK")),
        ScriptedCall::response(text_response("Done after compaction")),
    ])
    .model("test/small")
    .tools([read_file])
    .completing(vec![ScriptedCompletion::response(text_response(
        "## Goal\nSummary goes here.",
    ))])
    .options(CodingSessionOptions {
        enable_context_compaction: true,
        compaction_preserve_turns: 1,
        ..CodingSessionOptions::default()
    })
    .build();
    let mut events = session.subscribe();

    session
        .prompt("Read the file")
        .await
        .expect("the prompt succeeds");
    assert_eq!(
        session.file_tracker().file_count(),
        1,
        "the file the tool read is tracked"
    );

    session
        .prompt(&large_input())
        .await
        .expect("the second prompt succeeds");

    let requests = provider.completion_requests();
    let request = requests.first().expect("compaction asked the model");
    let system = message_text(
        request
            .messages()
            .first()
            .expect("the summarizing call is not empty"),
    );
    assert!(system.contains("## Goal"), "{system}");
    assert!(system.contains("## File Operations"), "{system}");
    assert!(system.contains("/src/main.rs"), "{system}");
    assert!(system.contains("COPY THIS SECTION VERBATIM"), "{system}");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published
            .iter()
            .any(|event| matches!(event, CodingEvent::CompactionCompleted {
                tracked_file_count: 1,
                ..
            })),
        "the file survives the summary"
    );
}

// --- What the session loads before it starts ---

/// A mock environment holding one file.
fn environment_with(files: Vec<(&str, &str)>, globs: Vec<&str>) -> Arc<MockEnvironment> {
    Arc::new(MockEnvironment {
        files: files
            .into_iter()
            .map(|(path, content)| (path.to_owned(), content.to_owned()))
            .collect::<HashMap<_, _>>(),
        glob_results: globs.into_iter().map(ToOwned::to_owned).collect(),
        ..MockEnvironment::linux()
    })
}

#[tokio::test]
async fn initializing_reports_the_memory_it_loaded() {
    let environment = environment_with(vec![("/home/test/AGENTS.md", "Hello world")], Vec::new());
    let (mut session, _provider) = TestSession::new(answers("ok"))
        .environment(environment)
        .options(CodingSessionOptions {
            memory_files: vec!["/home/test/AGENTS.md".to_owned()],
            ..CodingSessionOptions::default()
        })
        .build();
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");

    let published = settled(&mut session, &mut events).await;
    let loaded = published
        .iter()
        .find_map(|event| match event {
            CodingEvent::MemoryLoaded {
                profile,
                files,
                budget_bytes,
                ..
            } => Some((profile.clone(), files.clone(), *budget_bytes)),
            _ => None,
        })
        .expect("initialization reports what it loaded");
    let (profile, files, budget_bytes) = loaded;
    assert_eq!(profile, "anthropic");
    assert_eq!(budget_bytes, 32768);
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, "/home/test/AGENTS.md");
    assert_eq!(files[0].byte_count, "Hello world".len());
    assert_eq!(files[0].loaded_bytes, "Hello world".len());
    assert!(!files[0].truncated);
}

#[tokio::test]
async fn initializing_reports_an_empty_memory_too() {
    let (mut session, _provider) = TestSession::answering(answers("ok"));
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::MemoryLoaded { files, .. } if files.is_empty()
        )),
        "a session with no memory says so"
    );
}

#[tokio::test]
async fn initializing_reports_the_skills_it_found() {
    let environment = environment_with(
        vec![(
            "/skills/commit/SKILL.md",
            "---\nname: commit\ndescription: Make a commit\n---\nDo commit",
        )],
        vec!["/skills/commit/SKILL.md"],
    );
    let (mut session, _provider) = TestSession::new(answers("ok"))
        .environment(environment)
        .options(CodingSessionOptions {
            skill_dirs: vec!["/skills".to_owned()],
            ..CodingSessionOptions::default()
        })
        .build();
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");

    let published = settled(&mut session, &mut events).await;
    let discovered = published
        .iter()
        .find_map(|event| match event {
            CodingEvent::SkillsDiscovered {
                profile,
                source_dirs,
                skills,
            } => Some((profile.clone(), source_dirs.clone(), skills.clone())),
            _ => None,
        })
        .expect("initialization reports what it discovered");
    let (profile, source_dirs, skills) = discovered;
    assert_eq!(profile, "anthropic");
    assert_eq!(source_dirs, ["/skills".to_owned()]);
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "commit");
    assert_eq!(skills[0].description, "Make a commit");
}

#[tokio::test]
async fn initializing_reports_an_empty_skill_list_too() {
    let (mut session, _provider) = TestSession::answering(answers("ok"));
    let mut events = session.subscribe();

    session.initialize().await.expect("initialization succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::SkillsDiscovered { skills, .. } if skills.is_empty()
        )),
        "a session with no skills says so"
    );
}

#[tokio::test]
async fn a_slash_command_activates_the_skill_it_names() {
    let environment = environment_with(
        vec![(
            "/skills/commit/SKILL.md",
            "---\nname: commit\ndescription: Make a commit\n---\nRun commit. {{user_input}}",
        )],
        vec!["/skills/commit/SKILL.md"],
    );
    let (mut session, _provider) = TestSession::new(answers("ok"))
        .environment(environment)
        .options(CodingSessionOptions {
            skill_dirs: vec!["/skills".to_owned()],
            ..CodingSessionOptions::default()
        })
        .build();
    session.initialize().await.expect("initialization succeeds");
    let mut events = session.subscribe();

    session
        .prompt("/commit fix things")
        .await
        .expect("the prompt succeeds");

    let published = settled(&mut session, &mut events).await;
    assert!(
        published.iter().any(|event| matches!(
            event,
            CodingEvent::SkillActivated { skill_name, source }
                if skill_name == "commit" && *source == SkillActivationSource::Slash
        )),
        "the slash command is reported as the activation it is: {published:?}"
    );
    assert!(
        matches!(
            &session.history().turns()[0],
            Message::User { content, .. } if content.contains("Run commit.")
        ),
        "the skill's template is what the model reads"
    );
}
