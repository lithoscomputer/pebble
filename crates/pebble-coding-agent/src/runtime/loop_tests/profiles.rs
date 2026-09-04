//! Which harness a model resolves to.
//!
//! Pebble never guesses: the resolved catalog row names the harness, the model
//! answering for its provider where both say something. These tests run against
//! the catalog lithos actually ships, so a builtin row that stops naming a
//! profile pebble knows fails here rather than in an application.

use std::fmt::Write as _;
use std::result::Result as StdResult;
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::client::ClientBuild;
use lithos_llm::types::Role;

use super::super::testing::TestProfile;
use super::*;
use crate::human_input::{Answer, HumanInputError, HumanInputProvider, Question};
use crate::search::{SearchError, SearchProvider, SearchRequest, SearchResult};
use crate::subagent::{ChildObserver, SubagentOptions};
use crate::test_support::{
    MockEnvironment, ScriptedProvider, TEST_CATALOG, message_text, scripted_client,
    scripted_client_builder,
};
use crate::tool::NativeTool;

/// Two models added to the shipped `anthropic` provider, for the two answers
/// the shipped catalog cannot give on its own: a row that names no profile of
/// its own, and one that names a profile pebble does not know.
const OVERLAY: &str = r#"
schema_version = 1

[providers.anthropic.models."pebble-test-inherits"]
display_name = "Inherits the provider's profile"
api_model = "pebble-test-inherits"

[providers.anthropic.models."pebble-test-unknown"]
display_name = "Names a profile pebble does not know"
api_model = "pebble-test-unknown"

[providers.anthropic.models."pebble-test-unknown".metadata.pebble]
profile = "nonesuch"
"#;

/// A client over the catalog lithos ships, plus [`OVERLAY`].
fn shipped_client() -> Client {
    let catalog = Catalog::builder()
        .with_builtin()
        .overlay_toml(OVERLAY)
        .expect("the overlay parses")
        .build()
        .expect("the shipped catalog validates");
    let ClientBuild { client, .. } = Client::builder()
        .catalog(catalog)
        .build()
        .expect("the client builds");
    client
}

/// The harness a session on the shipped catalog resolves `selector` to.
fn resolved_profile(selector: &str) -> StdResult<AgentProfileKind, CodingAgentBuildError> {
    CodingRuntime::builder(shipped_client())
        .model(selector)
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .map(|session| session.profile_kind())
}

#[tokio::test]
async fn every_profile_the_shipped_catalog_names_is_one_pebble_runs() {
    let cases = [
        ("anthropic/claude-sonnet-5", AgentProfileKind::Claude5),
        ("openai/gpt-5.5", AgentProfileKind::OpenAi),
        ("openai/gpt-5.6-sol", AgentProfileKind::Gpt56),
        ("openai/gpt-6-astra", AgentProfileKind::Gpt6),
        ("gemini/gemini-3.5-flash", AgentProfileKind::Gemini),
        ("moonshot/kimi-k3", AgentProfileKind::Kimi),
        // The same model through a gateway: a Kimi row names the Kimi harness
        // wherever it is served from. The other half of that rule — a non-Kimi
        // model on the same gateway keeps the gateway's own profile — is
        // `openrouter/deepseek-v4-pro` in
        // `the_route_decides_which_file_editor_the_openai_harness_offers`.
        ("openrouter/kimi-k3", AgentProfileKind::Kimi),
        // The provider's own default, which no shipped model of that provider
        // leaves in place.
        (
            "anthropic/pebble-test-inherits",
            AgentProfileKind::Anthropic,
        ),
    ];

    for (selector, expected) in cases {
        assert_eq!(
            resolved_profile(selector).expect("the model names a profile pebble knows"),
            expected,
            "{selector} resolves to the harness its catalog row names"
        );
    }
}

#[tokio::test]
async fn the_shipped_catalog_says_which_models_reason_without_being_asked() {
    // Fabro's rule, read from the catalog lithos ships: a model the request
    // can set an effort level on reasons unless it is told not to, one that
    // takes only a thinking budget reasons when it is asked to, and a row
    // that knows better says so itself.
    let reasons = |selector: &str| {
        CodingRuntime::builder(shipped_client())
            .model(selector)
            .environment(Arc::new(MockEnvironment::linux()))
            .with_profile(TestProfile::shared())
            .build()
            .expect("the session builds")
            .model_facts()
            .reasons_by_default
    };

    assert!(
        reasons("anthropic/claude-sonnet-5"),
        "it takes effort levels"
    );
    assert!(
        !reasons("anthropic/claude-sonnet-4.5"),
        "it takes a thinking budget, and reasons when a request asks it to"
    );
    assert!(
        reasons("venice/glm-5.3"),
        "the row says it reasons on every call"
    );
}

#[tokio::test]
async fn a_model_naming_a_profile_pebble_does_not_know_is_refused() {
    let error = resolved_profile("anthropic/pebble-test-unknown")
        .expect_err("`nonesuch` is not a pebble profile");

    assert!(
        matches!(
            error,
            CodingAgentBuildError::UnknownProfile { ref profile, .. } if profile == "nonesuch"
        ),
        "{error:?}"
    );
}

// --- What a built-in harness gives one session ---

/// A search engine that is never asked anything: what these tests are about is
/// whether the tool is advertised at all.
struct Unused;

#[async_trait]
impl SearchProvider for Unused {
    async fn search(&self, _request: SearchRequest) -> StdResult<Vec<SearchResult>, SearchError> {
        Ok(Vec::new())
    }
}

#[async_trait]
impl HumanInputProvider for Unused {
    async fn ask_questions(
        &self,
        _tool_call_id: &str,
        _questions: Vec<Question>,
        _cancel_token: CancellationToken,
    ) -> StdResult<Vec<Answer>, HumanInputError> {
        Ok(Vec::new())
    }
}

/// How a session is configured beyond its model.
#[derive(Clone, Copy, Default)]
struct Configured {
    search:    bool,
    subagents: bool,
    questions: bool,
    skills:    bool,
}

/// One skill for `initialize` to find, so the tool that loads one is
/// registered.
fn one_skill() -> Arc<MockEnvironment> {
    Arc::new(MockEnvironment {
        files: [(
            "/skills/commit/SKILL.md".to_owned(),
            "---\nname: commit\ndescription: Make a commit\n---\nRun commit.".to_owned(),
        )]
        .into_iter()
        .collect(),
        glob_results: vec!["/skills/commit/SKILL.md".to_owned()],
        ..MockEnvironment::linux()
    })
}

/// An initialized session on the shipped catalog, so its prompt is built and
/// the tools `initialize` adds for itself are registered.
async fn initialized(selector: &str, configured: Configured) -> CodingRuntime {
    let (environment, options) = if configured.skills {
        (one_skill(), CodingAgentOptions {
            skill_dirs: vec!["/skills".to_owned()],
            ..CodingAgentOptions::default()
        })
    } else {
        (
            Arc::new(MockEnvironment::linux()),
            CodingAgentOptions::default(),
        )
    };
    let mut builder = CodingRuntime::builder(shipped_client())
        .model(selector)
        .environment(environment)
        .options(options);
    if configured.search {
        builder = builder.search_provider(Arc::new(Unused));
    }
    if configured.subagents {
        builder = builder.subagents(SubagentOptions::enabled());
    }
    if configured.questions {
        builder = builder.human_input(Arc::new(Unused));
    }
    let mut session = builder.build().expect("the session builds");
    session.initialize().await.expect("initialization succeeds");
    session
}

/// Whether `session` advertises a tool called `name`.
fn advertises(session: &CodingRuntime, name: &str) -> bool {
    session
        .registered_tools()
        .iter()
        .any(|tool| tool.definition.name == name)
}

/// Every tool `session` shows its model, sorted.
fn tool_names(session: &CodingRuntime) -> Vec<String> {
    let mut names: Vec<String> = session
        .registered_tools()
        .into_iter()
        .map(|tool| tool.definition.name)
        .collect();
    names.sort();
    names
}

fn child_tool_names(child: &CodingRuntime) -> Vec<String> {
    let mut names: Vec<String> = child
        .registered_tools()
        .into_iter()
        .map(|tool| tool.definition.name)
        .collect();
    names.sort();
    names
}

/// The harnesses pebble ships: a model that resolves to each, and whether
/// that harness's prompt has a section gated on the search tool.
///
/// Kimi Code's prompt is the one with no conditionals at all — it keeps every
/// piece of tool guidance in the tool descriptions — so its model learns it can
/// search from the tool list alone.
const SHIPPED: [(&str, AgentProfileKind, bool); 7] = [
    (
        "anthropic/pebble-test-inherits",
        AgentProfileKind::Anthropic,
        true,
    ),
    ("anthropic/claude-sonnet-5", AgentProfileKind::Claude5, true),
    ("openai/gpt-5.5", AgentProfileKind::OpenAi, true),
    ("openai/gpt-5.6-sol", AgentProfileKind::Gpt56, true),
    ("openai/gpt-6-astra", AgentProfileKind::Gpt6, true),
    ("gemini/gemini-3.5-flash", AgentProfileKind::Gemini, true),
    ("moonshot/kimi-k3", AgentProfileKind::Kimi, false),
];

/// What `session`'s model calls the search tool.
fn search_tool_name(session: &CodingRuntime) -> &'static str {
    NativeTool::WebSearch.name(session.resources.registry.vocabulary())
}

/// A prompt must never name a tool the session does not have, and must always
/// name one it does. Fabro kept these in step by reading the prompt's
/// conditionals out of the registry; pebble builds both from one answer, and
/// this is what says the two answers stayed the same one.
#[tokio::test]
async fn a_prompt_advertises_a_search_tool_exactly_when_the_session_has_one() {
    for (selector, kind, prompt_gates_search) in SHIPPED {
        for search in [false, true] {
            let session = initialized(selector, Configured {
                search,
                ..Configured::default()
            })
            .await;
            let name = search_tool_name(&session);

            assert_eq!(advertises(&session, name), search, "{kind}");
            assert_eq!(
                session.resources.system_prompt.contains(name),
                search && prompt_gates_search,
                "{kind} prompt disagrees with its registry"
            );
        }
    }
}

#[tokio::test]
async fn the_anthropic_prompt_mentions_subagents_exactly_when_the_session_can_spawn_them() {
    for subagents in [false, true] {
        let session = initialized("anthropic/pebble-test-inherits", Configured {
            subagents,
            ..Configured::default()
        })
        .await;

        assert_eq!(advertises(&session, "spawn_agent"), subagents);
        assert_eq!(
            session
                .resources
                .system_prompt
                .contains("Subagents are valuable for independent work"),
            subagents
        );
    }
}

/// Which editor a session offers is a property of the route, not of the model:
/// only the OpenAI Responses codec can carry a freeform patch grammar.
#[tokio::test]
async fn the_route_decides_which_file_editor_the_openai_harness_offers() {
    let responses = initialized("openai/gpt-5.5", Configured::default()).await;
    assert!(advertises(&responses, "apply_patch"));
    assert!(!advertises(&responses, "edit_file"));
    assert!(responses.resources.system_prompt.contains("## apply_patch"));

    let compatible = initialized("openrouter/deepseek-v4-pro", Configured::default()).await;
    assert_eq!(compatible.profile_kind(), AgentProfileKind::OpenAi);
    assert!(advertises(&compatible, "edit_file"));
    assert!(!advertises(&compatible, "apply_patch"));
    assert!(compatible.resources.system_prompt.contains("## edit_file"));
    assert!(
        compatible
            .registered_tools()
            .iter()
            .all(|tool| !tool.definition.is_custom()),
        "a codec that refuses freeform tools must be offered none"
    );
}

/// The same rule on the harness that has nothing else to edit files with.
///
/// The OpenAI harness losing its patch tool falls back to an editor it also
/// knows; 5.6 is given three tools in total, so a route that cannot carry the
/// patch grammar has to be handed the editor or the harness cannot write at
/// all. The shell's description names the editor too, and would point 5.6 at a
/// tool it was never given if the two came apart.
#[tokio::test]
async fn the_route_decides_which_file_editor_the_gpt56_harness_offers() {
    let describe_shell = |session: &CodingRuntime| {
        session
            .registered_tools()
            .into_iter()
            .find(|tool| tool.definition.name == "shell_command")
            .expect("the harness offers Codex's shell")
            .definition
            .description
    };

    let responses = initialized("openai/gpt-5.6-sol", Configured::default()).await;
    assert_eq!(responses.profile_kind(), AgentProfileKind::Gpt56);
    assert!(advertises(&responses, "apply_patch"));
    assert!(!advertises(&responses, "edit_file"));
    assert!(describe_shell(&responses).contains("`apply_patch`"));
    assert!(
        responses
            .resources
            .system_prompt
            .contains("Use `apply_patch` for local file edits")
    );

    let compatible = initialized("openrouter/gpt-5.6-sol", Configured::default()).await;
    assert_eq!(compatible.profile_kind(), AgentProfileKind::Gpt56);
    assert!(advertises(&compatible, "edit_file"));
    assert!(!advertises(&compatible, "apply_patch"));
    assert!(describe_shell(&compatible).contains("`edit_file`"));
    assert!(
        compatible
            .resources
            .system_prompt
            .contains("Use `edit_file` for local file edits")
    );
    assert!(
        compatible
            .registered_tools()
            .iter()
            .all(|tool| !tool.definition.is_custom()),
        "a codec that refuses freeform tools must be offered none"
    );
}

/// A question tool reaches the model only where the application named someone
/// to ask — and never on the Gemini harness, which has no such tool at all.
#[tokio::test]
async fn a_question_tool_is_registered_where_the_harness_has_one_and_a_person_to_ask() {
    let expected = [
        ("anthropic/pebble-test-inherits", Some("AskUserQuestion")),
        ("anthropic/claude-sonnet-5", Some("AskUserQuestion")),
        ("moonshot/kimi-k3", Some("AskUserQuestion")),
        ("openai/gpt-5.5", Some("request_user_input")),
        ("openai/gpt-5.6-sol", Some("request_user_input")),
        ("gemini/gemini-3.5-flash", None),
    ];

    for (selector, tool) in expected {
        let asked = initialized(selector, Configured {
            questions: true,
            ..Configured::default()
        })
        .await;
        let unasked = initialized(selector, Configured::default()).await;

        for name in ["AskUserQuestion", "request_user_input"] {
            assert_eq!(
                advertises(&asked, name),
                tool == Some(name),
                "{selector} with someone to ask"
            );
            assert!(!advertises(&unasked, name), "{selector} with nobody to ask");
        }
    }
}

// --- What the model is actually shown ---

/// The Claude 5 harness overrides the trait's default subagent family. Getting
/// pebble's own four here would be silent: the model would be shown four tools
/// it was never trained on, and the ones it *was* trained on would be missing.
#[tokio::test]
async fn the_claude_5_harness_gets_its_own_subagent_family() {
    let session = initialized("anthropic/claude-sonnet-5", Configured {
        subagents: true,
        ..Configured::default()
    })
    .await;

    for expected in ["Agent", "TaskOutput", "TaskStop", "SendMessage"] {
        assert!(
            advertises(&session, expected),
            "the harness lost {expected}"
        );
    }
    for absent in ["spawn_agent", "send_input", "wait", "close_agent"] {
        assert!(
            !advertises(&session, absent),
            "{absent} is pebble's own family, which this model was not trained on"
        );
    }
}

/// Every other harness inherits pebble's own four.
#[tokio::test]
async fn every_other_harness_gets_pebbles_own_subagent_family() {
    for (selector, kind, _) in SHIPPED {
        if kind == AgentProfileKind::Claude5 {
            continue;
        }
        let session = initialized(selector, Configured {
            subagents: true,
            ..Configured::default()
        })
        .await;

        for expected in ["spawn_agent", "send_input", "wait", "close_agent"] {
            assert!(advertises(&session, expected), "{kind} lost {expected}");
        }
        assert!(!advertises(&session, "Agent"), "{kind}");
    }
}

/// Every tool every harness shows its model, in every configuration that
/// changes the answer.
///
/// This is the plan's guard against a tool reaching a model silently. Five
/// separate things decide what a model is shown — the profile's own tools, the
/// search tool and question tool the builder adds, the subagent family the
/// profile contributes, and the skill tool `initialize` registers for itself —
/// and nothing in the crate shows all of them together except this snapshot, so
/// a change anywhere along that path has to be reviewed here. The names are the
/// wire ones, so a vocabulary that stopped renaming a tool shows up too.
///
/// The `search + subagents` row is also what a *child* session is shown, which
/// two tests check on sessions that really spawn one: a child inherits its
/// parent's search engine and factory through the spec, and deliberately
/// inherits no person to ask.
/// `a_child_is_shown_its_parents_tools_without_the_person_to_ask` measures it
/// on the Anthropic harness and
/// `a_child_is_never_told_to_ask_the_user_a_question` on the Claude 5 one,
/// which is the harness with its own vocabulary, its own subagent family and a
/// prompt section about asking.
#[tokio::test]
async fn every_harness_shows_its_model_exactly_these_tools() {
    let configurations = [
        ("nothing configured", Configured::default()),
        ("search", Configured {
            search: true,
            ..Configured::default()
        }),
        ("subagents", Configured {
            subagents: true,
            ..Configured::default()
        }),
        (
            "search + subagents (also what a child is shown)",
            Configured {
                search: true,
                subagents: true,
                ..Configured::default()
            },
        ),
        ("a skill was discovered", Configured {
            skills: true,
            ..Configured::default()
        }),
        ("everything, root only", Configured {
            search:    true,
            subagents: true,
            questions: true,
            skills:    true,
        }),
    ];

    for (selector, kind, _) in SHIPPED {
        let mut rendered = String::new();
        for (label, configured) in configurations {
            let mut session = initialized(selector, configured).await;
            let _ = writeln!(
                rendered,
                "{label}:\n  {}",
                tool_names(&session).join("\n  ")
            );
            session
                .shutdown(ShutdownReason::Completed)
                .await
                .expect("the session closes");
        }

        insta::assert_snapshot!(format!("effective_tools_{kind}"), rendered);
    }
}

/// The snapshot above claims its `search + subagents` row is also what a child
/// is shown. This is what checks the claim, on a session that really spawns
/// one.
///
/// A child inherits its parent's engine and its factory through the spec, and
/// deliberately inherits no person to ask: a child is built with no
/// `HumanInputProvider`, which is the whole of what makes questions root-only.
/// So a root configured with all three shows its model exactly one tool more
/// than its child does.
///
/// This runs on the Anthropic harness, which is what `test/model` resolves to.
/// `a_child_is_never_told_to_ask_the_user_a_question` runs the same spawn on
/// Claude 5, where the vocabulary, the subagent family and the prompt all
/// differ.
#[tokio::test]
async fn a_child_is_shown_its_parents_tools_without_the_person_to_ask() {
    let child_tools: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&child_tools);
    let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
        recorder
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(child_tool_names(child));
    });

    let (client, _provider) = scripted_client(answers("done"));
    let mut parent = CodingRuntime::builder(client)
        .model("test/model")
        .environment(Arc::new(MockEnvironment::linux()))
        .search_provider(Arc::new(Unused))
        .human_input(Arc::new(Unused))
        .observe_children(observer)
        .build()
        .expect("the session builds");
    parent.initialize().await.expect("initialization succeeds");
    let supervisor = parent
        .subagent_supervisor()
        .expect("the parent was given a factory")
        .clone();

    let agent_id = supervisor
        .spawn(parent.identity(), "work".to_owned())
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&agent_id, &CancellationToken::new())
        .await
        .expect("the child answers");

    let child = child_tools
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .first()
        .cloned()
        .expect("the factory built one child");
    let parent_tools = tool_names(&parent);

    assert!(
        parent_tools.contains(&"AskUserQuestion".to_owned()),
        "{parent_tools:?}"
    );
    assert_eq!(
        child,
        parent_tools
            .iter()
            .filter(|name| name.as_str() != "AskUserQuestion")
            .cloned()
            .collect::<Vec<_>>(),
        "a child is shown its parent's tools, minus the one that needs a person"
    );
    // Named rather than implied: these are the two the child inherits through
    // the spec, and the sharpest way the inheritance could break.
    assert!(child.contains(&"web_search".to_owned()), "{child:?}");
    assert!(child.contains(&"spawn_agent".to_owned()), "{child:?}");

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

/// A Claude 5 model on the scripted route.
///
/// The test above runs on `test/model`, which is the Anthropic harness. The
/// child rules it checks are the same for every harness, but only Claude 5's
/// prompt has a section about asking a person, and only Claude 5 replaces the
/// subagent family — so the harness that can get a child wrong is this one, and
/// the child has to be spawned on it.
const CLAUDE5_MODEL: &str = r#"
schema_version = 1

[providers.test.models."claude-5"]
display_name = "Claude 5"
api_model = "claude-5"
capabilities = { text = true, tools = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

[providers.test.models."claude-5".metadata.pebble]
profile = "claude-5"
"#;

/// A scripted client whose `test` provider also offers [`CLAUDE5_MODEL`].
fn claude5_scripted_client(calls: Vec<ScriptedCall>) -> (Client, Arc<ScriptedProvider>) {
    let catalog = Catalog::builder()
        .overlay_toml(TEST_CATALOG)
        .expect("the test catalog parses")
        .overlay_toml(CLAUDE5_MODEL)
        .expect("the overlay parses")
        .build()
        .expect("the catalog validates");
    let (builder, provider) = scripted_client_builder(ScriptedProvider::new(calls));
    let ClientBuild { client, .. } = builder
        .catalog(catalog)
        .build()
        .expect("the scripted client builds");
    (client, provider)
}

/// The system prompt of the one round `provider` was asked for.
fn requested_system_prompt(provider: &ScriptedProvider) -> String {
    let requests = provider.requests();
    assert_eq!(requests.len(), 1, "exactly one session ran a turn");
    let request = requests.first().expect("a round was requested");
    let system = request
        .messages()
        .first()
        .expect("the request is not empty");
    assert_eq!(system.role(), Role::System);
    message_text(system)
}

/// A prompt must never name a tool the session lacks — and a child is the
/// session where that is easiest to get wrong, because it runs its parent's
/// profile and inherits everything on it *except* the person to ask.
///
/// Claude 5 is the harness where it shows: its template is the only one with a
/// section about asking, so a child built from a profile that remembered its
/// parent's answer would be told to call `AskUserQuestion` and then not be
/// given it. The prompt is read off the wire rather than off the session,
/// because what reaches the model is the whole point.
#[tokio::test]
async fn a_child_is_never_told_to_ask_the_user_a_question() {
    let child_tools: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&child_tools);
    let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
        recorder
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(child_tool_names(child));
    });

    let (client, provider) = claude5_scripted_client(answers("done"));
    let mut parent = CodingRuntime::builder(client)
        .model("test/claude-5")
        .environment(Arc::new(MockEnvironment::linux()))
        .search_provider(Arc::new(Unused))
        .human_input(Arc::new(Unused))
        .observe_children(observer)
        .build()
        .expect("the session builds");
    parent.initialize().await.expect("initialization succeeds");
    assert_eq!(parent.profile_kind(), AgentProfileKind::Claude5);
    assert!(
        parent.resources.system_prompt.contains("# Asking the user"),
        "the root has someone to ask, so its own prompt says so"
    );

    let supervisor = parent
        .subagent_supervisor()
        .expect("the parent was given a factory")
        .clone();
    let agent_id = supervisor
        .spawn(parent.identity(), "work".to_owned())
        .expect("the spawn succeeds");
    supervisor
        .wait_with_cancel(&agent_id, &CancellationToken::new())
        .await
        .expect("the child answers");

    // The parent never ran a turn, so the one round asked for is the child's.
    let child_prompt = requested_system_prompt(&provider);
    assert!(
        child_prompt.contains("You are Claude, a software engineering agent running in Pebble."),
        "the child runs its parent's harness"
    );
    assert!(
        !child_prompt.contains("# Asking the user"),
        "a child has nobody to ask: {child_prompt}"
    );
    assert!(
        !child_prompt.contains("Use `AskUserQuestion`"),
        "{child_prompt}"
    );

    let child = child_tools
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .first()
        .cloned()
        .expect("the factory built one child");
    assert!(!child.contains(&"AskUserQuestion".to_owned()), "{child:?}");
    // The same inheritance the Anthropic-harness test checks, in this
    // harness's own vocabulary: its search tool and its background agents.
    assert!(child.contains(&"WebSearch".to_owned()), "{child:?}");
    assert!(child.contains(&"Agent".to_owned()), "{child:?}");

    parent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the parent shuts down");
}

#[tokio::test]
async fn a_model_naming_no_profile_anywhere_is_refused() {
    // The shipped catalog gives every provider a profile, so "names none" only
    // exists in a catalog an application wrote: the scripted `bare` provider.
    let (client, _provider) = scripted_client(answers("done"));

    let error = CodingRuntime::builder(client)
        .model("bare/plain")
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .expect_err("nothing names a harness");

    assert!(
        matches!(
            error,
            CodingAgentBuildError::MissingProfileMetadata { ref model } if model == "bare/plain"
        ),
        "{error:?}"
    );
}
