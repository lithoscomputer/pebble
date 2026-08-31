//! Which harness a model resolves to.
//!
//! Pebble never guesses: the resolved catalog row names the harness, the model
//! answering for its provider where both say something. These tests run against
//! the catalog lithos actually ships, so a builtin row that stops naming a
//! profile pebble knows fails here rather than in an application.

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::client::ClientBuild;

use super::super::testing::TestProfile;
use super::*;
use crate::test_support::{MockEnvironment, scripted_client};

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

/// Builds a session on the shipped catalog, which fails while pebble ships no
/// profile implementations — the failure names the profile that was resolved.
fn resolved_profile(selector: &str) -> StdResult<AgentProfileKind, SessionBuildError> {
    let error = Session::builder(shipped_client())
        .model(selector)
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .expect_err("pebble ships no built-in profiles yet, so every build fails");
    match error {
        SessionBuildError::ProfileUnavailable { profile } => Ok(profile),
        other => Err(other),
    }
}

#[tokio::test]
async fn every_profile_the_shipped_catalog_names_is_one_pebble_runs() {
    let cases = [
        ("anthropic/claude-sonnet-5", AgentProfileKind::Claude5),
        ("openai/gpt-5.5", AgentProfileKind::OpenAi),
        ("openai/gpt-5.6-sol", AgentProfileKind::Gpt56),
        ("gemini/gemini-3.5-flash", AgentProfileKind::Gemini),
        ("moonshot/kimi-k3", AgentProfileKind::Kimi),
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
        Session::builder(shipped_client())
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
            SessionBuildError::UnknownProfile { ref profile, .. } if profile == "nonesuch"
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_model_naming_no_profile_anywhere_is_refused() {
    // The shipped catalog gives every provider a profile, so "names none" only
    // exists in a catalog an application wrote: the scripted `bare` provider.
    let (client, _provider) = scripted_client(answers("done"));

    let error = Session::builder(client)
        .model("bare/plain")
        .environment(Arc::new(MockEnvironment::linux()))
        .build()
        .expect_err("nothing names a harness");

    assert!(
        matches!(
            error,
            SessionBuildError::MissingProfileMetadata { ref model } if model == "bare/plain"
        ),
        "{error:?}"
    );
}
