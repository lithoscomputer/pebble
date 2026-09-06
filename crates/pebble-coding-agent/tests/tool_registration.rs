//! Tool registration rejects collisions and requires explicit replacement.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lithos_llm::catalog::Catalog;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedProvider, TEST_CATALOG, scripted_client,
    scripted_client_builder, text_response, tool_call_response,
};
use pebble_coding_agent::tools::{CodingToolSet, RegisteredTool, ToolRegistrationError};
use pebble_coding_agent::{CodingAgent, CodingAgentBuildError, ShutdownReason};
use serde_json::json;

fn tool(name: &str) -> RegisteredTool {
    RegisteredTool::function(
        name,
        "Application tool",
        json!({"type": "object"}),
        |_, _| async { Ok("application result".to_owned()) },
    )
}

#[tokio::test]
async fn build_rejects_duplicate_names_and_identities() {
    for (profile, names, expected) in [
        (
            "anthropic",
            vec!["inspect", "inspect"],
            ToolRegistrationError::DuplicateName {
                name: "inspect".to_owned(),
            },
        ),
        ("kimi", vec!["Read"], ToolRegistrationError::DuplicateName {
            name: "Read".to_owned(),
        }),
        (
            "kimi",
            vec!["read_file"],
            ToolRegistrationError::DuplicateIdentity {
                id: "read_file".to_owned(),
            },
        ),
    ] {
        let (builder, provider) = scripted_client_builder(ScriptedProvider::new(vec![]));
        let catalog = Catalog::builder()
            .overlay_toml(&TEST_CATALOG.replace(
                "profile = \"anthropic\"",
                &format!("profile = \"{profile}\""),
            ))
            .expect("valid catalog")
            .build()
            .expect("catalog builds");
        let client = builder
            .catalog(catalog)
            .build()
            .expect("client builds")
            .client;
        let result = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .tools(names.into_iter().map(tool))
            .build()
            .await;
        assert!(
            matches!(result, Err(CodingAgentBuildError::ToolRegistration(error)) if error == expected)
        );
        assert_eq!(provider.call_count(), 0);
    }
}

#[tokio::test]
async fn replacement_keeps_the_profile_name_and_executes_the_new_tool() {
    for (profile, visible_name) in [("anthropic", "read_file"), ("kimi", "Read")] {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let replacement = RegisteredTool::function(
            "application_reader",
            "Read through the application",
            json!({"type": "object", "properties": {"key": {"type": "string"}}}),
            move |_, arguments| {
                observed.fetch_add(1, Ordering::SeqCst);
                async move {
                    assert_eq!(arguments["key"], "value");
                    Ok("replacement result".to_owned())
                }
            },
        );
        let (builder, provider) = scripted_client_builder(ScriptedProvider::new(vec![
            ScriptedCall::response(tool_call_response(
                visible_name,
                "call_1",
                json!({"key": "value"}),
            )),
            ScriptedCall::response(text_response("done")),
        ]));
        let catalog = Catalog::builder()
            .overlay_toml(&TEST_CATALOG.replace(
                "profile = \"anthropic\"",
                &format!("profile = \"{profile}\""),
            ))
            .expect("valid catalog")
            .build()
            .expect("catalog builds");
        let client = builder
            .catalog(catalog)
            .build()
            .expect("client builds")
            .client;
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .replace_tool("read_file", replacement)
            .build()
            .await
            .expect("replacement builds");
        agent
            .prompt("read a value")
            .await
            .result
            .expect("prompt completes");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let requests = provider.requests();
        let definition = requests[0]
            .tools()
            .iter()
            .find(|tool| tool.name == visible_name)
            .expect("visible tool");
        assert_eq!(definition.description, "Read through the application");
        assert!(
            requests[0]
                .tools()
                .iter()
                .all(|tool| tool.name != "application_reader")
        );
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shutdown");
    }
}

#[tokio::test]
async fn unknown_replacement_is_a_build_error() {
    let (client, _) = scripted_client(vec![]);
    let result = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .replace_tool("missing", tool("inspect"))
        .build()
        .await;
    assert!(
        matches!(result, Err(CodingAgentBuildError::ToolRegistration(
        ToolRegistrationError::UnknownReplacement { id }
    )) if id == "missing")
    );
}

#[test]
fn standalone_tools_require_explicit_replacement() {
    assert!(matches!(
        CodingToolSet::core().with_tool(tool("read_file")),
        Err(ToolRegistrationError::DuplicateName { .. })
    ));
    let tools = CodingToolSet::core()
        .replace_tool("read_file", tool("application_reader"))
        .expect("explicit replacement");
    let summaries = tools.summaries();
    let replaced = summaries
        .iter()
        .find(|tool| tool.name == "read_file")
        .expect("reader");
    assert_eq!(replaced.description, "Application tool");
    assert!(
        summaries
            .iter()
            .all(|tool| tool.name != "application_reader")
    );
}
