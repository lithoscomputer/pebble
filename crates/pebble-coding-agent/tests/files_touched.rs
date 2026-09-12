//! What a prompt changed, as its report says.
//!
//! An application that runs an agent to change files wants to know which
//! files, without reading the event stream back. The report names every file
//! the prompt wrote or edited, sorted, and the one it touched last; a call
//! that failed touched nothing; and each prompt starts from an empty list.

use std::sync::Arc;

use pebble_coding_agent::events::PermissionLevel;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedProvider, client_from, multi_tool_call_response,
    text_response, tool_call_response,
};
use pebble_coding_agent::{CodingAgent, CodingAgentOptions, ShutdownReason};
use serde_json::{Value, json};

fn write(path: &str) -> (&'static str, &str, Value) {
    (
        "write_file",
        path,
        json!({"file_path": path, "content": "text"}),
    )
}

async fn agent_with(calls: Vec<ScriptedCall>, level: PermissionLevel) -> CodingAgent {
    let (client, _provider) = client_from(ScriptedProvider::new(calls));
    CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .permission_level(level)
        .options(CodingAgentOptions::default().with_loop_detection(false))
        .build()
        .await
        .expect("the coding agent builds")
}

#[tokio::test]
async fn the_report_names_every_file_the_prompt_wrote_and_the_last_one() {
    let mut agent = agent_with(
        vec![
            ScriptedCall::response(multi_tool_call_response(vec![
                write("/home/test/zeta.txt"),
                write("/home/test/alpha.txt"),
            ])),
            ScriptedCall::response(tool_call_response(
                "write_file",
                "again",
                json!({"file_path": "/home/test/zeta.txt", "content": "more"}),
            )),
            ScriptedCall::response(text_response("done")),
        ],
        PermissionLevel::Full,
    )
    .await;

    let report = agent.prompt("write two files").await;

    assert!(report.result.is_ok(), "{report:?}");
    assert_eq!(
        report.files_touched,
        ["/home/test/alpha.txt", "/home/test/zeta.txt"],
        "sorted, each path once"
    );
    assert_eq!(
        report.last_file_touched.as_deref(),
        Some("/home/test/zeta.txt")
    );

    let next = agent.prompt("now just talk").await;
    assert!(
        next.result.is_err() || next.files_touched.is_empty(),
        "a new prompt starts with nothing touched: {next:?}"
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_refused_write_touches_nothing() {
    let mut agent = agent_with(
        vec![
            ScriptedCall::response(tool_call_response(
                "write_file",
                "denied",
                json!({"file_path": "/home/test/never.txt", "content": "text"}),
            )),
            ScriptedCall::response(text_response("could not")),
        ],
        PermissionLevel::ReadOnly,
    )
    .await;

    let report = agent.prompt("try to write").await;

    assert!(report.result.is_ok(), "{report:?}");
    assert!(report.files_touched.is_empty(), "{report:?}");
    assert!(report.last_file_touched.is_none());
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}
