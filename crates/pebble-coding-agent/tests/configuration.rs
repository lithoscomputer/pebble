//! Configuration is validated before the agent starts work.

use std::sync::Arc;

use pebble_agent::{Agent, AgentBuildError, AgentConfig};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, scripted_client, text_response,
};
use pebble_coding_agent::{
    CodingAgent, CodingAgentBuildError, CodingAgentOptions, CodingAgentOptionsError, ResumeMode,
    ShutdownReason,
};

#[tokio::test]
async fn invalid_limits_fail_build_before_any_model_call() {
    for value in [-1, 0, i64::from(u32::MAX) + 1, i64::MAX] {
        let (client, provider) = scripted_client(vec![]);
        let result = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .options(CodingAgentOptions::default().with_max_tokens(Some(value)))
            .build()
            .await;
        assert!(matches!(result, Err(CodingAgentBuildError::InvalidOptions {
            source: CodingAgentOptionsError::MaxTokens { value: actual },
        }) if actual == value));
        assert_eq!(provider.call_count(), 0);
    }
    for value in [0, 101, usize::MAX] {
        let (client, _) = scripted_client(vec![]);
        let result = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .options(CodingAgentOptions::default().with_compaction_threshold_percent(value))
            .build()
            .await;
        assert!(matches!(result, Err(CodingAgentBuildError::InvalidOptions {
            source: CodingAgentOptionsError::CompactionThreshold { value: actual },
        }) if actual == value));
    }
}

#[tokio::test]
async fn valid_limits_reach_the_request_and_none_uses_the_catalog() {
    for (limit, expected) in [(Some(1), 1), (Some(32000), 32000), (None, 32000)] {
        let (client, provider) =
            scripted_client(vec![ScriptedCall::response(text_response("done"))]);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .options(CodingAgentOptions::default().with_max_tokens(limit))
            .build()
            .await
            .expect("valid options");
        agent.prompt("work").await.expect("answers");
        assert_eq!(provider.requests()[0].max_output_tokens(), Some(expected));
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shutdown");
    }
    for percent in [1, 100] {
        let (client, _) = scripted_client(vec![]);
        let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
            .model("test/model")
            .options(CodingAgentOptions::default().with_compaction_threshold_percent(percent))
            .build()
            .await
            .expect("valid percentage");
        agent
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shutdown");
    }
}

#[tokio::test]
async fn resume_validates_options_too() {
    let (client, _) = scripted_client(vec![]);
    let mut agent = CodingAgent::builder(client.clone(), Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .build()
        .await
        .expect("builds");
    let record = agent.to_record();
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("shutdown");
    let result = CodingAgent::resume(
        client,
        Arc::new(MockEnvironment::linux()),
        record,
        ResumeMode::RecordedModel,
    )
    .options(CodingAgentOptions::default().with_max_tokens(Some(0)))
    .build()
    .await;
    assert!(matches!(
        result,
        Err(CodingAgentBuildError::InvalidOptions { .. })
    ));
}

#[test]
fn the_generic_builder_rejects_zero_output_tokens() {
    let (client, _) = scripted_client(vec![]);
    let result = Agent::builder(client, "test/model")
        .config(AgentConfig {
            max_output_tokens: Some(0),
            ..AgentConfig::default()
        })
        .build();
    assert!(matches!(result, Err(AgentBuildError::ZeroOutputTokens)));
}
