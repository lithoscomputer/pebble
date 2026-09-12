//! What a prompt compacted, as its report says.
//!
//! An application that meters compaction wants to know what each prompt
//! summarized and what the summary call cost, without reading the history
//! back. The report lists every compaction the prompt performed with the
//! facts its history turn records; the summary call's usage and cost are
//! already in the report's totals, so the list is a breakdown of them; a
//! prompt that compacted nothing lists nothing; and a compaction the
//! application asked for between prompts belongs to no prompt.

use std::sync::Arc;

use lithos_llm::types::TokenCounts;
use pebble_coding_agent::events::TokenUsage;
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedCompletion, ScriptedProvider, client_from,
    text_response, with_cost, with_usage,
};
use pebble_coding_agent::{
    CodingAgent, CodingAgentOptions, CompactionOptions, CompactionOutcome, CompactionReason,
    ShutdownReason,
};

/// An input long enough to fill the 100-token window of `test/small`.
fn large_input() -> String {
    "x".repeat(400)
}

/// The summary call's answer, with the usage and cost the provider reports
/// for it.
fn priced_summary(usage: TokenCounts, usd_micros: u64) -> ScriptedCompletion {
    ScriptedCompletion::response(with_cost(
        with_usage(
            text_response("Here is the summary of the conversation so far."),
            usage,
        ),
        usd_micros,
    ))
}

async fn agent_with(provider: ScriptedProvider, model: &str) -> CodingAgent {
    let (client, _provider) = client_from(provider);
    CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model(model)
        .options(CodingAgentOptions::default().with_compaction_preserve_turns(1))
        .build()
        .await
        .expect("the coding agent builds")
}

#[tokio::test]
async fn a_threshold_compaction_is_on_the_report_with_its_usage_and_cost() {
    let summary_usage = TokenCounts {
        input: 70,
        output: 12,
        ..TokenCounts::default()
    };
    // A response that reports no usage of its own, so the report's totals
    // are the summary call's alone.
    let provider = ScriptedProvider::new(vec![ScriptedCall::response(with_usage(
        text_response("OK"),
        TokenCounts::default(),
    ))])
    .completing(vec![priced_summary(summary_usage, 5)]);
    let mut agent = agent_with(provider, "test/small").await;

    let report = agent.prompt(&large_input()).await;

    assert!(report.result.is_ok(), "{report:?}");
    let [account] = report.compactions.as_slice() else {
        panic!("one compaction is on the report: {report:?}");
    };
    assert_eq!(account.reason, CompactionReason::Threshold);
    assert_eq!(account.original_turn_count, 2);
    assert_eq!(account.preserved_turn_count, 1);
    assert!(account.estimated_tokens_before > 80, "{account:?}");
    assert!(account.summary_token_estimate > 0, "{account:?}");
    assert_eq!(account.tracked_file_count, 0);
    assert!(!account.summary_truncated);
    assert_eq!(account.usage, TokenUsage::from(summary_usage));
    assert_eq!(account.cost_usd_micros, Some(5));
    assert_eq!(
        report.usage, account.usage,
        "the summary call is in the report's totals, which the list breaks down"
    );
    assert_eq!(report.cost_usd_micros, Some(5));
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_prompt_that_compacted_nothing_reports_no_compactions() {
    let provider = ScriptedProvider::new(vec![ScriptedCall::response(text_response("OK"))])
        .completing(vec![priced_summary(TokenCounts::default(), 5)]);
    let mut agent = agent_with(provider, "test/model").await;

    let report = agent.prompt("hi").await;

    assert!(report.result.is_ok(), "{report:?}");
    assert!(report.compactions.is_empty(), "{report:?}");
    assert_eq!(report.cost_usd_micros, None);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn a_manual_compaction_between_prompts_is_on_no_report() {
    let summary_usage = TokenCounts {
        input: 40,
        output: 9,
        ..TokenCounts::default()
    };
    let provider = ScriptedProvider::new(vec![
        ScriptedCall::response(text_response("first")),
        ScriptedCall::response(text_response("second")),
    ])
    .completing(vec![priced_summary(summary_usage, 9)]);
    let mut agent = agent_with(provider, "test/model").await;

    let first = agent.prompt("first task").await;
    let outcome = agent
        .compact(CompactionOptions::new().preserve_turns(1))
        .await
        .expect("the manual compaction succeeds");
    let second = agent.prompt("second task").await;

    assert!(first.result.is_ok(), "{first:?}");
    assert!(first.compactions.is_empty(), "{first:?}");
    let CompactionOutcome::Compacted(result) = outcome else {
        panic!("there was history to compact: {outcome:?}");
    };
    assert_eq!(result.reason(), CompactionReason::Manual);
    assert_eq!(result.usage(), TokenUsage::from(summary_usage));
    assert!(second.result.is_ok(), "{second:?}");
    assert!(
        second.compactions.is_empty(),
        "a compaction between prompts belongs to no prompt: {second:?}"
    );
    assert_eq!(
        second.cost_usd_micros, None,
        "nor is its summary call on the next prompt's bill"
    );
    assert_eq!(second.usage, first.usage);
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}
