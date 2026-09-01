//! Summarizing a conversation that has outgrown the context window.
//!
//! Compaction has two halves. [`check_context_usage`] answers whether the
//! session is close enough to the window to act, using the last usage the
//! provider reported plus a local estimate of everything since — the cheapest
//! answer that is still anchored to a real count. [`compact_context`] then
//! spends one non-streaming call on the session's own model to write a handoff
//! document, and replaces the summarized turns with it.
//!
//! Two rules keep compaction from destroying a conversation. It refuses to
//! replace anything with a summary that came back empty, and it never cuts
//! between a tool call and its result — [`History`] moves the boundary earlier
//! until the cut is safe, and a cut that would preserve everything is no cut at
//! all.

use std::fmt;
use std::fmt::Write as _;

use lithos_llm::client::Client;
use lithos_llm::types::Request;
use tracing::debug;

use crate::char_boundary::floor_char_boundary;
use crate::error::{CompactionError, Result};
use crate::event::Emitter;
use crate::file_tracker::FileTracker;
use crate::history::History;
use crate::profile::ModelFacts;
use crate::tool::result_text;
use crate::types::{AgentEvent, Message};

/// The characters-per-token ratio the local estimate assumes.
const APPROX_CHARS_PER_TOKEN: usize = 4;

/// The output budget for the summary text itself.
const SUMMARY_MAX_TOKENS: u32 = 4_096;

/// The extra output budget a model that reasons unasked is given.
///
/// A provider's output limit covers reasoning and visible text together, so a
/// reasoning model handed only [`SUMMARY_MAX_TOKENS`] can spend the whole
/// budget thinking and return a successful response with nothing in it.
const REASONING_HEADROOM_TOKENS: u32 = 16_384;

/// How much of a rendered turn the summarization transcript keeps.
const TRANSCRIPT_FIELD_BYTES: usize = 500;

/// How the size of the active conversation was arrived at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ContextEstimateMethod {
    /// The last usage the provider reported, plus a local estimate of the
    /// turns recorded since.
    ApiUsagePlusLocalDelta,
    /// A local estimate of the whole prompt, used before any turn carried
    /// usage.
    LocalEstimate,
}

impl ContextEstimateMethod {
    /// The stable identifier the warning event carries.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ApiUsagePlusLocalDelta => "api_usage_plus_local_delta",
            Self::LocalEstimate => "local_estimate",
        }
    }
}

impl fmt::Display for ContextEstimateMethod {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How large the active conversation is, and how that was worked out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextEstimate {
    /// The estimated prompt size, in tokens.
    pub tokens: usize,
    /// How the number was arrived at.
    pub method: ContextEstimateMethod,
}

/// What one compaction run needs to know about the session it is compacting.
///
/// An argument bundle rather than something an application reads back, so it is
/// plainly constructible and a member added later is a breaking change.
#[derive(Debug, Clone, Copy)]
pub struct CompactionRequest<'a> {
    /// The model selector the session runs on. Compaction summarizes with the
    /// same model, so the summary reads the conversation the way its author
    /// did.
    pub model:          &'a str,
    /// The budgets that model works within.
    pub facts:          ModelFacts,
    /// How many recent turns to leave untouched.
    ///
    /// One turn is always left, whatever this says. Compaction runs at a turn
    /// boundary where the newest turn may hold tool calls whose results have
    /// not been recorded yet, and summarizing that turn away would leave the
    /// results that follow it answering calls no provider can see.
    pub preserve_turns: usize,
    /// The estimate that triggered this run, reported on the started event.
    pub estimate:       ContextEstimate,
}

/// Whether the session has crossed the compaction threshold, and by how much.
///
/// Emits a `context_window` [`AgentEvent::Warning`] when it has, so an
/// application sees the pressure whether or not compaction is enabled. Answers
/// `None` for a model whose window is unknown, because a threshold on an
/// unknown window means nothing.
#[must_use]
pub fn check_context_usage(
    system_prompt: &str,
    history: &History,
    context_window_tokens: usize,
    threshold_percent: usize,
    emitter: &Emitter,
    session_id: &str,
) -> Option<ContextEstimate> {
    if context_window_tokens == 0 {
        return None;
    }

    let estimate = estimate_active_context_usage(system_prompt, history);
    let threshold = context_window_tokens.saturating_mul(threshold_percent) / 100;
    if estimate.tokens <= threshold {
        return None;
    }

    let usage_percent = estimate.tokens.saturating_mul(100) / context_window_tokens;
    emitter.emit(session_id.to_owned(), AgentEvent::Warning {
        kind:    "context_window".to_owned(),
        message: format!("Context window usage: {usage_percent}%"),
        details: serde_json::json!({
            "estimated_tokens": estimate.tokens,
            "context_window_size": context_window_tokens,
            "usage_percent": usage_percent,
            "estimate_method": estimate.method.as_str(),
        }),
    });
    Some(estimate)
}

/// Replaces the older turns of `history` with a summary of them.
///
/// Does nothing when preserving the recent turns safely would preserve all of
/// them: no call is made and no event is emitted, so an application never sees
/// a compaction start that cannot finish. The newest turn is always among the
/// preserved, however small
/// [`preserve_turns`](CompactionRequest::preserve_turns) is.
///
/// The summary text is bounded to the visible budget after the call, because a
/// provider enforces one combined ceiling for reasoning and output and cannot
/// be asked to bound the visible half on its own.
pub async fn compact_context(
    history: &mut History,
    client: &Client,
    file_tracker: &FileTracker,
    request: CompactionRequest<'_>,
    emitter: &Emitter,
    session_id: &str,
) -> Result<()> {
    let original_turn_count = history.len();
    // The newest turn stays whatever the caller asked for: it may be an
    // assistant turn whose tool calls are about to be answered, and its
    // results are pushed after this point.
    let preserve_start = history.compact_preserve_start(request.preserve_turns.max(1));
    if preserve_start == 0 {
        return Ok(());
    }
    let preserved_turn_count = original_turn_count - preserve_start;

    emitter.emit(session_id.to_owned(), AgentEvent::CompactionStarted {
        estimated_tokens:    request.estimate.tokens,
        context_window_size: request.facts.context_window_tokens,
    });

    let max_tokens = summary_max_tokens(
        request.facts.reasons_by_default,
        request.facts.max_output_tokens,
    );
    let summary_request = Request::builder()
        .model(request.model)
        .system(summarization_prompt(
            SUMMARY_MAX_TOKENS.min(max_tokens),
            file_tracker,
        ))
        .user(format!(
            "Here is the conversation to summarize:\n\n{}",
            render_turns_for_summary(&history.turns()[..preserve_start])
        ))
        .max_output_tokens(max_tokens)
        .build()
        .map_err(CompactionError::Request)?;

    let response = client
        .complete(summary_request)
        .await
        .map_err(CompactionError::Llm)?;

    // `compact_from` discards the summarized turns for good, so an empty
    // summary is refused before history is touched. Trimming first stops a
    // whitespace-only response from passing as a summary.
    let response_text = response.text();
    let summary = response_text.trim();
    if summary.is_empty() {
        return Err(CompactionError::EmptySummary {
            summarized_turn_count: preserve_start,
        }
        .into());
    }

    let (summary, summary_truncated) = truncate_summary_text(summary);
    debug!(
        summary_len = summary.len(),
        summary_truncated, max_tokens, "Compaction summary generated"
    );

    let content = format!(
        "A different assistant began this task and produced the following summary. Build on \
         their progress — do not repeat completed steps.\n\n{summary}"
    );
    let summary_token_estimate = local_tokens(content.len());

    history.compact_from(preserve_start, content);

    emitter.emit(session_id.to_owned(), AgentEvent::CompactionCompleted {
        original_turn_count,
        preserved_turn_count,
        summary_token_estimate,
        tracked_file_count: file_tracker.file_count(),
    });

    Ok(())
}

/// The instructions the summarizing call is given.
///
/// The file list is asked for verbatim because it is the part of a session a
/// summary most reliably loses, and the part the next assistant most needs.
fn summarization_prompt(visible_max_tokens: u32, file_tracker: &FileTracker) -> String {
    let file_operations = if file_tracker.is_empty() {
        String::new()
    } else {
        format!(
            "\n## File Operations\nCOPY THIS SECTION VERBATIM into your summary.\n\n{}",
            file_tracker.render()
        )
    };

    format!(
        "You are creating a handoff document for a different coding assistant that will take over \
this task. That assistant will only see your summary and the most recent messages — nothing else \
from the conversation so far.\n\n\
Write a summary using EXACTLY these sections:\n\n\
## Goal\nWhat the user asked for and any constraints or preferences stated.\n\n\
## Progress\nWhat was accomplished, with file paths and key decisions.\n\n\
## Key Decisions\nImportant choices made and their rationale.\n\n\
## Failed Approaches\nWhat was tried and didn't work, and why.\n\n\
## Open Issues\nBugs, edge cases, or TODOs that remain.\n\n\
## Next Steps\nWhat should happen next to make progress.\n\n\
Keep the entire response under {visible_max_tokens} tokens.\n\n\
Be thorough and specific — the assistant taking over has no prior context. Include file paths, \
function names, error messages, and exact values. Omit pleasantries and conversational filler.\
{file_operations}"
    )
}

/// The combined reasoning and output budget for the summarization request.
///
/// Compaction runs on the session's own model, so a reasoning session
/// summarizes with reasoning on and the budget has to cover the thinking as
/// well as the summary. Any budget is capped at what the model will produce.
fn summary_max_tokens(reasons_by_default: bool, max_output_tokens: Option<u64>) -> u32 {
    let budget = if reasons_by_default {
        SUMMARY_MAX_TOKENS.saturating_add(REASONING_HEADROOM_TOKENS)
    } else {
        SUMMARY_MAX_TOKENS
    };

    max_output_tokens.map_or(budget, |limit| {
        budget.min(u32::try_from(limit).unwrap_or(u32::MAX))
    })
}

/// Bounds the retained summary with the same local heuristic the context
/// estimate uses.
fn truncate_summary_text(summary: &str) -> (&str, bool) {
    let max_bytes = summary_max_approx_bytes();
    if summary.len() <= max_bytes {
        return (summary, false);
    }

    (&summary[..floor_char_boundary(summary, max_bytes)], true)
}

/// The byte budget [`SUMMARY_MAX_TOKENS`] works out to locally.
fn summary_max_approx_bytes() -> usize {
    usize::try_from(SUMMARY_MAX_TOKENS)
        .unwrap_or(usize::MAX)
        .saturating_mul(APPROX_CHARS_PER_TOKEN)
}

/// How large the prompt a session would send now is.
///
/// Anchored to the most recent usage a provider reported, which counts what
/// pebble cannot see — the provider's own framing, its tool serialization, its
/// cached prefix — plus a local estimate of the turns recorded since. Before
/// any assistant turn carries usage there is nothing to anchor to, so the whole
/// prompt is estimated locally.
#[must_use]
pub fn estimate_active_context_usage(system_prompt: &str, history: &History) -> ContextEstimate {
    let turns = history.turns();

    if let Some((index, reported)) = latest_reported_usage(turns) {
        return ContextEstimate {
            tokens: reported.saturating_add(local_tokens(turn_chars(&turns[index + 1..]))),
            method: ContextEstimateMethod::ApiUsagePlusLocalDelta,
        };
    }

    ContextEstimate {
        tokens: local_tokens(system_prompt.len().saturating_add(turn_chars(turns))),
        method: ContextEstimateMethod::LocalEstimate,
    }
}

/// The last assistant turn that reported any usage, and what it reported.
fn latest_reported_usage(turns: &[Message]) -> Option<(usize, usize)> {
    turns.iter().enumerate().rev().find_map(|(index, turn)| {
        let Message::Assistant { usage, .. } = turn else {
            return None;
        };
        let total = usage.total();
        (total > 0).then(|| (index, usize::try_from(total).unwrap_or(usize::MAX)))
    })
}

/// A character count as a local token estimate.
const fn local_tokens(chars: usize) -> usize {
    chars / APPROX_CHARS_PER_TOKEN
}

/// The characters a run of turns contributes to the prompt.
fn turn_chars(turns: &[Message]) -> usize {
    turns.iter().fold(0, |total, turn| {
        total.saturating_add(single_turn_chars(turn))
    })
}

/// The characters one turn contributes to the prompt.
fn single_turn_chars(turn: &Message) -> usize {
    match turn {
        Message::User { content, .. }
        | Message::System { content, .. }
        | Message::Steering { content, .. } => content.len(),
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let reasoning = turn.reasoning_text().map_or(0, str::len);
            let calls: usize = tool_calls
                .iter()
                .map(|call| call.name.len() + call.arguments.to_string().len())
                .sum();
            content.len() + reasoning + calls
        }
        Message::ToolResults { results, .. } => {
            results.iter().map(|result| result_text(result).len()).sum()
        }
    }
}

/// Renders turns as the transcript the summarizing call reads.
///
/// Long tool arguments and tool output are cut, because the summary is about
/// what happened rather than about every byte that moved.
#[must_use]
pub fn render_turns_for_summary(turns: &[Message]) -> String {
    let mut out = String::new();
    for turn in turns {
        match turn {
            Message::User { content, .. } => {
                let _ = writeln!(out, "User: {content}");
            }
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                if !content.is_empty() {
                    let _ = writeln!(out, "Assistant: {content}");
                }
                for call in tool_calls {
                    let _ = writeln!(
                        out,
                        "[Tool call: {}] {}",
                        call.name,
                        clip(&call.arguments.to_string())
                    );
                }
            }
            Message::ToolResults { results, .. } => {
                for result in results {
                    let _ = writeln!(
                        out,
                        "[Tool result: {}] {}",
                        result.tool_call_id,
                        clip(&result_text(result))
                    );
                }
            }
            Message::System { content, .. } => {
                let _ = writeln!(out, "System: {content}");
            }
            Message::Steering { content, .. } => {
                let _ = writeln!(out, "Steering: {content}");
            }
        }
    }
    out
}

/// Cuts one transcript field to its budget, marking where it was cut.
fn clip(text: &str) -> String {
    if text.len() <= TRANSCRIPT_FIELD_BYTES {
        return text.to_owned();
    }
    format!(
        "{}...",
        &text[..floor_char_boundary(text, TRANSCRIPT_FIELD_BYTES)]
    )
}

#[cfg(test)]
mod tests {
    use std::result::Result as StdResult;
    use std::sync::{Arc, Mutex, PoisonError};
    use std::time::SystemTime;

    use async_trait::async_trait;
    use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
    use lithos_llm::catalog::{AdapterId, Catalog};
    use lithos_llm::types::{
        ContentPart, Error as LlmError, ErrorKind as LlmErrorKind, FinishReason,
        Message as LlmMessage, Response, ResponseStream, ToolCall, ToolResult,
    };
    use lithos_llm::{Client, ClientBuild};
    use serde_json::json;

    use super::*;
    use crate::error::Error;
    use crate::event::{EventOptions, EventPump};
    use crate::record::StoredMessage;
    use crate::types::{SessionEvent, TokenUsage};

    const TEST_CATALOG: &str = r#"
schema_version = 1

[providers.test]
display_name = "Test"
adapter = "test-adapter"
codec = "test-codec"
base_url = "http://127.0.0.1"
default_model = "model"

[providers.test.auth]
type = "none"

[providers.test.models.model]
display_name = "Test model"
api_model = "model"
capabilities = { text = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

[providers.test.models.on-request]
display_name = "Reasons when asked"
api_model = "on-request"
capabilities = { text = true, reasoning = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

[providers.test.models.always]
display_name = "Reasons by default"
api_model = "always"
capabilities = { text = true, reasoning = true, reasoning_effort_levels = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }
"#;

    /// What the fake provider answers a summarization call with.
    #[derive(Clone)]
    enum Script {
        Summary(String),
        Fails,
    }

    struct ScriptedAdapter {
        id:       AdapterId,
        script:   Script,
        requests: Arc<Mutex<Vec<Request>>>,
    }

    impl ScriptedAdapter {
        fn new(script: Script) -> Self {
            Self {
                id: AdapterId::new("test-adapter"),
                script,
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl ProviderAdapter for ScriptedAdapter {
        fn id(&self) -> &AdapterId {
            &self.id
        }

        async fn complete(&self, call: &ResolvedCall) -> StdResult<Response, LlmError> {
            self.requests
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(call.request().clone());

            match &self.script {
                Script::Summary(text) => {
                    let mut response = Response::new(
                        call.route().provider().id().clone(),
                        call.route().model().id().clone(),
                        vec![ContentPart::Text { text: text.clone() }],
                    );
                    response.finish_reason = FinishReason::Stop;
                    Ok(response)
                }
                Script::Fails => Err(LlmError::new(LlmErrorKind::Server, "provider is down")),
            }
        }

        async fn stream(&self, _call: &ResolvedCall) -> StdResult<ResponseStream, LlmError> {
            Err(LlmError::new(
                LlmErrorKind::Middleware,
                "compaction never streams",
            ))
        }
    }

    fn client(script: Script) -> (Client, Arc<Mutex<Vec<Request>>>) {
        let adapter = ScriptedAdapter::new(script);
        let requests = Arc::clone(&adapter.requests);
        let catalog = Catalog::builder()
            .overlay_toml(TEST_CATALOG)
            .expect("the catalog layer parses")
            .build()
            .expect("the catalog validates");
        let ClientBuild { client, .. } = Client::builder()
            .catalog(catalog)
            .adapter("test", adapter)
            .build()
            .expect("the client builds");
        (client, requests)
    }

    fn now() -> SystemTime {
        SystemTime::now()
    }

    fn user(content: &str) -> Message {
        Message::User {
            content:   content.to_owned(),
            timestamp: now(),
        }
    }

    fn assistant(content: &str, usage: TokenUsage) -> Message {
        Message::Assistant {
            content: content.to_owned(),
            tool_calls: Vec::new(),
            provider_parts: Vec::new(),
            usage,
            response_id: "resp_1".to_owned(),
            timestamp: now(),
        }
    }

    fn tool_results(call_id: &str, text: &str) -> Message {
        Message::ToolResults {
            results:   vec![ToolResult {
                tool_call_id: call_id.to_owned(),
                name:         None,
                content:      vec![ContentPart::Text {
                    text: text.to_owned(),
                }],
                is_error:     false,
            }],
            timestamp: now(),
        }
    }

    fn history_from(turns: Vec<Message>) -> History {
        let mut history = History::default();
        for turn in turns {
            history.push(turn);
        }
        history
    }

    /// An emitter whose pump is drained on demand, so a test sees exactly the
    /// events one call produced.
    struct Events {
        emitter: Emitter,
        pump:    Option<EventPump>,
    }

    impl Events {
        fn new() -> Self {
            let (emitter, pump) = EventPump::new(EventOptions::default());
            Self {
                emitter,
                pump: Some(pump),
            }
        }

        /// Drops the emitter, runs the pump to exhaustion, and answers what was
        /// published.
        async fn drain(mut self) -> Vec<AgentEvent> {
            let mut receiver = self.emitter.subscribe();
            drop(self.emitter);
            self.pump
                .take()
                .expect("the pump is taken once")
                .run()
                .await
                .expect("no sink can refuse");
            let mut events = Vec::new();
            while let Ok(SessionEvent { event, .. }) = receiver.try_recv() {
                events.push(event);
            }
            events
        }
    }

    fn facts() -> ModelFacts {
        ModelFacts {
            context_window_tokens: 200_000,
            max_output_tokens:     Some(32_000),
            reasons_by_default:    false,
        }
    }

    fn compaction_request(estimate_tokens: usize) -> CompactionRequest<'static> {
        CompactionRequest {
            model:          "test/model",
            facts:          facts(),
            preserve_turns: 1,
            estimate:       ContextEstimate {
                tokens: estimate_tokens,
                method: ContextEstimateMethod::LocalEstimate,
            },
        }
    }

    // --- budget ---

    #[test]
    fn a_model_that_does_not_reason_gets_the_summary_allowance() {
        assert_eq!(summary_max_tokens(false, None), SUMMARY_MAX_TOKENS);
    }

    #[test]
    fn a_model_that_reasons_gets_headroom_on_top() {
        assert_eq!(
            summary_max_tokens(true, None),
            SUMMARY_MAX_TOKENS + REASONING_HEADROOM_TOKENS
        );
    }

    /// The facts the test catalog records about one of its models.
    fn catalog_facts(model: &str) -> ModelFacts {
        let catalog = Catalog::builder()
            .overlay_toml(TEST_CATALOG)
            .expect("the catalog layer parses")
            .build()
            .expect("the catalog validates");
        ModelFacts::from_catalog_model(catalog.model("test", model).expect("a known model"))
    }

    #[test]
    fn a_model_that_reasons_only_when_asked_gets_no_headroom() {
        // The case a `reasoning` capability alone gets wrong: the model can
        // reason, but a summarization call never asks it to, so the budget is
        // the summary's own.
        let facts = catalog_facts("on-request");

        assert!(!facts.reasons_by_default);
        assert_eq!(
            summary_max_tokens(facts.reasons_by_default, facts.max_output_tokens),
            SUMMARY_MAX_TOKENS
        );
    }

    #[test]
    fn a_model_that_always_reasons_gets_the_headroom_from_its_catalog_row() {
        let facts = catalog_facts("always");

        assert!(facts.reasons_by_default);
        assert_eq!(
            summary_max_tokens(facts.reasons_by_default, facts.max_output_tokens),
            SUMMARY_MAX_TOKENS + REASONING_HEADROOM_TOKENS
        );
    }

    #[test]
    fn a_budget_never_exceeds_what_the_model_will_produce() {
        assert_eq!(summary_max_tokens(true, Some(8_192)), 8_192);
        assert_eq!(summary_max_tokens(false, Some(2_048)), 2_048);
        assert_eq!(
            summary_max_tokens(false, Some(u64::from(u32::MAX) + 1)),
            SUMMARY_MAX_TOKENS
        );
    }

    // --- transcript ---

    #[test]
    fn a_transcript_labels_every_kind_of_turn() {
        let turns = vec![
            user("Hello"),
            Message::Assistant {
                content:        "Let me check".to_owned(),
                tool_calls:     vec![ToolCall::function(
                    "c1",
                    "read_file",
                    json!({ "path": "foo.rs" }),
                )],
                provider_parts: Vec::new(),
                usage:          TokenUsage::default(),
                response_id:    "resp_1".to_owned(),
                timestamp:      now(),
            },
            tool_results("c1", "file contents here"),
            Message::System {
                content:   "injected".to_owned(),
                timestamp: now(),
            },
            Message::Steering {
                content:   "focus".to_owned(),
                timestamp: now(),
            },
        ];

        let rendered = render_turns_for_summary(&turns);

        assert!(rendered.contains("User: Hello"));
        assert!(rendered.contains("Assistant: Let me check"));
        assert!(rendered.contains("[Tool call: read_file]"));
        assert!(rendered.contains("[Tool result: c1] file contents here"));
        assert!(rendered.contains("System: injected"));
        assert!(rendered.contains("Steering: focus"));
    }

    #[test]
    fn a_transcript_clips_long_tool_output() {
        let rendered = render_turns_for_summary(&[tool_results("c1", &"x".repeat(1_000))]);

        assert!(rendered.len() < 1_000);
        assert!(rendered.contains("..."));
    }

    #[test]
    fn clipping_keeps_whole_characters() {
        let clipped = clip(&"€".repeat(500));

        assert!(clipped.ends_with("..."));
        assert!(clipped.len() <= TRANSCRIPT_FIELD_BYTES + "...".len());
    }

    // --- estimate ---

    #[test]
    fn without_reported_usage_the_whole_prompt_is_estimated_locally() {
        let history = history_from(vec![user("Hello world")]);

        // "test" is four characters and "Hello world" is eleven: 15 / 4 = 3.
        let estimate = estimate_active_context_usage("test", &history);

        assert_eq!(estimate.tokens, 3);
        assert_eq!(estimate.method, ContextEstimateMethod::LocalEstimate);
    }

    #[test]
    fn a_local_estimate_counts_tool_calls_and_results() {
        let history = history_from(vec![
            user("Hello world"),
            Message::Assistant {
                content:        "No usage available".to_owned(),
                tool_calls:     vec![ToolCall::function(
                    "call_1",
                    "read_file",
                    json!({ "path": "foo.rs" }),
                )],
                provider_parts: Vec::new(),
                usage:          TokenUsage::default(),
                response_id:    "resp_1".to_owned(),
                timestamp:      now(),
            },
            tool_results("call_1", "1234"),
        ]);

        let estimate = estimate_active_context_usage("test", &history);

        assert_eq!(estimate.method, ContextEstimateMethod::LocalEstimate);
        // 4 prompt + 11 user + 18 content + 9 name + 16 arguments + 4 result.
        assert_eq!(estimate.tokens, 62 / 4);
    }

    #[test]
    fn reported_usage_anchors_the_estimate_and_later_turns_are_added() {
        let history = history_from(vec![
            user(&"ignored before the baseline".repeat(100)),
            assistant("baseline response", TokenUsage {
                input: 50,
                ..TokenUsage::default()
            }),
            tool_results("call_1", "1234"),
            user(&"u".repeat(16)),
            Message::Steering {
                content:   "s".repeat(8),
                timestamp: now(),
            },
        ]);

        let estimate = estimate_active_context_usage("ignored system prompt", &history);

        // 50 reported, then (4 + 16 + 8) / 4 = 7 for the turns since.
        assert_eq!(estimate.tokens, 57);
        assert_eq!(
            estimate.method,
            ContextEstimateMethod::ApiUsagePlusLocalDelta
        );
    }

    #[test]
    fn every_reported_bucket_counts_toward_the_baseline() {
        let history = history_from(vec![assistant("short", TokenUsage {
            input:       10,
            output:      20,
            reasoning:   30,
            cache_read:  40,
            cache_write: 50,
        })]);

        let estimate = estimate_active_context_usage("", &history);

        assert_eq!(estimate.tokens, 150);
        assert_eq!(
            estimate.method,
            ContextEstimateMethod::ApiUsagePlusLocalDelta
        );
    }

    #[test]
    fn the_latest_reported_usage_wins() {
        let history = history_from(vec![
            assistant("older response", TokenUsage {
                input: 1_000,
                ..TokenUsage::default()
            }),
            user(&"ignored before the latest baseline".repeat(100)),
            assistant("latest response", TokenUsage {
                input: 20,
                ..TokenUsage::default()
            }),
            user(&"u".repeat(8)),
        ]);

        let estimate = estimate_active_context_usage("", &history);

        assert_eq!(estimate.tokens, 22);
        assert_eq!(
            estimate.method,
            ContextEstimateMethod::ApiUsagePlusLocalDelta
        );
    }

    // --- threshold ---

    #[tokio::test]
    async fn usage_below_the_threshold_reports_nothing() {
        let events = Events::new();

        let estimate = check_context_usage(
            "short",
            &History::default(),
            200_000,
            80,
            &events.emitter,
            "s",
        );

        assert!(estimate.is_none());
        assert!(events.drain().await.is_empty());
    }

    #[tokio::test]
    async fn usage_above_the_threshold_warns_and_reports_the_estimate() {
        let events = Events::new();
        let history = history_from(vec![user(&"x".repeat(1_000))]);

        let estimate = check_context_usage("prompt", &history, 100, 80, &events.emitter, "sess");

        assert!(estimate.is_some());
        let published = events.drain().await;
        assert!(
            published.iter().any(|event| matches!(
                event,
                AgentEvent::Warning { kind, details, .. }
                    if kind == "context_window" && details["estimate_method"] == "local_estimate"
            )),
            "{published:?}"
        );
    }

    #[tokio::test]
    async fn a_model_with_no_known_window_is_never_over_the_threshold() {
        let events = Events::new();
        let history = history_from(vec![user(&"x".repeat(1_000))]);

        let estimate = check_context_usage("prompt", &history, 0, 80, &events.emitter, "sess");

        assert!(estimate.is_none());
        assert!(events.drain().await.is_empty());
    }

    // --- compaction ---

    struct Compacted {
        result:  Result<()>,
        history: History,
        before:  Vec<StoredMessage>,
        events:  Vec<AgentEvent>,
    }

    /// Compacts a four-turn history against a provider that answers `script`.
    async fn compact_with(script: Script) -> Compacted {
        let mut history = history_from(
            (0..4)
                .map(|index| user(&format!("message {index}")))
                .collect(),
        );
        let before = history.to_stored_messages();
        let (client, _) = client(script);
        let events = Events::new();

        let result = compact_context(
            &mut history,
            &client,
            &FileTracker::default(),
            compaction_request(1_000),
            &events.emitter,
            "sess",
        )
        .await;

        Compacted {
            result,
            history,
            before,
            events: events.drain().await,
        }
    }

    fn summary_turn(history: &History) -> &str {
        history
            .turns()
            .iter()
            .find_map(|turn| match turn {
                Message::System { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .expect("a compacted history carries a summary turn")
    }

    #[tokio::test]
    async fn a_summary_replaces_the_older_turns() {
        let compacted = compact_with(Script::Summary("Brief handoff.".to_owned())).await;

        compacted.result.expect("a summary compacts");
        let summary = summary_turn(&compacted.history);
        assert!(summary.contains("A different assistant began this task"));
        assert!(summary.contains("Brief handoff."));
        assert!(
            compacted
                .events
                .iter()
                .any(|event| matches!(event, AgentEvent::CompactionCompleted { .. }))
        );
    }

    #[tokio::test]
    async fn a_blank_summary_leaves_history_alone() {
        for blank in ["", "   \n\t  \n "] {
            let compacted = compact_with(Script::Summary(blank.to_owned())).await;

            let error = compacted
                .result
                .expect_err("a blank summary must not report success");
            assert!(
                matches!(
                    error,
                    Error::Compaction(CompactionError::EmptySummary {
                        summarized_turn_count: 3,
                    })
                ),
                "unexpected error: {error}"
            );
            assert_eq!(compacted.history.to_stored_messages(), compacted.before);
            assert!(
                compacted
                    .events
                    .iter()
                    .any(|event| matches!(event, AgentEvent::CompactionStarted { .. })),
                "the attempt is still recorded"
            );
            assert!(
                !compacted
                    .events
                    .iter()
                    .any(|event| matches!(event, AgentEvent::CompactionCompleted { .. })),
                "a refused summary must not complete"
            );
        }
    }

    #[tokio::test]
    async fn a_failed_summarization_call_leaves_history_alone() {
        let compacted = compact_with(Script::Fails).await;

        let error = compacted.result.expect_err("a failed call fails");
        assert!(
            matches!(error, Error::Compaction(CompactionError::Llm(_))),
            "unexpected error: {error}"
        );
        assert_eq!(compacted.history.to_stored_messages(), compacted.before);
    }

    #[tokio::test]
    async fn an_overlong_summary_is_bounded_before_it_is_kept() {
        let max_bytes = summary_max_approx_bytes();
        let overlong = format!("{}END", "€".repeat(max_bytes / 3 + 1));

        let compacted = compact_with(Script::Summary(overlong)).await;

        compacted
            .result
            .expect("an overlong summary compacts after truncation");
        let (_, retained) = summary_turn(&compacted.history)
            .split_once("\n\n")
            .expect("the summary turn separates its header from the generated text");
        assert!(retained.len() <= max_bytes);
        assert!(!retained.contains("END"));
    }

    #[tokio::test]
    async fn compaction_keeps_the_turn_whose_tool_calls_are_still_open() {
        // Compaction runs between committing an assistant turn and executing
        // the calls it made, so preserving nothing would summarize away the
        // calls the results about to be pushed answer.
        let mut history = history_from(vec![user("first"), user("second"), Message::Assistant {
            content:        "working on it".to_owned(),
            tool_calls:     vec![ToolCall::function("call_1", "shell", json!({}))],
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp_1".to_owned(),
            timestamp:      now(),
        }]);
        let (client, _) = client(Script::Summary("Brief handoff.".to_owned()));
        let events = Events::new();

        compact_context(
            &mut history,
            &client,
            &FileTracker::default(),
            CompactionRequest {
                preserve_turns: 0,
                ..compaction_request(1_000)
            },
            &events.emitter,
            "sess",
        )
        .await
        .expect("the summary compacts");

        history.push(tool_results("call_1", "done"));
        assert!(
            history.turns().iter().any(|turn| matches!(
                turn,
                Message::Assistant { tool_calls, .. }
                    if tool_calls.iter().any(|call| call.id == "call_1")
            )),
            "the open call survives its own compaction: {:?}",
            history.turns()
        );
        let _ = events.drain().await;
    }

    #[tokio::test]
    async fn compaction_with_nothing_to_summarize_makes_no_call() {
        let mut history = history_from(vec![user("only one turn")]);
        let (client, requests) = client(Script::Summary("unused".to_owned()));
        let events = Events::new();

        compact_context(
            &mut history,
            &client,
            &FileTracker::default(),
            CompactionRequest {
                preserve_turns: 4,
                ..compaction_request(1_000)
            },
            &events.emitter,
            "sess",
        )
        .await
        .expect("a no-op compaction succeeds");

        assert_eq!(history.len(), 1);
        assert!(
            requests
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        );
        assert!(events.drain().await.is_empty());
    }

    #[tokio::test]
    async fn the_summarization_call_carries_the_prompt_the_transcript_and_the_budget() {
        let mut history = history_from(vec![
            user("first"),
            user("second"),
            user("third"),
            user("fourth"),
        ]);
        let mut file_tracker = FileTracker::default();
        file_tracker.record_edit("src/lib.rs");
        let (client, requests) = client(Script::Summary("done".to_owned()));
        let events = Events::new();

        compact_context(
            &mut history,
            &client,
            &file_tracker,
            compaction_request(1_000),
            &events.emitter,
            "sess",
        )
        .await
        .expect("the summary compacts");

        let sent = requests.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(sent.len(), 1);
        let request = &sent[0];
        assert_eq!(request.model(), "test/model");
        assert_eq!(request.max_output_tokens(), Some(SUMMARY_MAX_TOKENS));
        assert!(request.tools().is_empty());

        let text: String = request
            .messages()
            .iter()
            .flat_map(LlmMessage::content)
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("handoff document"));
        assert!(text.contains("COPY THIS SECTION VERBATIM"));
        assert!(text.contains("- src/lib.rs (edited)"));
        assert!(text.contains("Keep the entire response under 4096 tokens"));
        assert!(text.contains("User: first"));
        // The preserved turn is not part of the transcript.
        assert!(!text.contains("User: fourth"));
    }

    #[tokio::test]
    async fn a_reasoning_model_asks_for_headroom() {
        let mut history = history_from(
            (0..4)
                .map(|index| user(&format!("message {index}")))
                .collect(),
        );
        let (client, requests) = client(Script::Summary("done".to_owned()));
        let events = Events::new();

        compact_context(
            &mut history,
            &client,
            &FileTracker::default(),
            CompactionRequest {
                facts: ModelFacts {
                    reasons_by_default: true,
                    ..facts()
                },
                ..compaction_request(1_000)
            },
            &events.emitter,
            "sess",
        )
        .await
        .expect("the summary compacts");

        let sent = requests.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(
            sent[0].max_output_tokens(),
            Some(SUMMARY_MAX_TOKENS + REASONING_HEADROOM_TOKENS)
        );
    }

    #[tokio::test]
    async fn an_unbuildable_request_is_reported_rather_than_sent() {
        let mut history = history_from(
            (0..4)
                .map(|index| user(&format!("message {index}")))
                .collect(),
        );
        let (client, requests) = client(Script::Summary("unused".to_owned()));
        let events = Events::new();

        let error = compact_context(
            &mut history,
            &client,
            &FileTracker::default(),
            CompactionRequest {
                model: "   ",
                ..compaction_request(1_000)
            },
            &events.emitter,
            "sess",
        )
        .await
        .expect_err("an empty model selector cannot build a request");

        assert!(
            matches!(error, Error::Compaction(CompactionError::Request(_))),
            "unexpected error: {error}"
        );
        assert!(
            requests
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        );
    }

    #[test]
    fn an_estimate_method_names_itself_for_the_event_stream() {
        assert_eq!(
            ContextEstimateMethod::LocalEstimate.to_string(),
            "local_estimate"
        );
        assert_eq!(
            ContextEstimateMethod::ApiUsagePlusLocalDelta.as_str(),
            "api_usage_plus_local_delta"
        );
    }
}
