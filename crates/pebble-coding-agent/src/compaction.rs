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
use std::io::{Error as IoError, Result as IoResult, Write};
use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::client::Client;
use lithos_llm::types::Request;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::char_boundary::floor_char_boundary;
use crate::error::{CompactionError, Error, ErrorData, InterruptReason, Result};
use crate::event::Emitter;
use crate::file_tracker::FileTracker;
use crate::history::{APPROX_CHARS_PER_TOKEN, History};
use crate::profile::ModelFacts;
use crate::tool::result_text;
use crate::types::{CodingEvent, Message, TokenUsage};

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

/// Why a conversation compaction ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CompactionReason {
    /// The context threshold was crossed.
    #[default]
    Threshold,
    /// The application requested compaction.
    Manual,
    /// A provider rejected the context as too large.
    Overflow,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's skip_serializing_if callback receives a reference"
)]
pub(crate) const fn is_threshold_reason(reason: &CompactionReason) -> bool {
    matches!(reason, CompactionReason::Threshold)
}

/// Options for a manual compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionOptions {
    instructions:   Option<String>,
    preserve_turns: Option<usize>,
}

impl CompactionOptions {
    /// Creates options with the session's normal compaction policy.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            instructions:   None,
            preserve_turns: None,
        }
    }

    /// Adds instructions for this summary only.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Overrides how many recent turns remain verbatim.
    #[must_use]
    pub const fn preserve_turns(mut self, preserve_turns: usize) -> Self {
        self.preserve_turns = Some(preserve_turns);
        self
    }

    pub(crate) fn instructions_ref(&self) -> Option<&str> {
        self.instructions.as_deref()
    }

    pub(crate) const fn preserve_turns_value(&self) -> Option<usize> {
        self.preserve_turns
    }
}

/// What one completed compaction recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionResult {
    summary:                 String,
    reason:                  CompactionReason,
    original_turn_count:     usize,
    preserved_turn_count:    usize,
    estimated_tokens_before: usize,
    summary_token_estimate:  usize,
    tracked_file_count:      usize,
    summary_truncated:       bool,
    usage:                   TokenUsage,
    cost_usd_micros:         Option<u64>,
}

impl CompactionResult {
    /// The model-visible handoff summary.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Why the compaction ran.
    #[must_use]
    pub const fn reason(&self) -> CompactionReason {
        self.reason
    }

    /// Turns present before compaction.
    #[must_use]
    pub const fn original_turn_count(&self) -> usize {
        self.original_turn_count
    }

    /// Turns preserved verbatim.
    #[must_use]
    pub const fn preserved_turn_count(&self) -> usize {
        self.preserved_turn_count
    }

    /// Estimated context tokens before compaction.
    #[must_use]
    pub const fn estimated_tokens_before(&self) -> usize {
        self.estimated_tokens_before
    }

    /// Estimated tokens in the generated summary.
    #[must_use]
    pub const fn summary_token_estimate(&self) -> usize {
        self.summary_token_estimate
    }

    /// Files represented in the compaction prompt.
    #[must_use]
    pub const fn tracked_file_count(&self) -> usize {
        self.tracked_file_count
    }

    /// Whether Pebble truncated the generated summary to its visible budget.
    #[must_use]
    pub const fn summary_was_truncated(&self) -> bool {
        self.summary_truncated
    }

    /// Provider-reported token usage for the summarization call.
    #[must_use]
    pub const fn usage(&self) -> TokenUsage {
        self.usage
    }

    /// Provider-reported or catalog-derived cost in USD micros.
    #[must_use]
    pub const fn cost_usd_micros(&self) -> Option<u64> {
        self.cost_usd_micros
    }

    #[cfg(test)]
    pub(crate) fn for_history(summary: String) -> Self {
        Self {
            summary,
            reason: CompactionReason::Threshold,
            original_turn_count: 0,
            preserved_turn_count: 0,
            estimated_tokens_before: 0,
            summary_token_estimate: 0,
            tracked_file_count: 0,
            summary_truncated: false,
            usage: TokenUsage::default(),
            cost_usd_micros: None,
        }
    }
}

/// The result of asking Pebble to compact now.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CompactionOutcome {
    /// There was not enough safely compactable history.
    Unchanged,
    /// Older history was replaced with this summary.
    Compacted(CompactionResult),
}

/// Shared cancellation for the one compaction a session can run at a time.
#[derive(Clone, Debug, Default)]
pub(crate) struct CompactionControl {
    inner: Arc<CompactionControlInner>,
}

#[derive(Debug, Default)]
struct CompactionControlInner {
    state: Mutex<CompactionControlState>,
    idle:  Notify,
}

#[derive(Debug, Default)]
struct CompactionControlState {
    next_generation: u64,
    active:          Option<(u64, CancellationToken)>,
}

impl CompactionControl {
    pub(crate) fn begin(&self, parent: &CancellationToken) -> CompactionRun {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        debug_assert!(state.active.is_none(), "compactions must not overlap");
        state.next_generation = state.next_generation.saturating_add(1);
        let generation = state.next_generation;
        let cancel = parent.child_token();
        state.active = Some((generation, cancel.clone()));
        CompactionRun {
            control: self.clone(),
            generation,
            cancel,
        }
    }

    pub(crate) fn cancel(&self) -> bool {
        let state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some((_, cancel)) = &state.active else {
            return false;
        };
        cancel.cancel();
        true
    }

    pub(crate) fn is_active(&self) -> bool {
        self.inner
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .active
            .is_some()
    }

    pub(crate) async fn wait_for_idle(&self) {
        loop {
            let notified = self.inner.idle.notified();
            if !self.is_active() {
                return;
            }
            notified.await;
        }
    }
}

pub(crate) struct CompactionRun {
    control:    CompactionControl,
    generation: u64,
    cancel:     CancellationToken,
}

impl CompactionRun {
    pub(crate) const fn token(&self) -> &CancellationToken {
        &self.cancel
    }
}

impl Drop for CompactionRun {
    fn drop(&mut self) {
        self.cancel.cancel();
        let cleared = {
            let mut state = self
                .control
                .inner
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if state
                .active
                .as_ref()
                .is_some_and(|(generation, _)| *generation == self.generation)
            {
                state.active = None;
                true
            } else {
                false
            }
        };
        if cleared {
            self.control.inner.idle.notify_waiters();
        }
    }
}

/// How the size of the active conversation was arrived at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum ContextEstimateMethod {
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
    pub(crate) const fn as_str(self) -> &'static str {
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
pub(crate) struct ContextEstimate {
    /// The estimated prompt size, in tokens.
    pub(crate) tokens: usize,
    /// How the number was arrived at.
    pub(crate) method: ContextEstimateMethod,
}

/// What one compaction run needs to know about the session it is compacting.
///
/// An argument bundle rather than something an application reads back, so it is
/// plainly constructible and a member added later is a breaking change.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CompactionRequest<'a> {
    /// The model selector the session runs on. Compaction summarizes with the
    /// same model, so the summary reads the conversation the way its author
    /// did.
    pub(crate) model:          &'a str,
    /// The budgets that model works within.
    pub(crate) facts:          ModelFacts,
    /// How many recent turns to leave untouched.
    ///
    /// One turn is always left, whatever this says. Compaction runs at a turn
    /// boundary where the newest turn may hold tool calls whose results have
    /// not been recorded yet, and summarizing that turn away would leave the
    /// results that follow it answering calls no provider can see.
    pub(crate) preserve_turns: usize,
    /// The estimate that triggered this run, reported on the started event.
    pub(crate) estimate:       ContextEstimate,
    /// Why this run started.
    pub(crate) reason:         CompactionReason,
    /// Additional instructions for the generated summary.
    pub(crate) instructions:   Option<&'a str>,
    /// Cancels this compaction without cancelling its surrounding prompt.
    pub(crate) cancel:         &'a CancellationToken,
}

/// Whether the session has crossed the compaction threshold, and by how much.
///
/// Emits a `context_window` [`CodingEvent::Warning`] when it has, so an
/// application sees the pressure whether or not compaction is enabled. Answers
/// `None` for a model whose window is unknown, because a threshold on an
/// unknown window means nothing.
#[must_use]
pub(crate) fn check_context_usage(
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
    emitter.emit(session_id.to_owned(), CodingEvent::Warning {
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
#[tracing::instrument(
    name = "context_compaction",
    skip_all,
    fields(
        session_id,
        model = request.model,
        estimated_tokens = request.estimate.tokens,
        context_window_size = request.facts.context_window_tokens
    )
)]
pub(crate) async fn compact_context(
    history: &mut History,
    client: &Client,
    file_tracker: &FileTracker,
    request: CompactionRequest<'_>,
    emitter: &Emitter,
    session_id: &str,
) -> Result<CompactionOutcome> {
    let original_turn_count = history.len();
    // The newest turn stays whatever the caller asked for: it may be an
    // assistant turn whose tool calls are about to be answered, and its
    // results are pushed after this point.
    let preserve_start = history.compact_preserve_start(request.preserve_turns.max(1));
    if preserve_start == 0 {
        return Ok(CompactionOutcome::Unchanged);
    }
    let preserved_turn_count = original_turn_count - preserve_start;

    emitter.emit(session_id.to_owned(), CodingEvent::CompactionStarted {
        estimated_tokens:    request.estimate.tokens,
        context_window_size: request.facts.context_window_tokens,
        reason:              request.reason,
    });

    let max_tokens = summary_max_tokens(
        request.facts.reasons_by_default,
        request.facts.max_output_tokens,
    );
    let summary_request = match Request::builder()
        .model(request.model)
        .system(summarization_prompt(
            SUMMARY_MAX_TOKENS.min(max_tokens),
            file_tracker,
            request.instructions,
        ))
        .user(format!(
            "Here is the conversation to summarize:\n\n{}",
            render_turns_for_summary(&history.turns()[..preserve_start])
        ))
        .max_output_tokens(max_tokens)
        .build()
        .map_err(CompactionError::Request)
    {
        Ok(request) => request,
        Err(source) => {
            let error = Error::from(source);
            emit_compaction_failure(emitter, session_id, request.reason, &error);
            return Err(error);
        }
    };

    let mut drop_guard = CompactionDropGuard {
        emitter,
        session_id,
        reason: request.reason,
        finished: false,
    };
    let response = tokio::select! {
        biased;
        () = request.cancel.cancelled() => {
            drop_guard.finished = true;
            emitter.emit(session_id.to_owned(), CodingEvent::CompactionCancelled {
                reason: request.reason,
            });
            return Err(Error::Interrupted(InterruptReason::Cancelled));
        }
        response = client.complete(summary_request) => {
            match response.map_err(CompactionError::Llm) {
                Ok(response) => response,
                Err(source) => {
                    drop_guard.finished = true;
                    let error = Error::from(source);
                    emit_compaction_failure(emitter, session_id, request.reason, &error);
                    return Err(error);
                }
            }
        }
    };

    drop_guard.finished = true;
    if request.cancel.is_cancelled() {
        emitter.emit(session_id.to_owned(), CodingEvent::CompactionCancelled {
            reason: request.reason,
        });
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    // `compact_from` discards the summarized turns for good, so an empty
    // summary is refused before history is touched. Trimming first stops a
    // whitespace-only response from passing as a summary.
    let response_text = response.text();
    let summary = response_text.trim();
    if summary.is_empty() {
        let error = Error::from(CompactionError::EmptySummary {
            summarized_turn_count: preserve_start,
        });
        emit_compaction_failure(emitter, session_id, request.reason, &error);
        return Err(error);
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

    let result = CompactionResult {
        summary: content,
        reason: request.reason,
        original_turn_count,
        preserved_turn_count,
        estimated_tokens_before: request.estimate.tokens,
        summary_token_estimate,
        tracked_file_count: file_tracker.file_count(),
        summary_truncated,
        usage: TokenUsage::from(response.usage),
        cost_usd_micros: response.cost.map(|cost| cost.usd_micros),
    };

    history.compact_from(preserve_start, &result);

    emitter.emit(session_id.to_owned(), CodingEvent::CompactionCompleted {
        original_turn_count,
        preserved_turn_count,
        summary_token_estimate,
        tracked_file_count: file_tracker.file_count(),
        reason: request.reason,
    });

    Ok(CompactionOutcome::Compacted(result))
}

/// Pairs a started compaction with cancellation when its model future is
/// dropped.
struct CompactionDropGuard<'a> {
    emitter:    &'a Emitter,
    session_id: &'a str,
    reason:     CompactionReason,
    finished:   bool,
}

impl Drop for CompactionDropGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.emitter.emit(
                self.session_id.to_owned(),
                CodingEvent::CompactionCancelled {
                    reason: self.reason,
                },
            );
        }
    }
}

fn emit_compaction_failure(
    emitter: &Emitter,
    session_id: &str,
    reason: CompactionReason,
    error: &Error,
) {
    emitter.emit(session_id.to_owned(), CodingEvent::CompactionFailed {
        reason,
        error: ErrorData::from(error),
    });
}

/// The instructions the summarizing call is given.
///
/// The file list is asked for verbatim because it is the part of a session a
/// summary most reliably loses, and the part the next assistant most needs.
fn summarization_prompt(
    visible_max_tokens: u32,
    file_tracker: &FileTracker,
    instructions: Option<&str>,
) -> String {
    let file_operations = if file_tracker.is_empty() {
        String::new()
    } else {
        format!(
            "\n## File Operations\nCOPY THIS SECTION VERBATIM into your summary.\n\n{}",
            file_tracker.render()
        )
    };

    let custom_instructions = instructions
        .filter(|instructions| !instructions.trim().is_empty())
        .map_or_else(String::new, |instructions| {
            format!("\n\n## Additional Instructions\n{}", instructions.trim())
        });

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
{file_operations}{custom_instructions}"
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
pub(crate) fn estimate_active_context_usage(
    system_prompt: &str,
    history: &History,
) -> ContextEstimate {
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
        Message::User { content, .. } | Message::Steering { content, .. } => {
            content.text_content().len()
        }
        Message::System { content, .. } => content.len(),
        Message::Compaction { summary, .. } => summary.len(),
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let reasoning = turn.reasoning_text().map_or(0, str::len);
            let calls: usize = tool_calls
                .iter()
                .map(|call| call.name.len() + call.input.raw().len())
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
pub(crate) fn render_turns_for_summary(turns: &[Message]) -> String {
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
                        clipped_arguments(
                            &call
                                .input
                                .to_value()
                                .unwrap_or_else(|_| Value::String(call.input.raw().to_owned()))
                        )
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
            Message::Compaction { summary, .. } => {
                let _ = writeln!(out, "Compaction: {summary}");
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

/// One tool call's arguments, serialized and cut to the transcript budget.
///
/// The arguments can hold an entire file, and the transcript keeps only
/// [`TRANSCRIPT_FIELD_BYTES`] of them, so serialization runs into a capped
/// buffer and stops at the cap rather than producing the whole value first.
fn clipped_arguments(arguments: &serde_json::Value) -> String {
    /// One byte past the budget, so a value that fills it exactly is
    /// distinguishable from one the cap cut.
    const CAPACITY: usize = TRANSCRIPT_FIELD_BYTES + 1;

    struct CappedWriter(Vec<u8>);

    impl Write for CappedWriter {
        fn write(&mut self, bytes: &[u8]) -> IoResult<usize> {
            let room = CAPACITY - self.0.len();
            if room == 0 {
                return Err(IoError::other("the transcript budget is spent"));
            }
            let taken = bytes.len().min(room);
            self.0.extend_from_slice(&bytes[..taken]);
            Ok(taken)
        }

        fn flush(&mut self) -> IoResult<()> {
            Ok(())
        }
    }

    let mut writer = CappedWriter(Vec::with_capacity(CAPACITY));
    let complete = serde_json::to_writer(&mut writer, arguments).is_ok();
    // The cap can land inside a multi-byte character; keep the valid prefix.
    let text = match String::from_utf8(writer.0) {
        Ok(text) => text,
        Err(error) => {
            let valid_up_to = error.utf8_error().valid_up_to();
            let mut bytes = error.into_bytes();
            bytes.truncate(valid_up_to);
            String::from_utf8(bytes).expect("the bytes below the cut are valid UTF-8")
        }
    };
    if complete {
        clip(&text)
    } else {
        format!(
            "{}...",
            &text[..floor_char_boundary(&text, TRANSCRIPT_FIELD_BYTES)]
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, LazyLock};
    use std::time::SystemTime;

    use lithos_llm::Client;
    use lithos_llm::types::{ContentPart, ErrorKind as LlmErrorKind, ToolCall, ToolResult};
    use serde_json::json;

    use super::*;
    use crate::error::Error;
    use crate::event::{EventOptions, EventPump};
    use crate::record::StoredMessage;
    use crate::runtime::testing::history_from;
    use crate::test_support::{
        ScriptedCompletion, ScriptedFailure, ScriptedProvider, client_from, message_text,
        test_catalog, text_response,
    };
    use crate::types::{CodingAgentEvent, TokenUsage};

    static NEVER_CANCEL: LazyLock<CancellationToken> = LazyLock::new(CancellationToken::new);

    /// A summarization call that answers with `text`.
    fn summary(text: &str) -> ScriptedCompletion {
        ScriptedCompletion::response(text_response(text))
    }

    /// A summarization call that fails outright.
    fn failure() -> ScriptedCompletion {
        ScriptedCompletion::Failure(ScriptedFailure::terminal(
            LlmErrorKind::Server,
            "provider is down",
        ))
    }

    /// A client whose non-streaming calls — compaction's — answer with
    /// `completion`, and the provider handle the sent requests are read from.
    fn client(completion: ScriptedCompletion) -> (Client, Arc<ScriptedProvider>) {
        client_from(ScriptedProvider::new(Vec::new()).completing(vec![completion]))
    }

    fn now() -> SystemTime {
        SystemTime::now()
    }

    fn user(content: &str) -> Message {
        Message::User {
            content:   content.into(),
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
        async fn drain(mut self) -> Vec<CodingEvent> {
            let mut receiver = self.emitter.subscribe();
            drop(self.emitter);
            self.pump
                .take()
                .expect("the pump is taken once")
                .run()
                .await
                .expect("no sink can refuse");
            let mut events = Vec::new();
            while let Ok(CodingAgentEvent { event, .. }) = receiver.try_recv() {
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
            reason:         CompactionReason::Threshold,
            instructions:   None,
            cancel:         &NEVER_CANCEL,
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

    /// The facts the shared test catalog records about one of its models.
    fn catalog_facts(model: &str) -> ModelFacts {
        ModelFacts::from_catalog_model(test_catalog().model("test", model).expect("a known model"))
    }

    #[test]
    fn a_reasoning_capability_alone_earns_no_headroom() {
        // The case a `reasoning` capability alone gets wrong: the model can
        // reason, but nothing in its capabilities says it does so unasked, so
        // the budget is the summary's own. The catalog's `always-thinking` row
        // is that shape — the `reasoning_by_default` metadata that corrects it
        // is applied at session build, not here.
        let facts = catalog_facts("always-thinking");

        assert!(!facts.reasons_by_default);
        assert_eq!(
            summary_max_tokens(facts.reasons_by_default, facts.max_output_tokens),
            SUMMARY_MAX_TOKENS
        );
    }

    #[test]
    fn a_model_that_always_reasons_gets_the_headroom_from_its_catalog_row() {
        // A model with named effort levels reasons unless told not to, so its
        // row alone earns the headroom.
        let facts = catalog_facts("thinking");

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
                content:   "focus".into(),
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
    fn a_transcript_clips_long_tool_arguments_as_the_whole_serialization_would() {
        // Multi-byte content, so the capped serialization has to cut at the
        // same character boundary the uncapped form would.
        let arguments = json!({ "content": "€".repeat(1_000) });
        let rendered = render_turns_for_summary(&[Message::Assistant {
            content:        String::new(),
            tool_calls:     vec![ToolCall::function("c1", "write_file", arguments.clone())],
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp_1".to_owned(),
            timestamp:      now(),
        }]);

        assert!(rendered.len() < arguments.to_string().len());
        assert!(rendered.contains(&format!(
            "[Tool call: write_file] {}",
            clip(&arguments.to_string())
        )));
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
                content:   "s".repeat(8).into(),
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
                CodingEvent::Warning { kind, details, .. }
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
        result:  Result<CompactionOutcome>,
        history: History,
        before:  Vec<StoredMessage>,
        events:  Vec<CodingEvent>,
    }

    /// Compacts a four-turn history against a provider that answers
    /// `completion`.
    async fn compact_with(completion: ScriptedCompletion) -> Compacted {
        let mut history = history_from(
            (0..4)
                .map(|index| user(&format!("message {index}")))
                .collect(),
        );
        let before = history.to_stored_messages();
        let (client, _) = client(completion);
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
                Message::Compaction { summary, .. } => Some(summary.as_str()),
                _ => None,
            })
            .expect("a compacted history carries a summary turn")
    }

    #[tokio::test]
    async fn a_summary_replaces_the_older_turns() {
        let compacted = compact_with(summary("Brief handoff.")).await;

        compacted.result.expect("a summary compacts");
        let summary = summary_turn(&compacted.history);
        assert!(summary.contains("A different assistant began this task"));
        assert!(summary.contains("Brief handoff."));
        assert!(
            compacted
                .events
                .iter()
                .any(|event| matches!(event, CodingEvent::CompactionCompleted { .. }))
        );
    }

    #[tokio::test]
    async fn a_blank_summary_leaves_history_alone() {
        for blank in ["", "   \n\t  \n "] {
            let compacted = compact_with(summary(blank)).await;

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
                    .any(|event| matches!(event, CodingEvent::CompactionStarted { .. })),
                "the attempt is still recorded"
            );
            assert!(
                compacted
                    .events
                    .iter()
                    .any(|event| matches!(event, CodingEvent::CompactionFailed { .. })),
                "a refused summary has a terminal failure event"
            );
            assert!(
                !compacted.events.iter().any(|event| matches!(
                    event,
                    CodingEvent::CompactionCompleted { .. }
                        | CodingEvent::CompactionCancelled { .. }
                )),
                "a refused summary has exactly one terminal outcome"
            );
        }
    }

    #[tokio::test]
    async fn a_failed_summarization_call_leaves_history_alone() {
        let compacted = compact_with(failure()).await;

        let error = compacted.result.expect_err("a failed call fails");
        assert!(
            matches!(error, Error::Compaction(CompactionError::Llm(_))),
            "unexpected error: {error}"
        );
        assert_eq!(compacted.history.to_stored_messages(), compacted.before);
        assert!(
            compacted
                .events
                .iter()
                .any(|event| matches!(event, CodingEvent::CompactionFailed { .. }))
        );
    }

    #[tokio::test]
    async fn an_overlong_summary_is_bounded_before_it_is_kept() {
        let max_bytes = summary_max_approx_bytes();
        let overlong = format!("{}END", "€".repeat(max_bytes / 3 + 1));

        let compacted = compact_with(summary(&overlong)).await;

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
        let (client, _) = client(summary("Brief handoff."));
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
        let (client, provider) = client(summary("unused"));
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
        assert!(provider.completion_requests().is_empty());
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
        let (client, provider) = client(summary("done"));
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

        let sent = provider.completion_requests();
        assert_eq!(sent.len(), 1);
        let request = &sent[0];
        assert_eq!(request.model(), "test/model");
        assert_eq!(request.max_output_tokens(), Some(SUMMARY_MAX_TOKENS));
        assert!(request.tools().is_empty());

        let text: String = request.messages().iter().map(message_text).collect();
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
        let (client, provider) = client(summary("done"));
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

        let sent = provider.completion_requests();
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
        let (client, provider) = client(summary("unused"));
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
        assert!(provider.completion_requests().is_empty());
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
