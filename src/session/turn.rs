//! The turn loop: one input, however many rounds it takes.
//!
//! A round is one exchange with the model. The session drains whatever was
//! steered in, builds a request, reads the response as it streams, commits the
//! turn, runs the tools it asked for, and goes round again. It stops when the
//! model answers with no tool calls, when something cancels the run, or when a
//! model call fails for good.
//!
//! Three rules shape everything here, and each is a thing that goes wrong in a
//! long-running agent if it is not held:
//!
//! - **Every tool call gets a result.** Tool results are committed whatever
//!   interrupted the round, and a tool that was cancelled answers "Cancelled"
//!   rather than being dropped mid-flight. A conversation carrying a call with
//!   no result is one a provider will refuse for the rest of the session.
//! - **A replayed turn withdraws what it showed.** Before any replay the
//!   session publishes an empty
//!   [`AssistantOutputReplace`](crate::AgentEvent::AssistantOutputReplace), so
//!   a reader never sees the first half of a turn twice.
//! - **An interrupt is announced exactly once.** Interrupt gestures are
//!   counted, and the loop settles the count as it publishes one
//!   [`RoundInterrupted`](crate::AgentEvent::RoundInterrupted) per gesture:
//!   never two for one, never none.

use std::sync::{Arc, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use futures_util::StreamExt as _;
use lithos_llm::middleware::{CallContext, CancellationToken as CallCancellation};
use lithos_llm::types::{
    ContentPart, Error as LlmError, ErrorKind as LlmErrorKind, FinishReason, Request, Response,
    ResponseStream, RetryClassification, StreamEvent, ToolCall, ToolChoice, ToolResult,
};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use super::control::SteeringItem;
use super::retry::RetryEventBridge;
use super::{RunTotals, Session};
use crate::compaction::{CompactionRequest, check_context_usage, compact_context};
use crate::context_window::{
    ContextWindowInput, build_local_snapshot, context_window_from_response_usage,
};
use crate::error::{Error, ErrorData, Result};
use crate::human_input::HumanInputProvider;
use crate::loop_detection::detect_loop;
use crate::reasoning::ReasoningOutput;
use crate::skills::{ExpandedInput, SkillExpansion, expand_skill};
use crate::task_reminder::maybe_task_reminder;
use crate::tool::{NativeTool, ToolDispatch, canonical_tool_name};
use crate::types::{
    AgentEvent, ContextWindowSnapshot, CostSource, LlmOutputKind, LlmRetryPhase, Message,
    SessionState, SkillActivationSource, TokenUsage,
};

/// How many times the session replays a turn whose stream broke after it had
/// already shown output.
///
/// Three replays, so four attempts in all. Failures before any visible output
/// never reach here: the client's retry middleware reconnects underneath the
/// session, which is the only layer that can do it without a reader noticing.
const STREAM_CONSUME_RETRIES: usize = 3;

/// What the model is told when it keeps making the same calls.
const LOOP_WARNING: &str = "WARNING: Loop detected. You appear to be repeating the same tool \
                            calls. Please try a different approach or ask for clarification.";

/// What a stream that stopped without finishing its response reports.
const TRUNCATED_STREAM: &str = "the stream ended without completing the response";

/// What a stream that stopped without finishing reports once the replays are
/// spent.
const TRUNCATED_STREAM_EXHAUSTED: &str =
    "the stream ended without completing the response, after every replay";

/// One request, and what it was measured to cost before it was sent.
struct BuiltRequest {
    request:        Request,
    context_window: ContextWindowSnapshot,
}

/// An opened provider stream and the handle that cancels the call behind it.
struct OpenStream {
    stream: ResponseStream,
    /// The client's own cancellation, which the session fires before it drops
    /// the stream so retry backoff, middleware, and the adapter all stop too.
    cancel: CallCancellation,
}

/// How opening a stream ended.
enum StreamOpen {
    Opened(OpenStream),
    /// The call failed for good; the error has already been published.
    Failed(Error),
    /// A cancellation won the race, so nothing was opened.
    Cancelled,
}

/// How one attempt at reading a stream ended.
enum AttemptOutcome {
    /// The response arrived whole.
    Completed(Box<Response>),
    /// The stream failed while it was being read.
    Failed(LlmError),
    /// The stream stopped without finishing its response.
    Truncated,
}

/// Takes the running span out of `start` and adds it to `total`.
fn record_elapsed(start: &mut Option<Instant>, total: &mut Duration) {
    if let Some(started) = start.take() {
        *total = total.saturating_add(started.elapsed());
    }
}

/// Which kind of output a stream event is the first sign of, if any.
///
/// The point is to name what the model started producing, so the event that
/// only proves the provider answered — the stream's own opening — is
/// deliberately not counted, and neither is accounting that can arrive before
/// any content exists.
fn first_output_kind(event: &StreamEvent) -> Option<LlmOutputKind> {
    use lithos_llm::types::ContentBlockKind;

    match event {
        StreamEvent::ContentBlockStart { kind, .. } => match kind {
            ContentBlockKind::Text => Some(LlmOutputKind::Text),
            ContentBlockKind::Reasoning => Some(LlmOutputKind::Reasoning),
            ContentBlockKind::ToolCall { .. } => Some(LlmOutputKind::ToolCall),
            // Provider-native blocks, and any block kind a later client adds,
            // carry nothing a reader would recognize as output.
            _ => None,
        },
        StreamEvent::TextDelta { .. } => Some(LlmOutputKind::Text),
        StreamEvent::ReasoningDelta { .. } => Some(LlmOutputKind::Reasoning),
        StreamEvent::ToolCallDelta { .. } => Some(LlmOutputKind::ToolCall),
        StreamEvent::ContentBlockEnd { part, .. } => match part {
            ContentPart::Text { .. } => Some(LlmOutputKind::Text),
            ContentPart::Reasoning(_) => Some(LlmOutputKind::Reasoning),
            ContentPart::ToolCall(_) => Some(LlmOutputKind::ToolCall),
            _ => None,
        },
        // The stream opening, its accounting, and its terminal event: all
        // proof that the provider answered, none of it output.
        _ => None,
    }
}

/// Adds one response's tokens to a run's running total.
fn add_usage(total: TokenUsage, one: TokenUsage) -> TokenUsage {
    TokenUsage {
        input:       total.input.saturating_add(one.input),
        output:      total.output.saturating_add(one.output),
        reasoning:   total.reasoning.saturating_add(one.reasoning),
        cache_read:  total.cache_read.saturating_add(one.cache_read),
        cache_write: total.cache_write.saturating_add(one.cache_write),
    }
}

/// The tool calls a response asked for, in the order it asked.
fn tool_calls_of(response: &Response) -> Vec<ToolCall> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

/// The parts of a response that only the provider understands.
///
/// Kept verbatim and replayed in place, because a reasoning block's signature
/// and a provider's own opaque items are what make a multi-turn conversation
/// valid on the next call.
fn provider_parts_of(response: &Response) -> Vec<ContentPart> {
    response
        .content
        .iter()
        .filter(|part| matches!(part, ContentPart::Opaque { .. } | ContentPart::Reasoning(_)))
        .cloned()
        .collect()
}

impl Session {
    /// Runs one input until the model stops asking for tools.
    ///
    /// Answers with the assistant's final text when it ended with any, and
    /// `None` when it ended with nothing worth showing.
    pub(super) async fn run_single_input(
        &mut self,
        input: &str,
        skill_expansion: SkillExpansion,
        human_input: Option<&Arc<dyn HumanInputProvider>>,
        totals: &mut RunTotals,
    ) -> Result<Option<String>> {
        if self.state == SessionState::Closed {
            return Err(Error::SessionClosed);
        }
        self.transition(SessionState::Thinking);

        let expanded = self.expand_input(input, skill_expansion)?;
        if let Some(name) = &expanded.skill_name {
            self.activated_skill_context_observed = true;
            self.emit(AgentEvent::SkillActivated {
                skill_name: name.clone(),
                source:     SkillActivationSource::Slash,
            });
        }
        self.history.push(Message::User {
            content:   expanded.text.clone(),
            timestamp: SystemTime::now(),
        });
        self.emit(AgentEvent::UserInput {
            text: expanded.text,
        });

        // One failed summarization is unlikely to succeed again inside the
        // same input, and both checkpoints would try. Suppressing further
        // attempts stops a provider returning nothing from turning into a run
        // of paid calls; the next input starts fresh.
        let mut compaction_failed = false;

        loop {
            // A refusing sink has already stopped the pipeline, so the run
            // ends here rather than working on with nothing recording it.
            self.check_pump().await?;

            let round_was_interrupted = self.refresh_round_token();

            // Ending the run beats a park: a session waiting for a steer that
            // was also cancelled is cancelled.
            if self.cancel_token.is_cancelled() {
                return Err(self.close_cancelled().await);
            }

            if round_was_interrupted {
                self.settle_interrupts();
            }

            // Steering pushed mid-round arrives as the first turn of the next
            // one. Drain, park if a bare interrupt left nothing to say, then
            // drain again — what woke the park is a steer that belongs to this
            // round.
            self.drain_steering();
            self.wait_for_steer_if_needed().await?;
            self.drain_steering();

            // Stable for this round even when a control handle swaps a fresh
            // token into the shared cell.
            let round_token = self
                .round_token
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();

            if !compaction_failed {
                compaction_failed = self.compact_if_needed().await;
            }

            // Staged, not committed: an interrupted round must not leave a
            // system message behind for later steering to answer. It commits
            // only alongside the assistant turn that read it.
            let pending_task_reminder = self.task_reminder_if_needed();

            let built = self.build_request(pending_task_reminder.as_ref())?;
            let local_context_window = built.context_window;
            let request = built.request;

            // The last moment at which nothing is known about the response.
            // One per round, however many times the turn is replayed inside
            // it.
            self.emit(AgentEvent::LlmRequestStarted {
                requested_model: self.model.clone(),
            });

            let mut inference_start = Some(Instant::now());
            let mut open = match self.open_stream(&request, &round_token).await {
                StreamOpen::Opened(open) => open,
                StreamOpen::Failed(error) => {
                    record_elapsed(&mut inference_start, &mut totals.timing.inference);
                    return Err(error);
                }
                StreamOpen::Cancelled => {
                    record_elapsed(&mut inference_start, &mut totals.timing.inference);
                    if self.cancel_token.is_cancelled() {
                        return Err(self.close_cancelled().await);
                    }
                    // The round was interrupted before the provider answered.
                    continue;
                }
            };

            let mut response: Option<Response> = None;
            let mut steer_interrupted = false;
            // Whether output from any attempt is still on a reader's screen.
            let mut visible_output_present = false;

            'attempts: for attempt in 0..=STREAM_CONSUME_RETRIES {
                let mut completed: Option<Response> = None;
                let mut stream_error: Option<LlmError> = None;
                let mut attempt_emitted_output = false;
                // Re-armed per attempt: a replay discards everything the last
                // attempt produced, so its first output is a new observation.
                let mut first_output_emitted = false;

                loop {
                    let chunk = tokio::select! {
                        biased;
                        () = round_token.cancelled() => None,
                        () = self.cancel_token.cancelled() => None,
                        next = open.stream.next() => Some(next),
                    };
                    // A cancellation won the race.
                    let Some(next) = chunk else {
                        break;
                    };
                    // The stream ended.
                    let Some(item) = next else {
                        break;
                    };

                    match item {
                        Ok(event) => {
                            if !first_output_emitted && let Some(kind) = first_output_kind(&event) {
                                first_output_emitted = true;
                                self.emit(AgentEvent::LlmFirstOutput { kind });
                            }
                            match event {
                                StreamEvent::TextDelta { text, .. } => {
                                    attempt_emitted_output = true;
                                    visible_output_present = true;
                                    self.emit(AgentEvent::TextDelta { delta: text });
                                }
                                StreamEvent::ReasoningDelta { text, .. } => {
                                    attempt_emitted_output = true;
                                    visible_output_present = true;
                                    self.emit(AgentEvent::ReasoningDelta { delta: text });
                                }
                                StreamEvent::Completed { response } => {
                                    completed = Some(response);
                                    break;
                                }
                                _ => {}
                            }
                        }
                        Err(error) => {
                            stream_error = Some(error);
                            break;
                        }
                    }
                }

                // Ending the run wins over everything else the attempt found.
                if self.cancel_token.is_cancelled() {
                    open.cancel.cancel();
                    drop(open);
                    record_elapsed(&mut inference_start, &mut totals.timing.inference);
                    return Err(self.close_cancelled().await);
                }

                // Only the round ended: drop the turn and pick up the steer.
                if round_token.is_cancelled() {
                    open.cancel.cancel();
                    drop(open);
                    steer_interrupted = true;
                    break 'attempts;
                }

                let outcome = match (completed, stream_error) {
                    (Some(finished), _) if finished.finish_reason == FinishReason::Incomplete => {
                        AttemptOutcome::Truncated
                    }
                    (Some(finished), _) => AttemptOutcome::Completed(Box::new(finished)),
                    (None, Some(error)) => AttemptOutcome::Failed(error),
                    (None, None) => AttemptOutcome::Truncated,
                };

                let (error, delay) = match outcome {
                    AttemptOutcome::Completed(finished) => {
                        // The stream is finished, so it is dropped here rather
                        // than at the end of the round: a middleware below
                        // holds its resources for exactly as long as the
                        // stream lives, and the tools this turn asks for run
                        // next.
                        drop(open);
                        response = Some(*finished);
                        break 'attempts;
                    }
                    AttemptOutcome::Failed(error) => {
                        let Some(delay) = self.consume_replay_delay(attempt, &error) else {
                            // Nothing more to try: take back what the turn
                            // showed and end the run on the failure.
                            if visible_output_present {
                                self.replace_visible_output();
                            }
                            record_elapsed(&mut inference_start, &mut totals.timing.inference);
                            return Err(self.emit_llm_error(error));
                        };
                        warn!(
                            attempt = attempt + 1,
                            max = STREAM_CONSUME_RETRIES,
                            error = %error,
                            delay_secs = delay.as_secs_f64(),
                            "The model stream failed mid-turn; replaying the turn"
                        );
                        (ErrorData::from(&error), delay)
                    }
                    AttemptOutcome::Truncated => {
                        if attempt >= STREAM_CONSUME_RETRIES {
                            break 'attempts;
                        }
                        warn!(
                            attempt = attempt + 1,
                            max = STREAM_CONSUME_RETRIES,
                            "The model stream stopped without finishing; replaying the turn"
                        );
                        // The one mid-turn restart with no error behind it.
                        // Without an event of its own the replay would be
                        // invisible to anyone reading the stream.
                        let error = LlmError::new(LlmErrorKind::StreamDecode, TRUNCATED_STREAM);
                        (ErrorData::from(&error), Duration::ZERO)
                    }
                };

                if attempt_emitted_output {
                    self.replace_visible_output();
                    visible_output_present = false;
                }
                // Published here rather than through the client's observer, so
                // `attempt` counts this loop's replays rather than the
                // client's reconnects.
                self.emit(AgentEvent::LlmRetry {
                    provider: self.provider.clone(),
                    model: self.model.clone(),
                    attempt,
                    delay_secs: delay.as_secs_f64(),
                    error,
                    phase: LlmRetryPhase::Consume,
                });

                // The failed attempt's stream is dropped before the wait and
                // the reopen. A middleware below may hold a resource — a
                // concurrency permit, a connection — for exactly as long as
                // its stream lives, and the reopen re-enters that layer to ask
                // for the same resource, so a stream kept alive here would
                // wait on itself. lithos's own stream retry drops its failed
                // stream for this reason.
                open.cancel.cancel();
                drop(open);

                if !delay.is_zero() {
                    let slept = tokio::select! {
                        biased;
                        () = round_token.cancelled() => false,
                        () = self.cancel_token.cancelled() => false,
                        () = sleep(delay) => true,
                    };
                    if !slept {
                        steer_interrupted =
                            round_token.is_cancelled() && !self.cancel_token.is_cancelled();
                        break 'attempts;
                    }
                }

                open = match self.open_stream(&request, &round_token).await {
                    StreamOpen::Opened(reopened) => reopened,
                    StreamOpen::Failed(error) => {
                        record_elapsed(&mut inference_start, &mut totals.timing.inference);
                        return Err(error);
                    }
                    StreamOpen::Cancelled => {
                        steer_interrupted =
                            round_token.is_cancelled() && !self.cancel_token.is_cancelled();
                        break 'attempts;
                    }
                };
            }
            record_elapsed(&mut inference_start, &mut totals.timing.inference);

            // A run ended while a replay was waiting out its delay or opening
            // its stream leaves the turn unfinished, and neither of those two
            // exits can tell that from a turn that ran out of replays. Ask
            // here, so ending a run always ends it as a cancellation.
            //
            // A turn that did finish is not asked: a response that arrived
            // whole is committed and answered, and the run then ends at the
            // next checkpoint. Withdrawing a complete answer because a
            // cancellation landed in the instant after it arrived would lose
            // work the model was already paid for.
            if response.is_none() && self.cancel_token.is_cancelled() {
                if visible_output_present {
                    self.replace_visible_output();
                }
                return Err(self.close_cancelled().await);
            }

            // A steer landed mid-turn: drop the uncommitted turn, take back
            // what it showed, and let the next round deliver the steer.
            if steer_interrupted {
                if visible_output_present {
                    self.replace_visible_output();
                }
                continue;
            }

            let Some(response) = response else {
                if visible_output_present {
                    self.replace_visible_output();
                }
                return Err(self.emit_llm_error(LlmError::new(
                    LlmErrorKind::StreamDecode,
                    TRUNCATED_STREAM_EXHAUSTED,
                )));
            };

            let committed = self.commit_turn(
                &response,
                pending_task_reminder,
                &local_context_window,
                totals,
            );

            // The turn just grew the conversation, so measure it again before
            // the next round is built on top of it.
            if !compaction_failed {
                compaction_failed = self.compact_if_needed().await;
            }

            let tool_calls = match committed {
                Committed::Answered { text } => {
                    if round_token.is_cancelled() {
                        // A steer landed during the final response; deliver it
                        // rather than ending the run on it.
                        continue;
                    }
                    let coordinator_continues = self
                        .completion_coordinator
                        .as_ref()
                        .is_some_and(|coordinator| coordinator.on_natural_completion());
                    if coordinator_continues {
                        continue;
                    }
                    return Ok((!text.trim().is_empty()).then_some(text));
                }
                Committed::CallsTools { tool_calls } => tool_calls,
            };

            // Tools watch one token covering both the end of the run and the
            // end of the round, and answer "Cancelled" cooperatively rather
            // than being dropped, which is what keeps every call paired with a
            // result.
            let composite = CancellationToken::new();
            self.transition(SessionState::Executing);
            let tool_start = Instant::now();
            let results = self
                .execute_tools(&tool_calls, &composite, &round_token, human_input)
                .await;
            totals.timing.tool = totals.timing.tool.saturating_add(tool_start.elapsed());

            if activated_a_skill(&tool_calls, &results) {
                self.activated_skill_context_observed = true;
            }
            self.file_tracker
                .record_from_tool_calls(&tool_calls, &results);

            // Committed whichever token fired, because a call without its
            // result is a conversation no provider will take back.
            self.history.push(Message::ToolResults {
                results,
                timestamp: SystemTime::now(),
            });

            if self.cancel_token.is_cancelled() {
                return Err(self.close_cancelled().await);
            }
            self.transition(SessionState::Thinking);
            if round_token.is_cancelled() {
                continue;
            }

            if self.config.enable_loop_detection
                && detect_loop(&self.history, self.config.loop_detection_window)
            {
                // Pushed straight into history rather than through the
                // steering queue: this is the session talking to the model,
                // not an operator, so it publishes no steering event.
                self.history.push(Message::Steering {
                    content:   LOOP_WARNING.to_owned(),
                    timestamp: SystemTime::now(),
                });
                self.emit(AgentEvent::LoopDetected);
            }
        }
    }

    /// Expands a `/name` reference in the input, where one is allowed.
    fn expand_input(&self, input: &str, expansion: SkillExpansion) -> Result<ExpandedInput> {
        if self.skills.is_empty() || expansion == SkillExpansion::Skip {
            return Ok(ExpandedInput {
                text:       input.to_owned(),
                skill_name: None,
            });
        }
        expand_skill(&self.skills, input).map_err(|error| Error::InvalidState(error.to_string()))
    }

    /// Arms a fresh round token when the last round's was cancelled.
    ///
    /// Answers whether the round that just ended was interrupted, which is what
    /// tells the loop there are interrupt generations to settle.
    fn refresh_round_token(&self) -> bool {
        let cancelled = self
            .round_token
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .is_cancelled();
        if cancelled {
            *self
                .round_token
                .write()
                .unwrap_or_else(PoisonError::into_inner) = CancellationToken::new();
        }
        cancelled
    }

    /// Publishes one event per interrupt gesture that has not been announced.
    fn settle_interrupts(&mut self) {
        let generations = {
            let mut control = self
                .control_state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let first = control.settled_interrupt_generation.saturating_add(1);
            let last = control.interrupt_generation;
            control.settled_interrupt_generation = last;
            if first <= last {
                (first..=last).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };
        for generation in generations {
            self.emit(AgentEvent::RoundInterrupted { generation });
        }
    }

    /// Commits everything queued as real conversation turns.
    fn drain_steering(&mut self) {
        let items: Vec<SteeringItem> = {
            let mut control = self
                .control_state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            control.queue.drain(..).collect()
        };
        for item in items {
            let timestamp = SystemTime::now();
            match item {
                SteeringItem::Steering { text, actor } => {
                    self.history.push(Message::Steering {
                        content: text.clone(),
                        timestamp,
                    });
                    self.emitter
                        .emit(self.id.clone(), AgentEvent::SteeringInjected {
                            text,
                            actor,
                        });
                }
                SteeringItem::User { text } => self.history.push(Message::User {
                    content: text,
                    timestamp,
                }),
                SteeringItem::System { text } => self.history.push(Message::System {
                    content: text,
                    timestamp,
                }),
            }
        }
    }

    /// Waits, where an interrupt parked the session, until someone says what to
    /// do next.
    ///
    /// Only ending the run wakes this with a failure. The round token means
    /// nothing here: the round it named is already over.
    async fn wait_for_steer_if_needed(&mut self) -> Result<()> {
        let notify = Arc::clone(&self.control_notify);
        loop {
            let cancelled = {
                let notified = notify.notified();
                tokio::pin!(notified);
                // Registered before the queue is read, so a steer that lands
                // in between still wakes this wait.
                notified.as_mut().enable();

                let should_wait = {
                    let control = self
                        .control_state
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    control.waiting_for_steer && control.queue.is_empty()
                };
                if !should_wait {
                    return Ok(());
                }

                tokio::select! {
                    biased;
                    () = self.cancel_token.cancelled() => true,
                    () = notified => false,
                }
            };
            if cancelled {
                return Err(self.close_cancelled().await);
            }
        }
    }

    /// Summarizes the older turns when the conversation is close to the
    /// model's window.
    ///
    /// Answers whether an attempt failed, which suppresses the rest of this
    /// input's attempts. The usage check runs even where compaction is turned
    /// off, so an application still hears that the window is filling up.
    async fn compact_if_needed(&mut self) -> bool {
        let Some(estimate) = check_context_usage(
            &self.system_prompt,
            &self.history,
            self.facts.context_window_tokens,
            self.config.compaction_threshold_percent,
            &self.emitter,
            &self.id,
        ) else {
            return false;
        };
        if !self.config.enable_context_compaction {
            return false;
        }

        let request = CompactionRequest {
            model: &self.model_selector,
            facts: self.facts,
            preserve_turns: self.config.compaction_preserve_turns,
            estimate,
        };
        if let Err(error) = compact_context(
            &mut self.history,
            &self.client,
            &self.file_tracker,
            request,
            &self.emitter,
            &self.id,
        )
        .await
        {
            // Not fatal: the conversation is intact, only larger than it
            // should be.
            self.emitter.emit(self.id.clone(), AgentEvent::Error {
                error: ErrorData::from(&error),
            });
            return true;
        }
        false
    }

    /// The reminder to stage this round, where the model has drifted from the
    /// task.
    fn task_reminder_if_needed(&self) -> Option<Message> {
        let tools = self.effective_tools();
        let names: Vec<&str> = tools
            .iter()
            .map(|tool| tool.definition.name.as_str())
            .collect();
        maybe_task_reminder(&self.history, &names).map(|content| Message::System {
            content,
            timestamp: SystemTime::now(),
        })
    }

    /// Builds this round's request, and measures what it will cost to send.
    ///
    /// A staged reminder goes last, through the same conversion durable turns
    /// use, so what the model reads now is what history will hold if the turn
    /// commits.
    fn build_request(&self, pending_task_reminder: Option<&Message>) -> Result<BuiltRequest> {
        let mut builder = Request::builder().model(self.model_selector.clone());
        if !self.system_prompt.trim().is_empty() {
            builder = builder.system(self.system_prompt.clone());
        }
        for message in self.history.to_llm_messages() {
            builder = builder.message(message);
        }
        if let Some(reminder) = pending_task_reminder {
            builder = builder.message(reminder.to_llm_message());
        }

        let tools = self.effective_tools();
        for tool in &tools {
            builder = builder.tool(tool.definition.clone());
        }
        if !tools.is_empty() {
            builder = builder.tool_choice(ToolChoice::Auto);
        }
        if let Some(max_output_tokens) = self.max_output_tokens() {
            builder = builder.max_output_tokens(max_output_tokens);
        }
        if let Some(effort) = self.config.reasoning_effort {
            builder = builder.reasoning_effort(effort);
        }
        if let Some(speed) = self.config.speed {
            builder = builder.speed(speed);
        }

        let request = builder.build().map_err(|error| {
            Error::InvalidState(format!("this round's request could not be built: {error}"))
        })?;
        let context_window = build_local_snapshot(ContextWindowInput {
            request: &request,
            tools: &tools,
            system_prompt: &self.system_prompt,
            memory: &self.memory,
            skills: &self.skills,
            tool_vocabulary: self.registry.vocabulary(),
            activated_skill_context_observed: self.activated_skill_context_observed,
            provider: &self.provider,
            model: &self.model,
            context_window_tokens: self.facts.context_window_tokens,
        });
        Ok(BuiltRequest {
            request,
            context_window,
        })
    }

    /// The most tokens the model may produce in one turn.
    ///
    /// The application's own budget where it set one, and the catalog's limit
    /// otherwise, so a model is asked for what it can actually give.
    fn max_output_tokens(&self) -> Option<u32> {
        let configured = self
            .config
            .max_tokens
            .and_then(|tokens| u32::try_from(tokens).ok());
        let from_catalog = || {
            self.facts
                .max_output_tokens
                .map(|tokens| u32::try_from(tokens).unwrap_or(u32::MAX))
        };
        configured
            .or_else(from_catalog)
            .filter(|tokens| *tokens > 0)
    }

    /// Opens the provider stream, racing it against both cancellations.
    ///
    /// The call carries a bridge back to this session's event stream, so
    /// retries the client decides for itself are published here; and its own
    /// cancellation, which the session fires so backoff and adapters stop when
    /// the session does.
    async fn open_stream(
        &mut self,
        request: &Request,
        round_token: &CancellationToken,
    ) -> StreamOpen {
        // Fresh per call, including every replay: the client's attempt budget
        // starts over for a turn the session chose to replay, and each call
        // gets its own identity in tracing.
        let mut context = CallContext::new();
        let cancel = context.cancellation().clone();
        context.extensions_mut().insert(RetryEventBridge::new(
            self.emitter.clone(),
            self.id.clone(),
            self.provider.clone(),
            self.model.clone(),
        ));

        let client = self.client.clone();
        let request = request.clone();
        let opening = client.stream_with_context(request, context);
        tokio::pin!(opening);
        let opened = tokio::select! {
            biased;
            () = round_token.cancelled() => None,
            () = self.cancel_token.cancelled() => None,
            result = &mut opening => Some(result),
        };

        match opened {
            Some(Ok(stream)) => StreamOpen::Opened(OpenStream { stream, cancel }),
            Some(Err(error)) => StreamOpen::Failed(self.emit_llm_error(error)),
            None => {
                cancel.cancel();
                StreamOpen::Cancelled
            }
        }
    }

    /// How long to wait before replaying this turn, or `None` to stop.
    ///
    /// Two budgets have to agree: the session replays a turn at most
    /// [`STREAM_CONSUME_RETRIES`] times, and the configured policy decides
    /// whether this particular failure is worth repeating at all.
    fn consume_replay_delay(&self, attempt: usize, error: &LlmError) -> Option<Duration> {
        if matches!(error.retry_classification(), RetryClassification::Never)
            || attempt >= STREAM_CONSUME_RETRIES
        {
            return None;
        }
        let failed_attempt = u32::try_from(attempt.saturating_add(1)).unwrap_or(u32::MAX);
        self.config.retry_policy.next_delay(failed_attempt, error)
    }

    /// Takes back everything the current turn has shown.
    fn replace_visible_output(&self) {
        self.emit(AgentEvent::AssistantOutputReplace {
            text:      String::new(),
            reasoning: None,
        });
    }

    /// Records the assistant turn and publishes it.
    fn commit_turn(
        &mut self,
        response: &Response,
        pending_task_reminder: Option<Message>,
        local_context_window: &ContextWindowSnapshot,
        totals: &mut RunTotals,
    ) -> Committed {
        let text = response.text();
        let tool_calls = tool_calls_of(response);
        // Normalized before the content moves into history.
        let reasoning = ReasoningOutput::from_content(&response.content);
        let provider_parts = provider_parts_of(response);
        let usage = TokenUsage::from(response.usage);
        let context_window = Some(context_window_from_response_usage(
            local_context_window,
            usage,
        ));

        totals.usage = add_usage(totals.usage, usage);
        if let Some(cost) = response.cost {
            totals.cost_usd_micros = Some(
                totals
                    .cost_usd_micros
                    .unwrap_or(0)
                    .saturating_add(cost.usd_micros),
            );
        }

        if let Some(reminder) = pending_task_reminder {
            self.history.push(reminder);
        }
        self.history.push(Message::Assistant {
            content: text.clone(),
            tool_calls: tool_calls.clone(),
            provider_parts,
            usage,
            response_id: response.id.clone().unwrap_or_default(),
            timestamp: SystemTime::now(),
        });

        let answering_model = response.model.model().as_str();
        self.emit(AgentEvent::AssistantMessage {
            text: text.clone(),
            model: if answering_model.is_empty() {
                self.model.clone()
            } else {
                answering_model.to_owned()
            },
            usage,
            cost_usd_micros: response.cost.map(|cost| cost.usd_micros),
            cost_source: response.cost.map(|cost| CostSource::from(cost.source)),
            tool_call_count: tool_calls.len(),
            context_window,
            reasoning,
        });

        if tool_calls.is_empty() {
            Committed::Answered { text }
        } else {
            Committed::CallsTools { tool_calls }
        }
    }

    /// Answers every tool call in one round, cancelling them together when
    /// either the round or the run ends.
    ///
    /// The round is watched from here rather than from a task of its own, so
    /// nothing outlives the round and the call to the tools is never dropped
    /// half-finished. Cancellation is therefore only as prompt as the tools
    /// are: a cancelled round waits for every call it made to answer, and a
    /// tool that ignores its token holds the round open until it returns.
    async fn execute_tools(
        &self,
        calls: &[ToolCall],
        composite: &CancellationToken,
        round_token: &CancellationToken,
        human_input: Option<&Arc<dyn HumanInputProvider>>,
    ) -> Vec<ToolResult> {
        let mut dispatch = ToolDispatch::new(
            &self.registry,
            &self.env,
            &self.config,
            &self.emitter,
            &self.id,
            &self.root_session_id,
        );
        if let Some(provider) = self.tool_env_provider.as_ref() {
            dispatch = dispatch.with_tool_env_provider(provider);
        }
        if let Some(provider) = human_input {
            dispatch = dispatch.with_human_input(provider);
        }
        dispatch = dispatch.with_redactor(&self.redactor);
        if let Some(summarizer) = self.web_fetch_summarizer.as_ref() {
            dispatch = dispatch.with_web_fetch_summarizer(summarizer);
        }

        let terminal = self.cancel_token.clone();
        let round = round_token.clone();
        let mut terminal_seen = false;
        let mut round_seen = false;

        let running = dispatch.execute(calls, true, composite);
        tokio::pin!(running);
        loop {
            tokio::select! {
                biased;
                () = terminal.cancelled(), if !terminal_seen => {
                    terminal_seen = true;
                    composite.cancel();
                }
                () = round.cancelled(), if !round_seen => {
                    round_seen = true;
                    composite.cancel();
                }
                results = &mut running => return results,
            }
        }
    }
}

/// What committing an assistant turn left the loop to do.
enum Committed {
    /// The model answered and asked for nothing.
    Answered { text: String },
    /// The model asked for tools.
    CallsTools { tool_calls: Vec<ToolCall> },
}

/// Whether any call in this round successfully loaded a skill.
fn activated_a_skill(calls: &[ToolCall], results: &[ToolResult]) -> bool {
    calls.iter().zip(results).any(|(call, result)| {
        !result.is_error && canonical_tool_name(&call.name) == NativeTool::UseSkill.canonical_name()
    })
}

#[cfg(test)]
mod tests {
    use std::slice;

    use lithos_llm::catalog::{ModelId, ProviderId};
    use lithos_llm::types::{
        ContentBlockId, ContentBlockKind, ReasoningContent, TokenCounts, ToolCallKind,
    };

    use super::*;

    fn response_with(content: Vec<ContentPart>) -> Response {
        Response::new(ProviderId::new("test"), ModelId::new("model"), content)
    }

    #[test]
    fn a_block_start_names_the_first_output_kind() {
        let cases = [
            (ContentBlockKind::Text, Some(LlmOutputKind::Text)),
            (ContentBlockKind::Reasoning, Some(LlmOutputKind::Reasoning)),
            (
                ContentBlockKind::ToolCall {
                    id:   "call_1".to_owned(),
                    name: Some("shell".to_owned()),
                    kind: ToolCallKind::Function,
                },
                Some(LlmOutputKind::ToolCall),
            ),
            (
                ContentBlockKind::Opaque {
                    kind: "openai.reasoning".to_owned(),
                },
                None,
            ),
        ];
        for (kind, expected) in cases {
            let event = StreamEvent::ContentBlockStart {
                id: ContentBlockId::new("block-0"),
                kind,
            };
            assert_eq!(first_output_kind(&event), expected);
        }
    }

    #[test]
    fn deltas_name_the_first_output_kind() {
        let id = ContentBlockId::new("block-0");
        assert_eq!(
            first_output_kind(&StreamEvent::TextDelta {
                id:   id.clone(),
                text: "hi".to_owned(),
            }),
            Some(LlmOutputKind::Text)
        );
        assert_eq!(
            first_output_kind(&StreamEvent::ReasoningDelta {
                id:   id.clone(),
                text: "hmm".to_owned(),
            }),
            Some(LlmOutputKind::Reasoning)
        );
        assert_eq!(
            first_output_kind(&StreamEvent::ToolCallDelta {
                id,
                arguments: "{".to_owned(),
            }),
            Some(LlmOutputKind::ToolCall)
        );
    }

    #[test]
    fn protocol_events_are_not_output() {
        assert_eq!(
            first_output_kind(&StreamEvent::Started { id: None }),
            None,
            "answering is not producing"
        );
        assert_eq!(
            first_output_kind(&StreamEvent::Usage {
                usage: TokenCounts::default(),
            }),
            None
        );
        assert_eq!(
            first_output_kind(&StreamEvent::Completed {
                response: response_with(Vec::new()),
            }),
            None
        );
    }

    #[test]
    fn a_finished_block_names_what_it_carried() {
        let id = ContentBlockId::new("block-0");
        assert_eq!(
            first_output_kind(&StreamEvent::ContentBlockEnd {
                id:   id.clone(),
                part: ContentPart::Text {
                    text: "hello".to_owned(),
                },
            }),
            Some(LlmOutputKind::Text)
        );
        assert_eq!(
            first_output_kind(&StreamEvent::ContentBlockEnd {
                id,
                part: ContentPart::ToolCall(ToolCall::function(
                    "call_1",
                    "shell",
                    serde_json::json!({}),
                )),
            }),
            Some(LlmOutputKind::ToolCall)
        );
    }

    #[test]
    fn tool_calls_come_out_in_the_order_they_were_asked() {
        let response = response_with(vec![
            ContentPart::Text {
                text: "working".to_owned(),
            },
            ContentPart::ToolCall(ToolCall::function("call_1", "read", serde_json::json!({}))),
            ContentPart::ToolCall(ToolCall::function("call_2", "shell", serde_json::json!({}))),
        ]);

        let calls = tool_calls_of(&response);

        assert_eq!(
            calls
                .iter()
                .map(|call| call.id.as_str())
                .collect::<Vec<_>>(),
            ["call_1", "call_2"]
        );
    }

    #[test]
    fn only_provider_native_parts_are_kept_for_replay() {
        let response = response_with(vec![
            ContentPart::Reasoning(ReasoningContent {
                text:             "thinking".to_owned(),
                signature:        None,
                signature_origin: None,
                redacted:         false,
            }),
            ContentPart::Text {
                text: "answer".to_owned(),
            },
            ContentPart::Opaque {
                kind: "openai.reasoning".to_owned(),
                data: serde_json::json!({"id": "rs_1"}),
            },
            ContentPart::ToolCall(ToolCall::function("call_1", "read", serde_json::json!({}))),
        ]);

        let parts = provider_parts_of(&response);

        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], ContentPart::Reasoning(_)));
        assert!(matches!(parts[1], ContentPart::Opaque { .. }));
    }

    #[test]
    fn usage_sums_across_a_run() {
        let first = TokenUsage {
            input:       10,
            output:      5,
            reasoning:   2,
            cache_read:  1,
            cache_write: 0,
        };
        let second = TokenUsage {
            input:       3,
            output:      7,
            reasoning:   0,
            cache_read:  0,
            cache_write: 4,
        };

        assert_eq!(add_usage(first, second), TokenUsage {
            input:       13,
            output:      12,
            reasoning:   2,
            cache_read:  1,
            cache_write: 4,
        });
    }

    #[test]
    fn elapsed_time_is_taken_once() {
        let mut start = Some(Instant::now());
        let mut total = Duration::ZERO;

        record_elapsed(&mut start, &mut total);
        let after_first = total;
        record_elapsed(&mut start, &mut total);

        assert_eq!(total, after_first, "a taken span is not counted again");
        assert!(start.is_none());
    }

    #[test]
    fn a_successful_skill_call_counts_as_an_activation() {
        let call = ToolCall::function(
            "call_1",
            NativeTool::UseSkill.canonical_name(),
            serde_json::json!({}),
        );
        let ok = ToolResult {
            tool_call_id: "call_1".to_owned(),
            name:         None,
            content:      vec![ContentPart::Text {
                text: "loaded".to_owned(),
            }],
            is_error:     false,
        };
        let failed = ToolResult {
            is_error: true,
            ..ok.clone()
        };

        assert!(activated_a_skill(slice::from_ref(&call), &[ok]));
        assert!(!activated_a_skill(&[call], &[failed]));
    }
}
