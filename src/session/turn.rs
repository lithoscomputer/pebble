//! The turn loop: one input, however many rounds it takes.
//!
//! A round is one exchange with the model. The session drains whatever was
//! steered in, builds a request, reads the response as it streams, commits the
//! turn, runs the tools it asked for, and goes round again. It stops when the
//! model answers with no tool calls, when something cancels the prompt, or when
//! a model call fails for good.
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

use lithos_llm::middleware::CallContext;
use lithos_llm::types::{
    ContentPart, Error as LlmError, Request, Response, ToolCall, ToolChoice, ToolResult,
};
use pebble_agent::FirstOutputKind;
use pebble_agent::advanced::{StreamObserver, StreamOutcome, stream_response};
use tokio_util::sync::CancellationToken;

use super::control::SteeringItem;
use super::retry::RetryEventBridge;
use super::{PromptTotals, Session};
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
use crate::tool::{NativeTool, ToolDefinitionWithSource, ToolDispatch, canonical_tool_name};
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

/// Takes the promptning span out of `start` and adds it to `total`.
fn record_elapsed(start: &mut Option<Instant>, total: &mut Duration) {
    if let Some(started) = start.take() {
        *total = total.saturating_add(started.elapsed());
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
    pub(super) async fn process_input(
        &mut self,
        input: &str,
        skill_expansion: SkillExpansion,
        human_input: Option<&Arc<dyn HumanInputProvider>>,
        totals: &mut PromptTotals,
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
        // attempts stops a provider returning nothing from turning into a prompt
        // of paid calls; the next input starts fresh.
        let mut compaction_failed = false;

        loop {
            // A refusing sink has already stopped the pipeline, so the prompt
            // ends here rather than working on with nothing recording it.
            self.check_pump().await?;

            self.refresh_round_token();

            // Ending the prompt beats a park: a session waiting for a steer that
            // was also cancelled is cancelled.
            if self.cancel_token.is_cancelled() {
                return Err(self.close_cancelled().await);
            }

            // Every round, whether or not this one looks interrupted: a
            // gesture that raised its generation and then cancelled the token
            // this loop was already replacing leaves an announcement owing on a
            // round that ended normally, and it is owed now rather than at the
            // next interrupt.
            self.settle_interrupts();

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

            // The policy filter runs once per round; the request builder takes
            // the only per-round copy of the definitions.
            let tools = self.effective_tools();

            // Staged, not committed: an interrupted round must not leave a
            // system message behind for later steering to answer. It commits
            // only alongside the assistant turn that read it.
            let pending_task_reminder = self.task_reminder_if_needed(&tools);

            let request = self.build_request(&tools, pending_task_reminder.as_ref())?;
            let local_context_window = self.measure_request(&request, &tools);

            // The last moment at which nothing is known about the response.
            // One per round, however many times the turn is replayed inside
            // it.
            self.emit(AgentEvent::LlmRequestStarted {
                requested_model: self.model.clone(),
            });

            let mut inference_start = Some(Instant::now());
            let bridge_emitter = self.emitter.clone();
            let bridge_session_id = self.id.clone();
            let bridge_provider = self.provider.clone();
            let bridge_model = self.model.clone();
            let observer = SessionStreamObserver { session: self };
            let outcome = stream_response(
                &self.client,
                request,
                &self.cancel_token,
                &round_token,
                self.config.turn_replay,
                u32::try_from(STREAM_CONSUME_RETRIES).unwrap_or(u32::MAX),
                move || {
                    let mut context = CallContext::new();
                    context.extensions_mut().insert(RetryEventBridge::new(
                        bridge_emitter.clone(),
                        bridge_session_id.clone(),
                        bridge_provider.clone(),
                        bridge_model.clone(),
                    ));
                    context
                },
                &observer,
            )
            .await;
            record_elapsed(&mut inference_start, &mut totals.timing.inference);

            let response = match outcome {
                StreamOutcome::Completed(response) => *response,
                StreamOutcome::Aborted => return Err(self.close_cancelled().await),
                StreamOutcome::Interrupted => continue,
                StreamOutcome::Failed(error) => return Err(self.emit_llm_error(*error)),
                _ => {
                    return Err(Error::InvalidState(
                        "the agent stream returned an unknown outcome".to_owned(),
                    ));
                }
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
                        // rather than ending the prompt on it.
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

            // Tools watch one token covering both the end of the prompt and the
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
    fn refresh_round_token(&self) {
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
    /// Only ending the prompt wakes this with a failure. The round token means
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
    fn task_reminder_if_needed(&self, tools: &[ToolDefinitionWithSource]) -> Option<Message> {
        let names: Vec<&str> = tools
            .iter()
            .map(|tool| tool.definition.name.as_str())
            .collect();
        maybe_task_reminder(&self.history, &names).map(|content| Message::System {
            content,
            timestamp: SystemTime::now(),
        })
    }

    /// Builds this round's request.
    ///
    /// A staged reminder goes last, through the same conversion durable turns
    /// use, so what the model reads now is what history will hold if the turn
    /// commits. Deterministic within a round — nothing it reads changes until
    /// the turn commits — which is what lets a replay rebuild the request
    /// instead of every round paying to clone one it will probably never
    /// need again.
    fn build_request(
        &self,
        tools: &[ToolDefinitionWithSource],
        pending_task_reminder: Option<&Message>,
    ) -> Result<Request> {
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

        for tool in tools {
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

        builder.build().map_err(|error| {
            Error::InvalidState(format!("this round's request could not be built: {error}"))
        })
    }

    /// Measures what this round's request will cost to send.
    fn measure_request(
        &self,
        request: &Request,
        tools: &[ToolDefinitionWithSource],
    ) -> ContextWindowSnapshot {
        build_local_snapshot(ContextWindowInput {
            request,
            tools,
            system_prompt: &self.system_prompt,
            memory_tokens: self.memory_tokens,
            skills_tokens: self.skills_tokens,
            activated_skill_context_observed: self.activated_skill_context_observed,
            provider: &self.provider,
            model: &self.model,
            context_window_tokens: self.facts.context_window_tokens,
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
        totals: &mut PromptTotals,
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

        totals.usage = totals.usage.saturating_add(usage);
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
    /// either the round or the prompt ends.
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

/// Projects the shared model-turn engine onto Pebble's durable coding events.
struct SessionStreamObserver<'a> {
    session: &'a Session,
}

impl StreamObserver for SessionStreamObserver<'_> {
    fn first_output(&self, kind: FirstOutputKind) {
        let kind = match kind {
            FirstOutputKind::Text => LlmOutputKind::Text,
            FirstOutputKind::Reasoning => LlmOutputKind::Reasoning,
            FirstOutputKind::ToolCall => LlmOutputKind::ToolCall,
            _ => return,
        };
        self.session.emit(AgentEvent::LlmFirstOutput { kind });
    }

    fn text_delta(&self, delta: &str) {
        self.session.emit(AgentEvent::TextDelta {
            delta: delta.to_owned(),
        });
    }

    fn reasoning_delta(&self, delta: &str) {
        self.session.emit(AgentEvent::ReasoningDelta {
            delta: delta.to_owned(),
        });
    }

    fn output_replaced(&self) {
        self.session.replace_visible_output();
    }

    fn replay(&self, failed_attempt: u32, delay: Duration, error: &LlmError) {
        self.session.emit(AgentEvent::LlmRetry {
            provider:   self.session.provider.clone(),
            model:      self.session.model.clone(),
            attempt:    usize::try_from(failed_attempt.saturating_sub(1)).unwrap_or(usize::MAX),
            delay_secs: delay.as_secs_f64(),
            error:      ErrorData::from(error),
            phase:      LlmRetryPhase::Consume,
        });
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
    use lithos_llm::types::ReasoningContent;

    use super::*;

    fn response_with(content: Vec<ContentPart>) -> Response {
        Response::new(ProviderId::new("test"), ModelId::new("model"), content)
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
    fn usage_sums_across_a_prompt() {
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

        assert_eq!(first.saturating_add(second), TokenUsage {
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
