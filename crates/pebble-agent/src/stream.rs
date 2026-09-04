//! One streamed model turn, shared by generic and specialized agent loops.

use std::time::Duration;

use futures_util::StreamExt as _;
use lithos_llm::middleware::{CallContext, CancellationToken as CallCancellation, RetryPolicy};
use lithos_llm::types::{
    ContentBlockKind, ContentPart, Error as LlmError, ErrorKind as LlmErrorKind, FinishReason,
    Request, Response, ResponseStream, StreamEvent,
};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::event::FirstOutputKind;
use crate::model::ModelService;

const TRUNCATED_STREAM: &str = "the stream ended without completing the response";
const TRUNCATED_STREAM_EXHAUSTED: &str =
    "the stream ended without completing the response, after every replay";

/// Observes output and replay decisions from one streamed model turn.
///
/// Every method runs inline on the task consuming the stream. Implementations
/// must not block. The default methods ignore every observation.
pub trait StreamObserver {
    /// The first model output in a stream attempt arrived.
    fn first_output(&self, _kind: FirstOutputKind) {}

    /// Assistant text arrived.
    fn text_delta(&self, _delta: &str) {}

    /// Assistant reasoning arrived.
    fn reasoning_delta(&self, _delta: &str) {}

    /// Visible output from the failed attempt must be replaced.
    fn output_replaced(&self) {}

    /// A failed stream will be opened again.
    fn replay(&self, _failed_attempt: u32, _delay: Duration, _error: &LlmError) {}
}

/// The outcome of consuming one model turn.
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamOutcome {
    /// A complete response arrived.
    Completed(Box<Response>),
    /// The active prompt was aborted.
    Aborted,
    /// The current turn was interrupted so steering can be applied.
    Interrupted,
    /// The model call failed after its replay budget was spent.
    Failed(Box<LlmError>),
}

/// Opens and consumes one model turn, replaying failed response streams.
///
/// `make_context` is called once for each attempt. It can attach
/// application-specific extensions or observers while keeping provider and
/// credential behavior in the injected [`ModelService`]. `max_replays` is the
/// hard limit after the initial attempt, even when the policy allows more.
#[tracing::instrument(
    name = "model_stream",
    skip_all,
    fields(model = %request.model(), max_replays)
)]
pub async fn stream_response<F, O>(
    service: &dyn ModelService,
    request: Request,
    prompt_cancel: &CancellationToken,
    round_cancel: &CancellationToken,
    replay_policy: RetryPolicy,
    max_replays: u32,
    mut make_context: F,
    observer: &O,
) -> StreamOutcome
where
    F: FnMut() -> CallContext,
    O: StreamObserver + ?Sized,
{
    let mut failed_attempt = 0_u32;

    loop {
        let context = make_context();
        let call_cancel = context.cancellation().clone();
        let opening = service.stream(request.clone(), context);
        tokio::pin!(opening);
        let opened = tokio::select! {
            biased;
            () = prompt_cancel.cancelled() => return StreamOutcome::Aborted,
            () = round_cancel.cancelled() => return StreamOutcome::Interrupted,
            result = &mut opening => result,
        };
        let stream = match opened {
            Ok(stream) => stream,
            Err(error) => return StreamOutcome::Failed(Box::new(error)),
        };
        let attempt =
            consume_stream(stream, call_cancel, prompt_cancel, round_cancel, observer).await;

        let (error, visible_output, truncated) = match attempt {
            StreamAttempt::Completed(response) => return StreamOutcome::Completed(response),
            StreamAttempt::Aborted => return StreamOutcome::Aborted,
            StreamAttempt::Interrupted => return StreamOutcome::Interrupted,
            StreamAttempt::Truncated { visible_output } => (
                Box::new(LlmError::new(LlmErrorKind::StreamDecode, TRUNCATED_STREAM)),
                visible_output,
                true,
            ),
            StreamAttempt::Failed {
                error,
                visible_output,
            } => (error, visible_output, false),
        };

        failed_attempt = failed_attempt.saturating_add(1);
        if failed_attempt > max_replays {
            if visible_output {
                observer.output_replaced();
            }
            return StreamOutcome::Failed(if truncated {
                Box::new(LlmError::new(
                    LlmErrorKind::StreamDecode,
                    TRUNCATED_STREAM_EXHAUSTED,
                ))
            } else {
                error
            });
        }
        let delay = if truncated {
            Some(Duration::ZERO)
        } else {
            replay_policy.next_delay(failed_attempt, &error)
        };
        let Some(delay) = delay else {
            if visible_output {
                observer.output_replaced();
            }
            return StreamOutcome::Failed(error);
        };

        if visible_output {
            observer.output_replaced();
        }
        observer.replay(failed_attempt, delay, &error);
        warn!(
            failed_attempt,
            delay_seconds = delay.as_secs_f64(),
            error = %error,
            "model stream failed; replaying the turn"
        );

        if !delay.is_zero() {
            tokio::select! {
                biased;
                () = prompt_cancel.cancelled() => return StreamOutcome::Aborted,
                () = round_cancel.cancelled() => return StreamOutcome::Interrupted,
                () = sleep(delay) => {}
            }
        }
    }
}

async fn consume_stream<O>(
    mut stream: ResponseStream,
    call_cancel: CallCancellation,
    prompt_cancel: &CancellationToken,
    round_cancel: &CancellationToken,
    observer: &O,
) -> StreamAttempt
where
    O: StreamObserver + ?Sized,
{
    let mut visible_output = false;
    let mut first_output = false;

    loop {
        let next = tokio::select! {
            biased;
            () = prompt_cancel.cancelled() => {
                call_cancel.cancel();
                if visible_output {
                    observer.output_replaced();
                }
                return StreamAttempt::Aborted;
            }
            () = round_cancel.cancelled() => {
                call_cancel.cancel();
                if visible_output {
                    observer.output_replaced();
                }
                return StreamAttempt::Interrupted;
            }
            next = stream.next() => next,
        };

        let Some(item) = next else {
            call_cancel.cancel();
            return StreamAttempt::Truncated { visible_output };
        };

        let event = match item {
            Ok(event) => event,
            Err(error) => {
                call_cancel.cancel();
                return StreamAttempt::Failed {
                    error: Box::new(error),
                    visible_output,
                };
            }
        };

        if !first_output && let Some(kind) = first_output_kind(&event) {
            first_output = true;
            observer.first_output(kind);
        }
        match event {
            StreamEvent::TextDelta { text, .. } => {
                visible_output = true;
                observer.text_delta(&text);
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                visible_output = true;
                observer.reasoning_delta(&text);
            }
            StreamEvent::Completed { response }
                if response.finish_reason == FinishReason::Incomplete =>
            {
                call_cancel.cancel();
                return StreamAttempt::Truncated { visible_output };
            }
            StreamEvent::Completed { response } => {
                return StreamAttempt::Completed(Box::new(response));
            }
            _ => {}
        }
    }
}

enum StreamAttempt {
    Completed(Box<Response>),
    Aborted,
    Interrupted,
    Truncated {
        visible_output: bool,
    },
    Failed {
        error:          Box<LlmError>,
        visible_output: bool,
    },
}

fn first_output_kind(event: &StreamEvent) -> Option<FirstOutputKind> {
    match event {
        StreamEvent::ContentBlockStart { kind, .. } => match kind {
            ContentBlockKind::Text => Some(FirstOutputKind::Text),
            ContentBlockKind::Reasoning => Some(FirstOutputKind::Reasoning),
            ContentBlockKind::ToolCall { .. } => Some(FirstOutputKind::ToolCall),
            _ => None,
        },
        StreamEvent::TextDelta { .. } => Some(FirstOutputKind::Text),
        StreamEvent::ReasoningDelta { .. } => Some(FirstOutputKind::Reasoning),
        StreamEvent::ToolCallDelta { .. } => Some(FirstOutputKind::ToolCall),
        StreamEvent::ContentBlockEnd { part, .. } => match part {
            ContentPart::Text { .. } => Some(FirstOutputKind::Text),
            ContentPart::Reasoning(_) => Some(FirstOutputKind::Reasoning),
            ContentPart::ToolCall(_) => Some(FirstOutputKind::ToolCall),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::result::Result as StdResult;
    use std::sync::{Mutex, PoisonError};

    use async_trait::async_trait;
    use futures_util::stream;
    use lithos_llm::catalog::{ModelId, ProviderId};
    use lithos_llm::types::{ContentBlockId, ErrorKind, ResponseStream, RetryClassification};

    use super::*;

    struct ScriptedModel {
        streams: Mutex<VecDeque<Vec<StdResult<StreamEvent, LlmError>>>>,
    }

    #[async_trait]
    impl ModelService for ScriptedModel {
        async fn stream(
            &self,
            _request: Request,
            _context: CallContext,
        ) -> StdResult<ResponseStream, LlmError> {
            let events = self
                .streams
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop_front()
                .expect("the script has another stream");
            Ok(ResponseStream::new(stream::iter(events)))
        }
    }

    #[derive(Default)]
    struct Observed {
        replacements: Mutex<usize>,
        replays:      Mutex<usize>,
    }

    impl StreamObserver for Observed {
        fn output_replaced(&self) {
            *self
                .replacements
                .lock()
                .unwrap_or_else(PoisonError::into_inner) += 1;
        }

        fn replay(&self, _failed_attempt: u32, _delay: Duration, _error: &LlmError) {
            *self.replays.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        }
    }

    fn request() -> Request {
        Request::builder()
            .model("test/model")
            .user("work")
            .build()
            .expect("the request is valid")
    }

    fn response(text: &str) -> Response {
        Response::new(ProviderId::new("test"), ModelId::new("model"), vec![
            ContentPart::Text {
                text: text.to_owned(),
            },
        ])
    }

    #[tokio::test]
    async fn visible_failed_output_is_replaced_before_replay() {
        let id = ContentBlockId::new("text-1");
        let model = ScriptedModel {
            streams: Mutex::new(VecDeque::from([
                vec![
                    Ok(StreamEvent::TextDelta {
                        id,
                        text: "partial".to_owned(),
                    }),
                    Err(LlmError::new(ErrorKind::Network, "reset")
                        .with_retry(RetryClassification::Safe)),
                ],
                vec![Ok(StreamEvent::Completed {
                    response: response("done"),
                })],
            ])),
        };
        let observed = Observed::default();
        let policy = RetryPolicy::exponential()
            .max_attempts(2)
            .initial_delay(Duration::ZERO);

        let outcome = stream_response(
            &model,
            request(),
            &CancellationToken::new(),
            &CancellationToken::new(),
            policy,
            1,
            CallContext::new,
            &observed,
        )
        .await;

        assert!(matches!(outcome, StreamOutcome::Completed(response) if response.text() == "done"));
        assert_eq!(
            *observed
                .replacements
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
            1
        );
        assert_eq!(
            *observed
                .replays
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
            1
        );
    }
}
