//! Reporting the retries the model client decides on its own.
//!
//! Two layers replay a failed model call, and they are separated by what the
//! reader has already seen. Before the provider produces visible output, the
//! client's [`RetryMiddleware`](lithos_llm::middleware::RetryMiddleware) owns
//! the replay: it reconnects underneath the session, which never learns that
//! anything went wrong. After visible output, only the session can replay,
//! because a replay has to withdraw what was shown first.
//!
//! Both halves publish the same
//! [`LlmRetry`](crate::events::CodingEvent::LlmRetry) event, so an application
//! sees one account of a struggling call. This module is the half that reaches
//! into the client: pebble puts a bridge into each call's context, and
//! [`RetryEventObserver`] — which the application installs on the middleware —
//! reads it back out.
//!
//! # Installing it
//!
//! The application builds the client, so the application installs the observer:
//!
//! ```no_run
//! use lithos_llm::Client;
//! use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
//! use pebble_coding_agent::events::RetryEventObserver;
//!
//! # fn build(catalog: lithos_llm::catalog::Catalog) -> Result<(), Box<dyn std::error::Error>> {
//! let policy = RetryPolicy::exponential().max_attempts(4);
//! let build = Client::builder()
//!     .catalog(catalog)
//!     .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
//!     .build()?;
//! # let _ = build;
//! # Ok(())
//! # }
//! ```
//!
//! [`CodingAgentOptions::turn_replay`](crate::CodingAgentOptions::turn_replay)
//! controls the separate replay after a response stream opens. A session whose
//! client has no retry middleware still runs correctly: it simply never
//! publishes an `open`-phase retry, because nothing retried.

use std::time::Duration;

use lithos_llm::middleware::{Call, Observer, RetryStage};
use lithos_llm::types::Error as LlmError;

use crate::error::ErrorData;
use crate::event::Emitter;
use crate::types::{CodingEvent, LlmRetryPhase};

/// Publishes the retries a client's retry middleware decides.
///
/// Install it on the middleware, once, when building the client. It reports
/// only for calls a pebble session made — it finds the session's event pipeline
/// on the call itself — so one client shared by several sessions sends each
/// session's retries to its tree's stream, and a call pebble did not make is
/// ignored.
#[derive(Clone, Copy, Debug, Default)]
pub struct RetryEventObserver;

impl Observer for RetryEventObserver {
    fn on_retry(
        &self,
        call: &Call,
        error: &LlmError,
        attempt: u32,
        delay: Duration,
        stage: RetryStage,
    ) {
        if let Some(bridge) = call.context.extensions().get::<RetryEventBridge>() {
            bridge.report(error, attempt, delay, stage);
        }
    }
}

/// What one session puts into a call so its retries find their way home.
///
/// The provider and model are copied in rather than read back from the route,
/// so the observer needs no catalog and reports the model the session asked
/// for even when a failure happened before one was resolved.
#[derive(Clone, Debug)]
pub(crate) struct RetryEventBridge {
    emitter:    Emitter,
    session_id: String,
    provider:   String,
    model:      String,
}

impl RetryEventBridge {
    /// Binds a bridge to one session and the model its round targets.
    pub(crate) fn new(
        emitter: Emitter,
        session_id: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            emitter,
            session_id: session_id.into(),
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Publishes one middleware retry.
    ///
    /// This runs inline inside the client while the session is parked, so it
    /// only queues: the session's event pump does the sequencing and the sink
    /// delivery.
    fn report(&self, error: &LlmError, attempt: u32, delay: Duration, stage: RetryStage) {
        self.emitter
            .emit(self.session_id.clone(), CodingEvent::LlmRetry {
                provider:   self.provider.clone(),
                model:      self.model.clone(),
                attempt:    attempt_index(attempt),
                delay_secs: delay.as_secs_f64(),
                error:      ErrorData::from(error),
                phase:      phase_for(stage),
            });
    }
}

/// The 0-based attempt number pebble publishes, from the client's 1-based one.
///
/// Pebble numbers attempts from zero on the wire: the first failure is attempt
/// 0, whichever layer reports it, so the session's own replays and the client's
/// read the same way.
fn attempt_index(attempt: u32) -> usize {
    usize::try_from(attempt.saturating_sub(1)).unwrap_or(usize::MAX)
}

/// Which retry loop the reported attempt belongs to.
///
/// A request the provider never answered with a stream is `open`; a stream that
/// failed while it was being read is `consume`, which is also what the
/// session's own post-visible replays report. A stage a later client adds is
/// reported as `consume`, because by then the call had got further than the
/// open.
fn phase_for(stage: RetryStage) -> LlmRetryPhase {
    match stage {
        RetryStage::Request => LlmRetryPhase::Open,
        // `Stream`, and any stage a later client adds: by then the call had
        // got further than the open.
        _ => LlmRetryPhase::Consume,
    }
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::ErrorKind as LlmErrorKind;
    use tokio::sync::broadcast::error::TryRecvError;

    use super::*;
    use crate::event::{EventOptions, EventPump};

    fn failure() -> LlmError {
        LlmError::new(LlmErrorKind::Network, "connection reset")
    }

    #[test]
    fn the_first_failure_is_attempt_zero() {
        assert_eq!(attempt_index(1), 0);
        assert_eq!(attempt_index(4), 3);
        assert_eq!(attempt_index(0), 0, "a client that counts from zero anyway");
    }

    #[test]
    fn a_request_failure_is_the_open_phase() {
        assert_eq!(phase_for(RetryStage::Request), LlmRetryPhase::Open);
        assert_eq!(phase_for(RetryStage::Stream), LlmRetryPhase::Consume);
    }

    #[tokio::test]
    async fn a_bridge_publishes_one_retry_on_its_session_stream() {
        let (emitter, pump) = EventPump::new(EventOptions::default());
        let mut events = emitter.subscribe();
        let pump = tokio::spawn(pump.run());
        let bridge = RetryEventBridge::new(emitter.clone(), "ses_1", "test", "model");

        bridge.report(
            &failure(),
            2,
            Duration::from_millis(250),
            RetryStage::Stream,
        );

        let event = events.recv().await.expect("the retry reaches the stream");
        assert_eq!(event.session_id, "ses_1");
        match event.event {
            CodingEvent::LlmRetry {
                provider,
                model,
                attempt,
                delay_secs,
                error,
                phase,
            } => {
                assert_eq!(provider, "test");
                assert_eq!(model, "model");
                assert_eq!(attempt, 1, "the client's attempt 2, numbered from zero");
                assert!((delay_secs - 0.25).abs() < f64::EPSILON);
                assert_eq!(error.message, "connection reset");
                assert_eq!(phase, LlmRetryPhase::Consume);
            }
            other => panic!("expected a retry event, got {other:?}"),
        }

        emitter.close().await.unwrap();
        pump.await
            .expect("the pump joins")
            .expect("no sink failure");
    }

    #[tokio::test]
    async fn an_observer_without_a_bridge_publishes_nothing() {
        let (emitter, pump) = EventPump::new(EventOptions::default());
        let mut events = emitter.subscribe();
        let pump = tokio::spawn(pump.run());

        // The observer reads its bridge from the call, so a call that carries
        // none — anything the application makes on the same client — is
        // ignored rather than reported onto a session's stream.
        emitter.emit("ses_1", CodingEvent::LoopDetected);

        assert!(matches!(
            events.recv().await.map(|event| event.event),
            Ok(CodingEvent::LoopDetected)
        ));
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));

        emitter.close().await.unwrap();
        pump.await
            .expect("the pump joins")
            .expect("no sink failure");
    }
}
