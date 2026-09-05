//! Human questions and per-call approvals sent to the terminal's input owner.

use async_trait::async_trait;
use pebble_agent::{ToolCallRequest, ToolSystemError};
use pebble_coding_agent::extensions::{
    Answer, AnswerStatus, HumanInputError, HumanInputProvider, Question,
};
use pebble_coding_agent::tools::{ApprovalDecision, ToolApprovalService};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(super) enum Request {
    Questions {
        questions: Vec<Question>,
        reply:     oneshot::Sender<Vec<Answer>>,
        cancel:    CancellationToken,
    },
    Approval {
        details: String,
        reply:   oneshot::Sender<bool>,
        cancel:  CancellationToken,
    },
}

pub(super) struct Services {
    sender: mpsc::Sender<Request>,
}

impl Services {
    pub(super) fn channel() -> (Self, mpsc::Receiver<Request>) {
        let (sender, receiver) = mpsc::channel(8);
        (Self { sender }, receiver)
    }
}

#[async_trait]
impl HumanInputProvider for Services {
    async fn ask_questions(
        &self,
        _tool_call_id: &str,
        questions: Vec<Question>,
        cancel: CancellationToken,
    ) -> Result<Vec<Answer>, HumanInputError> {
        let unanswered = || {
            questions
                .iter()
                .map(|question| Answer::unanswered(question, AnswerStatus::Cancelled))
                .collect()
        };
        let (reply, answer) = oneshot::channel();
        let request = Request::Questions {
            questions: questions.clone(),
            reply,
            cancel: cancel.clone(),
        };
        tokio::select! {
            () = cancel.cancelled() => return Ok(unanswered()),
            result = self.sender.send(request) => result.map_err(|error| HumanInputError::with_source("the terminal question channel closed", error))?,
        }
        tokio::select! {
            () = cancel.cancelled() => Ok(unanswered()),
            result = answer => result.map_err(|error| HumanInputError::with_source("the terminal question was closed", error)),
        }
    }
}

#[async_trait]
impl ToolApprovalService for Services {
    async fn approve(
        &self,
        request: &ToolCallRequest,
    ) -> Result<ApprovalDecision, ToolSystemError> {
        let denied = || ApprovalDecision::Deny {
            reason: "The user did not approve this tool call.".into(),
        };
        let Ok(details) = serde_json::to_string_pretty(request.call()) else {
            return Ok(denied());
        };
        let (reply, answer) = oneshot::channel();
        let prompt = Request::Approval {
            details,
            reply,
            cancel: request.cancellation().clone(),
        };
        tokio::select! {
            () = request.cancellation().cancelled() => return Ok(denied()),
            result = self.sender.send(prompt) => if result.is_err() { return Ok(denied()); },
        }
        tokio::select! {
            () = request.cancellation().cancelled() => Ok(denied()),
            result = answer => Ok(if result.unwrap_or(false) { ApprovalDecision::Allow } else { denied() }),
        }
    }
}
