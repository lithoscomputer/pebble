//! Thread-safe control of an active agent prompt.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::types::Message;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::UserMessage;

pub(crate) struct Control {
    state: Mutex<ControlState>,
    idle:  Notify,
}

struct ControlState {
    running:       bool,
    closed:        bool,
    steering:      VecDeque<Message>,
    follow_up:     VecDeque<Message>,
    prompt_cancel: CancellationToken,
    round_cancel:  CancellationToken,
}

impl Control {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(ControlState {
                running:       false,
                closed:        false,
                steering:      VecDeque::new(),
                follow_up:     VecDeque::new(),
                prompt_cancel: CancellationToken::new(),
                round_cancel:  CancellationToken::new(),
            }),
            idle:  Notify::new(),
        })
    }

    pub(crate) fn begin_prompt(&self) -> Option<CancellationToken> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return None;
        }
        debug_assert!(!state.running, "a mutable agent cannot start two prompts");
        state.running = true;
        state.prompt_cancel = CancellationToken::new();
        state.round_cancel = CancellationToken::new();
        Some(state.prompt_cancel.clone())
    }

    pub(crate) fn finish_prompt(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.running = false;
        state.round_cancel = CancellationToken::new();
        drop(state);
        self.idle.notify_waiters();
    }

    pub(crate) fn begin_round(&self) -> CancellationToken {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.round_cancel = CancellationToken::new();
        state.round_cancel.clone()
    }

    pub(crate) fn drain_steering(&self) -> Vec<Message> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .steering
            .drain(..)
            .collect()
    }

    pub(crate) fn pop_follow_up(&self) -> Option<Message> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .follow_up
            .pop_front()
    }

    pub(crate) fn clear_queues(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.steering.clear();
        state.follow_up.clear();
    }

    pub(crate) fn close(&self) -> bool {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return false;
        }
        state.closed = true;
        state.prompt_cancel.cancel();
        state.round_cancel.cancel();
        drop(state);
        self.idle.notify_waiters();
        true
    }

    pub(crate) fn is_running(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .running
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .closed
    }
}

/// Controls an agent while another task awaits
/// [`Agent::prompt`](crate::Agent::prompt).
#[derive(Clone)]
pub struct AgentControlHandle {
    control: Arc<Control>,
}

impl AgentControlHandle {
    pub(crate) fn new(control: Arc<Control>) -> Self {
        Self { control }
    }

    /// Queues steering for the next model turn and interrupts the current one.
    ///
    /// Returns `false` when the agent is closed. Steering queued while idle is
    /// applied after the next prompt's user message.
    pub fn steer(&self, message: impl Into<UserMessage>) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return false;
        }
        state.steering.push_back(message.into().into_message());
        if state.running {
            state.round_cancel.cancel();
        }
        true
    }

    /// Queues input to process after the current prompt reaches an answer.
    ///
    /// Returns `false` when the agent is closed.
    pub fn follow_up(&self, message: impl Into<UserMessage>) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return false;
        }
        state.follow_up.push_back(message.into().into_message());
        true
    }

    /// Aborts the active prompt.
    ///
    /// Returns whether a prompt was running.
    pub fn abort(&self) -> bool {
        let state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !state.running {
            return false;
        }
        state.prompt_cancel.cancel();
        state.round_cancel.cancel();
        true
    }

    /// Waits until no prompt is running.
    pub async fn wait_for_idle(&self) {
        loop {
            let notified = self.control.idle.notified();
            if !self.control.is_running() {
                return;
            }
            notified.await;
        }
    }

    /// Whether a prompt is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.control.is_running()
    }

    /// Whether the agent is closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.control.is_closed()
    }
}

impl fmt::Debug for AgentControlHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentControlHandle")
            .field("running", &self.is_running())
            .field("closed", &self.is_closed())
            .finish()
    }
}
