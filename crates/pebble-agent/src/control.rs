//! Thread-safe control of an active agent prompt.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use lithos_llm::types::Message;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::UserMessage;

pub(crate) struct Control {
    state:  Mutex<ControlState>,
    idle:   Notify,
    resume: Notify,
}

struct ControlState {
    running:       bool,
    closed:        bool,
    paused:        bool,
    steering:      VecDeque<Message>,
    follow_up:     VecDeque<Message>,
    prompt_cancel: CancellationToken,
    round_cancel:  CancellationToken,
}

impl Control {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state:  Mutex::new(ControlState {
                running:       false,
                closed:        false,
                paused:        false,
                steering:      VecDeque::new(),
                follow_up:     VecDeque::new(),
                prompt_cancel: CancellationToken::new(),
                round_cancel:  CancellationToken::new(),
            }),
            idle:   Notify::new(),
            resume: Notify::new(),
        })
    }

    pub(crate) fn begin_prompt(
        &self,
        parent_cancel: Option<&CancellationToken>,
    ) -> Option<CancellationToken> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return None;
        }
        debug_assert!(!state.running, "a mutable agent cannot start two prompts");
        state.running = true;
        state.paused = false;
        state.prompt_cancel =
            parent_cancel.map_or_else(CancellationToken::new, CancellationToken::child_token);
        state.round_cancel = CancellationToken::new();
        Some(state.prompt_cancel.clone())
    }

    pub(crate) fn finish_prompt(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.running = false;
        state.paused = false;
        state.round_cancel = CancellationToken::new();
        drop(state);
        self.idle.notify_waiters();
        self.resume.notify_waiters();
    }

    pub(crate) fn begin_round(&self) -> (CancellationToken, Vec<Message>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.round_cancel = CancellationToken::new();
        let cancel = state.round_cancel.clone();
        let steering = state.steering.drain(..).collect();
        (cancel, steering)
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
        state.paused = false;
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
        self.resume.notify_waiters();
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

    pub(crate) fn is_paused(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .paused
    }

    pub(crate) async fn wait_until_resumed(&self, cancel: &CancellationToken) -> bool {
        loop {
            let notified = self.resume.notified();
            if !self.is_paused() {
                return true;
            }
            tokio::select! {
                () = cancel.cancelled() => return false,
                () = notified => {}
            }
        }
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
        self.queue_steering(message.into(), true)
    }

    /// Queues steering for the next model turn without interrupting this one.
    ///
    /// Returns `false` when the agent is closed.
    pub fn enqueue_steering(&self, message: impl Into<UserMessage>) -> bool {
        self.queue_steering(message.into(), false)
    }

    fn queue_steering(&self, message: UserMessage, interrupt: bool) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return false;
        }
        state.steering.push_back(message.into_message());
        state.paused = false;
        if interrupt && state.running {
            state.round_cancel.cancel();
        }
        drop(state);
        self.control.resume.notify_waiters();
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
        drop(state);
        self.control.resume.notify_waiters();
        true
    }

    /// Interrupts the current turn and waits for steering before another.
    ///
    /// Returns whether a prompt was running.
    pub fn interrupt(&self) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !state.running {
            return false;
        }
        state.paused = true;
        state.round_cancel.cancel();
        true
    }

    /// Claims the next turn boundary for steering without interrupting now.
    ///
    /// Returns whether a prompt was running.
    pub fn park_for_steer(&self) -> bool {
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if !state.running {
            return false;
        }
        state.paused = true;
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
