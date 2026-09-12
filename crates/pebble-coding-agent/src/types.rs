//! The vocabulary a pebble session speaks: conversation turns, session state,
//! and the event stream an application observes.
//!
//! The serialized form of [`CodingAgentEvent`] and [`CodingEvent`] is public
//! API. Evolution is additive: new variants and new optional fields only.
//! Consumers should ignore unknown fields and tolerate unknown variants.
//!
//! Pebble builds these values and an application reads them, so the additive
//! promise covers reading. Building an event payload from outside the crate is
//! not covered: see the crate-level stability notes.

mod actor;
mod context_window;
mod llm;
mod process;
mod todo;
mod tool;

use std::fmt;
use std::ops::Deref;
use std::time::SystemTime;

use lithos_llm::types::{
    ContentPart, Message as LlmMessage, ReasoningOutput, Role, ToolCall, ToolResult,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use self::actor::Actor;
pub use self::context_window::{
    ContextWindowBreakdownItem, ContextWindowCategory, ContextWindowCountMethod,
    ContextWindowSnapshot, ContextWindowStaleness, ContextWindowWarning,
};
pub use self::llm::{CostSource, LlmOutputKind, LlmRetryPhase, TokenUsage};
pub use self::process::{CommandTermination, ExecOutputTail, ExecOutputTailTrace};
pub use self::todo::{
    TodoCreatedProps, TodoDeletedProps, TodoListKind, TodoListProjection, TodoProjection,
    TodoStatus, TodoUpdatedProps,
};
pub use self::tool::{PermissionLevel, ToolCategory, ToolErrorKind, ToolSource, ToolSummary};
use crate::compaction::{CompactionReason, CompactionResult, is_threshold_reason};
use crate::error::ErrorData;
use crate::record::StoredMessage;

/// The concatenated text of one message's `Text` content parts.
#[cfg(any(test, feature = "test-util"))]
pub(crate) fn message_text(message: &LlmMessage) -> String {
    InputContent::from(message.content())
        .text_content()
        .to_owned()
}

/// Provider-neutral content supplied by a person or another input source.
///
/// The readable text is cached for renderers and skill expansion. The full
/// ordered content remains available for model replay, including images,
/// audio, and documents.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputContent {
    parts: Vec<ContentPart>,
    text:  String,
}

impl InputContent {
    /// Creates input from provider-neutral content parts.
    #[must_use]
    pub fn new(parts: impl IntoIterator<Item = ContentPart>) -> Self {
        let parts: Vec<_> = parts.into_iter().collect();
        let text = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        Self { parts, text }
    }

    /// Creates plain text input.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::new([ContentPart::Text { text: text.into() }])
    }

    /// The ordered content sent to the model.
    #[must_use]
    pub fn parts(&self) -> &[ContentPart] {
        &self.parts
    }

    /// The concatenated readable text parts.
    #[must_use]
    pub fn text_content(&self) -> &str {
        &self.text
    }

    /// Consumes the value and returns its ordered content parts.
    #[must_use]
    pub fn into_parts(self) -> Vec<ContentPart> {
        self.parts
    }

    /// Rich content for an additive event field.
    ///
    /// Plain text is already present in the event's stable `text` member, so
    /// it does not need to be repeated.
    pub(crate) fn event_content(&self) -> Option<Self> {
        let plain = matches!(
            self.parts.as_slice(),
            [ContentPart::Text { text }] if text == &self.text
        );
        (!plain).then(|| self.clone())
    }
}

impl fmt::Display for InputContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.text_content())
    }
}

/// Treats string operations as operations on the readable text projection.
impl Deref for InputContent {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.text_content()
    }
}

impl PartialEq<str> for InputContent {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

impl PartialEq<&str> for InputContent {
    fn eq(&self, other: &&str) -> bool {
        self.text == *other
    }
}

impl PartialEq<String> for InputContent {
    fn eq(&self, other: &String) -> bool {
        self.text == *other
    }
}

impl From<String> for InputContent {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for InputContent {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<Vec<ContentPart>> for InputContent {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::new(parts)
    }
}

impl From<&[ContentPart]> for InputContent {
    fn from(parts: &[ContentPart]) -> Self {
        Self::new(parts.iter().cloned())
    }
}

impl Serialize for InputContent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.parts.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for InputContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StoredInputContent {
            Text(String),
            Parts(Vec<ContentPart>),
        }

        match StoredInputContent::deserialize(deserializer)? {
            StoredInputContent::Text(text) => Ok(Self::text(text)),
            StoredInputContent::Parts(parts) => Ok(Self::new(parts)),
        }
    }
}

/// Where an ordinary user-role turn came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InputSource {
    /// Input submitted through the main prompt operation.
    #[default]
    Prompt,
    /// Input queued to run after an answer.
    FollowUp,
    /// Input synthesized by the agent runtime.
    Agent,
    /// Input submitted by another application integration.
    External,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's skip_serializing_if callback receives a reference"
)]
pub(crate) const fn is_prompt_source(source: &InputSource) -> bool {
    matches!(source, InputSource::Prompt)
}

/// The generation of a subagent's first turn.
pub const INITIAL_SUBAGENT_GENERATION: u64 = 1;

/// The serde default for a subagent generation.
///
/// Events recorded before subagent session reuse existed carry no generation,
/// so they read back as [`INITIAL_SUBAGENT_GENERATION`].
pub(crate) const fn initial_subagent_generation() -> u64 {
    INITIAL_SUBAGENT_GENERATION
}

/// RFC 3339 timestamps with millisecond precision and a `Z` suffix.
///
/// Pebble represents every timestamp as a [`SystemTime`] and serializes it
/// through this module, so one format covers the event stream and the stored
/// record. Deserialization accepts any RFC 3339 offset and normalizes it to
/// UTC.
pub(crate) mod rfc3339_millis {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde::de::Error as DeError;
    use serde::{Deserialize, Deserializer, Serializer};

    const SECONDS_PER_DAY: i64 = 86_400;
    const DAYS_PER_ERA: i64 = 146_097;
    /// Days from 0000-03-01 to 1970-01-01, the shift the civil-date algorithm
    /// uses to put its year boundary on 1 March.
    const EPOCH_SHIFT_DAYS: i64 = 719_468;

    pub(crate) fn serialize<S: Serializer>(
        time: &SystemTime,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format(*time))
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<SystemTime, D::Error> {
        let text = String::deserialize(deserializer)?;
        parse(&text).map_err(DeError::custom)
    }

    /// Renders a moment as `YYYY-MM-DDTHH:MM:SS.mmmZ`, truncating toward the
    /// past below a millisecond.
    pub(crate) fn format(time: SystemTime) -> String {
        let (seconds, millis) = unix_parts(time);
        let (year, month, day) = civil_from_days(seconds.div_euclid(SECONDS_PER_DAY));
        let second_of_day = seconds.rem_euclid(SECONDS_PER_DAY);
        let hour = second_of_day / 3_600;
        let minute = (second_of_day % 3_600) / 60;
        let second = second_of_day % 60;
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
    }

    /// Parses an RFC 3339 timestamp with any offset.
    pub(crate) fn parse(text: &str) -> Result<SystemTime, String> {
        let bytes = text.as_bytes();
        let invalid = || format!("`{text}` is not an RFC 3339 timestamp");

        let year = digits(bytes, 0, 4).ok_or_else(invalid)?;
        let month = digits(bytes, 5, 2).ok_or_else(invalid)?;
        let day = digits(bytes, 8, 2).ok_or_else(invalid)?;
        let hour = digits(bytes, 11, 2).ok_or_else(invalid)?;
        let minute = digits(bytes, 14, 2).ok_or_else(invalid)?;
        let second = digits(bytes, 17, 2).ok_or_else(invalid)?;
        for (index, separator) in [(4, b'-'), (7, b'-'), (13, b':'), (16, b':')] {
            if bytes.get(index) != Some(&separator) {
                return Err(invalid());
            }
        }
        if !matches!(bytes.get(10), Some(b'T' | b't' | b' ')) {
            return Err(invalid());
        }
        if !(1..=12).contains(&month)
            || !(1..=days_in_month(year, month)).contains(&day)
            || hour > 23
            || minute > 59
            || second > 60
        {
            return Err(invalid());
        }

        let (nanos, rest) = fraction(&text[19..]).ok_or_else(invalid)?;
        let offset_seconds = offset(rest).ok_or_else(invalid)?;

        let seconds = days_from_civil(year, month, day) * SECONDS_PER_DAY
            + hour * 3_600
            + minute * 60
            + second.min(59)
            - offset_seconds;
        moment(seconds, nanos).ok_or_else(invalid)
    }

    /// Reads exactly `len` ASCII digits at `start`.
    fn digits(bytes: &[u8], start: usize, len: usize) -> Option<i64> {
        let mut value: i64 = 0;
        for offset in 0..len {
            let byte = *bytes.get(start + offset)?;
            if !byte.is_ascii_digit() {
                return None;
            }
            value = value * 10 + i64::from(byte - b'0');
        }
        Some(value)
    }

    /// Reads an optional `.fraction`, returning its nanoseconds and the rest.
    ///
    /// The fraction ends at the first byte that is not a digit, which is where
    /// the offset begins. A fraction shorter than nine digits is scaled up, and
    /// anything finer than a nanosecond is discarded.
    fn fraction(text: &str) -> Option<(u32, &str)> {
        let Some(rest) = text.strip_prefix('.') else {
            return Some((0, text));
        };
        let width = rest.bytes().take_while(u8::is_ascii_digit).count();
        if width == 0 {
            return None;
        }
        let significant = rest.as_bytes().get(..width.min(9))?;
        let mut nanos: u32 = 0;
        for byte in significant {
            nanos = nanos * 10 + u32::from(byte - b'0');
        }
        for _ in significant.len()..9 {
            nanos *= 10;
        }
        Some((nanos, &rest[width..]))
    }

    /// Reads the trailing UTC offset, in seconds east of UTC.
    fn offset(text: &str) -> Option<i64> {
        if matches!(text, "Z" | "z") {
            return Some(0);
        }
        let (sign, rest) = match text.as_bytes().first()? {
            b'+' => (1, &text[1..]),
            b'-' => (-1, &text[1..]),
            _ => return None,
        };
        let bytes = rest.as_bytes();
        if bytes.len() != 5 || bytes.get(2) != Some(&b':') {
            return None;
        }
        let hours = digits(bytes, 0, 2)?;
        let minutes = digits(bytes, 3, 2)?;
        if hours > 23 || minutes > 59 {
            return None;
        }
        Some(sign * (hours * 3_600 + minutes * 60))
    }

    /// Splits a moment into whole seconds since the epoch and its millisecond
    /// remainder, which is always non-negative.
    fn unix_parts(time: SystemTime) -> (i64, u32) {
        match time.duration_since(UNIX_EPOCH) {
            Ok(after) => (whole_seconds(after), after.subsec_millis()),
            Err(before) => {
                let before = before.duration();
                let seconds = whole_seconds(before);
                // Before the epoch the remainder counts backwards, so
                // truncating toward the past rounds it up rather than down.
                match before.subsec_nanos().div_ceil(1_000_000) {
                    0 => (-seconds, 0),
                    millis => (-seconds - 1, 1_000 - millis),
                }
            }
        }
    }

    fn whole_seconds(duration: Duration) -> i64 {
        i64::try_from(duration.as_secs()).unwrap_or(i64::MAX)
    }

    fn moment(seconds: i64, nanos: u32) -> Option<SystemTime> {
        let whole = Duration::from_secs(seconds.unsigned_abs());
        let base = if seconds < 0 {
            UNIX_EPOCH.checked_sub(whole)
        } else {
            UNIX_EPOCH.checked_add(whole)
        };
        base?.checked_add(Duration::from_nanos(u64::from(nanos)))
    }

    /// Converts days since the epoch into a civil year, month, and day.
    ///
    /// Howard Hinnant's `civil_from_days`, which is exact for every day the
    /// proleptic Gregorian calendar covers.
    fn civil_from_days(days: i64) -> (i64, i64, i64) {
        let shifted = days + EPOCH_SHIFT_DAYS;
        let era = shifted.div_euclid(DAYS_PER_ERA);
        let day_of_era = shifted - era * DAYS_PER_ERA;
        let year_of_era =
            (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let year = year_of_era + era * 400;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let shifted_month = (5 * day_of_year + 2) / 153;
        let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
        let month = if shifted_month < 10 {
            shifted_month + 3
        } else {
            shifted_month - 9
        };
        (if month <= 2 { year + 1 } else { year }, month, day)
    }

    /// How many days `month` has in `year`.
    ///
    /// Nonexistent civil dates are rejected rather than reinterpreted, so a
    /// corrupted timestamp fails loudly instead of naming a different instant.
    fn days_in_month(year: i64, month: i64) -> i64 {
        match month {
            4 | 6 | 9 | 11 => 30,
            2 if is_leap_year(year) => 29,
            2 => 28,
            _ => 31,
        }
    }

    /// Whether `year` is a leap year in the proleptic Gregorian calendar.
    const fn is_leap_year(year: i64) -> bool {
        year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
    }

    /// Converts a civil year, month, and day into days since the epoch.
    fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
        let year = if month <= 2 { year - 1 } else { year };
        let era = year.div_euclid(400);
        let year_of_era = year - era * 400;
        let shifted_month = if month > 2 { month - 3 } else { month + 9 };
        let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        era * DAYS_PER_ERA + day_of_era - EPOCH_SHIFT_DAYS
    }
}

/// One turn of durable conversation history.
///
/// History keeps its own turn enum rather than the wire form, because it
/// carries usage, timestamps, and the steering distinction that the wire form
/// drops. [`Message::to_llm_message`] is the single conversion to the wire
/// form, shared by durable history and staged turns so a staged turn produces
/// the same shape it will have once committed.
///
/// Non-exhaustive, like the [`StoredMessage`] it mirrors: a turn kind added to
/// the record has to be readable as a `Message` too, so the two grow together.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Message {
    /// Input from the person or system driving the session.
    User {
        /// The input content.
        content:   InputContent,
        /// When the turn was recorded.
        timestamp: SystemTime,
    },
    /// A committed assistant turn.
    Assistant {
        /// The assistant's text.
        content:        String,
        /// The tool calls the turn requested.
        tool_calls:     Vec<ToolCall>,
        /// Provider-native parts preserved for lossless replay: reasoning
        /// blocks with their signatures, and opaque provider items.
        provider_parts: Vec<ContentPart>,
        /// The token accounting the provider reported for this turn.
        usage:          TokenUsage,
        /// The provider's identifier for the response.
        response_id:    String,
        /// When the turn was committed.
        timestamp:      SystemTime,
    },
    /// The results of the tool calls a preceding assistant turn requested.
    ToolResults {
        /// One result per tool call, in call order.
        results:   Vec<ToolResult>,
        /// When the results were recorded.
        timestamp: SystemTime,
    },
    /// Injected content sent to the model with the system role.
    System {
        /// The injected text.
        content:   String,
        /// When the turn was recorded.
        timestamp: SystemTime,
    },
    /// A handoff summary that replaced older conversation turns.
    Compaction {
        /// The model-visible summary.
        summary:                 String,
        /// Why the compaction ran.
        reason:                  CompactionReason,
        /// Turns present before compaction.
        original_turn_count:     usize,
        /// Turns preserved verbatim.
        preserved_turn_count:    usize,
        /// Estimated context tokens before compaction.
        estimated_tokens_before: usize,
        /// Estimated tokens in the summary.
        summary_token_estimate:  usize,
        /// Files represented in the compaction prompt.
        tracked_file_count:      usize,
        /// Whether Pebble truncated the generated summary.
        summary_truncated:       bool,
        /// Usage from the summarization call.
        usage:                   TokenUsage,
        /// Cost of the summarization call in USD micros.
        cost_usd_micros:         Option<u64>,
        /// When the summary was recorded.
        timestamp:               SystemTime,
    },
    /// Injected steering sent to the model with the user role.
    ///
    /// Steering guides the assistant mid-conversation without appearing as
    /// actual user input.
    Steering {
        /// The steering content.
        content:   InputContent,
        /// When the turn was recorded.
        timestamp: SystemTime,
    },
}

impl Message {
    /// Builds the durable turn for a completed compaction.
    pub(crate) fn from_compaction(result: &CompactionResult) -> Self {
        Self::Compaction {
            summary:                 result.summary().to_owned(),
            reason:                  result.reason(),
            original_turn_count:     result.original_turn_count(),
            preserved_turn_count:    result.preserved_turn_count(),
            estimated_tokens_before: result.estimated_tokens_before(),
            summary_token_estimate:  result.summary_token_estimate(),
            tracked_file_count:      result.tracked_file_count(),
            summary_truncated:       result.summary_was_truncated(),
            usage:                   result.usage(),
            cost_usd_micros:         result.cost_usd_micros(),
            timestamp:               SystemTime::now(),
        }
    }

    /// When this turn was recorded.
    #[must_use]
    pub fn timestamp(&self) -> SystemTime {
        match self {
            Self::User { timestamp, .. }
            | Self::Assistant { timestamp, .. }
            | Self::ToolResults { timestamp, .. }
            | Self::System { timestamp, .. }
            | Self::Compaction { timestamp, .. }
            | Self::Steering { timestamp, .. } => *timestamp,
        }
    }

    /// The first readable reasoning text an assistant turn preserved, if any.
    #[must_use]
    pub fn reasoning_text(&self) -> Option<&str> {
        let Self::Assistant { provider_parts, .. } = self else {
            return None;
        };
        provider_parts.iter().find_map(|part| match part {
            ContentPart::Reasoning(reasoning) if !reasoning.redacted => {
                Some(reasoning.text.as_str())
            }
            _ => None,
        })
    }

    /// Converts this turn into the wire message sent to the provider.
    #[must_use]
    pub fn to_llm_message(&self) -> LlmMessage {
        match self {
            Self::User { content, .. } => {
                LlmMessage::new(Role::User, content.parts().iter().cloned())
            }
            Self::Assistant {
                content,
                tool_calls,
                provider_parts,
                ..
            } => {
                // Provider-native parts must precede tool calls, or a replayed
                // turn does not round-trip.
                let mut parts: Vec<ContentPart> = provider_parts.clone();
                if !content.is_empty() {
                    parts.push(ContentPart::Text {
                        text: content.clone(),
                    });
                }
                parts.extend(tool_calls.iter().cloned().map(ContentPart::ToolCall));
                LlmMessage::new(Role::Assistant, parts)
            }
            Self::ToolResults { results, .. } => {
                let parts = results.iter().cloned().map(ContentPart::ToolResult);
                let message = LlmMessage::new(Role::Tool, parts);
                match results.first() {
                    Some(first) => message.with_tool_call_id(first.tool_call_id.clone()),
                    None => message,
                }
            }
            Self::System { content, .. } => LlmMessage::text(Role::System, content.clone()),
            Self::Compaction { summary, .. } => LlmMessage::text(Role::System, summary.clone()),
            #[expect(
                clippy::match_same_arms,
                reason = "steering is its own turn kind that happens to share the user role"
            )]
            Self::Steering { content, .. } => {
                LlmMessage::new(Role::User, content.parts().iter().cloned())
            }
        }
    }

    /// Converts this turn into its stored form.
    ///
    /// The conversion itself keeps every fact, and
    /// [`Message::from_stored_message`] returns the turn it was given.
    /// Serializing the record is where precision is lost: timestamps are
    /// written with millisecond precision, so a turn stamped more finely than
    /// that comes back truncated to the millisecond.
    #[must_use]
    pub fn to_stored_message(&self) -> StoredMessage {
        match self {
            Self::User { content, timestamp } => StoredMessage::User {
                content:   content.clone(),
                timestamp: *timestamp,
            },
            Self::Assistant {
                content,
                tool_calls,
                provider_parts,
                usage,
                response_id,
                timestamp,
            } => StoredMessage::Assistant {
                content:        content.clone(),
                tool_calls:     tool_calls.clone(),
                provider_parts: provider_parts.clone(),
                usage:          *usage,
                response_id:    response_id.clone(),
                timestamp:      *timestamp,
            },
            Self::ToolResults { results, timestamp } => StoredMessage::ToolResults {
                results:   results.clone(),
                timestamp: *timestamp,
            },
            Self::System { content, timestamp } => StoredMessage::System {
                content:   content.clone(),
                timestamp: *timestamp,
            },
            Self::Compaction {
                summary,
                reason,
                original_turn_count,
                preserved_turn_count,
                estimated_tokens_before,
                summary_token_estimate,
                tracked_file_count,
                summary_truncated,
                usage,
                cost_usd_micros,
                timestamp,
            } => StoredMessage::Compaction {
                summary:                 summary.clone(),
                reason:                  *reason,
                original_turn_count:     *original_turn_count,
                preserved_turn_count:    *preserved_turn_count,
                estimated_tokens_before: *estimated_tokens_before,
                summary_token_estimate:  *summary_token_estimate,
                tracked_file_count:      *tracked_file_count,
                summary_truncated:       *summary_truncated,
                usage:                   *usage,
                cost_usd_micros:         *cost_usd_micros,
                timestamp:               *timestamp,
            },
            Self::Steering { content, timestamp } => StoredMessage::Steering {
                content:   content.clone(),
                timestamp: *timestamp,
            },
        }
    }

    /// Restores a turn from its stored form.
    #[must_use]
    pub fn from_stored_message(message: &StoredMessage) -> Self {
        match message {
            StoredMessage::User { content, timestamp } => Self::User {
                content:   content.clone(),
                timestamp: *timestamp,
            },
            StoredMessage::Assistant {
                content,
                tool_calls,
                provider_parts,
                usage,
                response_id,
                timestamp,
            } => Self::Assistant {
                content:        content.clone(),
                tool_calls:     tool_calls.clone(),
                provider_parts: provider_parts.clone(),
                usage:          *usage,
                response_id:    response_id.clone(),
                timestamp:      *timestamp,
            },
            StoredMessage::ToolResults { results, timestamp } => Self::ToolResults {
                results:   results.clone(),
                timestamp: *timestamp,
            },
            StoredMessage::System { content, timestamp } => Self::System {
                content:   content.clone(),
                timestamp: *timestamp,
            },
            StoredMessage::Compaction {
                summary,
                reason,
                original_turn_count,
                preserved_turn_count,
                estimated_tokens_before,
                summary_token_estimate,
                tracked_file_count,
                summary_truncated,
                usage,
                cost_usd_micros,
                timestamp,
            } => Self::Compaction {
                summary:                 summary.clone(),
                reason:                  *reason,
                original_turn_count:     *original_turn_count,
                preserved_turn_count:    *preserved_turn_count,
                estimated_tokens_before: *estimated_tokens_before,
                summary_token_estimate:  *summary_token_estimate,
                tracked_file_count:      *tracked_file_count,
                summary_truncated:       *summary_truncated,
                usage:                   *usage,
                cost_usd_micros:         *cost_usd_micros,
                timestamp:               *timestamp,
            },
            StoredMessage::Steering { content, timestamp } => Self::Steering {
                content:   content.clone(),
                timestamp: *timestamp,
            },
        }
    }
}

/// What a session is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodingAgentState {
    /// Waiting for input.
    Idle,
    /// Waiting on the model.
    Thinking,
    /// Running tools.
    Executing,
    /// Summarizing older conversation history.
    Compacting,
    /// Shut down.
    Closed,
}

/// One memory file that was loaded into the prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFileSummary {
    /// The path the file was read from.
    pub path:         String,
    /// The file's size on disk, in bytes.
    pub byte_count:   usize,
    /// How many bytes were actually loaded.
    pub loaded_bytes: usize,
    /// Whether the budget cut the file short.
    pub truncated:    bool,
}

/// One discovered skill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillSummary {
    /// The skill's name.
    pub name:        String,
    /// The skill's one-line description.
    pub description: String,
}

/// A `SKILL.md` file, or a skill directory, that discovery skipped.
///
/// Discovery never fails a session over one broken skill; it skips the file
/// and carries on. This is the record of what it skipped, so an application
/// can tell a person which file to fix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedSkill {
    /// The file that was skipped, or the directory that could not be searched.
    pub path:    String,
    /// Why it was skipped.
    pub reason:  SkippedSkillReason,
    /// The failure, rendered for a person.
    pub message: String,
}

/// Why discovery skipped a skill file or directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SkippedSkillReason {
    /// The configured directory could not be searched for `*/SKILL.md`.
    ///
    /// A directory that does not exist is not an error and is not reported;
    /// this is a search that failed.
    UnsearchableDirectory,
    /// The file was found but could not be read.
    UnreadableFile,
    /// The file was read but is not a skill: its frontmatter is missing,
    /// unterminated, or names no skill.
    Malformed,
    /// A directory the application required a skill discovery to search
    /// does not exist.
    MissingDirectory,
}

/// How a skill was activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SkillActivationSource {
    /// The operator typed the skill's slash command.
    Slash,
    /// The model called the skill tool.
    Tool,
}

/// Which built-in coding-agent harness a session is driving.
///
/// The identifier comes from the model catalog: pebble reads
/// `metadata.agent.profile` for the resolved model, falling back to the
/// provider's default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentProfileKind {
    /// Claude models before the current coding-agent harness.
    Anthropic,
    /// Claude 5 models, trained against Anthropic's current harness.
    #[serde(rename = "claude-5")]
    Claude5,
    /// OpenAI models without a family-specific harness.
    #[serde(rename = "openai")]
    OpenAi,
    /// Gemini models.
    Gemini,
    /// Kimi (Moonshot) models, wherever they are served from.
    Kimi,
    /// GPT-5.6 models, which Codex drives with a narrower core tool set.
    Gpt56,
    /// GPT-6 models, with Astra-specific behavior guidance.
    Gpt6,
}

impl AgentProfileKind {
    /// The instruction files this harness reads, in load order: the shared
    /// `AGENTS.md` first, then the vendor's own name where it has one.
    #[must_use]
    pub const fn memory_filenames(self) -> &'static [&'static str] {
        match self {
            Self::Anthropic | Self::Claude5 => &["AGENTS.md", "CLAUDE.md"],
            Self::OpenAi | Self::Gpt56 | Self::Gpt6 => &["AGENTS.md", ".codex/instructions.md"],
            Self::Gemini => &["AGENTS.md", "GEMINI.md"],
            Self::Kimi => &["AGENTS.md"],
        }
    }

    /// Every profile pebble ships.
    ///
    /// A slice rather than an array, so a profile added later does not change
    /// this constant's type.
    pub const ALL: &'static [Self] = &[
        Self::Anthropic,
        Self::Claude5,
        Self::OpenAi,
        Self::Gemini,
        Self::Kimi,
        Self::Gpt56,
        Self::Gpt6,
    ];

    /// The catalog identifier of this profile.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Claude5 => "claude-5",
            Self::OpenAi => "openai",
            Self::Gemini => "gemini",
            Self::Kimi => "kimi",
            Self::Gpt56 => "gpt56",
            Self::Gpt6 => "gpt6",
        }
    }
}

impl fmt::Display for AgentProfileKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// What became of one MCP server the application configured, as the
/// snapshot carries it: a view that starts after the agent was built reads
/// the outcome here, since the event that reported it has already passed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerStatus {
    /// The server's configured name.
    pub server: String,
    /// The tools it advertised, sorted by registered name; empty when it
    /// failed.
    pub tools:  Vec<McpToolSummary>,
    /// Why it did not start, when it did not.
    pub error:  Option<String>,
}

/// One tool an MCP server advertised, as the registry named it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolSummary {
    /// The name the model calls: `mcp__{server}__{tool}`.
    pub name:          String,
    /// The server's own name for the tool.
    pub original_name: String,
}

/// Something a session did, as seen by an observer.
///
/// The serialized form is externally tagged: `{"ToolCallStarted": {…}}` for
/// variants with a payload, and a bare `"SessionEnded"` for those without.
/// Variant names are permanent API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CodingEvent {
    /// The session opened.
    SessionStarted {
        /// The provider the session resolved to.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        /// The model the session resolved to.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model:    Option<String>,
    },
    /// The session closed.
    SessionEnded,
    /// A processing cycle finished and the session is idle again.
    ProcessingEnd,
    /// Input arrived from the caller.
    UserInput {
        /// The input text.
        text:    String,
        /// The ordered content when the input was not plain text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<InputContent>,
        /// Where this input came from.
        #[serde(default, skip_serializing_if = "is_prompt_source")]
        source:  InputSource,
    },
    /// An inference request is about to be dispatched for this round.
    ///
    /// Emitted after the request is built and compaction has run, immediately
    /// before the stream opens. Failover can re-target the call, so
    /// [`CodingEvent::AssistantMessage`] stays authoritative for what answered.
    LlmRequestStarted {
        /// The catalog identifier of the requested model.
        requested_model: String,
    },
    /// The provider produced its first output for the current attempt.
    ///
    /// Edge-triggered: emitted once per stream attempt and re-armed when a
    /// broken or finish-less stream restarts the turn.
    LlmFirstOutput {
        /// Which kind of output arrived first.
        kind: LlmOutputKind,
    },
    /// Replaces the in-progress assistant output buffers.
    ///
    /// Emitted before any turn replay, so a consumer never renders duplicated
    /// output.
    AssistantOutputReplace {
        /// The text the buffer should now hold.
        text:      String,
        /// The reasoning the buffer should now hold.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    /// A committed assistant turn.
    AssistantMessage {
        /// The assistant's text.
        text:            String,
        /// The catalog identifier of the model that answered.
        model:           String,
        /// The token accounting the provider reported.
        usage:           TokenUsage,
        /// The cost of this response, in USD micros.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost_usd_micros: Option<u64>,
        /// Where `cost_usd_micros` came from.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost_source:     Option<CostSource>,
        /// How many tool calls the turn requested.
        tool_call_count: usize,
        /// How much of the context window the session is using.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_window:  Option<ContextWindowSnapshot>,
        /// Readable reasoning normalized from the final response.
        ///
        /// Derived once the response is complete, so retried or replaced
        /// streaming buffers never become durable reasoning.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning:       Option<ReasoningOutput>,
    },
    /// A fragment of assistant text.
    TextDelta {
        /// The fragment.
        delta: String,
    },
    /// A fragment of assistant reasoning.
    ReasoningDelta {
        /// The fragment.
        delta: String,
    },
    /// A tool call started.
    ToolCallStarted {
        /// The tool's registered name.
        tool_name:    String,
        /// The provider's identifier for the call.
        tool_call_id: String,
        /// The parsed call arguments.
        arguments:    serde_json::Value,
    },
    /// A fragment of a running tool's output.
    ToolCallOutputDelta {
        /// The fragment.
        delta: String,
    },
    /// A tool call finished.
    ToolCallCompleted {
        /// The tool's registered name.
        tool_name:             String,
        /// The provider's identifier for the call.
        tool_call_id:          String,
        /// The output handed back to the model.
        output:                serde_json::Value,
        /// Structured details and artifacts for observers; never sent to the
        /// model.
        #[serde(
            default,
            skip_serializing_if = "pebble_agent::ToolOutputMetadata::is_empty"
        )]
        metadata:              pebble_agent::ToolOutputMetadata,
        /// Whether the call failed.
        is_error:              bool,
        /// Why the call failed, when it did.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_kind:            Option<ToolErrorKind>,
        /// Output bytes the tool produced.
        #[serde(default)]
        output_bytes_observed: usize,
        /// Output bytes kept in history.
        #[serde(default)]
        output_bytes_retained: usize,
        /// Output bytes dropped by the retention budget.
        #[serde(default)]
        output_bytes_omitted:  usize,
    },
    /// The outcome of a subordinate process a tool ran.
    ///
    /// Emitted before its owning [`CodingEvent::ToolCallCompleted`], which
    /// stays the single tool-protocol completion and the authoritative owner
    /// of `is_error`. CodingRuntime and tool-call identity come from the
    /// envelope.
    ToolProcessCompleted {
        /// The process's exit status, when it reported one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code:             Option<i32>,
        /// How the process ended.
        termination:           CommandTermination,
        /// How long the process ran, in milliseconds.
        duration_ms:           u64,
        /// Whether standard output and standard error were captured apart.
        streams_separated:     bool,
        /// The retained tail of the process's output.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exec_output_tail:      Option<ExecOutputTail>,
        /// Output bytes the process produced.
        #[serde(default)]
        output_bytes_observed: usize,
        /// Output bytes kept in history.
        #[serde(default)]
        output_bytes_retained: usize,
        /// Output bytes dropped by the retention budget.
        #[serde(default)]
        output_bytes_omitted:  usize,
    },
    /// The session failed.
    Error {
        /// The failure, projected for transport.
        error: ErrorData,
    },
    /// Something went wrong without stopping the session.
    Warning {
        /// A stable machine-readable code.
        kind:    String,
        /// The warning rendered for a human.
        message: String,
        /// Structured detail about the warning.
        details: serde_json::Value,
    },
    /// The loop detector fired.
    LoopDetected,
    /// The model asked for tools after the prompt had run every tool round
    /// its budget allows.
    ///
    /// The calls of that turn were recorded as `Cancelled` without running.
    /// The prompt ends with
    /// [`Error::ToolRoundsExhausted`](crate::Error::ToolRoundsExhausted), and
    /// the session stays open.
    ToolRoundsExhausted {
        /// The configured limit, which is also how many rounds ran.
        limit: usize,
    },
    /// An MCP server the application configured started and its tools are
    /// registered, each under `mcp__{server}__{tool}`.
    McpServerReady {
        /// The server's configured name.
        server: String,
        /// The tools it advertised, sorted by registered name.
        tools:  Vec<McpToolSummary>,
    },
    /// An MCP server the application configured did not start; the session
    /// runs without its tools.
    McpServerFailed {
        /// The server's configured name.
        server: String,
        /// Why it did not start.
        error:  String,
    },
    /// The prompt moved to a fallback route after its model failed.
    ///
    /// The conversation continued as it stood: no tool effect was repeated.
    /// Published by the session on its new route, after
    /// [`SessionStarted`](Self::SessionStarted) reports that route.
    RouteFailover {
        /// The `provider/model` that failed.
        from:    String,
        /// The `provider/model` the prompt continues on.
        to:      String,
        /// How many routes the prompt has moved through, this one included.
        attempt: u32,
        /// The failure that ended the previous route.
        error:   ErrorData,
    },
    /// Steering was injected into the conversation.
    SteeringInjected {
        /// The steering text.
        text:    String,
        /// The ordered content when the steer was not plain text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<InputContent>,
        /// Who authored the steer.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actor:   Option<Actor>,
    },
    /// The cancelled round has fully unwound.
    ///
    /// The session is now ready to consume queued steering or wait for more.
    RoundInterrupted {
        /// The interrupt generation that has settled.
        generation: u64,
    },
    /// Compaction of conversation history started.
    CompactionStarted {
        /// The estimated prompt size that triggered compaction.
        estimated_tokens:    usize,
        /// The model's context window, in tokens.
        context_window_size: usize,
        /// Why this compaction ran.
        #[serde(default, skip_serializing_if = "is_threshold_reason")]
        reason:              CompactionReason,
    },
    /// Compaction of conversation history finished.
    CompactionCompleted {
        /// Turns in history before compaction.
        original_turn_count:    usize,
        /// Turns kept verbatim.
        preserved_turn_count:   usize,
        /// The estimated size of the generated summary.
        summary_token_estimate: usize,
        /// How many tracked files survived compaction.
        tracked_file_count:     usize,
        /// Why this compaction ran.
        #[serde(default, skip_serializing_if = "is_threshold_reason")]
        reason:                 CompactionReason,
    },
    /// Compaction ended without changing history.
    CompactionFailed {
        /// Why this compaction ran.
        reason: CompactionReason,
        /// The failure projected for transport.
        error:  ErrorData,
    },
    /// Compaction was cancelled before it changed history.
    CompactionCancelled {
        /// Why this compaction ran.
        reason: CompactionReason,
    },
    /// An attempt failed to open **or sustain** a stream and the turn is being
    /// replayed.
    LlmRetry {
        /// The provider the attempt targeted.
        provider:   String,
        /// The catalog identifier of the model the attempt targeted.
        model:      String,
        /// The 0-based index of the attempt that failed.
        attempt:    usize,
        /// How long the loop waits before the next attempt.
        delay_secs: f64,
        /// The failure, projected for transport.
        error:      ErrorData,
        /// Which retry loop `attempt` counts.
        phase:      LlmRetryPhase,
    },
    /// A subagent session was created.
    SubAgentSpawned {
        /// The subagent's identifier.
        agent_id:   String,
        /// How deep the subagent sits below the root session.
        depth:      usize,
        /// The task the subagent was given.
        task:       String,
        /// Which reuse generation of the subagent this is.
        #[serde(default = "initial_subagent_generation")]
        generation: u64,
    },
    /// A subagent started a turn.
    SubAgentTurnStarted {
        /// The subagent's identifier.
        agent_id:   String,
        /// How deep the subagent sits below the root session.
        depth:      usize,
        /// The task the subagent was given.
        task:       String,
        /// Which reuse generation of the subagent this is.
        generation: u64,
    },
    /// A subagent finished its task.
    SubAgentCompleted {
        /// The subagent's identifier.
        agent_id:   String,
        /// How deep the subagent sits below the root session.
        depth:      usize,
        /// Which reuse generation of the subagent this is.
        #[serde(default = "initial_subagent_generation")]
        generation: u64,
        /// Whether the task succeeded.
        success:    bool,
        /// How many turns the subagent used.
        turns_used: usize,
    },
    /// A subagent failed.
    SubAgentFailed {
        /// The subagent's identifier.
        agent_id:   String,
        /// How deep the subagent sits below the root session.
        depth:      usize,
        /// Which reuse generation of the subagent this is.
        #[serde(default = "initial_subagent_generation")]
        generation: u64,
        /// The failure, projected for transport.
        error:      ErrorData,
    },
    /// A subagent session was closed and its slot released.
    SubAgentClosed {
        /// The subagent's identifier.
        agent_id:   String,
        /// How deep the subagent sits below the root session.
        depth:      usize,
        /// Which reuse generation of the subagent this is.
        #[serde(default = "initial_subagent_generation")]
        generation: u64,
    },
    /// Memory files were loaded into the prompt.
    MemoryLoaded {
        /// The identifier of the profile the session is driving, as
        /// [`AgentProfileKind::as_str`] spells it.
        ///
        /// Carried as text, so an identifier a later pebble adds still reads
        /// back on an older consumer instead of failing the whole event.
        profile:            String,
        /// The files that were considered.
        files:              Vec<MemoryFileSummary>,
        /// How many bytes were loaded in total.
        total_loaded_bytes: usize,
        /// The byte budget memory had to fit.
        budget_bytes:       usize,
    },
    /// Skills were discovered on the configured paths.
    SkillsDiscovered {
        /// The identifier of the profile the session is driving, as
        /// [`AgentProfileKind::as_str`] spells it.
        ///
        /// Carried as text, so an identifier a later pebble adds still reads
        /// back on an older consumer instead of failing the whole event.
        profile:     String,
        /// The directories that were searched.
        source_dirs: Vec<String>,
        /// The skills that were found.
        skills:      Vec<SkillSummary>,
        /// The files and directories discovery had to skip, so an application
        /// can report a broken skill without reading every file itself.
        ///
        /// Absent from streams recorded before it existed, which read back as
        /// nothing skipped.
        #[serde(default)]
        skipped:     Vec<SkippedSkill>,
    },
    /// A skill was activated.
    SkillActivated {
        /// The skill's name.
        skill_name: String,
        /// How the skill was activated.
        source:     SkillActivationSource,
    },
    /// A todo was created.
    ///
    /// Carries the whole row, so a projection can be reconstructed from the
    /// created event alone.
    TodoCreated(TodoCreatedProps),
    /// A todo was changed.
    ///
    /// Field-by-field optional patches; an absent field means "leave alone",
    /// and a `metadata_patch` key with a `null` value deletes that key.
    TodoUpdated(TodoUpdatedProps),
    /// A todo was removed.
    TodoDeleted(TodoDeletedProps),
}

impl CodingEvent {
    /// Whether this is a streaming-delta or buffer-replacement event.
    ///
    /// A presentation can filter these out when it does not need incremental
    /// updates. The durable event sink still receives them.
    #[must_use]
    pub fn is_streaming_noise(&self) -> bool {
        matches!(
            self,
            Self::AssistantOutputReplace { .. }
                | Self::TextDelta { .. }
                | Self::ReasoningDelta { .. }
                | Self::ToolCallOutputDelta { .. }
        )
    }

    /// Records this event on the tracing subscriber.
    ///
    /// Streaming noise logs nothing. Process output and subagent task prompts
    /// are reported by size rather than content. Consume-phase model retries
    /// are logged by the shared stream loop instead of repeated here.
    pub fn trace(&self, session_id: &str) {
        use tracing::{debug, error, info, warn};

        match self {
            Self::SessionStarted { provider, model } => {
                info!(
                    session_id,
                    provider = provider.as_deref().unwrap_or_default(),
                    model = model.as_deref().unwrap_or_default(),
                    "Agent session started"
                );
            }
            Self::SessionEnded => info!(session_id, "Agent session ended"),
            Self::ProcessingEnd => {
                debug!(session_id, "Processing cycle finished, session idle");
            }
            Self::UserInput { text, source, .. } => {
                debug!(
                    session_id,
                    text_len = text.len(),
                    source = ?source,
                    "User input received"
                );
            }
            Self::LlmRequestStarted { requested_model } => {
                debug!(
                    session_id,
                    model = requested_model.as_str(),
                    "LLM request started"
                );
            }
            Self::LlmFirstOutput { kind } => {
                debug!(
                    session_id,
                    kind = kind.as_str(),
                    "LLM produced first output"
                );
            }
            Self::AssistantMessage {
                model,
                usage,
                tool_call_count,
                ..
            } => {
                debug!(
                    session_id,
                    model = model.as_str(),
                    input_tokens = usage.input,
                    output_tokens = usage.output,
                    tool_call_count,
                    "Assistant message"
                );
            }
            Self::TextDelta { .. }
            | Self::ReasoningDelta { .. }
            | Self::AssistantOutputReplace { .. }
            | Self::ToolCallOutputDelta { .. } => {}
            Self::ToolCallStarted {
                tool_name,
                tool_call_id,
                ..
            } => {
                debug!(
                    session_id,
                    tool = tool_name.as_str(),
                    tool_call_id,
                    "Tool call started"
                );
            }
            Self::ToolCallCompleted {
                tool_name,
                tool_call_id,
                is_error,
                error_kind,
                output_bytes_observed,
                output_bytes_retained,
                output_bytes_omitted,
                ..
            } => {
                debug!(
                    session_id,
                    tool = tool_name.as_str(),
                    tool_call_id,
                    is_error,
                    error_kind = ?error_kind,
                    output_bytes_observed,
                    output_bytes_retained,
                    output_bytes_omitted,
                    "Tool call completed"
                );
            }
            Self::ToolProcessCompleted {
                exit_code,
                termination,
                duration_ms,
                streams_separated,
                exec_output_tail,
                output_bytes_observed,
                output_bytes_retained,
                output_bytes_omitted,
            } => {
                let tail = ExecOutputTail::trace_summary(exec_output_tail.as_ref());
                debug!(
                    session_id,
                    exit_code = ?exit_code,
                    termination = termination.as_str(),
                    duration_ms,
                    streams_separated,
                    output_tail_present = tail.present,
                    stdout_bytes = tail.stdout_bytes,
                    stderr_bytes = tail.stderr_bytes,
                    stdout_truncated = tail.stdout_truncated,
                    stderr_truncated = tail.stderr_truncated,
                    output_bytes_observed,
                    output_bytes_retained,
                    output_bytes_omitted,
                    "Tool process completed"
                );
            }
            Self::Error { error } => {
                error!(
                    session_id,
                    kind = ?error.kind,
                    error = error.message.as_str(),
                    "Agent error"
                );
            }
            Self::Warning { kind, message, .. } => {
                warn!(
                    session_id,
                    kind = kind.as_str(),
                    message = message.as_str(),
                    "Warning"
                );
            }
            Self::LoopDetected => warn!(session_id, "Loop detected"),
            Self::ToolRoundsExhausted { limit } => {
                warn!(session_id, limit, "Tool round budget exhausted");
            }
            Self::McpServerReady { server, tools } => {
                info!(
                    session_id,
                    server = server.as_str(),
                    tools = tools.len(),
                    "MCP server ready"
                );
            }
            Self::McpServerFailed { server, error } => {
                warn!(
                    session_id,
                    server = server.as_str(),
                    error = error.as_str(),
                    "MCP server failed"
                );
            }
            Self::RouteFailover {
                from,
                to,
                attempt,
                error,
            } => {
                warn!(
                    session_id,
                    from = from.as_str(),
                    to = to.as_str(),
                    attempt,
                    error = error.message.as_str(),
                    "Route failover"
                );
            }
            Self::SteeringInjected { text, .. } => {
                debug!(session_id, text_len = text.len(), "Steering injected");
            }
            Self::RoundInterrupted { generation } => {
                debug!(session_id, generation, "Agent round interrupted");
            }
            Self::CompactionStarted {
                estimated_tokens,
                context_window_size,
                reason,
            } => {
                info!(
                    session_id,
                    estimated_tokens,
                    context_window_size,
                    reason = ?reason,
                    "Context compaction started"
                );
            }
            Self::CompactionCompleted {
                original_turn_count,
                preserved_turn_count,
                summary_token_estimate,
                tracked_file_count,
                reason,
            } => {
                info!(
                    session_id,
                    original_turn_count,
                    preserved_turn_count,
                    summary_token_estimate,
                    tracked_file_count,
                    reason = ?reason,
                    "Context compaction completed"
                );
            }
            Self::CompactionFailed { reason, error } => {
                warn!(
                    session_id,
                    reason = ?reason,
                    error = error.message.as_str(),
                    "Context compaction failed"
                );
            }
            Self::CompactionCancelled { reason } => {
                info!(session_id, reason = ?reason, "Context compaction cancelled");
            }
            Self::LlmRetry {
                provider,
                model,
                attempt,
                delay_secs,
                error,
                phase,
            } => {
                if *phase == LlmRetryPhase::Consume {
                    return;
                }
                warn!(
                    session_id,
                    provider,
                    model,
                    attempt,
                    delay_secs,
                    phase = phase.as_str(),
                    error = error.message.as_str(),
                    "LLM request failed, retrying"
                );
            }
            Self::SubAgentSpawned {
                agent_id,
                depth,
                task,
                generation,
            } => {
                debug!(
                    session_id,
                    agent_id,
                    depth,
                    generation,
                    task_len = task.len(),
                    "Subagent spawned"
                );
            }
            Self::SubAgentTurnStarted {
                agent_id,
                depth,
                task,
                generation,
            } => {
                debug!(
                    session_id,
                    agent_id,
                    depth,
                    generation,
                    task_len = task.len(),
                    "Subagent turn started"
                );
            }
            Self::SubAgentCompleted {
                agent_id,
                depth,
                generation,
                success,
                turns_used,
            } => {
                debug!(
                    session_id,
                    agent_id, depth, generation, success, turns_used, "Subagent completed"
                );
            }
            Self::SubAgentFailed {
                agent_id,
                depth,
                generation,
                error,
            } => {
                warn!(
                    session_id,
                    agent_id,
                    depth,
                    generation,
                    error = error.message.as_str(),
                    "Subagent failed"
                );
            }
            Self::SubAgentClosed {
                agent_id,
                depth,
                generation,
            } => {
                debug!(session_id, agent_id, depth, generation, "Subagent closed");
            }
            Self::MemoryLoaded {
                profile,
                files,
                total_loaded_bytes,
                budget_bytes,
            } => {
                info!(
                    session_id,
                    profile = profile.as_str(),
                    file_count = files.len(),
                    total_loaded_bytes,
                    budget_bytes,
                    "Agent memory loaded"
                );
            }
            Self::SkillsDiscovered {
                profile,
                source_dirs,
                skills,
                skipped,
            } => {
                info!(
                    session_id,
                    profile = profile.as_str(),
                    skill_count = skills.len(),
                    source_dir_count = source_dirs.len(),
                    skipped_count = skipped.len(),
                    "Agent skills discovered"
                );
            }
            Self::SkillActivated { skill_name, source } => {
                debug!(
                    session_id,
                    skill = skill_name.as_str(),
                    source = ?source,
                    "Agent skill activated"
                );
            }
            Self::TodoCreated(props) => {
                debug!(
                    session_id,
                    list_id = props.list_id.as_str(),
                    todo_id = props.todo_id.as_str(),
                    "Todo created"
                );
            }
            Self::TodoUpdated(props) => {
                debug!(
                    session_id,
                    list_id = props.list_id.as_str(),
                    todo_id = props.todo_id.as_str(),
                    "Todo updated"
                );
            }
            Self::TodoDeleted(props) => {
                debug!(
                    session_id,
                    list_id = props.list_id.as_str(),
                    todo_id = props.todo_id.as_str(),
                    "Todo deleted"
                );
            }
        }
    }
}

/// One [`CodingEvent`] with the identity a consumer needs to place it.
///
/// Pebble builds the envelope and an application reads it. The struct is
/// `#[non_exhaustive]` because the envelope is where later identity lands —
/// `seq` arrived this way — so build one with [`CodingAgentEvent::new`] and
/// the `with_*` methods instead of a struct literal. Reading a field, and
/// assigning to one on an envelope you already hold, stay open.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CodingAgentEvent {
    /// A monotonic per-stream sequence number, assigned as the event is
    /// published. One root session and all its descendants share a stream.
    #[serde(default)]
    pub seq:               u64,
    /// The root session whose event pump owns this stream.
    ///
    /// Empty only when reading an envelope written before this field existed;
    /// in that case `session_id` identifies the legacy stream.
    #[serde(default)]
    pub stream_id:         String,
    /// What happened.
    pub event:             CodingEvent,
    /// When it happened.
    #[serde(with = "rfc3339_millis")]
    pub timestamp:         SystemTime,
    /// The session that produced the event.
    pub session_id:        String,
    /// The parent session, when this event came from a subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// The tool call the event belongs to, when it belongs to one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id:      Option<String>,
}

impl CodingAgentEvent {
    /// Builds an envelope for an event `session_id` produced at `timestamp`.
    ///
    /// The sequence number starts at zero, which is what an unpublished event
    /// carries; the session's event pump assigns the real one as it publishes.
    #[must_use]
    pub fn new(session_id: impl Into<String>, event: CodingEvent, timestamp: SystemTime) -> Self {
        let session_id = session_id.into();
        Self {
            seq: 0,
            stream_id: session_id.clone(),
            event,
            timestamp,
            session_id,
            parent_session_id: None,
            tool_call_id: None,
        }
    }

    /// Places this envelope at `seq` in its stream.
    #[must_use]
    pub fn with_seq(mut self, seq: u64) -> Self {
        self.seq = seq;
        self
    }

    /// Places this envelope in the stream owned by `stream_id`.
    #[must_use]
    pub fn with_stream_id(mut self, stream_id: impl Into<String>) -> Self {
        self.stream_id = stream_id.into();
        self
    }

    /// The stable identifier for the stream that contains this event.
    ///
    /// Envelopes written before `stream_id` existed use their producing
    /// session as the stream identity.
    #[must_use]
    pub fn stream_id(&self) -> &str {
        if self.stream_id.is_empty() {
            &self.session_id
        } else {
            &self.stream_id
        }
    }

    /// Records the parent session, for an event a subagent produced.
    #[must_use]
    pub fn with_parent_session_id(mut self, parent_session_id: impl Into<String>) -> Self {
        self.parent_session_id = Some(parent_session_id.into());
        self
    }

    /// Records the tool call this event belongs to.
    #[must_use]
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{Duration, UNIX_EPOCH};

    use lithos_llm::types::{ImageContent, MediaSource, ReasoningContent};
    use serde_json::json;

    use super::*;
    use crate::error::{Error, ErrorKind};

    fn moment() -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(1_767_225_600_500)
    }

    fn coding_agent_event(event: CodingEvent) -> CodingAgentEvent {
        CodingAgentEvent::new("ses_1", event, moment()).with_seq(1)
    }

    // --- Input content ---

    #[test]
    fn input_content_keeps_ordered_parts_and_readable_text() {
        let parts = vec![
            ContentPart::Text {
                text: "look at ".into(),
            },
            ContentPart::Image(ImageContent::new(MediaSource::url(
                "https://example.test/image.png",
            ))),
            ContentPart::Text {
                text: "this".into(),
            },
        ];

        let content = InputContent::new(parts.clone());

        assert_eq!(content.parts(), parts);
        assert_eq!(content.text_content(), "look at this");
        assert_eq!(content.into_parts(), parts);
    }

    #[test]
    fn input_content_reads_legacy_text_and_writes_parts() {
        let legacy: InputContent =
            serde_json::from_value(json!("hello")).expect("legacy text parses");
        assert_eq!(legacy.text_content(), "hello");
        assert_eq!(
            serde_json::to_value(legacy).expect("content serializes"),
            json!([{"type": "text", "text": "hello"}])
        );
    }

    // --- Timestamps ---

    #[test]
    fn timestamps_render_as_rfc3339_with_millisecond_precision() {
        assert_eq!(rfc3339_millis::format(moment()), "2026-01-01T00:00:00.500Z");
        assert_eq!(
            rfc3339_millis::format(UNIX_EPOCH),
            "1970-01-01T00:00:00.000Z"
        );
    }

    #[test]
    fn timestamps_truncate_below_a_millisecond() {
        let time = UNIX_EPOCH + Duration::from_nanos(1_999_999);
        assert_eq!(rfc3339_millis::format(time), "1970-01-01T00:00:00.001Z");
    }

    #[test]
    fn timestamps_round_trip_through_their_rendering() {
        let cases = [
            "1970-01-01T00:00:00.000Z",
            "2000-02-29T12:00:00.001Z",
            "2024-12-31T23:59:59.999Z",
            "2026-01-01T00:00:00.500Z",
            "2100-03-01T00:00:00.000Z",
        ];
        for text in cases {
            let parsed = rfc3339_millis::parse(text).expect("parses");
            assert_eq!(rfc3339_millis::format(parsed), text);
        }
    }

    #[test]
    fn timestamps_accept_any_offset_and_normalize_to_utc() {
        let cases = [
            ("2026-01-01T01:00:00.500+01:00", "2026-01-01T00:00:00.500Z"),
            ("2025-12-31T19:00:00.500-05:00", "2026-01-01T00:00:00.500Z"),
            ("2026-01-01T00:00:00.5Z", "2026-01-01T00:00:00.500Z"),
            ("2026-01-01t00:00:00.500z", "2026-01-01T00:00:00.500Z"),
            ("2026-01-01T00:00:00.500123456Z", "2026-01-01T00:00:00.500Z"),
        ];
        for (input, expected) in cases {
            let parsed = rfc3339_millis::parse(input).expect("parses");
            assert_eq!(rfc3339_millis::format(parsed), expected, "input {input}");
        }
    }

    #[test]
    fn a_short_fraction_ends_where_the_offset_begins() {
        let cases = [
            ("2026-01-01T12:00:00.5+12:00", "2026-01-01T00:00:00.500Z"),
            ("2026-01-01T00:00:00.52-11:30", "2026-01-01T11:30:00.520Z"),
            (
                "2026-01-01T00:00:00.123456789+00:00",
                "2026-01-01T00:00:00.123Z",
            ),
        ];
        for (input, expected) in cases {
            let parsed = rfc3339_millis::parse(input).expect("parses");
            assert_eq!(rfc3339_millis::format(parsed), expected, "input {input}");
        }
    }

    #[test]
    fn timestamps_before_the_epoch_round_trip() {
        for text in ["1969-12-31T23:59:59.750Z", "1900-01-01T00:00:00.000Z"] {
            let parsed = rfc3339_millis::parse(text).expect("parses");
            assert_eq!(rfc3339_millis::format(parsed), text);
        }
    }

    #[test]
    fn timestamps_before_the_epoch_also_truncate_toward_the_past() {
        let time = UNIX_EPOCH - Duration::from_micros(250_400);
        assert_eq!(rfc3339_millis::format(time), "1969-12-31T23:59:59.749Z");
    }

    #[test]
    fn leap_days_are_accepted_only_in_leap_years() {
        for text in ["2024-02-29T00:00:00.000Z", "2000-02-29T00:00:00.000Z"] {
            let parsed = rfc3339_millis::parse(text).expect("parses");
            assert_eq!(rfc3339_millis::format(parsed), text);
        }
    }

    #[test]
    fn malformed_timestamps_are_rejected() {
        let cases = [
            "",
            "2026-01-01",
            "2026-01-01T00:00:00",
            "2026-13-01T00:00:00.000Z",
            "2026-01-01T25:00:00.000Z",
            "2026/01/01T00:00:00.000Z",
            "2026-01-01T00:00:00.Z",
            "2026-01-01T00:00:00.000+1:00",
            // Days that no calendar has, which chrono-based writers reject.
            "2026-02-29T00:00:00.000Z",
            "2026-02-30T00:00:00.000Z",
            "2026-04-31T00:00:00.000Z",
            "1900-02-29T00:00:00.000Z",
            "2026-01-00T00:00:00.000Z",
        ];
        for text in cases {
            assert!(
                rfc3339_millis::parse(text).is_err(),
                "expected `{text}` to be rejected"
            );
        }
    }

    // --- History turns ---

    #[test]
    fn a_user_turn_converts_to_one_text_part() {
        let turn = Message::User {
            content:   "fix the test".into(),
            timestamp: moment(),
        };
        let message = turn.to_llm_message();
        assert_eq!(message.role(), Role::User);
        assert_eq!(message.content(), &[ContentPart::Text {
            text: "fix the test".into(),
        }]);
        assert_eq!(turn.timestamp(), moment());
    }

    #[test]
    fn an_assistant_turn_puts_provider_parts_before_text_and_tool_calls() {
        let turn = Message::Assistant {
            content:        "on it".into(),
            tool_calls:     vec![ToolCall::function("call_1", "shell", json!({"cmd": "ls"}))],
            provider_parts: vec![ContentPart::opaque("openai.reasoning", json!({"id": "r"}))],
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      moment(),
        };
        let message = turn.to_llm_message();

        assert_eq!(message.role(), Role::Assistant);
        assert!(matches!(message.content(), [
            ContentPart::Opaque { .. },
            ContentPart::Text { .. },
            ContentPart::ToolCall(_)
        ]));
    }

    #[test]
    fn an_assistant_turn_with_no_text_emits_no_text_part() {
        let turn = Message::Assistant {
            content:        String::new(),
            tool_calls:     vec![ToolCall::function("call_1", "shell", json!({}))],
            provider_parts: Vec::new(),
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      moment(),
        };
        assert!(matches!(turn.to_llm_message().content(), [
            ContentPart::ToolCall(_)
        ]));
    }

    #[test]
    fn tool_results_carry_the_first_call_id_on_the_message() {
        let turn = Message::ToolResults {
            results:   vec![
                ToolResult {
                    tool_call_id: "call_1".into(),
                    name:         None,
                    content:      vec![ContentPart::Text { text: "ok".into() }],
                    is_error:     false,
                },
                ToolResult {
                    tool_call_id: "call_2".into(),
                    name:         None,
                    content:      vec![ContentPart::Text {
                        text: "also ok".into(),
                    }],
                    is_error:     false,
                },
            ],
            timestamp: moment(),
        };
        let message = turn.to_llm_message();

        assert_eq!(message.role(), Role::Tool);
        assert_eq!(message.content().len(), 2);
        assert_eq!(message.tool_call_id(), Some("call_1"));
    }

    #[test]
    fn empty_tool_results_carry_no_message_level_call_id() {
        let turn = Message::ToolResults {
            results:   Vec::new(),
            timestamp: moment(),
        };
        assert_eq!(turn.to_llm_message().tool_call_id(), None);
    }

    #[test]
    fn system_and_steering_turns_keep_their_distinct_roles() {
        let system = Message::System {
            content:   "be brief".into(),
            timestamp: moment(),
        };
        let steering = Message::Steering {
            content:   "also update the changelog".into(),
            timestamp: moment(),
        };
        assert_eq!(system.to_llm_message().role(), Role::System);
        assert_eq!(steering.to_llm_message().role(), Role::User);
    }

    #[test]
    fn reasoning_text_reads_the_first_readable_reasoning_part() {
        let redacted = ContentPart::Reasoning(ReasoningContent {
            text:             "opaque".into(),
            signature:        Some("sig".into()),
            signature_origin: None,
            redacted:         true,
        });
        let readable = ContentPart::Reasoning(ReasoningContent {
            text:             "thought about it".into(),
            signature:        None,
            signature_origin: None,
            redacted:         false,
        });
        let turn = Message::Assistant {
            content:        String::new(),
            tool_calls:     Vec::new(),
            provider_parts: vec![redacted, readable],
            usage:          TokenUsage::default(),
            response_id:    "resp_1".into(),
            timestamp:      moment(),
        };
        assert_eq!(turn.reasoning_text(), Some("thought about it"));
    }

    #[test]
    fn reasoning_text_is_absent_on_other_turns() {
        let turn = Message::User {
            content:   "hi".into(),
            timestamp: moment(),
        };
        assert_eq!(turn.reasoning_text(), None);
    }

    // --- Event envelope ---

    #[test]
    fn coding_agent_event_omits_absent_identity_and_keeps_its_timestamp_format() {
        let event = coding_agent_event(CodingEvent::SessionStarted {
            provider: Some("anthropic".into()),
            model:    Some("claude-sonnet-5".into()),
        });
        let json = serde_json::to_string(&event).expect("serializes");

        assert!(json.contains("ses_1"), "{json}");
        assert!(json.contains("SessionStarted"), "{json}");
        assert!(!json.contains("parent_session_id"), "{json}");
        assert!(!json.contains("tool_call_id"), "{json}");
        assert!(json.contains("2026-01-01T00:00:00.500Z"), "{json}");

        let restored: CodingAgentEvent = serde_json::from_str(&json).expect("parses");
        assert_eq!(restored, event);
    }

    #[test]
    fn coding_agent_event_keeps_subagent_identity_when_present() {
        let mut event = coding_agent_event(CodingEvent::ProcessingEnd);
        event.parent_session_id = Some("ses_parent".into());
        event.tool_call_id = Some("call_1".into());

        let json = serde_json::to_string(&event).expect("serializes");
        let restored: CodingAgentEvent = serde_json::from_str(&json).expect("parses");

        assert_eq!(restored.parent_session_id.as_deref(), Some("ses_parent"));
        assert_eq!(restored.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn a_new_envelope_starts_unpublished_and_takes_its_identity_afterwards() {
        let envelope = CodingAgentEvent::new("ses_1", CodingEvent::SessionEnded, moment());

        assert_eq!(envelope.seq, 0, "an unpublished envelope carries no place");
        assert_eq!(envelope.stream_id, "ses_1");
        assert_eq!(envelope.parent_session_id, None);
        assert_eq!(envelope.tool_call_id, None);

        let placed = envelope
            .with_seq(9)
            .with_parent_session_id("ses_root")
            .with_tool_call_id("call_1");

        assert_eq!(placed.seq, 9);
        assert_eq!(placed.stream_id, "ses_1");
        assert_eq!(placed.session_id, "ses_1");
        assert_eq!(placed.timestamp, moment());
        assert_eq!(placed.parent_session_id.as_deref(), Some("ses_root"));
        assert_eq!(placed.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn a_coding_agent_event_without_a_sequence_reads_back_as_zero() {
        let event: CodingAgentEvent = serde_json::from_value(json!({
            "event": "SessionEnded",
            "timestamp": "2026-01-01T00:00:00.500Z",
            "session_id": "ses_1",
        }))
        .expect("parses");
        assert_eq!(event.seq, 0);
        assert!(event.stream_id.is_empty());
        assert_eq!(event.stream_id(), "ses_1");
    }

    // --- Event vocabulary ---

    #[test]
    fn unit_variants_serialize_as_bare_names() {
        assert_eq!(
            serde_json::to_value(CodingEvent::SessionEnded).expect("serializes"),
            json!("SessionEnded")
        );
        assert_eq!(
            serde_json::to_value(CodingEvent::LoopDetected).expect("serializes"),
            json!("LoopDetected")
        );
    }

    #[test]
    fn payload_variants_are_externally_tagged() {
        let event = CodingEvent::ToolCallStarted {
            tool_name:    "shell".into(),
            tool_call_id: "call_1".into(),
            arguments:    json!({"command": "ls"}),
        };
        assert_eq!(
            serde_json::to_value(&event).expect("serializes"),
            json!({"ToolCallStarted": {
                "tool_name": "shell",
                "tool_call_id": "call_1",
                "arguments": {"command": "ls"},
            }})
        );
    }

    #[test]
    fn streaming_noise_is_exactly_the_four_buffer_events() {
        let noisy = [
            CodingEvent::AssistantOutputReplace {
                text:      String::new(),
                reasoning: None,
            },
            CodingEvent::TextDelta { delta: "a".into() },
            CodingEvent::ReasoningDelta { delta: "a".into() },
            CodingEvent::ToolCallOutputDelta { delta: "a".into() },
        ];
        for event in &noisy {
            assert!(event.is_streaming_noise(), "{event:?}");
        }

        let quiet = [
            CodingEvent::SessionEnded,
            CodingEvent::UserInput {
                text:    "hi".into(),
                content: None,
                source:  InputSource::Prompt,
            },
            CodingEvent::LoopDetected,
        ];
        for event in &quiet {
            assert!(!event.is_streaming_noise(), "{event:?}");
        }
    }

    #[test]
    fn assistant_output_replace_round_trips() {
        let event = CodingEvent::AssistantOutputReplace {
            text:      "hello again".into(),
            reasoning: Some("retrying from scratch".into()),
        };
        let json = serde_json::to_string(&event).expect("serializes");
        assert_eq!(
            serde_json::from_str::<CodingEvent>(&json).expect("parses"),
            event
        );
    }

    #[test]
    fn assistant_message_carries_usage_and_cost() {
        let event = CodingEvent::AssistantMessage {
            text:            "hello".into(),
            model:           "claude-sonnet-5".into(),
            usage:           TokenUsage {
                input:       100,
                output:      50,
                reasoning:   20,
                cache_read:  80,
                cache_write: 10,
            },
            cost_usd_micros: Some(125_000),
            cost_source:     Some(CostSource::Catalog),
            tool_call_count: 2,
            context_window:  None,
            reasoning:       None,
        };
        let value = serde_json::to_value(&event).expect("serializes");
        let payload = &value["AssistantMessage"];

        assert_eq!(payload["model"], json!("claude-sonnet-5"));
        assert_eq!(payload["usage"]["cache_read"], json!(80));
        assert_eq!(payload["cost_usd_micros"], json!(125_000));
        assert_eq!(payload["cost_source"], json!("catalog"));
        assert!(payload.get("context_window").is_none());
        assert!(payload.get("reasoning").is_none());
        assert_eq!(
            serde_json::from_value::<CodingEvent>(value).expect("parses"),
            event
        );
    }

    #[test]
    fn legacy_subagent_events_default_to_the_initial_generation() {
        let event: CodingEvent = serde_json::from_str(
            r#"{"SubAgentSpawned":{"agent_id":"sa-1","depth":0,"task":"test"}}"#,
        )
        .expect("parses");
        assert!(matches!(event, CodingEvent::SubAgentSpawned {
            generation: 1,
            ..
        }));

        let event: CodingEvent = serde_json::from_str(
            r#"{"SubAgentCompleted":{"agent_id":"sa-1","depth":0,"success":true,"turns_used":2}}"#,
        )
        .expect("parses");
        assert!(matches!(event, CodingEvent::SubAgentCompleted {
            generation: 1,
            ..
        }));

        let error = serde_json::to_value(ErrorData::from(&Error::ToolExecution("boom".into())))
            .expect("serializes");
        let event: CodingEvent = serde_json::from_value(
            json!({ "SubAgentFailed": { "agent_id": "sa-1", "depth": 0, "error": error } }),
        )
        .expect("parses");
        assert!(matches!(event, CodingEvent::SubAgentFailed {
            generation: 1,
            ..
        }));

        let event: CodingEvent =
            serde_json::from_str(r#"{"SubAgentClosed":{"agent_id":"sa-1","depth":0}}"#)
                .expect("parses");
        assert!(matches!(event, CodingEvent::SubAgentClosed {
            generation: 1,
            ..
        }));

        assert_eq!(initial_subagent_generation(), INITIAL_SUBAGENT_GENERATION);
    }

    #[test]
    fn a_subagent_turn_has_no_legacy_generation_to_default() {
        // The asymmetry is deliberate, and this pins it: the other four
        // subagent events existed before the generation field and default it,
        // while a turn has never been published without one. Harmonizing the
        // serde attributes across all five would invent a first generation for
        // an event that is simply malformed.
        let error = serde_json::from_str::<CodingEvent>(
            r#"{"SubAgentTurnStarted":{"agent_id":"sa-1","depth":0,"task":"test"}}"#,
        )
        .expect_err("a turn with no generation does not parse");

        assert!(error.to_string().contains("generation"), "{error}");
    }

    #[test]
    fn legacy_tool_completions_default_their_byte_counts_and_error_kind() {
        let event: CodingEvent = serde_json::from_str(
            r#"{"ToolCallCompleted":{"tool_name":"shell","tool_call_id":"call_1","output":"ok","is_error":false}}"#,
        )
        .expect("parses");

        assert!(matches!(event, CodingEvent::ToolCallCompleted {
            error_kind: None,
            output_bytes_observed: 0,
            output_bytes_retained: 0,
            output_bytes_omitted: 0,
            ..
        }));
    }

    #[test]
    fn a_failed_tool_call_reports_why() {
        let event = CodingEvent::ToolCallCompleted {
            tool_name:             "shell".into(),
            tool_call_id:          "call_1".into(),
            output:                json!("permission denied"),
            metadata:              pebble_agent::ToolOutputMetadata::default(),
            is_error:              true,
            error_kind:            Some(ToolErrorKind::Denied),
            output_bytes_observed: 17,
            output_bytes_retained: 17,
            output_bytes_omitted:  0,
        };
        let value = serde_json::to_value(&event).expect("serializes");
        assert_eq!(value["ToolCallCompleted"]["error_kind"], json!("denied"));
        assert_eq!(
            serde_json::from_value::<CodingEvent>(value).expect("parses"),
            event
        );
    }

    #[test]
    fn error_events_carry_the_projection() {
        let event = CodingEvent::Error {
            error: ErrorData::from(&Error::ToolExecution("command failed".into())),
        };
        let value = serde_json::to_value(&event).expect("serializes");

        assert_eq!(value["Error"]["error"]["kind"], json!("tool_execution"));
        assert_eq!(
            value["Error"]["error"]["message"],
            json!("tool execution error: command failed")
        );

        let restored: CodingEvent = serde_json::from_value(value).expect("parses");
        assert!(matches!(
            restored,
            CodingEvent::Error { error } if error.kind == ErrorKind::ToolExecution
        ));
    }

    #[test]
    fn retry_events_pin_the_zero_based_attempt_and_phase() {
        let event = CodingEvent::LlmRetry {
            provider:   "openai".into(),
            model:      "gpt-5".into(),
            attempt:    0,
            delay_secs: 2.0,
            error:      ErrorData::new(ErrorKind::Llm, "too fast"),
            phase:      LlmRetryPhase::Open,
        };
        let value = serde_json::to_value(&event).expect("serializes");

        assert_eq!(value["LlmRetry"]["attempt"], json!(0));
        assert_eq!(value["LlmRetry"]["phase"], json!("open"));
        assert_eq!(
            serde_json::from_value::<CodingEvent>(value).expect("parses"),
            event
        );
    }

    #[test]
    fn subagent_failures_carry_the_projection() {
        let event = CodingEvent::SubAgentFailed {
            agent_id:   "sa-1".into(),
            depth:      1,
            generation: 2,
            error:      ErrorData::from(&Error::SessionClosed),
        };
        let json = serde_json::to_string(&event).expect("serializes");
        let restored: CodingEvent = serde_json::from_str(&json).expect("parses");
        assert_eq!(restored, event);
    }

    #[test]
    fn steering_records_who_authored_it() {
        let event = CodingEvent::SteeringInjected {
            text:    "also update the changelog".into(),
            content: None,
            actor:   Some(Actor::User {
                id:           Some("u_1".into()),
                display_name: None,
            }),
        };
        let value = serde_json::to_value(&event).expect("serializes");
        assert_eq!(
            value["SteeringInjected"]["actor"],
            json!({"kind": "user", "id": "u_1"})
        );
    }

    #[test]
    fn steering_without_an_author_omits_the_field() {
        let event = CodingEvent::SteeringInjected {
            text:    "keep going".into(),
            content: None,
            actor:   None,
        };
        assert_eq!(
            serde_json::to_value(&event).expect("serializes"),
            json!({"SteeringInjected": {"text": "keep going"}})
        );
    }

    #[test]
    fn todo_events_serialize_their_props_inline() {
        let event = CodingEvent::TodoDeleted(TodoDeletedProps {
            list_id:   "openai_plan:ses_1".into(),
            list_kind: TodoListKind::OpenAiPlan,
            todo_id:   "a".into(),
        });
        assert_eq!(
            serde_json::to_value(&event).expect("serializes"),
            json!({"TodoDeleted": {
                "list_id": "openai_plan:ses_1",
                "list_kind": "openai_plan",
                "todo_id": "a",
            }})
        );
    }

    #[test]
    fn memory_and_skill_events_name_the_profile() {
        let event = CodingEvent::SkillsDiscovered {
            profile:     AgentProfileKind::Claude5.as_str().to_owned(),
            source_dirs: vec!["/skills".into()],
            skills:      vec![SkillSummary {
                name:        "review".into(),
                description: "Review a diff".into(),
            }],
            skipped:     Vec::new(),
        };
        let value = serde_json::to_value(&event).expect("serializes");
        assert_eq!(value["SkillsDiscovered"]["profile"], json!("claude-5"));
        assert_eq!(
            serde_json::from_value::<CodingEvent>(value).expect("parses"),
            event
        );
    }

    #[test]
    fn a_profile_identifier_this_build_does_not_know_still_parses() {
        let event: CodingEvent = serde_json::from_value(json!({
            "SkillsDiscovered": {
                "profile": "a-later-profile",
                "source_dirs": [],
                "skills": [],
            },
        }))
        .expect("parses");

        assert!(
            matches!(event, CodingEvent::SkillsDiscovered { profile, .. } if profile == "a-later-profile")
        );
    }

    #[test]
    fn every_profile_identifier_round_trips() {
        for profile in AgentProfileKind::ALL.iter().copied() {
            let value = serde_json::to_value(profile).expect("serializes");
            assert_eq!(value, json!(profile.as_str()));
            assert_eq!(
                serde_json::from_value::<AgentProfileKind>(value).expect("parses"),
                profile
            );
        }
        let identifiers: Vec<&str> = AgentProfileKind::ALL
            .iter()
            .map(|profile| profile.as_str())
            .collect();
        assert_eq!(identifiers, vec![
            "anthropic",
            "claude-5",
            "openai",
            "gemini",
            "kimi",
            "gpt56",
            "gpt6"
        ]);
    }

    #[test]
    fn process_completions_report_a_bounded_tail() {
        let event = CodingEvent::ToolProcessCompleted {
            exit_code:             Some(1),
            termination:           CommandTermination::Exited,
            duration_ms:           42,
            streams_separated:     true,
            exec_output_tail:      Some(ExecOutputTail {
                stdout:           Some("out".into()),
                stderr:           None,
                stdout_truncated: true,
                stderr_truncated: false,
            }),
            output_bytes_observed: 100,
            output_bytes_retained: 3,
            output_bytes_omitted:  97,
        };
        let value = serde_json::to_value(&event).expect("serializes");
        assert_eq!(
            value["ToolProcessCompleted"]["termination"],
            json!("exited")
        );
        assert_eq!(
            serde_json::from_value::<CodingEvent>(value).expect("parses"),
            event
        );
    }

    #[test]
    fn unknown_members_are_ignored_when_parsing() {
        let event: CodingEvent = serde_json::from_value(json!({
            "UserInput": {"text": "hi", "future_field": 7},
        }))
        .expect("parses");
        assert_eq!(event, CodingEvent::UserInput {
            text:    "hi".into(),
            content: None,
            source:  InputSource::Prompt,
        });
    }

    #[test]
    fn every_event_survives_a_json_round_trip() {
        let events = vec![
            CodingEvent::SessionStarted {
                provider: None,
                model:    None,
            },
            CodingEvent::SessionEnded,
            CodingEvent::ProcessingEnd,
            CodingEvent::UserInput {
                text:    "hi".into(),
                content: None,
                source:  InputSource::Prompt,
            },
            CodingEvent::LlmRequestStarted {
                requested_model: "claude-sonnet-5".into(),
            },
            CodingEvent::LlmFirstOutput {
                kind: LlmOutputKind::Reasoning,
            },
            CodingEvent::AssistantOutputReplace {
                text:      String::new(),
                reasoning: None,
            },
            CodingEvent::TextDelta { delta: "a".into() },
            CodingEvent::ReasoningDelta { delta: "a".into() },
            CodingEvent::ToolCallOutputDelta { delta: "a".into() },
            CodingEvent::Warning {
                kind:    "budget".into(),
                message: "close to the limit".into(),
                details: json!({}),
            },
            CodingEvent::LoopDetected,
            CodingEvent::RoundInterrupted { generation: 3 },
            CodingEvent::CompactionStarted {
                estimated_tokens:    5_000,
                context_window_size: 8_000,
                reason:              CompactionReason::Threshold,
            },
            CodingEvent::CompactionCompleted {
                original_turn_count:    20,
                preserved_turn_count:   6,
                summary_token_estimate: 500,
                tracked_file_count:     3,
                reason:                 CompactionReason::Threshold,
            },
            CodingEvent::CompactionFailed {
                reason: CompactionReason::Manual,
                error:  ErrorData::new(ErrorKind::Compaction, "summary failed"),
            },
            CodingEvent::CompactionCancelled {
                reason: CompactionReason::Manual,
            },
            CodingEvent::SubAgentSpawned {
                agent_id:   "sa-1".into(),
                depth:      1,
                task:       "list files".into(),
                generation: 1,
            },
            CodingEvent::SubAgentTurnStarted {
                agent_id:   "sa-1".into(),
                depth:      1,
                task:       "list files".into(),
                generation: 2,
            },
            CodingEvent::SubAgentCompleted {
                agent_id:   "sa-1".into(),
                depth:      1,
                generation: 2,
                success:    true,
                turns_used: 5,
            },
            CodingEvent::SubAgentClosed {
                agent_id:   "sa-1".into(),
                depth:      1,
                generation: 2,
            },
            CodingEvent::MemoryLoaded {
                profile:            AgentProfileKind::Gpt56.as_str().to_owned(),
                files:              vec![MemoryFileSummary {
                    path:         "/memory/AGENTS.md".into(),
                    byte_count:   100,
                    loaded_bytes: 100,
                    truncated:    false,
                }],
                total_loaded_bytes: 100,
                budget_bytes:       1_000,
            },
            CodingEvent::SkillActivated {
                skill_name: "review".into(),
                source:     SkillActivationSource::Slash,
            },
            CodingEvent::TodoCreated(TodoCreatedProps {
                list_id:     "openai_plan:ses_1".into(),
                list_kind:   TodoListKind::OpenAiPlan,
                todo_id:     "a".into(),
                status:      TodoStatus::Pending,
                order:       0,
                subject:     "first".into(),
                description: String::new(),
                active_form: None,
                owner:       None,
                blocks:      Vec::new(),
                blocked_by:  Vec::new(),
                metadata:    BTreeMap::new(),
            }),
            CodingEvent::TodoUpdated(TodoUpdatedProps::new(
                "openai_plan:ses_1",
                TodoListKind::OpenAiPlan,
                "a",
            )),
        ];

        let json = serde_json::to_string(&events).expect("serializes");
        let restored: Vec<CodingEvent> = serde_json::from_str(&json).expect("parses");
        assert_eq!(restored, events);
    }
}
