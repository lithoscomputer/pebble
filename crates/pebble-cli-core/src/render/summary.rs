//! The closing report: what the stream added up to.

use pebble_coding_agent::events::CodingAgentEvent;
use pebble_coding_agent::projection::SessionProjection;

use super::Style;
use crate::terminal::print_err;

/// What the stream said, kept for the closing report.
///
/// The facts are the [`SessionProjection`]'s, the one fold every embedder
/// reads a session through, so this summary and an application's view of the
/// same stream agree. The one thing added is what this reader missed of the
/// stream, which is the reader's own and not the session's.
#[derive(Debug, Default)]
pub struct Summary {
    projection: SessionProjection,
    dropped:    u64,
}

impl Summary {
    /// Folds one event into the report.
    pub(super) fn apply(&mut self, event: &CodingAgentEvent) {
        self.projection.apply(event);
    }

    /// Counts events the reader missed because it fell behind the stream.
    pub(super) fn missed(&mut self, dropped: u64) {
        self.dropped += dropped;
    }

    /// The fold the report reads its facts from.
    #[must_use]
    pub fn projection(&self) -> &SessionProjection {
        &self.projection
    }

    /// Prints what the prompt used, after the answer.
    pub fn report(&self, style: Style) {
        if style != Style::Text {
            return;
        }
        print_err("");
        for line in self.lines() {
            print_err(&line);
        }
    }

    /// The report's lines, in order.
    ///
    /// Turns, tokens, and cost are the root session's, as the projection
    /// keeps them, and agree with the prompt's report: a compaction's summary
    /// call is on the stream with its usage, so the fold bills it as the
    /// report does. What the children did is on the `subagents` line. Tools
    /// and retries are the tree's.
    pub(super) fn lines(&self) -> Vec<String> {
        let projection = &self.projection;
        let mut lines = vec![format!("turns:  {}", projection.messages)];

        let calls: u64 = projection.tools.values().map(|tool| tool.calls).sum();
        let failed: u64 = projection.tools.values().map(|tool| tool.errors).sum();
        let open: u64 = projection.tools.values().map(|tool| tool.open).sum();
        let unfinished = if open > 0 {
            format!(", {open} unfinished")
        } else {
            String::new()
        };
        lines.push(format!(
            "tools:  {calls} call(s), {failed} failed{unfinished}"
        ));
        let named = projection
            .tools
            .iter()
            .map(|(name, tool)| format!("{name} x{}", tool.calls))
            .collect::<Vec<_>>()
            .join(", ");
        if !named.is_empty() {
            lines.push(format!("        {named}"));
        }

        let tokens = projection.usage.tokens;
        lines.push(format!(
            "tokens: {} in, {} out, {} reasoning, {} cached ({} total)",
            tokens.input,
            tokens.output,
            tokens.reasoning,
            tokens.cache_read + tokens.cache_write,
            tokens.total()
        ));
        lines.push(match projection.usage.cost {
            Some(cost) => format!("cost:   {}", dollars(cost.usd_micros)),
            None => "cost:   not reported for this model".to_owned(),
        });

        let spawned = projection.subagent_counts.spawned;
        if spawned > 0 || !projection.descendants.is_empty() {
            let usage = projection.descendant_usage();
            let turns: u64 = projection
                .descendants
                .values()
                .map(|account| account.messages)
                .sum();
            let cost = usage.cost.map_or_else(String::new, |cost| {
                format!(", {}", dollars(cost.usd_micros))
            });
            lines.push(format!(
                "subagents: {spawned} spawned, {turns} turn(s), {} tokens{cost}",
                usage.total_tokens()
            ));
        }
        if projection.retries > 0 {
            lines.push(format!("retries: {}", projection.retries));
        }
        if self.dropped > 0 {
            lines.push(format!(
                "dropped: {} event(s) this reader missed",
                self.dropped
            ));
        }
        lines
    }
}

/// Millionths of a dollar, as dollars.
fn dollars(usd_micros: u64) -> String {
    format!("${:.4}", usd_micros as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use pebble_coding_agent::events::{CodingEvent, CompactionReason};

    use super::super::fixtures::{message, priced, prompt, root, scripted_prompt};
    use super::*;

    fn summarize(events: &[CodingAgentEvent]) -> Summary {
        let mut summary = Summary::default();
        for event in events {
            summary.apply(event);
        }
        summary
    }

    #[test]
    fn the_summary_is_the_projection_of_the_stream() {
        let events = scripted_prompt();
        let summary = summarize(&events);

        let mut projection = SessionProjection::new();
        projection.apply_all(&events);
        assert_eq!(summary.projection(), &projection);
        assert_eq!(projection.messages, 2, "the root's turns");
        assert_eq!(projection.retries, 1);
        assert_eq!(projection.descendants["ses_child"].messages, 1);

        assert_eq!(summary.lines(), [
            "turns:  2",
            "tools:  3 call(s), 1 failed",
            "        edit_file x1, read_file x2",
            "tokens: 30 in, 10 out, 0 reasoning, 0 cached (40 total)",
            "cost:   $0.0003",
            "subagents: 1 spawned, 1 turn(s), 10 tokens, $0.0004",
            "retries: 1",
        ]);
    }

    #[test]
    fn a_prompt_without_children_or_retries_says_nothing_of_them() {
        let events = [
            root(prompt("go")),
            root(message(10, 5, None)),
            root(CodingEvent::ProcessingEnd),
            root(CodingEvent::SessionEnded),
        ];
        assert_eq!(summarize(&events).lines(), [
            "turns:  1",
            "tools:  0 call(s), 0 failed",
            "tokens: 10 in, 5 out, 0 reasoning, 0 cached (15 total)",
            "cost:   not reported for this model",
        ]);
    }

    #[test]
    fn a_compactions_summary_call_is_on_the_tokens_line() {
        let events = [
            root(prompt("go")),
            root(message(10, 5, Some(100))),
            root(CodingEvent::CompactionCompleted {
                original_turn_count:    4,
                preserved_turn_count:   1,
                summary_token_estimate: 20,
                tracked_file_count:     0,
                reason:                 CompactionReason::Threshold,
                usage:                  priced(30, 2, Some(200)),
            }),
            root(CodingEvent::ProcessingEnd),
            root(CodingEvent::SessionEnded),
        ];
        assert_eq!(summarize(&events).lines(), [
            "turns:  1",
            "tools:  0 call(s), 0 failed",
            "tokens: 40 in, 7 out, 0 reasoning, 0 cached (47 total)",
            "cost:   $0.0003",
        ]);
    }

    #[test]
    fn missed_events_are_the_last_line() {
        let mut summary = summarize(&scripted_prompt());
        summary.missed(3);
        assert_eq!(
            summary.lines().last().map(String::as_str),
            Some("dropped: 3 event(s) this reader missed")
        );
    }
}
