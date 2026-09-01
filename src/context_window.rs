//! Accounting for how much of the model's context window a session occupies.
//!
//! A session builds one [`ContextWindowSnapshot`] per request. The breakdown is
//! always a local estimate — the only way to attribute tokens to the system
//! prompt, the tools, and the conversation separately — and the total is
//! replaced by the provider's own count when there is one, with the breakdown
//! scaled to match it.
//!
//! Two things about the estimate are worth knowing before reading a snapshot.
//! Media and provider-native content are sized rather than tokenized, which the
//! snapshot says in its warnings. And skill instructions that have already been
//! expanded into the conversation are counted as conversation, not as skills,
//! because by then they are ordinary turns.

use std::collections::{BTreeMap, HashSet};
use std::time::SystemTime;

use lithos_llm::estimate::{
    EstimateWarning, TokenEstimate, message_tokens, request_control_tokens, text_tokens,
    tool_definition_tokens,
};
use lithos_llm::types::{ContentPart, Message, Request, Role};

use crate::memory::MemoryDocument;
use crate::skills::{Skill, format_skills_prompt_section};
use crate::tool::{ToolDefinitionWithSource, ToolVocabulary};
use crate::types::{
    ContextWindowBreakdownItem, ContextWindowCategory, ContextWindowCountMethod,
    ContextWindowSnapshot, ContextWindowStaleness, ContextWindowWarning, TokenUsage, ToolSource,
};

/// The warning a snapshot carries once a skill has been expanded into the
/// conversation.
pub const ACTIVATED_SKILL_WARNING: &str = "activated_skill_context_counted_as_conversation";

/// Everything one snapshot is measured from.
///
/// The request is what will actually be sent; the rest is what pebble knows
/// about where the request's parts came from, which is the only way the
/// breakdown can separate memory and skills from the system prompt that
/// contains them.
///
/// This is an argument bundle rather than something an application reads back,
/// so it is plainly constructible and a member added later is a breaking
/// change.
#[derive(Debug, Clone, Copy)]
pub struct ContextWindowInput<'a> {
    /// The request whose tokens are being counted.
    pub request: &'a Request,
    /// The tools the request advertises, with where each came from.
    pub tools: &'a [ToolDefinitionWithSource],
    /// The assembled system prompt, used to recognize the system message.
    pub system_prompt: &'a str,
    /// What the memory files contribute to the system prompt, from
    /// [`memory_prompt_tokens`]. Computed once per session rather than here,
    /// because the memory never changes after initialization.
    pub memory_tokens: u64,
    /// What the skills section contributes to the system prompt, from
    /// [`skills_prompt_tokens`]. Fixed per session, like the memory.
    pub skills_tokens: u64,
    /// Whether a skill has been expanded into the conversation this session.
    pub activated_skill_context_observed: bool,
    /// The provider the request will be routed to.
    pub provider: &'a str,
    /// The catalog identifier of the model the request will be routed to.
    pub model: &'a str,
    /// The model's context window, in tokens.
    pub context_window_tokens: usize,
}

/// Measures a request locally, attributing every token to a category.
#[must_use]
pub fn build_local_snapshot(input: ContextWindowInput<'_>) -> ContextWindowSnapshot {
    let mut builder = BreakdownBuilder::default();
    let mut warnings = Vec::new();

    add_message_breakdown(&mut builder, &mut warnings, &input);
    add_tool_breakdown(&mut builder, input.tools);
    add_request_control_breakdown(&mut builder, &mut warnings, input.request);

    if input.activated_skill_context_observed {
        warnings.push(ContextWindowWarning {
            code:    ACTIVATED_SKILL_WARNING.to_owned(),
            message: "Activated skill instructions are counted as conversation in this version."
                .to_owned(),
        });
    }

    builder.into_snapshot(SnapshotMeta {
        provider:              input.provider.to_owned(),
        model:                 input.model.to_owned(),
        context_window_tokens: u64::try_from(input.context_window_tokens).unwrap_or(u64::MAX),
        count_method:          ContextWindowCountMethod::LocalEstimate,
        staleness:             ContextWindowStaleness::Live,
        warnings:              dedupe_warnings_by_code(warnings),
    })
}

/// Rebuilds a snapshot around an authoritative total, scaling the breakdown to
/// match it.
///
/// The breakdown stays proportional to the local estimate, because no provider
/// reports tokens per category. Where the total itself is authoritative, the
/// warnings that only describe the estimator's imprecision are dropped: they
/// would tell a reader that the number in front of them is uncertain when it is
/// not.
#[must_use]
pub fn scaled_snapshot(
    local: &ContextWindowSnapshot,
    input_tokens: u64,
    count_method: ContextWindowCountMethod,
    warnings: Vec<ContextWindowWarning>,
) -> ContextWindowSnapshot {
    let warnings = if total_is_provider_authoritative(count_method) {
        warnings
            .into_iter()
            .filter(|warning| !is_local_estimator_warning(&warning.code))
            .collect()
    } else {
        warnings
    };

    ContextWindowSnapshot {
        provider: local.provider.clone(),
        model: local.model.clone(),
        context_window_tokens: local.context_window_tokens,
        input_tokens,
        usage_percent: usage_percent(input_tokens, local.context_window_tokens),
        count_method,
        staleness: ContextWindowStaleness::Live,
        generated_at: SystemTime::now(),
        event_seq: None,
        breakdown: scale_breakdown(&local.breakdown, input_tokens, local.context_window_tokens),
        warnings: dedupe_warnings_by_code(warnings),
    }
}

/// Rebuilds a snapshot around the prompt tokens a response reported.
///
/// A response that reported no prompt tokens leaves the local snapshot alone,
/// because zero is not a measurement.
#[must_use]
pub fn context_window_from_response_usage(
    local: &ContextWindowSnapshot,
    usage: TokenUsage,
) -> ContextWindowSnapshot {
    let input_tokens = usage.prompt();
    if input_tokens == 0 {
        return local.clone();
    }

    scaled_snapshot(
        local,
        input_tokens,
        ContextWindowCountMethod::ResponseUsageScaledBreakdown,
        local.warnings.clone(),
    )
}

/// Whether the displayed total came from the provider rather than the
/// estimator.
const fn total_is_provider_authoritative(method: ContextWindowCountMethod) -> bool {
    matches!(
        method,
        ContextWindowCountMethod::ProviderApiScaledBreakdown
            | ContextWindowCountMethod::ResponseUsageScaledBreakdown
    )
}

/// Whether a warning code is one the local token estimator raises.
///
/// A warning kind lithos-llm adds later has to be named here as well, or a
/// provider-authoritative total will keep carrying it.
fn is_local_estimator_warning(code: &str) -> bool {
    [
        EstimateWarning::Media,
        EstimateWarning::OpaqueContent,
        EstimateWarning::ProviderOptions,
    ]
    .into_iter()
    .any(|warning| warning.code() == code)
}

/// Collapses a warning list to one entry per code, keeping the first of each.
///
/// The estimator dedupes within one message, but a snapshot walks every message
/// in the request, so a long conversation would otherwise accumulate one copy
/// of the same caveat per turn that raised it.
fn dedupe_warnings_by_code(warnings: Vec<ContextWindowWarning>) -> Vec<ContextWindowWarning> {
    let mut seen: HashSet<String> = HashSet::new();
    warnings
        .into_iter()
        .filter(|warning| seen.insert(warning.code.clone()))
        .collect()
}

/// Attributes every message: the system prompt is split into its own text,
/// its memory, and its skills; everything else is conversation.
fn add_message_breakdown(
    builder: &mut BreakdownBuilder,
    warnings: &mut Vec<ContextWindowWarning>,
    input: &ContextWindowInput<'_>,
) {
    let mut system_prompt_seen = false;

    for message in input.request.messages() {
        let estimate = message_tokens(message);
        warnings.extend(warnings_from_estimate(&estimate));

        if !system_prompt_seen
            && message.role() == Role::System
            && message_text_is(message, input.system_prompt)
        {
            system_prompt_seen = true;
            let attributed = input.memory_tokens.saturating_add(input.skills_tokens);
            builder.add(
                ContextWindowCategory::SystemPrompt,
                estimate.tokens().saturating_sub(attributed),
            );
            builder.add(ContextWindowCategory::Memory, input.memory_tokens);
            builder.add(ContextWindowCategory::Skills, input.skills_tokens);
        } else {
            builder.add(ContextWindowCategory::Conversation, estimate.tokens());
        }
    }
}

/// Attributes every tool definition to the category its source implies.
fn add_tool_breakdown(builder: &mut BreakdownBuilder, tools: &[ToolDefinitionWithSource]) {
    for tool in tools {
        let tokens = tool_definition_tokens(&tool.definition);
        let category = match &tool.source {
            ToolSource::Mcp { .. } => ContextWindowCategory::McpTools,
            ToolSource::Skill => ContextWindowCategory::Skills,
            _ => ContextWindowCategory::Tools,
        };
        builder.add(category, tokens);
    }
}

/// Attributes the request settings that reach the prompt.
fn add_request_control_breakdown(
    builder: &mut BreakdownBuilder,
    warnings: &mut Vec<ContextWindowWarning>,
    request: &Request,
) {
    let estimate = request_control_tokens(request);
    warnings.extend(warnings_from_estimate(&estimate));
    builder.add(ContextWindowCategory::Other, estimate.tokens());
}

/// The caveats an estimate raised, as snapshot warnings.
fn warnings_from_estimate(estimate: &TokenEstimate) -> Vec<ContextWindowWarning> {
    estimate
        .warnings()
        .map(|warning| ContextWindowWarning {
            code:    warning.code().to_owned(),
            message: warning.to_string(),
        })
        .collect()
}

/// Whether the message's text parts, joined, spell exactly `expected`.
///
/// Compared part by part rather than by joining into a fresh `String`,
/// because this runs against the full system prompt on every snapshot.
fn message_text_is(message: &Message, expected: &str) -> bool {
    let mut rest = expected;
    for part in message.content() {
        if let ContentPart::Text { text } = part {
            match rest.strip_prefix(text.as_str()) {
                Some(remaining) => rest = remaining,
                None => return false,
            }
        }
    }
    rest.is_empty()
}

/// The tokens the memory files contribute to the system prompt.
///
/// A session computes this once at initialization and carries the number into
/// every [`ContextWindowInput`], because the memory is fixed for the session's
/// life.
#[must_use]
pub fn memory_prompt_tokens(memory: &[MemoryDocument]) -> u64 {
    text_tokens(&memory_prompt_suffix(memory))
}

/// The tokens the skills section contributes to the system prompt.
///
/// Fixed per session, like [`memory_prompt_tokens`].
#[must_use]
pub fn skills_prompt_tokens(skills: &[Skill], vocabulary: ToolVocabulary) -> u64 {
    text_tokens(&skills_prompt_suffix(skills, vocabulary))
}

/// The memory text a profile appends to the system prompt.
///
/// This mirrors what prompt assembly does, so the tokens attributed to memory
/// are the tokens memory actually contributed.
fn memory_prompt_suffix(memory: &[MemoryDocument]) -> String {
    if memory.is_empty() {
        return String::new();
    }

    format!(
        "\n\n{}",
        memory
            .iter()
            .map(|document| document.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    )
}

/// The skills section a profile appends to the system prompt.
fn skills_prompt_suffix(skills: &[Skill], vocabulary: ToolVocabulary) -> String {
    let section = format_skills_prompt_section(skills, vocabulary);
    if section.is_empty() {
        String::new()
    } else {
        format!("\n\n{section}")
    }
}

/// Accumulates tokens per category, in category order.
#[derive(Default)]
struct BreakdownBuilder {
    tokens: BTreeMap<ContextWindowCategory, u64>,
}

impl BreakdownBuilder {
    /// Adds tokens to a category, ignoring a zero contribution so the breakdown
    /// names only the categories that are actually present.
    fn add(&mut self, category: ContextWindowCategory, tokens: u64) {
        if tokens == 0 {
            return;
        }
        self.tokens
            .entry(category)
            .and_modify(|total| *total = total.saturating_add(tokens))
            .or_insert(tokens);
    }

    fn into_snapshot(self, meta: SnapshotMeta) -> ContextWindowSnapshot {
        let input_tokens = self.tokens.values().copied().sum::<u64>();
        let breakdown = self
            .tokens
            .into_iter()
            .map(|(category, tokens)| ContextWindowBreakdownItem {
                category,
                tokens,
                usage_percent: usage_percent(tokens, meta.context_window_tokens),
            })
            .collect();

        ContextWindowSnapshot {
            provider: meta.provider,
            model: meta.model,
            context_window_tokens: meta.context_window_tokens,
            input_tokens,
            usage_percent: usage_percent(input_tokens, meta.context_window_tokens),
            count_method: meta.count_method,
            staleness: meta.staleness,
            generated_at: SystemTime::now(),
            event_seq: None,
            breakdown,
            warnings: meta.warnings,
        }
    }
}

/// What a finished breakdown is wrapped in.
struct SnapshotMeta {
    provider:              String,
    model:                 String,
    context_window_tokens: u64,
    count_method:          ContextWindowCountMethod,
    staleness:             ContextWindowStaleness,
    warnings:              Vec<ContextWindowWarning>,
}

/// Scales a breakdown so it sums to `target_total`.
///
/// Rounding leftovers land in the last bucket, so the parts always add up to
/// the total a reader is shown. A local estimate of zero cannot be apportioned,
/// so the whole total becomes [`ContextWindowCategory::Other`].
fn scale_breakdown(
    breakdown: &[ContextWindowBreakdownItem],
    target_total: u64,
    context_window_tokens: u64,
) -> Vec<ContextWindowBreakdownItem> {
    let local_total = breakdown.iter().map(|item| item.tokens).sum::<u64>();
    if local_total == 0 {
        return (target_total > 0)
            .then(|| ContextWindowBreakdownItem {
                category:      ContextWindowCategory::Other,
                tokens:        target_total,
                usage_percent: usage_percent(target_total, context_window_tokens),
            })
            .into_iter()
            .collect();
    }

    let mut scaled: Vec<_> = breakdown
        .iter()
        .map(|item| {
            let tokens = u128::from(item.tokens).saturating_mul(u128::from(target_total))
                / u128::from(local_total);
            let tokens = u64::try_from(tokens).unwrap_or(u64::MAX);
            ContextWindowBreakdownItem {
                category: item.category,
                tokens,
                usage_percent: usage_percent(tokens, context_window_tokens),
            }
        })
        .collect();

    let allocated: u64 = scaled.iter().map(|item| item.tokens).sum();
    if let Some(last) = scaled.last_mut() {
        let leftover = target_total.saturating_sub(allocated);
        if leftover > 0 {
            last.tokens = last.tokens.saturating_add(leftover);
            last.usage_percent = usage_percent(last.tokens, context_window_tokens);
        }
    }
    scaled
}

/// A token count as a percentage of the window, with an unknown window
/// reported as zero rather than as a division by zero.
fn usage_percent(tokens: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        tokens as f64 * 100.0 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::{ToolChoice, ToolDefinition};
    use serde_json::json;

    use super::*;

    fn request(messages: Vec<Message>, tools: Vec<ToolDefinition>) -> Request {
        let mut builder = Request::builder().model("test/model-a");
        for message in messages {
            builder = builder.message(message);
        }
        for tool in tools {
            builder = builder.tool(tool);
        }
        builder
            .tool_choice(ToolChoice::Auto)
            .build()
            .expect("the request builds")
    }

    fn tool(name: &str, source: ToolSource) -> ToolDefinitionWithSource {
        ToolDefinitionWithSource {
            definition: ToolDefinition::function(
                name,
                format!("{name} description"),
                json!({ "type": "object" }),
            ),
            source,
        }
    }

    fn memory() -> Vec<MemoryDocument> {
        vec![MemoryDocument {
            path:         "/repo/AGENTS.md".to_owned(),
            content:      "memory instructions".to_owned(),
            byte_count:   19,
            loaded_bytes: 19,
            truncated:    false,
        }]
    }

    fn skills() -> Vec<Skill> {
        vec![Skill {
            name:        "commit".to_owned(),
            description: "Commit changes".to_owned(),
            template:    "commit template".to_owned(),
        }]
    }

    fn local_snapshot(breakdown: Vec<ContextWindowBreakdownItem>) -> ContextWindowSnapshot {
        let input_tokens = breakdown.iter().map(|item| item.tokens).sum();
        ContextWindowSnapshot {
            provider: "test".to_owned(),
            model: "model-a".to_owned(),
            context_window_tokens: 1_000,
            input_tokens,
            usage_percent: usage_percent(input_tokens, 1_000),
            count_method: ContextWindowCountMethod::LocalEstimate,
            staleness: ContextWindowStaleness::Live,
            generated_at: SystemTime::now(),
            event_seq: None,
            breakdown,
            warnings: Vec::new(),
        }
    }

    fn item(category: ContextWindowCategory, tokens: u64) -> ContextWindowBreakdownItem {
        ContextWindowBreakdownItem {
            category,
            tokens,
            usage_percent: usage_percent(tokens, 1_000),
        }
    }

    fn codes(snapshot: &ContextWindowSnapshot) -> Vec<&str> {
        snapshot
            .warnings
            .iter()
            .map(|warning| warning.code.as_str())
            .collect()
    }

    #[test]
    fn a_local_snapshot_separates_prompt_memory_skills_tools_and_conversation() {
        let memory = memory();
        let skills = skills();
        let system_prompt = format!(
            "core prompt{}{}",
            memory_prompt_suffix(&memory),
            skills_prompt_suffix(&skills, ToolVocabulary::Canonical)
        );
        let tools = vec![
            tool("read_file", ToolSource::Native),
            tool("mcp__server__search", ToolSource::Mcp {
                server_name:   "server".to_owned(),
                original_name: "search".to_owned(),
            }),
            tool("use_skill", ToolSource::Skill),
        ];
        let built = request(
            vec![
                Message::text(Role::System, system_prompt.clone()),
                Message::text(Role::User, "hello"),
            ],
            tools.iter().map(|tool| tool.definition.clone()).collect(),
        );

        let snapshot = build_local_snapshot(ContextWindowInput {
            request: &built,
            tools: &tools,
            system_prompt: &system_prompt,
            memory_tokens: memory_prompt_tokens(&memory),
            skills_tokens: skills_prompt_tokens(&skills, ToolVocabulary::Canonical),
            activated_skill_context_observed: true,
            provider: "test",
            model: "model-a",
            context_window_tokens: 100_000,
        });

        let categories: Vec<_> = snapshot
            .breakdown
            .iter()
            .map(|item| item.category)
            .collect();
        for expected in [
            ContextWindowCategory::SystemPrompt,
            ContextWindowCategory::Tools,
            ContextWindowCategory::McpTools,
            ContextWindowCategory::Skills,
            ContextWindowCategory::Memory,
            ContextWindowCategory::Conversation,
        ] {
            assert!(categories.contains(&expected), "missing {expected}");
        }
        assert_eq!(
            snapshot
                .breakdown
                .iter()
                .map(|item| item.tokens)
                .sum::<u64>(),
            snapshot.input_tokens
        );
        assert_eq!(
            snapshot.count_method,
            ContextWindowCountMethod::LocalEstimate
        );
        assert!(codes(&snapshot).contains(&ACTIVATED_SKILL_WARNING));
    }

    #[test]
    fn a_breakdown_is_ordered_by_category() {
        let memory = memory();
        let skills = skills();
        let system_prompt = format!(
            "core prompt{}{}",
            memory_prompt_suffix(&memory),
            skills_prompt_suffix(&skills, ToolVocabulary::Canonical)
        );
        let tools = vec![tool("read_file", ToolSource::Native)];
        let built = request(
            vec![
                Message::text(Role::System, system_prompt.clone()),
                Message::text(Role::User, "hello"),
            ],
            tools.iter().map(|tool| tool.definition.clone()).collect(),
        );

        let snapshot = build_local_snapshot(ContextWindowInput {
            request: &built,
            tools: &tools,
            system_prompt: &system_prompt,
            memory_tokens: memory_prompt_tokens(&memory),
            skills_tokens: skills_prompt_tokens(&skills, ToolVocabulary::Canonical),
            activated_skill_context_observed: false,
            provider: "test",
            model: "model-a",
            context_window_tokens: 100_000,
        });

        let categories: Vec<_> = snapshot
            .breakdown
            .iter()
            .map(|item| item.category)
            .collect();
        let mut sorted = categories.clone();
        sorted.sort_unstable();
        assert_eq!(categories, sorted);
        assert!(!codes(&snapshot).contains(&ACTIVATED_SKILL_WARNING));
    }

    #[test]
    fn the_skills_suffix_names_the_tool_the_profile_exposes() {
        let skills = skills();

        assert!(skills_prompt_suffix(&skills, ToolVocabulary::Canonical).contains("`use_skill`"));
        assert!(skills_prompt_suffix(&skills, ToolVocabulary::KimiCode).contains("`Skill`"));
    }

    #[test]
    fn a_scaled_breakdown_adds_up_to_the_provider_total() {
        let local = local_snapshot(vec![
            item(ContextWindowCategory::SystemPrompt, 10),
            item(ContextWindowCategory::Conversation, 20),
        ]);

        let scaled = scaled_snapshot(
            &local,
            101,
            ContextWindowCountMethod::ProviderApiScaledBreakdown,
            Vec::new(),
        );

        assert_eq!(scaled.input_tokens, 101);
        assert_eq!(
            scaled.breakdown.iter().map(|item| item.tokens).sum::<u64>(),
            101
        );
    }

    #[test]
    fn an_empty_local_breakdown_scales_into_one_unattributed_line() {
        let local = local_snapshot(Vec::new());

        let scaled = scaled_snapshot(
            &local,
            64,
            ContextWindowCountMethod::ProviderApiScaledBreakdown,
            Vec::new(),
        );

        assert_eq!(scaled.breakdown, vec![item(
            ContextWindowCategory::Other,
            64
        )]);
    }

    #[test]
    fn an_empty_local_breakdown_and_no_tokens_scale_into_nothing() {
        let scaled = scaled_snapshot(
            &local_snapshot(Vec::new()),
            0,
            ContextWindowCountMethod::ProviderApiScaledBreakdown,
            Vec::new(),
        );

        assert!(scaled.breakdown.is_empty());
    }

    fn estimator_noise_and_one_real_warning() -> Vec<ContextWindowWarning> {
        vec![
            ContextWindowWarning {
                code:    EstimateWarning::OpaqueContent.code().to_owned(),
                message: "noise".to_owned(),
            },
            ContextWindowWarning {
                code:    EstimateWarning::Media.code().to_owned(),
                message: "noise".to_owned(),
            },
            ContextWindowWarning {
                code:    ACTIVATED_SKILL_WARNING.to_owned(),
                message: "kept".to_owned(),
            },
        ]
    }

    #[test]
    fn a_provider_counted_total_drops_estimator_noise() {
        let local = local_snapshot(vec![item(ContextWindowCategory::Conversation, 50)]);

        let scaled = scaled_snapshot(
            &local,
            100,
            ContextWindowCountMethod::ProviderApiScaledBreakdown,
            estimator_noise_and_one_real_warning(),
        );

        assert_eq!(codes(&scaled), [ACTIVATED_SKILL_WARNING]);
    }

    #[test]
    fn a_response_counted_total_drops_estimator_noise() {
        let local = local_snapshot(vec![item(ContextWindowCategory::Conversation, 50)]);

        let scaled = scaled_snapshot(
            &local,
            100,
            ContextWindowCountMethod::ResponseUsageScaledBreakdown,
            estimator_noise_and_one_real_warning(),
        );

        assert_eq!(codes(&scaled), [ACTIVATED_SKILL_WARNING]);
    }

    #[test]
    fn a_locally_estimated_total_keeps_estimator_noise() {
        let local = local_snapshot(vec![item(ContextWindowCategory::Conversation, 50)]);

        let scaled = scaled_snapshot(
            &local,
            100,
            ContextWindowCountMethod::LocalEstimate,
            estimator_noise_and_one_real_warning(),
        );

        assert_eq!(codes(&scaled), [
            EstimateWarning::OpaqueContent.code(),
            EstimateWarning::Media.code(),
            ACTIVATED_SKILL_WARNING,
        ]);
    }

    #[test]
    fn a_repeated_warning_code_is_reported_once() {
        let local = local_snapshot(vec![item(ContextWindowCategory::Conversation, 50)]);
        let repeated: Vec<_> = (0..5)
            .map(|turn| ContextWindowWarning {
                code:    EstimateWarning::OpaqueContent.code().to_owned(),
                message: format!("turn {turn}"),
            })
            .collect();

        let scaled = scaled_snapshot(
            &local,
            100,
            ContextWindowCountMethod::LocalEstimate,
            repeated,
        );

        assert_eq!(codes(&scaled), [EstimateWarning::OpaqueContent.code()]);
    }

    #[test]
    fn reported_usage_replaces_the_estimated_total() {
        let local = local_snapshot(vec![
            item(ContextWindowCategory::SystemPrompt, 10),
            item(ContextWindowCategory::Conversation, 20),
        ]);

        let updated = context_window_from_response_usage(&local, TokenUsage {
            input:       100,
            output:      7,
            cache_read:  20,
            cache_write: 5,
            reasoning:   3,
        });

        // Prompt tokens only: output and reasoning never occupied the window.
        assert_eq!(updated.input_tokens, 125);
        assert_eq!(
            updated.count_method,
            ContextWindowCountMethod::ResponseUsageScaledBreakdown
        );
        assert_eq!(
            updated
                .breakdown
                .iter()
                .map(|item| item.tokens)
                .sum::<u64>(),
            125
        );
    }

    #[test]
    fn usage_without_prompt_tokens_leaves_the_local_snapshot_alone() {
        let local = local_snapshot(vec![item(ContextWindowCategory::Conversation, 20)]);

        let updated = context_window_from_response_usage(&local, TokenUsage {
            output: 12,
            ..TokenUsage::default()
        });

        assert_eq!(updated, local);
    }

    #[test]
    fn a_window_of_zero_reports_no_usage_rather_than_dividing_by_it() {
        assert!((usage_percent(50, 0) - 0.0).abs() < f64::EPSILON);
        assert!((usage_percent(50, 200) - 25.0).abs() < f64::EPSILON);
    }
}
