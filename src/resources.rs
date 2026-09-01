//! History, records, instructions, skills, and context management.

pub use crate::compaction::{
    CompactionRequest, ContextEstimate, ContextEstimateMethod, check_context_usage,
    compact_context, estimate_active_context_usage, render_turns_for_summary,
};
pub use crate::context_window::{
    ACTIVATED_SKILL_WARNING, ContextWindowInput, build_local_snapshot,
    context_window_from_response_usage, memory_prompt_tokens, scaled_snapshot,
    skills_prompt_tokens,
};
pub use crate::file_tracker::FileTracker;
pub use crate::history::History;
pub use crate::loop_detection::detect_loop;
pub use crate::memory::{MEMORY_BUDGET_BYTES, MemoryDocument, load_memory};
pub use crate::profile::{
    AgentProfile, DEFAULT_CONTEXT_WINDOW_TOKENS, EnvContext, ModelFacts, SubagentSupport,
};
pub use crate::record::{SESSION_RECORD_FORMAT_VERSION, SessionRecord, StoredMessage};
pub use crate::skills::{
    ExpandedInput, Skill, SkillExpansion, SkillExpansionError, SkillParseError, discover_skills,
    expand_skill, format_skills_prompt_section, parse_skill,
};
pub use crate::task_reminder::{TASK_REMINDER_TEXT, maybe_task_reminder};
pub use crate::types::Message;
