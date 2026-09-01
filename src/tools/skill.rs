//! Loading a skill the model asked for by name.
//!
//! A person activates a skill by typing `/name`, which
//! [`expand_skill`](crate::expand_skill) turns into the skill's prompt. A model
//! activates one by calling this tool, and gets the same prompt back as the
//! call's output. Both publish
//! [`SkillActivated`](crate::AgentEvent::SkillActivated), so an application
//! reading the stream sees every activation and which of the two it was.
//!
//! The tool's arguments are the one thing that differs by harness. Kimi Code
//! and Claude 5 call the fields `skill` and `args`; pebble's own name for the
//! first is `skill_name`, and Codex — which has no skill tool of its own, and
//! reads `SKILL.md` through the shell — keeps pebble's.

use std::sync::Arc;

use lithos_llm::types::ToolDefinition;

use crate::skills::{Skill, USER_INPUT_PLACEHOLDER};
use crate::tool::{NativeTool, RegisteredTool, ToolError, ToolVocabulary, required_str};
use crate::types::{AgentEvent, SkillActivationSource, ToolSource};

/// What the model is told the skill tool is for.
const USE_SKILL_DESCRIPTION: &str = "Load a skill's instructions by name. Call this when the \
                                     user's request matches an available skill.";

/// The tool that loads one of `skills`, in pebble's own vocabulary.
///
/// The registry renames it as it is registered, so a session speaking another
/// vocabulary still builds it this way.
#[must_use]
pub fn make_use_skill_tool(skills: Arc<[Skill]>) -> RegisteredTool {
    make_use_skill_tool_for_vocabulary(skills, ToolVocabulary::Canonical)
}

/// The tool that loads one of `skills`, with the arguments `vocabulary`
/// expects.
///
/// One executor serves every vocabulary: only the name of the argument holding
/// the skill's name changes, and the optional `args` string is read the same
/// way wherever it is offered.
#[must_use]
pub fn make_use_skill_tool_for_vocabulary(
    skills: Arc<[Skill]>,
    vocabulary: ToolVocabulary,
) -> RegisteredTool {
    let (name_parameter, parameters) = arguments_for(vocabulary);
    RegisteredTool {
        definition: ToolDefinition::function(
            NativeTool::UseSkill.canonical_name(),
            USE_SKILL_DESCRIPTION,
            parameters,
        ),
        executor:   Arc::new(move |args, ctx| {
            let skills = Arc::clone(&skills);
            Box::pin(async move {
                let name = required_str(&args, name_parameter)?;
                let skill = skills
                    .iter()
                    .find(|skill| skill.name == name)
                    .ok_or_else(|| {
                        ToolError::invalid_arguments(format!("Unknown skill: {name}"))
                    })?;
                // Published only once the skill was found: a lookup that
                // failed activated nothing.
                ctx.emit_agent_event(AgentEvent::SkillActivated {
                    skill_name: name.to_owned(),
                    source:     SkillActivationSource::Tool,
                });
                Ok(render(
                    skill,
                    args.get("args").and_then(|args| args.as_str()),
                ))
            })
        }),
        source:     ToolSource::Skill,
    }
}

/// The argument holding the skill's name, and the schema the model is shown.
fn arguments_for(vocabulary: ToolVocabulary) -> (&'static str, serde_json::Value) {
    match vocabulary {
        // Codex has no skill-loading tool of its own — it reads `SKILL.md`
        // through the shell — so there is no contract to match and it keeps
        // pebble's own.
        ToolVocabulary::Canonical | ToolVocabulary::Codex => (
            "skill_name",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to load (without the / prefix)"
                    }
                },
                "required": ["skill_name"]
            }),
        ),
        ToolVocabulary::Claude5 => (
            "skill",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Exact name of the skill to invoke"
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional argument string to pass to the skill"
                    }
                },
                "required": ["skill"],
                "additionalProperties": false
            }),
        ),
        ToolVocabulary::KimiCode => (
            "skill",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Exact name of the skill to invoke"
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional argument string to pass to the skill"
                    }
                },
                "required": ["skill"]
            }),
        ),
    }
}

/// The skill's prompt, with whatever the model sent along in it.
///
/// A template that says where the input goes gets it there; one that does not
/// keeps the arguments rather than dropping them, under a heading that says
/// what they are.
fn render(skill: &Skill, arguments: Option<&str>) -> String {
    match arguments.filter(|value| !value.is_empty()) {
        Some(value) if skill.template.contains(USER_INPUT_PLACEHOLDER) => {
            skill.template.replace(USER_INPUT_PLACEHOLDER, value)
        }
        Some(value) => format!("{}\n\nARGUMENTS:\n{value}", skill.template),
        None => skill.template.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, PoisonError};

    use serde_json::json;

    use super::*;
    use crate::test_support::MockEnvironment;
    use crate::tool::AgentEventEmitter;
    use crate::tools::testing::{context, schema_of};
    use crate::types::ToolErrorKind;

    /// The skills every test here loads from.
    fn test_skills() -> Arc<[Skill]> {
        Arc::from(vec![
            Skill {
                name:        "commit".to_owned(),
                description: "Create a git commit".to_owned(),
                template:    "Review changes and commit.\n\n{{user_input}}".to_owned(),
            },
            Skill {
                name:        "review".to_owned(),
                description: "Review a change".to_owned(),
                template:    "Read the diff and report what is wrong.".to_owned(),
            },
        ])
    }

    /// An emitter that keeps what it was given.
    #[derive(Default)]
    struct Recorder {
        events: Mutex<Vec<AgentEvent>>,
    }

    impl Recorder {
        fn events(&self) -> Vec<AgentEvent> {
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    impl AgentEventEmitter for Recorder {
        fn emit(&self, event: AgentEvent) {
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event);
        }
    }

    /// Calls the tool built for `vocabulary`, and reports what it published.
    async fn call_with_events(
        vocabulary: ToolVocabulary,
        args: serde_json::Value,
    ) -> (Result<String, ToolError>, Vec<AgentEvent>) {
        let recorder = Arc::new(Recorder::default());
        let tool = make_use_skill_tool_for_vocabulary(test_skills(), vocabulary);
        let context = context(MockEnvironment::default())
            .with_event_emitter(Arc::clone(&recorder) as Arc<dyn AgentEventEmitter>);
        let result = (tool.executor)(args, context).await;
        (result, recorder.events())
    }

    /// Calls the tool built for `vocabulary`.
    async fn call(
        vocabulary: ToolVocabulary,
        args: serde_json::Value,
    ) -> Result<String, ToolError> {
        call_with_events(vocabulary, args).await.0
    }

    #[tokio::test]
    async fn a_skill_the_model_names_is_answered_with_its_template() {
        let tool = make_use_skill_tool(test_skills());

        let output = (tool.executor)(
            json!({"skill_name": "commit"}),
            context(MockEnvironment::default()),
        )
        .await
        .expect("the skill is loaded");

        assert_eq!(output, "Review changes and commit.\n\n{{user_input}}");
    }

    #[tokio::test]
    async fn a_skill_nobody_discovered_is_refused_by_name() {
        let error = call(
            ToolVocabulary::Canonical,
            json!({"skill_name": "nonexistent"}),
        )
        .await
        .expect_err("there is no such skill");

        assert_eq!(error.message(), "Unknown skill: nonexistent");
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    #[tokio::test]
    async fn a_call_that_names_no_skill_is_refused() {
        let error = call(ToolVocabulary::Canonical, json!({}))
            .await
            .expect_err("the tool needs a name");

        assert!(
            error.message().contains("Missing required parameter"),
            "got: {}",
            error.message()
        );
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    /// The Kimi Code contract: `skill` and `args`, and no `skill_name`.
    #[tokio::test]
    async fn kimi_skill_schema_and_args_match_kimi_code() {
        let tool = make_use_skill_tool_for_vocabulary(test_skills(), ToolVocabulary::KimiCode);

        let output = call(
            ToolVocabulary::KimiCode,
            json!({"skill": "commit", "args": "only staged files"}),
        )
        .await
        .expect("the skill is loaded");

        assert!(output.contains("only staged files"), "{output}");
        let properties = &schema_of(&tool)["properties"];
        assert!(properties.get("skill").is_some());
        assert!(properties.get("args").is_some());
        assert!(properties.get("skill_name").is_none());
    }

    /// The Claude 5 contract: the same two fields, only `skill` required, and
    /// nothing else accepted.
    #[tokio::test]
    async fn claude5_skill_schema_uses_skill_and_optional_args() {
        let tool = make_use_skill_tool_for_vocabulary(test_skills(), ToolVocabulary::Claude5);

        let output = call(
            ToolVocabulary::Claude5,
            json!({"skill": "commit", "args": "only staged files"}),
        )
        .await
        .expect("the skill is loaded");

        assert!(output.contains("only staged files"), "{output}");
        let schema = schema_of(&tool);
        assert_eq!(schema["required"], json!(["skill"]));
        assert_eq!(schema["additionalProperties"], json!(false));
        assert!(schema["properties"].get("skill").is_some());
        assert!(schema["properties"].get("args").is_some());
        assert!(schema["properties"].get("skill_name").is_none());
    }

    /// A template with nowhere to put the arguments keeps them anyway: the
    /// model sent them because they matter to the work.
    #[tokio::test]
    async fn arguments_a_template_has_no_place_for_are_appended() {
        let output = call(
            ToolVocabulary::KimiCode,
            json!({"skill": "review", "args": "the auth change"}),
        )
        .await
        .expect("the skill is loaded");

        assert_eq!(
            output,
            "Read the diff and report what is wrong.\n\nARGUMENTS:\nthe auth change"
        );
    }

    /// An empty `args` is the same as none: appending an empty heading would
    /// tell the model there were arguments.
    #[tokio::test]
    async fn empty_arguments_leave_the_template_alone() {
        let output = call(
            ToolVocabulary::KimiCode,
            json!({"skill": "review", "args": ""}),
        )
        .await
        .expect("the skill is loaded");

        assert_eq!(output, "Read the diff and report what is wrong.");
    }

    #[tokio::test]
    async fn loading_a_skill_publishes_the_activation_it_performed() {
        let (result, events) =
            call_with_events(ToolVocabulary::Canonical, json!({"skill_name": "review"})).await;

        assert!(result.is_ok());
        assert_eq!(events, vec![AgentEvent::SkillActivated {
            skill_name: "review".to_owned(),
            source:     SkillActivationSource::Tool,
        }]);
    }

    #[tokio::test]
    async fn a_failed_lookup_publishes_nothing() {
        let (result, events) =
            call_with_events(ToolVocabulary::Canonical, json!({"skill_name": "nope"})).await;

        assert!(result.is_err());
        assert!(events.is_empty(), "got: {events:?}");
    }

    /// The tool is registered as a skill tool, which is what lets the registry
    /// rename it into the model's vocabulary.
    #[test]
    fn the_skill_tool_is_named_and_sourced_the_way_the_registry_expects() {
        for vocabulary in ToolVocabulary::ALL {
            let tool = make_use_skill_tool_for_vocabulary(test_skills(), *vocabulary);

            assert_eq!(tool.definition.name, "use_skill", "{vocabulary:?}");
            assert_eq!(tool.source, ToolSource::Skill, "{vocabulary:?}");
            assert_eq!(
                tool.definition.description,
                "Load a skill's instructions by name. Call this when the user's request matches \
                 an available skill."
            );
        }
    }
}
