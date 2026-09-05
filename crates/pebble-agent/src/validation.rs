//! Structural validation of model-supplied tool arguments.

use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;

use lithos_llm::types::{ContentPart, Message, Role, ToolDefinitionKind};
use serde_json::Value;

use crate::LifecycleError;

/// A structural mismatch between tool arguments and their input schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolArgumentsError {
    message: String,
}

impl fmt::Display for ToolArgumentsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for ToolArgumentsError {}

/// Checks the structural subset of JSON Schema used by agent tools.
///
/// Object and array shape, required properties, and declared property types
/// are checked recursively. Other JSON Schema keywords remain the executor's
/// responsibility. Provider-format custom tools are not checked.
///
/// # Errors
///
/// Returns every structural mismatch found in the arguments.
pub fn validate_tool_arguments(
    kind: &ToolDefinitionKind,
    arguments: &Value,
) -> Result<(), ToolArgumentsError> {
    let ToolDefinitionKind::Function { input_schema } = kind else {
        return Ok(());
    };

    let mut problems = Vec::new();
    check_against_schema(input_schema, arguments, "arguments", &mut problems);
    if problems.is_empty() {
        return Ok(());
    }

    Err(ToolArgumentsError {
        message: format!("Tool argument validation failed: {}", problems.join("; ")),
    })
}

/// Checks the structural subset of JSON Schema that tool executors can rely on.
fn check_against_schema(schema: &Value, value: &Value, path: &str, problems: &mut Vec<String>) {
    let Some(schema) = schema.as_object() else {
        return;
    };
    if schema.is_empty() {
        return;
    }

    if let Some(expected) = schema.get("type")
        && !matches_type(expected, value)
    {
        problems.push(format!(
            "{path}: expected {}, got {}",
            render_type(expected),
            type_name(value)
        ));
        return;
    }

    if let Some(members) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for name in required.iter().filter_map(Value::as_str) {
                if !members.contains_key(name) {
                    problems.push(format!("{path}: missing required property \"{name}\""));
                }
            }
        }

        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (name, property_schema) in properties {
                if let Some(member) = members.get(name) {
                    check_against_schema(
                        property_schema,
                        member,
                        &format!("{path}.{name}"),
                        problems,
                    );
                }
            }
        }
    }

    if let (Some(items), Some(item_schema)) = (value.as_array(), schema.get("items")) {
        for (index, item) in items.iter().enumerate() {
            check_against_schema(item_schema, item, &format!("{path}[{index}]"), problems);
        }
    }
}

fn matches_type(expected: &Value, value: &Value) -> bool {
    match expected {
        Value::String(name) => is_type(name, value),
        Value::Array(names) => names
            .iter()
            .filter_map(Value::as_str)
            .any(|name| is_type(name, value)),
        _ => true,
    }
}

fn is_type(name: &str, value: &Value) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64() || is_whole_float(value),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Whether `value` is a float with nothing after the point.
///
/// JSON has one number type, and some providers spell an integer as `2000.0`.
/// JSON Schema counts any number with a zero fractional part as an integer, so
/// pebble does too; `2.5` is not one. Infinity and NaN have no fractional
/// part to speak of and are not integers either.
fn is_whole_float(value: &Value) -> bool {
    value
        .as_f64()
        .is_some_and(|number| number.is_finite() && number.fract() == 0.0)
}

fn render_type(expected: &Value) -> String {
    match expected {
        Value::String(name) => name.clone(),
        Value::Array(names) => names
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" or "),
        other => other.to_string(),
    }
}

const fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Validates tool-call pairing in an application-prepared request view.
pub(crate) fn validate_context(messages: &[Message]) -> Result<(), LifecycleError> {
    let mut pending = HashSet::new();
    for message in messages {
        if !pending.is_empty() && message.role() != Role::Tool {
            return Err(LifecycleError::new(
                "prepared context separates tool calls from their results",
            ));
        }
        for part in message.content() {
            match part {
                ContentPart::ToolCall(call)
                    if (message.role() != Role::Assistant || !pending.insert(call.id.as_str())) =>
                {
                    return Err(LifecycleError::new(
                        "prepared context contains an invalid or duplicate tool call",
                    ));
                }
                ContentPart::ToolResult(result)
                    if message.role() != Role::Tool
                        || !pending.remove(result.tool_call_id.as_str()) =>
                {
                    return Err(LifecycleError::new(
                        "prepared context contains an unmatched tool result",
                    ));
                }
                _ => {}
            }
        }
    }
    if !pending.is_empty() {
        return Err(LifecycleError::new(
            "prepared context contains unanswered tool calls",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn limit_schema() -> ToolDefinitionKind {
        ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            }),
        }
    }

    #[test]
    fn an_integer_may_be_spelled_as_a_whole_float() {
        let kind = limit_schema();

        assert!(validate_tool_arguments(&kind, &json!({"limit": 2000})).is_ok());
        assert!(validate_tool_arguments(&kind, &json!({"limit": 2000.0})).is_ok());
        assert!(validate_tool_arguments(&kind, &json!({"limit": -3.0})).is_ok());
    }

    #[test]
    fn a_fraction_is_not_an_integer() {
        let kind = limit_schema();

        let error = validate_tool_arguments(&kind, &json!({"limit": 2.5}))
            .expect_err("2.5 has a fractional part");

        assert_eq!(
            error.to_string(),
            "Tool argument validation failed: arguments.limit: expected integer, got number"
        );
    }

    #[test]
    fn an_empty_schema_accepts_anything() {
        for schema in [json!(null), json!({})] {
            let kind = ToolDefinitionKind::Function {
                input_schema: schema,
            };
            assert!(validate_tool_arguments(&kind, &json!("anything")).is_ok());
        }
    }

    #[test]
    fn a_custom_tool_is_not_schema_checked() {
        let kind = ToolDefinitionKind::Custom {
            format: json!({"type": "grammar"}),
        };
        assert!(validate_tool_arguments(&kind, &json!("*** Begin Patch")).is_ok());
    }

    #[test]
    fn arguments_of_the_wrong_shape_are_rejected() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({"type": "object", "properties": {}}),
        };

        let error = validate_tool_arguments(&kind, &json!("not an object"))
            .expect_err("a string is not an object");

        assert!(
            error
                .to_string()
                .contains("arguments: expected object, got string")
        );
    }

    #[test]
    fn every_missing_required_property_is_named_at_once() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
                "required": ["name", "age"],
            }),
        };

        let error = validate_tool_arguments(&kind, &json!({}))
            .expect_err("both properties are missing")
            .to_string();

        assert!(error.contains("\"name\""), "{error}");
        assert!(error.contains("\"age\""), "{error}");
    }

    #[test]
    fn declared_properties_are_checked_against_their_types() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            }),
        };

        assert!(validate_tool_arguments(&kind, &json!({"count": 3})).is_ok());
        assert!(validate_tool_arguments(&kind, &json!({"count": "three"})).is_err());
        assert!(validate_tool_arguments(&kind, &json!({"other": "three"})).is_ok());
    }

    #[test]
    fn nested_objects_and_array_items_are_checked() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                },
            }),
        };

        assert!(
            validate_tool_arguments(&kind, &json!({"questions": [{"text": "Ship it?"}]})).is_ok()
        );
        let error = validate_tool_arguments(&kind, &json!({"questions": [{"header": "Decision"}]}))
            .expect_err("the item misses a required property")
            .to_string();
        assert!(error.contains("arguments.questions[0]"), "{error}");
    }

    #[test]
    fn a_type_union_accepts_either_member() {
        let kind = ToolDefinitionKind::Function {
            input_schema: json!({
                "type": "object",
                "properties": {"limit": {"type": ["integer", "null"]}},
            }),
        };

        assert!(validate_tool_arguments(&kind, &json!({"limit": 10})).is_ok());
        assert!(validate_tool_arguments(&kind, &json!({"limit": null})).is_ok());
        assert!(validate_tool_arguments(&kind, &json!({"limit": "ten"})).is_err());
    }
}
