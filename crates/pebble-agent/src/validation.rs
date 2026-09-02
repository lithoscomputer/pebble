//! Structural validation of model-supplied tool arguments.

use std::error::Error as StdError;
use std::fmt;

use lithos_llm::types::ToolDefinitionKind;
use serde_json::Value;

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
}
