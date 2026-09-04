//! Conversion of tool calls written by Pebble record formats 1 and 2.

use std::collections::BTreeMap;

use lithos_llm::types::{ToolArguments, ToolCall, ToolCallKind, ToolInput};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[derive(Deserialize)]
#[serde(untagged)]
enum StoredToolCall {
    Current(ToolCall),
    Legacy(LegacyToolCall),
}

#[derive(Deserialize)]
struct LegacyToolCall {
    id:                String,
    name:              String,
    arguments:         Value,
    #[serde(default)]
    kind:              ToolCallKind,
    #[serde(default)]
    raw_arguments:     Option<String>,
    #[serde(default)]
    provider_metadata: BTreeMap<String, Value>,
}

impl StoredToolCall {
    fn into_call(self) -> Result<ToolCall, &'static str> {
        let legacy = match self {
            Self::Current(call) => return Ok(call),
            Self::Legacy(legacy) => legacy,
        };
        let input = match legacy.kind {
            ToolCallKind::Function => ToolInput::Function(match legacy.raw_arguments {
                Some(raw) => ToolArguments::from_raw(raw),
                None => ToolArguments::from_json(legacy.arguments),
            }),
            ToolCallKind::Custom => ToolInput::Custom(match legacy.raw_arguments {
                Some(raw) => raw,
                None => legacy
                    .arguments
                    .as_str()
                    .ok_or("a stored custom tool input must be text")?
                    .to_owned(),
            }),
            _ => return Err("unknown stored tool call kind"),
        };
        let mut provider_metadata = legacy.provider_metadata;
        if let Some(openai) = provider_metadata
            .get_mut("openai")
            .and_then(Value::as_object_mut)
            && let Some(id) = openai.remove("id")
        {
            openai.entry("item_id").or_insert(id);
        }
        Ok(ToolCall {
            id: legacy.id,
            name: legacy.name,
            input,
            provider_metadata,
        })
    }
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<ToolCall>, D::Error> {
    Option::<Vec<StoredToolCall>>::deserialize(deserializer)?
        .unwrap_or_default()
        .into_iter()
        .map(StoredToolCall::into_call)
        .collect::<Result<_, _>>()
        .map_err(D::Error::custom)
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::ToolInput;
    use serde_json::json;

    use super::StoredToolCall;

    #[test]
    fn old_raw_function_input_is_preserved_even_when_invalid() {
        let stored: StoredToolCall = serde_json::from_value(json!({
            "id": "call", "name": "read", "kind": "function",
            "arguments": {}, "raw_arguments": "{broken"
        }))
        .expect("old call parses");
        let call = stored.into_call().expect("old call converts");
        assert_eq!(call.input.raw(), "{broken");
        assert!(call.input.to_value().is_err());
    }

    #[test]
    fn old_custom_input_remains_text() {
        let stored: StoredToolCall = serde_json::from_value(json!({
            "id": "call", "name": "patch", "kind": "custom", "arguments": "patch text"
        }))
        .expect("old call parses");
        assert_eq!(
            stored.into_call().expect("old call converts").input,
            ToolInput::Custom("patch text".to_owned())
        );
    }
}
