use super::ToolCallParser;
use crate::tool_calling::{ToolCall, Value};

pub struct ToolCallJsonParser {
    name_key: String,
    arguments_key: String,
    separator: Option<String>,
}

impl ToolCallJsonParser {
    pub fn new(
        name_key: String,
        arguments_key: String,
        separator: Option<String>,
    ) -> Self {
        Self {
            name_key,
            arguments_key,
            separator,
        }
    }

    fn parse_single_object(
        &self,
        text: String,
    ) -> Option<ToolCall> {
        let json_value: serde_json::Value =
            serde_json::from_str(text.trim()).ok()?;
        let object = json_value.as_object()?;
        let name = object.get(&self.name_key)?.as_str()?.to_string();
        let arguments = Value::from(object.get(&self.arguments_key)?.clone());
        Some(ToolCall::new(name, arguments))
    }
}

impl ToolCallParser for ToolCallJsonParser {
    fn parse(
        &self,
        content: String,
    ) -> Option<Vec<ToolCall>> {
        let text = content.trim().to_string();

        if let Some(tool_call) = self.parse_single_object(text.clone()) {
            return Some(vec![tool_call]);
        }

        if let Some(separator) = &self.separator {
            let mut tool_calls = Vec::new();
            for chunk in text.split(separator.as_str()) {
                match self.parse_single_object(chunk.to_string()) {
                    Some(tool_call) => tool_calls.push(tool_call),
                    None => return None,
                }
            }
            if !tool_calls.is_empty() {
                return Some(tool_calls);
            }
        }

        None
    }
}
