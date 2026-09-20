use serde_json::{Value as JsonValue, json};
use shoji::types::{
    basic::{ToolDescription, ToolFunction},
    session::chat::ChatMessage,
};

use crate::error::Error;

pub fn tools_json(messages: &[ChatMessage]) -> Result<String, Error> {
    let tools = collect_tools(messages)?;
    Ok(serde_json::to_string(&tools)?)
}

pub fn collect_tools(messages: &[ChatMessage]) -> Result<Vec<JsonValue>, Error> {
    let mut tools = Vec::new();
    let mut names = Vec::new();
    for message in messages {
        for namespace in message.tool_namespaces() {
            for description in namespace.tools {
                let ToolDescription::Function {
                    tool_function,
                } = description;
                if names.iter().any(|name| name == &tool_function.name) {
                    return Err(Error::DuplicateToolName {
                        name: tool_function.name,
                    });
                }
                names.push(tool_function.name.clone());
                tools.push(function_to_json(tool_function)?);
            }
        }
    }
    tools.sort_by(|left, right| {
        let left_name = left.get("name").and_then(JsonValue::as_str).unwrap_or("");
        let right_name = right.get("name").and_then(JsonValue::as_str).unwrap_or("");
        left_name.cmp(right_name)
    });
    Ok(tools)
}

fn function_to_json(function: ToolFunction) -> Result<JsonValue, Error> {
    let parameters = match function.parameters {
        Some(value) => serde_json::from_str(&value.json)?,
        None => json!({
            "type": "object",
            "properties": {}
        }),
    };
    Ok(json!({
        "name": function.name,
        "description": function.description,
        "parameters": parameters,
    }))
}
