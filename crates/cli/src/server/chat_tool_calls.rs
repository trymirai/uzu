use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uzu::types::{
    basic::{ToolCall, ToolDescription, ToolFunction, ToolNamespace, Value},
    session::chat::{ChatContentBlock, ChatMessage, ChatRole},
};

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiToolCall {
    // Present only in streaming deltas, per the OpenAI wire format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    pub function: OaiFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Deserialize)]
pub struct OaiTool {
    pub function: OaiToolFunction,
}

#[derive(Deserialize)]
pub struct OaiToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Option<Value>,
}

pub fn to_tool_call(tool_call: &OaiToolCall) -> ToolCall {
    // An invalid Value fails serialization inside template rendering and errors the whole
    // request, so arguments that are not valid JSON are re-wrapped instead of passed through.
    let arguments = &tool_call.function.arguments;
    let json = match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(_) => arguments.clone(),
        Err(_) if arguments.trim().is_empty() => "{}".to_string(),
        Err(_) => serde_json::Value::String(arguments.clone()).to_string(),
    };
    ToolCall {
        identifier: Some(tool_call.id.clone()),
        name: tool_call.function.name.clone(),
        arguments: Value {
            json,
        },
    }
}

pub fn tool_call_result_block(
    identifier: &str,
    content: String,
) -> ChatContentBlock {
    let value = serde_json::from_str::<serde_json::Value>(&content).unwrap_or(serde_json::Value::String(content));
    ChatContentBlock::ToolCallResult {
        identifier: Some(identifier.to_string()),
        name: None,
        value: value.into(),
    }
}

fn tools_message(tools: &[OaiTool]) -> ChatMessage {
    let descriptions = tools
        .iter()
        .map(|tool| ToolDescription::Function {
            tool_function: ToolFunction {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
                return_definition: None,
            },
        })
        .collect();
    ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
        name: "functions".to_string(),
        description: None,
        tools: descriptions,
    }])
}

// OpenAI tool messages carry only tool_call_id, but some chat templates render tool
// results as {name, response} pairs, so the name is recovered from the matching call.
pub fn backfill_tool_result_names(messages: &mut [ChatMessage]) {
    let names: HashMap<String, String> = messages
        .iter()
        .flat_map(|message| message.tool_calls())
        .filter_map(|tool_call| Some((tool_call.identifier?, tool_call.name)))
        .collect();
    for message in messages.iter_mut() {
        for block in message.content.iter_mut() {
            if let ChatContentBlock::ToolCallResult {
                identifier: Some(identifier),
                name,
                ..
            } = block
                && name.is_none()
            {
                *name = names.get(identifier.as_str()).cloned();
            }
        }
    }
}

pub fn insert_tools_message(
    messages: &mut Vec<ChatMessage>,
    tools: Option<&[OaiTool]>,
) {
    let Some(tools) = tools.filter(|tools| !tools.is_empty()) else {
        return;
    };
    let position = messages.iter().position(|message| message.role != (ChatRole::System {})).unwrap_or(messages.len());
    messages.insert(position, tools_message(tools));
}

pub fn oai_tool_call(
    index: Option<usize>,
    tool_call: &ToolCall,
) -> OaiToolCall {
    OaiToolCall {
        index,
        id: tool_call.identifier.clone().unwrap_or_default(),
        kind: "function".to_string(),
        function: OaiFunctionCall {
            name: tool_call.name.clone(),
            arguments: tool_call.arguments.json.clone(),
        },
    }
}

pub fn reply_tool_calls(message: &ChatMessage) -> Option<Vec<OaiToolCall>> {
    let tool_calls = message.tool_calls();
    (!tool_calls.is_empty()).then(|| tool_calls.iter().map(|tool_call| oai_tool_call(None, tool_call)).collect())
}

pub fn tool_call_deltas(
    tool_calls: &[ToolCall],
    emitted: usize,
) -> Vec<OaiToolCall> {
    tool_calls
        .iter()
        .enumerate()
        .skip(emitted)
        .map(|(index, tool_call)| oai_tool_call(Some(index), tool_call))
        .collect()
}

#[cfg(test)]
#[path = "../../unit/server/chat_tool_calls_test.rs"]
mod tests;
