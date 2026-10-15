use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;
pub use uzu::session::chat::tool_call::{ToolParameterTypes, coerce_tool_call, parse_parameter_text};
use uzu::{
    session::chat::{ChatSessionError, normalize_tool_call_arguments, tool_call::ToolCallArgumentStream},
    types::{
        basic::{ToolCall, ToolDescription, ToolFunction, ToolNamespace, Value},
        session::chat::{ChatContentBlock, ChatMessage, ChatRole},
    },
};

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiToolCall {
    // Present only in streaming deltas, per the OpenAI wire format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    // Empty after announcement; finish carries metadata only if no announcement occurred.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    #[serde(rename = "type", default, skip_serializing_if = "String::is_empty")]
    pub kind: String,
    pub function: OaiFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiFunctionCall {
    #[serde(default, skip_serializing_if = "String::is_empty")]
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

#[derive(Deserialize)]
#[serde(untagged)]
enum OaiToolChoice {
    Mode(String),
    Function {
        function: OaiToolChoiceFunction,
    },
}

#[derive(Deserialize)]
struct OaiToolChoiceFunction {
    name: String,
}

// Applies tool_choice to the declared tools: "none" hides all of them and a forced function
// exposes only the named one. Declarations are the only lever a local model has, so
// "required" cannot compel a call and keeps the full set like "auto".
pub fn choose_tools<'t>(
    tools: Option<&'t [OaiTool]>,
    tool_choice: Option<&serde_json::Value>,
) -> Result<Vec<&'t OaiTool>, String> {
    let tools: Vec<&OaiTool> = tools.unwrap_or_default().iter().collect();
    let Some(tool_choice) = tool_choice else {
        return Ok(tools);
    };
    let choice = serde_json::from_value::<OaiToolChoice>(tool_choice.clone())
        .map_err(|error| format!("tool_choice is not a recognized value: {error}"))?;
    match choice {
        OaiToolChoice::Mode(mode) => match mode.as_str() {
            "none" => Ok(vec![]),
            "auto" | "required" => Ok(tools),
            other => {
                Err(format!("tool_choice must be \"none\", \"auto\", \"required\" or a function object, got {other:?}"))
            },
        },
        OaiToolChoice::Function {
            function,
        } => {
            let selected: Vec<&OaiTool> =
                tools.into_iter().filter(|tool| tool.function.name == function.name).collect();
            if selected.is_empty() {
                return Err(format!("tool_choice names function {:?} but tools does not declare it", function.name));
            }
            Ok(selected)
        },
    }
}

pub fn tool_parameter_types(tools: Option<&[OaiTool]>) -> ToolParameterTypes {
    ToolParameterTypes::from_parameters(tools.unwrap_or_default().iter().filter_map(|tool| {
        tool.function.parameters.as_ref().map(|parameters| (tool.function.name.as_str(), parameters))
    }))
}

pub fn to_tool_call(tool_call: &OaiToolCall) -> ToolCall {
    // Templates render arguments as key/value pairs; normalization guarantees
    // an object no matter what the client echoes back.
    let arguments = &tool_call.function.arguments;
    let json = if arguments.trim().is_empty() {
        "{}".to_string()
    } else {
        normalize_tool_call_arguments(Value {
            json: arguments.clone(),
        })
        .json
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
    let value = serde_json::Value::String(content);
    ChatContentBlock::ToolCallResult {
        identifier: Some(identifier.to_string()),
        name: None,
        value: value.into(),
    }
}

fn tools_message(tools: &[&OaiTool]) -> ChatMessage {
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
    tools: &[&OaiTool],
) {
    if tools.is_empty() {
        return;
    }
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

pub fn reply_tool_calls(
    message: &ChatMessage,
    types: &ToolParameterTypes,
) -> Option<Vec<OaiToolCall>> {
    let tool_calls = message.tool_calls();
    (!tool_calls.is_empty())
        .then(|| tool_calls.iter().map(|tool_call| oai_tool_call(None, &coerce_tool_call(tool_call, types))).collect())
}

// Bare-JSON formats (e.g. llama-3) stream a tool call as ordinary text and only rewrite
// it into a ToolCall block when the turn finishes, so JSON-looking text must be withheld
// from delta.content until then; whatever survives in the final message is flushed at the end.
pub fn withhold_stream_text(
    has_tools: bool,
    text: &str,
) -> bool {
    let trimmed = text.trim_start();
    has_tools && (trimmed.is_empty() || trimmed.starts_with('{'))
}

pub struct ToolCallStreamer {
    id: String,
    announced: bool,
    arguments: Option<ToolCallArgumentStream>,
}

impl ToolCallStreamer {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            announced: false,
            arguments: None,
        }
    }

    pub fn update(
        &mut self,
        index: usize,
        raw: &str,
        types: &ToolParameterTypes,
    ) -> Result<Vec<OaiToolCall>, ChatSessionError> {
        let stream = self.arguments.get_or_insert_with(|| ToolCallArgumentStream::new(types.clone()));
        let delta = stream.update(raw.to_string())?;
        if delta.name.is_empty() {
            return Ok(Vec::new());
        }
        let mut deltas = Vec::new();
        if !self.announced {
            self.announced = true;
            deltas.push(OaiToolCall {
                index: Some(index),
                id: self.id.clone(),
                kind: "function".into(),
                function: OaiFunctionCall {
                    name: delta.name,
                    arguments: String::new(),
                },
            });
        }
        if !delta.arguments.is_empty() {
            deltas.push(OaiToolCall {
                index: Some(index),
                id: String::new(),
                kind: String::new(),
                function: OaiFunctionCall {
                    name: String::new(),
                    arguments: delta.arguments,
                },
            });
        }
        Ok(deltas)
    }

    pub fn finish(
        &mut self,
        index: usize,
        call: &ToolCall,
    ) -> Result<OaiToolCall, ChatSessionError> {
        let stream = self.arguments.get_or_insert_with(|| ToolCallArgumentStream::new(ToolParameterTypes::default()));
        let delta = stream.finish(call.clone())?;
        Ok(OaiToolCall {
            index: Some(index),
            id: if self.announced {
                String::new()
            } else {
                self.id.clone()
            },
            kind: if self.announced {
                String::new()
            } else {
                "function".into()
            },
            function: OaiFunctionCall {
                name: if self.announced {
                    String::new()
                } else {
                    delta.name
                },
                arguments: delta.arguments,
            },
        })
    }
}

#[cfg(test)]
#[path = "../../unit/server/chat_tool_calls_test.rs"]
mod tests;
