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

pub fn to_tool_call(tool_call: &OaiToolCall) -> ToolCall {
    // An invalid Value fails serialization inside template rendering and errors the whole request,
    // so arguments that are not valid JSON are re-wrapped instead of passed through.
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
    // Keep the client's tool output verbatim, the way llama.cpp and vLLM render OpenAI tool messages. Parsing it as
    // JSON re-serialized objects (different tokens from the original text) and failed outright on arrays, which the
    // chat template cannot run a containment check on.
    ChatContentBlock::ToolCallResult {
        identifier: Some(identifier.to_string()),
        name: None,
        value: serde_json::Value::String(content).into(),
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
            arguments: normalize_arguments(&tool_call.arguments.json),
        },
    }
}

/// The arguments the engine stores for a tool call, as the JSON object text a client expects. When the model's
/// argument JSON was invalid (typically a raw newline inside a string value), the engine keeps the text as a JSON
/// string instead of an object; unwrap that string and escape the control characters so the client can parse it.
/// llama.cpp never surfaces this case because its tool-call grammar forbids invalid JSON, so this levels the engines.
pub fn normalize_arguments(raw: &str) -> String {
    let out = match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::String(inner)) => {
            let repaired = repair_json_strings(&inner);
            match serde_json::from_str::<serde_json::Value>(&repaired) {
                Ok(value) if value.is_object() => repaired,
                _ => raw.to_string(),
            }
        },
        Ok(_) => raw.to_string(),
        Err(_) => repair_json_strings(raw),
    };
    if std::env::var("UZU_SESSION_TRACE").is_ok() && out != raw {
        eprintln!("tool call arguments normalized: {raw:?} -> {out:?}");
    }
    out
}

/// Escape raw control characters inside JSON string literals. Models sometimes emit a literal newline inside a
/// string argument; llama.cpp never surfaces that because its tool-call grammar forbids it, so accept it here too.
/// The input is returned unchanged when it already parses, or when the repair does not make it parse.
pub fn repair_json_strings(raw: &str) -> String {
    if serde_json::from_str::<serde_json::Value>(raw).is_ok() {
        return raw.to_string();
    }
    let mut out = String::with_capacity(raw.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    for ch in raw.chars() {
        if !in_string {
            if ch == '"' {
                in_string = true;
            }
            out.push(ch);
            continue;
        }
        if escaped {
            out.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => {
                escaped = true;
                out.push(ch);
            },
            '"' => {
                in_string = false;
                out.push(ch);
            },
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    if serde_json::from_str::<serde_json::Value>(&out).is_ok() {
        out
    } else {
        raw.to_string()
    }
}

pub fn reply_tool_calls(message: &ChatMessage) -> Option<Vec<OaiToolCall>> {
    let tool_calls = message.tool_calls();
    (!tool_calls.is_empty()).then(|| tool_calls.iter().map(|tool_call| oai_tool_call(None, tool_call)).collect())
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
