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

#[allow(dead_code)]
pub fn format_lfm_tool_preamble(tools: &[&OaiTool]) -> String {
    let items: Vec<String> = tools
        .iter()
        .map(|tool| {
            let mut function = serde_json::json!({
                "name": tool.function.name,
                "description": tool.function.description,
            });
            if let Some(parameters) = &tool.function.parameters {
                function["parameters"] =
                    serde_json::from_value(parameters.json.parse().unwrap_or(serde_json::Value::Null))
                        .unwrap_or(serde_json::Value::Null);
            }
            serde_json::json!({
                "type": "function",
                "function": function,
            })
            .to_string()
        })
        .collect();
    format!("List of tools: <|tool_list_start|>[{}]<|tool_list_end|>", items.join(", "))
}

#[allow(dead_code)]
pub fn format_lfm_assistant_tool_calls(calls: &[OaiToolCall]) -> String {
    let formatted: Vec<String> = calls
        .iter()
        .map(|c| {
            let args_map: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&c.function.arguments).unwrap_or_default();
            let kwargs: Vec<String> = args_map
                .into_iter()
                .map(|(k, v)| format!("{k}={}", serde_json::to_string(&v).unwrap_or_else(|_| "null".into())))
                .collect();
            format!("{}({})", c.function.name, kwargs.join(", "))
        })
        .collect();
    format!("<|tool_call_start|>[{}]<|tool_call_end|>", formatted.join(", "))
}

pub fn parse_lfm_tool_calls(
    raw_text: &str,
    _parameter_types: &ToolParameterTypes,
) -> (Option<Vec<OaiToolCall>>, String) {
    const START: &str = "<|tool_call_start|>";
    const END: &str = "<|tool_call_end|>";
    let mut calls = Vec::new();
    let mut clean = String::with_capacity(raw_text.len());
    let mut rest = raw_text;

    while let Some(start_idx) = rest.find(START) {
        clean.push_str(&rest[..start_idx]);
        let after_start = &rest[start_idx + START.len()..];
        let Some(end_idx) = after_start.find(END) else {
            rest = "";
            break;
        };
        let block = after_start[..end_idx].trim();
        parse_lfm_call_list(block, &mut calls);
        rest = &after_start[end_idx + END.len()..];
    }
    clean.push_str(rest);
    let clean_trimmed = clean
        .replace("<|im_end|>", "")
        .replace("<|text_end|>", "")
        .trim()
        .to_string();
    ((!calls.is_empty()).then_some(calls), clean_trimmed)
}

fn parse_lfm_call_list(block: &str, out: &mut Vec<OaiToolCall>) {
    let inner = block
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(block)
        .trim();
    for item in split_top_level(inner, ',') {
        let item = item.trim();
        let Some(open_paren) = item.find('(') else { continue };
        let Some(close_paren) = item.rfind(')') else { continue };
        if close_paren < open_paren {
            continue;
        }
        let name = item[..open_paren].trim();
        if name.is_empty() {
            continue;
        }
        let args_str = item[open_paren + 1..close_paren].trim();
        let mut args_obj = serde_json::Map::new();
        for kv in split_top_level(args_str, ',') {
            let kv = kv.trim();
            if kv.is_empty() {
                continue;
            }
            if let Some(eq_idx) = kv.find('=') {
                let k = kv[..eq_idx].trim().trim_matches(|c| c == '"' || c == '\'');
                let v_raw = kv[eq_idx + 1..].trim();
                args_obj.insert(k.to_string(), parse_pythonic_value(v_raw));
            }
        }
        let index = out.len();
        out.push(OaiToolCall {
            index: Some(index),
            id: format!("call_{}", index + 1),
            kind: "function".to_string(),
            function: OaiFunctionCall {
                name: name.to_string(),
                arguments: serde_json::Value::Object(args_obj).to_string(),
            },
        });
    }
}

fn split_top_level(s: &str, delim: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escape = false;
    for (idx, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if let Some(q) = quote {
            if ch == '\\' {
                escape = true;
            } else if ch == q {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            c if c == delim && depth == 0 => {
                parts.push(&s[start..idx]);
                start = idx + ch.len_utf8();
            },
            _ => {},
        }
    }
    if start <= s.len() {
        parts.push(&s[start..]);
    }
    parts
}

fn parse_pythonic_value(raw: &str) -> serde_json::Value {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return v;
    }
    if raw == "True" {
        return serde_json::Value::Bool(true);
    }
    if raw == "False" {
        return serde_json::Value::Bool(false);
    }
    if raw == "None" {
        return serde_json::Value::Null;
    }
    if let Some(unquoted) = raw.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        return serde_json::Value::String(unquoted.to_string());
    }
    let normalized = raw.replace('\'', "\"").replace("True", "true").replace("False", "false").replace("None", "null");
    serde_json::from_str(&normalized).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

/// Official LFM3-Audio tool preamble: one JSON definition per line inside `<tools>...</tools>`.
pub fn format_lfm3_audio_tool_preamble(tools: &[&OaiTool]) -> String {
    let items: Vec<String> = tools
        .iter()
        .map(|tool| {
            let mut function = serde_json::json!({
                "name": tool.function.name,
                "description": tool.function.description,
            });
            if let Some(parameters) = &tool.function.parameters {
                function["parameters"] =
                    serde_json::from_value(parameters.json.parse().unwrap_or(serde_json::Value::Null))
                        .unwrap_or(serde_json::Value::Null);
            }
            serde_json::json!({
                "type": "function",
                "function": function,
            })
            .to_string()
        })
        .collect();
    format!("List of tools:\n<tools>\n{}\n</tools>", items.join("\n"))
}

/// Official LFM3-Audio assistant tool call serialization in XML-inline format.
pub fn format_lfm3_audio_assistant_tool_calls(calls: &[OaiToolCall]) -> String {
    let mut blocks = Vec::with_capacity(calls.len());
    for call in calls {
        let args_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&call.function.arguments).unwrap_or_default();
        let mut body = format!("<tool_call>\n<function={}>\n", call.function.name);
        for (k, v) in args_map {
            let val_str = match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            body.push_str(&format!("<parameter={k}>\n{val_str}\n</parameter>\n"));
        }
        body.push_str("</function>\n</tool_call>");
        blocks.push(body);
    }
    blocks.join("\n\n")
}

/// Parse official LFM3-Audio XML-inline `<tool_call><function=...>...</function></tool_call>` blocks,
/// falling back to `<|tool_call_start|>...<|tool_call_end|>` if present.
pub fn parse_lfm3_audio_tool_calls(
    raw_text: &str,
    parameter_types: &ToolParameterTypes,
) -> (Option<Vec<OaiToolCall>>, String) {
    if raw_text.contains("<|tool_call_start|>") {
        return parse_lfm_tool_calls(raw_text, parameter_types);
    }
    const START: &str = "<tool_call>";
    const END: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut clean = String::with_capacity(raw_text.len());
    let mut rest = raw_text;

    while let Some(start_idx) = rest.find(START) {
        clean.push_str(&rest[..start_idx]);
        let after_start = &rest[start_idx + START.len()..];
        let Some(end_idx) = after_start.find(END) else {
            rest = "";
            break;
        };
        let block = after_start[..end_idx].trim();
        if let Some(fn_start) = block.find("<function=") {
            let after_fn = &block[fn_start + "<function=".len()..];
            if let Some(gt_idx) = after_fn.find('>') {
                let name = after_fn[..gt_idx].trim();
                let fn_body = after_fn[gt_idx + 1..]
                    .split("</function>")
                    .next()
                    .unwrap_or("");
                let mut args = serde_json::Map::new();
                let mut p_rest = fn_body;
                while let Some(p_start) = p_rest.find("<parameter=") {
                    let after_p = &p_rest[p_start + "<parameter=".len()..];
                    let Some(p_gt) = after_p.find('>') else { break };
                    let key = after_p[..p_gt].trim().to_string();
                    let after_val = &after_p[p_gt + 1..];
                    let Some(p_end) = after_val.find("</parameter>") else { break };
                    let raw_val = after_val[..p_end].trim();
                    let val = serde_json::from_str::<serde_json::Value>(raw_val)
                        .unwrap_or_else(|_| serde_json::Value::String(raw_val.to_string()));
                    args.insert(key, val);
                    p_rest = &after_val[p_end + "</parameter>".len()..];
                }
                if !name.is_empty() {
                    let idx = calls.len() + 1;
                    calls.push(OaiToolCall {
                        index: None,
                        id: format!("call_{idx}"),
                        kind: "function".to_string(),
                        function: OaiFunctionCall {
                            name: name.to_string(),
                            arguments: serde_json::Value::Object(args).to_string(),
                        },
                    });
                }
            }
        }
        rest = &after_start[end_idx + END.len()..];
    }
    clean.push_str(rest);
    let clean_trimmed = clean
        .replace("<|im_end|>", "")
        .replace("<|text_end|>", "")
        .trim()
        .to_string();
    ((!calls.is_empty()).then_some(calls), clean_trimmed)
}

#[cfg(test)]
#[path = "../../unit/server/chat_tool_calls_test.rs"]
mod tests;
