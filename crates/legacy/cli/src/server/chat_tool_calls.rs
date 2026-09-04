use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use uzu::{
    session::chat::normalize_tool_call_arguments,
    types::{
        basic::{ToolCall, ToolDescription, ToolFunction, ToolNamespace, Value, parse_lenient_json},
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

/// Declared JSON-Schema types of each tool parameter, keyed by function name.
/// Tool-call markup cannot carry scalar types — the parser keeps every scalar
/// parameter a string and types JSON-shaped values by their braces — so the
/// declared schema is what restores the wire types clients validate against.
#[derive(Clone, Default)]
pub struct ToolParameterTypes(HashMap<String, HashMap<String, Vec<String>>>);

impl ToolParameterTypes {
    pub fn from_tools(tools: Option<&[OaiTool]>) -> Self {
        let mut functions = HashMap::new();
        for tool in tools.unwrap_or_default() {
            let Some(parameters) = &tool.function.parameters else {
                continue;
            };
            let Ok(schema) = serde_json::from_str::<serde_json::Value>(&parameters.json) else {
                continue;
            };
            let Some(properties) = schema.get("properties").and_then(serde_json::Value::as_object) else {
                continue;
            };
            let parameters = properties
                .iter()
                .filter_map(|(name, property)| {
                    let types = match property.get("type")? {
                        serde_json::Value::String(kind) => vec![kind.clone()],
                        serde_json::Value::Array(kinds) => {
                            kinds.iter().filter_map(|kind| kind.as_str().map(str::to_string)).collect()
                        },
                        _ => return None,
                    };
                    (!types.is_empty()).then(|| (name.clone(), types))
                })
                .collect();
            functions.insert(tool.function.name.clone(), parameters);
        }
        Self(functions)
    }

    fn declared(
        &self,
        function: &str,
        parameter: &str,
    ) -> Option<&[String]> {
        Some(self.0.get(function)?.get(parameter)?.as_slice())
    }
}

fn matches_declared_type(
    value: &serde_json::Value,
    declared: &[String],
) -> bool {
    declared.iter().any(|kind| match kind.as_str() {
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        _ => false,
    })
}

/// Bare scalar text is strict JSON, plus the Python-style booleans some models
/// emit in tool markup (qwen3.5 writes `True`/`False`).
pub fn parse_scalar_text(text: &str) -> Option<serde_json::Value> {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        return Some(parsed);
    }
    match text.trim() {
        text if text.eq_ignore_ascii_case("true") => Some(serde_json::Value::Bool(true)),
        text if text.eq_ignore_ascii_case("false") => Some(serde_json::Value::Bool(false)),
        _ => None,
    }
}

fn coerce_parameter_value(
    value: &serde_json::Value,
    declared: &[String],
) -> Option<serde_json::Value> {
    let declares_string = declared.iter().any(|kind| kind == "string");
    match value {
        // The parser kept the bare markup text as a string; restore the
        // declared type when the text reads as it. A union that includes
        // "string" stays a string: the text is already schema-valid and the
        // intended type is unknowable.
        serde_json::Value::String(text) if !declares_string => {
            let parsed = parse_scalar_text(text)?;
            matches_declared_type(&parsed, declared).then_some(parsed)
        },
        // JSON-shaped markup text was typed by its braces although the
        // parameter is a plain string.
        value if declared == ["string"] && !value.is_string() => Some(serde_json::Value::String(value.to_string())),
        _ => None,
    }
}

/// Restores the declared scalar types the markup could not carry. Applied only
/// at the OpenAI boundary: the session keeps the parser's values so its stored
/// history stays consistent with what the template renders.
pub fn coerce_tool_call(
    tool_call: &ToolCall,
    types: &ToolParameterTypes,
) -> ToolCall {
    let Ok(serde_json::Value::Object(mut object)) = serde_json::from_str(&tool_call.arguments.json) else {
        return tool_call.clone();
    };
    let mut changed = false;
    for (parameter, value) in object.iter_mut() {
        if let Some(declared) = types.declared(&tool_call.name, parameter)
            && let Some(coerced) = coerce_parameter_value(value, declared)
        {
            *value = coerced;
            changed = true;
        }
    }
    if !changed {
        return tool_call.clone();
    }
    ToolCall {
        arguments: Value {
            json: serde_json::Value::Object(object).to_string(),
        },
        ..tool_call.clone()
    }
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

// Streaming state for one in-progress tool call.
//
// OpenAI clients assemble `function.arguments` by concatenating delta fragments and parsing the
// result when the call ends, so every fragment must extend one string. Candidate content comes in
// two shapes:
// - framed (qwen3.5-style): the candidate value is the model's raw `<function=name>` /
//   `<parameter=key>` markup, growing token by token. Arguments are synthesized into compact JSON;
//   the currently open parameter's value streams as escaped string content.
// - bare JSON (llama-3-style): the candidate value is partial JSON text; only the name is
//   announced, since the raw text is not a prefix of the canonical arguments string.
pub struct ToolCallStreamer {
    id: String,
    name_announced: bool,
    key_order: Vec<String>,
    open_text: String,
}

enum FramedParam {
    Complete(String),
    Open(String),
}

struct FramedCall {
    name: Option<String>,
    params: Vec<(String, FramedParam)>,
}

fn parse_framed_call(raw: &str) -> FramedCall {
    let name = raw.find("<function=").and_then(|start| {
        let rest = &raw[start + "<function=".len()..];
        rest.find('>').map(|end| rest[..end].to_string())
    });
    let mut params = Vec::new();
    let mut rest = raw;
    while let Some(start) = rest.find("<parameter=") {
        rest = &rest[start + "<parameter=".len()..];
        let Some(tag_end) = rest.find('>') else {
            break;
        };
        let key = rest[..tag_end].to_string();
        let value_part = &rest[tag_end + 1..];
        let value_part = value_part.strip_prefix('\n').unwrap_or(value_part);
        match value_part.find("\n</parameter>") {
            Some(end) => {
                params.push((key, FramedParam::Complete(value_part[..end].to_string())));
                rest = &value_part[end + "\n</parameter>".len()..];
            },
            None => {
                params.push((key, FramedParam::Open(value_part.to_string())));
                break;
            },
        }
    }
    FramedCall {
        name,
        params,
    }
}

// Mirrors the parser's synthesis rule composed with the schema coercion the
// final call goes through: values starting with `{` or `[` are typed JSON
// unless the parameter is declared a plain string, bare scalars take a
// declared non-string type when they parse as it, everything else is a string.
fn serialize_param_value(
    value: &str,
    declared: Option<&[String]>,
) -> String {
    let trimmed = value.trim_start();
    let json_shaped = trimmed.starts_with('{') || trimmed.starts_with('[');
    if json_shaped && let Some(parsed) = parse_lenient_json(value) {
        if declared.is_some_and(|declared| declared == ["string"]) {
            return serde_json::Value::String(parsed.to_string()).to_string();
        }
        return parsed.to_string();
    }
    if !json_shaped
        && let Some(declared) = declared
        && !declared.iter().any(|kind| kind == "string")
        && let Some(parsed) = parse_scalar_text(value)
        && matches_declared_type(&parsed, declared)
    {
        return parsed.to_string();
    }
    serde_json::Value::String(value.to_string()).to_string()
}

// The longest suffix of `content` that could be the start of the parameter close tag is
// ambiguous until more tokens arrive, so it is withheld from streaming.
const PARAMETER_CLOSE_SEQUENCE: &str = "\n</parameter>";

fn withhold_ambiguous_tail(content: &str) -> &str {
    // the leftmost char boundary whose suffix is a prefix of the close sequence
    // gives the longest ambiguous tail
    for (index, _) in content.char_indices() {
        let suffix = &content[index..];
        if suffix.len() <= PARAMETER_CLOSE_SEQUENCE.len() && PARAMETER_CLOSE_SEQUENCE.starts_with(suffix) {
            return &content[..index];
        }
    }
    content
}

// The arguments text with no closing brace: completed parameters are frozen, the open
// parameter's value streams as it grows.
fn open_arguments_text(
    params: &[(String, FramedParam)],
    function: Option<&str>,
    types: &ToolParameterTypes,
) -> Option<String> {
    let mut entries: Vec<(String, String)> = Vec::new();
    for (key, param) in params {
        let declared = function.and_then(|function| types.declared(function, key));
        let serialized = match param {
            FramedParam::Complete(value) => serialize_param_value(value, declared),
            FramedParam::Open(content) => {
                // an empty value has no known type yet, and typed values (JSON
                // starting with `{`/`[`, or declared non-string so the final
                // form is a bare literal) can only be serialized once
                // complete: either way nothing about the parameter may be emitted
                let streamable = withhold_ambiguous_tail(content);
                let declared_non_string =
                    declared.is_some_and(|declared| !declared.iter().any(|kind| kind == "string"));
                if streamable.is_empty()
                    || streamable.starts_with('{')
                    || streamable.starts_with('[')
                    || declared_non_string
                {
                    break;
                }
                let quoted = serde_json::Value::String(streamable.to_string()).to_string();
                quoted[..quoted.len() - 1].to_string()
            },
        };
        entries.push((key.clone(), serialized));
    }
    if entries.is_empty() {
        return None;
    }
    let mut out = String::from("{");
    for (index, (key, serialized)) in entries.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(key).expect("key serializes"));
        out.push(':');
        out.push_str(serialized);
    }
    Some(out)
}

fn json_candidate_name(raw: &str) -> Option<String> {
    let rest = &raw[raw.find("\"name\"")? + 6..];
    let rest = rest[rest.find(':')? + 1..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let mut name = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(name),
            '\\' => match chars.next() {
                Some('n') => name.push('\n'),
                Some(other) => name.push(other),
                None => return None,
            },
            c => name.push(c),
        }
    }
    None
}

impl ToolCallStreamer {
    pub fn new() -> Self {
        Self {
            // The id is assigned by the server at announcement time: clients key
            // in-progress tool call UI by id, so it must never change mid-call.
            id: Uuid::new_v4().to_string(),
            name_announced: false,
            key_order: Vec::new(),
            open_text: String::new(),
        }
    }

    pub fn update(
        &mut self,
        index: usize,
        raw: &str,
        types: &ToolParameterTypes,
    ) -> Vec<OaiToolCall> {
        let mut deltas = Vec::new();
        let name;
        if raw.trim_start().starts_with('<') {
            let call = parse_framed_call(raw);
            name = call.name.filter(|name| !name.is_empty());
            self.key_order = call.params.iter().map(|(key, _)| key.clone()).collect();
            if let Some(open_text) = open_arguments_text(&call.params, name.as_deref(), types)
                && open_text.len() > self.open_text.len()
                && open_text.starts_with(&self.open_text)
            {
                deltas.push(OaiToolCall {
                    index: Some(index),
                    id: String::new(),
                    kind: String::new(),
                    function: OaiFunctionCall {
                        name: String::new(),
                        arguments: open_text[self.open_text.len()..].to_string(),
                    },
                });
                self.open_text = open_text;
            }
        } else {
            name = json_candidate_name(raw);
        }
        if let Some(name) = name
            && !name.is_empty()
            && !self.name_announced
        {
            self.name_announced = true;
            deltas.insert(
                0,
                OaiToolCall {
                    index: Some(index),
                    id: self.id.clone(),
                    kind: "function".to_string(),
                    function: OaiFunctionCall {
                        name,
                        arguments: String::new(),
                    },
                },
            );
        }
        deltas
    }

    pub fn finish(
        &mut self,
        index: usize,
        call: &ToolCall,
    ) -> OaiToolCall {
        let arguments = if self.open_text.is_empty() {
            call.arguments.json.clone()
        } else {
            self.closing_fragment(call)
        };
        let (id, kind, name) = if self.name_announced {
            (String::new(), String::new(), String::new())
        } else {
            (self.id.clone(), "function".to_string(), call.name.clone())
        };
        OaiToolCall {
            index: Some(index),
            id,
            kind,
            function: OaiFunctionCall {
                name,
                arguments,
            },
        }
    }

    fn closing_fragment(
        &self,
        call: &ToolCall,
    ) -> String {
        // Rebuild the final arguments with keys in streamed order so the closing fragment
        // completes the exact string the earlier fragments started.
        let final_text = match serde_json::from_str::<serde_json::Value>(&call.arguments.json) {
            Ok(serde_json::Value::Object(map)) => {
                let mut keys: Vec<&str> =
                    self.key_order.iter().map(String::as_str).filter(|key| map.contains_key(*key)).collect();
                for key in map.keys() {
                    if !keys.contains(&key.as_str()) {
                        keys.push(key);
                    }
                }
                let entries = keys
                    .iter()
                    .map(|key| format!("{}:{}", serde_json::to_string(key).expect("key serializes"), map[*key]))
                    .collect::<Vec<_>>();
                format!("{{{}}}", entries.join(","))
            },
            _ => call.arguments.json.clone(),
        };
        match final_text.strip_prefix(&self.open_text) {
            Some(suffix) => suffix.to_string(),
            None => {
                eprintln!(
                    "[server] tool call stream diverged from the final call; the client may reject the assembled arguments"
                );
                final_text
            },
        }
    }
}

#[cfg(test)]
#[path = "../../unit/server/chat_tool_calls_test.rs"]
mod tests;
