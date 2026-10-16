use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use shoji::types::basic::{ToolCall, ToolDescription, ToolNamespace, Value, parse_lenient_json};

use super::ChatSessionError;

/// Declared JSON-Schema types of each tool parameter, keyed by function name.
/// Tool-call markup cannot carry types — the parser keeps every parameter the
/// text the model wrote — so the declared schema is what restores the wire
/// types clients validate against.
#[derive(Clone, Default)]
pub struct ToolParameterTypes(HashMap<String, HashMap<String, Vec<String>>>);

impl ToolParameterTypes {
    pub fn from_parameters<'a>(parameters: impl IntoIterator<Item = (&'a str, &'a Value)>) -> Self {
        let mut functions = HashMap::new();
        for (name, parameters) in parameters {
            let Ok(schema) = serde_json::from_str::<serde_json::Value>(&parameters.json) else {
                continue;
            };
            let Some(properties) = schema.get("properties").and_then(serde_json::Value::as_object) else {
                continue;
            };
            let parameters = properties
                .iter()
                .filter_map(|(name, property)| {
                    let types = declared_types(property, &schema, 8);
                    (!types.is_empty()).then(|| (name.clone(), types))
                })
                .collect();
            functions.insert(name.to_string(), parameters);
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

// Per parameter: count each queued schema and each entry in a type array.
const MAX_SCHEMA_TRAVERSAL_WORK: usize = 1024;

/// The JSON-Schema types a property schema declares: its `type` keyword, or the
/// union of its `anyOf`/`oneOf` branches, following local `$ref`s (the shape
/// pydantic and schemars clients emit for optional and nested objects).
fn declared_types(
    property: &serde_json::Value,
    root: &serde_json::Value,
    depth: u32,
) -> Vec<String> {
    let mut pending = vec![(property, depth)];
    let mut visited = HashMap::new();
    let mut types = HashSet::new();
    let mut remaining_work = MAX_SCHEMA_TRAVERSAL_WORK - 1;
    while let Some((property, depth)) = pending.pop() {
        if depth == 0 {
            continue;
        }
        let property = match property.get("$ref").and_then(serde_json::Value::as_str) {
            Some(reference) => {
                reference.strip_prefix('#').and_then(|pointer| root.pointer(pointer)).unwrap_or(property)
            },
            None => property,
        };
        // Track identity without hashing the schema's contents. Revisit a shared
        // node only when a shorter path leaves more depth to discover its types.
        let identity = std::ptr::from_ref(property);
        if visited.get(&identity).is_some_and(|seen_depth| *seen_depth >= depth) {
            continue;
        }
        visited.insert(identity, depth);
        match property.get("type") {
            Some(serde_json::Value::String(kind)) => {
                types.insert(kind.as_str());
            },
            Some(serde_json::Value::Array(kinds)) => {
                let Some(remaining) = remaining_work.checked_sub(kinds.len()) else {
                    return Vec::new();
                };
                remaining_work = remaining;
                types.extend(kinds.iter().filter_map(serde_json::Value::as_str));
            },
            _ if depth > 1 => {
                for branches in
                    ["anyOf", "oneOf"].iter().filter_map(|key| property.get(*key).and_then(serde_json::Value::as_array))
                {
                    // Charge work before enqueueing, bounding both visits and
                    // pending allocations even for a wide, nonrecursive union.
                    let Some(remaining) = remaining_work.checked_sub(branches.len()) else {
                        // Partial types may omit a string branch and cause an
                        // incorrect coercion, so leave the parameter unchanged.
                        return Vec::new();
                    };
                    remaining_work = remaining;
                    pending.extend(branches.iter().map(|branch| (branch, depth - 1)));
                }
            },
            _ => {},
        }
    }
    let mut types: Vec<String> = types.into_iter().map(str::to_string).collect();
    types.sort_unstable();
    types
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

/// Parameter text read as JSON: strict first, then with the bounded repair of
/// control characters and bare quotes models leave inside string values, plus
/// the Python-style booleans some models emit in tool markup (qwen3.5 writes
/// `True`/`False`).
pub fn parse_parameter_text(text: &str) -> Option<serde_json::Value> {
    if let Some(parsed) = parse_lenient_json(text) {
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
        // The parser kept the markup text as a string; restore the declared
        // type when the text reads as it. A union that includes "string" stays
        // a string: the text is already schema-valid and the intended type is
        // unknowable.
        serde_json::Value::String(text) if !declares_string => {
            let parsed = parse_parameter_text(text)?;
            matches_declared_type(&parsed, declared).then_some(parsed)
        },
        // A JSON-format parser typed a value although the parameter is a
        // plain string.
        value if declared == ["string"] && !value.is_string() => Some(serde_json::Value::String(value.to_string())),
        _ => None,
    }
}

/// Restores the declared types the markup could not carry. Applied only at
/// protocol boundaries: the session keeps the parser's values so its stored
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

enum FramedParam {
    Complete(String),
    Open(String),
}

struct FramedCall {
    name: Option<String>,
    params: Vec<(String, FramedParam)>,
    parameter_close: &'static str,
}

fn parse_framed_call(raw: &str) -> FramedCall {
    let atem = raw.trim_start().starts_with("<atem:");
    let (function_open, parameter_open, name_close, parameter_close) = if atem {
        ("<atem:invoke name=\"", "<atem:parameter name=\"", "\">", "</atem:parameter>")
    } else {
        ("<function=", "<parameter=", ">", "\n</parameter>")
    };
    let name = raw.find(function_open).and_then(|start| {
        let rest = &raw[start + function_open.len()..];
        rest.find(name_close).map(|end| rest[..end].to_string())
    });
    let mut params = Vec::new();
    let mut rest = raw;
    while let Some(start) = rest.find(parameter_open) {
        rest = &rest[start + parameter_open.len()..];
        let Some(tag_end) = rest.find(name_close) else {
            break;
        };
        let key = rest[..tag_end].to_string();
        let value_part = &rest[tag_end + name_close.len()..];
        let value_part = if atem {
            value_part
        } else {
            value_part.strip_prefix('\n').unwrap_or(value_part)
        };
        match value_part.find(parameter_close) {
            Some(end) => {
                params.push((key, FramedParam::Complete(value_part[..end].to_string())));
                rest = &value_part[end + parameter_close.len()..];
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
        parameter_close,
    }
}

// A completed parameter serializes exactly as coerce_tool_call serializes it in
// the final call, so the streamed prefix is a prefix of the final text.
fn serialize_param_value(
    value: &str,
    declared: Option<&[String]>,
) -> String {
    let text = serde_json::Value::String(value.to_string());
    declared.and_then(|declared| coerce_parameter_value(&text, declared)).unwrap_or(text).to_string()
}

// The longest suffix of `content` that could be the start of the parameter close tag is
// ambiguous until more tokens arrive, so it is withheld from streaming.
fn withhold_ambiguous_tail<'a>(
    content: &'a str,
    close_sequence: &str,
) -> &'a str {
    // the leftmost char boundary whose suffix is a prefix of the close sequence
    // gives the longest ambiguous tail
    let mut end = content.len();
    for (index, _) in content.char_indices().rev().take(close_sequence.len()) {
        let suffix = &content[index..];
        if suffix.len() <= close_sequence.len() && close_sequence.starts_with(suffix) {
            end = index;
        }
    }
    &content[..end]
}

// The arguments text with no closing brace: completed parameters are frozen, the open
// parameter's value streams as it grows.
fn open_arguments_text(
    call: &FramedCall,
    types: &ToolParameterTypes,
) -> Option<String> {
    let mut entries: Vec<(String, String)> = Vec::new();
    for (key, param) in &call.params {
        let declared = call.name.as_deref().and_then(|function| types.declared(function, key));
        let serialized = match param {
            FramedParam::Complete(value) => serialize_param_value(value, declared),
            FramedParam::Open(content) => {
                // an empty value has nothing to emit yet, and a declared non-string
                // value takes its final form (a bare literal) only once complete:
                // either way nothing about the parameter may be emitted
                let streamable = withhold_ambiguous_tail(content, call.parameter_close);
                let declared_non_string =
                    declared.is_some_and(|declared| !declared.iter().any(|kind| kind == "string"));
                if streamable.is_empty() || declared_non_string {
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
    let mut rest = raw.trim_start().strip_prefix('{')?;
    loop {
        let mut key = serde_json::Deserializer::from_str(rest).into_iter::<String>();
        let name = key.next()?.ok()?;
        rest = rest[key.byte_offset()..].trim_start().strip_prefix(':')?.trim_start();
        if name == "name" {
            return serde_json::Deserializer::from_str(rest).into_iter::<String>().next()?.ok();
        }
        let mut value = serde_json::Deserializer::from_str(rest).into_iter::<serde::de::IgnoredAny>();
        value.next()?.ok()?;
        rest = rest[value.byte_offset()..].trim_start().strip_prefix(',')?;
    }
}

/// Converts model tool-call candidates into append-only JSON argument fragments.
#[bindings::export(Class)]
#[derive(Clone)]
pub struct ToolCallArgumentStream {
    state: Arc<Mutex<ArgumentState>>,
    types: ToolParameterTypes,
}

#[derive(Default)]
struct ArgumentState {
    name: String,
    key_order: Vec<String>,
    open_text: String,
    finished: bool,
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone)]
pub struct ToolCallArgumentDelta {
    pub name: String,
    pub arguments: String,
}

impl ToolCallArgumentStream {
    pub fn new(types: ToolParameterTypes) -> Self {
        Self {
            state: Arc::new(Mutex::new(ArgumentState::default())),
            types,
        }
    }
}

#[bindings::export(Implementation)]
impl ToolCallArgumentStream {
    #[bindings::export(Method(Factory))]
    pub fn create(tools: Vec<ToolNamespace>) -> Self {
        let functions = tools.iter().flat_map(|namespace| &namespace.tools).filter_map(|tool| match tool {
            ToolDescription::Function {
                tool_function,
            } => tool_function.parameters.as_ref().map(|parameters| (tool_function.name.as_str(), parameters)),
        });
        Self::new(ToolParameterTypes::from_parameters(functions))
    }

    #[bindings::export(Method)]
    pub fn update(
        &self,
        raw: String,
    ) -> Result<ToolCallArgumentDelta, ChatSessionError> {
        let mut state = self.state.lock().expect("tool argument stream mutex poisoned");
        if state.finished {
            return Err(stream_error("Tool arguments changed after completion"));
        }
        let (name, next) = if raw.trim_start().starts_with('<') {
            let call = parse_framed_call(&raw);
            state.key_order = call.params.iter().map(|(key, _)| key.clone()).collect();
            let next = open_arguments_text(&call, &self.types);
            (call.name, next)
        } else {
            (json_candidate_name(&raw), None)
        };
        if let Some(name) = name.filter(|name| !name.is_empty()) {
            if !state.name.is_empty() && state.name != name {
                return Err(stream_error("Uzu revised a streamed tool name"));
            }
            state.name = name;
        }
        let mut arguments = String::new();
        if let Some(next) = next {
            let suffix = next
                .strip_prefix(&state.open_text)
                .ok_or_else(|| stream_error("Uzu revised streamed tool arguments"))?;
            arguments.push_str(suffix);
            state.open_text = next;
        }
        Ok(ToolCallArgumentDelta {
            name: state.name.clone(),
            arguments,
        })
    }

    #[bindings::export(Method)]
    pub fn finish(
        &self,
        call: ToolCall,
    ) -> Result<ToolCallArgumentDelta, ChatSessionError> {
        let mut state = self.state.lock().expect("tool argument stream mutex poisoned");
        if state.finished || call.name.is_empty() || (!state.name.is_empty() && state.name != call.name) {
            return Err(stream_error("Uzu revised a completed tool call"));
        }
        let call = coerce_tool_call(&call, &self.types);
        let map = serde_json::from_str::<serde_json::Value>(&call.arguments.json)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .filter(|map| !map.contains_key(super::UNPARSED_ARGUMENTS_KEY))
            .ok_or_else(|| stream_error("Uzu returned invalid tool arguments"))?;
        let mut keys = Vec::new();
        let mut seen = HashSet::new();
        for key in state.key_order.iter().chain(map.keys()) {
            if map.contains_key(key) && seen.insert(key) {
                keys.push(key);
            }
        }
        let entries = keys
            .iter()
            .map(|key| format!("{}:{}", serde_json::to_string(key).expect("key serializes"), map[*key]))
            .collect::<Vec<_>>();
        let final_text = format!("{{{}}}", entries.join(","));
        let arguments = final_text
            .strip_prefix(&state.open_text)
            .ok_or_else(|| stream_error("Uzu revised streamed tool arguments"))?
            .to_string();
        state.finished = true;
        Ok(ToolCallArgumentDelta {
            name: call.name,
            arguments,
        })
    }
}

fn stream_error(message: &str) -> ChatSessionError {
    ChatSessionError::Backend {
        message: message.to_string(),
    }
}

#[cfg(test)]
#[path = "tool_call_tests.rs"]
mod tests;
