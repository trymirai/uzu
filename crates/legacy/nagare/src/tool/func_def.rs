use std::{error::Error, future::Future, sync::Arc};

pub use shoji::types::basic::Value;
use shoji::types::basic::parse_lenient_json;

pub type ErrorFuture = Box<dyn Error + Send + Sync>;
pub type FunctionFuture = dyn Fn(Value) -> Box<dyn Future<Output = Result<Value, ErrorFuture>> + Send> + Send + Sync;

#[derive(Clone)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
    pub return_definition: Option<Value>,
    func: Arc<FunctionFuture>,
}

impl ToolDescriptor {
    pub fn new(
        name: String,
        description: String,
        parameters: Option<Value>,
        return_definition: Option<Value>,
        func: Box<FunctionFuture>,
    ) -> Self {
        Self {
            name,
            description,
            parameters,
            return_definition,
            func: Arc::new(func),
        }
    }

    pub async fn execute(
        &self,
        args: Value,
    ) -> Result<Value, ErrorFuture> {
        let args = self.coerce_arguments(args);
        Box::into_pin((self.func)(args)).await
    }

    // Markup parsers keep every argument the text the model wrote, and small models mistype scalars even in JSON
    // markup (e.g. Llama 3.2 1B passes "37" for a number parameter); coerce argument values to their
    // schema-declared types instead of failing the call — an error result makes such models retry the same call
    // indefinitely.
    fn coerce_arguments(
        &self,
        args: Value,
    ) -> Value {
        let Some(parameters) = &self.parameters else {
            return args;
        };
        let Ok(schema) = serde_json::Value::try_from(parameters.clone()) else {
            return args;
        };
        let Ok(json) = serde_json::Value::try_from(args.clone()) else {
            return args;
        };
        Value::from(coerce_to_schema(json, &schema))
    }
}

/// Runtime surface of the `uzu_tool_function`/`uzu_tool_closure` expansions;
/// public so generated code relies on ordinary API instead of hidden re-exports.
pub fn parse_arguments(args: Value) -> Result<serde_json::Value, ErrorFuture> {
    Ok(serde_json::Value::try_from(args)?)
}

pub fn extract_argument<T: serde::de::DeserializeOwned>(
    args: &serde_json::Value,
    name: &str,
    tool_name: &str,
) -> Result<T, ErrorFuture> {
    serde_json::from_value(args.get(name).cloned().unwrap_or(serde_json::Value::Null))
        .map_err(|error| format!("invalid value for parameter `{name}` of tool `{tool_name}`: {error}").into())
}

pub fn serialize_result<T: serde::Serialize>(result: &T) -> Result<Value, ErrorFuture> {
    Ok(Value::from(serde_json::to_value(result)?))
}

pub fn null_result() -> Value {
    Value::from(serde_json::Value::Null)
}

/// Runs a synchronous tool on the blocking thread pool so heavy work
/// doesn't stall the async runtime.
pub async fn run_blocking<T: Send + 'static>(func: impl FnOnce() -> T + Send + 'static) -> Result<T, ErrorFuture> {
    tokio::task::spawn_blocking(func).await.map_err(|error| -> ErrorFuture { error.into() })
}

fn coerce_to_schema(
    value: serde_json::Value,
    schema: &serde_json::Value,
) -> serde_json::Value {
    coerce_to_schema_with_root(value, schema, schema)
}

fn coerce_to_schema_with_root(
    value: serde_json::Value,
    schema: &serde_json::Value,
    root_schema: &serde_json::Value,
) -> serde_json::Value {
    use serde_json::Value as Json;

    let schema = resolve_local_schema(schema, root_schema);

    if let Some(branches) = schema.get("anyOf").and_then(Json::as_array)
        && branches.len() == 2
        && branches.iter().any(|branch| branch.get("type").and_then(Json::as_str) == Some("null"))
        && let Some(non_null_schema) =
            branches.iter().find(|branch| branch.get("type").and_then(Json::as_str) != Some("null"))
    {
        return coerce_to_schema_with_root(value, non_null_schema, root_schema);
    }
    // A union of shapes: container text goes to the first branch it parses as. Unlike the OpenAI boundary, which
    // keeps the text when the union admits a string, a tool implementation deserializes what it declared, so the
    // container branch wins here.
    if let Json::String(text) = &value
        && let Some(branches) = schema.get("anyOf").or_else(|| schema.get("oneOf")).and_then(Json::as_array)
    {
        for branch in branches {
            let branch = resolve_local_schema(branch, root_schema);
            let parses = match branch.get("type").and_then(Json::as_str) {
                Some("object") => parse_lenient_json(text).is_some_and(|parsed| parsed.is_object()),
                Some("array") => parse_lenient_json(text).is_some_and(|parsed| parsed.is_array()),
                _ => false,
            };
            if parses {
                return coerce_to_schema_with_root(value, branch, root_schema);
            }
        }
    }

    let schema_type = match schema.get("type") {
        Some(Json::String(schema_type)) => Some(schema_type.as_str()),
        Some(Json::Array(schema_types))
            if schema_types.len() == 2
                && schema_types.iter().any(|schema_type| schema_type.as_str() == Some("null")) =>
        {
            schema_types.iter().filter_map(Json::as_str).find(|schema_type| *schema_type != "null")
        },
        _ => None,
    };

    // an object or array parameter arrives as text from the markup parsers; read it when it parses as that shape
    let value = match (schema_type, value) {
        (Some("object"), Json::String(text)) => {
            parse_lenient_json(&text).filter(Json::is_object).unwrap_or(Json::String(text))
        },
        (Some("array"), Json::String(text)) => {
            parse_lenient_json(&text).filter(Json::is_array).unwrap_or(Json::String(text))
        },
        (_, value) => value,
    };

    match schema_type {
        Some("object") => match value {
            Json::Object(map) => Json::Object(
                map.into_iter()
                    .map(|(key, value)| {
                        let value = match schema.get("properties").and_then(|properties| properties.get(&key)) {
                            Some(property_schema) => coerce_to_schema_with_root(value, property_schema, root_schema),
                            None => value,
                        };
                        (key, value)
                    })
                    .collect(),
            ),
            other => other,
        },
        Some("array") => match (value, schema.get("items")) {
            (Json::Array(items), Some(item_schema)) => Json::Array(
                items.into_iter().map(|item| coerce_to_schema_with_root(item, item_schema, root_schema)).collect(),
            ),
            (other, _) => other,
        },
        Some("number") => match &value {
            Json::String(text) => match text.trim().parse::<f64>().ok().and_then(serde_json::Number::from_f64) {
                Some(number) => Json::Number(number),
                None => value,
            },
            _ => value,
        },
        Some("integer") => match &value {
            Json::String(text) => {
                let text = text.trim();
                match text.parse::<i64>() {
                    Ok(number) => Json::Number(number.into()),
                    Err(_) => match text.parse::<u64>() {
                        Ok(number) => Json::Number(number.into()),
                        Err(_) => value,
                    },
                }
            },
            _ => value,
        },
        Some("boolean") => match &value {
            Json::String(text) if text.trim().eq_ignore_ascii_case("true") => Json::Bool(true),
            Json::String(text) if text.trim().eq_ignore_ascii_case("false") => Json::Bool(false),
            _ => value,
        },
        Some("string") => match value {
            Json::Number(number) => Json::String(number.to_string()),
            Json::Bool(boolean) => Json::String(boolean.to_string()),
            other => other,
        },
        _ => value,
    }
}

fn resolve_local_schema<'a>(
    mut schema: &'a serde_json::Value,
    root_schema: &'a serde_json::Value,
) -> &'a serde_json::Value {
    use serde_json::Value as Json;

    let mut visited = Vec::new();
    while let Some(reference) = schema.get("$ref").and_then(Json::as_str) {
        if visited.contains(&reference) {
            break;
        }
        visited.push(reference);

        let Some(pointer) = reference.strip_prefix('#') else {
            break;
        };
        let target = if pointer.is_empty() {
            Some(root_schema)
        } else if pointer.starts_with('/') {
            root_schema.pointer(pointer)
        } else {
            // Anchor fragments are not JSON Pointers.
            None
        };
        let Some(target) = target else {
            break;
        };
        schema = target;
    }
    schema
}

#[cfg(test)]
mod tests {
    use super::coerce_to_schema;

    #[test]
    fn coerce_to_schema_parses_container_text_and_keeps_string_text() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "options": {"type": "object", "properties": {"retries": {"type": "integer"}}},
                "tags": {"type": ["array", "null"], "items": {"type": "integer"}}
            }
        });
        // markup parsers keep every parameter as text; containers are parsed by the schema, strings never are
        let coerced = coerce_to_schema(
            serde_json::json!({
                "content": "{\"name\": \"arcade\"}",
                "options": "{\"retries\": \"3\"}",
                "tags": "[\"1\", 2]"
            }),
            &schema,
        );
        assert_eq!(
            coerced,
            serde_json::json!({"content": "{\"name\": \"arcade\"}", "options": {"retries": 3}, "tags": [1, 2]})
        );

        let kept = coerce_to_schema(serde_json::json!({"options": "{ broken", "tags": "not a list"}), &schema);
        assert_eq!(kept, serde_json::json!({"options": "{ broken", "tags": "not a list"}));
    }

    #[test]
    fn coerce_to_schema_reads_container_text_through_refs_and_unions() {
        // the shapes schemars emits for Option<Box<T>> and for a data enum
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "child": {"anyOf": [{"$ref": "#/$defs/Child"}, {"type": "null"}]},
                "shape": {"oneOf": [{"type": "object", "properties": {"r": {"type": "number"}}}, {"type": "string"}]}
            },
            "$defs": {"Child": {"type": "object", "properties": {"n": {"type": "integer"}}}}
        });
        let coerced =
            coerce_to_schema(serde_json::json!({"child": "{\"n\": \"2\"}", "shape": "{\"r\": \"1.5\"}"}), &schema);
        assert_eq!(coerced, serde_json::json!({"child": {"n": 2}, "shape": {"r": 1.5}}));
    }
}
