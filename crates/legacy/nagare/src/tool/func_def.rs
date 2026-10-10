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
    mut value: serde_json::Value,
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
    if let Some(branches) = schema.get("anyOf").or_else(|| schema.get("oneOf")).and_then(Json::as_array) {
        let branches = branches.iter().map(|branch| resolve_local_schema(branch, root_schema)).collect::<Vec<_>>();
        // Container text still prefers its declared container shape over a string branch.
        // Native containers need the same traversal to coerce their nested fields.
        let parsed = value.as_str().and_then(parse_lenient_json);
        let container_type = match parsed.as_ref().unwrap_or(&value) {
            Json::Object(_) => Some("object"),
            Json::Array(_) => Some("array"),
            _ => None,
        };
        if let Some(container_type) = container_type {
            let mut matching =
                branches.iter().filter(|branch| branch.get("type").and_then(Json::as_str) == Some(container_type));
            if let Some(branch) = matching.next() {
                if matching.next().is_none() {
                    return coerce_to_schema_with_root(value, branch, root_schema);
                }
                // Reading valid container JSON does not require choosing a
                // union branch. Keep its nested values unchanged when several
                // branches share this shape; serde can select the actual arm.
                if let Some(parsed) = parsed {
                    value = parsed;
                }
            }
        }
        // Do not reinterpret scalar text if a branch permits strings or has an unknown shape.
        if value.is_string()
            && branches.iter().all(|branch| {
                matches!(
                    branch.get("type").and_then(Json::as_str),
                    Some("number" | "integer" | "boolean" | "object" | "array" | "null")
                )
            })
        {
            let mut converted = branches.iter().filter_map(|branch| {
                if !matches!(branch.get("type").and_then(Json::as_str), Some("number" | "integer" | "boolean")) {
                    return None;
                }
                let converted = coerce_to_schema_with_root(value.clone(), branch, root_schema);
                (converted != value).then_some(converted)
            });
            if let Some(converted_value) = converted.next()
                && converted.next().is_none()
            {
                return converted_value;
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

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[serde(untagged)]
    enum Choice {
        First {
            first: u32,
        },
        Second {
            second: u32,
        },
    }

    #[crate::tool::uzu_tool_function]
    fn choose(request: Choice) -> u32 {
        match request {
            Choice::First {
                first,
            } => first,
            Choice::Second {
                second,
            } => second,
        }
    }

    #[tokio::test]
    async fn tool_accepts_markup_json_for_either_object_union_branch() {
        let tool: super::ToolDescriptor = choose.into();
        for request in [serde_json::json!({ "first": 1 }), serde_json::json!({ "second": 2 })] {
            let expected = request.as_object().unwrap().values().next().unwrap().clone();
            for argument in [request.clone(), serde_json::json!(request.to_string())] {
                let result = tool.execute(serde_json::json!({ "request": argument }).into()).await.unwrap();
                assert_eq!(serde_json::Value::try_from(result).unwrap(), expected);
            }
        }
        // Parsing the container must not guess which branch should coerce a
        // nested scalar, or make a malformed container acceptable.
        for request in [r#"{"first":"1"}"#, r#"{"first":1}}"#] {
            assert!(tool.execute(serde_json::json!({ "request": request }).into()).await.is_err());
        }
    }

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

    #[test]
    fn coerce_to_schema_traverses_union_containers_and_scalar_strings() {
        for union in ["anyOf", "oneOf"] {
            let schema = serde_json::json!({
                "type": "array",
                "items": {(union): [{"type": "number"}, {"$ref": "#/$defs/Point"}]},
                "$defs": {"Point": {"type": "object", "properties": {
                    "x": {"type": "number"}, "y": {"type": "number"}
                }}}
            });
            let coerced = coerce_to_schema(
                serde_json::json!(["97.42", {"x": "1", "y": "2"}, "{\"x\":\"3\",\"y\":\"4\"}"]),
                &schema,
            );
            assert_eq!(coerced, serde_json::json!([97.42, {"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]));

            let schema = serde_json::json!({
                (union): [{"type": "array", "items": {"type": "integer"}}, {"type": "boolean"}]
            });
            assert_eq!(coerce_to_schema(serde_json::json!(["1", "2"]), &schema), serde_json::json!([1, 2]));
            assert_eq!(coerce_to_schema(serde_json::json!("true"), &schema), serde_json::json!(true));
        }
    }

    #[test]
    fn coerce_to_schema_keeps_ambiguous_or_invalid_union_input() {
        for union in ["anyOf", "oneOf"] {
            for branches in [
                serde_json::json!([{"type": "number"}, {"type": "string"}]),
                serde_json::json!([{"type": "number"}, {"$ref": "#/$defs/Text"}]),
                serde_json::json!([{"type": "number"}, {"type": "integer"}]),
                serde_json::json!([{"type": "number"}, {"enum": ["other"]}]),
            ] {
                let schema = serde_json::json!({(union): branches, "$defs": {"Text": {"type": "string"}}});
                let input = serde_json::json!("2");
                assert_eq!(coerce_to_schema(input.clone(), &schema), input);
            }

            let schema = serde_json::json!({(union): [
                {"type": "object", "properties": {"x": {"type": "number"}}},
                {"type": "object", "properties": {"x": {"type": "integer"}}}
            ]});
            for input in [serde_json::json!({"x": "1"}), serde_json::json!("{\"x\":\"1\"}")] {
                assert_eq!(coerce_to_schema(input, &schema), serde_json::json!({"x": "1"}));
            }
            let schema = serde_json::json!({(union): [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "integer"}}
            ]});
            for input in [serde_json::json!(["1"]), serde_json::json!("[\"1\"]")] {
                assert_eq!(coerce_to_schema(input, &schema), serde_json::json!(["1"]));
            }

            let schema = serde_json::json!({(union): [
                {"type": "number"}, {"type": "object", "properties": {"x": {"type": "number"}}}
            ]});
            for input in [
                serde_json::json!("not a number"),
                serde_json::json!("NaN"),
                serde_json::json!("Infinity"),
                serde_json::json!({"x": "bad"}),
                serde_json::json!(null),
            ] {
                assert_eq!(coerce_to_schema(input.clone(), &schema), input);
            }
        }
    }
}
