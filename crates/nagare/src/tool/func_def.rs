use std::{error::Error, future::Future, sync::Arc};

pub use shoji::types::basic::Value;

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

    // Small models often mistype scalar tool arguments (e.g. Llama 3.2 1B passes "37" for a number parameter);
    // coerce argument values to their schema-declared scalar types instead of failing the call — an error result
    // makes such models retry the same call indefinitely.
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

fn coerce_to_schema(
    value: serde_json::Value,
    schema: &serde_json::Value,
) -> serde_json::Value {
    use serde_json::Value as Json;

    if let Some(branches) = schema.get("anyOf").and_then(Json::as_array)
        && branches.len() == 2
        && branches.iter().any(|branch| branch.get("type").and_then(Json::as_str) == Some("null"))
        && let Some(non_null_schema) =
            branches.iter().find(|branch| branch.get("type").and_then(Json::as_str) != Some("null"))
    {
        return coerce_to_schema(value, non_null_schema);
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

    match schema_type {
        Some("object") => match value {
            Json::Object(map) => Json::Object(
                map.into_iter()
                    .map(|(key, value)| {
                        let value = match schema.get("properties").and_then(|properties| properties.get(&key)) {
                            Some(property_schema) => coerce_to_schema(value, property_schema),
                            None => value,
                        };
                        (key, value)
                    })
                    .collect(),
            ),
            other => other,
        },
        Some("array") => match (value, schema.get("items")) {
            (Json::Array(items), Some(item_schema)) => {
                Json::Array(items.into_iter().map(|item| coerce_to_schema(item, item_schema)).collect())
            },
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
