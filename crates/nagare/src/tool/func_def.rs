use std::{future::Future, pin::Pin, sync::Arc};

pub use shoji::types::basic::Value;

pub type FutureError = Box<dyn std::error::Error + Send + Sync>;
pub type FutureFunction =
    dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value, FutureError>> + Send>> + Send + Sync;

#[derive(Clone)]
pub struct ToolFunctionDefinition {
    name: String,
    description: String,
    parameters: Option<Value>,
    return_definition: Option<Value>,
    func: Arc<FutureFunction>,
}

impl ToolFunctionDefinition {
    pub fn new(
        name: String,
        description: String,
        parameters: Option<Value>,
        return_definition: Option<Value>,
        func: Box<FutureFunction>,
    ) -> Self {
        Self {
            name,
            description,
            parameters,
            return_definition,
            func: Arc::new(func),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn parameters(&self) -> &Option<Value> {
        &self.parameters
    }

    pub fn return_definition(&self) -> &Option<Value> {
        &self.return_definition
    }

    pub async fn execute(
        &self,
        args: Value,
    ) -> Result<Value, FutureError> {
        let args = self.coerce_arguments(args);
        (self.func)(args).await
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

fn coerce_to_schema(
    value: serde_json::Value,
    schema: &serde_json::Value,
) -> serde_json::Value {
    use serde_json::Value as Json;

    match schema.get("type").and_then(Json::as_str) {
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
            Json::String(text) => match text.trim().parse::<i64>() {
                Ok(number) => Json::Number(number.into()),
                Err(_) => value,
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
