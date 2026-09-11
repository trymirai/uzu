use indexmap::IndexMap;
use serde_json::Value;

use crate::{
    TransformError, TransformSchema,
    execution::{execute_pipeline, operation::Pipeline},
};

#[tracing::instrument(skip_all, fields(key = key))]
pub fn execute_resolve(
    key: &str,
    map: &IndexMap<String, Value>,
    default: Option<&Value>,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::Object(mut object) = input else {
        return Ok(Value::Null);
    };
    let field_value = object.get(key).and_then(|value| value.as_str()).unwrap_or("");
    let resolved = map.get(field_value).or(default).cloned().unwrap_or(Value::Null);
    object.insert(key.to_string(), resolved);
    Ok(Value::Object(object))
}

#[tracing::instrument(skip(fields, input, schema), fields(field_count = fields.len()))]
pub fn execute_object(
    fields: &IndexMap<String, Pipeline>,
    required: &[String],
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    let mut map = serde_json::Map::new();
    for (field_name, pipeline) in fields {
        let value = execute_pipeline(pipeline, input.clone(), schema)?;
        let is_empty = match &value {
            Value::Null => true,
            Value::String(text) => text.is_empty(),
            Value::Array(items) => items.is_empty(),
            _ => false,
        };
        if is_empty && !required.contains(field_name) {
            continue;
        }
        map.insert(field_name.clone(), value);
    }
    Ok(Value::Object(map))
}

#[tracing::instrument(skip_all, fields(value = %value))]
pub fn execute_literal(value: &Value) -> Result<Value, TransformError> {
    Ok(value.clone())
}

#[tracing::instrument(skip_all)]
pub fn execute_to_array(input: Value) -> Result<Value, TransformError> {
    Ok(Value::Array(vec![input]))
}

#[tracing::instrument(skip_all)]
pub fn execute_default(
    fallback: &Value,
    input: Value,
) -> Result<Value, TransformError> {
    if input.is_null() {
        Ok(fallback.clone())
    } else {
        Ok(input)
    }
}
