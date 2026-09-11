use serde_json::Value;

use crate::{
    TransformError,
    execution::operation::{GetTarget, PathSegment},
};

#[tracing::instrument(skip(input), fields(target = %target))]
pub fn execute_get(
    target: &GetTarget,
    input: Value,
) -> Result<Value, TransformError> {
    match target {
        GetTarget::Key {
            key,
        } => execute_get_key(key, input),
        GetTarget::Path {
            path,
        } => execute_get_path(path, input),
    }
}

fn execute_get_key(
    key: &str,
    input: Value,
) -> Result<Value, TransformError> {
    match input {
        Value::Object(map) => Ok(map.get(key).cloned().unwrap_or(Value::Null)),
        _ => Ok(Value::Null),
    }
}

fn execute_get_path(
    path: &[PathSegment],
    input: Value,
) -> Result<Value, TransformError> {
    let mut current = input;
    for segment in path {
        current = match segment {
            PathSegment::Key(key) => match current {
                Value::Object(map) => map.get(key).cloned().unwrap_or(Value::Null),
                _ => Value::Null,
            },
            PathSegment::Index(index) => match current {
                Value::Array(items) => items.into_iter().nth(*index).unwrap_or(Value::Null),
                _ => Value::Null,
            },
        };
    }
    Ok(current)
}

#[tracing::instrument(skip_all)]
pub fn execute_first(input: Value) -> Result<Value, TransformError> {
    match input {
        Value::Array(items) => Ok(items.into_iter().next().unwrap_or(Value::Null)),
        _ => Ok(Value::Null),
    }
}
