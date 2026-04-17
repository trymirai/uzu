use serde_json::Value;

use crate::{
    TransformError, TransformSchema,
    execution::{Condition, execute_pipeline, operation::Pipeline},
};

pub fn execute_each(
    apply: &Pipeline,
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    let Value::Array(items) = input else {
        return Ok(Value::Null);
    };
    let _span = tracing::info_span!("execute_each", items = items.len()).entered();
    let results: Result<Vec<Value>, TransformError> =
        items.into_iter().map(|element| execute_pipeline(apply, element, schema)).collect();
    Ok(Value::Array(results?))
}

pub fn execute_flat_map(
    apply: &Pipeline,
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    let Value::Array(items) = input else {
        return Ok(input);
    };
    let _span = tracing::info_span!("execute_flat_map", items = items.len()).entered();
    let mut results: Vec<Value> = Vec::new();
    for element in items {
        let result = execute_pipeline(apply, element, schema)?;
        match result {
            Value::Array(inner_items) => results.extend(inner_items),
            other => results.push(other),
        }
    }
    Ok(Value::Array(results))
}

pub fn execute_filter(
    condition: &Condition,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::Array(items) = input else {
        return Ok(Value::Null);
    };
    let input_count = items.len();
    let filtered: Vec<Value> = items.into_iter().filter(|element| condition.evaluate(element)).collect();
    let _span = tracing::info_span!(
        "execute_filter",
        condition = %condition,
        input = input_count,
        output = filtered.len()
    )
    .entered();
    Ok(Value::Array(filtered))
}

#[tracing::instrument(skip(input))]
pub fn execute_join(
    separator: &str,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::Array(items) = input else {
        return Ok(Value::Null);
    };
    let parts: Vec<&str> = items.iter().filter_map(|item| item.as_str()).collect();
    Ok(Value::String(parts.join(separator)))
}

#[tracing::instrument(skip_all)]
pub fn execute_reduce(
    key_pipeline: &Pipeline,
    r#if: &Option<Condition>,
    then_pipeline: &Pipeline,
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    let Value::Array(items) = input else {
        return Ok(Value::Null);
    };

    let mut result: Vec<Value> = Vec::new();
    let mut group: Vec<Value> = Vec::new();
    let mut group_key: Option<Value> = None;

    for item in items {
        let item_key = execute_pipeline(key_pipeline, item.clone(), schema)?;

        let should_group = match r#if {
            Some(condition) => condition.evaluate(&item_key),
            None => true,
        };

        if !should_group {
            flush_group(&mut group, &mut group_key, &mut result, then_pipeline, schema)?;
            result.push(item);
            continue;
        }

        let same_group = group_key.as_ref().map(|existing| existing == &item_key).unwrap_or(false);

        if !group.is_empty() && !same_group {
            flush_group(&mut group, &mut group_key, &mut result, then_pipeline, schema)?;
        }

        group_key = Some(item_key);
        group.push(item);
    }

    flush_group(&mut group, &mut group_key, &mut result, then_pipeline, schema)?;

    Ok(Value::Array(result))
}

fn flush_group(
    group: &mut Vec<Value>,
    group_key: &mut Option<Value>,
    result: &mut Vec<Value>,
    then_pipeline: &Pipeline,
    schema: &TransformSchema,
) -> Result<(), TransformError> {
    if group.is_empty() {
        return Ok(());
    }
    let group_array = Value::Array(std::mem::take(group));
    let merged = execute_pipeline(then_pipeline, group_array, schema)?;
    result.push(merged);
    *group_key = None;
    Ok(())
}
