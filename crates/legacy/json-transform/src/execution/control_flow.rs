use indexmap::IndexMap;
use serde_json::Value;

use crate::{
    TransformError, TransformSchema,
    execution::{
        execute_pipeline,
        operation::{CallTarget, Pipeline, SwitchCase},
    },
};

pub fn execute_switch(
    key_pipeline: &Pipeline,
    cases: &[SwitchCase],
    default: &Option<Pipeline>,
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    let key_description: String =
        key_pipeline.iter().map(|operation| operation.to_string()).collect::<Vec<_>>().join(", ");
    let cases_description: String = cases.iter().map(|case| case.when.to_string()).collect::<Vec<_>>().join(", ");
    let _span = tracing::info_span!(
        "execute_switch",
        key = %key_description,
        cases = %cases_description,
    )
    .entered();

    let match_value = execute_pipeline(key_pipeline, input.clone(), schema)?;

    for case in cases {
        if case.when.evaluate(&match_value) {
            tracing::trace!(matched = %case.when);
            return execute_pipeline(&case.then, input, schema);
        }
    }

    tracing::trace!("matched default");
    match default {
        Some(pipeline) => execute_pipeline(pipeline, input, schema),
        None => Ok(Value::Null),
    }
}

#[tracing::instrument(skip(arguments, input, schema), fields(target = %target))]
pub fn execute_call(
    target: &CallTarget,
    arguments: &IndexMap<String, Value>,
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    let resolved_name = match target {
        CallTarget::Static {
            name,
        } => name.clone(),
        CallTarget::Dynamic {
            key,
        } => match &input {
            Value::Object(map) => {
                map.get(key).and_then(|value| value.as_str()).map(|text| text.to_string()).ok_or_else(|| {
                    TransformError::UndefinedPipeline {
                        name: format!("dynamic call key {key} not found"),
                    }
                })?
            },
            _ => {
                return Err(TransformError::UndefinedPipeline {
                    name: "dynamic call requires object input".to_string(),
                });
            },
        },
    };

    let effective_input = if arguments.is_empty() {
        input
    } else {
        let mut map = match input {
            Value::Object(map) => map,
            _ => serde_json::Map::new(),
        };
        for (key, value) in arguments {
            map.insert(key.clone(), value.clone());
        }
        Value::Object(map)
    };

    let pipeline = schema.pipelines.get(&resolved_name).ok_or_else(|| TransformError::UndefinedPipeline {
        name: resolved_name.clone(),
    })?;
    execute_pipeline(pipeline, effective_input, schema)
}

#[tracing::instrument(skip_all, fields(field = field, with = with))]
pub fn execute_on(
    field: &str,
    with: Option<&str>,
    do_pipeline: &Pipeline,
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    let working_value = match with {
        Some(key) => input.as_object().and_then(|map| map.get(key).cloned()).unwrap_or(Value::Null),
        None => input.clone(),
    };

    let is_active =
        input.as_object().and_then(|map| map.get(field)).map(|value| value == &Value::Bool(true)).unwrap_or(false);

    if is_active {
        execute_pipeline(do_pipeline, working_value, schema)
    } else {
        Ok(working_value)
    }
}
