mod access;
mod collection;
mod condition;
mod construction;
mod control_flow;
mod operation;
mod string;

pub use condition::Condition;
pub use operation::{CallTarget, GetTarget, Operation, PathSegment, Pipeline, SwitchCase};
use serde_json::Value;

use crate::{TransformError, TransformSchema};

pub(crate) fn execute_pipeline(
    pipeline: &Pipeline,
    input: Value,
    schema: &TransformSchema,
) -> Result<Value, TransformError> {
    if pipeline.is_empty() {
        return Ok(input);
    }
    let mut output = input;
    for operation in pipeline {
        output = operation.execute(output, schema)?;
    }
    Ok(output)
}
