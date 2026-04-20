use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{TransformError, execution::Pipeline};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(transparent)]
pub struct TransformSchema {
    pub pipelines: IndexMap<String, Pipeline>,
}

impl TransformSchema {
    pub fn execute(
        &self,
        name: &str,
        input: Value,
    ) -> Result<Value, TransformError> {
        let pipeline = self.pipelines.get(name).ok_or_else(|| TransformError::UndefinedPipeline {
            name: name.to_string(),
        })?;
        crate::execution::execute_pipeline(pipeline, input, self)
    }
}
