use std::collections::HashMap;

use serde::{Serialize, Serializer};

use crate::tool_calling::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallResultContent {
    Success(Value),
    Failure(String),
}

impl ToolCallResultContent {
    pub fn to_value(&self) -> Value {
        match self {
            ToolCallResultContent::Success(value) => value.clone(),
            ToolCallResultContent::Failure(error) => {
                let mut error_object = HashMap::new();
                error_object.insert("error".to_string(), Value::String(error.clone()));
                Value::Object(error_object)
            }
        }
    }
}

impl Serialize for ToolCallResultContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_value().serialize(serializer)
    }
}
