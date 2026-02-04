use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tool_calling::{ToolParameterType, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    #[serde(rename = "type")]
    pub parameter_type: ToolParameterType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub is_required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, Value>>,
}

impl ToolParameter {
    pub fn required(
        name: String,
        parameter_type: ToolParameterType,
        description: String,
    ) -> Self {
        Self {
            name,
            parameter_type,
            description: Some(description),
            is_required: true,
            extra: None,
        }
    }

    pub fn optional(
        name: String,
        parameter_type: ToolParameterType,
        description: String,
    ) -> Self {
        Self {
            name,
            parameter_type,
            description: Some(description),
            is_required: false,
            extra: None,
        }
    }

    pub fn with_extra(
        mut self,
        extra: HashMap<String, Value>,
    ) -> Self {
        self.extra = Some(extra);
        self
    }

    pub fn with_enum(
        mut self,
        values: Vec<Value>,
    ) -> Self {
        let extra = self.extra.get_or_insert_with(HashMap::new);
        extra.insert("enum".to_string(), Value::Array(values));
        self
    }

    pub fn to_schema(&self) -> serde_json::Value {
        let mut schema = self.parameter_type.to_schema();

        if let Some(description) = &self.description {
            schema["description"] =
                serde_json::Value::String(description.clone());
        }

        if let Some(extra) = &self.extra {
            if let serde_json::Value::Object(ref mut schema_object) = schema {
                for (key, value) in extra {
                    schema_object.insert(
                        key.clone(),
                        serde_json::Value::from(value.clone()),
                    );
                }
            }
        }

        schema
    }
}
