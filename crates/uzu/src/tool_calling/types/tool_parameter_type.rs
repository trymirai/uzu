use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tool_calling::ToolParameter;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolParameterType {
    String,
    Bool,
    Int,
    Double,
    Array {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        items: Vec<ToolParameterType>,
    },
    Object {
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        properties: HashMap<String, ToolParameter>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        required: Vec<String>,
    },
    Data,
}

impl ToolParameterType {
    pub fn to_schema(&self) -> serde_json::Value {
        match self {
            ToolParameterType::String => serde_json::json!({"type": "string"}),
            ToolParameterType::Bool => serde_json::json!({"type": "boolean"}),
            ToolParameterType::Int => serde_json::json!({"type": "integer"}),
            ToolParameterType::Double => serde_json::json!({"type": "number"}),
            ToolParameterType::Array { items } => {
                let mut schema = serde_json::json!({"type": "array"});
                if let Some(item_type) = items.first() {
                    schema["items"] = item_type.to_schema();
                }
                schema
            }
            ToolParameterType::Object { properties, required } => {
                let mut schema = serde_json::json!({"type": "object"});
                if !properties.is_empty() {
                    let properties_schema: serde_json::Map<String, serde_json::Value> = properties
                        .iter()
                        .map(|(key, parameter)| (key.clone(), parameter.to_schema()))
                        .collect();
                    schema["properties"] = serde_json::Value::Object(properties_schema);
                }
                if !required.is_empty() {
                    schema["required"] = serde_json::json!(required);
                }
                schema
            }
            ToolParameterType::Data => serde_json::json!({}),
        }
    }

    pub fn string() -> Self {
        ToolParameterType::String
    }

    pub fn bool() -> Self {
        ToolParameterType::Bool
    }

    pub fn int() -> Self {
        ToolParameterType::Int
    }

    pub fn double() -> Self {
        ToolParameterType::Double
    }

    pub fn array(element_type: ToolParameterType) -> Self {
        ToolParameterType::Array {
            items: vec![element_type],
        }
    }

    pub fn object(properties: HashMap<String, ToolParameter>, required: Vec<String>) -> Self {
        ToolParameterType::Object {
            properties,
            required,
        }
    }
}
