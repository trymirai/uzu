use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tool_calling::Value;

fn default_parameter_type() -> String {
    "object".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolFunctionParameters {
    #[serde(rename = "type", default = "default_parameter_type")]
    parameter_type: String,
    #[serde(default)]
    pub properties: HashMap<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

impl Default for ToolFunctionParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolFunctionParameters {
    pub fn new() -> Self {
        Self {
            parameter_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
        }
    }

    pub fn with_properties(
        mut self,
        properties: HashMap<String, Value>,
    ) -> Self {
        self.properties = properties;
        self
    }

    pub fn with_required(
        mut self,
        required: Vec<String>,
    ) -> Self {
        self.required = required;
        self
    }

    pub fn add_property(
        mut self,
        name: String,
        schema: Value,
        is_required: bool,
    ) -> Self {
        self.properties.insert(name.clone(), schema);
        if is_required {
            self.required.push(name);
        }
        self
    }
}

impl From<ToolFunctionParameters> for Value {
    fn from(function_parameters: ToolFunctionParameters) -> Self {
        let properties: HashMap<String, Value> = function_parameters.properties;

        let mut map = HashMap::new();
        map.insert("type".to_string(), Value::String("object".to_string()));
        map.insert("properties".to_string(), Value::Object(properties));

        if !function_parameters.required.is_empty() {
            map.insert(
                "required".to_string(),
                Value::Array(
                    function_parameters
                        .required
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }

        Value::Object(map)
    }
}

impl From<serde_json::Value> for ToolFunctionParameters {
    fn from(value: serde_json::Value) -> Self {
        let mut function_parameters = ToolFunctionParameters::new();

        if let serde_json::Value::Object(object) = value {
            if let Some(serde_json::Value::Object(properties_map)) =
                object.get("properties")
            {
                for (name, schema) in properties_map {
                    function_parameters
                        .properties
                        .insert(name.clone(), Value::from(schema.clone()));
                }
            }

            if let Some(serde_json::Value::Array(required_array)) =
                object.get("required")
            {
                for required_item in required_array {
                    if let serde_json::Value::String(required_name) =
                        required_item
                    {
                        function_parameters
                            .required
                            .push(required_name.clone());
                    }
                }
            }
        }

        function_parameters
    }
}
