use serde::{Deserialize, Serialize};

use crate::tool_calling::{ToolFunctionParameters, ToolParameter, Tool, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: ToolFunctionParameters,
}

impl ToolFunction {
    pub fn new(name: String, description: String, parameters: ToolFunctionParameters) -> Self {
        Self {
            name,
            description,
            parameters,
        }
    }

    pub fn from_parameters(name: String, description: String, parameters: Vec<ToolParameter>) -> Self {
        let mut function_parameters = ToolFunctionParameters::new();

        for parameter in parameters {
            function_parameters = function_parameters.add_property(
                parameter.name.clone(),
                Value::from(parameter.to_schema()),
                parameter.is_required,
            );
        }

        Self {
            name,
            description,
            parameters: function_parameters,
        }
    }

    pub fn into_tool(self) -> Tool {
        Tool::Function { function: self }
    }
}
