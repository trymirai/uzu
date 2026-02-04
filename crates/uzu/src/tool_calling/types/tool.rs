use serde::{Deserialize, Serialize};

use crate::tool_calling::{ToolFunction, ToolFunctionParameters};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Tool {
    Function { function: ToolFunction },
}

impl Tool {
    pub fn function_with(name: String, description: String, parameters: ToolFunctionParameters) -> Self {
        Tool::Function {
            function: ToolFunction::new(name, description, parameters),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Tool::Function { function } => &function.name,
        }
    }
}
