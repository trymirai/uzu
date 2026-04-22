use async_openai::types::responses::{FunctionTool, Tool};
use shoji::types::basic::ToolDescription;

use crate::openai::Error;

pub fn build(description: ToolDescription) -> Result<Tool, Error> {
    let ToolDescription::Function {
        function,
    } = description;
    let parameters = match function.parameters {
        Some(value) => {
            Some(serde_json::from_str::<serde_json::Value>(&value.json).map_err(|error| Error::Serialization {
                message: error.to_string(),
            })?)
        },
        None => None,
    };
    Ok(Tool::Function(FunctionTool {
        name: function.name,
        parameters,
        strict: None,
        description: Some(function.description),
        defer_loading: None,
    }))
}
