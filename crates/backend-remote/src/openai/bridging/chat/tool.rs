use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject};
use shoji::types::basic::ToolDescription;

use crate::openai::Error;

pub fn build(description: ToolDescription) -> Result<ChatCompletionTools, Error> {
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
    Ok(ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: function.name,
            description: Some(function.description),
            parameters,
            strict: None,
        },
    }))
}
