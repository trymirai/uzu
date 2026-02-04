use schemars::generate::SchemaSettings;

use crate::tool_calling::{
    Tool, ToolCall, ToolCallResult, ToolError, ToolFunctionParameters, ToolImplementationCallable,
    Value,
};

pub struct ToolImplementationLambda<Parameters, Output, HandlerError, Handler>
where
    Parameters: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync,
    Output: serde::Serialize + Send + Sync,
    HandlerError: std::fmt::Display + Send + Sync,
    Handler: Fn(Parameters) -> Result<Output, HandlerError> + Send + Sync,
{
    pub name: String,
    pub description: String,
    handler: Handler,
    _phantom: std::marker::PhantomData<(Parameters, Output, HandlerError)>,
}

impl<Parameters, Output, HandlerError, Handler>
    ToolImplementationLambda<Parameters, Output, HandlerError, Handler>
where
    Parameters: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync,
    Output: serde::Serialize + Send + Sync,
    HandlerError: std::fmt::Display + Send + Sync,
    Handler: Fn(Parameters) -> Result<Output, HandlerError> + Send + Sync,
{
    pub fn new(name: String, description: String, handler: Handler) -> Self {
        Self {
            name,
            description,
            handler,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Parameters, Output, HandlerError, Handler> ToolImplementationCallable
    for ToolImplementationLambda<Parameters, Output, HandlerError, Handler>
where
    Parameters: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync,
    Output: serde::Serialize + Send + Sync,
    HandlerError: std::fmt::Display + Send + Sync,
    Handler: Fn(Parameters) -> Result<Output, HandlerError> + Send + Sync,
{
    fn tool(&self) -> Tool {
        let settings = SchemaSettings::default().with(|schema_settings| {
            schema_settings.inline_subschemas = true;
        });
        let schema = settings
            .into_generator()
            .into_root_schema_for::<Parameters>()
            .to_value();

        Tool::function_with(
            self.name.clone(),
            self.description.clone(),
            ToolFunctionParameters::from(schema),
        )
    }

    fn call(&self, tool_call: &ToolCall) -> Result<ToolCallResult, ToolError> {
        let json_parameters: serde_json::Value = tool_call.arguments.clone().into();

        let parsed: Parameters = serde_json::from_value(json_parameters).map_err(|error| {
            ToolError::InvalidParameters {
                message: error.to_string(),
            }
        })?;

        match (self.handler)(parsed) {
            Ok(output) => {
                let json_result = serde_json::to_value(output).map_err(|error| {
                    ToolError::SerializationError {
                        message: error.to_string(),
                    }
                })?;
                Ok(ToolCallResult::success(
                    tool_call.id.clone(),
                    tool_call.name.clone(),
                    Value::from(json_result),
                ))
            }
            Err(error) => Ok(ToolCallResult::failure(
                tool_call.id.clone(),
                tool_call.name.clone(),
                error.to_string(),
            )),
        }
    }
}
