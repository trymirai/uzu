use std::{collections::HashMap, sync::Arc};

use crate::tool_calling::{
    Tool, ToolCall, ToolCallResult, ToolError, ToolImplementationCallable,
    ToolImplementationLambda, Value,
};

pub type ToolHandler = Arc<dyn Fn(&ToolCall) -> Result<Value, String> + Send + Sync>;

#[derive(Default)]
pub struct ToolRegistry {
    implementations: HashMap<String, Box<dyn ToolImplementationCallable>>,
    handlers: HashMap<String, (Tool, ToolHandler)>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: ToolImplementationCallable + 'static>(
        &mut self,
        implementation: T,
    ) -> &mut Self {
        let name = implementation.tool().name().to_string();
        self.implementations.insert(name, Box::new(implementation));
        self
    }

    pub fn register_handler(
        &mut self,
        tool: Tool,
        handler: ToolHandler,
    ) -> &mut Self {
        let name = tool.name().to_string();
        self.handlers.insert(name, (tool, handler));
        self
    }

    pub fn register_lambda<Parameters, Output, HandlerError, Handler>(
        &mut self,
        name: String,
        description: String,
        handler: Handler,
    ) -> &mut Self
    where
        Parameters: serde::de::DeserializeOwned
            + schemars::JsonSchema
            + Send
            + Sync
            + 'static,
        Output: serde::Serialize + Send + Sync + 'static,
        HandlerError: std::fmt::Display + Send + Sync + 'static,
        Handler: Fn(Parameters) -> Result<Output, HandlerError>
            + Send
            + Sync
            + 'static,
    {
        let implementation =
            ToolImplementationLambda::new(name, description, handler);
        let tool_name = implementation.tool().name().to_string();
        self.implementations.insert(tool_name, Box::new(implementation));
        self
    }

    pub fn execute(
        &self,
        tool_call: &ToolCall,
    ) -> Result<ToolCallResult, ToolError> {
        let name = &tool_call.name;

        if let Some((_, handler)) = self.handlers.get(name) {
            return match handler(tool_call) {
                Ok(value) => Ok(ToolCallResult::success(
                    tool_call.id.clone(),
                    name.clone(),
                    value,
                )),
                Err(error) => Ok(ToolCallResult::failure(
                    tool_call.id.clone(),
                    name.clone(),
                    error,
                )),
            };
        }

        match self.implementations.get(name) {
            Some(implementation) => implementation.call(tool_call),
            None => Err(ToolError::NotFound { name: name.clone() }),
        }
    }

    pub fn tools(&self) -> Vec<Tool> {
        let mut result: Vec<Tool> = self
            .implementations
            .values()
            .map(|implementation| implementation.tool())
            .collect();
        result.extend(self.handlers.values().map(|(tool, _)| tool.clone()));
        result
    }

    pub fn contains(&self, name: &str) -> bool {
        self.implementations.contains_key(name) || self.handlers.contains_key(name)
    }
}
