use std::{future::Future, pin::Pin, sync::Arc};

use shoji::types::basic::Value;

pub type FutureError = Box<dyn std::error::Error + Send + Sync>;
pub type FutureFunction =
    dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value, FutureError>> + Send>> + Send + Sync;

#[derive(Clone)]
pub struct ToolFunctionDefinition {
    name: String,
    description: String,
    parameters: Option<Value>,
    return_definition: Option<Value>,
    func: Arc<FutureFunction>,
}

impl ToolFunctionDefinition {
    pub fn new(
        name: String,
        description: String,
        parameters: Option<Value>,
        return_definition: Option<Value>,
        func: Box<FutureFunction>,
    ) -> Self {
        Self {
            name,
            description,
            parameters,
            return_definition,
            func: Arc::new(func),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn parameters(&self) -> &Option<Value> {
        &self.parameters
    }

    pub fn return_definition(&self) -> &Option<Value> {
        &self.return_definition
    }

    pub async fn execute(
        &self,
        args: Value,
    ) -> Result<Value, FutureError> {
        (self.func)(args).await
    }
}
