use std::{future::Future, pin::Pin, sync::Arc};

use shoji::types::basic::Value;

pub type FutureFunction = dyn Fn(Option<Value>) -> Pin<Box<dyn Future<Output = Option<Value>> + Send>> + Send + Sync;

#[derive(Clone)]
pub struct ToolFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
    pub return_def: Option<Value>,
    pub func: Arc<FutureFunction>,
}
