use crate::tool_calling::{Tool, ToolCall, ToolCallResult, ToolError};

pub trait ToolImplementationCallable: Send + Sync {
    fn tool(&self) -> Tool;
    fn call(&self, tool_call: &ToolCall) -> Result<ToolCallResult, ToolError>;
}
