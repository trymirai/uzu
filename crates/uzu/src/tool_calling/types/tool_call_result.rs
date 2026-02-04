use serde::Serialize;

use crate::tool_calling::{ToolCallResultContent, Value};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCallResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: ToolCallResultContent,
}

impl ToolCallResult {
    pub fn success(tool_call_id: String, name: String, content: Value) -> Self {
        Self {
            tool_call_id,
            name,
            content: ToolCallResultContent::Success(content),
        }
    }

    pub fn failure(tool_call_id: String, name: String, error: String) -> Self {
        Self {
            tool_call_id,
            name,
            content: ToolCallResultContent::Failure(error),
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self.content, ToolCallResultContent::Success(_))
    }
}
