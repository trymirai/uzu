use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::tool_calling::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: String, arguments: Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            arguments,
        }
    }
}
