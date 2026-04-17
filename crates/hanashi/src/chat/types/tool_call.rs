use serde::{Deserialize, Serialize};

use crate::chat::types::Value;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}
