use serde::{Deserialize, Serialize};

use crate::types::basic::Value;

#[bindings::export(Struct, name = "ChatToolCall")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolCall {
    #[serde(rename = "id")]
    pub identifier: Option<String>,
    pub name: String,
    pub arguments: Value,
}
