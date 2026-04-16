use serde::{Deserialize, Serialize};

use crate::chat::types::Value;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
    pub r#return: Option<Value>,
}
