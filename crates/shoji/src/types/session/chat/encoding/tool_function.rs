use serde::{Deserialize, Serialize};

use crate::types::basic::Value;

#[bindings::export(Struct, name = "ToolFunction")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
    pub r#return: Option<Value>,
}
