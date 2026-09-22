use serde::{Deserialize, Serialize};

use crate::types::basic::Value;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolCall {
    #[serde(rename = "id")]
    pub identifier: Option<String>,
    pub name: String,
    /// A JSON object. Markup parsers (Qwen3.5/3.6) deliver every parameter as
    /// the text the model wrote, and the session stores that; consumers type
    /// the values from the tool's declared schema at their boundary, as the
    /// OpenAI server and the tool registry do.
    pub arguments: Value,
}
