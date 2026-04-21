use serde::{Deserialize, Serialize};

use crate::types::session::chat::ToolDescription;

#[bindings::export(Struct, name = "ToolNamespace")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolNamespace {
    pub name: String,
    pub description: Option<String>,
    pub tools: Vec<ToolDescription>,
}
