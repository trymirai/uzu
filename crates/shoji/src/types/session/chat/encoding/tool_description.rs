use serde::{Deserialize, Serialize};

use crate::types::session::chat::ToolFunction;

#[bindings::export(Enum)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDescription {
    Function {
        function: ToolFunction,
    },
}
