use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Section {
    #[serde(rename = "$text")]
    Text {
        value: Option<String>,
    },
    Reasoning {
        value: Option<String>,
    },
    ToolCall {
        value: Option<Value>,
    },
    ToolCallResult {
        value: Option<Value>,
    },
}
