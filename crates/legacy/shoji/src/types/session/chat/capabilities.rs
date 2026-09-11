use serde::{Deserialize, Serialize};

use crate::types::basic::ReasoningEffort;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatModelCapabilities {
    pub supports_reasoning: bool,
    pub supports_disable_reasoning: bool,
    #[serde(default)]
    pub reasoning_efforts: Vec<ReasoningEffort>,
    pub supports_tools: bool,
    pub supports_multiple_tool_calls: bool,
    pub requires_tools: bool,
}
