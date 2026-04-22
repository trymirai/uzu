use serde::{Deserialize, Serialize};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub supports_reasoning: bool,
    pub supports_disable_reasoning: bool,
    pub supports_tools: bool,
    pub supports_multiple_tool_calls: bool,
    pub requires_tools: bool,
}
