use serde::{Deserialize, Serialize};

#[bindings::export(Enum, name = "ChatFinishReason")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    ContextLimitReached,
    ToolCalls,
    Rejected,
}
