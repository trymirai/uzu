use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatReplyFinishReason {
    Stop,
    Length,
    Cancelled,
    ContextLimitReached,
    ToolCalls,
    Rejected,
}
