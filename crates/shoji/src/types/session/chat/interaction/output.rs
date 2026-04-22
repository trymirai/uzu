use serde::{Deserialize, Serialize};

use crate::types::session::chat::{ChatFinishReason, ChatMessage, ChatStats};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatOutput {
    pub message: ChatMessage,
    pub stats: ChatStats,
    pub finish_reason: Option<ChatFinishReason>,
}
