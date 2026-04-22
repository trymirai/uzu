mod config;
mod finish_reason;
mod stats;

pub use config::ChatReplyConfig;
pub use finish_reason::ChatReplyFinishReason;
use serde::{Deserialize, Serialize};
pub use stats::ChatReplyStats;

use crate::types::session::chat::ChatMessage;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatReply {
    pub message: ChatMessage,
    pub stats: ChatReplyStats,
    pub finish_reason: Option<ChatReplyFinishReason>,
}
