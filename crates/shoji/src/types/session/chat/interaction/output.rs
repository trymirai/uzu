use serde::{Deserialize, Serialize};

use crate::types::session::chat::{FinishReason, Message, Stats};

#[bindings::export(Struct, name = "ChatOutput")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub message: Message,
    pub stats: Stats,
    pub finish_reason: Option<FinishReason>,
}
