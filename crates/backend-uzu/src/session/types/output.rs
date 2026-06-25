use serde::{Deserialize, Serialize};

use crate::session::types::{Stats, Text};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    ContextLimitReached,
}

#[derive(Debug, Clone)]
pub struct Output {
    pub text: Text,
    pub stats: Stats,
    pub finish_reason: Option<FinishReason>,
}

impl Output {
    pub fn clone_with_finish_reason(
        &self,
        finish_reason: Option<FinishReason>,
    ) -> Self {
        Self {
            text: self.text.clone(),
            stats: self.stats.clone(),
            finish_reason,
        }
    }
}
