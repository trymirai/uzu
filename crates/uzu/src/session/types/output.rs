use serde::{Deserialize, Serialize};

use crate::session::types::{Stats, TotalStats};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct Output {
    pub text: String,
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
            finish_reason: finish_reason,
        }
    }

    pub fn clone_with_duration(
        &self,
        duration: f64,
    ) -> Self {
        Self {
            text: self.text.clone(),
            stats: Stats {
                prefill_stats: self.stats.prefill_stats.clone(),
                generate_stats: self.stats.generate_stats.clone(),
                total_stats: TotalStats {
                    duration: duration,
                    ..self.stats.total_stats.clone()
                },
            },
            finish_reason: self.finish_reason.clone(),
        }
    }
}
