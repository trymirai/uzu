use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionOutputRunStats {
    pub count: u64,
    pub average_duration: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionOutputStepStats {
    pub duration: f64,
    pub suffix_length: u64,
    pub tokens_count: u64,
    pub tokens_per_second: f64,
    pub model_run: SessionOutputRunStats,
    pub run: Option<SessionOutputRunStats>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionOutputTotalStats {
    pub duration: f64,
    pub tokens_count_input: u64,
    pub tokens_count_output: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionOutputStats {
    pub prefill_stats: SessionOutputStepStats,
    pub generate_stats: Option<SessionOutputStepStats>,
    pub total_stats: SessionOutputTotalStats,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionOutputFinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SessionOutput {
    pub text: String,
    pub stats: SessionOutputStats,
    pub finish_reason: Option<SessionOutputFinishReason>,
}

impl SessionOutput {
    pub fn clone_with_finish_reason(
        &self,
        finish_reason: Option<SessionOutputFinishReason>,
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
            stats: SessionOutputStats {
                prefill_stats: self.stats.prefill_stats.clone(),
                generate_stats: self.stats.generate_stats.clone(),
                total_stats: SessionOutputTotalStats {
                    duration: duration,
                    ..self.stats.total_stats.clone()
                },
            },
            finish_reason: self.finish_reason.clone(),
        }
    }
}
