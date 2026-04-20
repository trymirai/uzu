use serde::{Deserialize, Serialize};

use crate::types::session::chat::RunStats;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepStats {
    pub duration: f64,
    pub suffix_length: u32,
    pub tokens_count: u32,
    pub tokens_per_second: f64,
    pub processed_tokens_per_second: f64,
    pub model_run: RunStats,
    pub run: Option<RunStats>,
    pub speculator_proposed: u32,
    pub speculator_accepted: u32,
}
