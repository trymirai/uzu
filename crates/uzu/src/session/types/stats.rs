use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RunStats {
    pub count: u64,
    pub average_duration: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StepStats {
    pub duration: f64,
    pub suffix_length: u64,
    pub tokens_count: u64,
    pub tokens_per_second: f64,
    pub processed_tokens_per_second: f64,
    pub model_run: RunStats,
    pub run: Option<RunStats>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TotalStats {
    pub duration: f64,
    pub tokens_count_input: u64,
    pub tokens_count_output: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Stats {
    pub prefill_stats: StepStats,
    pub generate_stats: Option<StepStats>,
    pub total_stats: TotalStats,
}
