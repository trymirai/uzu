#[derive(Debug, Clone)]
pub struct PrefillResult {
    pub tokens: Vec<u64>,
    pub forwardpass_durations: Vec<f64>,
    pub prefix_cache_restored_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub tokens: Vec<u64>,
    pub forwardpass_duration: f64,
    pub speculator_proposed: usize,
    pub speculator_accepted: usize,
}
