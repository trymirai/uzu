#[derive(Debug, Clone)]
pub struct PrefillResult {
    pub tokens: Vec<u64>,
    pub forwardpass_durations: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub tokens: Vec<u64>,
    pub forwardpass_duration: f64,
}
