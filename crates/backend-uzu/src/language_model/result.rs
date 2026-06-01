#[derive(Debug, Clone)]
pub struct PrefillResult {
    pub tokens: Vec<u64>,
    pub forwardpass_durations: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub tokens: Vec<u64>,
    pub forward_pass_cpu_duration: f64,
    pub forward_pass_gpu_duration: f64,
    pub speculator_proposed: usize,
    pub speculator_accepted: usize,
}
