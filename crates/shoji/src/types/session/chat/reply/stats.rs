use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyPowerStats {
    pub samples_count: i64,
    pub average_cpu_watts: f64,
    pub average_gpu_watts: f64,
    pub average_ane_watts: f64,
    pub average_ram_watts: f64,
    pub average_total_watts: f64,
    pub average_package_watts: f64,
    pub max_package_watts: f64,
    pub energy_joules: f64,
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatReplyStats {
    pub duration: f64,
    pub time_to_first_token: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub generate_tokens_per_second: Option<f64>,
    pub tokens_count_input: Option<u32>,
    pub tokens_count_output: Option<u32>,
    pub memory_used_bytes: Option<i64>,
    pub power_stats: Option<ChatReplyPowerStats>,
}

#[bindings::export(Implementation)]
impl ChatReplyStats {
    #[bindings::export(Method(Getter))]
    pub fn tokens_count(&self) -> Option<u32> {
        self.tokens_count_input.and_then(|input| self.tokens_count_output.map(|output| input + output))
    }
}
