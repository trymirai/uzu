use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    pub cpu_watts: f32,
    pub gpu_watts: f32,
    pub gpu_sram_watts: f32,
    pub ane_watts: f32,
    pub ram_watts: f32,
    pub total_watts: f32,
}
