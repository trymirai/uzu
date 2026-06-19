use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub usage_percent: f32, // combined, core-count weighted
    pub ecpu_frequency_megahertz: u32,
    pub ecpu_usage_percent: f32,
    pub pcpu_frequency_megahertz: u32,
    pub pcpu_usage_percent: f32,
}
